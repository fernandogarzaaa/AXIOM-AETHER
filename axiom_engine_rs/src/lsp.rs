//! Native Axiom Language Server Protocol daemon.
//!
//! The LSP front-end stays deliberately thin: document notifications update an
//! in-memory text map and enqueue full snapshots onto a background worker. The
//! worker builds Tree-sitter structural digests and runs bounded TTT prefill so
//! editor typing never waits on Candle computation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use candle_core::Result as CandleResult;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio::task::spawn_blocking;
use tower_lsp::jsonrpc::Result as LspResult;
use tower_lsp::lsp_types::{
    DidChangeTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams,
    InitializeParams, InitializeResult, MessageType, ServerCapabilities,
    TextDocumentSyncCapability, TextDocumentSyncKind,
};
use tower_lsp::{async_trait, Client, LanguageServer, LspService, Server};

use crate::context_compressor::{adapt_session_blocking, TttSessionStore};
use crate::inference::InferencePipeline;
use crate::skeleton::build_digest;

#[derive(Clone)]
struct LspUpdate {
    uri: String,
    text: String,
}

pub struct AxiomLspBackend {
    client: Client,
    documents: Arc<Mutex<HashMap<String, String>>>,
    updates: mpsc::Sender<LspUpdate>,
}

pub async fn run_lsp_server(pipeline: InferencePipeline) -> Result<(), String> {
    let (updates_tx, updates_rx) = mpsc::channel(64);
    let pipeline = Arc::new(RwLock::new(pipeline));
    let sessions = Arc::new(TttSessionStore::new());
    tokio::spawn(run_prefill_worker(updates_rx, pipeline, sessions));

    let documents = Arc::new(Mutex::new(HashMap::new()));
    let documents_for_service = documents.clone();
    let updates_for_service = updates_tx.clone();
    let (service, socket) = LspService::new(move |client| AxiomLspBackend {
        client,
        documents: documents_for_service.clone(),
        updates: updates_for_service.clone(),
    });
    Server::new(tokio::io::stdin(), tokio::io::stdout(), socket)
        .serve(service)
        .await;
    Ok(())
}

#[async_trait]
impl LanguageServer for AxiomLspBackend {
    async fn initialize(&self, _: InitializeParams) -> LspResult<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                ..Default::default()
            },
            server_info: Some(tower_lsp::lsp_types::ServerInfo {
                name: "axiom-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: tower_lsp::lsp_types::InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Axiom LSP initialized")
            .await;
    }

    async fn shutdown(&self) -> LspResult<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        let text = params.text_document.text;
        self.store_and_enqueue(uri, text).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        let Some(change) = params.content_changes.into_iter().last() else {
            return;
        };
        self.store_and_enqueue(uri, change.text).await;
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        let text = params.text.or_else(|| {
            self.documents
                .lock()
                .ok()
                .and_then(|docs| docs.get(&uri).cloned())
        });
        if let Some(text) = text {
            self.store_and_enqueue(uri, text).await;
        }
    }
}

impl AxiomLspBackend {
    async fn store_and_enqueue(&self, uri: String, text: String) {
        if let Ok(mut docs) = self.documents.lock() {
            docs.insert(uri.clone(), text.clone());
        }
        if let Err(e) = self.updates.try_send(LspUpdate { uri, text }) {
            self.client
                .log_message(
                    MessageType::WARNING,
                    format!("Axiom LSP update dropped: {e}"),
                )
                .await;
        }
    }
}

async fn run_prefill_worker(
    mut updates: mpsc::Receiver<LspUpdate>,
    pipeline: Arc<RwLock<InferencePipeline>>,
    sessions: Arc<TttSessionStore>,
) {
    while let Some(update) = updates.recv().await {
        let pipeline = pipeline.clone();
        let sessions = sessions.clone();
        let _ = spawn_blocking(move || prefill_update(update, pipeline, sessions)).await;
    }
}

fn prefill_update(
    update: LspUpdate,
    pipeline: Arc<RwLock<InferencePipeline>>,
    sessions: Arc<TttSessionStore>,
) -> CandleResult<usize> {
    let digest = structural_digest_for_lsp(&update.uri, &update.text);
    let pipeline = pipeline
        .read()
        .map_err(|_| candle_core::Error::Msg("pipeline lock poisoned".into()))?;
    let handle = sessions.get_or_create(&update.uri, &pipeline)?;
    let mut states = handle.blocking_lock();
    let tokens = pipeline.encode_text(&digest);
    adapt_session_blocking(&pipeline, &mut states, &tokens)?;
    Ok(tokens.len())
}

pub fn structural_digest_for_lsp(uri: &str, text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(uri.as_bytes());
    hasher.update(text.as_bytes());
    let state_hash = format!("sha256:{:x}", hasher.finalize());
    let started = Instant::now();
    build_digest(
        text,
        uri,
        text.split_whitespace().count(),
        0.0,
        &state_hash,
        8,
    )
    .replace(
        "recall_norm=\"0.000\"",
        &format!("recall_norm=\"{:.3}\"", started.elapsed().as_secs_f32()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsp_digest_uses_tree_sitter_skeleton() {
        let digest = structural_digest_for_lsp(
            "file:///demo.rs",
            "pub struct Demo { value: i32 }\nimpl Demo { pub fn value(&self) -> i32 { self.value } }",
        );
        assert!(digest.contains("pub struct Demo"));
        assert!(digest.contains("impl Demo"));
        assert!(!digest.contains("self.value }"));
    }
}
