use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub struct JitContextStreamer {
    vocab_size: u32,
    max_context_tokens: usize,
    client: Client,
    api_url: Option<String>,
    api_key: Option<String>,
}

impl JitContextStreamer {
    pub fn new(
        vocab_size: u32,
        max_context_tokens: usize,
        api_url: Option<String>,
        api_key: Option<String>,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(6))
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            vocab_size: vocab_size.max(1),
            max_context_tokens: max_context_tokens.max(1),
            client,
            api_url,
            api_key,
        }
    }

    pub fn fetch_and_pack_context(&self, user_prompt: &str) -> Vec<u32> {
        let docs = self.collect_context(user_prompt);
        let ranked = self.rank_lines(user_prompt, &docs);
        let mut packed = Vec::with_capacity(self.max_context_tokens);

        for line in ranked {
            for tok in tokenize(&line) {
                let digest = Sha256::digest(tok.as_bytes());
                let id = u64::from_le_bytes([
                    digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6],
                    digest[7],
                ]) % self.vocab_size as u64;
                packed.push(id as u32);
                if packed.len() >= self.max_context_tokens {
                    break;
                }
            }
            if packed.len() >= self.max_context_tokens {
                break;
            }
        }

        if packed.is_empty() {
            vec![0]
        } else {
            packed
        }
    }

    fn collect_context(&self, query: &str) -> Vec<String> {
        let subqueries = decompose_query(query, 4);
        let mut docs = Vec::new();

        if let Some(api_url) = &self.api_url {
            for sq in subqueries {
                if let Ok(mut fetched) = self.fetch_live_docs(api_url, &sq) {
                    docs.append(&mut fetched);
                }
            }
        }

        if docs.is_empty() {
            return vec![
                format!(
                    "# Offline Context\nNo live API response for query: {query}\nUsing local fallback retrieval surface."
                ),
                format!(
                    "## Notes\nEnable --context-api-url (or AXIOM_CONTEXT_API_URL) to inject live markdown context for: {query}"
                ),
            ];
        }

        docs
    }

    fn fetch_live_docs(&self, base_url: &str, query: &str) -> Result<Vec<String>, reqwest::Error> {
        let encoded = urlencoding::encode(query);
        let target = if base_url.contains("{query}") {
            base_url.replace("{query}", &encoded)
        } else if base_url.contains('?') {
            format!("{base_url}&q={encoded}")
        } else {
            format!("{base_url}?q={encoded}")
        };

        let mut request = self.client.get(target).header(
            "accept",
            "application/json,text/markdown;q=0.9,text/plain;q=0.9",
        );
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = request.send()?.error_for_status()?;
        let body = response.text()?;

        let parsed = parse_documents(&body);
        if parsed.is_empty() {
            Ok(vec![body])
        } else {
            Ok(parsed)
        }
    }

    fn rank_lines(&self, query: &str, docs: &[String]) -> Vec<String> {
        let query_terms = tokenize(query);
        let query_set: HashSet<&str> = query_terms.iter().map(String::as_str).collect();
        let mut seen = HashSet::new();
        let mut scored: Vec<(f32, String)> = Vec::new();

        for doc in docs {
            for raw_line in doc.lines() {
                let line = raw_line.trim().to_lowercase();
                if line.is_empty() || !seen.insert(line.clone()) {
                    continue;
                }
                let line_terms = tokenize(&line);
                if line_terms.is_empty() {
                    continue;
                }
                let overlap = line_terms
                    .iter()
                    .filter(|term| query_set.contains(term.as_str()))
                    .count() as f32;
                let score = overlap / (line_terms.len() as f32);
                if score > 0.0 {
                    scored.push((score, line));
                }
            }
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored.into_iter().map(|(_, line)| line).collect()
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn decompose_query(query: &str, max_subqueries: usize) -> Vec<String> {
    let mut out = vec![query.to_string()];
    for term in tokenize(query)
        .into_iter()
        .filter(|t| t.len() > 3)
        .take(max_subqueries.saturating_sub(1))
    {
        out.push(format!("{query} {term}"));
    }
    out
}

fn parse_documents(body: &str) -> Vec<String> {
    fn collect_strings(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::String(s) => {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    out.push(trimmed.to_string());
                }
            }
            Value::Array(values) => {
                for item in values {
                    collect_strings(item, out);
                }
            }
            Value::Object(map) => {
                for value in map.values() {
                    collect_strings(value, out);
                }
            }
            _ => {}
        }
    }

    let mut docs = Vec::new();
    if let Ok(json) = serde_json::from_str::<Value>(body) {
        collect_strings(&json, &mut docs);
    }
    docs
}
