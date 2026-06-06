//! build_pairs — crawl source dirs, mine doc<->body + markdown pairs, and
//! (optionally) generate synthetic NL questions per content chunk via the
//! running Axiom proxy. Writes checkpoints/pairs.jsonl.
//!
//!   AXIOM_PAIR_DIRS="C:/Users/garza/AXIOM-AETHER;C:/Users/garza/ChimeraLang" \
//!   AXIOM_SYNTH=200 cargo run --release --bin build_pairs
//!
//! Env: AXIOM_PAIR_DIRS (";"-separated roots; default = this repo)
//!      AXIOM_PAIRS_OUT (default checkpoints/pairs.jsonl)
//!      AXIOM_SYNTH (synthetic-query count; 0 = skip)
//!      AXIOM_PROXY_URL (default http://127.0.0.1:3000)

use axiom_engine::corpus::{collect_files, Deduper};
use axiom_engine::pairs::{mine_doc_body, mine_markdown, write_pairs_jsonl, Pair};
use serde_json::json;

fn repo() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn main() {
    let dirs = std::env::var("AXIOM_PAIR_DIRS").unwrap_or_else(|_| repo().to_string_lossy().into());
    let out = std::env::var("AXIOM_PAIRS_OUT")
        .unwrap_or_else(|_| repo().join("checkpoints/pairs.jsonl").to_string_lossy().into());
    let synth_n: usize = std::env::var("AXIOM_SYNTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let _ = std::fs::remove_file(&out); // fresh build

    let mut dedup = Deduper::new();
    let mut mined: Vec<Pair> = Vec::new();
    for root in dirs.split(';').filter(|s| !s.trim().is_empty()) {
        let files = collect_files(std::path::Path::new(root.trim()), 50_000, 512 * 1024);
        eprintln!("[pairs] {root}: {} files", files.len());
        for f in files {
            let Ok(text) = std::fs::read_to_string(&f) else { continue };
            if !dedup.accept(text.as_bytes()) {
                continue;
            }
            let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "md" {
                mined.extend(mine_markdown(&text, 40));
            } else {
                mined.extend(mine_doc_body(&text, 25));
            }
        }
    }
    eprintln!("[pairs] mined {} pairs", mined.len());
    write_pairs_jsonl(&out, &mined).expect("write mined pairs");

    // Synthetic queries via the proxy (bounded). Each call asks Claude to produce
    // one natural-language question answered by a content chunk → (question, chunk).
    if synth_n > 0 && !mined.is_empty() {
        let base = std::env::var("AXIOM_PROXY_URL").unwrap_or_else(|_| "http://127.0.0.1:3000".into());
        let url = format!("{}/v1/messages", base.trim_end_matches('/'));
        let client = reqwest::blocking::Client::new();
        let step = (mined.len() / synth_n).max(1);
        let mut synth: Vec<Pair> = Vec::new();
        for (i, p) in mined.iter().enumerate().step_by(step).take(synth_n) {
            let prompt = format!(
                "Read this code/text and write ONE short natural-language question a developer would ask whose answer is this snippet. Output ONLY the question.\n\n{}",
                &p.positive.chars().take(1200).collect::<String>()
            );
            let body = json!({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 64,
                "messages": [{"role":"user","content": prompt}]
            });
            match client.post(&url).json(&body).send().and_then(|r| r.json::<serde_json::Value>()) {
                Ok(v) => {
                    let q = v["content"][0]["text"].as_str().unwrap_or("").trim().to_string();
                    if q.len() > 8 {
                        synth.push(Pair { anchor: q, positive: p.positive.clone(), source: "synthetic".into() });
                    }
                }
                Err(e) => eprintln!("[pairs] synth {i} failed (proxy down?): {e}"),
            }
        }
        eprintln!("[pairs] synthetic pairs: {}", synth.len());
        write_pairs_jsonl(&out, &synth).expect("write synth pairs");
    }
    eprintln!("[pairs] DONE → {out}");
}
