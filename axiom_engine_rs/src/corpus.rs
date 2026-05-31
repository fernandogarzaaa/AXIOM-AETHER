//! Corpus utilities for the multi-language code crawler. All RAM-bounded:
//! callers stream files; we only ever hold one file's bytes + a hash set.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Source-code extensions we ingest (multi-language code specialist).
pub const CODE_EXTS: &[&str] = &[
    "rs", "go", "py", "ts", "tsx", "js", "jsx", "c", "h", "cpp", "hpp", "cc", "java", "rb", "php",
    "swift", "kt", "scala", "sh", "toml", "json", "md",
];

pub fn is_code_file(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => CODE_EXTS.contains(&ext.to_ascii_lowercase().as_str()),
        None => false,
    }
}

/// Content hash for dedup (cargo registry has heavy duplication).
pub fn content_hash(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

/// Tracks seen content hashes to skip duplicate files across the crawl.
#[derive(Default)]
pub struct Deduper {
    seen: HashSet<[u8; 32]>,
}

impl Deduper {
    pub fn new() -> Self {
        Self { seen: HashSet::new() }
    }
    /// Returns true if this content is new (and records it); false if duplicate.
    pub fn accept(&mut self, bytes: &[u8]) -> bool {
        self.seen.insert(content_hash(bytes))
    }
    pub fn len(&self) -> usize {
        self.seen.len()
    }
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

/// Recursively collect code-file paths under `root`, skipping hidden dirs and
/// `target/`, and files over `max_file_bytes`. Bounded by `max_files`.
/// (node_modules is intentionally kept — we want JS/TS.)
pub fn collect_files(root: &Path, max_files: usize, max_file_bytes: u64) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if out.len() >= max_files {
            break;
        }
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let p = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if p.is_dir() {
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                stack.push(p);
            } else if is_code_file(&p) {
                if let Ok(meta) = entry.metadata() {
                    if meta.len() <= max_file_bytes && meta.len() > 0 {
                        out.push(p);
                        if out.len() >= max_files {
                            break;
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn detects_code_extensions() {
        assert!(is_code_file(Path::new("a/b.rs")));
        assert!(is_code_file(Path::new("x.py")));
        assert!(!is_code_file(Path::new("img.png")));
        assert!(!is_code_file(Path::new("noext")));
    }

    #[test]
    fn deduper_rejects_duplicate_content() {
        let mut d = Deduper::new();
        assert!(d.accept(b"fn main() {}"));
        assert!(!d.accept(b"fn main() {}"));
        assert!(d.accept(b"different"));
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn collect_files_finds_only_code_under_root() {
        let tmp = std::env::temp_dir().join(format!("axiom_corpus_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(tmp.join("sub"));
        std::fs::write(tmp.join("a.rs"), b"fn a() {}").unwrap();
        std::fs::write(tmp.join("b.png"), b"\x89PNG").unwrap();
        std::fs::write(tmp.join("sub/c.py"), b"def c(): pass").unwrap();
        let files = collect_files(&tmp, 100, 1_000_000);
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"a.rs".to_string()));
        assert!(names.contains(&"c.py".to_string()));
        assert!(!names.contains(&"b.png".to_string()));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
