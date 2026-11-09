use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

pub static COUNTER_TOTAL_TOKENS_PREFILLED: AtomicU64 = AtomicU64::new(0);
pub static GAUGE_ACTIVE_SESSIONS: AtomicU64 = AtomicU64::new(0);
pub static GAUGE_QUANTIZED_SESSIONS: AtomicU64 = AtomicU64::new(0);
pub static HISTOGRAM_PREFILL_LATENCY: LazyLock<Mutex<Histogram>> =
    LazyLock::new(|| Mutex::new(Histogram::new(&HISTOGRAM_BUCKETS)));

const HISTOGRAM_BUCKETS: [f64; 9] = [0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0];

#[derive(Clone)]
pub struct HistogramSnapshot {
    pub buckets: Vec<(f64, u64)>,
    pub count: u64,
    pub sum: f64,
}

pub struct Histogram {
    boundaries: Vec<f64>,
    counts: Vec<u64>,
    count: u64,
    sum: f64,
}

impl Histogram {
    pub fn new(boundaries: &[f64]) -> Self {
        Self {
            boundaries: boundaries.to_vec(),
            counts: vec![0; boundaries.len()],
            count: 0,
            sum: 0.0,
        }
    }

    pub fn observe(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        for (index, boundary) in self.boundaries.iter().enumerate() {
            if value <= *boundary {
                self.counts[index] += 1;
            }
        }
    }

    pub fn snapshot(&self) -> HistogramSnapshot {
        HistogramSnapshot {
            buckets: self
                .boundaries
                .iter()
                .copied()
                .zip(self.counts.iter().copied())
                .collect(),
            count: self.count,
            sum: self.sum,
        }
    }
}

#[derive(Clone, Default)]
struct SessionMetricEntry {
    active: bool,
    quantized: bool,
}

/// Maximum number of per-session series `render_metrics` emits. Each emitted
/// session produces two series (`state="active"` and `state="quantized"`), so
/// the per-session cardinality is bounded at `2 * MAX_EMITTED_SESSIONS`.
/// The aggregate gauges (`axiom_active_sessions`, `axiom_quantized_sessions`)
/// already carry the totals, so per-session detail is diagnostic only and a
/// cap costs nothing an operator needs. The emitted set is chosen
/// deterministically (sorted by hashed label), so output is stable across
/// scrapes for a fixed registry.
pub const MAX_EMITTED_SESSIONS: usize = 100;

/// Hash a client-supplied session id into the label emitted in
/// `axiom_session_residency`: SHA-256 hex, truncated to 16 characters
/// (64 bits). SHA-256 is one-way, so the raw id is not recoverable from the
/// label; a preimage requires guessing the id. The hash is unsalted and
/// deterministic, so one session keeps a stable series across scrapes and
/// restarts. Truncation to 64 bits keeps labels short while accidental
/// collision stays negligible (birthday bound: about 2.7e-12 at 10,000
/// sessions, about 2.7e-8 at 1,000,000). Residual risk: anyone who can guess
/// a session id can confirm its presence in the output. That oracle is
/// limited to key holders, because `/metrics` is gated by `require_api_key`
/// whenever `AXIOM_API_KEY` is set. Hashing happens at render time only: the
/// registry still keys on the raw id, so `remove_session` and
/// `mark_session_quantized` keep working unchanged.
fn hash_session_id(session_id: &str) -> String {
    let digest = Sha256::digest(session_id.as_bytes());
    format!("{digest:x}")[..16].to_string()
}

static SESSION_METRICS: LazyLock<Mutex<HashMap<String, SessionMetricEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn add_prefilled_tokens(tokens: usize) {
    COUNTER_TOTAL_TOKENS_PREFILLED.fetch_add(tokens as u64, Ordering::Relaxed);
}

pub fn set_active_sessions(count: usize) {
    GAUGE_ACTIVE_SESSIONS.store(count as u64, Ordering::Relaxed);
}

pub fn set_quantized_sessions(count: usize) {
    GAUGE_QUANTIZED_SESSIONS.store(count as u64, Ordering::Relaxed);
}

pub fn observe_prefill_latency(seconds: f64) {
    let seconds = seconds.max(0.0);
    if let Ok(mut histogram) = HISTOGRAM_PREFILL_LATENCY.lock() {
        histogram.observe(seconds);
    }
}

pub fn register_session(session_id: &str) {
    if let Ok(mut registry) = SESSION_METRICS.lock() {
        registry.insert(
            session_id.to_string(),
            SessionMetricEntry {
                active: true,
                quantized: false,
            },
        );
    }
}

pub fn remove_session(session_id: &str) {
    if let Ok(mut registry) = SESSION_METRICS.lock() {
        registry.remove(session_id);
    }
}

pub fn mark_session_quantized(session_id: &str, quantized: bool) {
    if let Ok(mut registry) = SESSION_METRICS.lock() {
        let entry = registry.entry(session_id.to_string()).or_default();
        entry.quantized = quantized;
        entry.active = !quantized;
    }
}

pub fn render_metrics() -> String {
    let total_tokens = COUNTER_TOTAL_TOKENS_PREFILLED.load(Ordering::Relaxed);
    let active_sessions = GAUGE_ACTIVE_SESSIONS.load(Ordering::Relaxed);
    let quantized_sessions = GAUGE_QUANTIZED_SESSIONS.load(Ordering::Relaxed);
    let histogram = HISTOGRAM_PREFILL_LATENCY
        .lock()
        .map(|guard| guard.snapshot())
        .unwrap_or(HistogramSnapshot {
            buckets: HISTOGRAM_BUCKETS
                .iter()
                .copied()
                .map(|bucket| (bucket, 0))
                .collect(),
            count: 0,
            sum: 0.0,
        });
    let session_entries = SESSION_METRICS
        .lock()
        .map(|registry| {
            let mut entries = registry
                .iter()
                .map(|(id, entry)| (hash_session_id(id), entry.clone()))
                .collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            entries.truncate(MAX_EMITTED_SESSIONS);
            entries
        })
        .unwrap_or_default();

    let mut body = String::new();
    body.push_str("# HELP axiom_total_tokens_prefilled Total count of tokens ingested.\n");
    body.push_str("# TYPE axiom_total_tokens_prefilled counter\n");
    body.push_str(&format!("axiom_total_tokens_prefilled {}\n", total_tokens));

    body.push_str("# HELP axiom_active_sessions Current number of allocated sessions in memory.\n");
    body.push_str("# TYPE axiom_active_sessions gauge\n");
    body.push_str(&format!("axiom_active_sessions {}\n", active_sessions));

    body.push_str(
        "# HELP axiom_quantized_sessions Number of idle sessions parked in compressed form.\n",
    );
    body.push_str("# TYPE axiom_quantized_sessions gauge\n");
    body.push_str(&format!(
        "axiom_quantized_sessions {}\n",
        quantized_sessions
    ));

    body.push_str(
        "# HELP axiom_prefill_latency_seconds Time spent inside prefill execution paths.\n",
    );
    body.push_str("# TYPE axiom_prefill_latency_seconds histogram\n");
    for (bucket, count) in &histogram.buckets {
        body.push_str(&format!(
            "axiom_prefill_latency_seconds_bucket{{le=\"{}\"}} {}\n",
            bucket, count
        ));
    }
    body.push_str(&format!(
        "axiom_prefill_latency_seconds_bucket{{le=\"+Inf\"}} {}\n",
        histogram.count
    ));
    body.push_str(&format!(
        "axiom_prefill_latency_seconds_sum {}\n",
        histogram.sum
    ));
    body.push_str(&format!(
        "axiom_prefill_latency_seconds_count {}\n",
        histogram.count
    ));

    body.push_str("# HELP axiom_session_residency Session residency states.\n");
    body.push_str("# TYPE axiom_session_residency gauge\n");
    for (session_label, entry) in session_entries {
        body.push_str(&format!(
            "axiom_session_residency{{session_id=\"{}\",state=\"active\"}} {}\n",
            escape_label_value(&session_label),
            if entry.active { 1 } else { 0 }
        ));
        body.push_str(&format!(
            "axiom_session_residency{{session_id=\"{}\",state=\"quantized\"}} {}\n",
            escape_label_value(&session_label),
            if entry.quantized { 1 } else { 0 }
        ));
    }

    body
}

fn escape_label_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent oracle for the label the renderer must emit for a session
    /// id: SHA-256 hex, truncated to 16 characters. Computed here from the
    /// spec, not by calling the implementation, so the test fails if the
    /// renderer emits anything else (including the raw id).
    fn expected_session_label(session_id: &str) -> String {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(session_id.as_bytes());
        format!("{digest:x}")[..16].to_string()
    }

    /// Every `axiom_session_residency` label in the rendered output must be
    /// a 16-char lowercase hex hash, never a raw client id. This is the
    /// invariant that holds regardless of how many sessions are registered
    /// (the cap may truncate a given session from the emitted set, so a
    /// per-id presence assertion would be order-dependent through the shared
    /// global registry).
    fn assert_labels_are_truncated_hashes(rendered: &str) {
        let mut found = 0;
        for line in rendered.lines() {
            if !line.starts_with("axiom_session_residency{") {
                continue;
            }
            let label = line
                .split("session_id=\"")
                .nth(1)
                .and_then(|rest| rest.split('"').next())
                .expect("residency line must carry a session_id label");
            assert_eq!(label.len(), 16, "session label is not 16 chars: {label}");
            assert!(
                label
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "session label is not lowercase hex: {label}"
            );
            found += 1;
        }
        assert!(found > 0, "no residency lines rendered");
    }

    #[test]
    fn test_hash_session_id_is_truncated_sha256() {
        // Direct unit test of the mapping: deterministic, and equal to the
        // independently computed SHA-256 hex truncation.
        assert_eq!(
            hash_session_id("session-a"),
            expected_session_label("session-a")
        );
        assert_eq!(
            hash_session_id("session-a"),
            hash_session_id("session-a"),
            "same id must map to a stable label across renders"
        );
        assert_ne!(
            hash_session_id("session-a"),
            hash_session_id("session-b"),
            "distinct ids must map to distinct labels"
        );
    }

    #[test]
    fn test_render_metrics_contains_histogram_and_session_registry() {
        add_prefilled_tokens(4);
        set_active_sessions(2);
        set_quantized_sessions(1);
        observe_prefill_latency(0.01);
        register_session("session-a");
        mark_session_quantized("session-b", true);

        let rendered = render_metrics();

        assert!(rendered.contains("axiom_total_tokens_prefilled"));
        assert!(rendered.contains("axiom_prefill_latency_seconds_bucket"));
        // Session ids are emitted only as truncated SHA-256 labels, never raw.
        assert!(!rendered.contains("session-a"));
        assert!(!rendered.contains("session-b"));
        assert_labels_are_truncated_hashes(&rendered);
    }

    #[test]
    fn test_render_metrics_never_emits_raw_adversarial_session_ids() {
        let long_id = "a".repeat(5_000);
        let tricky_id = "we\"ird\\\\id\nwith\nnewlines";
        let quote_id = "\"quoted\"";
        register_session(&long_id);
        register_session(tricky_id);
        register_session(quote_id);

        let rendered = render_metrics();

        for raw in [&long_id, tricky_id, quote_id] {
            assert!(
                !rendered.contains(raw),
                "rendered metrics leaked raw session id"
            );
        }
        // Every emitted label is a truncated hash, so no adversarial id can
        // break out of the label value either.
        assert_labels_are_truncated_hashes(&rendered);
    }

    #[test]
    fn test_render_metrics_bounds_per_session_series() {
        // 250 distinct sessions must not produce 250 per-session series: the
        // bound is 100 emitted sessions x 2 states = 200 residency lines.
        for index in 0..250 {
            register_session(&format!("bound-test-{index}"));
        }

        let rendered = render_metrics();
        let residency_lines = rendered
            .lines()
            .filter(|line| line.starts_with("axiom_session_residency{"))
            .count();

        assert!(residency_lines > 0);
        assert!(
            residency_lines <= 200,
            "rendered {residency_lines} per-session residency lines, bound is 200"
        );
    }
}
