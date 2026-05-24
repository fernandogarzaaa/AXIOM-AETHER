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
                .map(|(id, entry)| (id.clone(), entry.clone()))
                .collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
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
    for (session_id, entry) in session_entries {
        body.push_str(&format!(
            "axiom_session_residency{{session_id=\"{}\",state=\"active\"}} {}\n",
            escape_label_value(&session_id),
            if entry.active { 1 } else { 0 }
        ));
        body.push_str(&format!(
            "axiom_session_residency{{session_id=\"{}\",state=\"quantized\"}} {}\n",
            escape_label_value(&session_id),
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
        assert!(rendered.contains("session-a"));
        assert!(rendered.contains("session-b"));
    }
}
