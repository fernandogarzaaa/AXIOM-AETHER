//! The shared conformance suite.
//!
//! Every CP/1 binding, in every repository, runs the same five checks against
//! the same golden corpus. That is the entire mechanism keeping three
//! hand-written bindings in three languages agreeing about the wire — there is
//! no code generation and no shared library to enforce it, so this suite is
//! load-bearing rather than decorative.
//!
//! The checks, and the failure each one catches:
//!
//! 1. **Round trip.** Parsing a fixture and re-encoding it in canonical form
//!    reproduces the exact bytes. Catches a binding whose key ordering, number
//!    rendering or string escaping differs from the normative source — the
//!    class of bug that silently produces documents other components reject.
//! 2. **Seal.** Each fixture's `provenance.content_hash` is the true hash of
//!    the document with that member removed. Catches a binding that hashes a
//!    different byte sequence than it transmits.
//! 3. **Structure.** Required members are present with the right shapes, and
//!    basis-point members are integers in range. Catches a binding that would
//!    accept a float where the protocol forbids one.
//! 4. **Manifest.** The vendored copy of the corpus hashes to what the
//!    normative source recorded. Catches a binding running against a stale
//!    fixture file, which would make checks 1–3 pass against the wrong contract.
//! 5. **Provenance edges.** `derived_from` ids resolve within the corpus, and a
//!    `FitnessResult` that reports runs references the `SimulationCompleted`
//!    that produced them. Catches a measurement that cannot be chained back to
//!    the work behind it — structurally indistinguishable from a fabricated one.

use serde_json::Value;

use super::canonical;
use super::event::EventKind;
use crate::provenance::sha256_hex;

/// One conformance failure, named precisely enough to act on without a debugger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub fixture_line: usize,
    pub document_type: String,
    pub detail: String,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fixture line {} ({}): {}",
            self.fixture_line, self.document_type, self.detail
        )
    }
}

/// Members every CP/1 document carries.
const REQUIRED_PROVENANCE_MEMBERS: [&str; 6] = [
    "authored_by",
    "produced_at",
    "origin",
    "evidence",
    "derived_from",
    "content_hash",
];

/// Every document type the corpus must cover.
///
/// Twelve of these are canonical types (SPEC.md section 3). `ValidationRequest`
/// is not — it is a protocol message (section 3.0) — but it crosses the wire and
/// must round-trip identically, so it is covered here alongside them. Naming the
/// constant for canonical types alone would make it contradict the spec.
const COVERED_TYPES: [&str; 13] = [
    "Identity",
    "Genome",
    "Capability",
    "Belief",
    "Memory",
    "Skill",
    "Mutation",
    "Reflection",
    "Observation",
    "Experience",
    "FitnessResult",
    "Context",
    "ValidationRequest",
];

/// Run checks 1–3 over a fixture corpus.
///
/// `corpus` is the contents of `fixtures/canonical.jsonl`: one canonical-form
/// document per line. Returns every failure rather than the first, so a binding
/// under repair sees the whole picture in one run.
pub fn check_corpus(corpus: &str) -> Vec<Failure> {
    let mut failures = Vec::new();
    let mut seen_types = std::collections::BTreeSet::new();
    // Check 5 reads across documents, so it needs the whole corpus indexed
    // before any edge can be resolved. Collected here rather than in a second
    // pass over the text so each line is parsed once.
    let mut type_by_id: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    let mut edges: Vec<(usize, String, String, Vec<String>, u64)> = Vec::new();

    for (index, line) in corpus.lines().enumerate() {
        let lineno = index + 1;
        if line.trim().is_empty() {
            continue;
        }

        let document: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(err) => {
                failures.push(Failure {
                    fixture_line: lineno,
                    document_type: "?".to_string(),
                    detail: format!("not valid JSON: {err}"),
                });
                continue;
            }
        };

        let doc_type = document
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();

        // An event's `type` is its event name; its canonical type is `Event`.
        seen_types.insert(if EventKind::parse(&doc_type).is_some() {
            "Event".to_string()
        } else {
            doc_type.clone()
        });

        let fail = |detail: String| Failure {
            fixture_line: lineno,
            document_type: doc_type.clone(),
            detail,
        };

        // 1. Round trip.
        match canonical::to_canonical(&document) {
            Ok(reencoded) if reencoded == line => {}
            Ok(reencoded) => failures.push(fail(format!(
                "re-encoding changed the bytes:\n    read:  {line}\n    wrote: {reencoded}"
            ))),
            Err(err) => failures.push(fail(format!("could not be canonicalized: {err}"))),
        }

        // 2. Seal.
        match canonical::verify_seal(&document) {
            Ok(true) => {}
            Ok(false) => failures.push(fail(
                "provenance.content_hash does not match the document's content".to_string(),
            )),
            Err(err) => failures.push(fail(format!("seal could not be checked: {err}"))),
        }

        // 3. Structure.
        for detail in structural_problems(&document) {
            failures.push(fail(detail));
        }

        // Indexed for check 5, below.
        if let Some(id) = document.get("id").and_then(Value::as_str) {
            type_by_id.insert(id.to_string(), doc_type.clone());
            let derived = document
                .get("provenance")
                .and_then(|p| p.get("derived_from"))
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let baseline_runs = document
                .get("baseline")
                .and_then(|b| b.get("runs"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            edges.push((
                lineno,
                doc_type.clone(),
                id.to_string(),
                derived,
                baseline_runs,
            ));
        }
    }

    // Coverage: a corpus missing a type would let that type's encoding drift
    // in every binding at once, undetected.
    for expected in COVERED_TYPES {
        if !seen_types.contains(expected) {
            failures.push(Failure {
                fixture_line: 0,
                document_type: expected.to_string(),
                detail: "no fixture covers this document type".to_string(),
            });
        }
    }
    if !seen_types.contains("Event") {
        failures.push(Failure {
            fixture_line: 0,
            document_type: "Event".to_string(),
            detail: "no fixture covers the event envelope".to_string(),
        });
    }

    failures.extend(provenance_edge_problems(&type_by_id, &edges));
    failures
}

/// Check 5: `derived_from` edges resolve, and a `FitnessResult` names its run.
///
/// See SPEC.md section 4.2. A `FitnessResult` asserts that baseline and
/// candidate each ran *n* times at a given seed; without a reference to the
/// `SimulationCompleted` that produced those runs, a measured result and a
/// fabricated one are structurally identical and the receiver cannot tell them
/// apart. This is the one place a component reports on work only it can see,
/// which is where the chain has to be checkable rather than conventional.
///
/// The edge is required only when `baseline.runs` is non-zero — a result
/// reporting no runs is EVE declining to measure, and has no run to name.
///
/// Scoped to the corpus on purpose: a binding cannot resolve an id it was never
/// given, so an edge pointing outside the supplied set is not a failure.
fn provenance_edge_problems(
    type_by_id: &std::collections::BTreeMap<String, String>,
    edges: &[(usize, String, String, Vec<String>, u64)],
) -> Vec<Failure> {
    let mut failures = Vec::new();

    for (lineno, doc_type, id, derived, baseline_runs) in edges {
        if derived.contains(id) {
            failures.push(Failure {
                fixture_line: *lineno,
                document_type: doc_type.clone(),
                detail: "derives from itself, which is not a provenance edge".to_string(),
            });
        }

        // A result reporting no runs is the honest encoding of "EVE declined to
        // measure this". There is no simulation for it to name, and demanding
        // one would force it to invent the very reference this rule exists to
        // make meaningful.
        if doc_type != "FitnessResult" || *baseline_runs == 0 {
            continue;
        }

        let names_a_run = derived
            .iter()
            .filter_map(|ref_id| type_by_id.get(ref_id))
            .any(|referenced| referenced == "SimulationCompleted");

        if !names_a_run {
            failures.push(Failure {
                fixture_line: *lineno,
                document_type: doc_type.clone(),
                detail: "provenance.derived_from names no SimulationCompleted; a measurement \
                         that cannot be chained back to its run is indistinguishable from a \
                         fabricated one (SPEC.md section 4.2)"
                    .to_string(),
            });
        }
    }

    failures
}

fn structural_problems(document: &Value) -> Vec<String> {
    let mut problems = Vec::new();

    if document.get("cp").and_then(Value::as_str) != Some("cp1") {
        problems.push("`cp` must be the string \"cp1\"".to_string());
    }
    if document.get("id").and_then(Value::as_str).is_none() {
        problems.push("`id` must be a string".to_string());
    }

    match document.get("provenance").and_then(Value::as_object) {
        None => problems.push("`provenance` must be an object".to_string()),
        Some(provenance) => {
            for member in REQUIRED_PROVENANCE_MEMBERS {
                if !provenance.contains_key(member) {
                    problems.push(format!("provenance is missing `{member}`"));
                }
            }
            if !matches!(
                provenance.get("authored_by").and_then(Value::as_str),
                Some("adam" | "eve" | "axiom")
            ) {
                problems.push(
                    "provenance.authored_by must be one of adam, eve, axiom".to_string(),
                );
            }
        }
    }

    walk_basis_points(document, "$", &mut problems);
    problems
}

/// The only members the schema declares as `signedBasisPoints`.
///
/// Everything else ending in `_bp` is a plain `basisPoints`, whose range starts
/// at zero. Getting this wrong in the permissive direction is worse than it
/// looks: this check is the one vendored into EVE and ADAM, so a range wider
/// than the schema's would let those bindings accept a document the contract
/// forbids and only the normative repository would notice.
const SIGNED_BASIS_POINT_MEMBERS: [&str; 1] = ["delta_bp"];

/// Every member whose name ends in `_bp` must be an integer in range.
///
/// Checked by name rather than by schema position so a new basis-point member
/// added to any type is covered the moment it appears in a fixture — the
/// alternative, an enumerated list, silently fails open on additions.
fn walk_basis_points(value: &Value, path: &str, problems: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let child_path = format!("{path}.{key}");
                if key.ends_with("_bp") {
                    let low = if SIGNED_BASIS_POINT_MEMBERS.contains(&key.as_str()) {
                        -10_000
                    } else {
                        0
                    };
                    match child.as_i64() {
                        None => problems.push(format!(
                            "{child_path} ends in `_bp` and must be an integer, found {child}"
                        )),
                        Some(n) if !(low..=10_000).contains(&n) => problems.push(format!(
                            "{child_path} is {n}, outside the basis-point range [{low}, 10000]"
                        )),
                        Some(_) => {}
                    }
                }
                walk_basis_points(child, &child_path, problems);
            }
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                walk_basis_points(item, &format!("{path}[{i}]"), problems);
            }
        }
        _ => {}
    }
}

/// Check 4: verify a vendored corpus against the normative manifest.
///
/// `manifest` is the contents of `MANIFEST.sha256`; `files` pairs each
/// manifest-relative path with the bytes this repository actually has. Paths
/// the caller did not supply are skipped, so a binding that vendors only the
/// fixtures need not also carry `SPEC.md`.
pub fn check_manifest(manifest: &str, files: &[(&str, &[u8])]) -> Vec<String> {
    let mut failures = Vec::new();
    let mut matched = 0usize;

    for line in manifest.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Some((expected, path)) = line.split_once("  ") else {
            // A manifest is an integrity control. Silently skipping a line we
            // cannot parse would let a truncated or corrupted manifest report
            // success while verifying nothing.
            failures.push(format!(
                "malformed manifest line (expected `<sha256>  <path>`): {line:?}"
            ));
            continue;
        };
        let Some((_, bytes)) = files.iter().find(|(name, _)| *name == path) else {
            continue;
        };
        matched += 1;
        let actual = sha256_hex(bytes);
        if actual != expected {
            failures.push(format!(
                "{path}: manifest records {}…, vendored copy hashes {}…  \
                 (re-vendor from the normative source in AXIOM-AETHER)",
                &expected[..12.min(expected.len())],
                &actual[..12]
            ));
        }
    }

    if matched == 0 {
        failures.push(
            "no supplied file matched any manifest entry; the vendored paths have drifted \
             from the manifest's"
                .to_string(),
        );
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The normative corpus, read from the repository's own protocol directory.
    fn corpus() -> String {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../protocol/cp1/fixtures/canonical.jsonl"
        );
        std::fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("cannot read the CP/1 corpus at {path}: {err}"))
    }

    fn manifest() -> String {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../protocol/cp1/MANIFEST.sha256");
        std::fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("cannot read the CP/1 manifest at {path}: {err}"))
    }

    #[test]
    fn the_normative_corpus_conforms() {
        let failures = check_corpus(&corpus());
        assert!(
            failures.is_empty(),
            "AXIOM's binding disagrees with the normative CP/1 corpus:\n{}",
            failures
                .iter()
                .map(|f| format!("  - {f}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn the_corpus_matches_the_manifest() {
        let corpus = corpus();
        let failures = check_manifest(
            &manifest(),
            &[("fixtures/canonical.jsonl", corpus.as_bytes())],
        );
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[test]
    fn the_corpus_is_not_empty() {
        // A conformance suite that silently passes on an empty corpus is worse
        // than no suite: it reports success while testing nothing.
        assert_eq!(
            corpus().lines().filter(|l| !l.trim().is_empty()).count(),
            15,
            "the corpus should hold one document per covered type, plus the two \
             events: a GenomeCommitted for the envelope and a SimulationCompleted \
             for the FitnessResult to chain back to"
        );
    }

    #[test]
    fn a_fitness_result_that_names_no_run_is_rejected() {
        // The fabricated-evidence case: a well-formed, correctly sealed
        // FitnessResult that points only at the mutation it scores. Nothing
        // about its bytes is wrong — it simply cannot be chained back to any
        // work, which is the whole property check 5 exists to enforce.
        let corpus = corpus()
            .lines()
            .filter(|line| !line.contains("\"SimulationCompleted\""))
            .collect::<Vec<_>>()
            .join("\n");
        let failures = check_corpus(&corpus);
        assert!(
            failures
                .iter()
                .any(|f| f.document_type == "FitnessResult"
                    && f.detail.contains("names no SimulationCompleted")),
            "check 5 did not fire; failures were {failures:#?}"
        );
    }

    #[test]
    fn a_declined_measurement_need_not_name_a_run() {
        // The other half of the rule: EVE reports an unmeasurable mutation with
        // both sides zeroed, and there is no simulation for it to point at.
        // Demanding one would force it to invent the reference.
        let mut declined = serde_json::from_str::<Value>(
                r#"{"baseline":{"cognitive_load_bp":0,"composite_bp":0,"frustration_bp":0,"runs":0,"task_success_bp":0,"trust_bp":0},"candidate":{"cognitive_load_bp":0,"composite_bp":0,"frustration_bp":0,"runs":0,"task_success_bp":0,"trust_bp":0},"cp":"cp1","delta_bp":0,"id":"3b3b3b3b-3b3b-4b3b-8b3b-3b3b3b3b3b3b","mutation_id":"88888888-8888-4888-8888-888888888888","provenance":{"authored_by":"eve","content_hash":"","derived_from":["88888888-8888-4888-8888-888888888888"],"evidence":["runs=0"],"origin":"eve:cp1/validate","produced_at":"2026-01-01T00:00:00.000Z"},"reason":"not measurable by simulation","recommendation":"needs_review","scenario_ids":["excellent"],"seed":1337,"trials":1,"type":"FitnessResult"}"#,
        )
        .unwrap();
        canonical::seal(&mut declined).unwrap();
        let line = canonical::to_canonical(&declined).unwrap();

        let failures = check_corpus(&line);
        assert!(
            !failures
                .iter()
                .any(|f| f.detail.contains("names no SimulationCompleted")),
            "check 5 fired on a declined measurement: {failures:#?}"
        );
    }

    #[test]
    fn a_document_deriving_from_itself_is_rejected() {
        let corpus = corpus().lines().next().unwrap().replace(
            "\"derived_from\":[]",
            "\"derived_from\":[\"11111111-1111-4111-8111-111111111111\"]",
        );
        let failures = check_corpus(&corpus);
        assert!(
            failures.iter().any(|f| f.detail.contains("derives from itself")),
            "{failures:#?}"
        );
    }

    #[test]
    fn a_reordered_document_fails_the_round_trip_check() {
        let corpus = "{\"type\":\"Identity\",\"cp\":\"cp1\",\"id\":\"x\",\"provenance\":{}}";
        let failures = check_corpus(corpus);
        assert!(
            failures.iter().any(|f| f.detail.contains("re-encoding changed the bytes")),
            "{failures:#?}"
        );
    }

    #[test]
    fn a_float_basis_point_is_reported() {
        let corpus = r#"{"confidence_bp":0.82,"cp":"cp1","id":"x","provenance":{"authored_by":"adam","content_hash":"","derived_from":[],"evidence":[],"origin":"o","produced_at":"p"},"type":"Belief"}"#;
        let failures = check_corpus(corpus);
        assert!(
            failures
                .iter()
                .any(|f| f.detail.contains("must be an integer")),
            "{failures:#?}"
        );
    }

    #[test]
    fn a_negative_confidence_is_reported_but_a_negative_delta_is_accepted() {
        // `confidence_bp` is a plain basisPoints and may not be negative;
        // `delta_bp` is the one signed member, because a mutation may make the
        // organism worse and that is the finding.
        let negative_confidence = r#"{"confidence_bp":-5000,"cp":"cp1","id":"x","provenance":{"authored_by":"adam","content_hash":"","derived_from":[],"evidence":[],"origin":"o","produced_at":"p"},"type":"Belief"}"#;
        assert!(check_corpus(negative_confidence)
            .iter()
            .any(|f| f.detail.contains("outside the basis-point range [0, 10000]")));

        let negative_delta = r#"{"cp":"cp1","delta_bp":-700,"id":"x","provenance":{"authored_by":"eve","content_hash":"","derived_from":[],"evidence":[],"origin":"o","produced_at":"p"},"type":"FitnessResult"}"#;
        assert!(!check_corpus(negative_delta)
            .iter()
            .any(|f| f.detail.contains("basis-point range")));
    }

    #[test]
    fn a_malformed_manifest_line_is_reported_rather_than_skipped() {
        let manifest = "this line has no separator\n";
        let failures = check_manifest(manifest, &[("fixtures/canonical.jsonl", b"x")]);
        assert!(
            failures.iter().any(|f| f.contains("malformed manifest line")),
            "{failures:#?}"
        );
    }

    #[test]
    fn an_out_of_range_basis_point_is_reported() {
        let corpus = r#"{"confidence_bp":20000,"cp":"cp1","id":"x","provenance":{"authored_by":"adam","content_hash":"","derived_from":[],"evidence":[],"origin":"o","produced_at":"p"},"type":"Belief"}"#;
        let failures = check_corpus(corpus);
        assert!(
            failures
                .iter()
                .any(|f| f.detail.contains("outside the basis-point range")),
            "{failures:#?}"
        );
    }

    #[test]
    fn a_missing_type_is_reported_as_a_coverage_gap() {
        let failures = check_corpus("");
        assert!(failures.iter().any(|f| f.detail.contains("no fixture covers")));
    }

    #[test]
    fn manifest_check_detects_a_stale_vendored_copy() {
        let manifest = format!("{}  fixtures/canonical.jsonl\n", "0".repeat(64));
        let failures = check_manifest(&manifest, &[("fixtures/canonical.jsonl", b"stale")]);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("re-vendor"));
    }

    #[test]
    fn manifest_check_reports_when_no_path_matches() {
        let manifest = format!("{}  fixtures/canonical.jsonl\n", "0".repeat(64));
        let failures = check_manifest(&manifest, &[("some/other/path.jsonl", b"x")]);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("drifted"));
    }
}
