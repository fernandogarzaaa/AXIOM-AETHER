//! CP/1 canonical form: the single byte-exact encoding a document hashes and
//! signs over.
//!
//! `serde_json::to_string` is *almost* canonical already, but not quite: it
//! preserves whatever key order the value carries, and CP/1 requires keys
//! sorted by UTF-8 code unit. This module walks the value and re-emits it with
//! that ordering imposed, reusing serde_json only for scalar rendering — where
//! its escaping rules (escape `"`, `\`, and `U+0000`–`U+001F`, using the short
//! forms `\b \f \n \r \t` where they exist; leave non-ASCII as literal UTF-8)
//! already match SPEC.md section 2 rule 3 exactly.
//!
//! Floats are rejected rather than rendered. CP/1 puts no floating point on the
//! wire (SPEC.md section 2.1), so encountering one means a caller built a
//! document that will not survive a round trip through a JavaScript binding —
//! failing loudly here is far better than emitting bytes that hash differently
//! on the other side of the boundary.

use serde_json::Value;

use crate::provenance::sha256_hex;

/// Why a value could not be rendered in CP/1 canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalError {
    /// A non-integer number was present. See the module docs.
    FloatNotPermitted { path: String },
    /// A `null` was present. CP/1 writes absent keys instead (SPEC.md 2 rule 6).
    NullNotPermitted { path: String },
}

impl std::fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CanonicalError::FloatNotPermitted { path } => write!(
                f,
                "CP/1 canonical form permits integers only; found a floating-point number at {path}"
            ),
            CanonicalError::NullNotPermitted { path } => write!(
                f,
                "CP/1 canonical form omits absent values rather than writing null; found null at {path}"
            ),
        }
    }
}

impl std::error::Error for CanonicalError {}

/// Render `value` in CP/1 canonical form.
///
/// ```
/// use axiom_engine::cp1::canonical;
/// let value = serde_json::json!({ "b": 1, "a": [2, { "d": 3, "c": 4 }] });
/// assert_eq!(canonical::to_canonical(&value).unwrap(), r#"{"a":[2,{"c":4,"d":3}],"b":1}"#);
/// ```
pub fn to_canonical(value: &Value) -> Result<String, CanonicalError> {
    let mut out = String::new();
    write_value(value, "$", &mut out)?;
    Ok(out)
}

fn write_value(value: &Value, path: &str, out: &mut String) -> Result<(), CanonicalError> {
    match value {
        Value::Null => Err(CanonicalError::NullNotPermitted {
            path: path.to_string(),
        }),
        Value::Bool(b) => {
            out.push_str(if *b { "true" } else { "false" });
            Ok(())
        }
        Value::Number(n) => {
            if n.is_f64() {
                return Err(CanonicalError::FloatNotPermitted {
                    path: path.to_string(),
                });
            }
            out.push_str(&n.to_string());
            Ok(())
        }
        Value::String(s) => {
            // Delegate escaping to serde_json, whose rules match CP/1's.
            out.push_str(&Value::String(s.clone()).to_string());
            Ok(())
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, &format!("{path}[{i}]"), out)?;
            }
            out.push(']');
            Ok(())
        }
        Value::Object(map) => {
            // `serde_json::Map` is a BTreeMap unless the `preserve_order`
            // feature is on, in which case it is insertion-ordered. Sorting
            // explicitly makes this correct either way rather than depending on
            // a feature flag a transitive dependency could switch on.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                write_value(&map[key], &format!("{path}.{key}"), out)?;
            }
            out.push('}');
            Ok(())
        }
    }
}

/// SHA-256 over the canonical form of `document` with
/// `provenance.content_hash` removed.
///
/// A document cannot commit to its own hash, so the member is stripped before
/// hashing and written back afterwards by [`seal`]. Everything else — including
/// timestamps, evidence and `derived_from` — is inside the hash on purpose: the
/// provenance chain is only unforgeable if substituting the evidence changes
/// the hash (SPEC.md section 4.1).
pub fn content_hash(document: &Value) -> Result<String, CanonicalError> {
    let mut unsealed = document.clone();
    if let Some(provenance) = unsealed
        .get_mut("provenance")
        .and_then(Value::as_object_mut)
    {
        provenance.remove("content_hash");
    }
    Ok(sha256_hex(to_canonical(&unsealed)?.as_bytes()))
}

/// Compute and write `provenance.content_hash` into `document`.
///
/// Returns the hash that was written. A document with no `provenance` object is
/// left untouched and reported as an error by [`verify_seal`] rather than being
/// silently accepted here.
pub fn seal(document: &mut Value) -> Result<String, CanonicalError> {
    let hash = content_hash(document)?;
    if let Some(provenance) = document.get_mut("provenance").and_then(Value::as_object_mut) {
        provenance.insert("content_hash".to_string(), Value::String(hash.clone()));
    }
    Ok(hash)
}

/// Whether `document` carries a `provenance.content_hash` equal to its true hash.
pub fn verify_seal(document: &Value) -> Result<bool, CanonicalError> {
    let recorded = document
        .get("provenance")
        .and_then(|p| p.get("content_hash"))
        .and_then(Value::as_str);
    match recorded {
        Some(recorded) => Ok(recorded == content_hash(document)?),
        None => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn object_keys_are_sorted_at_every_depth() {
        let value = json!({ "z": 1, "a": { "y": 2, "b": 3 } });
        assert_eq!(to_canonical(&value).unwrap(), r#"{"a":{"b":3,"y":2},"z":1}"#);
    }

    #[test]
    fn array_order_is_preserved() {
        let value = json!({ "xs": [3, 1, 2] });
        assert_eq!(to_canonical(&value).unwrap(), r#"{"xs":[3,1,2]}"#);
    }

    #[test]
    fn strings_use_short_escapes_and_keep_non_ascii_literal() {
        let value = json!({ "s": "a\"b\\c\nd\te—f" });
        assert_eq!(
            to_canonical(&value).unwrap(),
            "{\"s\":\"a\\\"b\\\\c\\nd\\te—f\"}"
        );
    }

    #[test]
    fn floats_are_rejected_with_their_location() {
        let value = json!({ "outer": { "fitness": 0.5 } });
        assert_eq!(
            to_canonical(&value),
            Err(CanonicalError::FloatNotPermitted {
                path: "$.outer.fitness".to_string()
            })
        );
    }

    #[test]
    fn nulls_are_rejected_with_their_location() {
        let value = json!({ "xs": [1, null] });
        assert_eq!(
            to_canonical(&value),
            Err(CanonicalError::NullNotPermitted {
                path: "$.xs[1]".to_string()
            })
        );
    }

    #[test]
    fn integers_render_without_fraction_or_exponent() {
        let value = json!({ "big": 4294967295u32, "neg": -10000, "zero": 0 });
        assert_eq!(
            to_canonical(&value).unwrap(),
            r#"{"big":4294967295,"neg":-10000,"zero":0}"#
        );
    }

    #[test]
    fn seal_then_verify_round_trips_and_detects_tampering() {
        let mut document = json!({
            "cp": "cp1",
            "type": "Belief",
            "statement": "tests catch regressions",
            "provenance": { "authored_by": "adam", "evidence": [] }
        });

        let hash = seal(&mut document).unwrap();
        assert_eq!(hash.len(), 64);
        assert!(verify_seal(&document).unwrap());

        document["statement"] = json!("tests do not catch regressions");
        assert!(
            !verify_seal(&document).unwrap(),
            "changing the payload must invalidate the seal"
        );
    }

    #[test]
    fn seal_ignores_a_previously_recorded_hash() {
        // Sealing twice must be idempotent: the second seal must not hash the
        // first seal's output, or the operation would not be reproducible.
        let mut document = json!({
            "statement": "x",
            "provenance": { "content_hash": "0".repeat(64) }
        });
        let first = seal(&mut document).unwrap();
        let second = seal(&mut document).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn a_document_without_provenance_never_verifies() {
        let document = json!({ "statement": "x" });
        assert!(!verify_seal(&document).unwrap());
    }
}
