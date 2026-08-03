//! CP/1 transport: line-delimited JSON and the signed envelope.
//!
//! The envelope carries its payload as a **string**, not a nested object, so
//! the bytes that were hashed are exactly the bytes transmitted. Nesting the
//! document would let the receiver's JSON writer re-render it — different key
//! order, different escaping — and invalidate a hash that was perfectly valid
//! when it was computed.
//!
//! Signing reuses [`crate::provenance`], which already implements the
//! SHA-256 + optional HMAC-SHA256 construction CP/1 specifies. This module adds
//! the CP/1 framing and the line protocol; it does not reimplement the crypto.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::canonical::{self, CanonicalError};
use crate::provenance::{hmac_sha256_hex, sha256_hex};

pub const ENVELOPE_SCHEMA: &str = "cp1_signed_envelope";

/// A CP/1 document wrapped for transport.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedEnvelope {
    pub cp: String,
    pub schema: String,
    /// Canonical-form JSON of the document, as a string.
    pub payload: String,
    /// SHA-256 of `payload`'s UTF-8 bytes.
    pub sha256: String,
    /// HMAC-SHA256 over `sha256`, keyed by the fleet secret. Optional: over a
    /// stdio subprocess boundary the parent already controls the child, so
    /// requiring a shared secret there would be ceremony without a threat.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hmac: Option<String>,
}

/// Why an envelope was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    /// Not a CP/1 envelope at all.
    BadSchema,
    /// `payload` does not hash to `sha256`: the payload was altered in transit.
    HashMismatch,
    /// A fleet key is configured but the envelope is unsigned.
    SignatureMissing,
    /// The HMAC does not verify under the configured fleet key.
    SignatureInvalid,
    /// The envelope is signed but this node has no key to check it with.
    NoKeyToVerify,
    /// `payload` is not valid JSON.
    MalformedPayload,
    /// The payload parsed but is not in canonical form, so its `content_hash`
    /// could not have been computed over what was sent.
    NotCanonical,
    /// The document's `provenance.content_hash` does not match its content.
    SealBroken,
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            EnvelopeError::BadSchema => "not a CP/1 signed envelope",
            EnvelopeError::HashMismatch => "payload hash mismatch (tampered in transit)",
            EnvelopeError::SignatureMissing => "fleet key configured but envelope is unsigned",
            EnvelopeError::SignatureInvalid => "HMAC signature does not verify",
            EnvelopeError::NoKeyToVerify => "envelope is signed but no fleet key is configured",
            EnvelopeError::MalformedPayload => "payload is not valid JSON",
            EnvelopeError::NotCanonical => "payload is not in CP/1 canonical form",
            EnvelopeError::SealBroken => "document content_hash does not match its content",
        };
        f.write_str(message)
    }
}

impl std::error::Error for EnvelopeError {}

impl SignedEnvelope {
    /// Wrap a document, sealing it first if it is not already sealed.
    pub fn seal(document: &Value, fleet_key: Option<&[u8]>) -> Result<Self, CanonicalError> {
        let mut document = document.clone();
        canonical::seal(&mut document)?;
        let payload = canonical::to_canonical(&document)?;
        let sha256 = sha256_hex(payload.as_bytes());
        let hmac = fleet_key.map(|key| hmac_sha256_hex(key, sha256.as_bytes()));
        Ok(Self {
            cp: "cp1".to_string(),
            schema: ENVELOPE_SCHEMA.to_string(),
            payload,
            sha256,
            hmac,
        })
    }

    /// Verify the envelope and return the document it carries.
    ///
    /// Checks run outermost-first — schema, transport hash, signature, then the
    /// document's own seal — so the cheapest rejection happens first and an
    /// attacker learns the least from which error came back.
    pub fn open(&self, fleet_key: Option<&[u8]>) -> Result<Value, EnvelopeError> {
        if self.cp != "cp1" || self.schema != ENVELOPE_SCHEMA {
            return Err(EnvelopeError::BadSchema);
        }
        if sha256_hex(self.payload.as_bytes()) != self.sha256 {
            return Err(EnvelopeError::HashMismatch);
        }
        match (fleet_key, &self.hmac) {
            (Some(key), Some(mac)) => {
                let expected = hmac_sha256_hex(key, self.sha256.as_bytes());
                if !constant_time_eq(expected.as_bytes(), mac.as_bytes()) {
                    return Err(EnvelopeError::SignatureInvalid);
                }
            }
            (Some(_), None) => return Err(EnvelopeError::SignatureMissing),
            (None, Some(_)) => return Err(EnvelopeError::NoKeyToVerify),
            (None, None) => {}
        }

        let document: Value =
            serde_json::from_str(&self.payload).map_err(|_| EnvelopeError::MalformedPayload)?;
        if canonical::to_canonical(&document).map_err(|_| EnvelopeError::NotCanonical)?
            != self.payload
        {
            return Err(EnvelopeError::NotCanonical);
        }
        if !canonical::verify_seal(&document).map_err(|_| EnvelopeError::NotCanonical)? {
            return Err(EnvelopeError::SealBroken);
        }
        Ok(document)
    }

    /// Render as one line of the line-delimited JSON transport.
    ///
    /// The envelope itself is emitted in canonical form too, so a reader can
    /// hash the whole line for a transport-level audit trail.
    pub fn to_line(&self) -> String {
        let value = serde_json::to_value(self).expect("SignedEnvelope always serializes");
        canonical::to_canonical(&value)
            .expect("SignedEnvelope contains only strings")
    }

    /// Parse one line of the line-delimited JSON transport.
    pub fn from_line(line: &str) -> Result<Self, EnvelopeError> {
        serde_json::from_str(line).map_err(|_| EnvelopeError::BadSchema)
    }
}

/// Compare two byte strings without an early exit on the first difference.
///
/// Signature comparison with `==` leaks, through timing, how many leading bytes
/// of a forged MAC were correct — enough to reconstruct one byte at a time.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn document() -> Value {
        json!({
            "cp": "cp1",
            "type": "Belief",
            "id": "55555555-5555-4555-8555-555555555555",
            "statement": "tests catch regressions",
            "provenance": {
                "authored_by": "adam",
                "produced_at": "2026-01-01T00:00:00.000Z",
                "origin": "reasoning",
                "evidence": [],
                "derived_from": []
            }
        })
    }

    #[test]
    fn unsigned_round_trip_returns_the_same_document() {
        let envelope = SignedEnvelope::seal(&document(), None).unwrap();
        let opened = envelope.open(None).unwrap();
        assert_eq!(opened["statement"], json!("tests catch regressions"));
        assert!(canonical::verify_seal(&opened).unwrap());
    }

    #[test]
    fn signed_round_trip_verifies_under_the_same_key() {
        let key = b"fleet-secret";
        let envelope = SignedEnvelope::seal(&document(), Some(key)).unwrap();
        assert!(envelope.hmac.is_some());
        assert!(envelope.open(Some(key)).is_ok());
    }

    #[test]
    fn a_different_key_does_not_verify() {
        let envelope = SignedEnvelope::seal(&document(), Some(b"fleet-secret")).unwrap();
        assert_eq!(
            envelope.open(Some(b"other-secret")),
            Err(EnvelopeError::SignatureInvalid)
        );
    }

    #[test]
    fn an_unsigned_envelope_is_refused_when_a_key_is_required() {
        let envelope = SignedEnvelope::seal(&document(), None).unwrap();
        assert_eq!(
            envelope.open(Some(b"fleet-secret")),
            Err(EnvelopeError::SignatureMissing)
        );
    }

    #[test]
    fn a_signed_envelope_is_refused_when_no_key_is_configured() {
        let envelope = SignedEnvelope::seal(&document(), Some(b"fleet-secret")).unwrap();
        assert_eq!(envelope.open(None), Err(EnvelopeError::NoKeyToVerify));
    }

    #[test]
    fn tampering_with_the_payload_is_caught_by_the_transport_hash() {
        let mut envelope = SignedEnvelope::seal(&document(), None).unwrap();
        envelope.payload = envelope.payload.replace("catch", "miss");
        assert_eq!(envelope.open(None), Err(EnvelopeError::HashMismatch));
    }

    #[test]
    fn a_consistently_rehashed_payload_is_still_caught_by_the_document_seal() {
        // The interesting attack: an attacker who edits the payload *and*
        // recomputes the transport hash. The document's own content_hash is
        // what stops them, which is why CP/1 has both.
        let mut envelope = SignedEnvelope::seal(&document(), None).unwrap();
        envelope.payload = envelope.payload.replace("catch", "misss");
        envelope.sha256 = sha256_hex(envelope.payload.as_bytes());
        assert_eq!(envelope.open(None), Err(EnvelopeError::SealBroken));
    }

    #[test]
    fn a_non_canonical_payload_is_refused() {
        let envelope = SignedEnvelope::seal(&document(), None).unwrap();
        let reordered = format!(" {}", envelope.payload);
        let tampered = SignedEnvelope {
            sha256: sha256_hex(reordered.as_bytes()),
            payload: reordered,
            ..envelope
        };
        assert_eq!(tampered.open(None), Err(EnvelopeError::NotCanonical));
    }

    #[test]
    fn a_foreign_schema_is_refused_before_anything_else() {
        let mut envelope = SignedEnvelope::seal(&document(), None).unwrap();
        envelope.schema = "axiom_signed_export_v1".to_string();
        assert_eq!(envelope.open(None), Err(EnvelopeError::BadSchema));
    }

    #[test]
    fn line_protocol_round_trips_without_embedded_newlines() {
        let envelope = SignedEnvelope::seal(&document(), Some(b"k")).unwrap();
        let line = envelope.to_line();
        assert!(!line.contains('\n'), "a line must not contain a newline");
        assert_eq!(SignedEnvelope::from_line(&line).unwrap(), envelope);
    }

    #[test]
    fn constant_time_eq_matches_ordinary_equality() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }
}
