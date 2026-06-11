//! Tamper-evident provenance for swarm exchange — verify-before-trust.
//!
//! Ported and hardened from chimeralang-mcp's replay/protocol/integrity modules.
//! Chimera's idea — *cryptographic equality is the trust primitive* — is exactly
//! what Axiom's swarm-immunity merge was missing (`/v1/immunity/merge` trusted
//! any peer's JSON blindly). Two gaps in the original are fixed here:
//!   1. Chimera truncated its hashes to 128-bit (`[:32]` hex). We use the full
//!      256-bit SHA-256.
//!   2. Chimera had *no peer authentication* (documented: "no shared secret, no
//!      signature"). We add an optional HMAC-SHA256 over the hash, keyed by a
//!      shared fleet secret, so a node can require that an export came from a
//!      trusted peer — not merely that it is internally consistent.
//!
//! A [`SignedExport`] wraps a payload (e.g. an exported heal memory) with its
//! SHA-256 and, optionally, an HMAC. [`verify_export`] gates a merge: the hash
//! must match the payload bytes, and — when a fleet key is configured — the
//! HMAC must verify. No new dependencies: HMAC is built over `sha2` directly.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A payload wrapped with tamper-evident provenance for cross-node transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedExport {
    /// Schema marker so a receiver can tell a signed export from a raw payload.
    pub schema: String,
    /// The exact payload bytes the hash/HMAC are computed over.
    pub payload: String,
    /// Lowercase hex SHA-256 of `payload` (full 256-bit).
    pub sha256: String,
    /// Optional hex HMAC-SHA256 over `sha256`, keyed by the fleet secret.
    #[serde(default)]
    pub hmac: Option<String>,
}

pub const SCHEMA: &str = "axiom_signed_export_v1";

/// Why a [`SignedExport`] was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceError {
    /// `payload` does not hash to `sha256` — the payload was tampered with.
    HashMismatch,
    /// A fleet key is configured but the export carries no HMAC.
    SignatureMissing,
    /// The HMAC does not verify under the configured fleet key.
    SignatureInvalid,
    /// The export carries an HMAC but this node has no key to verify it.
    NoKeyToVerify,
    /// Unrecognised schema.
    BadSchema,
}

impl std::fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ProvenanceError::HashMismatch => "payload hash mismatch (tampered)",
            ProvenanceError::SignatureMissing => "fleet key configured but export is unsigned",
            ProvenanceError::SignatureInvalid => "HMAC signature does not verify",
            ProvenanceError::NoKeyToVerify => "export is signed but no fleet key is configured",
            ProvenanceError::BadSchema => "unrecognised export schema",
        };
        f.write_str(s)
    }
}

/// SHA-256 of `bytes` as lowercase hex (full 256-bit).
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// HMAC-SHA256(key, msg) as lowercase hex (RFC 2104, built over `sha2`).
pub fn hmac_sha256_hex(key: &[u8], msg: &[u8]) -> String {
    const BLOCK: usize = 64;
    let mut block = [0u8; BLOCK];
    if key.len() > BLOCK {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= block[i];
        opad[i] ^= block[i];
    }
    let inner = Sha256::new().chain_update(ipad).chain_update(msg).finalize();
    let outer = Sha256::new().chain_update(opad).chain_update(inner).finalize();
    let mut out = String::with_capacity(64);
    for b in outer {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Constant-time equality for two hex strings of equal length (avoids leaking
/// where a signature first differs).
fn ct_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.bytes().zip(b.bytes()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Wrap `payload` with provenance. When `fleet_key` is `Some`, an HMAC over the
/// SHA-256 is added so receivers can authenticate the sender.
pub fn sign_export(payload: &str, fleet_key: Option<&[u8]>) -> SignedExport {
    let sha256 = sha256_hex(payload.as_bytes());
    let hmac = fleet_key.map(|k| hmac_sha256_hex(k, sha256.as_bytes()));
    SignedExport {
        schema: SCHEMA.to_string(),
        payload: payload.to_string(),
        sha256,
        hmac,
    }
}

/// Verify a [`SignedExport`] before its payload is trusted.
///
/// Always checks the SHA-256 integrity. When `fleet_key` is `Some`, the HMAC is
/// required and must verify (authenticity). When `fleet_key` is `None`, a signed
/// export is rejected with [`ProvenanceError::NoKeyToVerify`] so a node that
/// expects unsigned exports cannot be fed a (forgeable-by-anyone) HMAC it cannot
/// check. Returns the verified payload on success.
pub fn verify_export<'a>(
    export: &'a SignedExport,
    fleet_key: Option<&[u8]>,
) -> Result<&'a str, ProvenanceError> {
    if export.schema != SCHEMA {
        return Err(ProvenanceError::BadSchema);
    }
    // 1. Integrity: the payload must hash to the claimed digest.
    if !ct_eq(&sha256_hex(export.payload.as_bytes()), &export.sha256) {
        return Err(ProvenanceError::HashMismatch);
    }
    // 2. Authenticity (when a fleet key is in play).
    match (fleet_key, export.hmac.as_deref()) {
        (Some(key), Some(mac)) => {
            let expected = hmac_sha256_hex(key, export.sha256.as_bytes());
            if !ct_eq(&expected, mac) {
                return Err(ProvenanceError::SignatureInvalid);
            }
        }
        (Some(_), None) => return Err(ProvenanceError::SignatureMissing),
        (None, Some(_)) => return Err(ProvenanceError::NoKeyToVerify),
        (None, None) => {}
    }
    Ok(&export.payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_sha256_not_truncated() {
        assert_eq!(sha256_hex(b"").len(), 64, "full 256-bit hex, not Chimera's 32");
        // Known SHA-256("abc").
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hmac_matches_rfc_test_vector() {
        // RFC 4231 Test Case 2: key="Jefe", data="what do ya want for nothing?"
        let mac = hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            mac,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn unsigned_roundtrip_verifies() {
        let e = sign_export("hello memory", None);
        assert_eq!(verify_export(&e, None).unwrap(), "hello memory");
    }

    #[test]
    fn tampered_payload_is_rejected() {
        let mut e = sign_export("original", None);
        e.payload = "tampered".to_string();
        assert_eq!(verify_export(&e, None), Err(ProvenanceError::HashMismatch));
    }

    #[test]
    fn signed_roundtrip_verifies_with_key() {
        let key = b"fleet-secret";
        let e = sign_export("payload", Some(key));
        assert_eq!(verify_export(&e, Some(key)).unwrap(), "payload");
    }

    #[test]
    fn wrong_key_is_rejected() {
        let e = sign_export("payload", Some(b"right-key"));
        assert_eq!(
            verify_export(&e, Some(b"wrong-key")),
            Err(ProvenanceError::SignatureInvalid)
        );
    }

    #[test]
    fn signed_export_rejected_when_node_has_no_key() {
        let e = sign_export("payload", Some(b"k"));
        assert_eq!(verify_export(&e, None), Err(ProvenanceError::NoKeyToVerify));
    }

    #[test]
    fn unsigned_export_rejected_when_key_required() {
        let e = sign_export("payload", None);
        assert_eq!(
            verify_export(&e, Some(b"k")),
            Err(ProvenanceError::SignatureMissing)
        );
    }
}
