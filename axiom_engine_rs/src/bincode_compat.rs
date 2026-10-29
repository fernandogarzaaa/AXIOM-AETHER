//! bincode 2.x shims preserving the bincode 1.3 wire format.
//!
//! bincode 1.3 is unmaintained, so we depend on bincode 2 with the `serde`
//! feature. bincode 2's default (`standard()`) encoding uses variable-length
//! integers and is NOT byte-compatible with 1.3. We use
//! [`bincode::config::legacy()`] instead, which reproduces the 1.3 default
//! encoding (little-endian, fixed-width ints) byte-for-byte (verified by test
//! below).
//!
//! Wire stability matters here:
//! - `crate::dwe::fragment_preimage` feeds serialized bytes into HMAC
//!   signatures; a format change would invalidate existing signatures.
//! - `PersistedCompressionCache` blobs are written to disk and read back;
//!   byte-identical encoding means old cache files keep decoding.

use serde::{de::Deserialize, Serialize};

/// Serialize `value`, mirroring the old `bincode::serialize` signature.
pub fn serialize<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, bincode::error::EncodeError> {
    bincode::serde::encode_to_vec(value, bincode::config::legacy())
}

/// Deserialize from `bytes`, mirroring the old `bincode::deserialize` signature.
///
/// Accepts borrowed types like `&str` (via `Deserialize<'a>`), matching the
/// bincode 1.3 API. The returned value borrows from `bytes`.
pub fn deserialize<'a, T: Deserialize<'a>>(
    bytes: &'a [u8],
) -> Result<T, bincode::error::DecodeError> {
    bincode::serde::borrow_decode_from_slice(bytes, bincode::config::legacy())
        .map(|(value, _)| value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Sample {
        schema: String,
        sequence: u64,
        layers: Vec<Vec<f32>>,
        hash: Option<String>,
        flag: bool,
    }

    fn sample() -> Sample {
        Sample {
            schema: "axiom.dwe.v1".into(),
            sequence: 42,
            layers: vec![vec![1.0, 2.5, -3.75], vec![], vec![0.0; 16]],
            hash: Some("abc123".into()),
            flag: true,
        }
    }

    #[test]
    fn round_trip() {
        let v = sample();
        let bytes = serialize(&v).unwrap();
        let back: Sample = deserialize(&bytes).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn tuple_round_trip_like_fragment_preimage() {
        let shadow = (
            "axiom.dwe.v1".to_string(),
            "sess-1".to_string(),
            7u64,
            vec![vec![1.5f32]],
            "deadbeef".to_string(),
        );
        let bytes = serialize(&shadow).unwrap();
        let back: (String, String, u64, Vec<Vec<f32>>, String) = deserialize(&bytes).unwrap();
        assert_eq!(shadow, back);
    }

    /// Golden bytes produced by bincode 1.3.3 for `sample()` above.
    ///
    /// Generated once with bincode 1.3.3; this test fails if the encoding
    /// ever drifts from the 1.3 wire format (e.g. someone swaps `legacy()`
    /// for `standard()`), which would break HMAC preimages and on-disk
    /// caches.
    #[test]
    fn wire_format_matches_bincode_1_3() {
        #[rustfmt::skip]
        let expected: [u8; 152] = [
           12, 0, 0, 0, 0, 0, 0, 0, 97, 120, 105, 111,
           109, 46, 100, 119, 101, 46, 118, 49, 42, 0, 0, 0,
           0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
           3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63,
           0, 0, 32, 64, 0, 0, 112, 192, 0, 0, 0, 0,
           0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 6, 0, 0, 0, 0, 0, 0,
           0, 97, 98, 99, 49, 50, 51, 1,
        ];
        let bytes = serialize(&sample()).unwrap();
        assert_eq!(bytes, expected, "wire format drifted from bincode 1.3");
        // Decode the independently fixed 1.3 bytes to prove the decoder
        // reads them, not just that our encoder reproduces them.
        let back: Sample = deserialize(&expected).unwrap();
        assert_eq!(back, sample(), "decoder failed on bincode 1.3 golden bytes");
    }
}
