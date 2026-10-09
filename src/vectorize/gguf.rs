//! Just enough GGUF to read string metadata from a model header.

use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use super::VectorizeError;

/// GGUF value type ids, from the format spec.
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;

/// Size in bytes of each fixed-width GGUF value type.
fn fixed_size(ty: u32) -> Option<u64> {
    Some(match ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

/// Returns the string value stored under `key`, or `None` if it is absent.
///
/// Only the key/value header is read; tensor data is never touched, so this
/// is cheap even on a multi-hundred-gigabyte model.
pub fn string_value(path: &Path, key: &str) -> Result<Option<String>, VectorizeError> {
    let file = std::fs::File::open(path).map_err(|e| VectorizeError::io(path, e))?;
    let mut r = BufReader::new(file);
    read_string_value(&mut r, key).map_err(|e| match e {
        Failure::Io(e) => VectorizeError::io(path, e),
        Failure::Format(why) => {
            VectorizeError::msg(format!("{}: not a usable GGUF file: {why}", path.display()))
        }
    })
}

/// Why the header could not be read.
#[derive(Debug)]
enum Failure {
    Io(std::io::Error),
    Format(&'static str),
}

impl From<std::io::Error> for Failure {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

fn read_string_value<R: Read + Seek>(r: &mut R, key: &str) -> Result<Option<String>, Failure> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(Failure::Format("bad magic"));
    }
    let version = u32_le(r)?;
    if version < 2 {
        return Err(Failure::Format("GGUF v1 is not supported"));
    }
    let _tensors = u64_le(r)?;
    let kv_count = u64_le(r)?;
    for _ in 0..kv_count {
        let name = string(r)?;
        let ty = u32_le(r)?;
        if name == key && ty == T_STRING {
            return Ok(Some(string(r)?));
        }
        skip_value(r, ty)?;
    }
    Ok(None)
}

fn skip_value<R: Read + Seek>(r: &mut R, ty: u32) -> Result<(), Failure> {
    match ty {
        T_STRING => {
            let len = u64_le(r)?;
            skip(r, len)
        }
        T_ARRAY => {
            let elem = u32_le(r)?;
            let count = u64_le(r)?;
            if let Some(size) = fixed_size(elem) {
                let bytes = count
                    .checked_mul(size)
                    .ok_or(Failure::Format("array too large"))?;
                skip(r, bytes)
            } else {
                for _ in 0..count {
                    skip_value(r, elem)?;
                }
                Ok(())
            }
        }
        other => skip(
            r,
            fixed_size(other).ok_or(Failure::Format("unknown value type"))?,
        ),
    }
}

fn skip<R: Seek>(r: &mut R, bytes: u64) -> Result<(), Failure> {
    let bytes = i64::try_from(bytes).map_err(|_| Failure::Format("length overflow"))?;
    r.seek(SeekFrom::Current(bytes))?;
    Ok(())
}

fn string<R: Read>(r: &mut R) -> Result<String, Failure> {
    let len = u64_le(r)?;
    if len > 1 << 20 {
        return Err(Failure::Format("string too long"));
    }
    let mut buf = vec![0u8; usize::try_from(len).map_err(|_| Failure::Format("length overflow"))?];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| Failure::Format("string is not UTF-8"))
}

fn u32_le<R: Read>(r: &mut R) -> Result<u32, Failure> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn u64_le<R: Read>(r: &mut R) -> Result<u64, Failure> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// A header with a u32, a string array, a f32 array, then the wanted key.
    fn header() -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&4u64.to_le_bytes());
        push_str(&mut b, "x.block_count");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&43u32.to_le_bytes());
        push_str(&mut b, "tokenizer.tokens");
        b.extend_from_slice(&T_ARRAY.to_le_bytes());
        b.extend_from_slice(&T_STRING.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        push_str(&mut b, "a");
        push_str(&mut b, "bc");
        push_str(&mut b, "scores");
        b.extend_from_slice(&T_ARRAY.to_le_bytes());
        b.extend_from_slice(&6u32.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes());
        b.extend_from_slice(&[0u8; 12]);
        push_str(&mut b, "general.architecture");
        b.extend_from_slice(&T_STRING.to_le_bytes());
        push_str(&mut b, "deepseek4");
        b
    }

    #[test]
    fn finds_a_string_after_skipping_scalars_and_arrays() {
        let got = read_string_value(&mut Cursor::new(header()), "general.architecture").unwrap();
        assert_eq!(got.as_deref(), Some("deepseek4"));
    }

    #[test]
    fn a_missing_key_is_none() {
        let got = read_string_value(&mut Cursor::new(header()), "general.name").unwrap();
        assert_eq!(got, None);
    }

    #[test]
    fn a_non_gguf_file_is_rejected() {
        let err = read_string_value(&mut Cursor::new(b"nope....".to_vec()), "k").unwrap_err();
        assert!(matches!(err, Failure::Format("bad magic")));
    }
}
