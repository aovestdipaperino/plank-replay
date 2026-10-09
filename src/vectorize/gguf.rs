//! Just enough GGUF to read string metadata and the tensor table from a model
//! header; tensor data is located, never parsed here.

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
///
/// # Errors
/// Fails when the file cannot be read or is not a usable GGUF.
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

/// GGUF `general.alignment` default: tensor data starts on this boundary.
const DEFAULT_ALIGNMENT: u64 = 32;

/// One tensor's entry in the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Its name, such as `blk.10.attn_output_b.weight`.
    pub name: String,
    /// GGML dimensions, fastest-varying first (`ne0` is the row length).
    pub dims: Vec<u64>,
    /// The GGML type id.
    pub ty: u32,
    /// Absolute file offset of its first byte.
    pub offset: u64,
    /// Its size in bytes: exact for the types [`type_layout`] knows, else
    /// the distance to the next tensor (padding included).
    pub bytes: u64,
}

impl TensorInfo {
    /// Number of elements.
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// `(elements per block, bytes per block)` of a GGML type, for the types the
/// dequantizer reads; `None` for the rest (IQ, MXFP4, ...).
#[must_use]
pub fn type_layout(ty: u32) -> Option<(u64, u64)> {
    Some(match ty {
        0 => (1, 4),      // F32
        1 | 30 => (1, 2), // F16, BF16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        _ => return None,
    })
}

/// A GGML type id's name, for messages.
#[must_use]
pub fn type_name(ty: u32) -> String {
    match ty {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        6 => "Q5_0".into(),
        7 => "Q5_1".into(),
        8 => "Q8_0".into(),
        10 => "Q2_K".into(),
        11 => "Q3_K".into(),
        12 => "Q4_K".into(),
        13 => "Q5_K".into(),
        14 => "Q6_K".into(),
        16 => "IQ2_XXS".into(),
        17 => "IQ2_XS".into(),
        18 => "IQ3_XXS".into(),
        19 => "IQ1_S".into(),
        20 => "IQ4_NL".into(),
        21 => "IQ3_S".into(),
        22 => "IQ2_S".into(),
        23 => "IQ4_XS".into(),
        29 => "IQ1_M".into(),
        30 => "BF16".into(),
        39 => "MXFP4".into(),
        other => format!("type {other}"),
    }
}

/// Reads the tensor table of the GGUF at `path`.
///
/// # Errors
/// Fails when the file cannot be read or is not a usable GGUF.
pub fn tensors(path: &Path) -> Result<Vec<TensorInfo>, VectorizeError> {
    let file = std::fs::File::open(path).map_err(|e| VectorizeError::io(path, e))?;
    let len = file
        .metadata()
        .map_err(|e| VectorizeError::io(path, e))?
        .len();
    let mut r = BufReader::new(file);
    read_tensors(&mut r, len).map_err(|e| match e {
        Failure::Io(e) => VectorizeError::io(path, e),
        Failure::Format(why) => {
            VectorizeError::msg(format!("{}: not a usable GGUF file: {why}", path.display()))
        }
    })
}

fn read_tensors<R: Read + Seek>(r: &mut R, file_len: u64) -> Result<Vec<TensorInfo>, Failure> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(Failure::Format("bad magic"));
    }
    if u32_le(r)? < 2 {
        return Err(Failure::Format("GGUF v1 is not supported"));
    }
    let tensor_count = u64_le(r)?;
    let kv_count = u64_le(r)?;
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..kv_count {
        let name = string(r)?;
        let ty = u32_le(r)?;
        if name == "general.alignment" && ty == 4 {
            alignment = u64::from(u32_le(r)?).max(1);
        } else {
            skip_value(r, ty)?;
        }
    }
    if tensor_count > 1 << 24 {
        return Err(Failure::Format("too many tensors"));
    }
    let mut infos = Vec::new();
    for _ in 0..tensor_count {
        let name = string(r)?;
        let n_dims = u32_le(r)?;
        if n_dims > 8 {
            return Err(Failure::Format("too many dimensions"));
        }
        let dims = (0..n_dims)
            .map(|_| u64_le(r))
            .collect::<Result<Vec<_>, _>>()?;
        let ty = u32_le(r)?;
        let offset = u64_le(r)?;
        infos.push(TensorInfo {
            name,
            dims,
            ty,
            offset,
            bytes: 0,
        });
    }
    let data = r.stream_position()?.div_ceil(alignment) * alignment;
    let mut order: Vec<usize> = (0..infos.len()).collect();
    order.sort_by_key(|&i| infos[i].offset);
    for (k, &i) in order.iter().enumerate() {
        let next = order
            .get(k + 1)
            .map_or(file_len, |&j| data + infos[j].offset);
        let info = &mut infos[i];
        info.offset += data;
        let exact = type_layout(info.ty).and_then(|(block, size)| {
            let n = info.elements();
            n.is_multiple_of(block).then(|| n / block * size)
        });
        info.bytes = exact.unwrap_or_else(|| next.saturating_sub(info.offset));
        if info.offset + info.bytes > file_len {
            return Err(Failure::Format("a tensor runs past the end of the file"));
        }
    }
    Ok(infos)
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
