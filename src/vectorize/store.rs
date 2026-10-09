//! Named steering vectors kept in `~/.plank/models/vectors.json`.
//!
//! The file is a JSON array with one entry per model, each holding its
//! named vectors as base64 of the raw little-endian `f32` matrix:
//!
//! ```json
//! [
//!   {
//!     "model": "ds4vision",
//!     "vectors": [
//!       { "name": "succinct", "value": "AACAPwAAAAA..." }
//!     ]
//!   }
//! ]
//! ```
//!
//! Fields this module does not know are kept as they are, so other tools can
//! annotate entries without losing their notes on the next write.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::{Direction, VectorizeError};

/// A `vectors.json` file of named steering vectors.
#[derive(Debug, Clone)]
pub struct VectorStore {
    path: PathBuf,
}

impl VectorStore {
    /// The store at `path`; the file need not exist yet.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// `models/vectors.json` under the plank home (`~/.plank`).
    #[must_use]
    pub fn default_path() -> PathBuf {
        super::plank_dir().join("models").join("vectors.json")
    }

    /// The file this store reads and writes.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether `model` already has a vector called `name`.
    ///
    /// # Errors
    /// Fails when the file exists but is not a valid store.
    pub fn contains(&self, model: &str, name: &str) -> Result<bool, VectorizeError> {
        let entries = self.read()?;
        Ok(entries
            .iter()
            .filter(|e| e.get("model").and_then(Value::as_str) == Some(model))
            .filter_map(|e| e.get("vectors").and_then(Value::as_array))
            .flatten()
            .any(|v| v.get("name").and_then(Value::as_str) == Some(name)))
    }

    /// Stores `direction` as `name` under `model`, creating the file and its
    /// directory when missing. Returns `true` if an existing vector was
    /// replaced.
    ///
    /// # Errors
    /// Fails when the pair exists and `force` is false, when the file is not a
    /// valid store, or when it cannot be written.
    pub fn put(
        &self,
        model: &str,
        name: &str,
        direction: &Direction,
        force: bool,
    ) -> Result<bool, VectorizeError> {
        let mut entries = self.read()?;
        let value = Value::String(base64(&direction.to_le_bytes()));
        let fresh_vector = || json!({ "name": name, "value": value.clone() });

        let index = if let Some(i) = entries
            .iter()
            .position(|e| e.get("model").and_then(Value::as_str) == Some(model))
        {
            i
        } else {
            entries.push(json!({ "model": model, "vectors": [] }));
            entries.len() - 1
        };
        let entry = &mut entries[index];
        let vectors = entry
            .as_object_mut()
            .ok_or_else(|| self.invalid("an entry is not an object"))?
            .entry("vectors")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| self.invalid(&format!("`vectors` of `{model}` is not an array")))?;

        let replaced = match vectors
            .iter_mut()
            .find(|v| v.get("name").and_then(Value::as_str) == Some(name))
        {
            Some(_) if !force => {
                return Err(VectorizeError::msg(format!(
                    "{}: `{model}` already has a vector named `{name}`; \
                     pass --force to replace it",
                    self.path.display()
                )));
            }
            Some(existing) => {
                match existing.as_object_mut() {
                    Some(fields) => {
                        fields.insert("value".into(), value.clone());
                    }
                    None => *existing = fresh_vector(),
                }
                true
            }
            None => {
                vectors.push(fresh_vector());
                false
            }
        };
        self.write(&entries)?;
        Ok(replaced)
    }

    /// The entries in the file; a missing or empty file is an empty store.
    fn read(&self) -> Result<Vec<Value>, VectorizeError> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(VectorizeError::io(&self.path, e)),
        };
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        match serde_json::from_str(&text) {
            Ok(Value::Array(entries)) => {
                if entries.iter().all(Value::is_object) {
                    Ok(entries)
                } else {
                    Err(self.invalid("every entry must be an object"))
                }
            }
            Ok(_) => Err(self.invalid("the top level must be a JSON array")),
            Err(e) => Err(self.invalid(&e.to_string())),
        }
    }

    /// Replaces the file through a temporary sibling, so a crash mid-write
    /// never leaves a truncated store.
    fn write(&self, entries: &[Value]) -> Result<(), VectorizeError> {
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| VectorizeError::io(dir, e))?;
        }
        let text = serde_json::to_string_pretty(entries)
            .map_err(|e| VectorizeError::msg(e.to_string()))?
            + "\n";
        let tmp = self
            .path
            .with_extension(format!("json.tmp-{}", std::process::id()));
        std::fs::write(&tmp, text).map_err(|e| VectorizeError::io(&tmp, e))?;
        std::fs::rename(&tmp, &self.path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            VectorizeError::io(&self.path, e)
        })
    }

    fn invalid(&self, why: &str) -> VectorizeError {
        VectorizeError::msg(format!(
            "{}: not a vector store ({why}); fix or move it aside",
            self.path.display()
        ))
    }
}

/// Standard base64 with padding.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            if i <= chunk.len() {
                out.push(char::from(ALPHABET[(n >> shift & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorize::{Accumulator, Profile};

    const TINY: Profile = Profile {
        name: "tiny",
        layers: 1,
        width: 2,
        residual_dump: "ffn_out",
        residual_branches: 1,
    };

    fn direction(x: f32) -> Direction {
        let mut acc = Accumulator::new(TINY);
        acc.add_pair(&[x, 0.0], &[0.0, 0.0]);
        acc.finish(false, false)
    }

    fn temp_store(label: &str) -> VectorStore {
        let dir = std::env::temp_dir().join(format!(
            "plank-tools-test-store-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        VectorStore::new(dir.join("models").join("vectors.json"))
    }

    #[test]
    fn base64_matches_the_rfc_vectors() {
        for (raw, enc) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(raw.as_bytes()), enc);
        }
    }

    #[test]
    fn the_first_put_creates_the_file_and_its_directory() {
        let store = temp_store("create");
        assert!(!store.contains("m.gguf", "v").unwrap());
        assert!(!store.put("m.gguf", "v", &direction(1.0), false).unwrap());
        assert!(store.contains("m.gguf", "v").unwrap());
        let json: Value =
            serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        // [1.0, 0.0] as little-endian f32.
        assert_eq!(json[0]["model"], "m.gguf");
        assert_eq!(json[0]["vectors"][0]["name"], "v");
        assert_eq!(
            json[0]["vectors"][0]["value"],
            base64(&[0, 0, 0x80, 0x3f, 0, 0, 0, 0])
        );
    }

    #[test]
    fn an_existing_pair_needs_force_and_other_fields_survive() {
        let store = temp_store("force");
        std::fs::create_dir_all(store.path().parent().unwrap()).unwrap();
        std::fs::write(
            store.path(),
            r#"[{"model":"m.gguf","note":"keep","vectors":[{"name":"v","value":"x","scale":2},{"name":"w","value":"y"}]},{"model":"other.gguf","vectors":[]}]"#,
        )
        .unwrap();
        let err = store
            .put("m.gguf", "v", &direction(1.0), false)
            .unwrap_err();
        assert!(err.to_string().contains("--force"), "{err}");
        assert!(store.put("m.gguf", "v", &direction(1.0), true).unwrap());

        let json: Value =
            serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(json[0]["note"], "keep");
        assert_eq!(json[0]["vectors"][0]["scale"], 2);
        assert_ne!(json[0]["vectors"][0]["value"], "x");
        assert_eq!(json[0]["vectors"][1]["value"], "y");
        assert_eq!(json[1]["model"], "other.gguf");
    }

    #[test]
    fn a_new_name_or_model_is_appended() {
        let store = temp_store("append");
        store.put("a.gguf", "v", &direction(1.0), false).unwrap();
        store.put("a.gguf", "w", &direction(1.0), false).unwrap();
        store.put("b.gguf", "v", &direction(1.0), false).unwrap();
        let json: Value =
            serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(json[0]["vectors"].as_array().unwrap().len(), 2);
        assert_eq!(json[1]["model"], "b.gguf");
    }

    #[test]
    fn a_file_of_another_shape_is_refused_not_clobbered() {
        let store = temp_store("shape");
        std::fs::create_dir_all(store.path().parent().unwrap()).unwrap();
        std::fs::write(store.path(), r#"{"models":[]}"#).unwrap();
        let err = store.put("m.gguf", "v", &direction(1.0), true).unwrap_err();
        assert!(err.to_string().contains("JSON array"), "{err}");
        assert_eq!(
            std::fs::read_to_string(store.path()).unwrap(),
            r#"{"models":[]}"#
        );
    }
}
