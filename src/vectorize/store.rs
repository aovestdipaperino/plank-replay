//! `pt`'s view of the `~/.plank/models/vectors.json` store.
//!
//! The store, its format and its merge live in `plank-lib`
//! (`../plank/crates/plank-lib`), which plank reads the same file with. This
//! wrapper stores a [`Direction`] and reports errors as [`VectorizeError`].

use std::path::{Path, PathBuf};

use super::{Direction, VectorizeError};

/// A `vectors.json` file of named steering vectors.
#[derive(Debug, Clone)]
pub struct VectorStore(plank_lib::vectors::VectorStore);

impl VectorStore {
    /// The store at `path`; the file need not exist yet.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self(plank_lib::vectors::VectorStore::new(path))
    }

    /// `models/vectors.json` under the plank home (`~/.plank`).
    #[must_use]
    pub fn default_path() -> PathBuf {
        plank_lib::vectors::store_path_in(&plank_lib::home::plank_dir())
    }

    /// The file this store reads and writes.
    #[must_use]
    pub fn path(&self) -> &Path {
        self.0.path()
    }

    /// The shared store underneath, for listing and merging.
    #[must_use]
    pub fn shared(&self) -> &plank_lib::vectors::VectorStore {
        &self.0
    }

    /// Whether `model` already has a vector called `name`.
    ///
    /// # Errors
    /// Fails when the file exists but is not a valid store.
    pub fn contains(&self, model: &str, name: &str) -> Result<bool, VectorizeError> {
        self.0.contains(model, name).map_err(VectorizeError::from)
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
        self.0
            .put(model, name, &direction.to_le_bytes(), force)
            .map_err(VectorizeError::from)
    }
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

    #[test]
    fn a_direction_is_stored_as_its_le_bytes_in_base64() {
        let dir = std::env::temp_dir().join(format!(
            "plank-tools-test-store-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let store = VectorStore::new(dir.join("models").join("vectors.json"));
        let mut acc = Accumulator::new(TINY);
        acc.add_pair(&[1.0, 0.0], &[0.0, 0.0]);
        assert!(
            !store
                .put("m", "v", &acc.finish(false, false), false)
                .unwrap()
        );
        assert!(store.contains("m", "v").unwrap());
        assert!(
            store
                .put("m", "v", &acc.finish(false, false), false)
                .is_err()
        );
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        // [1.0, 0.0] as little-endian f32.
        assert_eq!(json[0]["vectors"][0]["value"], "AACAPwAAAAA=");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
