//! Resolves a plank model name to its GGUF file and steering shape.

use std::path::{Path, PathBuf};

use super::{VectorizeError, gguf};

/// The steering shape of a model family: layers by hidden width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Profile {
    /// Name as used by ds4's `build_direction.py --profile`.
    pub name: &'static str,
    /// Normal transformer layers; MTP predictor layers are excluded.
    pub layers: usize,
    /// Hidden width of one direction.
    pub width: usize,
}

/// Every model family ds4 can steer, with the GGUF architecture it reports.
pub const PROFILES: &[(Profile, &str)] = &[
    (
        Profile {
            name: "deepseek-v4-flash",
            layers: 43,
            width: 4096,
        },
        "deepseek4",
    ),
    (
        Profile {
            name: "glm-5.3-flash",
            layers: 45,
            width: 4096,
        },
        "glm5-next",
    ),
    (
        Profile {
            name: "qwen3.8-flash-next",
            layers: 48,
            width: 2560,
        },
        "qwen4exp",
    ),
];

impl Profile {
    /// Looks a profile up by its name.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        PROFILES
            .iter()
            .find(|(p, _)| p.name == name)
            .map(|(p, _)| *p)
    }

    /// Looks a profile up by the GGUF `general.architecture` value.
    #[must_use]
    pub fn for_architecture(arch: &str) -> Option<Self> {
        PROFILES.iter().find(|(_, a)| *a == arch).map(|(p, _)| *p)
    }
}

/// A model on disk, found from a plank engine name or a GGUF path.
#[derive(Debug, Clone)]
pub struct Model {
    name: String,
    path: PathBuf,
    architecture: Option<String>,
}

impl Model {
    /// Resolves `name` the way plank does, rooted at `~/.plank`.
    ///
    /// # Errors
    /// Fails when no GGUF can be found for `name`, or its header is unreadable.
    pub fn resolve(name: &str) -> Result<Self, VectorizeError> {
        Self::resolve_in(&plank_dir(), name)
    }

    /// Resolves `name` against the plank directory `root`.
    ///
    /// The lookup order is: an existing `.gguf` path, a `main.path` entry in
    /// `engines.local.json`, then the managed file `<root>/<name>.gguf`.
    ///
    /// # Errors
    /// Fails when no GGUF can be found for `name`, or its header is unreadable.
    pub fn resolve_in(root: &Path, name: &str) -> Result<Self, VectorizeError> {
        let direct = Path::new(name);
        let path = if direct.extension().is_some_and(|e| e == "gguf") && direct.is_file() {
            direct.to_path_buf()
        } else if let Some(path) = local_engine_path(root, name)? {
            path
        } else {
            let managed = root.join(format!("{name}.gguf"));
            if !managed.is_file() {
                return Err(VectorizeError::msg(format!(
                    "no model `{name}`: not a .gguf file, not an engine in {}, \
                     and {} does not exist",
                    root.join("engines.local.json").display(),
                    managed.display()
                )));
            }
            managed
        };
        let architecture = gguf::string_value(&path, "general.architecture")?;
        Ok(Self {
            name: name.to_string(),
            path,
            architecture,
        })
    }

    /// The name the model was asked for by.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The main GGUF file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The GGUF `general.architecture`, if the header carries one.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.architecture.as_deref()
    }

    /// The steering shape for this model's architecture.
    ///
    /// ds4 treats a GGUF without an architecture key as `DeepSeek` V4 Flash, so
    /// this does too.
    ///
    /// # Errors
    /// Fails for architectures ds4 has no steering support for.
    pub fn profile(&self) -> Result<Profile, VectorizeError> {
        let arch = self.architecture().unwrap_or("deepseek4");
        Profile::for_architecture(arch).ok_or_else(|| {
            VectorizeError::msg(format!(
                "{}: ds4 has no steering shape for architecture `{arch}` \
                 (steerable: deepseek4, glm5-next, qwen4exp); \
                 pass --profile only if your ds4 build supports it",
                self.path.display()
            ))
        })
    }
}

/// `~/.plank`, or `./.plank` when `HOME` is unset.
fn plank_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
        .join(".plank")
}

/// The `main.path` of engine `name` in `<root>/engines.local.json`, if any.
fn local_engine_path(root: &Path, name: &str) -> Result<Option<PathBuf>, VectorizeError> {
    let file = root.join("engines.local.json");
    let Ok(text) = std::fs::read_to_string(&file) else {
        return Ok(None);
    };
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| VectorizeError::msg(format!("{}: {e}", file.display())))?;
    let Some(path) = json
        .pointer(&format!(
            "/engines/{}/main/path",
            name.replace('~', "~0").replace('/', "~1")
        ))
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    Ok(Some(expand_tilde(path)))
}

/// Expands a leading `~/` to `HOME`, as plank does for local engine paths.
fn expand_tilde(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "plank-tools-test-model-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A minimal GGUF header carrying only `general.architecture`.
    fn gguf(arch: &str) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let key = "general.architecture";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&(arch.len() as u64).to_le_bytes());
        b.extend_from_slice(arch.as_bytes());
        b
    }

    #[test]
    fn a_managed_engine_resolves_to_its_gguf_in_the_plank_dir() {
        let root = temp_root("managed");
        std::fs::write(root.join("qwen.gguf"), gguf("qwen4exp")).unwrap();
        let model = Model::resolve_in(&root, "qwen").unwrap();
        assert_eq!(model.path(), root.join("qwen.gguf"));
        assert_eq!(model.profile().unwrap().name, "qwen3.8-flash-next");
    }

    #[test]
    fn a_local_engine_path_wins_over_the_managed_file() {
        let root = temp_root("local");
        let elsewhere = root.join("elsewhere.gguf");
        std::fs::write(&elsewhere, gguf("glm5-next")).unwrap();
        std::fs::write(root.join("mine.gguf"), gguf("deepseek4")).unwrap();
        std::fs::write(
            root.join("engines.local.json"),
            format!(
                r#"{{"engines":{{"mine":{{"main":{{"path":"{}"}}}}}}}}"#,
                elsewhere.display()
            ),
        )
        .unwrap();
        let model = Model::resolve_in(&root, "mine").unwrap();
        assert_eq!(model.path(), elsewhere);
        assert_eq!(model.profile().unwrap().layers, 45);
    }

    #[test]
    fn an_unknown_name_says_where_it_looked() {
        let root = temp_root("unknown");
        let err = Model::resolve_in(&root, "nope").unwrap_err().to_string();
        assert!(err.contains("nope.gguf"), "{err}");
    }

    #[test]
    fn an_unsteerable_architecture_asks_for_a_profile() {
        let root = temp_root("unsteerable");
        std::fs::write(root.join("v41.gguf"), gguf("deepseek41")).unwrap();
        let err = Model::resolve_in(&root, "v41")
            .unwrap()
            .profile()
            .unwrap_err();
        assert!(err.to_string().contains("--profile"));
    }
}
