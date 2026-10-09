//! Resolves a plank engine (or a GGUF path) to its file, key and steering shape.

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
    /// The ds4 dump holding the residual stream after each layer, the
    /// activation heretic reads (`hidden_states` of the first generated
    /// token, at the last prompt position).
    pub residual_dump: &'static str,
    /// Hyper-connection branches that dump holds per token, `width` floats
    /// each; they are averaged into one row, as ds4 does when it hands the
    /// residual to its drafter. 1 when the dump is already a single row.
    pub residual_branches: usize,
}

/// Every model family ds4 can steer, with the GGUF architecture it reports.
pub const PROFILES: &[(Profile, &str)] = &[
    (
        Profile {
            name: "deepseek-v4-flash",
            layers: 43,
            width: 4096,
            // `ffn_out` here is the FFN block's own output; the residual is
            // the four-branch hyper-connection state after it.
            residual_dump: "hc_ffn_post",
            residual_branches: 4,
        },
        "deepseek4",
    ),
    (
        Profile {
            name: "glm-5.3-flash",
            layers: 45,
            width: 4096,
            // GLM's `ffn_out` dump is `next`, the residual after the layer.
            residual_dump: "ffn_out",
            residual_branches: 1,
        },
        "glm5-next",
    ),
    (
        Profile {
            name: "qwen3.8-flash-next",
            layers: 48,
            width: 2560,
            // Qwen's `ffn_out` dump is already the branch mean of the
            // residual `R` at the last prompt token.
            residual_dump: "ffn_out",
            residual_branches: 1,
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

/// A model on disk, found through plank's engine catalog.
#[derive(Debug, Clone)]
pub struct Model {
    name: String,
    engine: Option<String>,
    path: PathBuf,
    architecture: Option<String>,
}

/// plank's engine catalog as shipped, the layer every other one sits on. It
/// comes from the plank checkout `pt` builds against (`../plank`, the same
/// one `local-inference-engine` links).
const COMPILED_IN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../plank/engines.json"
));

/// One catalog engine, reduced to what resolving a model needs.
#[derive(Debug, Clone, PartialEq)]
struct Engine {
    name: String,
    /// Where plank loads the main file from: a local `path`, else the
    /// managed install `<root>/<name>.gguf`.
    main: PathBuf,
    /// The published file name (`main.name`), which a downloaded copy kept
    /// elsewhere still carries.
    published: Option<String>,
}

impl Model {
    /// Resolves `spec` the way plank does, against its catalog under the
    /// plank home (`~/.plank`, or the shared `/Users/.plank`).
    ///
    /// # Errors
    /// As [`Model::resolve_in`].
    pub fn resolve(spec: &str) -> Result<Self, VectorizeError> {
        Self::resolve_in(&super::plank_dir(), spec)
    }

    /// Resolves `spec` against the plank directory `root`.
    ///
    /// `spec` is an engine name from plank's catalog: the shipped
    /// `engines.json`, the fetched `engines.remote.json` when it is newer, and
    /// `engines.local.json`, each later layer replacing engines by name, as
    /// in plank. The engine's main file is its local `path`, else the managed
    /// `<root>/<name>.gguf`.
    ///
    /// A path to a `.gguf` is accepted too, and mapped back to the engine it
    /// belongs to when it is that engine's main file or carries its published
    /// file name, so vectors built from a downloaded copy are still stored
    /// under the engine plank will run.
    ///
    /// # Errors
    /// Fails when `spec` is neither a catalog engine nor an existing `.gguf`,
    /// when the engine is not installed, when a catalog layer is malformed,
    /// or when the model's header is unreadable.
    pub fn resolve_in(root: &Path, spec: &str) -> Result<Self, VectorizeError> {
        let engines = catalog(root)?;
        let (engine, path) = if let Some(e) = engines.iter().find(|e| e.name == spec) {
            if !e.main.is_file() {
                return Err(VectorizeError::msg(format!(
                    "engine `{spec}` is not installed: {} does not exist \
                     (plank downloads it the first time it runs with --model {spec})",
                    e.main.display()
                )));
            }
            (Some(e.name.clone()), e.main.clone())
        } else {
            let direct = Path::new(spec);
            if !(direct.extension().is_some_and(|x| x == "gguf") && direct.is_file()) {
                let names: Vec<&str> = engines.iter().map(|e| e.name.as_str()).collect();
                return Err(VectorizeError::msg(format!(
                    "no model `{spec}`: not an engine in plank's catalog ({}) \
                     and not a .gguf file",
                    names.join(", ")
                )));
            }
            (owner(&engines, direct), direct.to_path_buf())
        };
        let architecture = gguf::string_value(&path, "general.architecture")?;
        Ok(Self {
            name: spec.to_string(),
            engine,
            path,
            architecture,
        })
    }

    /// The name or path the model was asked for by.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The plank engine this model is, if it is one.
    #[must_use]
    pub fn engine(&self) -> Option<&str> {
        self.engine.as_deref()
    }

    /// The key plank looks this model's vectors up by: the engine name, or
    /// the file name for a GGUF that is no engine.
    #[must_use]
    pub fn key(&self) -> String {
        self.engine.clone().unwrap_or_else(|| {
            self.path.file_name().map_or_else(
                || self.path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            )
        })
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

/// The engine a GGUF path belongs to: one whose main file it is, else one
/// whose published file name it carries.
fn owner(engines: &[Engine], path: &Path) -> Option<String> {
    let real = |p: &Path| p.canonicalize().ok();
    let target = real(path);
    let by_file = engines
        .iter()
        .find(|e| target.is_some() && real(&e.main) == target);
    let file = path.file_name().and_then(|n| n.to_str());
    by_file
        .or_else(|| {
            engines
                .iter()
                .find(|e| file.is_some() && e.published.as_deref() == file)
        })
        .map(|e| e.name.clone())
}

/// plank's catalog under `root`: shipped, then the fetched cache when its
/// `version` is higher, then the local layer, later layers replacing engines
/// of the same name.
fn catalog(root: &Path) -> Result<Vec<Engine>, VectorizeError> {
    let parse = |text: &str, what: &Path| -> Result<serde_json::Value, VectorizeError> {
        serde_json::from_str(text)
            .map_err(|e| VectorizeError::msg(format!("{}: {e}", what.display())))
    };
    let shipped = parse(COMPILED_IN, Path::new("plank engines.json"))?;
    let mut layers = vec![shipped];
    let remote_path = root.join("engines.remote.json");
    if let Ok(text) = std::fs::read_to_string(&remote_path) {
        let remote = parse(&text, &remote_path)?;
        let version = |v: &serde_json::Value| v.get("version").and_then(serde_json::Value::as_u64);
        if version(&remote) > version(&layers[0]) {
            layers.push(remote);
        }
    }
    let local_path = root.join("engines.local.json");
    if let Ok(text) = std::fs::read_to_string(&local_path) {
        layers.push(parse(&text, &local_path)?);
    }

    let mut engines: Vec<Engine> = Vec::new();
    for layer in &layers {
        let Some(map) = layer.get("engines").and_then(serde_json::Value::as_object) else {
            continue;
        };
        for (name, entry) in map {
            let main = entry.get("main");
            let path = main
                .and_then(|m| m.get("path"))
                .and_then(serde_json::Value::as_str)
                .map_or_else(|| root.join(format!("{name}.gguf")), expand_tilde);
            let published = main
                .and_then(|m| m.get("name"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            let engine = Engine {
                name: name.clone(),
                main: path,
                published,
            };
            match engines.iter_mut().find(|e| e.name == *name) {
                Some(slot) => *slot = engine,
                None => engines.push(engine),
            }
        }
    }
    Ok(engines)
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

    /// The shipped catalog's file name for `ds4vision`'s main model.
    fn ds4vision_published() -> String {
        let shipped: serde_json::Value = serde_json::from_str(COMPILED_IN).unwrap();
        shipped["engines"]["ds4vision"]["main"]["name"]
            .as_str()
            .unwrap()
            .to_owned()
    }

    #[test]
    fn a_shipped_engine_resolves_to_its_managed_install() {
        let root = temp_root("managed");
        std::fs::write(root.join("qwen.gguf"), gguf("qwen4exp")).unwrap();
        let model = Model::resolve_in(&root, "qwen").unwrap();
        assert_eq!(model.path(), root.join("qwen.gguf"));
        assert_eq!(model.key(), "qwen");
        assert_eq!(model.profile().unwrap().name, "qwen3.8-flash-next");
    }

    #[test]
    fn an_engine_that_is_not_downloaded_says_so() {
        let root = temp_root("missing");
        let err = Model::resolve_in(&root, "ds4vision")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not installed"), "{err}");
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
        assert_eq!(model.key(), "mine");
        assert_eq!(model.profile().unwrap().layers, 45);
    }

    #[test]
    fn a_path_is_mapped_back_to_its_engine() {
        let root = temp_root("by-path");
        // The managed install itself.
        let installed = root.join("ds4vision.gguf");
        std::fs::write(&installed, gguf("deepseek4")).unwrap();
        let model = Model::resolve_in(&root, installed.to_str().unwrap()).unwrap();
        assert_eq!(model.engine(), Some("ds4vision"));
        // A downloaded copy elsewhere, under the published file name.
        let copy = root.join("models").join(ds4vision_published());
        std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
        std::fs::write(&copy, gguf("deepseek4")).unwrap();
        let model = Model::resolve_in(&root, copy.to_str().unwrap()).unwrap();
        assert_eq!(model.path(), copy);
        assert_eq!(model.key(), "ds4vision");
    }

    #[test]
    fn a_stray_gguf_is_keyed_by_its_file_name() {
        let root = temp_root("stray");
        let stray = root.join("my-finetune.gguf");
        std::fs::write(&stray, gguf("deepseek4")).unwrap();
        let model = Model::resolve_in(&root, stray.to_str().unwrap()).unwrap();
        assert_eq!(model.engine(), None);
        assert_eq!(model.key(), "my-finetune.gguf");
    }

    #[test]
    fn an_unknown_name_lists_the_catalog() {
        let root = temp_root("unknown");
        let err = Model::resolve_in(&root, "nope").unwrap_err().to_string();
        assert!(err.contains("ds4vision"), "{err}");
    }

    #[test]
    fn an_unsteerable_architecture_asks_for_a_profile() {
        let root = temp_root("unsteerable");
        let file = root.join("v41.gguf");
        std::fs::write(&file, gguf("deepseek41")).unwrap();
        let err = Model::resolve_in(&root, file.to_str().unwrap())
            .unwrap()
            .profile()
            .unwrap_err();
        assert!(err.to_string().contains("--profile"));
    }
}
