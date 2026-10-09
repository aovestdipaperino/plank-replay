//! `pt install`: install a vectors file or a profile from a repository.

use std::io::{BufRead as _, IsTerminal as _, Write as _};
use std::path::Path;
use std::process::ExitCode;

use plank_lib::profiles::{self, State};
use plank_lib::source::{Repo, RepoPath};
use plank_lib::vectors::{self, Planned, Status, VectorStore};

use super::CliError;

/// Usage text for `pt install --help`.
pub const USAGE: &str = "\
pt install - install steering vectors or a profile from a repository

USAGE:
    pt install <repo>:<path> [--skip | --overwrite]

<repo> is a local folder, a GitHub owner/repo, or a git URL (cloned shallow);
<path> is a file or folder inside it, e.g.

    pt install aovestdipaperino/plank-profiles:/profiles/3v1l
    pt install ~/Code/vectors:ds4vision/vectors.json

What the path holds decides what is installed:

    a vectors file      a JSON array of {model, vectors: [{name, value}]};
                        each vector is merged into ~/.plank/models/vectors.json
    a profile           a folder with .plank-plugin/plugin.json (or that file,
                        or its folder); copied to ~/.plank/profiles/<name> with
                        a .plank-source record, so `plank --profile <source>`
                        finds it. A vectors.json inside the profile folder is
                        merged too.

Nothing identical is touched. Anything that would replace something different
is listed and asked about (skip or overwrite, one by one or all at once)
before anything is written. Without a terminal, pass --skip or --overwrite.

OPTIONS:
        --skip          Keep everything already installed
        --overwrite     Replace everything that differs
    -h, --help          Show this help
";

/// How conflicts are settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    Ask,
    SkipAll,
    OverwriteAll,
}

/// Runs `pt install`.
pub fn run(args: &[String]) -> Result<ExitCode, CliError> {
    let mut spec = None;
    let mut policy = Policy::Ask;
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => return Err(CliError::Help),
            "--skip" if policy != Policy::OverwriteAll => policy = Policy::SkipAll,
            "--overwrite" if policy != Policy::SkipAll => policy = Policy::OverwriteAll,
            "--skip" | "--overwrite" => {
                return Err(CliError::Usage(
                    "`--skip` and `--overwrite` exclude each other".into(),
                ));
            }
            other if other.starts_with('-') => {
                return Err(CliError::Usage(format!("unknown option `{other}`")));
            }
            other if spec.is_none() => spec = Some(other.to_owned()),
            other => return Err(CliError::Usage(format!("unexpected argument `{other}`"))),
        }
    }
    let spec = spec.ok_or_else(|| CliError::Usage("a <repo>:<path> is required".into()))?;
    let source = RepoPath::parse(&spec).map_err(|e| CliError::Usage(e.to_string()))?;
    let interactive = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
    let mut decider = Decider {
        policy,
        interactive,
        input: Box::new(std::io::stdin().lock()),
    };
    install(
        &source,
        &VectorStore::at_home(),
        &profiles::root(),
        &mut decider,
    )?;
    Ok(ExitCode::SUCCESS)
}

/// Fetches `source` and installs what it points at into `store` and the
/// profiles root `root`, settling conflicts through `decider`.
fn install(
    source: &RepoPath,
    store: &VectorStore,
    root: &Path,
    decider: &mut Decider<'_>,
) -> Result<(), CliError> {
    let failed = |e: plank_lib::Error| CliError::Failed(e.to_string());
    if !matches!(source.repo, Repo::Local(_)) {
        eprintln!("fetching {} …", source.repo_label());
    }
    let checkout = source.fetch().map_err(failed)?;
    let target = checkout.resolve(&source.path).map_err(failed)?;
    let label = format!("{}:{}", source.repo_label(), source.path);

    if let Some(dir) = profiles::folder_of(&target) {
        let plan = profiles::plan(&dir, root).map_err(failed)?;
        let bundled = dir.join("vectors.json");
        let vector_plan = if bundled.is_file() {
            plan_vectors(store, &bundled, &format!("{label}/vectors.json"))?
        } else {
            Vec::new()
        };

        // Every question first, so a refusal writes nothing at all.
        let replace_profile = match &plan.state {
            State::Differs { installed_version } => decider.decide(&format!(
                "profile `{}` is installed ({}) and differs from {label} ({})",
                plan.manifest.name,
                version(installed_version.as_ref()),
                version(plan.manifest.version.as_ref()),
            ))?,
            State::New | State::Unchanged => true,
        };
        let overwrite = decide_vectors(&vector_plan, decider)?;

        let rel = dir.strip_prefix(checkout.root()).unwrap_or(&dir);
        let record = source.record(rel);
        match (&plan.state, replace_profile) {
            (State::Differs { .. }, false) => {
                println!("profile `{}`: kept the installed copy", plan.manifest.name);
            }
            (state, _) => {
                profiles::install(&plan, Some(&record)).map_err(failed)?;
                let what = match state {
                    State::New => "installed",
                    State::Unchanged => "already installed, unchanged",
                    State::Differs { .. } => "replaced",
                };
                println!(
                    "profile `{}` {} {what} in {}",
                    plan.manifest.name,
                    version(plan.manifest.version.as_ref()),
                    plan.dest.display()
                );
            }
        }
        if !vector_plan.is_empty() {
            merge_vectors(store, &vector_plan, &overwrite)?;
        }
        warn_missing_direction(store, &plan.manifest);
        println!("launch it with: plank --profile {}", plan.manifest.name);
        return Ok(());
    }

    if target.is_file() {
        let text = std::fs::read_to_string(&target)
            .map_err(|e| CliError::Failed(format!("{}: {e}", target.display())))?;
        if vectors::looks_like_file(&text) {
            let plan = plan_vectors(store, &target, &label)?;
            let overwrite = decide_vectors(&plan, decider)?;
            return merge_vectors(store, &plan, &overwrite);
        }
    }
    Err(CliError::Failed(format!(
        "{label}: neither a vectors file (a JSON array of {{model, vectors}}) \
         nor a profile (a folder with .plank-plugin/plugin.json)"
    )))
}

fn version(v: Option<&String>) -> String {
    v.map_or_else(|| "no version".to_owned(), |v| format!("v{v}"))
}

/// Reads and plans merging the vectors file at `path`.
fn plan_vectors(store: &VectorStore, path: &Path, label: &str) -> Result<Vec<Planned>, CliError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| CliError::Failed(format!("{}: {e}", path.display())))?;
    let incoming =
        vectors::parse_file(&text, label).map_err(|e| CliError::Failed(e.to_string()))?;
    store
        .plan(incoming)
        .map_err(|e| CliError::Failed(e.to_string()))
}

/// Asks about every conflicting vector, returning the `(model, name)` pairs
/// to overwrite.
fn decide_vectors(
    plan: &[Planned],
    decider: &mut Decider<'_>,
) -> Result<Vec<(String, String)>, CliError> {
    let mut overwrite = Vec::new();
    for p in plan.iter().filter(|p| p.status == Status::Conflict) {
        let i = &p.incoming;
        if decider.decide(&format!(
            "vector `{}`/`{}` is already stored with a different value \
             (incoming: {} floats)",
            i.model,
            i.name,
            i.byte_len() / 4
        ))? {
            overwrite.push((i.model.clone(), i.name.clone()));
        }
    }
    Ok(overwrite)
}

/// Applies a vectors plan and reports what it did.
fn merge_vectors(
    store: &VectorStore,
    plan: &[Planned],
    overwrite: &[(String, String)],
) -> Result<(), CliError> {
    let report = store
        .merge(plan, |p| {
            overwrite
                .iter()
                .any(|(m, n)| *m == p.incoming.model && *n == p.incoming.name)
        })
        .map_err(|e| CliError::Failed(e.to_string()))?;
    let list = |pairs: &[(String, String)]| {
        pairs
            .iter()
            .map(|(m, n)| format!("{m}/{n}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    for (what, pairs) in [
        ("added", &report.added),
        ("replaced", &report.replaced),
        ("kept (skipped)", &report.skipped),
        ("unchanged", &report.unchanged),
    ] {
        if !pairs.is_empty() {
            println!("vectors {what}: {}", list(pairs));
        }
    }
    println!("vectors file: {}", store.path().display());
    Ok(())
}

/// Warns when a profile steers along a direction the store will not hold
/// for its model, which would stop plank at launch.
fn warn_missing_direction(store: &VectorStore, manifest: &profiles::Manifest) {
    let (Some(steering), Some(model)) = (&manifest.steering, &manifest.recommended_model) else {
        return;
    };
    let present = store
        .directions(std::slice::from_ref(model))
        .is_ok_and(|names| names.contains(&steering.direction));
    if !present {
        eprintln!(
            "warning: profile `{}` steers `{model}` along `{}`, which {} does not hold; \
             install a vectors file with it, or build it with \
             `pt vectorize {model} --from … --to … -n {}`",
            manifest.name,
            steering.direction,
            store.path().display(),
            steering.direction
        );
    }
}

/// Settles conflicts: by policy, or by asking on the terminal.
struct Decider<'a> {
    policy: Policy,
    interactive: bool,
    input: Box<dyn std::io::BufRead + 'a>,
}

impl Decider<'_> {
    /// Whether to overwrite what `what` describes.
    fn decide(&mut self, what: &str) -> Result<bool, CliError> {
        match self.policy {
            Policy::SkipAll => {
                eprintln!("{what}: skipped");
                return Ok(false);
            }
            Policy::OverwriteAll => {
                eprintln!("{what}: overwriting");
                return Ok(true);
            }
            Policy::Ask => {}
        }
        if !self.interactive {
            return Err(CliError::Failed(format!(
                "{what}; nothing was installed. Rerun with --skip or --overwrite to decide \
                 without a terminal"
            )));
        }
        loop {
            eprint!("{what}.\n  [s]kip, [o]verwrite, [S]kip all, [O]verwrite all? ");
            let _ = std::io::stderr().flush();
            let mut line = String::new();
            if self
                .input
                .read_line(&mut line)
                .map_err(|e| CliError::Failed(e.to_string()))?
                == 0
            {
                return Err(CliError::Failed("no answer; nothing was installed".into()));
            }
            match line.trim() {
                "s" | "skip" | "" => return Ok(false),
                "o" | "overwrite" => return Ok(true),
                "S" => {
                    self.policy = Policy::SkipAll;
                    return Ok(false);
                }
                "O" => {
                    self.policy = Policy::OverwriteAll;
                    return Ok(true);
                }
                _ => eprintln!("  please answer s, o, S or O"),
            }
        }
    }
}

impl std::fmt::Debug for Decider<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decider")
            .field("policy", &self.policy)
            .field("interactive", &self.interactive)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "plank-tools-test-install-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    const ONE_ZERO: &str = "AACAPwAAAAA=";
    const ZERO_ONE: &str = "AAAAAAAAgD8=";

    fn vectors_file(path: &Path, heretic: &str) {
        std::fs::write(
            path,
            format!(
                r#"[{{"model":"ds4vision","vectors":[{{"name":"heretic","value":"{heretic}"}},{{"name":"terse","value":"{ONE_ZERO}"}}]}}]"#
            ),
        )
        .unwrap();
    }

    fn decider(policy: Policy, answers: &'static str) -> Decider<'static> {
        Decider {
            policy,
            interactive: true,
            input: Box::new(answers.as_bytes()),
        }
    }

    fn stored(store: &VectorStore, name: &str) -> String {
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        json[0]["vectors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["name"] == name)
            .unwrap()["value"]
            .as_str()
            .unwrap()
            .to_owned()
    }

    #[test]
    fn vectors_merge_and_conflicts_are_asked_about() {
        let dir = scratch("vectors");
        let repo = dir.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        vectors_file(&repo.join("v.json"), ONE_ZERO);
        let store = VectorStore::new(dir.join("home").join("vectors.json"));
        let src = RepoPath::parse(&format!("{}:v.json", repo.display())).unwrap();
        let root = dir.join("profiles");

        install(&src, &store, &root, &mut decider(Policy::Ask, "")).unwrap();
        assert_eq!(stored(&store, "heretic"), ONE_ZERO);

        // Now the repository's `heretic` differs: skip, then overwrite.
        vectors_file(&repo.join("v.json"), ZERO_ONE);
        install(&src, &store, &root, &mut decider(Policy::Ask, "s\n")).unwrap();
        assert_eq!(stored(&store, "heretic"), ONE_ZERO, "skipped");
        install(&src, &store, &root, &mut decider(Policy::Ask, "x\no\n")).unwrap();
        assert_eq!(
            stored(&store, "heretic"),
            ZERO_ONE,
            "overwritten after a bad answer"
        );
    }

    #[test]
    fn without_a_terminal_a_conflict_writes_nothing() {
        let dir = scratch("noninteractive");
        let repo = dir.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let store = VectorStore::new(dir.join("vectors.json"));
        vectors_file(&repo.join("v.json"), ONE_ZERO);
        let src = RepoPath::parse(&format!("{}:v.json", repo.display())).unwrap();
        install(&src, &store, &dir, &mut decider(Policy::Ask, "")).unwrap();
        let before = std::fs::read_to_string(store.path()).unwrap();

        vectors_file(&repo.join("v.json"), ZERO_ONE);
        let mut d = decider(Policy::Ask, "");
        d.interactive = false;
        let err = install(&src, &store, &dir, &mut d).unwrap_err();
        assert!(matches!(err, CliError::Failed(m) if m.contains("--overwrite")));
        assert_eq!(std::fs::read_to_string(store.path()).unwrap(), before);

        install(&src, &store, &dir, &mut decider(Policy::OverwriteAll, "")).unwrap();
        assert_eq!(stored(&store, "heretic"), ZERO_ONE);
    }

    #[test]
    fn a_profile_installs_with_its_bundled_vectors_and_asks_before_replacing() {
        let dir = scratch("profile");
        let profile = dir.join("repo").join("profiles").join("evil");
        std::fs::create_dir_all(profile.join(".plank-plugin")).unwrap();
        let manifest = |v: &str| {
            format!(
                r#"{{"name":"evil","version":"{v}","profile":{{"systemPrompt":"prompt.md",
                    "recommendedModel":"ds4vision","steering":{{"direction":"heretic"}}}}}}"#
            )
        };
        std::fs::write(profile.join(".plank-plugin/plugin.json"), manifest("1")).unwrap();
        std::fs::write(profile.join("prompt.md"), "be evil").unwrap();
        vectors_file(&profile.join("vectors.json"), ONE_ZERO);

        let store = VectorStore::new(dir.join("vectors.json"));
        let root = dir.join("profiles");
        let src = RepoPath::parse(&format!(
            "{}:profiles/evil/.plank-plugin/plugin.json",
            dir.join("repo").display()
        ))
        .unwrap();
        install(&src, &store, &root, &mut decider(Policy::Ask, "")).unwrap();
        let installed = root.join("evil");
        assert_eq!(
            std::fs::read_to_string(installed.join("prompt.md")).unwrap(),
            "be evil"
        );
        assert!(
            std::fs::read_to_string(installed.join(profiles::SOURCE_FILE))
                .unwrap()
                .trim()
                .ends_with("profiles/evil")
        );
        assert_eq!(stored(&store, "heretic"), ONE_ZERO);

        // A changed profile is kept on "skip" and replaced on "overwrite".
        std::fs::write(profile.join(".plank-plugin/plugin.json"), manifest("2")).unwrap();
        std::fs::write(profile.join("prompt.md"), "be worse").unwrap();
        install(&src, &store, &root, &mut decider(Policy::Ask, "s\n")).unwrap();
        assert_eq!(
            std::fs::read_to_string(installed.join("prompt.md")).unwrap(),
            "be evil"
        );
        install(&src, &store, &root, &mut decider(Policy::Ask, "o\n")).unwrap();
        assert_eq!(
            std::fs::read_to_string(installed.join("prompt.md")).unwrap(),
            "be worse"
        );
    }

    #[test]
    fn something_that_is_neither_is_refused() {
        let dir = scratch("neither");
        std::fs::write(dir.join("x.json"), r#"{"hello":1}"#).unwrap();
        let src = RepoPath::parse(&format!("{}:x.json", dir.display())).unwrap();
        let store = VectorStore::new(dir.join("vectors.json"));
        let err = install(&src, &store, &dir, &mut decider(Policy::Ask, "")).unwrap_err();
        assert!(matches!(err, CliError::Failed(m) if m.contains("neither")));
    }
}
