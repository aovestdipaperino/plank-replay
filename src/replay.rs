//! Applies decoded tool calls to a fresh output directory.

use std::fmt;
use std::path::{Component, Path, PathBuf};

use crate::error::ReplayError;
use crate::parse::{Call, Event, UPTO};
use crate::seed::Seed;

/// What happened when one call was replayed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// A file was created or fully replaced.
    Wrote {
        /// Path relative to the output root.
        path: PathBuf,
        /// Byte length of the new contents.
        bytes: usize,
    },
    /// An existing file was patched in place.
    Edited {
        /// Path relative to the output root.
        path: PathBuf,
        /// Byte length after the patch.
        bytes: usize,
    },
    /// A shell command was run.
    Ran {
        /// The command line, as recorded in the transcript.
        command: String,
        /// Process exit status, or `None` if it was killed by a signal.
        status: Option<i32>,
    },
    /// A file was restored from a `read` result before an edit needed it.
    Seeded {
        /// Path relative to the output root.
        path: PathBuf,
        /// Byte length of the restored contents.
        bytes: usize,
    },
    /// The call was not replayable and was passed over.
    Skipped {
        /// Tool name that was skipped.
        tool: String,
        /// Why it was skipped.
        reason: String,
    },
    /// The call was replayable but failed, exactly as it may have in the session.
    Failed {
        /// Tool name that failed.
        tool: String,
        /// Rendered error.
        reason: String,
    },
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wrote { path, bytes } => write!(f, "write  {} ({bytes} B)", path.display()),
            Self::Edited { path, bytes } => write!(f, "edit   {} ({bytes} B)", path.display()),
            Self::Seeded { path, bytes } => write!(f, "seed   {} ({bytes} B)", path.display()),
            Self::Ran { command, status } => {
                let code = status.map_or_else(|| "signal".to_string(), |c| c.to_string());
                write!(f, "bash   [{code}] {}", first_line(command))
            }
            Self::Skipped { tool, reason } => write!(f, "skip   {tool}: {reason}"),
            Self::Failed { tool, reason } => write!(f, "FAIL   {tool}: {reason}"),
        }
    }
}

/// One replayed event paired with its position in the repro file.
#[derive(Debug, Clone)]
pub struct StepReport {
    /// 1-based index of the event within the transcript.
    pub index: usize,
    /// 1-based line in the repro file where the call opened.
    pub line: usize,
    /// Result of replaying it.
    pub outcome: Outcome,
}

/// Rebuilds a workspace by applying a repro's calls under one root directory.
#[derive(Debug, Clone)]
pub struct Replayer {
    root: PathBuf,
    run_bash: bool,
    stop_on_error: bool,
    seed: bool,
}

impl Replayer {
    /// Creates a replayer targeting `root`, which is created on first write.
    #[must_use]
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            run_bash: false,
            stop_on_error: false,
            seed: true,
        }
    }

    /// Enables execution of recorded `bash` commands inside the output root.
    #[must_use]
    pub fn run_bash(mut self, yes: bool) -> Self {
        self.run_bash = yes;
        self
    }

    /// Aborts the replay at the first failing call instead of continuing.
    #[must_use]
    pub fn stop_on_error(mut self, yes: bool) -> Self {
        self.stop_on_error = yes;
        self
    }

    /// Restores files from complete `read` results, so edits have a base. On by default.
    #[must_use]
    pub fn seed(mut self, yes: bool) -> Self {
        self.seed = yes;
        self
    }

    /// Replays every event in order, returning one report per event.
    ///
    /// Individual failures are recorded as [`Outcome::Failed`] and do not stop
    /// the run unless [`Replayer::stop_on_error`] was set.
    ///
    /// # Errors
    ///
    /// Returns an error only when `stop_on_error` is set and an event fails.
    pub fn replay(&self, events: &[Event]) -> Result<Vec<StepReport>, ReplayError> {
        let mut reports = Vec::with_capacity(events.len());
        for (i, event) in events.iter().enumerate() {
            let (line, result) = match event {
                Event::Call(call) => (call.line, self.apply(call)),
                Event::Seed(seed) => (seed.line, self.apply_seed(seed)),
            };
            let tool = match event {
                Event::Call(call) => call.name.clone(),
                Event::Seed(_) => "seed".to_string(),
            };
            let outcome = match result {
                Ok(outcome) => outcome,
                Err(e) => {
                    if self.stop_on_error {
                        return Err(e);
                    }
                    Outcome::Failed {
                        tool,
                        reason: e.to_string(),
                    }
                }
            };
            reports.push(StepReport {
                index: i + 1,
                line,
                outcome,
            });
        }
        Ok(reports)
    }

    /// Restores a file recovered from a `read` result, never overwriting one
    /// the transcript already produced.
    ///
    /// # Errors
    ///
    /// Returns an error when the path escapes the root or the write fails.
    pub fn apply_seed(&self, seed: &Seed) -> Result<Outcome, ReplayError> {
        if !self.seed {
            return Ok(Outcome::Skipped {
                tool: "seed".into(),
                reason: "seeding disabled (--no-seed)".into(),
            });
        }
        let rel = Self::sandbox(&seed.path)?;
        let target = self.root.join(&rel);
        if target.exists() {
            return Ok(Outcome::Skipped {
                tool: "seed".into(),
                reason: format!("{} already reconstructed", rel.display()),
            });
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ReplayError::io(parent, e))?;
        }
        std::fs::write(&target, &seed.content).map_err(|e| ReplayError::io(&target, e))?;
        Ok(Outcome::Seeded {
            path: rel,
            bytes: seed.content.len(),
        })
    }

    /// Applies a single call.
    ///
    /// # Errors
    ///
    /// Returns an error when the call is malformed, its path escapes the root,
    /// or the underlying file operation fails.
    pub fn apply(&self, call: &Call) -> Result<Outcome, ReplayError> {
        match call.name.as_str() {
            "write" => self.write(call),
            "edit" => self.edit(call),
            "bash" if self.run_bash => self.bash(call),
            "bash" => Ok(Outcome::Skipped {
                tool: "bash".into(),
                reason: "not executed (pass --run-bash)".into(),
            }),
            other => Ok(Outcome::Skipped {
                tool: other.into(),
                reason: "read-only tool, nothing to replay".into(),
            }),
        }
    }

    /// Creates or replaces a file.
    fn write(&self, call: &Call) -> Result<Outcome, ReplayError> {
        let rel = Self::sandbox(call.require("path")?)?;
        let content = call.param("content").unwrap_or_default();
        let target = self.root.join(&rel);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ReplayError::io(parent, e))?;
        }
        std::fs::write(&target, content).map_err(|e| ReplayError::io(&target, e))?;
        Ok(Outcome::Wrote {
            path: rel,
            bytes: content.len(),
        })
    }

    /// Patches a file, honouring the `[upto]` anchored form.
    fn edit(&self, call: &Call) -> Result<Outcome, ReplayError> {
        let rel = Self::sandbox(call.require("path")?)?;
        let old = call.require("old")?;
        let new = call.require("new")?;
        let target = self.root.join(&rel);
        let current = match std::fs::read_to_string(&target) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ReplayError::no_base_contents(&rel));
            }
            Err(e) => return Err(ReplayError::io(&target, e)),
        };
        let patched = apply_edit(&current, old, new, &rel)?;
        std::fs::write(&target, &patched).map_err(|e| ReplayError::io(&target, e))?;
        Ok(Outcome::Edited {
            path: rel,
            bytes: patched.len(),
        })
    }

    /// Runs a recorded shell command with the output root as the working directory.
    fn bash(&self, call: &Call) -> Result<Outcome, ReplayError> {
        let command = call.require("command")?.to_string();
        std::fs::create_dir_all(&self.root).map_err(|e| ReplayError::io(&self.root, e))?;
        let status = std::process::Command::new("sh")
            .arg("-c")
            .arg(&command)
            .current_dir(&self.root)
            .status()
            .map_err(|e| ReplayError::io(&self.root, e))?;
        Ok(Outcome::Ran {
            command,
            status: status.code(),
        })
    }

    /// Maps a transcript path onto a path inside the output root.
    ///
    /// Absolute paths are re-rooted under `_abs/`; `..` components are rejected.
    fn sandbox(raw: &str) -> Result<PathBuf, ReplayError> {
        let raw = raw.trim();
        let path = Path::new(raw);
        let mut out = PathBuf::new();
        if path.is_absolute() {
            out.push("_abs");
        }
        for component in path.components() {
            match component {
                Component::Normal(part) => out.push(part),
                Component::CurDir | Component::RootDir => {}
                Component::ParentDir | Component::Prefix(_) => {
                    return Err(ReplayError::escaping_path(raw));
                }
            }
        }
        if out.as_os_str().is_empty() {
            return Err(ReplayError::escaping_path(raw));
        }
        Ok(out)
    }
}

/// Returns `text` with `old` replaced by `new`, matching plank's `edit` rules.
///
/// When `old` contains `[upto]` the head and tail halves each have to match
/// exactly once, and everything between them is replaced.
///
/// # Panics
///
/// Never panics: each `expect` is guarded by the match count checked just above it.
///
/// # Errors
///
/// Returns an error when the anchor is absent or matches more than once.
pub fn apply_edit(text: &str, old: &str, new: &str, path: &Path) -> Result<String, ReplayError> {
    let Some((head, tail)) = old.split_once(UPTO) else {
        let count = text.matches(old).count();
        if count == 0 {
            return Err(ReplayError::edit_not_found(path));
        }
        if count > 1 {
            return Err(ReplayError::edit_ambiguous(path, count));
        }
        return Ok(text.replacen(old, new, 1));
    };

    let head = head.trim_end_matches('\n');
    let tail = tail.trim_start_matches('\n');
    if tail.is_empty() {
        return Err(ReplayError::edit_not_found(path));
    }

    let head_hits = text.match_indices(head).count();
    if head_hits == 0 {
        return Err(ReplayError::edit_not_found(path));
    }
    if head_hits > 1 {
        return Err(ReplayError::edit_ambiguous(path, head_hits));
    }
    let start = text.find(head).expect("head matched once");
    let after = start + head.len();
    let region = &text[after..];
    let tail_hits = region.matches(tail).count();
    if tail_hits == 0 {
        return Err(ReplayError::edit_not_found(path));
    }
    if tail_hits > 1 {
        return Err(ReplayError::edit_ambiguous(path, tail_hits));
    }
    let end = after + region.find(tail).expect("tail matched once") + tail.len();

    let mut out = String::with_capacity(text.len() + new.len());
    out.push_str(&text[..start]);
    out.push_str(new);
    out.push_str(&text[end..]);
    Ok(out)
}

/// Returns the first line of a command, for one-line reporting.
fn first_line(command: &str) -> &str {
    command.lines().next().unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{Call, Event};

    fn call(name: &str, params: &[(&str, &str)]) -> Call {
        Call {
            name: name.to_string(),
            params: params
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
            line: 1,
        }
    }

    fn tmp(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "plank-replay-test-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn write_creates_nested_files() {
        let root = tmp("write");
        let r = Replayer::new(&root);
        let outcome = r
            .apply(&call(
                "write",
                &[("path", "src/main.rs"), ("content", "fn main() {}")],
            ))
            .unwrap();
        assert!(matches!(outcome, Outcome::Wrote { .. }));
        assert_eq!(
            std::fs::read_to_string(root.join("src/main.rs")).unwrap(),
            "fn main() {}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_then_edit_round_trips() {
        let root = tmp("edit");
        let r = Replayer::new(&root);
        r.apply(&call(
            "write",
            &[("path", "a.txt"), ("content", "one\ntwo\nthree\n")],
        ))
        .unwrap();
        r.apply(&call(
            "edit",
            &[("path", "a.txt"), ("old", "two"), ("new", "TWO")],
        ))
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "one\nTWO\nthree\n"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn absolute_paths_are_rerooted() {
        let root = tmp("abs");
        let r = Replayer::new(&root);
        r.apply(&call(
            "write",
            &[("path", "/tmp/example.c"), ("content", "x")],
        ))
        .unwrap();
        assert!(root.join("_abs/tmp/example.c").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn parent_components_are_rejected() {
        let r = Replayer::new(tmp("escape"));
        assert!(
            r.apply(&call("write", &[("path", "../outside"), ("content", "x")]))
                .is_err()
        );
    }

    #[test]
    fn bash_is_skipped_unless_enabled() {
        let r = Replayer::new(tmp("nobash"));
        let outcome = r.apply(&call("bash", &[("command", "echo hi")])).unwrap();
        assert!(matches!(outcome, Outcome::Skipped { .. }));
    }

    #[test]
    fn read_only_tools_are_skipped() {
        let r = Replayer::new(tmp("ro"));
        let outcome = r.apply(&call("read", &[("path", "a.txt")])).unwrap();
        assert!(matches!(outcome, Outcome::Skipped { .. }));
    }

    #[test]
    fn missing_parameters_are_reported() {
        let r = Replayer::new(tmp("missing"));
        let err = r.apply(&call("write", &[("content", "x")])).unwrap_err();
        assert!(err.to_string().contains("path"));
    }

    #[test]
    fn replay_records_failures_and_continues() {
        let root = tmp("continue");
        let r = Replayer::new(&root);
        let events = vec![
            Event::Call(call(
                "edit",
                &[("path", "missing.txt"), ("old", "a"), ("new", "b")],
            )),
            Event::Call(call("write", &[("path", "ok.txt"), ("content", "hi")])),
        ];
        let reports = r.replay(&events).unwrap();
        assert!(matches!(reports[0].outcome, Outcome::Failed { .. }));
        assert!(matches!(reports[1].outcome, Outcome::Wrote { .. }));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn stop_on_error_aborts() {
        let r = Replayer::new(tmp("stop")).stop_on_error(true);
        let events = vec![Event::Call(call(
            "edit",
            &[("path", "missing.txt"), ("old", "a"), ("new", "b")],
        ))];
        assert!(r.replay(&events).is_err());
    }

    #[test]
    fn plain_edit_requires_a_unique_match() {
        let p = Path::new("a.txt");
        assert!(apply_edit("a a", "a", "b", p).is_err());
        assert!(apply_edit("zzz", "a", "b", p).is_err());
        assert_eq!(apply_edit("xax", "a", "b", p).unwrap(), "xbx");
    }

    #[test]
    fn anchored_edit_replaces_head_through_tail() {
        let text = "start\nstatic int parse(void) {\n    int ok = 0;\n    work();\n    return ok;\n}\nend\n";
        let old = "static int parse(void) {\n[upto]\n    return ok;\n}";
        let new = "static int parse(void) {\n    return parse_impl();\n}";
        let got = apply_edit(text, old, new, Path::new("x.c")).unwrap();
        assert_eq!(got, format!("start\n{new}\nend\n"));
    }

    #[test]
    fn anchored_edit_rejects_an_ambiguous_tail() {
        let text = "head\nA\nend\nfiller\nend\n";
        let err = apply_edit(text, "head\n[upto]\nend", "X", Path::new("x")).unwrap_err();
        assert!(err.to_string().contains("matched 2 times"));
    }

    #[test]
    fn anchored_edit_rejects_an_empty_tail() {
        assert!(apply_edit("abc", "a\n[upto]\n", "X", Path::new("x")).is_err());
    }

    #[test]
    fn bash_runs_in_the_output_root_when_enabled() {
        let root = tmp("bash");
        std::fs::create_dir_all(&root).unwrap();
        let r = Replayer::new(&root).run_bash(true);
        let outcome = r
            .apply(&call("bash", &[("command", "touch made-by-bash")]))
            .unwrap();
        assert!(matches!(
            outcome,
            Outcome::Ran {
                status: Some(0),
                ..
            }
        ));
        assert!(root.join("made-by-bash").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn seeding_lets_an_edit_apply_to_a_pre_existing_file() {
        let root = tmp("seed");
        let r = Replayer::new(&root);
        let seed = Seed {
            path: "old.txt".into(),
            content: "alpha\nbeta\n".into(),
            line: 1,
        };
        assert!(matches!(
            r.apply_seed(&seed).unwrap(),
            Outcome::Seeded { .. }
        ));
        r.apply(&call(
            "edit",
            &[("path", "old.txt"), ("old", "beta"), ("new", "BETA")],
        ))
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("old.txt")).unwrap(),
            "alpha\nBETA\n"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn seeding_never_overwrites_a_written_file() {
        let root = tmp("seed-noclobber");
        let r = Replayer::new(&root);
        r.apply(&call("write", &[("path", "a.txt"), ("content", "fresh")]))
            .unwrap();
        let seed = Seed {
            path: "a.txt".into(),
            content: "stale".into(),
            line: 1,
        };
        assert!(matches!(
            r.apply_seed(&seed).unwrap(),
            Outcome::Skipped { .. }
        ));
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "fresh"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn seeding_can_be_disabled() {
        let r = Replayer::new(tmp("noseed")).seed(false);
        let seed = Seed {
            path: "a.txt".into(),
            content: "x".into(),
            line: 1,
        };
        assert!(matches!(
            r.apply_seed(&seed).unwrap(),
            Outcome::Skipped { .. }
        ));
    }

    #[test]
    fn editing_an_unseen_file_explains_why() {
        let r = Replayer::new(tmp("nobase"));
        let err = r
            .apply(&call(
                "edit",
                &[("path", "never.rs"), ("old", "a"), ("new", "b")],
            ))
            .unwrap_err();
        assert!(err.to_string().contains("no base contents"), "{err}");
    }
}
