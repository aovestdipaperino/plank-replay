//! Prompt sets from local files or Hugging Face datasets.

use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::VectorizeError;
use super::hub::{self, Fetch, HubCache};

/// Fields tried, in order, when a structured row has no `--column` given.
const PREFERRED_COLUMNS: &[&str] = &["text", "prompt", "instruction", "question", "input", "goal"];

/// Where a prompt set comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptSource {
    /// A file on this machine.
    Local(PathBuf),
    /// A Hugging Face dataset, written `owner/name`, `owner:name` or
    /// `owner/name:path`. The path is a file (`.txt`, `.jsonl`, `.json`)
    /// fetched as is, or a split such as `train` (or a data file named after
    /// one) read through datasets-server. Without a path, the `train` split
    /// is read, or the first split when there is none by that name.
    Hub {
        /// The dataset repository, `owner/name`.
        repo: String,
        /// File path or split inside the repository, if one was given.
        path: Option<String>,
    },
}

impl PromptSource {
    /// Interprets a command-line argument as a prompt source.
    ///
    /// An existing local path always wins, so a file whose name contains a
    /// colon is never mistaken for a dataset.
    #[must_use]
    pub fn parse(spec: &str) -> Self {
        if Path::new(spec).exists() {
            return Self::Local(PathBuf::from(spec));
        }
        let hub = |repo: String, path: Option<&str>| Self::Hub {
            repo,
            path: path
                .map(|p| p.trim_start_matches('/'))
                .filter(|p| !p.is_empty())
                .map(str::to_string),
        };
        match spec.split_once(':') {
            Some((repo, path)) if is_repo_id(repo) => hub(repo.to_string(), Some(path)),
            Some((owner, name)) if is_segment(owner) && is_segment(name) => {
                hub(format!("{owner}/{name}"), None)
            }
            // A missing `dir/file.txt` is a mistyped path, not a dataset.
            None if is_repo_id(spec) && !is_plain_file(spec) => hub(spec.to_string(), None),
            _ => Self::Local(PathBuf::from(spec)),
        }
    }

    /// Reads the prompts, keeping at most `limit` of them.
    ///
    /// Text files hold one prompt per line; blank lines and lines starting
    /// with `#` are skipped. JSON Lines and JSON arrays may hold strings or
    /// objects, from which `column` (or the first of `text`, `prompt`,
    /// `instruction`, `question`, `input`, `goal`) is taken.
    ///
    /// Hugging Face data goes through the default [`HubCache`].
    ///
    /// # Errors
    /// Fails when the source cannot be read or fetched, or holds no prompts.
    pub fn load(
        &self,
        column: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<String>, VectorizeError> {
        self.load_with(column, limit, &HubCache::default(), &mut |_| {})
    }

    /// How many prompts the source holds, as cheaply as it can be told.
    ///
    /// A Hugging Face split reports its row count from a single-row request
    /// (or the cache), so a large dataset is sized without being downloaded;
    /// rows with an empty prompt are counted, so the true number can be a
    /// little lower. Files are read in full and counted exactly.
    ///
    /// # Errors
    /// Fails when the source cannot be read or fetched.
    pub fn size(
        &self,
        column: Option<&str>,
        cache: &HubCache,
        report: &mut dyn FnMut(Fetch),
    ) -> Result<Option<usize>, VectorizeError> {
        match self {
            Self::Hub { repo, path } if !path.as_deref().is_some_and(is_plain_file) => {
                hub::split_total(repo, path.as_deref(), cache, report)
            }
            _ => Ok(Some(self.load_with(column, None, cache, report)?.len())),
        }
    }

    /// Like [`load`](Self::load), with an explicit cache and progress reports.
    ///
    /// # Errors
    /// Fails when the source cannot be read or fetched, or holds no prompts.
    pub fn load_with(
        &self,
        column: Option<&str>,
        limit: Option<usize>,
        cache: &HubCache,
        report: &mut dyn FnMut(Fetch),
    ) -> Result<Vec<String>, VectorizeError> {
        let mut prompts = match self {
            Self::Local(path) => {
                let text = std::fs::read_to_string(path).map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        VectorizeError::msg(format!(
                            "{}: no such file (a Hugging Face dataset is written \
                             owner/name, owner:name or owner/name:split)",
                            path.display()
                        ))
                    } else {
                        VectorizeError::io(path, e)
                    }
                })?;
                parse_by_extension(&text, &path.to_string_lossy(), column)?
            }
            Self::Hub {
                repo,
                path: Some(path),
            } if is_plain_file(path) => {
                let text = hub::load_file(repo, path, cache, report)?;
                parse_by_extension(&text, path, column)?
            }
            Self::Hub { repo, path } => {
                let rows = hub::load_split(repo, path.as_deref(), limit, cache, report)?;
                prompts_from_values(&rows, &self.to_string(), column)?
            }
        };
        if let Some(limit) = limit {
            prompts.truncate(limit);
        }
        if prompts.is_empty() {
            return Err(VectorizeError::msg(format!("{self}: no prompts found")));
        }
        Ok(prompts)
    }
}

impl fmt::Display for PromptSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local(path) => write!(f, "{}", path.display()),
            Self::Hub { repo, path: None } => f.write_str(repo),
            Self::Hub {
                repo,
                path: Some(path),
            } => write!(f, "{repo}:{path}"),
        }
    }
}

/// Whether `s` looks like a Hugging Face `owner/name` repository id.
fn is_repo_id(s: &str) -> bool {
    matches!(s.split_once('/'), Some((owner, name)) if is_segment(owner) && is_segment(name))
}

/// Whether `s` is a valid Hugging Face owner or repository name.
fn is_segment(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('.')
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Whether a repository path names a file read as is rather than a split.
fn is_plain_file(path: &str) -> bool {
    let ext = Path::new(path).extension().and_then(|e| e.to_str());
    matches!(ext, Some("txt" | "jsonl" | "json"))
}

fn parse_by_extension(
    text: &str,
    name: &str,
    column: Option<&str>,
) -> Result<Vec<String>, VectorizeError> {
    let ext = Path::new(name).extension().and_then(|e| e.to_str());
    match ext {
        Some("jsonl") => {
            let mut rows = Vec::new();
            for (i, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                rows.push(
                    serde_json::from_str(line)
                        .map_err(|e| VectorizeError::msg(format!("{name}: line {}: {e}", i + 1)))?,
                );
            }
            prompts_from_values(&rows, name, column)
        }
        Some("json") => {
            let value: Value = serde_json::from_str(text)
                .map_err(|e| VectorizeError::msg(format!("{name}: {e}")))?;
            let Value::Array(rows) = value else {
                return Err(VectorizeError::msg(format!(
                    "{name}: expected a JSON array"
                )));
            };
            prompts_from_values(&rows, name, column)
        }
        _ => Ok(text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect()),
    }
}

/// Takes the prompt out of each row: the row itself if it is a string, else
/// the chosen column of an object. The column is chosen from the first row.
fn prompts_from_values(
    rows: &[Value],
    name: &str,
    column: Option<&str>,
) -> Result<Vec<String>, VectorizeError> {
    let Some(first) = rows.first() else {
        return Ok(Vec::new());
    };
    if first.is_string() {
        return Ok(rows
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect());
    }
    let key = pick_column(first, name, column)?;
    Ok(rows
        .iter()
        .filter_map(|row| row.get(&key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect())
}

fn pick_column(row: &Value, name: &str, column: Option<&str>) -> Result<String, VectorizeError> {
    let Value::Object(fields) = row else {
        return Err(VectorizeError::msg(format!(
            "{name}: rows must be strings or objects"
        )));
    };
    let strings: Vec<&str> = fields
        .iter()
        .filter(|(_, v)| v.is_string())
        .map(|(k, _)| k.as_str())
        .collect();
    if let Some(column) = column {
        return if strings.contains(&column) {
            Ok(column.to_string())
        } else {
            Err(VectorizeError::msg(format!(
                "{name}: no text column `{column}`; text columns are: {}",
                strings.join(", ")
            )))
        };
    }
    if let Some(found) = PREFERRED_COLUMNS.iter().find(|c| strings.contains(c)) {
        return Ok((*found).to_string());
    }
    match strings.as_slice() {
        [only] => Ok((*only).to_string()),
        _ => Err(VectorizeError::msg(format!(
            "{name}: cannot tell which column holds the prompt; pass --column (text columns: {})",
            strings.join(", ")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repo_pointer_parses_as_a_hub_source() {
        assert_eq!(
            PromptSource::parse("mlabonne/harmful_behaviors:train"),
            PromptSource::Hub {
                repo: "mlabonne/harmful_behaviors".into(),
                path: Some("train".into())
            }
        );
    }

    #[test]
    fn a_bare_dataset_has_no_path_in_either_spelling() {
        let want = PromptSource::Hub {
            repo: "mlabonne/harmful_behaviors".into(),
            path: None,
        };
        assert_eq!(PromptSource::parse("mlabonne:harmful_behaviors"), want);
        assert_eq!(PromptSource::parse("mlabonne/harmful_behaviors"), want);
        assert_eq!(PromptSource::parse("mlabonne/harmful_behaviors:"), want);
        assert_eq!(want.to_string(), "mlabonne/harmful_behaviors");
    }

    #[test]
    fn plain_paths_and_odd_colons_stay_local() {
        for spec in [
            "prompts.txt",
            "/abs/p.txt",
            "C:/stuff",
            "a/b/c:train",
            "./x/y:z",
            "x/y/z",
            "examples/missing.txt",
        ] {
            assert!(
                matches!(PromptSource::parse(spec), PromptSource::Local(_)),
                "{spec}"
            );
        }
    }

    #[test]
    fn text_files_skip_blanks_and_comments() {
        let got = parse_by_extension("# header\n\n one \ntwo\n", "p.txt", None).unwrap();
        assert_eq!(got, ["one", "two"]);
    }

    #[test]
    fn jsonl_rows_use_the_preferred_column() {
        let text = "{\"id\":1,\"prompt\":\"a\",\"note\":\"x\"}\n{\"id\":2,\"prompt\":\"b\",\"note\":\"y\"}\n";
        assert_eq!(
            parse_by_extension(text, "p.jsonl", None).unwrap(),
            ["a", "b"]
        );
        assert_eq!(
            parse_by_extension(text, "p.jsonl", Some("note")).unwrap(),
            ["x", "y"]
        );
    }

    #[test]
    fn an_ambiguous_object_asks_for_a_column() {
        let err = parse_by_extension(r#"[{"a":"x","b":"y"}]"#, "p.json", None).unwrap_err();
        assert!(err.to_string().contains("--column"), "{err}");
    }

    #[test]
    fn a_json_array_of_strings_is_used_directly() {
        let got = parse_by_extension(r#"["x", " y ", ""]"#, "p.json", None).unwrap();
        assert_eq!(got, ["x", "y"]);
    }
}
