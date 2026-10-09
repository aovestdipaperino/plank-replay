//! Hugging Face dataset access, with an on-disk cache and rate-limit backoff.
//!
//! Rows come from datasets-server, which serves at most 100 rows per request
//! and answers `429 Too Many Requests` when pushed. Every page is appended to
//! the cache as soon as it arrives, so an interrupted or rate-limited fetch
//! resumes where it stopped, and a dataset is downloaded once per machine.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::Value;

use super::VectorizeError;

/// Hugging Face's datasets-server, which serves any dataset's rows as JSON.
const ROWS_API: &str = "https://datasets-server.huggingface.co";

/// Rows per datasets-server request; the server's maximum.
const PAGE: usize = 100;

/// Waits between attempts when the Hub rate-limits or fails; about six
/// minutes in all before giving up.
const BACKOFF: &[Duration] = &[
    Duration::from_secs(10),
    Duration::from_secs(20),
    Duration::from_secs(40),
    Duration::from_secs(60),
    Duration::from_secs(90),
    Duration::from_secs(120),
];

/// The variables a Hugging Face key is read from, in order of preference.
/// `HF_TOKEN` is the name the Hugging Face tools themselves use.
pub const HF_KEY_VARS: &[&str] = &["HF_API_KEY", "HF_TOKEN"];

/// Where downloaded datasets are kept, and whether to trust what is there.
///
/// The default lives in `$XDG_CACHE_HOME/plank-tools/hf`, falling back to
/// `~/.cache/plank-tools/hf`. Writing to the cache is best effort: a cache
/// that cannot be written slows a fetch down but never fails it.
#[derive(Debug, Clone)]
pub struct HubCache {
    dir: Option<PathBuf>,
    refresh: bool,
}

impl HubCache {
    /// A cache rooted at `dir`.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: Some(dir.into()),
            refresh: false,
        }
    }

    /// No cache: every load goes to the network.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            dir: None,
            refresh: false,
        }
    }

    /// The default cache directory, if a home or cache directory is known.
    #[must_use]
    pub fn default_dir() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CACHE_HOME")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
        Some(base.join("plank-tools").join("hf"))
    }

    /// Ignores cached data and downloads it again, replacing the cache.
    #[must_use]
    pub fn refresh(mut self, refresh: bool) -> Self {
        self.refresh = refresh;
        self
    }

    /// The cache directory, or `None` when caching is off.
    #[must_use]
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// The cache directory for one dataset repository.
    fn repo_dir(&self, repo: &str) -> Option<PathBuf> {
        let dir = self.dir.as_ref()?.join("datasets");
        Some(repo.split('/').fold(dir, |d, part| d.join(safe(part))))
    }
}

impl Default for HubCache {
    fn default() -> Self {
        Self::default_dir().map_or_else(Self::disabled, Self::new)
    }
}

/// What a Hub load is doing, for progress display.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fetch {
    /// `rows` rows were found in the cache.
    Cached {
        /// Rows already on disk.
        rows: usize,
    },
    /// A page arrived; `have` rows are now on disk.
    Rows {
        /// Rows fetched so far, cached ones included.
        have: usize,
        /// Rows in the split, once the server has said.
        total: Option<usize>,
    },
    /// A request failed in a way worth retrying; waiting before the next try.
    Waiting {
        /// Pause before the next attempt.
        wait: Duration,
        /// The attempt that just failed, from 1.
        attempt: usize,
        /// Attempts allowed in all.
        attempts: usize,
        /// Why the request failed, e.g. `HTTP 429`.
        reason: String,
    },
}

/// A failed HTTP request, keeping the status for the retry decision.
#[derive(Debug)]
struct HttpError {
    status: Option<u16>,
    message: String,
}

impl HttpError {
    /// Rate limits, server errors, and failures with no status at all
    /// (dropped connections) are retried; client errors are final.
    fn retryable(&self) -> bool {
        self.status.is_none_or(|s| s == 429 || s >= 500)
    }

    fn reason(&self) -> String {
        self.status
            .map_or_else(|| self.message.clone(), |s| format!("HTTP {s}"))
    }
}

/// Fetches a URL, the seam that tests replace.
type Get<'a> = dyn FnMut(&str) -> Result<String, HttpError> + 'a;

/// Loads the rows of a split, from the cache where possible.
///
/// At most `limit` rows are needed; more may be returned if the cache already
/// holds them. Returned values are the row objects as datasets-server sends.
pub(super) fn load_split(
    repo: &str,
    path: Option<&str>,
    limit: Option<usize>,
    cache: &HubCache,
    report: &mut dyn FnMut(Fetch),
) -> Result<Vec<Value>, VectorizeError> {
    load_split_with(&mut curl_get, BACKOFF, repo, path, limit, cache, report)
}

/// Loads a file from the dataset repository, from the cache where possible.
pub(super) fn load_file(
    repo: &str,
    path: &str,
    cache: &HubCache,
    report: &mut dyn FnMut(Fetch),
) -> Result<String, VectorizeError> {
    load_file_with(&mut curl_get, BACKOFF, repo, path, cache, report)
}

/// The number of rows in a split, without fetching them.
///
/// The cache answers when it knows; otherwise one single-row page is asked
/// for, since datasets-server reports `num_rows_total` with every page, and
/// the answer is cached for next time.
pub(super) fn split_total(
    repo: &str,
    path: Option<&str>,
    cache: &HubCache,
    report: &mut dyn FnMut(Fetch),
) -> Result<Option<usize>, VectorizeError> {
    split_total_with(&mut curl_get, BACKOFF, repo, path, cache, report)
}

fn split_total_with(
    get: &mut Get<'_>,
    delays: &[Duration],
    repo: &str,
    path: Option<&str>,
    cache: &HubCache,
    report: &mut dyn FnMut(Fetch),
) -> Result<Option<usize>, VectorizeError> {
    let name = path.map_or_else(|| repo.to_string(), |p| format!("{repo}:{p}"));
    let base = cache.repo_dir(repo);
    let (config, split) = resolve_split(
        get,
        delays,
        repo,
        path,
        &name,
        base.as_deref(),
        cache,
        report,
    )?;
    let (_, meta_file) = split_files(base.as_deref(), &config, &split);
    if !cache.refresh
        && let Some(total) = meta_file.as_deref().and_then(read_total)
    {
        return Ok(Some(total));
    }
    let url = format!(
        "{ROWS_API}/rows?dataset={}&config={}&split={}&offset=0&length=1",
        encode(repo),
        encode(&config),
        encode(&split)
    );
    let text = with_backoff(get, delays, &url, report)?;
    let page: Value =
        serde_json::from_str(&text).map_err(|e| VectorizeError::msg(format!("{url}: {e}")))?;
    let total = page
        .get("num_rows_total")
        .and_then(Value::as_u64)
        .and_then(|t| usize::try_from(t).ok());
    if let (Some(file), Some(t)) = (&meta_file, total) {
        write_atomic(file, format!("{{\"total\":{t}}}\n").as_bytes());
    }
    Ok(total)
}

/// Where a split's rows and its `{"total": N}` note are cached.
fn split_files(
    base: Option<&Path>,
    config: &str,
    split: &str,
) -> (Option<PathBuf>, Option<PathBuf>) {
    let rows = base.map(|b| {
        b.join("rows")
            .join(safe(config))
            .join(format!("{}.jsonl", safe(split)))
    });
    let meta = rows.as_ref().map(|f| f.with_extension("meta.json"));
    (rows, meta)
}

fn load_split_with(
    get: &mut Get<'_>,
    delays: &[Duration],
    repo: &str,
    path: Option<&str>,
    limit: Option<usize>,
    cache: &HubCache,
    report: &mut dyn FnMut(Fetch),
) -> Result<Vec<Value>, VectorizeError> {
    let name = path.map_or_else(|| repo.to_string(), |p| format!("{repo}:{p}"));
    let base = cache.repo_dir(repo);

    let (config, split) = resolve_split(
        get,
        delays,
        repo,
        path,
        &name,
        base.as_deref(),
        cache,
        report,
    )?;
    let (config, split) = (config.as_str(), split.as_str());

    let (rows_file, meta_file) = split_files(base.as_deref(), config, split);
    if cache.refresh {
        for file in rows_file.iter().chain(&meta_file) {
            let _ = std::fs::remove_file(file);
        }
    }
    let mut rows = rows_file.as_deref().map(read_rows).unwrap_or_default();
    let mut total = meta_file.as_deref().and_then(read_total);
    let need = limit.unwrap_or(usize::MAX);
    let complete = |rows: &[Value], total: Option<usize>| total.is_some_and(|t| rows.len() >= t);
    if !rows.is_empty() {
        report(Fetch::Cached { rows: rows.len() });
    }

    while rows.len() < need && !complete(&rows, total) {
        let offset = rows.len();
        let want = PAGE.min(need - offset);
        let url = format!(
            "{ROWS_API}/rows?dataset={}&config={}&split={}&offset={offset}&length={want}",
            encode(repo),
            encode(config),
            encode(split)
        );
        let text = with_backoff(get, delays, &url, report)?;
        let page: Value =
            serde_json::from_str(&text).map_err(|e| VectorizeError::msg(format!("{url}: {e}")))?;
        let fresh: Vec<Value> = page
            .get("rows")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|r| r.get("row").cloned())
            .collect();
        let reported = page
            .get("num_rows_total")
            .and_then(Value::as_u64)
            .and_then(|t| usize::try_from(t).ok());
        total = reported.or(total);
        let short = fresh.len() < want;
        if let Some(file) = &rows_file {
            append_rows(file, &fresh);
        }
        rows.extend(fresh);
        if short {
            // The split ended early: whatever the header said, this is all.
            total = Some(rows.len());
        }
        if let (Some(file), Some(t)) = (&meta_file, total) {
            write_atomic(file, format!("{{\"total\":{t}}}\n").as_bytes());
        }
        report(Fetch::Rows {
            have: rows.len(),
            total,
        });
        if short {
            break;
        }
    }
    Ok(rows)
}

/// Picks the `(config, split)` that `path` names, listing the splits from
/// the cache or the server.
#[allow(clippy::too_many_arguments, reason = "internal plumbing of one load")]
fn resolve_split(
    get: &mut Get<'_>,
    delays: &[Duration],
    repo: &str,
    path: Option<&str>,
    name: &str,
    base: Option<&Path>,
    cache: &HubCache,
    report: &mut dyn FnMut(Fetch),
) -> Result<(String, String), VectorizeError> {
    let splits_url = format!("{ROWS_API}/splits?dataset={}", encode(repo));
    let splits_file = base.map(|b| b.join("splits.json"));
    let cached = splits_file
        .as_deref()
        .filter(|f| !cache.refresh && f.is_file())
        .and_then(|f| std::fs::read_to_string(f).ok());
    let splits_text = if let Some(text) = cached {
        text
    } else {
        let text = with_backoff(get, delays, &splits_url, report).map_err(|e| {
            VectorizeError::msg(format!(
                "{name}: cannot list the dataset's splits; check the name, \
                 or set HF_API_KEY if it is gated or private ({e})"
            ))
        })?;
        if let Some(file) = &splits_file {
            write_atomic(file, text.as_bytes());
        }
        text
    };
    let splits: Value = serde_json::from_str(&splits_text)
        .map_err(|e| VectorizeError::msg(format!("{splits_url}: {e}")))?;
    let listed: Vec<(&str, &str)> = splits
        .get("splits")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|s| Some((s.get("config")?.as_str()?, s.get("split")?.as_str()?)))
        .collect();
    let (config, split) = choose_split(&listed, path).ok_or_else(|| {
        let names: Vec<String> = listed.iter().map(|(c, s)| format!("{c}/{s}")).collect();
        VectorizeError::msg(format!(
            "{name}: no matching split; available: {}",
            names.join(", ")
        ))
    })?;
    Ok((config.to_string(), split.to_string()))
}

fn load_file_with(
    get: &mut Get<'_>,
    delays: &[Duration],
    repo: &str,
    path: &str,
    cache: &HubCache,
    report: &mut dyn FnMut(Fetch),
) -> Result<String, VectorizeError> {
    let relative = Path::new(path);
    if !relative
        .components()
        .all(|c| matches!(c, Component::Normal(_)))
    {
        return Err(VectorizeError::msg(format!(
            "{repo}:{path}: the path must stay inside the repository"
        )));
    }
    let file = cache.repo_dir(repo).map(|b| b.join("files").join(relative));
    if let Some(file) = file.as_deref().filter(|f| !cache.refresh && f.is_file())
        && let Ok(text) = std::fs::read_to_string(file)
    {
        report(Fetch::Cached {
            rows: text.lines().count(),
        });
        return Ok(text);
    }
    let url = format!("https://huggingface.co/datasets/{repo}/resolve/main/{path}");
    let text = with_backoff(get, delays, &url, report)?;
    if let Some(file) = &file {
        write_atomic(file, text.as_bytes());
    }
    Ok(text)
}

/// Fetches `url`, waiting and retrying while the failure is retryable.
fn with_backoff(
    get: &mut Get<'_>,
    delays: &[Duration],
    url: &str,
    report: &mut dyn FnMut(Fetch),
) -> Result<String, VectorizeError> {
    let attempts = delays.len() + 1;
    for attempt in 1..=attempts {
        match get(url) {
            Ok(body) => return Ok(body),
            Err(e) if e.retryable() && attempt < attempts => {
                let wait = delays[attempt - 1];
                report(Fetch::Waiting {
                    wait,
                    attempt,
                    attempts,
                    reason: e.reason(),
                });
                std::thread::sleep(wait);
            }
            Err(e) => {
                let hint = if e.status == Some(429) {
                    " (rate limited; rerun later to resume from the cache, \
                     or set HF_API_KEY for a higher limit)"
                } else {
                    ""
                };
                return Err(VectorizeError::msg(format!("{url}: {}{hint}", e.message)));
            }
        }
    }
    unreachable!("the last attempt always returns")
}

/// Downloads `url` with `curl`, sending the Hugging Face key when one is set.
///
/// The key is passed on stdin rather than argv so it never shows up in the
/// process list.
fn curl_get(url: &str) -> Result<String, HttpError> {
    let fail = |message: String| HttpError {
        status: None,
        message,
    };
    let token = hf_api_key();
    let mut cmd = Command::new("curl");
    cmd.args(["-fsSL", "--connect-timeout", "20"]);
    if token.is_some() {
        cmd.args(["-H", "@-"]);
    }
    cmd.arg(url)
        .stdin(if token.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| fail(format!("cannot run curl: {e}")))?;
    if let (Some(token), Some(mut stdin)) = (token, child.stdin.take()) {
        let _ = writeln!(stdin, "Authorization: Bearer {token}");
    }
    let out = child
        .wait_with_output()
        .map_err(|e| fail(format!("curl: {e}")))?;
    if !out.status.success() {
        let message = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(HttpError {
            status: http_status(&message),
            message,
        });
    }
    String::from_utf8(out.stdout).map_err(|_| fail("response is not UTF-8".into()))
}

/// The status in curl's `The requested URL returned error: 429`.
fn http_status(message: &str) -> Option<u16> {
    let (_, rest) = message.rsplit_once("error: ")?;
    rest.get(..3)?.parse().ok()
}

/// The first non-empty key among [`HF_KEY_VARS`].
fn hf_api_key() -> Option<String> {
    first_key(|name| std::env::var(name).ok())
}

fn first_key(lookup: impl Fn(&str) -> Option<String>) -> Option<String> {
    HF_KEY_VARS
        .iter()
        .filter_map(|name| lookup(name))
        .map(|key| key.trim().to_string())
        .find(|key| !key.is_empty())
}

/// Matches `path` to a listed `(config, split)`: by split name, by
/// `config/split`, or by a data file named `<split>-…` such as
/// `data/train-00000-of-00001.parquet`. With no path, `train` is taken, else
/// the first split. The `default` config is preferred.
fn choose_split<'a>(
    listed: &[(&'a str, &'a str)],
    path: Option<&str>,
) -> Option<(&'a str, &'a str)> {
    let Some(path) = path else {
        return choose_split(listed, Some("train")).or_else(|| {
            listed
                .iter()
                .find(|(c, _)| *c == "default")
                .or_else(|| listed.first())
                .copied()
        });
    };
    if let Some(exact) = listed.iter().find(|(c, s)| path == format!("{c}/{s}")) {
        return Some(*exact);
    }
    let stem = Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.split(['-', '.']).next().unwrap_or(n));
    let by = |want: Option<&str>| {
        let hits: Vec<_> = listed.iter().filter(|(_, s)| Some(*s) == want).collect();
        hits.iter()
            .find(|(c, _)| *c == "default")
            .or_else(|| hits.first())
            .map(|cs| **cs)
    };
    by(Some(path)).or_else(|| by(stem))
}

/// Reads cached rows, keeping those before the first damaged line. A damaged
/// tail (an interrupted append) is cut off so later appends stay valid.
fn read_rows(file: &Path) -> Vec<Value> {
    let Ok(text) = std::fs::read_to_string(file) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    let mut good_bytes = 0;
    for line in text.split_inclusive('\n') {
        match serde_json::from_str::<Value>(line) {
            Ok(row) if line.ends_with('\n') => {
                rows.push(row);
                good_bytes += line.len();
            }
            _ => break,
        }
    }
    if good_bytes < text.len() {
        write_atomic(file, &text.as_bytes()[..good_bytes]);
    }
    rows
}

fn read_total(file: &Path) -> Option<usize> {
    let text = std::fs::read_to_string(file).ok()?;
    let meta: Value = serde_json::from_str(&text).ok()?;
    usize::try_from(meta.get("total")?.as_u64()?).ok()
}

/// Appends rows as JSON lines; best effort.
fn append_rows(file: &Path, rows: &[Value]) {
    let mut text = String::new();
    for row in rows {
        text.push_str(&row.to_string());
        text.push('\n');
    }
    let _ = file
        .parent()
        .map(std::fs::create_dir_all)
        .transpose()
        .and_then(|_| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file)
        })
        .and_then(|mut f| f.write_all(text.as_bytes()));
}

/// Replaces `file` with `bytes` through a temporary sibling; best effort.
fn write_atomic(file: &Path, bytes: &[u8]) {
    let Some(parent) = file.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let tmp = file.with_extension(format!("tmp-{}", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, file).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// A name made safe to use as one path component.
fn safe(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() || out.starts_with('.') {
        out.insert(0, '_');
    }
    out
}

/// Percent-encodes a query value.
fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b'/') {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn temp_cache(label: &str) -> HubCache {
        let dir = std::env::temp_dir().join(format!(
            "plank-tools-test-hub-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        HubCache::new(dir)
    }

    const SPLITS: &str = r#"{"splits":[{"config":"default","split":"train"}]}"#;

    /// A fake datasets-server holding `total` rows `{"text":"p<i>"}`, which
    /// answers 429 to the first `limited` row requests.
    struct FakeHub {
        total: usize,
        limited: usize,
        requests: Vec<String>,
    }

    impl FakeHub {
        fn get(&mut self, url: &str) -> Result<String, HttpError> {
            self.requests.push(url.to_string());
            if url.contains("/splits") {
                return Ok(SPLITS.to_string());
            }
            if self.limited > 0 {
                self.limited -= 1;
                return Err(HttpError {
                    status: Some(429),
                    message: "The requested URL returned error: 429".into(),
                });
            }
            let param = |key: &str| -> usize {
                url.split(['?', '&'])
                    .find_map(|kv| kv.strip_prefix(key))
                    .and_then(|v| v.parse().ok())
                    .unwrap()
            };
            let (offset, length) = (param("offset="), param("length="));
            let rows: Vec<String> = (offset..(offset + length).min(self.total))
                .map(|i| format!(r#"{{"row_idx":{i},"row":{{"text":"p{i}"}}}}"#))
                .collect();
            Ok(format!(
                r#"{{"rows":[{}],"num_rows_total":{}}}"#,
                rows.join(","),
                self.total
            ))
        }

        fn row_requests(&self) -> usize {
            self.requests.iter().filter(|u| u.contains("/rows")).count()
        }
    }

    fn load(
        hub: &mut FakeHub,
        cache: &HubCache,
        limit: Option<usize>,
        events: &mut Vec<Fetch>,
    ) -> Result<Vec<Value>, VectorizeError> {
        let mut get = |url: &str| hub.get(url);
        load_split_with(
            &mut get,
            &[Duration::ZERO; 3],
            "o/n",
            None,
            limit,
            cache,
            &mut |f| events.push(f),
        )
    }

    #[test]
    fn a_second_load_is_served_from_the_cache() {
        let cache = temp_cache("second");
        let mut hub = FakeHub {
            total: 250,
            limited: 0,
            requests: Vec::new(),
        };
        let rows = load(&mut hub, &cache, None, &mut Vec::new()).unwrap();
        assert_eq!(rows.len(), 250);
        assert_eq!(hub.row_requests(), 3);

        let mut events = Vec::new();
        let again = load(&mut hub, &cache, None, &mut events).unwrap();
        assert_eq!(again, rows);
        assert_eq!(hub.row_requests(), 3, "no new requests");
        assert_eq!(events, [Fetch::Cached { rows: 250 }]);
    }

    #[test]
    fn the_size_costs_one_row_and_is_then_cached() {
        let cache = temp_cache("size");
        let mut hub = FakeHub {
            total: 25_058,
            limited: 0,
            requests: Vec::new(),
        };
        let size = |hub: &mut FakeHub| {
            let mut get = |url: &str| hub.get(url);
            split_total_with(&mut get, &[], "o/n", None, &cache, &mut |_| {}).unwrap()
        };
        assert_eq!(size(&mut hub), Some(25_058));
        assert!(hub.requests.last().unwrap().ends_with("offset=0&length=1"));
        let before = hub.requests.len();
        assert_eq!(size(&mut hub), Some(25_058));
        assert_eq!(hub.requests.len(), before, "answered from the cache");
    }

    #[test]
    fn a_limited_load_resumes_where_the_cache_stops() {
        let cache = temp_cache("resume");
        let mut hub = FakeHub {
            total: 1000,
            limited: 0,
            requests: Vec::new(),
        };
        assert_eq!(
            load(&mut hub, &cache, Some(150), &mut Vec::new())
                .unwrap()
                .len(),
            150
        );
        assert_eq!(hub.row_requests(), 2);
        let rows = load(&mut hub, &cache, Some(320), &mut Vec::new()).unwrap();
        assert_eq!(rows.len(), 320);
        assert_eq!(rows[150]["text"], "p150");
        assert!(
            hub.requests
                .last()
                .unwrap()
                .contains("offset=250&length=70"),
            "{:?}",
            hub.requests
        );
        assert_eq!(hub.row_requests(), 4);
    }

    #[test]
    fn rate_limits_are_waited_out() {
        let cache = temp_cache("ratelimit");
        let mut hub = FakeHub {
            total: 5,
            limited: 2,
            requests: Vec::new(),
        };
        let mut events = Vec::new();
        let rows = load(&mut hub, &cache, None, &mut events).unwrap();
        assert_eq!(rows.len(), 5);
        let waits = events
            .iter()
            .filter(|e| matches!(e, Fetch::Waiting { reason, .. } if reason == "HTTP 429"))
            .count();
        assert_eq!(waits, 2);
    }

    #[test]
    fn a_persistent_rate_limit_fails_with_a_hint_and_keeps_progress() {
        let cache = temp_cache("giveup");
        let mut hub = FakeHub {
            total: 300,
            limited: 0,
            requests: Vec::new(),
        };
        load(&mut hub, &cache, Some(200), &mut Vec::new()).unwrap();
        hub.limited = 99;
        let err = load(&mut hub, &cache, None, &mut Vec::new()).unwrap_err();
        assert!(err.to_string().contains("rerun later"), "{err}");
        hub.limited = 0;
        let mut events = Vec::new();
        let rows = load(&mut hub, &cache, None, &mut events).unwrap();
        assert_eq!(rows.len(), 300);
        assert_eq!(events[0], Fetch::Cached { rows: 200 });
    }

    #[test]
    fn client_errors_are_not_retried() {
        let mut calls = 0;
        let mut get = |_: &str| {
            calls += 1;
            Err(HttpError {
                status: Some(404),
                message: "The requested URL returned error: 404".into(),
            })
        };
        let err = with_backoff(&mut get, &[Duration::ZERO; 3], "u", &mut |_| {}).unwrap_err();
        assert_eq!(calls, 1);
        assert!(err.to_string().contains("404"));
    }

    #[test]
    fn refresh_discards_the_cache() {
        let cache = temp_cache("refresh");
        let mut hub = FakeHub {
            total: 10,
            limited: 0,
            requests: Vec::new(),
        };
        load(&mut hub, &cache, None, &mut Vec::new()).unwrap();
        hub.total = 12;
        let rows = load(
            &mut hub,
            &cache.clone().refresh(true),
            None,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(rows.len(), 12);
    }

    #[test]
    fn a_damaged_cache_tail_is_dropped() {
        let cache = temp_cache("damaged");
        let file = cache.dir().unwrap().join("rows.jsonl");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, "{\"a\":1}\n{\"a\":2}\n{\"a\":").unwrap();
        assert_eq!(read_rows(&file).len(), 2);
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "{\"a\":1}\n{\"a\":2}\n"
        );
    }

    #[test]
    fn repository_files_are_cached_and_paths_contained() {
        let cache = temp_cache("files");
        let mut bodies = VecDeque::from(["one\ntwo\n".to_string()]);
        let mut get = |_: &str| Ok(bodies.pop_front().expect("fetched twice"));
        let first = load_file_with(&mut get, &[], "o/n", "p/a.txt", &cache, &mut |_| {}).unwrap();
        let second = load_file_with(&mut get, &[], "o/n", "p/a.txt", &cache, &mut |_| {}).unwrap();
        assert_eq!(first, second);
        let err = load_file_with(&mut get, &[], "o/n", "../x.txt", &cache, &mut |_| {});
        assert!(err.is_err());
    }

    #[test]
    fn statuses_are_read_from_curl_messages() {
        assert_eq!(
            http_status("curl: (56) The requested URL returned error: 429"),
            Some(429)
        );
        assert_eq!(http_status("curl: (6) Could not resolve host"), None);
    }

    #[test]
    fn splits_match_by_name_config_or_data_file() {
        let listed = [
            ("other", "train"),
            ("default", "train"),
            ("default", "test"),
        ];
        assert_eq!(
            choose_split(&listed, Some("train")),
            Some(("default", "train"))
        );
        assert_eq!(
            choose_split(&listed, Some("other/train")),
            Some(("other", "train"))
        );
        assert_eq!(
            choose_split(&listed, Some("data/test-00000-of-00001.parquet")),
            Some(("default", "test"))
        );
        assert_eq!(choose_split(&listed, Some("validation")), None);
        assert_eq!(choose_split(&listed, None), Some(("default", "train")));
        assert_eq!(
            choose_split(&[("default", "test"), ("x", "eval")], None),
            Some(("default", "test"))
        );
    }

    #[test]
    fn hf_api_key_wins_and_blank_keys_are_skipped() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| (*v).to_string())
            }
        };
        assert_eq!(
            first_key(env(&[("HF_API_KEY", "a"), ("HF_TOKEN", "b")])).as_deref(),
            Some("a")
        );
        assert_eq!(
            first_key(env(&[("HF_API_KEY", "  "), ("HF_TOKEN", "b")])).as_deref(),
            Some("b")
        );
        assert_eq!(first_key(env(&[])), None);
    }

    #[test]
    fn names_are_made_safe_for_the_cache() {
        assert_eq!(safe("default"), "default");
        assert_eq!(safe("../x y"), "_.._x_y");
        assert_eq!(safe(""), "_");
    }

    #[test]
    fn query_values_are_percent_encoded() {
        assert_eq!(encode("o/n a&b"), "o/n%20a%26b");
    }
}
