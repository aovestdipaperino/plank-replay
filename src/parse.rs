//! Extracts the embedded transcript from a repro file and decodes its tool calls.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::ReplayError;
use crate::seed::{Seed, scan};

/// Opening marker of the embedded transcript.
const BEGIN: &str = "----- BEGIN TRANSCRIPT -----";
/// Closing marker of the embedded transcript.
const END: &str = "----- END TRANSCRIPT -----";
/// Opening prefix every DSML tag shares, before the optional separating space.
const OPEN: &str = "<\u{ff5c}DSML\u{ff5c}";
/// Closing prefix every DSML end tag shares.
const CLOSE: &str = "</\u{ff5c}DSML\u{ff5c}";

/// Names a DSML tag may use to open or close a block of tool calls.
///
/// The `dsml` dialect writes `tool_calls`; `dsml41` writes `calls`.
const BLOCK_NAMES: [&str; 2] = ["calls", "tool_calls"];

/// Renders the closing tag for `name`, in the spaced and unspaced spellings.
fn close_tags(name: &str) -> [String; 2] {
    [format!("{CLOSE} {name}>"), format!("{CLOSE}{name}>")]
}

/// Strips a DSML opening tag prefix, tolerating the optional space.
fn strip_open<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let rest = line.trim_start().strip_prefix(OPEN)?;
    let rest = rest.strip_prefix(' ').unwrap_or(rest);
    rest.strip_prefix(name)
}

/// Returns true when `trimmed` is exactly the closing tag for `name`.
fn is_close(trimmed: &str, name: &str) -> bool {
    close_tags(name).iter().any(|t| t == trimmed)
}

/// Finds the earliest `</DSML parameter>` in `haystack`, in either spelling.
fn find_param_end(haystack: &str) -> Option<(usize, usize)> {
    close_tags("parameter")
        .into_iter()
        .filter_map(|tag| haystack.find(&tag).map(|i| (i, tag.len())))
        .min_by_key(|(i, _)| *i)
}

/// Anchor separating the head and tail halves of an anchored `edit`.
pub const UPTO: &str = "[upto]";

/// One thing the replayer can act on, in transcript order.
#[derive(Debug, Clone)]
pub enum Event {
    /// A tool call the model made.
    Call(Call),
    /// A file recovered from a complete `read` result.
    Seed(Seed),
}

/// A repro file: its header metadata plus every replayable event, in order.
#[derive(Debug, Clone)]
pub struct Repro {
    /// Key/value pairs from the `- key: value` lines of the report header.
    pub meta: BTreeMap<String, String>,
    /// Tool calls and recovered file contents, oldest first.
    pub events: Vec<Event>,
}

impl Repro {
    /// Returns a header value, e.g. `session` or `date`.
    #[must_use]
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.meta.get(key).map(String::as_str)
    }

    /// Returns every tool call, oldest first.
    #[must_use]
    pub fn calls(&self) -> Vec<&Call> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Call(c) => Some(c),
                Event::Seed(_) => None,
            })
            .collect()
    }

    /// Returns only the calls that mutate files (`write` and `edit`).
    #[must_use]
    pub fn file_calls(&self) -> Vec<&Call> {
        self.calls()
            .into_iter()
            .filter(|c| matches!(c.name.as_str(), "write" | "edit"))
            .collect()
    }

    /// Returns the files recovered from complete `read` results.
    #[must_use]
    pub fn seeds(&self) -> Vec<&Seed> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Seed(s) => Some(s),
                Event::Call(_) => None,
            })
            .collect()
    }
}

/// One decoded DSML tool call.
#[derive(Debug, Clone)]
pub struct Call {
    /// Tool name, such as `write`, `edit`, or `bash`.
    pub name: String,
    /// Parameters in declaration order.
    pub params: Vec<(String, String)>,
    /// 1-based line in the repro file where the invocation opened.
    pub line: usize,
}

impl Call {
    /// Returns the value of a named parameter.
    #[must_use]
    pub fn param(&self, name: &str) -> Option<&str> {
        self.params
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// Returns a required parameter.
    ///
    /// # Errors
    ///
    /// Returns an error when the call does not carry the parameter.
    pub fn require(&self, name: &'static str) -> Result<&str, ReplayError> {
        self.param(name)
            .ok_or_else(|| ReplayError::missing_parameter(&self.name, name))
    }
}

/// Reads a repro file from disk and decodes it.
///
/// # Errors
///
/// Returns an error if the file cannot be read, has no transcript, or contains
/// a structurally broken DSML block.
///
/// # Examples
///
/// ```no_run
/// let repro = plank_tools::parse_repro("repro-debug-1789376559.md")?;
/// println!("{} calls", repro.calls().len());
/// # Ok::<(), plank_tools::ReplayError>(())
/// ```
pub fn parse_repro(path: impl AsRef<Path>) -> Result<Repro, ReplayError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| ReplayError::io(path, e))?;
    parse_str(&text)
}

/// Decodes an already-loaded repro document.
///
/// # Errors
///
/// Returns an error when the transcript is absent or a DSML block is broken.
pub fn parse_str(text: &str) -> Result<Repro, ReplayError> {
    let lines: Vec<&str> = text.lines().collect();
    let begin = lines
        .iter()
        .position(|l| l.trim() == BEGIN)
        .ok_or_else(ReplayError::no_transcript)?;
    let end = lines[begin..]
        .iter()
        .position(|l| l.trim() == END)
        .map_or(lines.len(), |i| begin + i);

    let meta = parse_meta(&lines[..begin]);
    let events = parse_events(&lines[begin + 1..end], begin + 2)?;
    Ok(Repro { meta, events })
}

/// Collects `- key: value` pairs from the report header.
pub(crate) fn parse_meta(lines: &[&str]) -> BTreeMap<String, String> {
    let mut meta = BTreeMap::new();
    for line in lines {
        let Some(rest) = line.strip_prefix("- ") else {
            continue;
        };
        let Some((key, value)) = rest.split_once(": ") else {
            continue;
        };
        if key.contains(' ') && key.len() > 20 {
            continue;
        }
        meta.entry(key.to_string())
            .or_insert_with(|| value.trim().to_string());
    }
    meta
}

/// Walks transcript lines, decoding DSML blocks that appear in assistant turns.
///
/// `offset` is the 1-based repro line number of `lines[0]`, used for reporting.
fn parse_events(lines: &[&str], offset: usize) -> Result<Vec<Event>, ReplayError> {
    let mut events = Vec::new();
    let mut in_assistant = false;
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        match line.trim_end() {
            "[assistant]" => {
                in_assistant = true;
                i += 1;
                continue;
            }
            "[user]" | "[system]" | "[tool]" => {
                in_assistant = false;
                let start = i + 1;
                let mut j = start;
                while j < lines.len()
                    && !matches!(
                        lines[j].trim_end(),
                        "[assistant]" | "[user]" | "[system]" | "[tool]"
                    )
                {
                    j += 1;
                }
                events.extend(
                    scan(&lines[start..j], offset + start)
                        .into_iter()
                        .map(Event::Seed),
                );
                i = j;
                continue;
            }
            _ => {}
        }

        let opens_block = BLOCK_NAMES
            .iter()
            .any(|name| strip_open(line, name).is_some_and(|rest| rest.trim() == ">"));
        if !in_assistant || !opens_block {
            i += 1;
            continue;
        }

        i += 1;
        while i < lines.len() {
            let trimmed = lines[i].trim();
            let closes_block = BLOCK_NAMES.iter().any(|name| is_close(trimmed, name));
            if closes_block || trimmed.is_empty() {
                i += 1;
                if closes_block {
                    break;
                }
                continue;
            }
            let Some(name) = invoke_name(trimmed) else {
                // A stray line inside a call block; skip rather than abort.
                i += 1;
                continue;
            };
            let start = i;
            i += 1;
            let (params, next) = parse_params(lines, i, offset)?;
            i = next;
            events.push(Event::Call(Call {
                name,
                params,
                line: offset + start,
            }));
        }
    }

    Ok(events)
}

/// Returns the tool name if the line opens an invocation.
fn invoke_name(trimmed: &str) -> Option<String> {
    let rest = strip_open(trimmed, "invoke")?.strip_prefix(" name=\"")?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Reads every parameter of one invocation, returning the line after it closes.
fn parse_params(
    lines: &[&str],
    mut i: usize,
    offset: usize,
) -> Result<(Vec<(String, String)>, usize), ReplayError> {
    let mut params = Vec::new();
    loop {
        if i >= lines.len() {
            return Err(ReplayError::malformed(
                offset + i,
                "invocation never closed",
            ));
        }
        let trimmed = lines[i].trim();
        if is_close(trimmed, "invoke") {
            return Ok((params, i + 1));
        }
        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        let Some((name, first)) = param_head(lines[i]) else {
            i += 1;
            continue;
        };
        let (value, next) = read_value(lines, first, i, offset)?;
        params.push((name, unescape(&value)));
        i = next;
    }
}

/// Splits a parameter opening line into its name and the first chunk of value.
fn param_head(line: &str) -> Option<(String, &str)> {
    let rest = strip_open(line, "parameter")?.strip_prefix(" name=\"")?;
    let quote = rest.find('"')?;
    let name = rest[..quote].to_string();
    let after = &rest[quote + 1..];
    let gt = after.find('>')?;
    Some((name, &after[gt + 1..]))
}

/// Accumulates a parameter value, which may span many lines.
fn read_value(
    lines: &[&str],
    first: &str,
    start: usize,
    offset: usize,
) -> Result<(String, usize), ReplayError> {
    if let Some((idx, _)) = find_param_end(first) {
        return Ok((first[..idx].to_string(), start + 1));
    }
    let mut value = String::from(first);
    let mut i = start + 1;
    while i < lines.len() {
        value.push('\n');
        if let Some((idx, _)) = find_param_end(lines[i]) {
            value.push_str(&lines[i][..idx]);
            return Ok((value, i + 1));
        }
        value.push_str(lines[i]);
        i += 1;
    }
    Err(ReplayError::malformed(
        offset + start,
        "parameter value never closed",
    ))
}

/// Reverses the two escapes plank applies inside string parameter values.
///
/// Both the spaced and unspaced spellings of the closing tag are recognised.
fn unescape(value: &str) -> String {
    // For each spelling of the close tag: (once-escaped, twice-escaped).
    let forms: Vec<(String, String)> = close_tags("parameter")
        .into_iter()
        .map(|close| (format!("&lt;{}", &close[1..]), close))
        .collect();

    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(idx) = rest.find('&') {
        out.push_str(&rest[..idx]);
        let tail = &rest[idx..];
        let hit = forms.iter().find_map(|(escaped, close)| {
            let double = format!("&amp;{}", &escaped[1..]);
            tail.strip_prefix(double.as_str())
                .map(|after| (escaped.clone(), after))
                .or_else(|| {
                    tail.strip_prefix(escaped.as_str())
                        .map(|after| (close.clone(), after))
                })
        });
        if let Some((text, after)) = hit {
            out.push_str(&text);
            rest = after;
        } else {
            out.push('&');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAG: &str = "<\u{ff5c}DSML\u{ff5c} ";
    const PARAM_END: &str = "</\u{ff5c}DSML\u{ff5c} parameter>";
    const INVOKE_END: &str = "</\u{ff5c}DSML\u{ff5c} invoke>";
    const CALLS_END: &str = "</\u{ff5c}DSML\u{ff5c} calls>";

    fn doc(body: &str) -> String {
        format!("# plank repro v5.1.2\n\n- session: test\n\n{BEGIN}\n{body}\n{END}\n")
    }

    fn call_block(inner: &str) -> String {
        format!("{TAG}calls>\n{inner}\n{CALLS_END}")
    }

    #[test]
    fn reads_header_metadata() {
        let repro = parse_str(&doc("[assistant]\nhi")).unwrap();
        assert_eq!(repro.meta("session"), Some("test"));
    }

    #[test]
    fn missing_transcript_is_an_error() {
        assert!(parse_str("# not a repro\n").is_err());
    }

    #[test]
    fn decodes_a_write_call() {
        let block = call_block(&format!(
            "{TAG}invoke name=\"write\">\n{TAG}parameter name=\"path\" string=\"true\">a.txt{PARAM_END}\n{TAG}parameter name=\"content\" string=\"true\">one\ntwo{PARAM_END}\n{INVOKE_END}"
        ));
        let repro = parse_str(&doc(&format!("[assistant]\n{block}"))).unwrap();
        assert_eq!(repro.calls().len(), 1);
        assert_eq!(repro.calls()[0].name, "write");
        assert_eq!(repro.calls()[0].param("path"), Some("a.txt"));
        assert_eq!(repro.calls()[0].param("content"), Some("one\ntwo"));
    }

    #[test]
    fn ignores_calls_outside_assistant_turns() {
        let block = call_block(&format!(
            "{TAG}invoke name=\"write\">\n{TAG}parameter name=\"path\" string=\"true\">a.txt{PARAM_END}\n{INVOKE_END}"
        ));
        let repro = parse_str(&doc(&format!("[system]\n{block}\n[user]\nhello"))).unwrap();
        assert!(repro.calls().is_empty());
    }

    #[test]
    fn a_user_turn_ends_assistant_scope() {
        let block = call_block(&format!(
            "{TAG}invoke name=\"write\">\n{TAG}parameter name=\"path\" string=\"true\">a.txt{PARAM_END}\n{INVOKE_END}"
        ));
        let body = format!("[assistant]\n{block}\n[user]\n{block}");
        let repro = parse_str(&doc(&body)).unwrap();
        assert_eq!(repro.calls().len(), 1);
    }

    #[test]
    fn decodes_multiple_invocations_in_one_block() {
        let block = call_block(&format!(
            "{TAG}invoke name=\"skill\">\n{INVOKE_END}\n{TAG}invoke name=\"read\">\n{TAG}parameter name=\"path\" string=\"true\">b.md{PARAM_END}\n{INVOKE_END}"
        ));
        let repro = parse_str(&doc(&format!("[assistant]\n{block}"))).unwrap();
        assert_eq!(repro.calls().len(), 2);
        assert_eq!(repro.calls()[1].param("path"), Some("b.md"));
    }

    #[test]
    fn file_calls_filters_to_mutations() {
        let block = call_block(&format!(
            "{TAG}invoke name=\"bash\">\n{TAG}parameter name=\"command\" string=\"true\">ls{PARAM_END}\n{INVOKE_END}\n{TAG}invoke name=\"write\">\n{TAG}parameter name=\"path\" string=\"true\">a.txt{PARAM_END}\n{INVOKE_END}"
        ));
        let repro = parse_str(&doc(&format!("[assistant]\n{block}"))).unwrap();
        assert_eq!(repro.file_calls().len(), 1);
    }

    #[test]
    fn unescapes_parameter_close_tags() {
        assert_eq!(
            unescape("x&lt;/\u{ff5c}DSML\u{ff5c} parameter>y"),
            format!("x{PARAM_END}y")
        );
        assert_eq!(
            unescape("&amp;lt;/\u{ff5c}DSML\u{ff5c} parameter>"),
            "&lt;/\u{ff5c}DSML\u{ff5c} parameter>"
        );
        assert_eq!(unescape("a & b &amp; c"), "a & b &amp; c");
    }

    #[test]
    fn unclosed_parameter_is_an_error() {
        let body = format!(
            "[assistant]\n{TAG}calls>\n{TAG}invoke name=\"write\">\n{TAG}parameter name=\"content\" string=\"true\">oops"
        );
        assert!(parse_str(&doc(&body)).is_err());
    }

    #[test]
    fn records_the_repro_line_of_each_call() {
        let block = call_block(&format!(
            "{TAG}invoke name=\"write\">\n{TAG}parameter name=\"path\" string=\"true\">a.txt{PARAM_END}\n{INVOKE_END}"
        ));
        let repro = parse_str(&doc(&format!("[assistant]\n{block}"))).unwrap();
        assert!(repro.calls()[0].line > 0);
    }

    const OLD_TAG: &str = "<\u{ff5c}DSML\u{ff5c}";
    const OLD_PARAM_END: &str = "</\u{ff5c}DSML\u{ff5c}parameter>";

    #[test]
    fn decodes_the_legacy_unspaced_dialect() {
        let body = format!(
            "[assistant]\n{OLD_TAG}tool_calls>\n{OLD_TAG}invoke name=\"write\">\n{OLD_TAG}parameter name=\"path\" string=\"true\">a.txt{OLD_PARAM_END}\n{OLD_TAG}parameter name=\"content\" string=\"true\">one\ntwo{OLD_PARAM_END}\n</\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>"
        );
        let repro = parse_str(&doc(&body)).unwrap();
        assert_eq!(repro.calls().len(), 1);
        assert_eq!(repro.calls()[0].param("path"), Some("a.txt"));
        assert_eq!(repro.calls()[0].param("content"), Some("one\ntwo"));
    }

    #[test]
    fn unescapes_the_legacy_spelling_too() {
        assert_eq!(
            unescape("x&lt;/\u{ff5c}DSML\u{ff5c}parameter>y"),
            format!("x{OLD_PARAM_END}y")
        );
    }
}
