//! Recovers pre-existing file contents from `read` tool results.
//!
//! A repro only records the calls the model made, so an `edit` against a file
//! the session did not itself create has nothing to apply to. When the model
//! read that file first, though, the transcript carries its full text in a
//! numbered-line tool result, which is enough to seed the file before the edit.

/// A file whose contents were recovered from a complete `read` result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seed {
    /// Path exactly as the read result reported it.
    pub path: String,
    /// Full file contents, with the line numbers stripped.
    pub content: String,
    /// 1-based line in the repro file where the result appeared.
    pub line: usize,
}

/// Scans one `[user]` turn's lines for complete `read` results.
///
/// Partial reads are ignored: seeding from a truncated file would let a later
/// `edit` succeed against text that is not what the session actually saw.
pub(crate) fn scan(lines: &[&str], offset: usize) -> Vec<Seed> {
    let mut seeds = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if !is_read_result(lines[i]) {
            i += 1;
            continue;
        }
        i += 1;
        let Some(header) = lines.get(i) else { break };
        let Some(path) = complete_read_path(header) else {
            continue;
        };
        let start = i + 1;
        let (content, next) = take_numbered(lines, start);
        if next > start {
            seeds.push(Seed {
                path,
                content,
                line: offset + i,
            });
        }
        i = next;
    }
    seeds
}

/// Returns true for the `Tool result N (read):` banner, with or without a prefix tag.
fn is_read_result(line: &str) -> bool {
    let line = line.trim_start_matches("<tool_result>");
    line.starts_with("Tool result ") && line.trim_end().ends_with("(read):")
}

/// Returns the path when the header describes a read covering the whole file.
fn complete_read_path(header: &str) -> Option<String> {
    let (path, rest) = header.rsplit_once(": lines ")?;
    let rest = rest.split(';').next()?.trim();
    let (range, total) = rest.split_once(" of ")?;
    let (first, last) = range.split_once('-')?;
    if first.trim() != "1" || last.trim() != total.trim() {
        return None;
    }
    total.trim().parse::<usize>().ok()?;
    Some(path.trim().to_string())
}

/// Consumes consecutive `N content` lines, returning the joined text.
fn take_numbered(lines: &[&str], start: usize) -> (String, usize) {
    let mut out = String::new();
    let mut expected = 1usize;
    let mut i = start;
    while i < lines.len() {
        let Some(body) = strip_number(lines[i], expected) else {
            break;
        };
        if expected > 1 {
            out.push('\n');
        }
        out.push_str(body);
        expected += 1;
        i += 1;
    }
    if i > start {
        out.push('\n');
    }
    (out, i)
}

/// Strips the leading `expected` line number and its single separating space.
fn strip_number(line: &str, expected: usize) -> Option<&str> {
    let rest = line.strip_prefix(&expected.to_string())?;
    if rest.is_empty() {
        return Some("");
    }
    rest.strip_prefix(' ')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_a_complete_read() {
        let lines = vec![
            "<tool_result>Tool result 1 (read):",
            "src/a.rs: lines 1-3 of 3",
            "1 one",
            "2 ",
            "3 three",
            "</tool_result>",
        ];
        let seeds = scan(&lines, 1);
        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].path, "src/a.rs");
        assert_eq!(seeds[0].content, "one\n\nthree\n");
    }

    #[test]
    fn ignores_a_partial_read() {
        let lines = vec![
            "Tool result 2 (read):",
            "src/ui.rs: lines 1-100 of 9441; continue_offset=101",
            "1 one",
        ];
        assert!(scan(&lines, 1).is_empty());
    }

    #[test]
    fn ignores_results_from_other_tools() {
        let lines = vec!["Tool result 1 (bash):", "src/a.rs: lines 1-1 of 1", "1 one"];
        assert!(scan(&lines, 1).is_empty());
    }

    #[test]
    fn stops_at_the_first_non_numbered_line() {
        let lines = vec![
            "Tool result 1 (read):",
            "a.txt: lines 1-2 of 2",
            "1 one",
            "2 two",
            "Tool result 2 (bash):",
            "unrelated",
        ];
        assert_eq!(scan(&lines, 1)[0].content, "one\ntwo\n");
    }

    #[test]
    fn handles_absolute_paths() {
        let lines = vec!["Tool result 1 (read):", "/tmp/x.md: lines 1-1 of 1", "1 hi"];
        assert_eq!(scan(&lines, 1)[0].path, "/tmp/x.md");
    }
}
