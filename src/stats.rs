//! Summarises a repro file: generation throughput, tool usage, and guard errors.
//!
//! Nothing here touches the filesystem. [`Stats::collect`] walks the already
//! loaded document once and [`Stats::render`] prints a single-page report.

#![allow(
    clippy::cast_precision_loss,
    reason = "counts and rates are reported for humans, not accounting"
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::parse::{Event, Repro, parse_meta, parse_str};

/// Marker that opens the embedded transcript.
const BEGIN: &str = "----- BEGIN TRANSCRIPT -----";
/// Marker that closes the embedded transcript.
const END: &str = "----- END TRANSCRIPT -----";
/// Prefix every tool failure line shares inside a tool result.
const TOOL_ERROR: &str = "Tool error:";
/// Width, in characters, of the histogram bars.
const BAR: usize = 24;
/// Lines of the recorded prompt the report echoes before eliding the rest.
const PROMPT_LINES: usize = 12;

/// Tags wrapping text plank injects into a user turn around the real prompt.
const WRAPPERS: [&str; 3] = ["hook_context", "system-reminder", "tool_result"];

/// Openings of a user turn plank generated wholesale, with no prompt in it.
const GENERATED: [&str; 2] = ["Agent instructions:", "Today's date is "];

/// The ANSI escapes the report paints with, or empty strings when colour is off.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    /// Section headings.
    pub head: &'static str,
    /// Field labels and other secondary text.
    pub dim: &'static str,
    /// Counts and other figures worth spotting.
    pub value: &'static str,
    /// Something that went wrong.
    pub bad: &'static str,
    /// Something that went well.
    pub good: &'static str,
    /// The prompt the session started from.
    pub quote: &'static str,
    /// Returns to the terminal default.
    pub off: &'static str,
}

impl Style {
    /// The colourless style, for pipes, files, and `NO_COLOR`.
    #[must_use]
    pub const fn plain() -> Self {
        Self {
            head: "",
            dim: "",
            value: "",
            bad: "",
            good: "",
            quote: "",
            off: "",
        }
    }

    /// The 16-colour ANSI style.
    #[must_use]
    pub const fn ansi() -> Self {
        Self {
            head: "\u{1b}[1;36m",
            dim: "\u{1b}[2m",
            value: "\u{1b}[1m",
            bad: "\u{1b}[31m",
            good: "\u{1b}[32m",
            quote: "\u{1b}[33m",
            off: "\u{1b}[0m",
        }
    }

    /// Picks the style for a stream: colour only on a terminal, and never when
    /// `NO_COLOR` is set.
    #[must_use]
    pub fn detect(is_terminal: bool) -> Self {
        if is_terminal && std::env::var_os("NO_COLOR").is_none() {
            Self::ansi()
        } else {
            Self::plain()
        }
    }

    /// Colours `text` as a count: green at zero, red above it.
    fn tally(self, count: usize) -> String {
        let colour = if count == 0 { self.good } else { self.bad };
        format!("{colour}{count}{}", self.off)
    }
}

/// How the session ended, as read from the last generation pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The model answered instead of calling another tool: the goal was reached.
    GoalReached,
    /// The user stopped generation.
    Interrupted,
    /// A loop guard cut the pass short.
    Guarded,
    /// The session ended part-way through a tool call, or left no passes.
    Unfinished,
}

impl Verdict {
    /// Reads the verdict off the stop reason of the last pass.
    #[must_use]
    pub fn of(stop: Option<&str>) -> Self {
        match stop {
            Some("answer") => Self::GoalReached,
            Some("interrupted by user") => Self::Interrupted,
            Some(other) if other.starts_with("guard:") => Self::Guarded,
            _ => Self::Unfinished,
        }
    }

    /// The phrase the report prints.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::GoalReached => "goal reached",
            Self::Interrupted => "interrupted by user",
            Self::Guarded => "cut by a loop guard",
            Self::Unfinished => "ended mid-task",
        }
    }

    /// The colour that phrase is painted in.
    #[must_use]
    const fn colour(self, style: Style) -> &'static str {
        match self {
            Self::GoalReached => style.good,
            Self::Interrupted | Self::Guarded => style.bad,
            Self::Unfinished => style.quote,
        }
    }
}

/// One row of the `## Passes` table: a single generation pass.
#[derive(Debug, Clone)]
pub struct Pass {
    /// Wall-clock time the pass ended, as written in the table.
    pub ended: String,
    /// Which agent generated it: `main`, or a sub-agent name.
    pub agent: String,
    /// Tokens the pass emitted.
    pub tokens: u64,
    /// Decode speed the engine reported for the pass.
    pub tok_per_sec: f64,
    /// Bytes of reasoning the loop guard saw inside `<think>`.
    pub reasoning: u64,
    /// Why generation stopped, e.g. `tool calls: 2` or `guard: draft`.
    pub stop: String,
}

/// Everything the `--stats` report needs, collected in one pass over the file.
#[derive(Debug, Default)]
pub struct Stats {
    /// Header metadata, `key: value`, from every `- key: value` line.
    pub meta: BTreeMap<String, String>,
    /// The `## Passes` table, oldest first.
    pub passes: Vec<Pass>,
    /// Tool results seen in the transcript, counted per tool name.
    pub tools: BTreeMap<String, usize>,
    /// `Tool error:` messages, counted per distinct message.
    pub errors: BTreeMap<String, usize>,
    /// Transcript turns counted per role marker.
    pub turns: BTreeMap<String, usize>,
    /// First and last `[timestamp]` lines of the transcript.
    pub span: Option<(String, String)>,
    /// Seconds between the first and last transcript timestamp, when stated.
    pub elapsed_sec: Option<u64>,
    /// Bash commands whose result reported a non-zero exit status.
    pub failed_shells: usize,
    /// Bash commands whose result reported a clean exit.
    pub ok_shells: usize,
    /// Title line of the repro, e.g. `# plank repro v5.1.3 BETA`.
    pub title: String,
    /// Size of the repro file in bytes.
    pub bytes: usize,
    /// Lines in the repro file.
    pub lines: usize,
    /// DSML calls the replayer decoded.
    pub calls: usize,
    /// Of those, the `write` and `edit` calls.
    pub file_calls: usize,
    /// Files recovered from complete `read` results.
    pub seeds: usize,
    /// Distinct paths the decoded `write`/`edit` calls touch.
    pub touched: BTreeMap<String, usize>,
    /// Bytes of file content the decoded `write` calls carry.
    pub written_bytes: usize,
    /// Why the tool calls could not be decoded, when the transcript is broken.
    pub parse_note: Option<String>,
    /// The prompt the session started from, as the human typed it.
    pub prompt: Option<String>,
}

impl Stats {
    /// Summarises a repro document, tolerating a transcript the parser rejects.
    ///
    /// A truncated session can carry an unterminated tool call; the header and
    /// the transcript are still worth reporting, so the decode failure is kept
    /// as [`Stats::parse_note`] rather than propagated.
    #[must_use]
    pub fn of(text: &str) -> Self {
        match parse_str(text) {
            Ok(repro) => Self::collect(text, &repro),
            Err(error) => {
                let empty = Repro {
                    meta: BTreeMap::new(),
                    events: Vec::new(),
                };
                let mut stats = Self::collect(text, &empty);
                stats.parse_note = Some(error.to_string());
                stats
            }
        }
    }

    /// Summarises an already-parsed repro together with its raw text.
    #[must_use]
    pub fn collect(text: &str, repro: &Repro) -> Self {
        let lines: Vec<&str> = text.lines().collect();
        let begin = lines.iter().position(|l| l.trim() == BEGIN);
        let end = lines.iter().position(|l| l.trim() == END);
        let body = begin.map_or(0, |b| b + 1)..end.unwrap_or(lines.len());

        let mut stats = Self {
            meta: parse_meta(&lines[..begin.unwrap_or(lines.len())]),
            title: lines.first().unwrap_or(&"").trim().to_string(),
            bytes: text.len(),
            lines: lines.len(),
            ..Self::default()
        };
        stats.read_passes(&lines[..begin.unwrap_or(lines.len())]);
        stats.read_transcript(&lines[body]);
        stats.read_events(repro);
        stats
    }

    /// Parses the `## Passes` markdown table out of the report header.
    fn read_passes(&mut self, header: &[&str]) {
        let Some(start) = header.iter().position(|l| l.trim() == "## Passes") else {
            return;
        };
        for line in &header[start..] {
            let trimmed = line.trim();
            if trimmed.starts_with("## ") && trimmed != "## Passes" {
                break;
            }
            let Some(row) = trimmed.strip_prefix('|').and_then(|r| r.strip_suffix('|')) else {
                continue;
            };
            let cells: Vec<&str> = row.split('|').map(str::trim).collect();
            // `#, ended, delta, agent, tokens, tok/s, reasoning, cycle, headings, fenced, stop`
            if cells.len() < 11 || cells[0].parse::<u32>().is_err() {
                continue;
            }
            self.passes.push(Pass {
                ended: cells[1].to_string(),
                agent: cells[3].to_string(),
                tokens: cells[4].parse().unwrap_or(0),
                tok_per_sec: cells[5].parse().unwrap_or(0.0),
                reasoning: cells[6].parse().unwrap_or(0),
                stop: cells[10].to_string(),
            });
        }
    }

    /// Counts turns, tool results, tool errors, and shell outcomes.
    ///
    /// The walk also keeps the last human user turn before the first assistant
    /// reply, which is the prompt the session started from.
    fn read_transcript(&mut self, body: &[&str]) {
        let mut turn: Option<Vec<&str>> = None;
        let mut answered = false;

        for line in body {
            let trimmed = line.trim_end();
            if let Some(role) = trimmed
                .strip_prefix('[')
                .and_then(|r| r.strip_suffix(']'))
                .filter(|r| matches!(*r, "assistant" | "user" | "system" | "tool"))
            {
                *self.turns.entry(role.to_string()).or_default() += 1;
                if let Some(lines) = turn.take() {
                    self.note_prompt(&lines);
                }
                answered |= role == "assistant";
                turn = (role == "user" && !answered).then(Vec::new);
                continue;
            }
            if let Some(lines) = turn.as_mut()
                && !trimmed.starts_with("[timestamp] ")
            {
                lines.push(trimmed);
            }
            if let Some(rest) = trimmed.strip_prefix("[timestamp] ") {
                self.note_timestamp(rest);
            } else if let Some(name) = tool_result_name(trimmed) {
                *self.tools.entry(name).or_default() += 1;
            } else if let Some(idx) = trimmed.find(TOOL_ERROR) {
                let message = trimmed[idx + TOOL_ERROR.len()..].trim();
                *self.errors.entry(shorten(message)).or_default() += 1;
            } else if let Some(code) = field(trimmed, "exit_status=") {
                if code == "0" {
                    self.ok_shells += 1;
                } else {
                    self.failed_shells += 1;
                }
            }
        }
        if let Some(lines) = turn {
            self.note_prompt(&lines);
        }
    }

    /// Keeps a user turn as the prompt unless it is one plank injected itself.
    ///
    /// Sessions open with generated turns — the date, hook context, the agent
    /// instructions assembled from `CLAUDE.md` — and a sub-agent's task arrives
    /// wrapped in a `<system-reminder>`. The prompt is what is left of the last
    /// such turn before the model first answers.
    fn note_prompt(&mut self, lines: &[&str]) {
        let text = strip_wrappers(&lines.join("\n"));
        let body = text.trim();
        let first = body.lines().next().unwrap_or_default();
        if body.is_empty()
            || first.starts_with('<')
            || GENERATED.iter().any(|tag| first.starts_with(tag))
        {
            return;
        }
        self.prompt = Some(body.to_string());
    }

    /// Records the transcript span and the `+5h7m` offset the line carries.
    fn note_timestamp(&mut self, rest: &str) {
        let stamp = rest
            .split_whitespace()
            .take(2)
            .collect::<Vec<_>>()
            .join(" ");
        match &mut self.span {
            Some((_, last)) => *last = stamp,
            none => *none = Some((stamp.clone(), stamp)),
        }
        if let Some(seconds) = rest
            .rsplit_once('+')
            .and_then(|(_, tail)| parse_delta(tail))
        {
            self.elapsed_sec = Some(self.elapsed_sec.unwrap_or(0).max(seconds));
        }
    }

    /// Tallies what the replayer would actually rebuild.
    fn read_events(&mut self, repro: &Repro) {
        self.calls = repro.calls().len();
        self.seeds = repro.seeds().len();
        for event in &repro.events {
            let Event::Call(call) = event else { continue };
            if !matches!(call.name.as_str(), "write" | "edit") {
                continue;
            }
            self.file_calls += 1;
            if let Some(path) = call.param("path") {
                *self.touched.entry(path.to_string()).or_default() += 1;
            }
            if call.name == "write" {
                self.written_bytes += call.param("content").map_or(0, str::len);
            }
        }
    }

    /// Tokens generated across every pass.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.passes.iter().map(|p| p.tokens).sum()
    }

    /// Throughput over the whole session: tokens divided by the decode time
    /// implied by each pass's own rate, so idle time between passes is excluded.
    #[must_use]
    pub fn average_tok_per_sec(&self) -> Option<f64> {
        let decode: f64 = self
            .passes
            .iter()
            .filter(|p| p.tok_per_sec > 0.0)
            .map(|p| p.tokens as f64 / p.tok_per_sec)
            .sum();
        let tokens: u64 = self
            .passes
            .iter()
            .filter(|p| p.tok_per_sec > 0.0)
            .map(|p| p.tokens)
            .sum();
        (decode > 0.0).then(|| tokens as f64 / decode)
    }

    /// Passes that a loop guard cut short.
    #[must_use]
    pub fn guard_stops(&self) -> usize {
        self.passes
            .iter()
            .filter(|p| p.stop.starts_with("guard:"))
            .count()
    }

    /// How the session ended, from the stop reason of its last pass.
    #[must_use]
    pub fn verdict(&self) -> Verdict {
        Verdict::of(self.passes.last().map(|p| p.stop.as_str()))
    }

    /// Seconds actually spent decoding, summed over the passes.
    #[must_use]
    pub fn decode_sec(&self) -> f64 {
        self.passes
            .iter()
            .filter(|p| p.tok_per_sec > 0.0)
            .map(|p| p.tokens as f64 / p.tok_per_sec)
            .sum()
    }

    /// Tool errors the model was handed, across every message.
    #[must_use]
    pub fn tool_errors(&self) -> usize {
        self.errors.values().sum()
    }

    /// Renders the whole report as one page of text.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one straight-line report layout")]
    pub fn render(&self, path: &std::path::Path, style: Style) -> String {
        let mut out = String::new();
        let meta = |k: &str| self.meta.get(k).map_or("-", String::as_str);
        let row = |out: &mut String, label: &str, value: &str| {
            let _ = writeln!(out, "  {}{label:<8}{} {value}", style.dim, style.off);
        };

        rule(&mut out, &self.title, style);
        row(&mut out, "file", &path.display().to_string());
        row(
            &mut out,
            "size",
            &format!("{} in {} lines", human_bytes(self.bytes), self.lines),
        );
        row(
            &mut out,
            "session",
            &format!(
                "{}{}{}   {}date{} {}   {}note{} {}",
                style.value,
                meta("session"),
                style.off,
                style.dim,
                style.off,
                meta("date"),
                style.dim,
                style.off,
                meta("note"),
            ),
        );
        row(
            &mut out,
            "model",
            &format!(
                "{}{}{}   {}family{} {}   {}dialect{} {}",
                style.value,
                meta("name"),
                style.off,
                style.dim,
                style.off,
                meta("family"),
                style.dim,
                style.off,
                meta("tool dialect"),
            ),
        );
        row(
            &mut out,
            "sampling",
            &format!(
                "think {}   temp {}   top_p {}   seed {}   guards {}",
                meta("think mode"),
                meta("temperature"),
                meta("top_p"),
                meta("seed"),
                meta("loop guards"),
            ),
        );
        let used = meta("last ctx used").parse::<f64>().ok();
        let size = meta("context size").parse::<f64>().ok();
        let share = match (used, size) {
            (Some(u), Some(s)) if s > 0.0 => format!("  ({:.1}% of window)", 100.0 * u / s),
            _ => String::new(),
        };
        row(
            &mut out,
            "context",
            &format!(
                "{}{}{} of {} tokens{}   transcript {}",
                style.value,
                meta("last ctx used"),
                style.off,
                meta("context size"),
                share,
                meta("transcript tokens"),
            ),
        );

        let verdict = self.verdict();
        let wall = self.elapsed_sec.map_or_else(
            || "unrecorded".to_string(),
            |seconds| format!("{} total", human_time(seconds)),
        );
        let decode = self.decode_sec();
        row(
            &mut out,
            "outcome",
            &format!(
                "{}{}{}   {}wall{} {}{}",
                verdict.colour(style),
                verdict.label(),
                style.off,
                style.dim,
                style.off,
                wall,
                if decode > 0.0 {
                    format!(", {} generating", human_time(whole_seconds(decode)))
                } else {
                    String::new()
                },
            ),
        );

        if let Some(prompt) = &self.prompt {
            rule(&mut out, "Prompt", style);
            let lines: Vec<&str> = prompt.lines().collect();
            for line in lines.iter().take(PROMPT_LINES) {
                let text: String = line.chars().take(96).collect();
                let _ = writeln!(out, "  {}{}{}", style.quote, text, style.off);
            }
            if lines.len() > PROMPT_LINES {
                let _ = writeln!(
                    out,
                    "  {}... {} more line(s){}",
                    style.dim,
                    lines.len() - PROMPT_LINES,
                    style.off,
                );
            }
        }

        rule(&mut out, "Generation", style);
        if self.passes.is_empty() {
            let _ = writeln!(
                out,
                "  {}(this repro carries no Passes table){}",
                style.dim, style.off
            );
        } else {
            let rates: Vec<f64> = self
                .passes
                .iter()
                .map(|p| p.tok_per_sec)
                .filter(|r| *r > 0.0)
                .collect();
            let fastest = rates.iter().copied().fold(f64::MIN, f64::max);
            let slowest = rates.iter().copied().fold(f64::MAX, f64::min);
            row(
                &mut out,
                "passes",
                &format!(
                    "{}{}{}   tokens {}{}{}   reasoning {}",
                    style.value,
                    self.passes.len(),
                    style.off,
                    style.value,
                    self.total_tokens(),
                    style.off,
                    human_bytes(
                        self.passes
                            .iter()
                            .map(|p| usize::try_from(p.reasoning).unwrap_or(usize::MAX))
                            .sum()
                    ),
                ),
            );
            row(
                &mut out,
                "tok/s",
                &format!(
                    "{}{}{}  (fastest {:.1}, slowest {:.1})",
                    style.value,
                    self.average_tok_per_sec()
                        .map_or_else(|| "-".to_string(), |r| format!("{r:.1} avg")),
                    style.off,
                    if rates.is_empty() { 0.0 } else { fastest },
                    if rates.is_empty() { 0.0 } else { slowest },
                ),
            );
            if let Some(p) = self.passes.iter().max_by_key(|p| p.tokens) {
                row(
                    &mut out,
                    "longest",
                    &format!(
                        "pass of {} tokens at {:.1} tok/s, stopped on {}",
                        p.tokens, p.tok_per_sec, p.stop,
                    ),
                );
            }
            let mut agents: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
            for pass in &self.passes {
                let entry = agents.entry(pass.agent.as_str()).or_default();
                entry.0 += 1;
                entry.1 += pass.tokens;
            }
            let summary = agents
                .iter()
                .map(|(name, (n, tok))| format!("{name} {n}x/{tok}tok"))
                .collect::<Vec<_>>()
                .join("   ");
            row(&mut out, "agents", &summary);

            let mut stops: BTreeMap<&str, usize> = BTreeMap::new();
            for pass in &self.passes {
                *stops.entry(pass.stop.as_str()).or_default() += 1;
            }
            let _ = writeln!(out, "\n  {}stop reasons{}", style.dim, style.off);
            histogram(&mut out, &stops, self.passes.len(), 8, style, stop_is_bad);
        }

        rule(&mut out, "Tools", style);
        let invocations: usize = self.tools.values().sum();
        if invocations == 0 {
            let _ = writeln!(
                out,
                "  {}(no tool results in the transcript){}",
                style.dim, style.off
            );
        } else {
            let _ = writeln!(
                out,
                "  {}{invocations}{} invocation(s) across {} distinct tool(s)\n",
                style.value,
                style.off,
                self.tools.len()
            );
            let by_name: BTreeMap<&str, usize> =
                self.tools.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            histogram(&mut out, &by_name, invocations, 12, style, |_| false);
        }
        if self.ok_shells + self.failed_shells > 0 {
            out.push('\n');
            row(
                &mut out,
                "shell",
                &format!(
                    "{}{}{} command(s) exited clean, {} non-zero",
                    style.good,
                    self.ok_shells,
                    style.off,
                    style.tally(self.failed_shells),
                ),
            );
        }

        rule(&mut out, "Guard and tool errors", style);
        let _ = writeln!(
            out,
            "  {} tool error(s) returned to the model, {} pass(es) cut by a loop guard",
            style.tally(self.tool_errors()),
            style.tally(self.guard_stops()),
        );
        if self.errors.is_empty() {
            let _ = writeln!(
                out,
                "  {}clean run: nothing was refused or failed{}",
                style.good, style.off
            );
        } else {
            out.push('\n');
            let by_msg: BTreeMap<&str, usize> =
                self.errors.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            histogram(&mut out, &by_msg, self.tool_errors(), 8, style, |_| true);
        }

        rule(&mut out, "Workspace the replay would rebuild", style);
        if let Some(note) = &self.parse_note {
            let _ = writeln!(
                out,
                "  {}calls could not be decoded: {note}{}",
                style.bad, style.off
            );
        }
        let _ = writeln!(
            out,
            "  {}{}{} decoded call(s), {}{}{} file-mutating, {} file(s) recoverable from reads",
            style.value, self.calls, style.off, style.value, self.file_calls, style.off, self.seeds,
        );
        let _ = writeln!(
            out,
            "  {} distinct path(s), {} of written content",
            self.touched.len(),
            human_bytes(self.written_bytes),
        );
        let mut hottest: Vec<(&String, &usize)> = self.touched.iter().collect();
        hottest.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        for (path, count) in hottest.iter().take(8) {
            let _ = writeln!(out, "    {}{count:>3}x{}  {path}", style.value, style.off);
        }
        if hottest.len() > 8 {
            let _ = writeln!(
                out,
                "    {}... and {} more{}",
                style.dim,
                hottest.len() - 8,
                style.off
            );
        }

        rule(&mut out, "Transcript", style);
        let turns = self
            .turns
            .iter()
            .map(|(role, n)| format!("{role} {n}"))
            .collect::<Vec<_>>()
            .join("   ");
        row(&mut out, "turns", &turns);
        if let Some((first, last)) = &self.span {
            row(
                &mut out,
                "span",
                &format!(
                    "{first}  ->  {last}{}",
                    self.elapsed_sec.map_or_else(String::new, |s| format!(
                        "   ({}{}{})",
                        style.value,
                        human_time(s),
                        style.off
                    )),
                ),
            );
        }
        let tokens = self.total_tokens();
        if let Some(seconds) = self.elapsed_sec.filter(|s| *s > 0 && tokens > 0) {
            row(
                &mut out,
                "wall",
                &format!(
                    "{:.1} tok/s including thinking, tools, and idle time",
                    tokens as f64 / seconds as f64,
                ),
            );
        }
        out
    }
}

/// Whether a pass stop reason is worth flagging in red.
fn stop_is_bad(stop: &str) -> bool {
    stop.starts_with("guard:") || stop == "tool error" || stop == "interrupted by user"
}

/// Removes the `<system-reminder>`-style envelopes plank wraps a turn in.
///
/// An envelope that is never closed runs to the end of the turn.
fn strip_wrappers(text: &str) -> String {
    let mut out = text.to_string();
    for tag in WRAPPERS {
        let (open, close) = (format!("<{tag}>"), format!("</{tag}>"));
        while let Some(start) = out.find(&open) {
            let rest = &out[start + open.len()..];
            let end = rest
                .find(&close)
                .map_or(out.len(), |i| start + open.len() + i + close.len());
            out.replace_range(start..end, "");
        }
    }
    out
}

/// Returns the tool name of a `Tool result N (name):` line.
fn tool_result_name(line: &str) -> Option<String> {
    const HEAD: &str = "Tool result ";
    let rest = &line[line.find(HEAD)? + HEAD.len()..];
    let open = rest.find('(')?;
    rest[..open].trim().parse::<u32>().ok()?;
    let close = rest[open + 1..].find(')')?;
    Some(rest[open + 1..open + 1 + close].to_string())
}

/// Reads the value of a `key=value` field, up to the next space.
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let rest = &line[line.find(key)? + key.len()..];
    Some(rest.split_whitespace().next().unwrap_or(rest))
}

/// Trims a long error message to one readable line.
fn shorten(message: &str) -> String {
    let first = message.lines().next().unwrap_or(message).trim();
    if first.chars().count() <= 64 {
        return first.to_string();
    }
    let cut: String = first.chars().take(61).collect();
    format!("{cut}...")
}

/// Writes a section heading.
fn rule(out: &mut String, title: &str, style: Style) {
    let _ = writeln!(
        out,
        "\n{}{title}{}\n{}{}{}",
        style.head,
        style.off,
        style.dim,
        "-".repeat(title.chars().count()),
        style.off,
    );
}

/// Writes a sorted bar chart of `counts`, keeping the `top` largest rows.
///
/// `alarming` decides which rows are painted as trouble rather than as data.
fn histogram(
    out: &mut String,
    counts: &BTreeMap<&str, usize>,
    total: usize,
    top: usize,
    style: Style,
    alarming: impl Fn(&str) -> bool,
) {
    let mut rows: Vec<(&&str, &usize)> = counts.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let width = rows
        .iter()
        .take(top)
        .map(|(label, _)| label.chars().count().min(40))
        .max()
        .unwrap_or(0);
    for (label, count) in rows.iter().take(top) {
        let filled = if total == 0 {
            0
        } else {
            (**count * BAR).div_ceil(total)
        };
        let colour = if alarming(label) {
            style.bad
        } else {
            style.head
        };
        let text: String = label.chars().take(40).collect();
        let share = if total == 0 {
            0.0
        } else {
            100.0 * **count as f64 / total as f64
        };
        let bar = format!("{colour}{}{}", "\u{2588}".repeat(filled), style.off);
        let pad = " ".repeat(BAR - filled);
        let _ = writeln!(
            out,
            "    {text:<width$}  {bar}{pad} {}{count:>4}{}  {}{share:>5.1}%{}",
            style.value, style.off, style.dim, style.off,
        );
    }
    if rows.len() > top {
        let rest: usize = rows.iter().skip(top).map(|(_, c)| **c).sum();
        let _ = writeln!(
            out,
            "    {}... {} more ({rest} total){}",
            style.dim,
            rows.len() - top,
            style.off,
        );
    }
}

/// Formats a byte count in the largest unit that keeps it readable.
fn human_bytes(bytes: usize) -> String {
    match bytes {
        0..1024 => format!("{bytes} B"),
        b if b < 1024 * 1024 => format!("{:.1} KiB", b as f64 / 1024.0),
        b => format!("{:.1} MiB", b as f64 / (1024.0 * 1024.0)),
    }
}

/// Reads a duration written as `5h7m`, `4m53s`, or `42s`.
fn parse_delta(text: &str) -> Option<u64> {
    let mut seconds = 0u64;
    let mut digits = String::new();
    let mut seen = false;
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            continue;
        }
        let Ok(value) = digits.parse::<u64>() else {
            break;
        };
        digits.clear();
        seconds += match ch {
            'h' => value * 3600,
            'm' => value * 60,
            's' => value,
            _ => break,
        };
        seen = true;
    }
    seen.then_some(seconds)
}

/// Rounds a duration in seconds to whole seconds, clamping the absurd.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to a non-negative range a u64 holds"
)]
fn whole_seconds(seconds: f64) -> u64 {
    seconds.round().clamp(0.0, 1e18) as u64
}

/// Formats a duration in seconds as `1h 02m 03s`, dropping empty leading units.
fn human_time(seconds: u64) -> String {
    let (h, m, s) = (seconds / 3600, (seconds % 3600) / 60, seconds % 60);
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "\
# plank repro v5.1.3 BETA

- date: 2026-09-14
- session: dorky-gauss
- context size: 1000
- last ctx used: 250
- name: Qwen3.8 Flash Next

## Passes

| # | ended | \u{394} | agent | tokens | tok/s | reasoning | cycle | headings | fenced | stop |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-14 21:41:30 |  | main | 100 | 50.0 | 120 | - | 0 | 0 | tool calls: 2 |
| 2 | 2026-09-14 21:41:34 | +4s | main | 300 | 25.0 | 180 | - | 0 | 0 | guard: draft |

----- BEGIN TRANSCRIPT -----
[system]
setup
[user]
Today's date is 2026-09-14.
[user]
<hook_context>you have superpowers</hook_context>
[user]
write the snake game
[assistant]
hi
[user]
<tool_result>Tool result 1 (bash):
bash job=1 pid=1 status=done elapsed_sec=0.1 timed_out=0
exit_status=1
[user]
<tool_result>Tool result 2 (edit):
Tool error: refused by loop guard - same call twice
[timestamp] 2026-09-14 21:41:17 (msg 0, +0s)
[timestamp] 2026-09-14 21:41:59 (end, +42s)
----- END TRANSCRIPT -----
";

    fn stats() -> Stats {
        Stats::of(DOC)
    }

    #[test]
    fn a_broken_transcript_still_reports_the_header() {
        let broken = DOC.replace("[system]", "[assistant]\n<\u{ff5c}DSML\u{ff5c} calls>\n<\u{ff5c}DSML\u{ff5c} invoke name=\"write\">");
        let stats = Stats::of(&broken);
        assert!(stats.parse_note.is_some());
        assert_eq!(stats.passes.len(), 2);
        assert_eq!(stats.calls, 0);
    }

    #[test]
    fn reads_the_passes_table() {
        let stats = stats();
        assert_eq!(stats.passes.len(), 2);
        assert_eq!(stats.total_tokens(), 400);
        assert_eq!(stats.passes[1].stop, "guard: draft");
    }

    #[test]
    fn averages_throughput_over_decode_time_only() {
        // 100 tokens at 50/s plus 300 at 25/s is 400 tokens in 14 seconds.
        let rate = stats().average_tok_per_sec().unwrap();
        assert!((rate - 400.0 / 14.0).abs() < 1e-6, "{rate}");
    }

    #[test]
    fn counts_guard_stops_and_tool_errors() {
        let stats = stats();
        assert_eq!(stats.guard_stops(), 1);
        assert_eq!(stats.tool_errors(), 1);
        assert!(stats.errors.keys().next().unwrap().starts_with("refused"));
    }

    #[test]
    fn counts_tool_results_and_shell_outcomes() {
        let stats = stats();
        assert_eq!(stats.tools.get("bash"), Some(&1));
        assert_eq!(stats.tools.get("edit"), Some(&1));
        assert_eq!(stats.failed_shells, 1);
        assert_eq!(stats.ok_shells, 0);
    }

    #[test]
    fn records_turns_and_span() {
        let stats = stats();
        assert_eq!(stats.turns.get("user"), Some(&5));
        assert_eq!(stats.elapsed_sec, Some(42));
        let (first, last) = stats.span.clone().unwrap();
        assert_eq!(first, "2026-09-14 21:41:17");
        assert_eq!(last, "2026-09-14 21:41:59");
    }

    #[test]
    fn strips_wrappers_around_a_delegated_task() {
        let turn = [
            "<system-reminder>",
            "you are a subagent",
            "</system-reminder>",
            "review src/app.rs",
        ];
        let mut stats = Stats::default();
        stats.note_prompt(&turn);
        assert_eq!(stats.prompt.as_deref(), Some("review src/app.rs"));
    }

    #[test]
    fn an_unterminated_wrapper_swallows_the_rest_of_the_turn() {
        let mut stats = Stats::default();
        stats.note_prompt(&["<hook_context>skills are on"]);
        assert!(stats.prompt.is_none());
    }

    #[test]
    fn captures_the_prompt_the_session_started_from() {
        assert_eq!(stats().prompt.as_deref(), Some("write the snake game"));
    }

    #[test]
    fn reads_the_verdict_off_the_last_pass() {
        assert_eq!(stats().verdict(), Verdict::Guarded);
        assert_eq!(Verdict::of(Some("answer")), Verdict::GoalReached);
        assert_eq!(
            Verdict::of(Some("interrupted by user")),
            Verdict::Interrupted
        );
        assert_eq!(Verdict::of(Some("tool calls: 1")), Verdict::Unfinished);
        assert_eq!(Verdict::of(None), Verdict::Unfinished);
    }

    #[test]
    fn sums_decode_time_across_passes() {
        // 100 tokens at 50/s is 2 seconds; 300 at 25/s is 12 more.
        assert!((stats().decode_sec() - 14.0).abs() < 1e-6);
    }

    #[test]
    fn the_report_states_the_outcome_and_the_wall_total() {
        let text = stats().render(std::path::Path::new("r.md"), Style::plain());
        assert!(text.contains("cut by a loop guard"));
        assert!(text.contains("42s total, 14s generating"));
    }

    #[test]
    fn colour_wraps_the_report_in_escapes() {
        let plain = stats().render(std::path::Path::new("r.md"), Style::plain());
        let painted = stats().render(std::path::Path::new("r.md"), Style::ansi());
        assert!(!plain.contains('\u{1b}'));
        assert!(painted.contains("\u{1b}[1;36m"));
    }

    #[test]
    fn renders_a_report_without_panicking() {
        let text = stats().render(std::path::Path::new("repro.md"), Style::plain());
        assert!(text.contains("Generation"));
        assert!(text.contains("guard: draft"));
        assert!(text.contains("Qwen3.8 Flash Next"));
    }

    #[test]
    fn formats_helpers() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KiB");
        assert_eq!(human_time(3723), "1h 02m 03s");
        assert_eq!(human_time(75), "1m 15s");
        assert_eq!(parse_delta("5h7m)"), Some(5 * 3600 + 7 * 60));
        assert_eq!(parse_delta("4m53s)"), Some(4 * 60 + 53));
        assert_eq!(parse_delta("42s)"), Some(42));
        assert_eq!(parse_delta("0s)"), Some(0));
        assert_eq!(parse_delta("nope"), None);
        assert_eq!(
            tool_result_name("Tool result 3 (read):").as_deref(),
            Some("read")
        );
        assert_eq!(tool_result_name("not a tool result"), None);
    }
}
