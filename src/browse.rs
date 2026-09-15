//! Full-screen browser over a directory of repro files.
//!
//! [`browse`] lists every `*.md` in the repro folder, newest first, and paints
//! the [`Stats`](crate::stats::Stats) report of the highlighted entry in a side
//! panel. Backspace deletes the selected repro after a confirmation, `u`
//! uploads it to a secret GitHub gist through the `gh` command line tool, and
//! `c` copies its full path to the system clipboard.
//!
//! The terminal is driven with bare ANSI escapes and `stty`; only the clipboard
//! write goes through the `arboard` crate. Raw mode is restored by
//! [`RawMode`]'s destructor on every exit path, panics included.

use std::fmt::Write as _;
use std::io::{Read, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::SystemTime;

use crate::stats::{Stats, Style, human_bytes, human_time, whole_seconds};

/// Escape that switches to the alternate screen and hides the cursor.
const ENTER_SCREEN: &str = "\u{1b}[?1049h\u{1b}[?25l";
/// Escape that restores the primary screen and the cursor.
const LEAVE_SCREEN: &str = "\u{1b}[?25h\u{1b}[?1049l";
/// Widest the entry list is allowed to grow.
const LIST_MAX: usize = 38;
/// Rows the detail panel jumps by when scrolled.
const PANEL_STEP: usize = 8;
/// Lines of the prompt the panel headline shows before eliding the rest.
const HEADLINE_PROMPT: usize = 10;
/// Column the headline wraps the prompt at.
const HEADLINE_WIDTH: usize = 72;
/// Fallback terminal size when `stty size` cannot be read.
const FALLBACK_SIZE: (usize, usize) = (24, 100);

/// Restores the terminal settings captured at construction.
///
/// Dropping this value leaves raw mode and the alternate screen, whether the
/// browser returned normally or unwound through a panic.
#[derive(Debug)]
struct RawMode {
    /// The `stty -g` snapshot taken before raw mode was entered.
    saved: Option<String>,
}

impl RawMode {
    /// Puts the terminal into raw mode, remembering how to undo it.
    ///
    /// # Errors
    ///
    /// Returns a message when `stty` is missing or refuses the terminal.
    fn enter() -> Result<Self, String> {
        let saved = stty(&["-g"]).map(|state| state.trim().to_string());
        stty(&["raw", "-echo"]).ok_or("cannot put the terminal into raw mode")?;
        let mut out = std::io::stdout();
        let _ = out.write_all(ENTER_SCREEN.as_bytes());
        let _ = out.flush();
        Ok(Self { saved })
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        let mut out = std::io::stdout();
        let _ = out.write_all(LEAVE_SCREEN.as_bytes());
        let _ = out.flush();
        match &self.saved {
            Some(state) => drop(stty(&[state.as_str()])),
            None => drop(stty(&["sane"])),
        }
    }
}

/// Runs `stty` against the controlling terminal, returning its output.
fn stty(args: &[&str]) -> Option<String> {
    let output = Command::new("stty")
        .args(args)
        .stdin(Stdio::inherit())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Reads the terminal size as `(rows, cols)`, falling back to a sane default.
fn terminal_size() -> (usize, usize) {
    let Some(text) = stty(&["size"]) else {
        return FALLBACK_SIZE;
    };
    let mut parts = text.split_whitespace().filter_map(|n| n.parse().ok());
    match (parts.next(), parts.next()) {
        (Some(rows), Some(cols)) if rows > 4 && cols > 20 => (rows, cols),
        _ => FALLBACK_SIZE,
    }
}

/// One repro file in the browsed directory.
#[derive(Debug)]
struct Entry {
    /// Full path to the repro.
    path: PathBuf,
    /// File name shown in the list.
    name: String,
    /// Size on disk, for the list column.
    size: usize,
    /// Modification time, used only to order the list.
    modified: SystemTime,
}

/// What the status bar is currently asking or saying.
#[derive(Debug)]
enum Mode {
    /// Ordinary navigation; the key legend is shown.
    Normal,
    /// Backspace was pressed and the deletion awaits a `y`.
    Confirm,
    /// A one-shot notice, cleared by the next keystroke.
    Notice(String),
}

/// The browser's whole state between keystrokes.
#[derive(Debug)]
struct Browser {
    /// Directory being listed.
    dir: PathBuf,
    /// Entries, newest first.
    entries: Vec<Entry>,
    /// Index of the highlighted entry.
    cursor: usize,
    /// First entry row drawn, for list scrolling.
    top: usize,
    /// First report row drawn, for panel scrolling.
    panel: usize,
    /// Rendered report of the highlighted entry, and which entry it belongs to.
    cached: Option<(PathBuf, Vec<String>)>,
    /// Status bar state.
    mode: Mode,
    /// Colours the report and the chrome are painted with.
    style: Style,
}

/// Browses `dir`, painting each repro's `--stats` report beside the list.
///
/// Returns once the user quits with `q`. `colour` switches the ANSI palette off
/// for terminals that asked for no colour.
///
/// # Errors
///
/// Returns a message when the directory cannot be read or the terminal cannot
/// be put into raw mode.
///
/// # Examples
///
/// ```no_run
/// plank_replay::browse::browse(std::path::Path::new("/tmp/repro"), true)?;
/// # Ok::<(), String>(())
/// ```
pub fn browse(dir: &Path, colour: bool) -> Result<(), String> {
    let entries = read_entries(dir)?;
    if entries.is_empty() {
        return Err(format!("{}: no repro files to browse", dir.display()));
    }

    let mut browser = Browser {
        dir: dir.to_path_buf(),
        entries,
        cursor: 0,
        top: 0,
        panel: 0,
        cached: None,
        mode: Mode::Normal,
        style: if colour {
            Style::ansi()
        } else {
            Style::plain()
        },
    };

    let _raw = RawMode::enter()?;
    let mut stdin = std::io::stdin().lock();
    let mut byte = [0u8; 1];
    loop {
        browser.draw();
        if stdin.read(&mut byte).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let key = match byte[0] {
            0x1b => read_escape(&mut stdin),
            other => Key::Char(other),
        };
        if !browser.handle(key) {
            break;
        }
    }
    Ok(())
}

/// A keystroke the browser understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Key {
    /// A printable byte, or a control byte such as backspace.
    Char(u8),
    /// Cursor up.
    Up,
    /// Cursor down.
    Down,
    /// Page up.
    PageUp,
    /// Page down.
    PageDown,
    /// A sequence with no meaning here.
    Other,
}

/// Decodes the remainder of a CSI sequence after the leading escape.
fn read_escape(stdin: &mut impl Read) -> Key {
    let mut byte = [0u8; 1];
    if stdin.read(&mut byte).unwrap_or(0) == 0 || byte[0] != b'[' {
        return Key::Other;
    }
    if stdin.read(&mut byte).unwrap_or(0) == 0 {
        return Key::Other;
    }
    match byte[0] {
        b'A' => Key::Up,
        b'B' => Key::Down,
        b'5' | b'6' => {
            let page = if byte[0] == b'5' {
                Key::PageUp
            } else {
                Key::PageDown
            };
            // Swallow the terminating `~`.
            let _ = stdin.read(&mut byte);
            page
        }
        _ => Key::Other,
    }
}

/// Rows the entry list can draw into on the current terminal.
fn list_rows() -> usize {
    terminal_size().0.saturating_sub(3).max(1)
}

/// Lists the `*.md` files in `dir`, newest first.
fn read_entries(dir: &Path) -> Result<Vec<Entry>, String> {
    let read = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut entries: Vec<Entry> = read
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            Some(Entry {
                name: e.file_name().to_string_lossy().into_owned(),
                size: usize::try_from(meta.len()).unwrap_or(usize::MAX),
                modified: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                path: e.path(),
            })
        })
        .collect();
    entries.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(entries)
}

impl Browser {
    /// Handles one keystroke, returning `false` when the browser should quit.
    fn handle(&mut self, key: Key) -> bool {
        if matches!(self.mode, Mode::Confirm) {
            self.mode = Mode::Normal;
            if key == Key::Char(b'y') || key == Key::Char(b'Y') {
                self.delete_selected();
            } else {
                self.mode = Mode::Notice("deletion cancelled".into());
            }
            return !self.entries.is_empty();
        }
        self.mode = Mode::Normal;

        match key {
            Key::Char(b'q') => return false,
            Key::Up | Key::Char(b'k') => self.move_by(-1),
            Key::Down | Key::Char(b'j') => self.move_by(1),
            Key::PageUp => self.move_by(-(i64::try_from(list_rows()).unwrap_or(10))),
            Key::PageDown => self.move_by(i64::try_from(list_rows()).unwrap_or(10)),
            Key::Char(b'g') => self.move_to(0),
            Key::Char(b'G') => self.move_to(self.entries.len().saturating_sub(1)),
            Key::Char(b'K') => self.panel = self.panel.saturating_sub(PANEL_STEP),
            Key::Char(b'J') => self.panel = self.panel.saturating_add(PANEL_STEP),
            Key::Char(b'r') => self.reload(),
            Key::Char(0x7f | 0x08) => self.mode = Mode::Confirm,
            Key::Char(b'u' | b'U') => self.upload_selected(),
            Key::Char(b'c') => self.copy_selected(),
            Key::Char(_) | Key::Other => {}
        }
        !self.entries.is_empty()
    }

    /// Moves the selection by `delta`, clamped to the list.
    fn move_by(&mut self, delta: i64) {
        let last = self.entries.len().saturating_sub(1);
        let target = i64::try_from(self.cursor)
            .unwrap_or(0)
            .saturating_add(delta);
        let target = usize::try_from(target.max(0)).unwrap_or(0);
        self.move_to(target.min(last));
    }

    /// Selects `index`, resetting the panel scroll.
    fn move_to(&mut self, index: usize) {
        if index != self.cursor {
            self.panel = 0;
        }
        self.cursor = index;
    }

    /// Re-reads the directory, keeping the selection in range.
    fn reload(&mut self) {
        if let Ok(entries) = read_entries(&self.dir) {
            self.entries = entries;
            self.cursor = self.cursor.min(self.entries.len().saturating_sub(1));
            self.cached = None;
            self.panel = 0;
        }
    }

    /// Deletes the highlighted repro and drops it from the list.
    fn delete_selected(&mut self) {
        let Some(entry) = self.entries.get(self.cursor) else {
            return;
        };
        let (path, name) = (entry.path.clone(), entry.name.clone());
        self.mode = match std::fs::remove_file(&path) {
            Ok(()) => {
                self.entries.remove(self.cursor);
                self.cursor = self.cursor.min(self.entries.len().saturating_sub(1));
                self.cached = None;
                self.panel = 0;
                Mode::Notice(format!("deleted {name}"))
            }
            Err(e) => Mode::Notice(format!("cannot delete {name}: {e}")),
        };
    }

    /// Uploads the highlighted repro to a secret gist with `gh`.
    fn upload_selected(&mut self) {
        let Some(entry) = self.entries.get(self.cursor) else {
            return;
        };
        let (path, name) = (entry.path.clone(), entry.name.clone());
        self.note(&format!("uploading {name}..."));

        let output = Command::new("gh")
            .arg("gist")
            .arg("create")
            .arg(&path)
            .arg("--desc")
            .arg(format!("plank repro {name}"))
            .stdin(Stdio::null())
            .output();

        self.mode = match output {
            Ok(done) if done.status.success() => {
                let url = String::from_utf8_lossy(&done.stdout);
                Mode::Notice(format!(
                    "secret gist: {}",
                    url.lines().last().unwrap_or("").trim()
                ))
            }
            Ok(done) => {
                let err = String::from_utf8_lossy(&done.stderr);
                Mode::Notice(format!(
                    "gh failed: {}",
                    err.lines().next().unwrap_or("unknown error").trim()
                ))
            }
            Err(e) => Mode::Notice(format!("cannot run gh: {e}")),
        };
    }

    /// Copies the highlighted repro's full path to the system clipboard.
    fn copy_selected(&mut self) {
        let Some(entry) = self.entries.get(self.cursor) else {
            return;
        };
        let path = entry.path.display().to_string();
        self.mode = match arboard::Clipboard::new() {
            Ok(mut clipboard) => match clipboard.set_text(path.clone()) {
                Ok(()) => Mode::Notice(format!("copied {path}")),
                Err(e) => Mode::Notice(format!("cannot copy: {e}")),
            },
            Err(e) => Mode::Notice(format!("cannot copy: {e}")),
        };
    }

    /// Paints a transient status line without disturbing the rest of the frame.
    fn note(&mut self, text: &str) {
        let (rows, cols) = terminal_size();
        let mut out = std::io::stdout();
        let _ = write!(
            out,
            "\u{1b}[{rows};1H\u{1b}[K{}{}{}",
            self.style.dim,
            clip(text, cols),
            self.style.off
        );
        let _ = out.flush();
    }

    /// The report lines of the highlighted entry, rendering them on first use.
    fn report(&mut self) -> &[String] {
        let Some(entry) = self.entries.get(self.cursor) else {
            return &[];
        };
        let path = entry.path.clone();
        if self.cached.as_ref().is_none_or(|(p, _)| *p != path) {
            let lines = match std::fs::read_to_string(&path) {
                Ok(text) => {
                    let stats = Stats::of(&text);
                    let mut lines = headline(&stats, self.style);
                    lines.extend(stats.render(&path, self.style).lines().map(str::to_string));
                    lines
                }
                Err(e) => vec![format!("  cannot read this repro: {e}")],
            };
            self.cached = Some((path, lines));
        }
        self.cached
            .as_ref()
            .map_or(&[], |(_, lines)| lines.as_slice())
    }

    /// Repaints the whole screen.
    fn handle_scroll(&mut self, rows: usize) {
        if self.cursor < self.top {
            self.top = self.cursor;
        } else if self.cursor >= self.top + rows {
            self.top = self.cursor + 1 - rows;
        }
    }

    /// Draws the list, the detail panel, and the status bar.
    fn draw(&mut self) {
        let (rows, cols) = terminal_size();
        let body = rows.saturating_sub(2).max(1);
        let list_width = LIST_MAX.min(cols / 3).max(12);
        let panel_width = cols.saturating_sub(list_width + 3);
        self.handle_scroll(body);

        let style = self.style;
        let header = format!(
            "{}{}{}  {}{} repro(s){}",
            style.head,
            clip(&self.dir.display().to_string(), cols.saturating_sub(20)),
            style.off,
            style.dim,
            self.entries.len(),
            style.off,
        );

        let (top, cursor) = (self.top, self.cursor);
        let list: Vec<String> = (0..body)
            .map(|row| match self.entries.get(top + row) {
                Some(entry) => self.list_cell(entry, top + row == cursor, list_width),
                None => " ".repeat(list_width),
            })
            .collect();

        let panel_top = self.panel;
        let report = self.report();
        let panel: Vec<String> = (0..body)
            .map(|row| {
                report
                    .get(panel_top + row)
                    .map_or_else(String::new, |line| clip(line, panel_width))
            })
            .collect();

        let mut frame = String::from("\u{1b}[H\u{1b}[K");
        let _ = writeln!(frame, "{header}\r");
        for (left, right) in list.iter().zip(&panel) {
            let _ = writeln!(
                frame,
                "{left} {}\u{2502}{} {right}\u{1b}[K\r",
                style.dim, style.off
            );
        }
        let _ = write!(frame, "\u{1b}[K{}", self.status(cols));
        let _ = write!(frame, "\u{1b}[J");

        let mut out = std::io::stdout();
        let _ = out.write_all(frame.as_bytes());
        let _ = out.flush();
    }

    /// Formats one entry row, padded to exactly `width` visible columns.
    fn list_cell(&self, entry: &Entry, selected: bool, width: usize) -> String {
        let size = human_bytes(entry.size);
        let room = width.saturating_sub(size.len() + 3);
        let name: String = entry.name.chars().take(room).collect();
        let text = format!(
            " {name}{} {size} ",
            " ".repeat(width.saturating_sub(name.chars().count() + size.len() + 3)),
        );
        if selected {
            format!("\u{1b}[7m{text}{}", self.style.off)
        } else {
            text
        }
    }

    /// The bottom line: the key legend, a confirmation, or a notice.
    fn status(&self, cols: usize) -> String {
        let style = self.style;
        match &self.mode {
            Mode::Confirm => {
                let name = self
                    .entries
                    .get(self.cursor)
                    .map_or("", |e| e.name.as_str());
                format!("{}delete {name} permanently? [y/N]{}", style.bad, style.off)
            }
            Mode::Notice(text) => {
                format!("{}{}{}", style.value, clip(text, cols), style.off)
            }
            Mode::Normal => format!(
                "{}\u{2191}\u{2193}/jk move  J/K scroll report  \u{232b} delete  u gist  c copy path  r reload  q quit{}",
                style.dim, style.off
            ),
        }
    }
}

/// Formats when the session ran, from the transcript span or the header date.
///
/// A span inside one day is shortened to `2026-09-15  09:47:23 -> 10:00:33`;
/// one crossing midnight prints both timestamps in full.
fn when(stats: &Stats) -> String {
    let Some((first, last)) = &stats.span else {
        return stats
            .meta
            .get("date")
            .map_or("-", String::as_str)
            .to_string();
    };
    match (first.split_once(' '), last.split_once(' ')) {
        (Some((day, start)), Some((end_day, end))) if day == end_day => {
            format!("{day}  {start} -> {end}")
        }
        _ => format!("{first} -> {last}"),
    }
}

/// Builds the block the panel opens with: prompt, model, skills, when, and time.
///
/// These answer "what was asked, of what, with skills or not, when, and for how
/// long" at a glance, so they lead the panel; the full `--stats` report follows
/// underneath.
fn headline(stats: &Stats, style: Style) -> Vec<String> {
    let mut lines = Vec::with_capacity(HEADLINE_PROMPT + 6);
    let meta = |k: &str| stats.meta.get(k).map_or("-", String::as_str);

    let model = meta("name");
    let time = stats
        .elapsed_sec
        .map_or_else(|| "unrecorded".to_string(), human_time);
    let generating = stats.decode_sec();

    lines.push(format!("{}{}{}", style.head, "PROMPT", style.off));
    match &stats.prompt {
        Some(prompt) => {
            let wrapped = wrap(prompt, HEADLINE_WIDTH);
            for line in wrapped.iter().take(HEADLINE_PROMPT) {
                lines.push(format!("  {}{}{}", style.quote, line, style.off));
            }
            if wrapped.len() > HEADLINE_PROMPT {
                lines.push(format!(
                    "  {}... {} more line(s){}",
                    style.dim,
                    wrapped.len() - HEADLINE_PROMPT,
                    style.off,
                ));
            }
        }
        None => lines.push(format!("  {}(no prompt recorded){}", style.dim, style.off)),
    }

    lines.push(String::new());
    lines.push(format!(
        "{}MODEL{}   {}{}{}",
        style.head, style.off, style.value, model, style.off,
    ));
    lines.push(format!(
        "  {}family{} {}   {}dialect{} {}   {}think{} {}",
        style.dim,
        style.off,
        meta("family"),
        style.dim,
        style.off,
        meta("tool dialect"),
        style.dim,
        style.off,
        meta("think mode"),
    ));

    lines.push(String::new());
    let invoked = stats.tools.get("skill").copied().unwrap_or(0);
    let (colour, skills) = if stats.skills {
        (
            style.good,
            match invoked {
                0 => "enabled, never invoked".to_string(),
                1 => "enabled, 1 invocation".to_string(),
                n => format!("enabled, {n} invocations"),
            },
        )
    } else {
        (style.dim, "disabled".to_string())
    };
    lines.push(format!(
        "{}SKILLS{}  {colour}{skills}{}",
        style.head, style.off, style.off,
    ));

    lines.push(String::new());
    lines.push(format!(
        "{}WHEN{}    {}{}{}",
        style.head,
        style.off,
        style.value,
        when(stats),
        style.off,
    ));

    lines.push(String::new());
    lines.push(format!(
        "{}TIME{}    {}{}{}   {}generating{} {}",
        style.head,
        style.off,
        style.value,
        time,
        style.off,
        style.dim,
        style.off,
        if generating > 0.0 {
            human_time(whole_seconds(generating))
        } else {
            "-".to_string()
        },
    ));
    lines.push(String::new());
    lines
}

/// Wraps `text` to `width` columns on whitespace, keeping its own line breaks.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for paragraph in text.lines() {
        if paragraph.trim().is_empty() {
            out.push(String::new());
            continue;
        }
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            let extra = usize::from(!line.is_empty()) + word.chars().count();
            if !line.is_empty() && line.chars().count() + extra > width {
                out.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        if !line.is_empty() {
            out.push(line);
        }
    }
    out
}

/// Truncates `line` to `width` visible columns, skipping ANSI escapes.
///
/// Escape sequences are copied through untouched and cost no columns, so a
/// coloured report line keeps its colours after being clipped.
fn clip(line: &str, width: usize) -> String {
    let mut out = String::with_capacity(line.len().min(width * 2));
    let mut used = 0usize;
    let mut chars = line.chars();
    let mut coloured = false;
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            let start = out.len();
            out.push(ch);
            for esc in chars.by_ref() {
                out.push(esc);
                if esc.is_ascii_alphabetic() {
                    break;
                }
            }
            // A reset turns colour back off, so nothing needs closing after it.
            coloured = !out[start..].ends_with("[0m");
            continue;
        }
        if used >= width {
            return if coloured { out + "\u{1b}[0m" } else { out };
        }
        out.push(ch);
        used += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Browser, Mode, Style, clip};
    use std::path::PathBuf;

    #[test]
    fn status_legend_lists_copy_command() {
        let browser = Browser {
            dir: PathBuf::from("/tmp"),
            entries: vec![],
            cursor: 0,
            top: 0,
            panel: 0,
            cached: None,
            mode: Mode::Normal,
            style: Style::plain(),
        };
        assert!(browser.status(100).contains("c copy path"));
    }

    #[test]
    fn clip_counts_only_visible_characters() {
        assert_eq!(clip("hello", 3), "hel");
        assert_eq!(clip("hello", 99), "hello");
    }

    #[test]
    fn clip_keeps_escapes_free_of_charge() {
        let line = "\u{1b}[1mhello\u{1b}[0m world";
        assert_eq!(clip(line, 5), "\u{1b}[1mhello\u{1b}[0m");
    }

    #[test]
    fn clip_resets_colour_when_it_truncates() {
        assert!(clip("\u{1b}[31mabcdef", 3).ends_with("\u{1b}[0m"));
    }
}
