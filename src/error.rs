//! Error type shared by parsing and replay.

use std::backtrace::Backtrace;
use std::fmt;
use std::path::PathBuf;

/// Anything that stops a repro from being parsed or replayed.
#[derive(Debug)]
pub struct ReplayError {
    kind: ErrorKind,
    backtrace: Backtrace,
}

/// What went wrong, without exposing the variants as the public error type.
#[derive(Debug)]
enum ErrorKind {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    NoTranscript,
    Malformed {
        line: usize,
        detail: String,
    },
    MissingParameter {
        call: String,
        name: &'static str,
    },
    EditNotFound {
        path: PathBuf,
    },
    NoBaseContents {
        path: PathBuf,
    },
    EditAmbiguous {
        path: PathBuf,
        count: usize,
    },
    EscapingPath {
        path: PathBuf,
    },
}

impl ReplayError {
    /// Wraps an I/O failure together with the path that caused it.
    #[must_use]
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::new(ErrorKind::Io {
            path: path.into(),
            source,
        })
    }

    /// Reports that the file has no `BEGIN TRANSCRIPT` marker.
    #[must_use]
    pub fn no_transcript() -> Self {
        Self::new(ErrorKind::NoTranscript)
    }

    /// Reports a structurally broken DSML block at a transcript line.
    #[must_use]
    pub fn malformed(line: usize, detail: impl Into<String>) -> Self {
        Self::new(ErrorKind::Malformed {
            line,
            detail: detail.into(),
        })
    }

    /// Reports a tool call that lacks a parameter the replayer needs.
    #[must_use]
    pub fn missing_parameter(call: impl Into<String>, name: &'static str) -> Self {
        Self::new(ErrorKind::MissingParameter {
            call: call.into(),
            name,
        })
    }

    /// Reports an `edit` whose `old` text is absent from the current file.
    #[must_use]
    pub fn edit_not_found(path: impl Into<PathBuf>) -> Self {
        Self::new(ErrorKind::EditNotFound { path: path.into() })
    }

    /// Reports an `edit` against a file the transcript never produced.
    #[must_use]
    pub fn no_base_contents(path: impl Into<PathBuf>) -> Self {
        Self::new(ErrorKind::NoBaseContents { path: path.into() })
    }

    /// Reports an `edit` whose `old` text matches more than once.
    #[must_use]
    pub fn edit_ambiguous(path: impl Into<PathBuf>, count: usize) -> Self {
        Self::new(ErrorKind::EditAmbiguous {
            path: path.into(),
            count,
        })
    }

    /// Reports a tool call whose path would land outside the output root.
    pub fn escaping_path(path: impl Into<PathBuf>) -> Self {
        Self::new(ErrorKind::EscapingPath { path: path.into() })
    }

    fn new(kind: ErrorKind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns the captured backtrace, empty unless `RUST_BACKTRACE` is set.
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::Io { path, source } => write!(f, "{}: {source}", path.display()),
            ErrorKind::NoTranscript => {
                f.write_str("no `----- BEGIN TRANSCRIPT -----` marker in this file")
            }
            ErrorKind::Malformed { line, detail } => {
                write!(f, "malformed tool call at transcript line {line}: {detail}")
            }
            ErrorKind::MissingParameter { call, name } => {
                write!(f, "`{call}` call is missing the `{name}` parameter")
            }
            ErrorKind::EditNotFound { path } => {
                write!(f, "{}: `old` text not found", path.display())
            }
            ErrorKind::NoBaseContents { path } => write!(
                f,
                "{}: no base contents (the session edited a file it never wrote, \
                 and never read in full, so the repro does not carry its text)",
                path.display()
            ),
            ErrorKind::EditAmbiguous { path, count } => {
                write!(
                    f,
                    "{}: `old` text matched {count} times, expected once",
                    path.display()
                )
            }
            ErrorKind::EscapingPath { path } => {
                write!(f, "{}: path escapes the output directory", path.display())
            }
        }
    }
}

impl std::error::Error for ReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ErrorKind::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
