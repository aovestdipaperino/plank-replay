//! Parses `plank` repro transcripts and replays their file-mutating tool calls.
//!
//! A repro file is a Markdown report written by `plank` when a session ends.
//! It embeds the exact engine transcript between `----- BEGIN TRANSCRIPT -----`
//! and `----- END TRANSCRIPT -----` markers. Assistant turns inside that
//! transcript contain DSML tool calls; replaying the `write` and `edit` calls in
//! order reconstructs the workspace the session produced.

pub mod browse;
pub mod error;
pub mod parse;
pub mod replay;
pub mod seed;
pub mod stats;

pub use browse::browse;
pub use error::ReplayError;
pub use parse::{Call, Event, Repro, parse_repro};
pub use replay::{Outcome, Replayer, StepReport};
pub use seed::Seed;
pub use stats::{Pass, Stats, Style, Verdict};
