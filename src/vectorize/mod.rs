//! Builds ds4 directional-steering vectors for plank models.
//!
//! A steering vector is a flat little-endian `f32` matrix with one unit-length
//! direction per normal transformer layer. ds4 applies it at runtime as
//! `y = y - scale * d[layer] * dot(d[layer], y)`, so a positive scale removes
//! the direction and a negative scale amplifies it.
//!
//! The direction is extracted from two prompt sets. Each prompt runs once
//! through the ds4 engine (linked in-process from plank's `local-inference-engine` crate)
//! with its activation dump enabled; the last prompt row of every
//! layer is averaged per set, and the direction is `target - control`, made
//! orthogonal to the control mean and normalized per layer. This mirrors
//! `dir-steering/tools/build_direction.py` in the ds4 tree, with target as its
//! good file and control as its bad file.
//!
//! `pt vectorize` passes its `from` prompts as the target and its `to`
//! prompts as the control, so the vector is `from - to` and a positive scale
//! pushes the model towards `to`.
//!
//! ```no_run
//! use plank_tools::vectorize::{Accumulator, Capture, Model, PromptSource};
//!
//! let model = Model::resolve("ds4vision")?;
//! let profile = model.profile()?;
//! let to = PromptSource::parse("succinct.txt").load(None, None)?;
//! let from = PromptSource::parse("mlabonne/harmless_alpaca:train").load(None, None)?;
//! let mut capture = Capture::new(model.path(), profile);
//! let mut acc = Accumulator::new(profile);
//! for (t, f) in to.iter().zip(&from) {
//!     // `from` is the target and `to` the control: positive scale goes to `to`.
//!     acc.add_pair(&capture.activations(f)?, &capture.activations(t)?);
//! }
//! acc.finish(true, false).write_f32("out.f32")?;
//! # Ok::<(), plank_tools::vectorize::VectorizeError>(())
//! ```

mod capture;
mod direction;
mod gguf;
mod hub;
mod model;
mod prompts;
mod store;

use std::backtrace::Backtrace;
use std::fmt;
use std::path::PathBuf;

pub use capture::{Capture, Component, Tick};
pub use direction::{Accumulator, Direction};
pub use hub::{Fetch, HF_KEY_VARS, HubCache};
pub use model::{Model, PROFILES, Profile};
pub use prompts::PromptSource;
pub use store::VectorStore;

/// Anything that stops a steering vector from being built.
#[derive(Debug)]
pub struct VectorizeError {
    kind: ErrorKind,
    backtrace: Backtrace,
}

/// What went wrong, kept private so variants can change freely.
#[derive(Debug)]
enum ErrorKind {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Message(String),
}

impl VectorizeError {
    /// Wraps an I/O failure together with the path that caused it.
    #[must_use]
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::new(ErrorKind::Io {
            path: path.into(),
            source,
        })
    }

    /// Reports a failure described by `message`.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Message(message.into()))
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

impl fmt::Display for VectorizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::Io { path, source } => write!(f, "{}: {source}", path.display()),
            ErrorKind::Message(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for VectorizeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ErrorKind::Io { source, .. } => Some(source),
            ErrorKind::Message(_) => None,
        }
    }
}
