//! Runs prompts through the ds4 engine in-process and collects its per-layer
//! activation dump.

use std::fmt::{self, Write as _};
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::os::fd::AsRawFd as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use local_inference_engine::{Family, Model, Options, Session, Think};

use super::{Profile, VectorizeError};

/// The activation stream a direction is extracted from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Component {
    /// The residual stream after each layer, what heretic measures; the
    /// direction is applied to the FFN output with `ffn`, as heretic ablates
    /// each layer's MLP output against the residual direction after it.
    #[default]
    Residual,
    /// The dump ds4 calls `ffn_out`, which `build_direction.py` reads: the
    /// FFN block's output on `DeepSeek`, the post-layer residual on GLM and Qwen.
    FfnOut,
    /// Output of each layer's attention projection; applied with `attn`.
    AttnOut,
}

impl Component {
    /// The name `--component` takes.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Residual => "residual",
            Self::FfnOut => "ffn_out",
            Self::AttnOut => "attn_out",
        }
    }

    /// Parses `residual`, `ffn_out` or `attn_out`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "residual" => Some(Self::Residual),
            "ffn_out" => Some(Self::FfnOut),
            "attn_out" => Some(Self::AttnOut),
            _ => None,
        }
    }

    /// The ds4 dump to read for `profile`, and how many hyper-connection
    /// branches of `width` floats it holds per token.
    #[must_use]
    pub fn dump(self, profile: Profile) -> (&'static str, usize) {
        match self {
            Self::Residual => (profile.residual_dump, profile.residual_branches),
            Self::FfnOut => ("ffn_out", 1),
            Self::AttnOut => ("attn_out", 1),
        }
    }
}

impl fmt::Display for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Captures last-prompt-token activations through the linked ds4 engine.
///
/// The model is loaded once, at [`load`](Self::load) or the first capture,
/// and every prompt then runs on the same session: its cache is dropped, the
/// prompt is prefilled in one chunk from position 0, and the engine dumps one
/// row per prompt token per layer. The scratch directory is removed when the
/// `Capture` is dropped.
///
/// The engine reads its dump settings once per process, so a process holds
/// one `Capture` for one [`Component`]; loading a second for another
/// component fails.
#[derive(Debug)]
pub struct Capture {
    model: PathBuf,
    profile: Profile,
    ctx: u32,
    system: String,
    think: bool,
    component: Component,
    work: PathBuf,
    metal: Option<PathBuf>,
    session: Option<Session>,
}

/// The dump the process's settings were fixed to, once loaded.
static DUMPING: Mutex<Option<&'static str>> = Mutex::new(None);

impl Capture {
    /// Prepares captures of `model`.
    #[must_use]
    pub fn new(model: impl AsRef<Path>, profile: Profile) -> Self {
        Self {
            model: model.as_ref().to_path_buf(),
            profile,
            ctx: 512,
            system: "You are a helpful assistant.".to_string(),
            think: false,
            component: Component::Residual,
            work: std::env::temp_dir().join(format!("pt-vectorize-{}", std::process::id())),
            metal: None,
            session: None,
        }
    }

    /// Sets the context size; a prompt that does not fit fails its capture.
    #[must_use]
    pub fn ctx(mut self, ctx: u32) -> Self {
        self.ctx = ctx;
        self
    }

    /// Sets the system prompt each capture runs under.
    #[must_use]
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = system.into();
        self
    }

    /// Captures after an open `<think>` instead of a direct answer.
    #[must_use]
    pub fn think(mut self, think: bool) -> Self {
        self.think = think;
        self
    }

    /// Chooses the activation stream to capture.
    #[must_use]
    pub fn component(mut self, component: Component) -> Self {
        self.component = component;
        self
    }

    /// Compiles the engine's Metal kernels from `dir` instead of the sources
    /// the engine was built from.
    ///
    /// # Errors
    /// Fails when `dir` holds no `flash_attn.metal`.
    pub fn metal_dir(mut self, dir: impl AsRef<Path>) -> Result<Self, VectorizeError> {
        let dir = dir.as_ref();
        if !dir.join("flash_attn.metal").is_file() {
            return Err(VectorizeError::msg(format!(
                "{}: no Metal kernel sources here (no flash_attn.metal)",
                dir.display()
            )));
        }
        self.metal = Some(dir.to_path_buf());
        Ok(self)
    }

    /// The Metal kernel directory the engine will compile from.
    #[must_use]
    pub fn metal_source_dir(&self) -> PathBuf {
        self.metal
            .clone()
            .unwrap_or_else(local_inference_engine::metal::source_dir)
    }

    /// Checks what can be known before the model loads: that this build
    /// linked the engine and that the Metal kernels are where it will look.
    ///
    /// # Errors
    /// Fails on either.
    pub fn preflight(&self) -> Result<(), VectorizeError> {
        if !local_inference_engine::DS4_AVAILABLE {
            return Err(VectorizeError::msg(
                "this pt was built without the ds4 engine (it needs macOS and plank's refs/ds4)",
            ));
        }
        let dir = self.metal_source_dir();
        if cfg!(target_os = "macos") && !dir.join("flash_attn.metal").is_file() {
            return Err(VectorizeError::msg(format!(
                "no Metal kernel sources in {}: pass --metal DIR with the metal/ directory \
                 of the plank/refs/ds4 tree pt was built from, or set DS4_METAL_DIR",
                dir.display()
            )));
        }
        Ok(())
    }

    /// Whether the model is loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.session.is_some()
    }

    /// Loads the model, if it is not loaded yet. The engine reports its load
    /// on stderr.
    ///
    /// The C engine exits the process instead of returning when another
    /// process (a running plank or ds4) holds `/tmp/ds4.lock`.
    ///
    /// # Errors
    /// Fails when this build has no engine, the dump settings were fixed for
    /// another component, or the engine refuses the model.
    pub fn load(&mut self) -> Result<(), VectorizeError> {
        if self.session.is_some() {
            return Ok(());
        }
        self.preflight()?;
        self.fix_dump_settings()?;
        std::fs::create_dir_all(&self.work).map_err(|e| VectorizeError::io(&self.work, e))?;
        let ctx = i32::try_from(self.ctx).unwrap_or(i32::MAX);
        let chunk = self.ctx.max(1024);
        let options = Options::new(&self.model).ctx_size(ctx).prefill_chunk(chunk);
        if Family::of(&self.model) != Family::Ds4 {
            return Err(VectorizeError::msg(format!(
                "{}: activation capture needs a model the ds4 engine runs",
                self.model.display()
            )));
        }
        let model = Model::open(&options).map_err(|e| VectorizeError::msg(e.to_string()))?;
        let session =
            Session::new(&Arc::new(model), ctx).map_err(|e| VectorizeError::msg(e.to_string()))?;
        self.session = Some(session);
        Ok(())
    }

    /// Sets the process-wide variables the engine reads once: the dump
    /// prefix, name and position, the Qwen prefill chunk, and `--metal`.
    fn fix_dump_settings(&self) -> Result<(), VectorizeError> {
        let mut dumping = DUMPING
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (name, _) = self.component.dump(self.profile);
        match *dumping {
            Some(d) if d != name => {
                return Err(VectorizeError::msg(format!(
                    "this process already captures {d}; the engine cannot switch to {name} \
                     without a restart"
                )));
            }
            Some(_) => return Ok(()),
            None => {}
        }
        let chunk = self.ctx.max(1024).to_string();
        // SAFETY: set once, before the engine is opened and before any thread
        // that reads the environment is spawned.
        unsafe {
            std::env::set_var("DS4_METAL_GRAPH_DUMP_PREFIX", self.dump_prefix());
            std::env::set_var("DS4_METAL_GRAPH_DUMP_NAME", name);
            std::env::set_var("DS4_METAL_GRAPH_DUMP_POS", "0");
            std::env::set_var("DS4_QWEN4_PREFILL_CHUNK", chunk);
            if let Some(dir) = &self.metal {
                std::env::set_var("DS4_METAL_DIR", dir);
            }
        }
        *dumping = Some(name);
        Ok(())
    }

    fn dump_prefix(&self) -> PathBuf {
        self.work.join("dump")
    }

    fn dump_path(&self, layer: usize) -> PathBuf {
        let (name, _) = self.component.dump(self.profile);
        self.work.join(format!("dump_{name}-{layer}_pos0.bin"))
    }

    /// Runs `prompt` and returns `layers * width` floats, layer-major.
    ///
    /// # Errors
    /// Fails when the model cannot load, the prefill fails, or the engine does
    /// not write the expected dumps.
    pub fn activations(&mut self, prompt: &str) -> Result<Vec<f32>, VectorizeError> {
        self.activations_with(prompt, &mut |_| {})
    }

    /// Like [`activations`](Self::activations), reporting while the engine
    /// prefills.
    ///
    /// `on_tick` is called as the engine reports prefill progress, with the
    /// time spent so far and a short status, so a caller can keep a live line
    /// moving. The engine's own stderr is diverted during the prefill (it
    /// prints a line per dumped layer) and given back around each call, so
    /// `on_tick` can draw on the terminal.
    ///
    /// # Errors
    /// Fails when the model cannot load, the prefill fails, or the engine does
    /// not write the expected dumps.
    pub fn activations_with(
        &mut self,
        prompt: &str,
        on_tick: &mut dyn FnMut(Tick<'_>),
    ) -> Result<Vec<f32>, VectorizeError> {
        self.load()?;
        for layer in 0..self.profile.layers {
            let _ = std::fs::remove_file(self.dump_path(layer));
        }
        let think = if self.think { Think::High } else { Think::Off };
        let started = Instant::now();
        let log_path = self.work.join("engine.log");
        let Some(session) = self.session.as_mut() else {
            return Err(VectorizeError::msg("the model is not loaded"));
        };
        let tokens = session
            .model()
            .encode_chat(&self.system, prompt, think)
            .map_err(|e| VectorizeError::msg(e.to_string()))?;
        if tokens.len() > usize::try_from(session.ctx()).unwrap_or(usize::MAX) {
            return Err(VectorizeError::msg(format!(
                "a prompt of {} tokens does not fit --ctx {}; raise --ctx or drop the prompt",
                tokens.len(),
                session.ctx()
            )));
        }
        // A shared prefix (the system prompt) would otherwise be reused and
        // the prefill would start past position 0, where nothing is dumped.
        session.invalidate();
        on_tick(Tick {
            elapsed: started.elapsed(),
            ds4_says: "prefill",
        });
        let mut quiet = Diverted::to(&log_path)?;
        let mut status = String::new();
        let result = session.sync_with_progress(&tokens, &mut |event, cur, total| {
            status.clear();
            let what = if event.starts_with("prefill") {
                "prefill"
            } else {
                event
            };
            let _ = write!(status, "{what} {cur}/{total}");
            quiet.pause();
            on_tick(Tick {
                elapsed: started.elapsed(),
                ds4_says: &status,
            });
            quiet.resume();
        });
        let log = quiet.finish();
        if let Err(e) = result {
            let tail: Vec<&str> = log.lines().rev().take(15).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(VectorizeError::msg(format!(
                "ds4 activation capture failed: {e}{}",
                if tail.is_empty() {
                    String::new()
                } else {
                    format!("\n{}", tail.join("\n"))
                }
            )));
        }

        let w = self.profile.width;
        let (_, branches) = self.component.dump(self.profile);
        let mut rows = Vec::with_capacity(self.profile.layers * w);
        for layer in 0..self.profile.layers {
            let path = self.dump_path(layer);
            let bytes = std::fs::read(&path).map_err(|e| {
                VectorizeError::msg(format!(
                    "{}: {e} (this ds4 build or model may not support activation dumps)",
                    path.display()
                ))
            })?;
            let row = last_row(&bytes, w * branches).ok_or_else(|| {
                VectorizeError::msg(format!(
                    "{}: {} bytes is not a whole number of {}-float rows",
                    path.display(),
                    bytes.len(),
                    w * branches
                ))
            })?;
            rows.extend(branch_mean(&row, w));
        }
        Ok(rows)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        // The session (and with it the engine) goes before its scratch files.
        self.session = None;
        let _ = std::fs::remove_dir_all(&self.work);
    }
}

/// A progress report from a running capture.
#[derive(Debug, Clone, Copy)]
pub struct Tick<'a> {
    /// Time since the capture started.
    pub elapsed: Duration,
    /// What the engine is doing, such as `prefill 256/512`.
    pub ds4_says: &'a str,
}

/// File descriptor 2 sent to a log file, restorable around callbacks.
///
/// The engine writes straight to fd 2 from C, so redirecting the descriptor
/// is the only way to keep its per-layer chatter off the terminal.
struct Diverted {
    saved: libc::c_int,
    log: File,
}

impl Diverted {
    fn to(path: &Path) -> Result<Self, VectorizeError> {
        let log = File::options()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| VectorizeError::io(path, e))?;
        // SAFETY: duplicating a descriptor the process owns.
        let saved = unsafe { libc::dup(libc::STDERR_FILENO) };
        if saved < 0 {
            return Err(VectorizeError::io(path, std::io::Error::last_os_error()));
        }
        let mut diverted = Self { saved, log };
        diverted.resume();
        Ok(diverted)
    }

    /// Gives fd 2 back to the terminal.
    fn pause(&mut self) {
        // SAFETY: `saved` is a live duplicate of the original stderr.
        unsafe { libc::dup2(self.saved, libc::STDERR_FILENO) };
    }

    /// Sends fd 2 to the log again.
    fn resume(&mut self) {
        // SAFETY: both descriptors are live.
        unsafe { libc::dup2(self.log.as_raw_fd(), libc::STDERR_FILENO) };
    }

    /// Restores fd 2 for good and returns what was logged, lossily decoded.
    fn finish(mut self) -> String {
        self.pause();
        let mut bytes = Vec::new();
        let _ = self.log.seek(SeekFrom::Start(0));
        let _ = self.log.read_to_end(&mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl Drop for Diverted {
    fn drop(&mut self) {
        self.pause();
        // SAFETY: `saved` is ours and closed only here.
        unsafe { libc::close(self.saved) };
    }
}

/// The mean of the `width`-float branches `row` holds, branch-major as the
/// hyper-connection kernels lay them out (`d + branch * width`). A row of
/// one branch is returned as it is.
fn branch_mean(row: &[f32], width: usize) -> Vec<f32> {
    let branches = row.len() / width;
    if branches <= 1 {
        return row.to_vec();
    }
    #[allow(clippy::cast_precision_loss, reason = "a handful of branches")]
    let inv = 1.0 / branches as f32;
    (0..width)
        .map(|d| (0..branches).map(|b| row[b * width + d]).sum::<f32>() * inv)
        .collect()
}

/// The last `width` little-endian floats of a dump of whole rows.
fn last_row(bytes: &[u8], width: usize) -> Option<Vec<f32>> {
    let row = width * 4;
    if bytes.len() < row || !bytes.len().is_multiple_of(row) {
        return None;
    }
    Some(
        bytes[bytes.len() - row..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyper_connection_branches_are_averaged_per_component() {
        // Two branches of width 3, branch-major.
        let row = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0];
        assert_eq!(branch_mean(&row, 3), [2.0, 3.0, 4.0]);
        assert_eq!(branch_mean(&row[..3], 3), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_residual_is_read_from_each_familys_own_dump() {
        let ds4 = crate::vectorize::Profile::named("deepseek-v4-flash").unwrap();
        let qwen = crate::vectorize::Profile::named("qwen3.8-flash-next").unwrap();
        assert_eq!(Component::Residual.dump(ds4), ("hc_ffn_post", 4));
        assert_eq!(Component::Residual.dump(qwen), ("ffn_out", 1));
        assert_eq!(Component::FfnOut.dump(ds4), ("ffn_out", 1));
    }

    #[test]
    fn the_last_row_of_a_multi_row_dump_is_taken() {
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        assert_eq!(last_row(&bytes, 2), Some(vec![3.0, 4.0]));
        assert_eq!(last_row(&bytes, 3), None);
        assert_eq!(last_row(&bytes[..4], 2), None);
    }

    #[test]
    fn components_round_trip_through_their_names() {
        for c in [Component::Residual, Component::FfnOut, Component::AttnOut] {
            assert_eq!(Component::parse(c.as_str()), Some(c));
        }
        assert_eq!(Component::parse("resid"), None);
    }
}
