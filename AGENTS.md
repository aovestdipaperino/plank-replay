# AGENTS.md

This file provides guidance to the agent when working with code in this repository.

## Build, lint, and test

- `cargo test` — runs the unit tests in `src/`, the `pt` binary tests in `src/cli/`, plus the doctests in `src/parse.rs`, `src/browse.rs`, and `src/vectorize/mod.rs`. Currently 97 library tests, 8 binary tests and 3 doctests pass.
- `cargo clippy --all-targets` — the lint command. The crate enables `clippy::pedantic` and `clippy::perf` as warnings, plus `missing_debug_implementations`, `unsafe_op_in_unsafe_fn`, and `unused_lifetimes`. Use `--all-targets` so the binary and tests are checked too.
- `cargo fmt --check` — formatting check; the code is formatted.
- `cargo build` — normal build. Dependencies are `arboard` (clipboard write in `pt browse`) and `serde_json` (`pt vectorize`).
- `cargo install --path .` — installs the binary, but see the sandbox gotcha below.

There is no CI configuration, Makefile, or scripted test harness in the repo; the commands above are the full verification surface.

## Architecture

`plank-tools` is a single Rust crate (edition 2024, library `plank_tools`) with two dependencies (`arboard`, `serde_json`). Its binary is `pt`, a multi-command front end over plank repro transcripts: `pt replay` replays their file-mutating tool calls into a fresh directory, `pt stats` summarises a session, `pt browse` is a TUI over the repro folder, and `pt vectorize` builds ds4 directional-steering vectors for plank models. More commands are expected.

The pipeline is: repro text → locate transcript markers → split into role turns → decode DSML calls in assistant turns → scan user turns for complete reads → ordered event stream → `Replayer` → output directory.

- `src/lib.rs` is the crate root and public API re-exports.
- `src/main.rs` is the `pt` dispatcher: it picks a command from `cli::COMMANDS`, handles `help`/`--version`, and maps `CliError` to exit codes (2 for usage errors).
- `src/cli/` holds one module per subcommand (`replay.rs`, `stats.rs`, `browse.rs`, `vectorize.rs`), each exporting `USAGE` and `run(&[String])` and owning its own argument parsing. Adding a command means adding a module and one `Command` entry in `src/cli/mod.rs`; the top-level help is generated from that table.
- `src/parse.rs` extracts the transcript and decodes DSML tool calls into an ordered `Vec<Event>` of `Call` and `Seed` events; it touches no filesystem.
- `src/replay.rs` applies events to an output root, handling `write`, `edit`, optional `bash`, path sandboxing, and per-step `Outcome` reporting.
- `src/seed.rs` recovers pre-existing file contents from complete `read` tool results.
- `src/stats.rs` builds the read-only `pt stats` report; it shares the parser but never writes files.
- `src/browse.rs` is the full-screen TUI over a repro directory, driven with ANSI escapes and `stty`.
- `src/vectorize/` builds steering vectors: `model.rs` resolves a plank engine name to its GGUF and steering `Profile`, `gguf.rs` reads header metadata, `prompts.rs` parses local or Hugging Face prompt sets, `hub.rs` fetches from Hugging Face with an on-disk cache and 429 backoff (the HTTP call is injected so tests run offline), `capture.rs` runs ds4 per prompt and reads its activation dumps, `direction.rs` does the mean-difference math, `store.rs` keeps named vectors in `~/.plank/models/vectors.json` (a JSON array of `{model, vectors: [{name, value}]}`, `model` being the GGUF file name and `value` base64 of the little-endian f32 matrix; unknown fields are preserved and writes are atomic). It has its own `VectorizeError`.
- `src/error.rs` defines `ReplayError`, a struct wrapping a private kind enum and a captured backtrace.
- `docs/DESIGN.md` documents the transcript format, edit semantics, read recovery, and stats decisions.

Parsing and replaying are deliberately separate so the decoder is testable against string fixtures without filesystem setup. The event stream is ordered rather than grouped by kind, because a recovered file must be restored at the point in the session where the model first saw it.

## Setup and environment

- Rust toolchain supporting edition 2024 is required.
- `HOME` — used to find the default repro directory (`~/.plank/repro`) when no path is given.
- `HF_API_KEY` — Hugging Face API key sent by `pt vectorize` when fetching datasets, for gated or private ones. `HF_TOKEN` is the fallback; `HF_API_KEY` wins when both are set.
- `NO_COLOR` — disables ANSI colour in `pt stats` and `pt browse` output.
- `RUST_BACKTRACE` — controls whether `ReplayError::backtrace()` captures a real backtrace.
- `gh` CLI — required only for the `u` upload action in `pt browse`; it must be installed and authenticated. The gist is created as secret.
- `stty` — required by `pt browse` for raw mode and terminal size; the TUI needs a real terminal.
- `local-inference-engine` — plank's crate (built here without its Gemma backend) at `../plank/crates/local-inference-engine`, a path dependency: it compiles the ds4 C engine from `../plank/refs/ds4` (or `DS4_SRC`) and links it into `pt`. `Capture` loads the model once and runs each prompt as `invalidate` + `sync` on one session. The engine reads `DS4_METAL_GRAPH_DUMP_*` once per process, so `Capture::load` sets them before opening the model and a process captures one component only; `invalidate` before each prompt is what makes the prefill start at position 0, where the dump is written. Its fd-2 chatter (a line per dumped layer) is diverted to a log during each prefill and given back around every progress tick. Metal kernels: `--metal`, `$DS4_METAL_DIR`, the built-against `refs/ds4/metal`, or `../share/plank/metal`. Holds `/tmp/ds4.lock` while loaded.
- `curl` — used by `pt vectorize` to fetch Hugging Face prompt sources; the key from `HF_API_KEY` (else `HF_TOKEN`) is forwarded on stdin, never argv.
- `arboard` — the crate behind the `c` copy action in `pt browse`; no external CLI tool is needed for the clipboard.

## Gotchas and workflow quirks

- **DSML dialects**: the parser handles both the legacy `dsml` (`tool_calls`, no space after `<｜DSML｜`) and current `dsml41` (`calls`, with a space). It never matches a fixed tag string; it strips the shared prefix and tolerates the optional space. Both spellings appear across a user's repro directory.
- **Only assistant turns are decoded**: DSML examples in the system prompt are ignored. A naive scan would replay plank's own documentation as if the model had requested it.
- **Replay scope**: only `write` and `edit` are replayed. `bash` is skipped unless `pt replay --run-bash` is passed; read-only tools are skipped. `--stop-on-error` aborts at the first failure; otherwise failures are recorded as `Outcome::Failed` and the replay continues.
- **Edit fidelity**: a plain `old` must match exactly once — zero or multiple matches is a failure. The anchored `[upto]` form splits into head/tail anchors, each must be unique, and an empty tail is rejected. Ambiguity is reproduced rather than resolved to the first match.
- **Seeding**: files are recovered only from complete `read` results (`lines 1-N of N`). Partial reads are deliberately ignored because a chunk can contain an edit anchor. Seeds never overwrite a path the transcript itself produced; `--no-seed` disables recovery entirely.
- **Path containment**: absolute paths are re-rooted under `_abs/`; any `..` component rejects the call. Transcript paths are untrusted input describing a machine that may not be this one.
- **Output directory**: must be empty unless `--force` is passed. Default output is `./replay-<repro stem>`. The exit status is non-zero when any call failed, so a replay can gate a script.
- **`pt stats` tolerates broken transcripts**: a truncated session can carry an unterminated tool call, which is a hard error for the replayer but only a note for the report. `Stats::of` keeps the decode failure as `parse_note` and reports what the header and transcript still hold.
- **`pt browse` deletion is permanent**: backspace deletes the selected repro after a `y/N` confirmation and does not go through trash. The terminal is restored on every exit path via `RawMode`'s destructor.
- **`cargo install --path .` can fail inside a sandboxed shell**: the macOS Seatbelt profile denies writes to `~/.cargo` with `EPERM` ("Operation not permitted"), even though the directory is writable by the owner. Build, test, and lint are unaffected because they only write to `target/`. Workaround: run the install from a normal terminal, or use `cargo install --path . --root /tmp/plank-install` to install into a writable location. See `BUG-REPRO.md` for the full analysis.
- **`pt vectorize` must stay bit-identical to ds4's `dir-steering/tools/build_direction.py` run with `--good-file <from> --bad-file <to>`**: `from` is the target and `to` the control, so the vector is `from - to`, orthogonalized against the normalized `to` mean, and a positive runtime scale moves the model towards `to`. Swapping roles is not a sign flip: ds4's projection edit is invariant to the vector's sign, and only the choice of control mean changes behaviour. Sums are in `f64`, the direction is normalized, then orthogonalized and renormalized. Pairing truncates to the shorter set. Shapes come from `PROFILES` keyed by GGUF `general.architecture` (a missing key means `deepseek4`, as in ds4); GGUF `block_count` is not used because it disagrees with the steering shape for Qwen.
- **Hugging Face cache**: rows live in `~/.cache/plank-tools/hf/datasets/<owner>/<name>/rows/<config>/<split>.jsonl` with a `.meta.json` total; pages are appended as they arrive and a damaged tail is truncated on read, so interrupted fetches resume. Cache writes are best effort and never fail a load. `pt vectorize` sizes both sides first (`PromptSource::size`: one `length=1` request per split, cached in the `.meta.json`) and loads each only up to the smaller size or `--limit`; datasets-server rate-limits unpaged bulk reads with 429.
- **`pt vectorize` flags**: `-n/--name` stores into `vectors.json`, `-o/--out` writes a raw file; at least one is required and both may be given. `--limit` is `-l` (it was `-n` before names existed). `--force` replaces both an existing file and an existing model/name pair; the pair check runs before any capture.
- **Tests use per-test temp directories** keyed by process and thread id, so they neither collide nor leak. The real regression suite is the repro archive in `~/.plank/repro/` — running every repro and comparing file/failure counts against the previous run catches format drift that fixtures would not.
