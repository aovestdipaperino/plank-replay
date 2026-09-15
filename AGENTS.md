# AGENTS.md

This file provides guidance to the agent when working with code in this repository.

## Build, lint, and test

- `cargo test` — runs the unit tests in `src/` plus the doctests in `src/lib.rs`, `src/parse.rs`, and `src/browse.rs`. 56 unit tests + 2 doctests currently pass.
- `cargo clippy --all-targets` — the lint command. The crate enables `clippy::pedantic` and `clippy::perf` as warnings, plus `missing_debug_implementations`, `unsafe_op_in_unsafe_fn`, and `unused_lifetimes`. Run with `--all-targets` so the binary and tests are checked too.
- `cargo fmt --check` — formatting check; the code is formatted.
- `cargo build` / `cargo install --path .` — normal build/install. The only dependency is `arboard`, used for the clipboard write in `--browse`.

## Architecture

`plank-replay` is a single Rust crate (edition 2024) with one dependency (`arboard`, for the clipboard write in `--browse`). It parses plank repro transcripts and replays their file-mutating tool calls into a fresh directory.

- `src/lib.rs` — crate root and public API re-exports.
- `src/main.rs` — CLI argument parsing and the `run` path (list, stats, browse, replay).
- `src/parse.rs` — transcript extraction and DSML tool-call decoding. Produces an ordered `Vec<Event>` of `Call` and `Seed` events; touches no filesystem.
- `src/replay.rs` — `Replayer` applies events to an output root. Handles `write`, `edit`, optional `bash`, path sandboxing, and per-step `Outcome` reporting.
- `src/seed.rs` — recovers pre-existing file contents from complete `read` tool results.
- `src/stats.rs` — read-only `--stats` report; shares the parser but never writes files.
- `src/browse.rs` — full-screen TUI over a repro directory, driven with ANSI escapes and `stty`; `c` copies the highlighted path to the clipboard via `arboard`.
- `src/error.rs` — `ReplayError`, a struct wrapping a private kind enum and a captured backtrace.
- `docs/DESIGN.md` — detailed design notes on the transcript format, edit semantics, read recovery, and stats decisions.

The pipeline is: repro text → locate transcript markers → split into role turns → decode DSML calls in assistant turns → scan user turns for complete reads → ordered event stream → `Replayer` → output directory. Parsing and replaying are deliberately separate so the decoder is testable against string fixtures without filesystem setup.

## Setup and environment

- Rust toolchain supporting edition 2024 is required.
- `HOME` — used to find the default repro directory (`~/.plank/repro`) when no path is given.
- `NO_COLOR` — disables ANSI colour in `--stats` and `--browse` output.
- `RUST_BACKTRACE` — controls whether `ReplayError::backtrace()` captures a real backtrace.
- `gh` CLI — required only for the `u` upload action in `--browse`; it must be installed and authenticated. The gist is created as secret.
- `arboard` — the crate behind the `c` copy action in `--browse`; no external CLI tool is needed for the clipboard.
- `stty` — required by `--browse` for raw mode and terminal size; the TUI needs a real terminal.

## Gotchas and workflow quirks

- **DSML dialects**: the parser handles both the legacy `dsml` (`tool_calls`, no space after `<｜DSML｜`) and current `dsml41` (`calls`, with a space). It never matches a fixed tag string; it strips the shared prefix and tolerates the optional space. Both spellings appear across a user's repro directory.
- **Only assistant turns are decoded**: DSML examples in the system prompt are ignored. A naive scan would replay plank's own documentation as if the model had requested it.
- **Replay scope**: only `write` and `edit` are replayed. `bash` is skipped unless `--run-bash` is passed; read-only tools are skipped. `--stop-on-error` aborts at the first failure; otherwise failures are recorded as `Outcome::Failed` and the replay continues.
- **Edit fidelity**: a plain `old` must match exactly once — zero or multiple matches is a failure. The anchored `[upto]` form splits into head/tail anchors, each must be unique, and an empty tail is rejected. Ambiguity is reproduced rather than resolved to the first match.
- **Seeding**: files are recovered only from complete `read` results (`lines 1-N of N`). Partial reads are deliberately ignored because a chunk can contain an edit anchor. Seeds never overwrite a path the transcript itself produced; `--no-seed` disables recovery entirely.
- **Path containment**: absolute paths are re-rooted under `_abs/`; any `..` component rejects the call. This is untrusted input describing a machine that may not be this one.
- **Output directory**: must be empty unless `--force` is passed. Default output is `./replay-<repro stem>`. The exit status is non-zero when any call failed, so a replay can gate a script.
- **`--stats` tolerates broken transcripts**: a truncated session can carry an unterminated tool call, which is a hard error for the replayer but only a note for the report. `Stats::of` keeps the decode failure as `parse_note` and reports what the header and transcript still hold.
- **`--browse` deletion is permanent**: backspace deletes the selected repro after a `y/N` confirmation and does not go through trash. The terminal is restored on every exit path via `RawMode`'s destructor.
- **`--browse` copy**: `c` writes the highlighted repro's full path to the system clipboard through `arboard`; failures are reported in the status bar.
- **Tests use per-test temp directories** keyed by process and thread id, so they neither collide nor leak. The real regression suite is the repro archive in `~/.plank/repro/` — running every repro and comparing file/failure counts against the previous run catches format drift that fixtures would not.
