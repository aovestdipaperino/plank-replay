<p align="center">
  <img src="assets/logo.svg" alt="plank-replay logo" width="140">
</p>

<h1 align="center">plank-replay</h1>

<p align="center">
  <em>Rebuild the workspace a plank session produced, from the repro file it left behind.</em>
</p>

When a [plank](https://plank-agent.dev) session ends it writes a report into `~/.plank/repro/` containing the exact
engine transcript. Everything the model built is in there, but only as a sequence of tool
calls. plank-replay decodes those calls and applies them to a fresh directory, so the
session's output becomes a tree you can actually compile.

```
$ plank-replay ~/.plank/repro/repro-debug-1789376559.md -o /tmp/parola
repro : /Users/enzo/.plank/repro/repro-debug-1789376559.md
session: plucky-vivaldi   date: 2026-09-14   model: DeepSeek V4.1 Flash
calls : 48 total, 16 file-mutating, 2 file(s) recovered from reads

  3. seed   PAROLA-PROMPT.md (5649 B)
 12. write  Cargo.toml (139 B)
 21. write  src/words.rs (6684 B)
 23. edit   src/words.rs (6666 B)
 ...
rebuilt 9 file(s) in /tmp/parola

$ cd /tmp/parola && cargo test
test result: ok. 24 passed; 0 failed
```

## Install

```
cargo install --path .
```

No dependencies beyond the standard library. Rust 2024 edition.

## Usage

```
plank-replay <repro.md> [-o DIR] [OPTIONS]

  -o, --out DIR       Output directory (default: ./replay-<repro stem>)
  -l, --list          List the recorded calls without touching the filesystem
      --no-seed       Do not restore pre-existing files from `read` tool results
      --run-bash      Also execute the recorded bash commands in the output dir
      --stop-on-error Abort at the first failing call
  -f, --force         Reuse a non-empty output directory
  -q, --quiet         Only print the summary
```

The exit status is non-zero when any call failed, so a replay can gate a script.

## What gets replayed

`write` creates or replaces a file. `edit` patches one, following plank's rules exactly:
the `old` text has to match once, and the anchored `[upto]` form replaces everything from
a unique head anchor through a unique tail anchor. An edit that was ambiguous in the
original session fails here too and is reported, rather than being quietly resolved to the
first match.

Read-only tools have nothing to replay and are skipped. `bash` is skipped unless you pass
`--run-bash`, which runs the recorded commands with the output directory as their working
directory.

## Files the session did not create

Sessions routinely edit files that already existed in their workspace, and a repro does
not carry those as content. It does carry them when the model read one in full, as a
numbered-line tool result, so plank-replay restores those files before the edits that need
them.

Partial reads are deliberately ignored. plank serves large files in chunks, and an edit
anchor can match inside a chunk, so seeding from a truncated file would let an edit succeed
against text the session never saw. Use `--no-seed` to turn recovery off entirely.

A file that was neither written nor read in full has no text anywhere in the repro, and
its edits fail with `no base contents`.

## Where things land

Everything is written inside the output directory. Relative paths keep their shape;
absolute ones are re-rooted under `_abs/`, so `/Users/enzo/Code/x/src/a.rs` becomes
`_abs/Users/enzo/Code/x/src/a.rs`. Paths containing `..` are rejected.

## Dialects

Both DSML spellings plank has emitted are handled: the current `dsml41`
(`<｜DSML｜ invoke …>`, block tag `calls`) and the older `dsml`
(`<｜DSML｜invoke …>`, block tag `tool_calls`).

## Library use

The crate is usable directly. `parse_repro` returns the header metadata and an ordered
event stream; `Replayer` applies it.

```rust
use plank_replay::{Replayer, parse_repro};

let repro = parse_repro("repro-debug-1789376559.md")?;
let reports = Replayer::new("out").replay(&repro.events)?;
for report in &reports {
    println!("{}", report.outcome);
}
# Ok::<(), plank_replay::ReplayError>(())
```

## Design

[`docs/DESIGN.md`](docs/DESIGN.md) covers the transcript format, why parsing is scoped to
assistant turns, the edit semantics, and the reasoning behind read recovery.

## Development

```
cargo test
cargo clippy --all-targets
```

## License

MIT. See [LICENSE](LICENSE).
