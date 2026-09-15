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
cargo install plank-replay
```

Or from a clone:

```
cargo install --path .
```

As a library dependency:

```
cargo add plank-replay
```

No dependencies beyond the standard library. Rust 2024 edition.

## Usage

```
plank-replay <repro.md> [-o DIR] [OPTIONS]
plank-replay --browse [DIR]

  -o, --out DIR       Output directory (default: ./replay-<repro stem>)
  -l, --list          List the recorded calls without touching the filesystem
      --browse        Browse a repro directory in a full-screen TUI
      --stats         Print a one-page summary of the session and exit
      --no-color      Never colour the --stats report
      --no-seed       Do not restore pre-existing files from `read` tool results
      --run-bash      Also execute the recorded bash commands in the output dir
      --stop-on-error Abort at the first failing call
  -f, --force         Reuse a non-empty output directory
  -q, --quiet         Only print the summary
```

The exit status is non-zero when any call failed, so a replay can gate a script.

## Reading a session at a glance

`--stats` parses the repro and prints a single page instead of rebuilding anything. It
opens with the prompt the session started from, recovered from the last human turn before
the model first answered, with the date, hook context, and agent instructions plank
injects around it stripped away. Then it reports the model and sampling settings, how much of the context window the session ended
on, and the generation passes: tokens emitted, decode throughput averaged over decode time
only, reasoning bytes the loop guard saw, and a histogram of why each pass stopped. Below
that it counts every tool result in the transcript by tool, tallies clean and non-zero
shell exits, and lists the `Tool error:` messages the model was handed, including loop
guard refusals. It closes with what a real replay would rebuild and the wall-clock span of
the transcript.

```
plank-replay --stats ~/.plank/repro/repro-debug-1789414920.md
```

An `outcome` line states how the session ended, read from the stop reason of the last
generation pass: `goal reached` in green when the model answered instead of calling another
tool, and red for `interrupted by user` or `cut by a loop guard`. Beside it is the wall
time the transcript spans and, of that, how much was spent generating.

The report is coloured when it goes to a terminal, and plain when it is piped, redirected,
or `NO_COLOR` is set; `--no-color` forces plain.

Older repros predate the passes table; the report then prints the sections it can fill and
says so for the rest.

## Browsing the repro folder

`--browse` opens a full-screen browser over a directory of repros, `~/.plank/repro` unless
another is given. The left pane lists every `*.md` newest first; the right pane shows the
highlighted repro's `--stats` report, led by the things that identify a session at a glance:
the prompt it started from, the model that ran it, whether skills were enabled, when it ran,
and how long it took.

```
PROMPT
  write the game described in SNAKE-PROMPT.md

MODEL   Qwen3.8 Flash Next
  family qwen   dialect qwen   think low

SKILLS  enabled, 1 invocation

WHEN    2026-09-15  09:47:19 -> 10:00:29

TIME    13m 10s   generating 9m 44s
```

`SKILLS` reads the tool schema plank writes into the transcript: a declared `skill` tool
means the session was offered skills, and the invocation count comes from the tool results.
A session that had skills available but never reached for them reads `enabled, never
invoked`, which is the interesting case when comparing runs. `WHEN` uses the transcript's
first and last timestamps rather than the header date, and collapses to a single day when
the session did not cross midnight.

```
plank-replay --browse
```

| Key | Action |
|---|---|
| `↑`/`↓`, `j`/`k` | Move the selection |
| `PgUp`/`PgDn`, `g`/`G` | Page, or jump to the first or last repro |
| `J`/`K` | Scroll the report panel |
| `⌫` | Delete the selected repro, after a `y/N` confirmation |
| `u` | Upload the selected repro to a secret GitHub gist |
| `r` | Re-read the directory |
| `q` | Quit |

Deletion is permanent and does not go through the trash. The upload shells out to the
[`gh`](https://cli.github.com) command line tool and needs it installed and logged in; the
gist it creates is secret, not public, since a repro embeds the session's whole transcript.
The resulting URL is printed in the status bar.

The browser drives the terminal with plain ANSI escapes and `stty`, so it adds no
dependencies, and it restores the terminal on every exit path.

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
