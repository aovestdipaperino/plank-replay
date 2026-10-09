<p align="center">
  <img src="assets/logo.svg" alt="plank-tools logo" width="140">
</p>

<h1 align="center">plank-tools</h1>

<p align="center">
  <em>Tools for the repro files plank sessions leave behind, starting with a replayer that rebuilds their workspace.</em>
</p>

When a [plank](https://plank-agent.dev) session ends it writes a report into `~/.plank/repro/` containing the exact
engine transcript. Everything the model built is in there, but only as a sequence of tool
calls. `pt`, the plank-tools binary, works on those files through a set of commands:
`pt replay` decodes the calls and applies them to a fresh directory, so the session's
output becomes a tree you can actually compile; `pt stats` summarises a session on one
page; `pt browse` walks the whole repro folder; `pt vectorize` builds a steering vector
for a plank model.

```
$ pt replay ~/.plank/repro/repro-debug-1789376559.md -o /tmp/parola
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
cargo install plank-tools
```

Or from a clone:

```
cargo install --path .
```

As a library dependency:

```
cargo add plank-tools
```

Two dependencies: `arboard`, for the clipboard write in `pt browse`, and `serde_json`.
`pt vectorize` also shells out to `curl` for Hugging Face sources, and links the ds4 engine from a plank checkout at `../plank` (`crates/local-inference-engine`). Rust 2024 edition.

## Usage

```
pt <command> [ARGS]
pt help <command>

  replay     Rebuild the workspace a repro recorded
  stats      Print a one-page summary of a recorded session
  browse     Browse a repro directory in a full-screen TUI
  vectorize  Build a steering vector for a model from two prompt sets
```

`pt replay` takes a repro file and these options:

```
pt replay <repro.md> [-o DIR] [OPTIONS]

  -o, --out DIR       Output directory (default: ./replay-<repro stem>)
  -l, --list          List the recorded calls without touching the filesystem
      --no-seed       Do not restore pre-existing files from `read` tool results
      --run-bash      Also execute the recorded bash commands in the output dir
      --stop-on-error Abort at the first failing call
  -f, --force         Reuse a non-empty output directory
  -q, --quiet         Only print the summary
```

`pt stats <repro.md>` and `pt browse [DIR]` both accept `--no-color`. Every command takes
`-h`/`--help`.

The exit status of `pt replay` is non-zero when any call failed, so a replay can gate a
script.

## Reading a session at a glance

`pt stats` parses the repro and prints a single page instead of rebuilding anything. It
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
pt stats ~/.plank/repro/repro-debug-1789414920.md
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

`pt browse` opens a full-screen browser over a directory of repros, `~/.plank/repro` unless
another is given. The left pane lists every `*.md` newest first; the right pane shows the
highlighted repro's `pt stats` report, led by the things that identify a session at a glance:
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
pt browse
```

| Key | Action |
|---|---|
| `↑`/`↓`, `j`/`k` | Move the selection |
| `PgUp`/`PgDn`, `g`/`G` | Page, or jump to the first or last repro |
| `J`/`K` | Scroll the report panel |
| `⌫` | Delete the selected repro, after a `y/N` confirmation |
| `u` | Upload the selected repro to a secret GitHub gist |
| `c` | Copy the selected repro's full path to the system clipboard |
| `r` | Re-read the directory |
| `q` | Quit |

Deletion is permanent and does not go through the trash. The upload shells out to the
[`gh`](https://cli.github.com) command line tool and needs it installed and logged in; the
gist it creates is secret, not public, since a repro embeds the session's whole transcript.
The resulting URL is printed in the status bar.

The browser drives the terminal with plain ANSI escapes and `stty`; the clipboard
write goes through the `arboard` crate. It restores the terminal on every exit path.

## Building a steering vector

`pt vectorize` builds a ds4 directional-steering vector, the `.f32` file a plank engine's
`steering` block points at. It takes a plank model, a set of baseline prompts and a set of
target prompts, runs every prompt through the ds4 engine with the activation dump on, and writes one
unit direction per layer separating the two sets, oriented so that a positive scale pushes
the model towards the `to` prompts.

```
pt vectorize ds4vision \
  --to dir-steering/examples/succinct.txt \
  --from dir-steering/examples/verbose.txt \
  -n succinct
```

The vector is stored by name in `~/.plank/models/vectors.json`, which is created on first
use. The file is a JSON array with one entry per model file, each listing its vectors with
the raw little-endian `f32` matrix in base64:

```json
[
  {
    "model": "ds4vision.gguf",
    "vectors": [
      { "name": "succinct", "value": "Pq3vPLnF..." }
    ]
  }
]
```

`model` is the file name of the GGUF the model name resolved to. A model/name pair that
already exists is refused before any work starts, unless `--force` is given to replace it;
other entries, and any extra fields in them, are left as they were. `-o FILE` writes the raw
`.f32` file instead, or as well, with a metadata sidecar beside it. If the store cannot be
written at the end of a long run and no `-o` was given, the vector is saved to
`./<name>.f32` so the work is not lost.

The model is a plank engine name, resolved the way plank resolves it: a `main.path` in
`~/.plank/engines.local.json`, else the managed `~/.plank/<name>.gguf`. A path to a `.gguf`
works too. The vector's shape comes from the GGUF architecture: DeepSeek V4 Flash is
43 x 4096, GLM 5.3 Flash 45 x 4096, Qwen3.8 Flash Next 48 x 2560. Gemma engines run on
plank's native engine rather than ds4 and cannot be steered this way.

A prompt source is a local file or a Hugging Face dataset. Text files hold one prompt per
line; `.jsonl` and `.json` files hold strings or objects. A dataset is written
`owner/name` or `owner:name`, which reads its `train` split (or its first split when it has
no `train`), or `owner/name:path` to pick something else. A `.txt`, `.jsonl` or `.json`
path is downloaded as is, and anything else names a split read through the
datasets-server API, so `mlabonne:harmful_behaviors`, `mlabonne/harmful_behaviors:train`
and `mlabonne/harmful_behaviors:data/train-00000-of-00001.parquet` all read the same rows. The
prompt comes from `--column`, or else the first of `text`, `prompt`, `instruction`,
`question`, `input` or `goal`. `--from` and `--to` can be repeated to combine sources. A
Hugging Face API key is sent when `HF_API_KEY` is set, for gated or private datasets;
`HF_TOKEN`, the name the Hugging Face tools use, works too, and `HF_API_KEY` wins when both
are set.

Dataset rows are cached in `~/.cache/plank-tools/hf` (or under `$XDG_CACHE_HOME`), one
JSON line per row, written as each page of 100 arrives. A dataset is downloaded once, and a
fetch that is interrupted or turned away by Hugging Face's rate limit resumes from the rows
already on disk. Rate limits and server errors are retried with a growing wait of up to two
minutes; an API key raises the limit. Since prompts past the shorter set are never paired,
both sides are sized first, from a single-row request per split (or the cache), and each is
fetched only up to the smaller of the two, or `-l` if that is lower. Pairing
`harmful_behaviors` with `harmless_alpaca` costs 416 rows from each, whichever side the large
one is on, rather than all 25,058. `--refresh` downloads again and `--no-cache` skips
the cache.

The two sets are paired line by line and cut to the shorter one, or to `-l`/`--limit`. The
model is loaded once, which takes seconds to minutes depending on the page cache, and each
capture after that is a single prefill (about a quarter of a second for a short prompt on
`ds4vision`). The vector is the difference
of the two means per layer, `from - to`, made orthogonal to the `to` mean and normalized.
`--pair-normalize` and `--no-orthogonalize` switch those steps the same way the flags of
ds4's own `build_direction.py` do, and the result matches that script bit for bit when it
is given the `from` prompts as `--good-file` and the `to` prompts as `--bad-file`. With `-o`,
metadata recording the sources and settings is written next to the file as `<out>.json`.

ds4 applies the vector as a projection, `y - scale * d * dot(d, y)`, which cannot tell `d`
from `-d`; what sets the direction of travel is which set the vector is orthogonalized
against. Building it as `from - to` against the `to` mean leaves `to`-like activations near
zero along it, so a positive scale strips the `from` component and pushes the model towards
the `to` prompts, and a negative scale pushes it towards `from`:

```json
"steering": { "file": "/Users/me/.plank/steering/succinct.f32", "ffn": 1 }
```

This is the reverse of `build_direction.py`'s own convention, where the target goes in
`--good-file` and a negative scale amplifies it. Vectors built before this change, such as
a heretic vector with harmful prompts as `--to`, keep their old meaning; rebuild them with
`--to` and `--from` as the side to move towards and the side to move away from.

The ds4 engine is linked into `pt` through plank's `local-inference-engine` crate, which builds it from
`refs/ds4` in a plank checkout beside this repository (`../plank`); no ds4 binary is
involved. On macOS the engine compiles its Metal kernels from source when the model loads,
and finds them the way plank does: `--metal DIR`, then `$DS4_METAL_DIR`, then the
`refs/ds4/metal` directory `pt` was built against, then `../share/plank/metal` beside the
`pt` binary. Without kernels the run stops before downloading anything. The kernels must
come from the same ds4 version `pt` was built with. Like plank, the engine holds
`/tmp/ds4.lock` while loaded, so quit a running plank or ds4 first.

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
numbered-line tool result, so `pt replay` restores those files before the edits that need
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
use plank_tools::{Replayer, parse_repro};

let repro = parse_repro("repro-debug-1789376559.md")?;
let reports = Replayer::new("out").replay(&repro.events)?;
for report in &reports {
    println!("{}", report.outcome);
}
# Ok::<(), plank_tools::ReplayError>(())
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
