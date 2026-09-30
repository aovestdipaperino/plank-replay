# Bug: `cargo install --path .` fails with "Operation not permitted" on `~/.cargo/.crates.toml`

## Environment

- macOS (filesystem `/dev/disk3s5` mounted at `/System/Volumes/Data`)
- Rust crate `plank-replay`, edition 2024, single dependency `arboard`
- Agent runs inside a sandboxed shell (ds4-agent)

## Steps to reproduce

1. `cargo install --path .`
2. Observe:

```
error: failed to open: /Users/enzo/.cargo/.crates.toml
Caused by: Operation not permitted (os error 1)
```

## Expected

`cargo install` writes the crate's registry entry to `~/.cargo/.crates.toml` and installs the `plank-replay` binary into `~/.cargo/bin`, then reports success.

## Actual

The install aborts before writing anything. The same denial hits even a bare write attempt:

```
$ touch /Users/enzo/.cargo/.crates.toml
touch: /Users/enzo/.cargo/.crates.toml: Operation not permitted
$ touch /Users/enzo/.cargo/.plank-write-test
touch: /Users/enzo/.cargo/.plank-write-test: Operation not permitted
```

## Root cause analysis

The failure is a process-level sandbox denial, not a normal permission problem. Evidence:

- `~/.cargo/.crates.toml` is owned by `enzo`, mode `0644`, and has **no** macOS flags (no `uchg`/immutable, no `nodump`) — `ls -lO` shows `-` in the flags column.
- `~/.cargo` itself is owned by `enzo` with `drwxr-xr-x` — writable by owner.
- The filesystem is not read-only: `/dev/disk3s5` on `/System/Volumes/Data` reports 93% used with free capacity.
- Writing a **new** file in `~/.cargo` is denied too, so it is not a single locked file.
- No sandbox-related environment variables are visible (`env | grep -i sandbox` is empty).
- Writes inside the project directory succeed (edits, `target/`, git commits all worked).

The error is `EPERM` ("Operation not permitted", errno 1) rather than `EACCES` ("Permission denied", errno 13). That combination — a writable, owner-owned directory on a writable filesystem, denied with EPERM for both existing and new files — is the classic signature of a macOS Seatbelt/sandbox profile that restricts the process to a designated workspace. The agent process is allowed to write inside the project tree but is denied writes to `~/.cargo`.

## Impact

- `cargo install --path .` cannot run inside the sandbox, so the binary cannot be installed to the user's cargo bin from here.
- Build, test, and lint are unaffected: `cargo test`, `cargo clippy --all-targets`, and the pre-commit hooks all pass because they only write to `target/` inside the project.

## Workaround

Run `cargo install --path .` from a normal terminal outside the sandbox. If the goal is only to prove the install path works, `cargo install --path . --root /tmp/plank-install` installs into a writable location instead of `~/.cargo`.
