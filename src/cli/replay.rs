//! `pt replay`: parse a plank repro and rebuild its workspace.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::ExitCode;

use plank_tools::{Outcome, Replayer, parse::parse_str};

use super::CliError;

/// Usage text for `pt replay --help`.
pub const USAGE: &str = "\
pt replay - rebuild the workspace a plank repro recorded

USAGE:
    pt replay <repro.md> [-o DIR] [OPTIONS]

ARGS:
    <repro.md>          Path to a plank repro file, e.g.
                        ~/.plank/repro/repro-debug-1789376559.md

OPTIONS:
    -o, --out DIR       Output directory (default: ./replay-<repro stem>)
    -l, --list          List the recorded calls without touching the filesystem
        --no-seed       Do not restore pre-existing files from `read` tool results
        --run-bash      Also execute the recorded bash commands in the output dir
        --stop-on-error Abort at the first failing call
    -f, --force         Reuse a non-empty output directory
    -q, --quiet         Only print the summary
    -h, --help          Show this help
";

/// Parsed `pt replay` arguments.
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool is an independent CLI flag"
)]
struct Args {
    repro: PathBuf,
    out: Option<PathBuf>,
    list: bool,
    run_bash: bool,
    stop_on_error: bool,
    force: bool,
    quiet: bool,
    no_seed: bool,
}

/// Runs `pt replay`.
pub fn run(args: &[String]) -> Result<ExitCode, CliError> {
    let args = parse_args(args)?;
    replay(&args)
}

/// Reads the arguments that follow `replay`.
fn parse_args(args: &[String]) -> Result<Args, CliError> {
    let mut repro = None;
    let mut out = None;
    let mut list = false;
    let (mut run_bash, mut stop_on_error, mut force, mut quiet, mut no_seed) =
        (false, false, false, false, false);

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(CliError::Help),
            "-l" | "--list" => list = true,
            "--run-bash" => run_bash = true,
            "--stop-on-error" => stop_on_error = true,
            "-f" | "--force" => force = true,
            "-q" | "--quiet" => quiet = true,
            "--no-seed" => no_seed = true,
            "-o" | "--out" => {
                let dir = it
                    .next()
                    .ok_or_else(|| CliError::Usage("`--out` needs a directory".into()))?;
                out = Some(PathBuf::from(dir));
            }
            other if other.starts_with('-') => {
                return Err(CliError::Usage(format!("unknown option `{other}`")));
            }
            other if repro.is_none() => repro = Some(PathBuf::from(other)),
            other => return Err(CliError::Usage(format!("unexpected argument `{other}`"))),
        }
    }

    let repro = repro.ok_or_else(|| CliError::Usage("a repro file is required".into()))?;
    Ok(Args {
        repro,
        out,
        list,
        run_bash,
        stop_on_error,
        force,
        quiet,
        no_seed,
    })
}

/// Parses the repro and either lists or replays it.
fn replay(args: &Args) -> Result<ExitCode, CliError> {
    let repro_path = &args.repro;
    let text = std::fs::read_to_string(repro_path)
        .map_err(|e| format!("{}: {e}", repro_path.display()))?;
    let repro = parse_str(&text).map_err(|e| e.to_string())?;

    if !args.quiet {
        let label = |k: &str| repro.meta(k).unwrap_or("?");
        println!(
            "repro : {}\nsession: {}   date: {}   model: {}",
            repro_path.display(),
            label("session"),
            label("date"),
            label("name"),
        );
        println!(
            "calls : {} total, {} file-mutating, {} file(s) recovered from reads\n",
            repro.calls().len(),
            repro.file_calls().len(),
            repro.seeds().len(),
        );
    }

    if args.list {
        for (i, call) in repro.calls().iter().enumerate() {
            let detail = call
                .param("path")
                .or_else(|| call.param("command"))
                .unwrap_or("");
            println!(
                "{:>3}. line {:<6} {:<6} {}",
                i + 1,
                call.line,
                call.name,
                detail.lines().next().unwrap_or("")
            );
        }
        return Ok(ExitCode::SUCCESS);
    }

    let out = args.out.clone().unwrap_or_else(|| default_out(repro_path));
    prepare_out(&out, args.force)?;

    let replayer = Replayer::new(&out)
        .run_bash(args.run_bash)
        .stop_on_error(args.stop_on_error)
        .seed(!args.no_seed);
    let reports = replayer.replay(&repro.events).map_err(|e| e.to_string())?;

    let mut touched = BTreeSet::new();
    let mut failures = 0usize;
    for report in &reports {
        match &report.outcome {
            Outcome::Skipped { .. } if args.quiet => continue,
            Outcome::Wrote { path, .. }
            | Outcome::Edited { path, .. }
            | Outcome::Seeded { path, .. } => {
                touched.insert(path.clone());
            }
            Outcome::Failed { .. } => failures += 1,
            Outcome::Ran { .. } | Outcome::Skipped { .. } => {}
        }
        if !args.quiet || matches!(report.outcome, Outcome::Failed { .. }) {
            println!("{:>3}. {}", report.index, report.outcome);
        }
    }

    println!(
        "\nrebuilt {} file(s) in {}{}",
        touched.len(),
        out.display(),
        if failures == 0 {
            String::new()
        } else {
            format!("  ({failures} call(s) failed)")
        }
    );
    for path in &touched {
        println!("  {}", path.display());
    }

    Ok(if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// Derives `./replay-<stem>` from the repro file name.
fn default_out(repro: &std::path::Path) -> PathBuf {
    let stem = repro
        .file_stem()
        .map_or_else(|| "repro".into(), |s| s.to_string_lossy().into_owned());
    PathBuf::from(format!("replay-{stem}"))
}

/// Creates the output directory, refusing to reuse a non-empty one without `--force`.
fn prepare_out(out: &std::path::Path, force: bool) -> Result<(), String> {
    if out.exists() {
        let mut entries = std::fs::read_dir(out).map_err(|e| format!("{}: {e}", out.display()))?;
        if entries.next().is_some() && !force {
            return Err(format!(
                "{} is not empty; pass --force to replay into it anyway",
                out.display()
            ));
        }
        return Ok(());
    }
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))
}
