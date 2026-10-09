//! `pt stats`: a one-page summary of a recorded session.

use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;

use plank_tools::{Stats, Style};

use super::CliError;

/// Usage text for `pt stats --help`.
pub const USAGE: &str = "\
pt stats - print a one-page summary of a recorded session

USAGE:
    pt stats <repro.md> [OPTIONS]

Writes nothing: reports throughput, tool usage, guard and tool errors.

ARGS:
    <repro.md>          Path to a plank repro file

OPTIONS:
        --no-color      Never colour the report
    -h, --help          Show this help
";

/// Runs `pt stats`.
pub fn run(args: &[String]) -> Result<ExitCode, CliError> {
    let mut repro = None;
    let mut no_color = false;
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => return Err(CliError::Help),
            "--no-color" => no_color = true,
            other if other.starts_with('-') => {
                return Err(CliError::Usage(format!("unknown option `{other}`")));
            }
            other if repro.is_none() => repro = Some(PathBuf::from(other)),
            other => return Err(CliError::Usage(format!("unexpected argument `{other}`"))),
        }
    }
    let repro = repro.ok_or_else(|| CliError::Usage("a repro file is required".into()))?;

    let text = std::fs::read_to_string(&repro).map_err(|e| format!("{}: {e}", repro.display()))?;
    let style = Style::detect(!no_color && std::io::stdout().is_terminal());
    print!("{}", Stats::of(&text).render(&repro, style));
    Ok(ExitCode::SUCCESS)
}
