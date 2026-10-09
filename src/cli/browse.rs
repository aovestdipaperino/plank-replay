//! `pt browse`: a full-screen TUI over a repro directory.

use std::path::PathBuf;
use std::process::ExitCode;

use super::{CliError, colour_enabled, default_repro_dir};

/// Usage text for `pt browse --help`.
pub const USAGE: &str = "\
pt browse - browse a repro directory in a full-screen TUI

USAGE:
    pt browse [DIR] [OPTIONS]

Lists the repros newest first, with the `pt stats` report of the highlighted
one in a side panel.

ARGS:
    [DIR]               Directory of repro files (default: ~/.plank/repro)

OPTIONS:
        --no-color      Never colour the report panel
    -h, --help          Show this help
";

/// Runs `pt browse`.
pub fn run(args: &[String]) -> Result<ExitCode, CliError> {
    let mut dir = None;
    let mut no_color = false;
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => return Err(CliError::Help),
            "--no-color" => no_color = true,
            other if other.starts_with('-') => {
                return Err(CliError::Usage(format!("unknown option `{other}`")));
            }
            other if dir.is_none() => dir = Some(PathBuf::from(other)),
            other => return Err(CliError::Usage(format!("unexpected argument `{other}`"))),
        }
    }

    let dir = dir.unwrap_or_else(default_repro_dir);
    plank_tools::browse(&dir, colour_enabled(no_color))?;
    Ok(ExitCode::SUCCESS)
}
