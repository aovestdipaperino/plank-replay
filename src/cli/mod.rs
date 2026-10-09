//! The `pt` subcommands. Each one owns its argument parsing and usage text;
//! adding a tool means adding a module and an entry in [`COMMANDS`].

mod browse;
mod install;
mod replay;
mod stats;
mod vecdiff;
mod vectorize;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;

/// One `pt` subcommand.
#[derive(Debug)]
pub struct Command {
    /// The word typed after `pt`.
    pub name: &'static str,
    /// One line for the top-level command list.
    pub summary: &'static str,
    /// Full usage text for `pt <name> --help`.
    pub usage: &'static str,
    /// Runs the command on the arguments that follow its name.
    pub run: fn(&[String]) -> Result<ExitCode, CliError>,
}

/// Every command `pt` knows, in the order they are listed in the help.
pub const COMMANDS: &[Command] = &[
    Command {
        name: "replay",
        summary: "Rebuild the workspace a repro recorded",
        usage: replay::USAGE,
        run: replay::run,
    },
    Command {
        name: "stats",
        summary: "Print a one-page summary of a recorded session",
        usage: stats::USAGE,
        run: stats::run,
    },
    Command {
        name: "browse",
        summary: "Browse a repro directory in a full-screen TUI",
        usage: browse::USAGE,
        run: browse::run,
    },
    Command {
        name: "install",
        summary: "Install steering vectors or a profile from a repository",
        usage: install::USAGE,
        run: install::run,
    },
    Command {
        name: "vectorize",
        summary: "Build a steering vector for a model from two prompt sets",
        usage: vectorize::USAGE,
        run: vectorize::run,
    },
];

/// Why a command did not run to completion.
#[derive(Debug)]
pub enum CliError {
    /// `-h`/`--help` was passed; the dispatcher prints the command's usage.
    Help,
    /// The arguments were wrong; the usage is printed after the message.
    Usage(String),
    /// The command ran and failed.
    Failed(String),
}

impl From<String> for CliError {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}

impl From<&str> for CliError {
    fn from(message: &str) -> Self {
        Self::Failed(message.into())
    }
}

/// The directory plank writes its repro reports into.
fn default_repro_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".plank").join("repro")
}

/// Whether to colour output on stdout, honouring `--no-color` and `NO_COLOR`.
fn colour_enabled(no_color: bool) -> bool {
    !no_color && std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}
