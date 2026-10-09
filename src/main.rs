//! Command-line front end: `pt <command>` dispatches to one of the plank tools.

mod cli;

use std::fmt::Write as _;
use std::process::ExitCode;

use cli::{COMMANDS, CliError, Command};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((name, rest)) = args.split_first() else {
        eprint!("{}", usage());
        return ExitCode::from(2);
    };

    match name.as_str() {
        "-h" | "--help" => return help(None),
        "help" => return help(rest.first().map(String::as_str)),
        "-V" | "--version" => {
            println!("pt {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        _ => {}
    }

    let Some(command) = find(name) else {
        eprintln!("error: unknown command `{name}`\n\n{}", usage());
        return ExitCode::from(2);
    };

    match (command.run)(rest) {
        Ok(code) => code,
        Err(CliError::Help) => {
            print!("{}", command.usage);
            ExitCode::SUCCESS
        }
        Err(CliError::Usage(message)) => {
            eprintln!("error: {message}\n\n{}", command.usage);
            ExitCode::from(2)
        }
        Err(CliError::Failed(message)) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// Looks a command up by name.
fn find(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.name == name)
}

/// Handles `pt help [command]`.
fn help(topic: Option<&str>) -> ExitCode {
    let Some(name) = topic else {
        print!("{}", usage());
        return ExitCode::SUCCESS;
    };
    if let Some(command) = find(name) {
        print!("{}", command.usage);
        ExitCode::SUCCESS
    } else {
        eprintln!("error: unknown command `{name}`\n\n{}", usage());
        ExitCode::from(2)
    }
}

/// Top-level usage, listing every command from the table.
fn usage() -> String {
    let width = COMMANDS.iter().map(|c| c.name.len()).max().unwrap_or(0);
    let mut text = String::from(
        "pt - tools for plank repro files\n\n\
         USAGE:\n    pt <command> [ARGS]\n    pt help <command>\n\n\
         COMMANDS:\n",
    );
    for command in COMMANDS {
        let _ = writeln!(text, "    {:<width$}  {}", command.name, command.summary);
    }
    text.push_str(
        "\nOPTIONS:\n    -h, --help          Show this help\n    -V, --version       Show the version\n",
    );
    text
}
