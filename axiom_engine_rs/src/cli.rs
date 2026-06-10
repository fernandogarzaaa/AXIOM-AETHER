use std::path::PathBuf;

use clap::{Args, CommandFactory, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "axiom",
    about = "Axiom neural hypervisor runtime",
    version,
    disable_help_subcommand = true
)]
pub struct AxiomCli {
    #[command(subcommand)]
    pub command: Option<AxiomCommand>,
}

#[derive(Debug, Subcommand)]
pub enum AxiomCommand {
    /// Scaffold ~/.axiom with config, logs, run state, and model cache paths.
    Init(InitArgs),
    /// Manage the background Neural VFS and swarm listener daemon.
    Daemon {
        #[command(subcommand)]
        command: DaemonCommand,
    },
    /// Attach a directory to the running user-mode VFS hypervisor.
    Mount {
        /// Directory to mount into the hypervisor.
        path: PathBuf,
    },
    /// Warm-start the persistent vibe memory by absorbing a codebase through TTT.
    Prime {
        /// Directory to crawl and absorb (defaults to the current directory).
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Measure compression: token savings + structural round-trip fidelity.
    Bench {
        /// Directory to crawl and measure (defaults to the current directory).
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Run a program under self-healing supervision: failures are absorbed into
    /// the TTT fast-weights, the environment is repaired (e.g. missing
    /// directories created), and the program is restarted until it succeeds.
    Run {
        /// Restart budget after the first attempt.
        #[arg(long, default_value_t = 3)]
        max_restarts: usize,
        /// The command to supervise, followed by its arguments.
        #[arg(trailing_var_arg = true, required = true, num_args = 1..)]
        command: Vec<String>,
    },
    /// Manage DWE swarm peers.
    Swarm {
        #[command(subcommand)]
        command: SwarmCommand,
    },
    /// Report what Axiom has learned about program failures (acquired immunity):
    /// remembered heals, per-program failure-tension history, and confidence.
    Immunity {
        /// Optional case-insensitive command substring to filter by.
        query: Option<String>,
        /// Forget faded heals: drop records whose confidence has decayed below
        /// the prune floor (memory waning → clonal deletion).
        #[arg(long)]
        prune: bool,
    },
}

#[derive(Debug, Args, Default)]
pub struct InitArgs {
    /// Skip automatic base-model download during initialization.
    #[arg(long)]
    pub no_fetch: bool,
    /// Skip bootstrapping a local checkpoint. By default `init` trains a small
    /// model locally (no network) when none exists, so the proxy never boots on
    /// random weights.
    #[arg(long)]
    pub no_train: bool,
}

#[derive(Debug, Subcommand)]
pub enum DaemonCommand {
    /// Start the Axiom hypervisor in the background.
    Start,
    /// Stop the background hypervisor if it is running.
    Stop,
    /// Print daemon PID, health, and log path.
    Status,
}

#[derive(Debug, Subcommand)]
pub enum SwarmCommand {
    /// Persist a DWE peer address in ~/.axiom/config.toml.
    Connect {
        /// Peer IP or host:port. Bare hosts default to port 9191.
        ip: String,
    },
    /// Pull a peer's heal memory and merge it into this node's (swarm
    /// immunity): heals learned anywhere in the fleet immunize this machine.
    Immunity {
        /// Peer Axiom server, host:port (the HTTP proxy port, e.g. 3000).
        peer: String,
    },
}

#[derive(Debug)]
pub enum ParsedCli {
    Command(AxiomCommand),
    HelpPrinted,
    Legacy,
}

pub fn parse_entry() -> Result<ParsedCli, clap::Error> {
    let argv: Vec<String> = std::env::args().collect();
    if should_use_legacy_parser(&argv) {
        return Ok(ParsedCli::Legacy);
    }
    if argv.len() == 1 {
        AxiomCli::command().print_long_help()?;
        println!();
        return Ok(ParsedCli::HelpPrinted);
    }
    let cli = match AxiomCli::try_parse_from(argv) {
        Ok(cli) => cli,
        Err(err)
            if matches!(
                err.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            let _ = err.print();
            return Ok(ParsedCli::HelpPrinted);
        }
        Err(err) => return Err(err),
    };
    match cli.command {
        Some(command) => Ok(ParsedCli::Command(command)),
        None => {
            AxiomCli::command().print_long_help()?;
            println!();
            Ok(ParsedCli::HelpPrinted)
        }
    }
}

fn should_use_legacy_parser(argv: &[String]) -> bool {
    let Some(first) = argv.get(1).map(String::as_str) else {
        return false;
    };
    matches!(
        first,
        "--mode"
            | "--lsp"
            | "--epochs"
            | "--steps-per-epoch"
            | "--batch-size"
            | "--seq-len"
            | "--max-new-tokens"
            | "--checkpoint"
            | "--tokenizer"
            | "--context-api-url"
            | "--context-api-key"
            | "--max-context-tokens"
            | "--host"
            | "--port"
            | "--device"
    )
}

pub fn normalize_peer(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() || trimmed.contains(':') {
        trimmed.to_string()
    } else {
        format!("{trimmed}:9191")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_mode_flags_stay_compatible() {
        let argv = vec![
            "axiom".to_string(),
            "--mode".to_string(),
            "server".to_string(),
        ];
        assert!(should_use_legacy_parser(&argv));
    }

    #[test]
    fn bare_swarm_peer_gets_default_port() {
        assert_eq!(normalize_peer("10.0.0.2"), "10.0.0.2:9191");
        assert_eq!(normalize_peer("10.0.0.2:9292"), "10.0.0.2:9292");
    }
}
