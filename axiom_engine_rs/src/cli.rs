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
    /// Autonomy (Pillar 3): drive a failing verify command to green by chaining
    /// environment self-healing and Poly JIT / LLM source repair, then
    /// remembering what worked. With no --source, the faulty file is localized
    /// automatically from the verify command's own output (works across Rust,
    /// Python, JS/TS, Go, C/C++, …). `axiom solve [--source PATH] -- <cmd>`.
    Solve {
        /// Max environment-heal + source-repair rounds.
        #[arg(long, default_value_t = 2)]
        max_rounds: usize,
        /// Restart budget for the environment supervisor each round.
        #[arg(long, default_value_t = 3)]
        max_restarts: usize,
        /// Source file to repair (reversible). Optional: when omitted, the file
        /// is auto-detected from the failure trace. Pass it to pin the target.
        #[arg(long)]
        source: Option<PathBuf>,
        /// The verify command to drive to green, followed by its arguments.
        /// Optional: with none, the project language is detected and a default
        /// check is used (e.g. `cargo test`, `pytest`, `go test`).
        #[arg(trailing_var_arg = true, num_args = 0..)]
        command: Vec<String>,
    },
    /// Agentic Core: goal-directed autonomous coding. Drive a verify command to
    /// green toward an objective, editing files under an all-or-nothing,
    /// verifier-gated, reversible transaction loop (a rejected change is rolled
    /// back byte-for-byte; identical failed changes are never retried). Repair is
    /// the special case (omit --goal). `axiom task --goal "<desc>" [--file P]... -- <cmd>`.
    Task {
        /// What to accomplish, in natural language. Omit to mean "make the verify
        /// command pass" (pure repair).
        #[arg(long, default_value = "")]
        goal: String,
        /// A file the agent may edit (repeatable). With none, target files are
        /// localized automatically from the verify command's failure output.
        #[arg(long = "file")]
        files: Vec<PathBuf>,
        /// Max verifier-gated attempts.
        #[arg(long, default_value_t = 4)]
        max_attempts: usize,
        /// The verify command that grounds acceptance, followed by its arguments.
        #[arg(trailing_var_arg = true, required = true, num_args = 1..)]
        command: Vec<String>,
    },
    /// Measure the autonomous loop's capability as a reproducible score: run the
    /// built-in seeded broken-repo fixtures through the real solve loop and report
    /// how many it repairs end-to-end (deterministic, no LLM required).
    EvalAgentic {},
    /// Run a program under self-healing supervision: failures are absorbed into
    /// the TTT fast-weights, the environment is repaired (e.g. missing
    /// directories created), and the program is restarted until it succeeds.
    Run {
        /// Restart budget after the first attempt.
        #[arg(long, default_value_t = 3)]
        max_restarts: usize,
        /// Predict from learned immunity whether the command would fail now,
        /// then exit WITHOUT running it (anticipatory pre-flight).
        #[arg(long)]
        dry_run: bool,
        /// The command to supervise, followed by its arguments.
        #[arg(trailing_var_arg = true, required = true, num_args = 1..)]
        command: Vec<String>,
    },
    /// Manage DWE swarm peers.
    Swarm {
        #[command(subcommand)]
        command: SwarmCommand,
    },
    /// Fleet operations: inspect DWE/immunity wiring and print peer-join config.
    Fleet {
        #[command(subcommand)]
        command: FleetCommand,
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
    /// Run AXIOM's in-tree ChimeraLang (AI-cognition DSL): check, run, prove
    /// (emit a signed certificate), or verify a certificate. `.chimera` programs
    /// execute on the same belief/provenance substrate as the engine itself.
    Chimera {
        #[command(subcommand)]
        command: ChimeraCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum ChimeraCommand {
    /// Lex + parse a `.chimera` file, reporting syntax errors (the verify gate).
    Check {
        /// Path to the `.chimera` source file.
        file: PathBuf,
    },
    /// Execute a `.chimera` file and print emitted values + any guard violations.
    Run {
        /// Path to the `.chimera` source file.
        file: PathBuf,
    },
    /// Execute and emit a tamper-evident certificate of the run (SHA-256, plus
    /// HMAC when `AXIOM_FLEET_KEY` is set), reusing AXIOM's provenance layer.
    Prove {
        /// Path to the `.chimera` source file.
        file: PathBuf,
        /// Where to write the certificate JSON (defaults to stdout).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Verify a certificate produced by `prove`, offline.
    Verify {
        /// Path to the certificate JSON.
        cert: PathBuf,
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
pub enum FleetCommand {
    /// Show DWE wiring and whether a fleet key is set on this node.
    Status,
    /// Print the environment a peer node must set to join this fleet.
    Join {
        /// Peer address as host:port (its AXIOM_DWE_LISTEN).
        peer: String,
    },
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
