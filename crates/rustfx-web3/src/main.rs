//! rustFX Web3 plane.
//! Read-only catalog and health JSON for Laura. No custody. No signing.

use std::env;
use std::io::{self, Write};
use std::process::ExitCode;

const CONTRACT: &str = "laura-rustfx-web3-v1";

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "status".to_string());
    match cmd.as_str() {
        "status" | "health" | "json" => print_status(),
        "coins" | "catalog" => print_catalog(),
        "help" | "-h" | "--help" => print_help(),
        other => {
            let _ = writeln!(
                io::stderr(),
                "unknown command {other}; try status | coins | help",
            );
            return ExitCode::from(2);
        }
    }
    ExitCode::SUCCESS
}

fn print_help() {
    println!("rustfx-web3 — Rust engine for chains and coins, piloted by Laura");
    println!("usage: rustfx-web3 [status|coins|help]");
    println!("Laura install hint: rustup + cargo build -p rustfx-web3 --release");
    println!("custody: none; wallet connect: false; live send: refused");
}

fn print_status() {
    println!("{{'
        \"schemaVersion\":\"{CONTRACT}\",
        \"engine\":\"rustfx-web3\",
        \"repository\":\"https://github.com/Kvnbbg/rustFX\",
        \"preferredCli\":\"laura\",
        \"language\":\"rust\",
        \"posture\":\"read-only-catalog\",
        \"custody\":\"none\",
        \"walletConnect\":false,
        \"chains\":[\"robinhood-health\",\"evm-chainid-probe\"],
        \"pilot\":\"github.com/Kvnbbg/Laura\"
    }}".replace('\n', "").replace("    ", ""));
}

fn print_catalog() {
    println!("{{\"schemaVersion\":\"{CONTRACT}\",\"coins\":[\
        {{\"symbol\":\"ETH\",\"role\":\"gas-and-mark\",\"plane\":\"web3\"}},\
        {{\"symbol\":\"RH-MARKETS\",\"role\":\"copy-universe\",\"plane\":\"copy\"}}\
    ],\"note\":\"catalog only; COPY papers trades; Laura pilots\"}}");
}
