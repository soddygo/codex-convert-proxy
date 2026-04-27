//! Codex Convert Proxy - Main Entry Point
//!
//! A proxy server that converts between OpenAI Responses API and Chat API.

use std::sync::Arc;

use codex_convert_proxy::cli::{Cli, Commands, StartArgs};
use codex_convert_proxy::config::{BackendRouter, ProxyConfig};
use codex_convert_proxy::logger;
use codex_convert_proxy::proxy::CodexProxy;
use codex_convert_proxy::server;

fn main() {
    let cli = Cli::parse_args();

    match cli.command {
        Commands::Start(args) => {
            if let Err(e) = run_proxy(args) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Init(args) => {
            if let Err(e) = generate_config(&args.output) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// Run the proxy server.
fn run_proxy(args: StartArgs) -> anyhow::Result<()> {
    // Create proxy config from args
    let config = args.to_proxy_config()?;

    // Initialize logging
    let log_dir = std::path::PathBuf::from(&config.log_dir);
    logger::init_logging(&log_dir, config.log_body, true)?;

    // Create backend router
    if config.backends.is_empty() {
        anyhow::bail!("No backend configured");
    }

    let router = Arc::new(BackendRouter::new(config.backends.clone())?);
    let listen = &config.listen;

    eprintln!("Starting Codex Convert Proxy");
    eprintln!("  Listen: {}", listen);
    eprintln!("  Backends:");
    for name in router.backend_names() {
        eprintln!("    - {}", name);
    }
    eprintln!();

    // Create CodexProxy
    let proxy = CodexProxy::new(router, config.log_body);

    // Start the pingora server (this blocks)
    server::start_proxy_server(proxy, listen);

    Ok(())
}

/// Generate a config file template.
fn generate_config(output: &std::path::Path) -> anyhow::Result<()> {
    let config = ProxyConfig::default();

    let json = serde_json::to_string_pretty(&config)?;
    std::fs::write(output, json)?;

    println!("Config file generated: {}", output.display());
    println!();
    println!("Edit the config file to add your backends.");

    Ok(())
}
