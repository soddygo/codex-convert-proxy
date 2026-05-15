//! Codex Convert Proxy - Main Entry Point
//!
//! A proxy server that converts between OpenAI Responses API and Chat API.
//!
//! This binary requires the `binary` feature to be enabled.

use std::sync::Arc;

use codex_convert_proxy::cli::{Cli, Commands, ServerArgs};
use codex_convert_proxy::config::{BackendRouter, ProxyConfig};
use codex_convert_proxy::logger;
use codex_convert_proxy::server;

fn main() {
    let cli = Cli::parse_args();

    match cli.command {
        Commands::Server(args) => {
            if let Err(e) = run_server(args) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        #[cfg(feature = "acp")]
        Commands::Acp(args) => {
            if let Err(e) = codex_convert_proxy::acp::run_acp(args) {
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
fn run_server(args: ServerArgs) -> anyhow::Result<()> {
    codex_utils_rustls_provider::ensure_rustls_crypto_provider();

    // Create proxy config from args
    let config = args.to_proxy_config()?;

    // Initialize logging
    let log_dir = std::path::PathBuf::from(&config.log_dir);
    logger::init_logging(&log_dir, config.log_body, true)?;

    let listen = config.listen.clone();
    let router = Arc::new(BackendRouter::new(config.backends.clone())?);

    eprintln!("Starting Codex Convert Proxy");
    eprintln!("  Listen: {}", listen);
    eprintln!("  Backends:");
    for name in router.backend_names() {
        eprintln!("    - {name}");
    }
    eprintln!();

    let proxy = server::build_proxy(config)?;

    // Start the pingora server (this blocks)
    server::start_proxy_server(proxy, &listen);

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
