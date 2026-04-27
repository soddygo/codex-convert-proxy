//! Codex Convert Proxy - Main Entry Point
//!
//! A proxy server that converts between OpenAI Responses API and Chat API.

use std::sync::Arc;

use codex_convert_proxy::cli::{Cli, Commands, StartArgs};
use codex_convert_proxy::config::ProxyConfig;
use codex_convert_proxy::logger;
use codex_convert_proxy::providers::create_provider;
use codex_convert_proxy::proxy::CodexProxy;

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

    // Create provider
    if config.backends.is_empty() {
        anyhow::bail!("No backend configured");
    }

    let backend = &config.backends[0];
    let provider = create_provider(&backend.name)?;
    let provider_name = provider.name();

    eprintln!("Starting Codex Convert Proxy");
    eprintln!("  Provider: {}", provider_name);
    eprintln!("  Upstream: {}", backend.url);
    eprintln!("  Listen: {}", config.listen);
    eprintln!();

    // Create CodexProxy (convert Box<dyn Provider> to Arc<dyn Provider>)
    let _proxy = CodexProxy::new(
        Arc::from(provider),
        &backend.url,
        &backend.api_key,
        config.log_body,
    );

    // Note: Starting the actual pingora server requires more setup
    // This is a placeholder that demonstrates the proxy creation
    eprintln!("Proxy configuration complete.");
    eprintln!("To start the full server, use the pingora integration.");
    eprintln!();
    eprintln!("Example usage:");
    eprintln!("  # With config file:");
    eprintln!("  codex-convert-proxy start --config config.json");
    eprintln!();
    eprintln!("  # With CLI args:");
    eprintln!("  codex-convert-proxy start --provider glm --upstream-url https://api.example.com --api-key YOUR_KEY --listen 0.0.0.0:8080");

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
