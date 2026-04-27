//! Pingora server startup and configuration.
//!
//! This module handles the initialization and startup of the pingora proxy server.

use pingora_core::server::configuration::ServerConf;
use pingora_core::server::Server;
use pingora_proxy::http_proxy_service;
use tracing::info;

use crate::proxy::CodexProxy;

/// Graceful shutdown timeout in seconds (wait for existing requests to complete)
const GRACEFUL_SHUTDOWN_TIMEOUT: u64 = 30;

/// Grace period before starting final shutdown step
const GRACEFUL_PERIOD: u64 = 10;

/// Start the pingora proxy server.
///
/// This function bootstraps the pingora server, adds the CodexProxy service,
/// and runs it forever. The server handles graceful shutdown automatically
/// (SIGTERM for graceful, SIGINT for fast shutdown).
pub fn start_proxy_server(
    proxy: CodexProxy,
    listen: &str,
) {
    // Create server configuration with custom graceful shutdown settings
    let mut server_conf = ServerConf::new().expect("Failed to create server config");
    server_conf.grace_period_seconds = Some(GRACEFUL_PERIOD);
    server_conf.graceful_shutdown_timeout_seconds = Some(GRACEFUL_SHUTDOWN_TIMEOUT);

    let mut server = Server::new_with_opt_and_conf(None, server_conf);

    let mut http_proxy = http_proxy_service(&server.configuration, proxy);
    http_proxy.add_tcp(listen);

    server.add_service(http_proxy);

    info!("Server listening on {}", listen);
    info!("Graceful shutdown: {}s grace period, {}s timeout",
          GRACEFUL_PERIOD, GRACEFUL_SHUTDOWN_TIMEOUT);
    server.run_forever();
}
