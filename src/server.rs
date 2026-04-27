//! Pingora server startup and configuration.
//!
//! This module handles the initialization and startup of the pingora proxy server.

use pingora_core::server::configuration::Opt;
use pingora_core::server::Server;
use pingora_proxy::http_proxy_service;
use tracing::info;

use crate::proxy::CodexProxy;

/// Start the pingora proxy server.
///
/// This function bootstraps the pingora server, adds the CodexProxy service,
/// and runs it forever. The server handles graceful shutdown automatically
/// (SIGTERM for graceful, SIGINT for fast shutdown).
pub fn start_proxy_server(
    proxy: CodexProxy,
    listen: &str,
) {
    let opt = Opt::default();
    let mut server = Server::new(Some(opt)).expect("Failed to create server");
    server.bootstrap();

    let mut http_proxy = http_proxy_service(&server.configuration, proxy);
    http_proxy.add_tcp(listen);

    server.add_service(http_proxy);

    info!("Server listening on {}", listen);
    server.run_forever();
}
