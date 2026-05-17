//! Proxy routing and upstream request helpers.

use std::time::Duration;

use pingora_core::protocols::{Digest, TcpKeepalive};
use pingora_core::upstreams::peer::{ALPN, HttpPeer, Peer};
use pingora_core::Result as PingoraResult;
use pingora_http::RequestHeader;
use pingora_proxy::Session;
use tracing::{debug, error, info, warn};

use crate::config::BackendRouter;
use crate::proxy::context::ProxyContext;
use crate::proxy::core::CodexProxy;

impl CodexProxy {
    pub(crate) async fn handle_request_filter(
        &self,
        session: &mut Session,
        ctx: &mut ProxyContext,
    ) -> PingoraResult<bool> {
        let method = session.req_header().method.as_str().to_string();
        let path = session.req_header().uri.path().to_string();
        let query = session.req_header().uri.query().unwrap_or("");

        debug!("[REQUEST_FILTER] ★★★ RECEIVED REQUEST ★★★");
        debug!("[REQUEST_FILTER] Method: {}, Path: {}, Query: {}", method, path, query);

        let headers: Vec<(String, String)> = session
            .req_header()
            .headers
            .iter()
            .map(|(name, value)| {
                (
                    name.as_str().to_string(),
                    value.to_str().unwrap_or("<binary>").to_string(),
                )
            })
            .collect();

        for (name, value) in &headers {
            debug!("[REQUEST_FILTER]   Header: {}: {}", name, value);
        }

        debug!("[REQUEST_FILTER] ★★★ END REQUEST HEADER ★★★");

        let (backend_config, backend) = match self.router.select_with_config(&path, &headers) {
            Some(pair) => pair,
            None => {
                warn!("[REQUEST] No matching backend for path: {}", path);
                return Err(pingora_core::Error::new_str("No matching backend"));
            }
        };

        let normalized_path = if let Some(prefix) = backend_config.match_rules.path_prefix.as_deref()
        {
            BackendRouter::strip_path_prefix(&path, prefix).unwrap_or_else(|| path.clone())
        } else {
            path.clone()
        };
        ctx.route.normalized_path = Some(normalized_path.clone());

        let is_conversion = !self.no_convert
            && (normalized_path.starts_with("/v1/responses")
                || normalized_path.starts_with("/responses"))
            && method == "POST";
        ctx.flags.is_conversion_request = is_conversion;

        if is_conversion {
            debug!(
                "[REQUEST] {} {} -> conversion (CONVERSION)",
                method, normalized_path
            );
        }

        ctx.route.selected_backend = Some(backend.clone());
        ctx.route.provider_name = Some(backend.name.clone());

        debug!("[REQUEST] {} {} -> {}", method, normalized_path, backend.name);

        Ok(false)
    }

    pub(crate) async fn select_upstream_peer(
        &self,
        ctx: &mut ProxyContext,
    ) -> PingoraResult<Box<HttpPeer>> {
        let backend = ctx.route.selected_backend.as_ref().ok_or_else(|| {
            error!("No backend selected");
            pingora_core::Error::new_str("No backend selected")
        })?;

        let mut peer = HttpPeer::new(
            (backend.host.as_str(), backend.port),
            backend.use_tls,
            backend.host.clone(),
        );

        peer.options.alpn = ALPN::H2H1;
        peer.options.connection_timeout = Some(Duration::from_secs(10));
        peer.options.total_connection_timeout = Some(Duration::from_secs(30));
        peer.options.idle_timeout = Some(Duration::from_secs(90));
        peer.options.tcp_keepalive = Some(TcpKeepalive {
            idle: Duration::from_secs(60),
            interval: Duration::from_secs(5),
            count: 5,
        });

        if backend.use_tls {
            peer.options.h2_ping_interval = Some(Duration::from_secs(30));
        }

        Ok(Box::new(peer))
    }

    pub(crate) async fn rewrite_upstream_request(
        &self,
        session: &mut Session,
        upstream_request: &mut RequestHeader,
        ctx: &mut ProxyContext,
    ) -> PingoraResult<()> {
        let backend = ctx.route.selected_backend.as_ref().ok_or_else(|| {
            error!("No backend selected");
            pingora_core::Error::new_str("No backend selected")
        })?;

        let original_uri = session.req_header().uri.clone();
        let path = original_uri.path().to_string();
        let query = original_uri.query();
        let normalized_path = ctx.route.normalized_path.as_deref().unwrap_or(path.as_str());

        let is_conversion_request = (normalized_path.starts_with("/v1/responses")
            || normalized_path.starts_with("/responses"))
            && upstream_request.method.as_str() == "POST";

        let chat_api_path = if is_conversion_request
            || normalized_path.starts_with("/v1/chat/completions")
        {
            self.get_provider(&backend.name)
                .map(|provider| provider.chat_completions_path())
                .unwrap_or_else(|| "/chat/completions".to_string())
        } else {
            normalized_path.to_string()
        };

        let new_path = if !backend.base_path.is_empty() {
            format!("{}{}", backend.base_path, chat_api_path)
        } else {
            chat_api_path
        };

        let new_uri_str = if let Some(q) = query {
            format!("{}?{}", new_path, q)
        } else {
            new_path.clone()
        };

        let new_uri: http::Uri = new_uri_str.parse().map_err(|e| {
            error!("URI rewrite failed: {}", e);
            pingora_core::Error::new_str("URI rewrite failed")
        })?;
        upstream_request.set_uri(new_uri);
        ctx.route.rewritten_path = Some(new_path);

        upstream_request.remove_header("x-api-key");
        upstream_request.remove_header("authorization");

        if is_conversion_request {
            upstream_request.remove_header("content-length");
            debug!("[UPSTREAM] Removed content-length for body transformation");
        }

        upstream_request
            .insert_header("authorization", format!("Bearer {}", backend.api_key))
            .map_err(|e| {
                error!("Failed to inject authorization header: {}", e);
                pingora_core::Error::new_str("Header injection failed")
            })?;

        upstream_request
            .insert_header("host", &backend.host)
            .map_err(|e| {
                error!("Failed to set host header: {}", e);
                pingora_core::Error::new_str("Host header failed")
            })?;

        debug!(
            "[UPSTREAM] {} {} -> {}",
            upstream_request.method.as_str(),
            upstream_request.uri,
            backend.name
        );

        Ok(())
    }

    pub(crate) async fn log_connected_to_upstream(
        &self,
        reused: bool,
        peer: &HttpPeer,
        digest: Option<&Digest>,
        ctx: &mut ProxyContext,
    ) -> PingoraResult<()> {
        let tls_version = digest
            .and_then(|d| d.ssl_digest.as_ref())
            .map(|ssl| ssl.version.to_string())
            .unwrap_or_else(|| "none".to_string());

        let use_tls = ctx
            .route
            .selected_backend
            .as_ref()
            .map(|b| b.use_tls)
            .unwrap_or(false);
        let backend_name = ctx.route.provider_name.as_deref().unwrap_or("unknown");

        info!(
            "[CONNECT] {} -> {} (backend={}, TLS={}, reused={}, tls_version={})",
            peer.sni(),
            peer.address(),
            backend_name,
            use_tls,
            reused,
            tls_version
        );

        Ok(())
    }
}
