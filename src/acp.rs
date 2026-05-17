//! ACP mode: run Codex ACP with an embedded conversion proxy.

use std::net::TcpListener;
use std::path::Path;
use std::time::Duration;

use codex_arg0::arg0_dispatch_or_else;
use codex_utils_cli::CliConfigOverrides;
use nuwax_codex_acp::CodexRuntimeOverrides;

use crate::cli::AcpArgs;
use crate::logger;
use crate::server;

pub fn run_acp(args: AcpArgs) -> anyhow::Result<()> {
    codex_utils_rustls_provider::ensure_rustls_crypto_provider();
    if args.clean_log_dir {
        clean_acp_log_dir(&args.log_dir)?;
    }
    logger::init_logging_with_stdout(&args.log_dir, args.log_body, true, false)?;

    let proxy_listen = allocate_listen_addr(&args.proxy_listen)?;
    let codex_base_url = format!("http://{proxy_listen}/v1");
    let proxy_config = args.to_proxy_config(proxy_listen.clone());
    let runtime_overrides = CodexRuntimeOverrides {
        model: Some(args.model.clone()),
        base_url: Some(codex_base_url.clone()),
        api_key: Some(args.api_key.clone()),
        provider_id: args.provider.clone(),
        provider_name: args.provider.clone(),
        model_context_window: Some(args.context_window),
        personality_enabled: args.personality.as_runtime_override(),
        wire_api: Some("chat".to_string()),
    };

    eprintln!("Starting Codex ACP with embedded conversion proxy");
    eprintln!("  Provider: {}", args.provider.as_deref().unwrap_or("(auto)"));
    eprintln!("  Model: {}", args.model);
    eprintln!("  Upstream: {}", args.base_url);
    eprintln!("  Embedded proxy: http://{proxy_listen}");
    eprintln!("  Personality: {:?}", args.personality);

    arg0_dispatch_or_else(|arg0_paths| async move {
        let proxy = server::build_proxy(proxy_config)?;
        std::thread::Builder::new()
            .name("codex-convert-proxy-acp".to_string())
            .spawn(move || {
                server::start_proxy_server(proxy, &proxy_listen);
            })?;

        wait_for_embedded_proxy(&codex_base_url).await?;

        nuwax_codex_acp::run_main_with_runtime_overrides(
            arg0_paths.codex_linux_sandbox_exe.clone(),
            CliConfigOverrides::default(),
            runtime_overrides,
        )
        .await?;
        Ok(())
    })
}

fn clean_acp_log_dir(log_dir: &Path) -> anyhow::Result<()> {
    if !log_dir.exists() {
        return Ok(());
    }

    for entry in std::fs::read_dir(log_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with("proxy.log.") || file_name.ends_with(".converted_request.json") {
            std::fs::remove_file(&path)?;
        }
    }

    Ok(())
}

fn allocate_listen_addr(requested: &str) -> anyhow::Result<String> {
    let listener = TcpListener::bind(requested)?;
    let addr = listener.local_addr()?;
    if !addr.ip().is_loopback() {
        anyhow::bail!("ACP embedded proxy must listen on loopback, got {addr}");
    }
    drop(listener);
    Ok(addr.to_string())
}

async fn wait_for_embedded_proxy(codex_base_url: &str) -> anyhow::Result<()> {
    let url = url::Url::parse(codex_base_url)?;
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("embedded proxy URL is missing host"))?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| anyhow::anyhow!("embedded proxy URL is missing port"))?;
    let addr = format!("{host}:{port}");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    loop {
        match tokio::net::TcpStream::connect(&addr).await {
            Ok(_) => return Ok(()),
            Err(err) if tokio::time::Instant::now() < deadline => {
                tracing::debug!("waiting for embedded proxy: {err}");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Err(err) => return Err(err.into()),
        }
    }
}
