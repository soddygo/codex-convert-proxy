//! JSON error response helpers.

use bytes::Bytes;
use pingora_proxy::{FailToProxy, Session};
use tracing::error;

use crate::proxy::context::ProxyContext;

pub(crate) fn conversion_error_json(_code: u16, message: impl ToString) -> String {
    serde_json::json!({
        "error": {
            "type": "invalid_request_error",
            "code": "proxy_conversion_error",
            "message": message.to_string()
        }
    })
    .to_string()
}

pub(crate) fn proxy_error_json(code: u16, message: impl ToString) -> String {
    serde_json::json!({
        "error": {
            "type": "proxy_error",
            "code": code,
            "message": message.to_string()
        }
    })
    .to_string()
}

pub(crate) fn streaming_error_sse(message: impl ToString, sequence_number: u64) -> String {
    let body = serde_json::json!({
        "type": "response.error",
        "sequence_number": sequence_number,
        "error": {
            "type": "server_error",
            "code": "proxy_stream_conversion_error",
            "message": message.to_string()
        }
    });
    format!("event: response.error\ndata: {}\n\n", body)
}

pub(crate) async fn write_fail_response(
    session: &mut Session,
    e: &pingora_core::Error,
    ctx: &mut ProxyContext,
) -> FailToProxy {
    let code = match *e.etype() {
        pingora_core::ErrorType::ConnectTimedout => 504,
        pingora_core::ErrorType::ConnectRefused => 502,
        pingora_core::ErrorType::TLSHandshakeFailure => 502,
        pingora_core::ErrorType::HTTPStatus(status) => status,
        _ => 502,
    };

    let method = session.req_header().method.as_str();
    let uri = &session.req_header().uri;

    error!(
        "[FAIL] {} {} -> {} (provider: {}, model: {:?}): {}",
        method,
        uri,
        code,
        ctx.route.provider_name.as_deref().unwrap_or("unknown"),
        ctx.model.as_ref(),
        e
    );

    let error_body = if ctx.flags.is_conversion_request {
        conversion_error_json(code, e)
    } else {
        proxy_error_json(code, e)
    };
    if let Ok(mut resp) = pingora_http::ResponseHeader::build(code, None) {
        let _ = resp.insert_header("content-type", "application/json");
        let _ = resp.insert_header("content-length", error_body.len().to_string());
        let _ = session.write_response_header(Box::new(resp), false).await;
        let _ = session
            .write_response_body(Some(Bytes::from(error_body)), true)
            .await;
    }

    FailToProxy {
        error_code: code,
        can_reuse_downstream: false,
    }
}
