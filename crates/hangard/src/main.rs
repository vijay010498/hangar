//! `hangard` - the Hangar daemon.

use anyhow::{bail, Context, Result};
use axum::{
    body::{Body, Bytes},
    extract::{OriginalUri, State},
    http::{header, HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use hangar_core::{
    config::HangarConfig,
    is_loopback_url,
    router::{self, Surface},
};
use serde_json::Value;
use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::Duration,
};
use tracing::{error, info, warn};

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(
    name = "hangard",
    version,
    about = "Hangar daemon - offline chat-completions proxy with task routing"
)]
struct Cli {
    #[arg(long, default_value = "hangar.toml")]
    config: String,
}

#[derive(Clone)]
struct AppState {
    http: reqwest::Client,
    upstream: String,
    config: Arc<HangarConfig>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let cli = Cli::parse();
    let cfg = Arc::new(
        HangarConfig::load_or_default(&cli.config)
            .with_context(|| format!("loading config from {}", cli.config))?,
    );

    let upstream = cfg.runtime.ollama_url.trim_end_matches('/').to_string();

    let bind_ip: IpAddr = cfg
        .server
        .host
        .parse()
        .with_context(|| format!("parsing server.host '{}'", cfg.server.host))?;

    if cfg.runtime.offline {
        if !bind_ip.is_loopback() {
            bail!(
                "flight mode is on but server.host '{}' is not loopback - refusing to bind off-loopback",
                cfg.server.host
            );
        }
        if !is_loopback_url(&upstream) {
            bail!(
                "flight mode is on but runtime.ollama_url '{}' is not loopback - refusing a non-local upstream",
                upstream
            );
        }
    } else if !bind_ip.is_loopback() {
        warn!(host = %cfg.server.host, "binding to a non-loopback address with flight mode OFF");
    }

    let addr = SocketAddr::new(bind_ip, cfg.server.port);
    let state = AppState {
        http: reqwest::Client::new(),
        upstream: upstream.clone(),
        config: cfg.clone(),
    };

    if probe_upstream(&state.http, &state.upstream).await {
        info!(upstream = %state.upstream, "runtime reachable");
    } else {
        warn!(upstream = %state.upstream, "runtime NOT reachable yet - start it with `ollama serve`");
    }

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(passthrough))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(chat))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;

    info!(
        %addr,
        upstream = %upstream,
        offline = cfg.runtime.offline,
        backend = %cfg.runtime.backend,
        "hangard {VERSION} listening"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;

    info!("hangard shutting down");
    Ok(())
}

async fn healthz(State(st): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "hangard",
        "version": VERSION,
        "upstream": st.upstream,
    }))
}

// Classify the request, rewrite its `model` to the routed model, then proxy.
async fn chat(
    State(st): State<AppState>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let path = uri.path();
    let surface = if path.ends_with("/chat/completions") {
        Surface::ChatCompletions
    } else {
        Surface::Completions
    };
    let path_q = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or(path);

    let Ok(mut body_json) = serde_json::from_slice::<Value>(&body) else {
        warn!("request body is not JSON; forwarding unrouted");
        return forward(&st, method, path_q, &headers, body).await;
    };

    let requested = body_json
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let has_tools = body_json
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    let approx_tokens = estimate_tokens(&body_json, surface);

    let decision = router::route(
        &st.config,
        surface,
        requested.as_deref(),
        has_tools,
        approx_tokens,
    );

    info!(
        tier = %decision.tier,
        model = %decision.model,
        reason = %decision.reason,
        approx_tokens,
        "routed"
    );

    if let Some(obj) = body_json.as_object_mut() {
        obj.insert("model".to_string(), Value::String(decision.model));
    }

    match serde_json::to_vec(&body_json) {
        Ok(routed) => forward(&st, method, path_q, &headers, routed.into()).await,
        Err(e) => {
            error!(error = %e, "re-serializing routed body failed; forwarding original");
            forward(&st, method, path_q, &headers, body).await
        }
    }
}

// Proxy without routing (e.g. GET /v1/models).
async fn passthrough(
    State(st): State<AppState>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let path_q = uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or_else(|| uri.path());
    forward(&st, method, path_q, &headers, body).await
}

// Send the request upstream and stream the response straight back, so SSE token
// streaming is preserved unchanged.
async fn forward(
    st: &AppState,
    method: Method,
    path_q: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Response {
    let url = format!("{}{}", st.upstream, path_q);
    let mut req = st.http.request(method, &url);
    for name in [header::CONTENT_TYPE, header::ACCEPT] {
        if let Some(v) = headers.get(&name) {
            req = req.header(name, v);
        }
    }
    if !body.is_empty() {
        req = req.body(body);
    }

    match req.send().await {
        Ok(resp) => {
            let status =
                StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let mut builder = Response::builder().status(status);
            if let Some(ct) = resp.headers().get(reqwest::header::CONTENT_TYPE) {
                builder = builder.header(header::CONTENT_TYPE, ct.as_bytes());
            }
            builder
                .body(Body::from_stream(resp.bytes_stream()))
                .unwrap_or_else(|e| {
                    error!(error = %e, "failed to build proxied response");
                    StatusCode::BAD_GATEWAY.into_response()
                })
        }
        Err(e) => {
            error!(error = %e, url = %url, "upstream request failed");
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("upstream request failed: {e}"),
                        "type": "upstream_unreachable",
                    }
                })),
            )
                .into_response()
        }
    }
}

// Rough token estimate from request text (chars / 4). Used only to decide tier
// escalation; the real usage counts come back in the response.
fn estimate_tokens(body_json: &Value, surface: Surface) -> usize {
    let chars = match surface {
        Surface::Completions => body_json
            .get("prompt")
            .and_then(Value::as_str)
            .map(str::len)
            .unwrap_or(0),
        Surface::ChatCompletions => body_json
            .get("messages")
            .and_then(Value::as_array)
            .map(|msgs| {
                msgs.iter()
                    .filter_map(|m| m.get("content").and_then(Value::as_str))
                    .map(str::len)
                    .sum()
            })
            .unwrap_or(0),
    };
    chars / 4
}

async fn probe_upstream(client: &reqwest::Client, base: &str) -> bool {
    client
        .get(format!("{base}/api/version"))
        .timeout(Duration::from_millis(800))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}