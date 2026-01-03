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
    memory::{Admission, MemoryManager, DEFAULT_MODEL_FOOTPRINT_BYTES},
    router::{self, Surface},
    runtime::RuntimeClient,
};
use serde_json::{json, Value};
use std::{
    net::{IpAddr, SocketAddr},
    sync::{Arc, Mutex},
    time::Duration,
};
use tracing::{error, info, warn};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Parser)]
#[command(
    name = "hangard",
    version,
    about = "Hangar daemon - offline chat-completions proxy with routing and memory management"
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
    runtime: Arc<RuntimeClient>,
    mem: Arc<Mutex<MemoryManager>>,
    // Serializes load/evict sequences so concurrent requests can never push the
    // runtime past the ceiling. Resident models skip it via the fast path.
    admission: Arc<tokio::sync::Mutex<()>>,
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

    let ceiling_bytes = match cfg.memory.ceiling_gb {
        Some(g) if g > 0.0 => (g * BYTES_PER_GB) as u64,
        _ => {
            warn!("no memory.ceiling_gb set - memory management disabled (no ceiling)");
            u64::MAX
        }
    };

    let state = AppState {
        http: reqwest::Client::new(),
        upstream: upstream.clone(),
        config: cfg.clone(),
        runtime: Arc::new(RuntimeClient::new(&upstream)),
        mem: Arc::new(Mutex::new(MemoryManager::new(
            ceiling_bytes,
            &cfg.memory.warm_pool,
        ))),
        admission: Arc::new(tokio::sync::Mutex::new(())),
    };

    if probe_upstream(&state.http, &state.upstream).await {
        info!(upstream = %state.upstream, "runtime reachable");
        preload_warm_pool(&state).await;
    } else {
        warn!(upstream = %state.upstream, "runtime NOT reachable yet - start it with `ollama serve`");
    }

    spawn_reconcile_loop(&state);

    let addr = SocketAddr::new(bind_ip, cfg.server.port);
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/admin/memory", get(mem_state))
        .route("/v1/models", get(passthrough))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(chat))
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;

    info!(
        %addr,
        upstream = %upstream,
        offline = cfg.runtime.offline,
        ceiling_gb = cfg.memory.ceiling_gb.unwrap_or(0.0),
        "hangard {VERSION} listening"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;

    info!("hangard shutting down");
    if state.config.memory.unload_on_exit {
        unload_all(&state).await;
    }
    Ok(())
}

// Unload everything resident, so models loaded with keep_alive = -1 don't stay
// in the runtime after we exit.
async fn unload_all(st: &AppState) {
    let resident: Vec<String> = {
        let m = st.mem.lock().unwrap();
        m.snapshot()
            .models
            .into_iter()
            .filter(|x| x.resident)
            .map(|x| x.name)
            .collect()
    };
    for model in resident {
        info!(model = %model, "unloading on shutdown");
        if let Err(e) = st.runtime.unload(&model).await {
            warn!(model = %model, error = %e, "shutdown unload failed");
        }
    }
}

async fn preload_warm_pool(st: &AppState) {
    for model in &st.config.memory.warm_pool {
        info!(model = %model, "preloading warm-pool model");
        if let Err(e) = st.runtime.load(model).await {
            warn!(model = %model, error = %e, "warm-pool preload failed (is it pulled?)");
        }
    }
    reconcile_now(st).await;
}

fn spawn_reconcile_loop(st: &AppState) {
    let runtime = st.runtime.clone();
    let mem = st.mem.clone();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(5));
        loop {
            ticker.tick().await;
            if let Ok(ps) = runtime.ps().await {
                let pairs: Vec<(String, u64)> =
                    ps.into_iter().map(|m| (m.name, m.size_bytes)).collect();
                if let Ok(mut m) = mem.lock() {
                    m.reconcile(&pairs);
                }
            }
        }
    });
}

async fn reconcile_now(st: &AppState) {
    if let Ok(ps) = st.runtime.ps().await {
        let pairs: Vec<(String, u64)> = ps.into_iter().map(|m| (m.name, m.size_bytes)).collect();
        if let Ok(mut m) = st.mem.lock() {
            m.reconcile(&pairs);
        }
    }
}

async fn healthz(State(st): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "service": "hangard",
        "version": VERSION,
        "upstream": st.upstream,
    }))
}

async fn mem_state(State(st): State<AppState>) -> impl IntoResponse {
    let snapshot = st.mem.lock().map(|m| m.snapshot()).ok();
    Json(snapshot)
}

// Ensure `model` is resident within the ceiling before serving, evicting LRU
// non-pinned models if needed. Returns an error Response to apply backpressure.
#[allow(clippy::result_large_err)]
async fn ensure_resident(st: &AppState, model: &str) -> Result<(), Response> {
    if resident_touch(st, model) {
        return Ok(());
    }

    let _permit = st.admission.lock().await;
    // Another request may have loaded it while we waited for the permit.
    if resident_touch(st, model) {
        return Ok(());
    }

    let estimate = footprint_estimate(st, model).await;
    let plan = st.mem.lock().unwrap().plan_admission(model, estimate);

    match plan {
        Admission::Serve => {
            st.mem.lock().unwrap().touch(model);
            Ok(())
        }
        Admission::CantFit => {
            warn!(
                model,
                "cannot fit within memory ceiling - applying backpressure"
            );
            Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": {
                    "message": format!("model '{model}' does not fit within the memory ceiling"),
                    "type": "memory_backpressure",
                }})),
            )
                .into_response())
        }
        Admission::Load { victims } => {
            for v in &victims {
                info!(victim = %v, "evicting LRU model to make room");
                if let Err(e) = st.runtime.unload(v).await {
                    warn!(victim = %v, error = %e, "unload failed");
                }
                st.mem.lock().unwrap().mark_unloaded(v);
            }
            info!(model, "loading model");
            if let Err(e) = st.runtime.load(model).await {
                error!(model, error = %e, "load failed");
                return Err((
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": {
                        "message": format!("failed to load model '{model}': {e}"),
                        "type": "model_load_failed",
                    }})),
                )
                    .into_response());
            }
            reconcile_now(st).await;
            // The real footprint (from /api/ps) may exceed the pre-load
            // estimate; trim back under the ceiling if so, sparing the model
            // we just loaded.
            let overflow = st.mem.lock().unwrap().overflow_victims(model);
            for v in &overflow {
                info!(victim = %v, "evicting to restore ceiling after load");
                if let Err(e) = st.runtime.unload(v).await {
                    warn!(victim = %v, error = %e, "unload failed");
                }
                st.mem.lock().unwrap().mark_unloaded(v);
            }
            st.mem.lock().unwrap().touch(model);
            Ok(())
        }
    }
}

// Footprint estimate for admission: learned resident size if known, else the
// model's on-disk size from the runtime, else a conservative default.
async fn footprint_estimate(st: &AppState, model: &str) -> u64 {
    if let Some(f) = st.mem.lock().unwrap().learned_footprint(model) {
        return f;
    }
    if let Ok(tags) = st.runtime.tags().await {
        if let Some(found) = tags.iter().find(|m| m.name == model) {
            return found.size_bytes;
        }
    }
    DEFAULT_MODEL_FOOTPRINT_BYTES
}

fn resident_touch(st: &AppState, model: &str) -> bool {
    let mut m = st.mem.lock().unwrap();
    if m.is_resident(model) {
        m.touch(model);
        true
    } else {
        false
    }
}

// Classify, ensure the routed model is resident, rewrite the body's model, proxy.
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

    if let Err(resp) = ensure_resident(&st, &decision.model).await {
        return resp;
    }

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

// Proxy without routing or admission (e.g. GET /v1/models).
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
                Json(json!({"error": {
                    "message": format!("upstream request failed: {e}"),
                    "type": "upstream_unreachable",
                }})),
            )
                .into_response()
        }
    }
}

// Rough token estimate from request text (chars / 4). Used only to decide tier
// escalation; real usage counts come back in the response.
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
