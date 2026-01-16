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
    admin::{AdminState, FeedEvent, TokenUsage},
    config::HangarConfig,
    is_loopback_url,
    memory::{Admission, MemoryManager, DEFAULT_MODEL_FOOTPRINT_BYTES},
    router::{self, Surface},
    runtime::RuntimeClient,
};
use async_stream::stream;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, VecDeque},
    net::{IpAddr, SocketAddr},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tracing::{error, info, warn};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Default)]
struct TokenCounters {
    input: AtomicU64,
    output: AtomicU64,
    requests: AtomicU64,
}

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
    feed: Arc<Mutex<VecDeque<FeedEvent>>>,
    seq: Arc<AtomicU64>,
    tokens: Arc<TokenCounters>,
    last_seen: Arc<Mutex<HashMap<String, Instant>>>,
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
        feed: Arc::new(Mutex::new(VecDeque::new())),
        seq: Arc::new(AtomicU64::new(0)),
        tokens: Arc::new(TokenCounters::default()),
        last_seen: Arc::new(Mutex::new(HashMap::new())),
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
        .route("/admin/state", get(admin_state))
        .route("/v1/models", get(list_models))
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
        push_feed(st, "load", format!("{model} (warm pool)"));
        if let Err(e) = st.runtime.load(model).await {
            warn!(model = %model, error = %e, "warm-pool preload failed (is it pulled?)");
        }
    }
    reconcile_now(st).await;
}

fn spawn_reconcile_loop(st: &AppState) {
    let runtime = st.runtime.clone();
    let mem = st.mem.clone();
    let admission = st.admission.clone();
    let last_seen = st.last_seen.clone();
    let feed = st.feed.clone();
    let seq = st.seq.clone();
    let idle_minutes = st.config.memory.idle_unload_minutes;

    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(15));
        loop {
            ticker.tick().await;

            // Keep the manager's view in sync with the runtime.
            if let Ok(ps) = runtime.ps().await {
                let pairs: Vec<(String, u64)> =
                    ps.into_iter().map(|m| (m.name, m.size_bytes)).collect();
                if let Ok(mut m) = mem.lock() {
                    m.reconcile(&pairs);
                }
            }

            // Unload non-pinned models idle past the threshold, freeing RAM. The
            // next request reloads them on demand via ensure_resident.
            if idle_minutes == 0 {
                continue;
            }
            let threshold = Duration::from_secs(idle_minutes * 60);
            let victims: Vec<String> = {
                let snapshot = match mem.lock() {
                    Ok(m) => m.snapshot(),
                    Err(_) => continue,
                };
                let seen = last_seen.lock().ok();
                snapshot
                    .models
                    .into_iter()
                    .filter(|m| m.resident && !m.pinned)
                    .filter(|m| {
                        seen.as_ref()
                            .and_then(|s| s.get(&m.name))
                            .map(|t| t.elapsed() >= threshold)
                            .unwrap_or(false)
                    })
                    .map(|m| m.name)
                    .collect()
            };
            if victims.is_empty() {
                continue;
            }

            let _permit = admission.lock().await;
            for victim in victims {
                let still_resident = mem.lock().map(|m| m.is_resident(&victim)).unwrap_or(false);
                if !still_resident {
                    continue;
                }
                info!(model = %victim, "idle unload");
                if runtime.unload(&victim).await.is_ok() {
                    if let Ok(mut m) = mem.lock() {
                        m.mark_unloaded(&victim);
                    }
                    let s = seq.fetch_add(1, Ordering::Relaxed);
                    if let Ok(mut f) = feed.lock() {
                        f.push_back(FeedEvent {
                            seq: s,
                            kind: "evict".into(),
                            message: format!("{victim} (idle {idle_minutes}m)"),
                        });
                        while f.len() > FEED_CAP {
                            f.pop_front();
                        }
                    }
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

async fn admin_state(State(st): State<AppState>) -> impl IntoResponse {
    let memory = st.mem.lock().unwrap().snapshot();
    let feed = st
        .feed
        .lock()
        .map(|f| f.iter().cloned().collect())
        .unwrap_or_default();
    let tokens = TokenUsage {
        input_tokens: st.tokens.input.load(Ordering::Relaxed),
        output_tokens: st.tokens.output.load(Ordering::Relaxed),
        requests: st.tokens.requests.load(Ordering::Relaxed),
    };
    Json(AdminState {
        memory,
        feed,
        tokens,
    })
}

const FEED_CAP: usize = 200;

fn push_feed(st: &AppState, kind: &str, message: String) {
    let seq = st.seq.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut f) = st.feed.lock() {
        f.push_back(FeedEvent {
            seq,
            kind: kind.to_string(),
            message,
        });
        while f.len() > FEED_CAP {
            f.pop_front();
        }
    }
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
            mark_seen(st, model);
            Ok(())
        }
        Admission::CantFit => {
            warn!(
                model,
                "cannot fit within memory ceiling - applying backpressure"
            );
            push_feed(st, "backpressure", format!("{model} exceeds ceiling"));
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
                push_feed(st, "evict", format!("{v} (LRU, to make room)"));
                if let Err(e) = st.runtime.unload(v).await {
                    warn!(victim = %v, error = %e, "unload failed");
                }
                st.mem.lock().unwrap().mark_unloaded(v);
            }
            info!(model, "loading model");
            push_feed(st, "load", model.to_string());
            if let Err(e) = st.runtime.load(model).await {
                error!(model, error = %e, "load failed");
                push_feed(st, "error", format!("load {model} failed"));
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
                push_feed(st, "evict", format!("{v} (overflow trim)"));
                if let Err(e) = st.runtime.unload(v).await {
                    warn!(victim = %v, error = %e, "unload failed");
                }
                st.mem.lock().unwrap().mark_unloaded(v);
            }
            st.mem.lock().unwrap().touch(model);
            mark_seen(st, model);
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

fn mark_seen(st: &AppState, model: &str) {
    if let Ok(mut seen) = st.last_seen.lock() {
        seen.insert(model.to_string(), Instant::now());
    }
}

fn resident_touch(st: &AppState, model: &str) -> bool {
    let resident = {
        let mut m = st.mem.lock().unwrap();
        if m.is_resident(model) {
            m.touch(model);
            true
        } else {
            false
        }
    };
    if resident {
        mark_seen(st, model);
    }
    resident
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
        return forward(&st, method, path_q, &headers, body, true).await;
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
    push_feed(
        &st,
        "routed",
        format!("{} -> {}  ({})", decision.tier, decision.model, decision.reason),
    );

    if let Err(resp) = ensure_resident(&st, &decision.model).await {
        return resp;
    }

    if let Some(obj) = body_json.as_object_mut() {
        obj.insert("model".to_string(), Value::String(decision.model));
    }

    match serde_json::to_vec(&body_json) {
        Ok(routed) => forward(&st, method, path_q, &headers, routed.into(), true).await,
        Err(e) => {
            error!(error = %e, "re-serializing routed body failed; forwarding original");
            forward(&st, method, path_q, &headers, body, true).await
        }
    }
}

async fn list_models(State(st): State<AppState>) -> impl IntoResponse {
    let mut data: Vec<Value> = st
        .config
        .routing
        .keys()
        .map(|tier| {
            json!({
                "id": tier,
                "object": "model",
                "created": 0,
                "owned_by": "hangar",
            })
        })
        .collect();

    if let Ok(models) = st.runtime.tags().await {
        for m in models {
            data.push(json!({
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": "library",
            }));
        }
    }

    Json(json!({ "object": "list", "data": data }))
}

// Send the request upstream and stream the response straight back, so SSE token
// streaming is preserved. When `meter` is set, the body is also observed as it
// streams (never buffered off the wire) to accumulate token usage at the end.
async fn forward(
    st: &AppState,
    method: Method,
    path_q: &str,
    headers: &HeaderMap,
    body: Bytes,
    meter: bool,
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
            let ct = resp.headers().get(reqwest::header::CONTENT_TYPE).cloned();

            let response_body = if meter {
                let tokens = st.tokens.clone();
                let upstream = resp.bytes_stream();
                Body::from_stream(stream! {
                    let mut upstream = upstream;
                    let mut buf: Vec<u8> = Vec::new();
                    while let Some(item) = upstream.next().await {
                        if let Ok(ref chunk) = item {
                            buf.extend_from_slice(chunk);
                        }
                        yield item;
                    }
                    if let Some((input, output)) = parse_usage(&buf) {
                        tokens.input.fetch_add(input, Ordering::Relaxed);
                        tokens.output.fetch_add(output, Ordering::Relaxed);
                        tokens.requests.fetch_add(1, Ordering::Relaxed);
                    }
                })
            } else {
                Body::from_stream(resp.bytes_stream())
            };

            let mut builder = Response::builder().status(status);
            if let Some(ct) = ct {
                builder = builder.header(header::CONTENT_TYPE, ct.as_bytes());
            }
            builder.body(response_body).unwrap_or_else(|e| {
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

// Pull prompt/completion token counts from a chat response body. Non-streaming
// bodies are one JSON object with `usage`; streamed SSE bodies carry it in a
// final `data:` chunk (only when the client requested usage).
fn parse_usage(buf: &[u8]) -> Option<(u64, u64)> {
    if let Ok(v) = serde_json::from_slice::<Value>(buf) {
        if let Some(u) = v.get("usage") {
            return extract_usage(u);
        }
    }
    let text = std::str::from_utf8(buf).ok()?;
    let mut last = None;
    for line in text.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("data:") {
            let rest = rest.trim();
            if rest == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(rest) {
                if let Some(pair) = v.get("usage").and_then(extract_usage) {
                    last = Some(pair);
                }
            }
        }
    }
    last
}

fn extract_usage(u: &Value) -> Option<(u64, u64)> {
    let input = u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
    let output = u
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if input == 0 && output == 0 {
        None
    } else {
        Some((input, output))
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
