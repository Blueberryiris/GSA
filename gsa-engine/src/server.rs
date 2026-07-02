//! WebSocket server that exposes GSA to any client on the LAN.
//!
//! Binds to `0.0.0.0` (not `127.0.0.1`) so phones, tablets, and other
//! computers on the same network can reach the engine, not just the Mac
//! running it. The engine performs 100% of the sorting; clients only ever
//! send an unsorted array and render whatever frames come back.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;

use crate::gpu::GpuContext;
use crate::sort::{gsa_sort_tuned, SortEvent};

pub const DEFAULT_PORT: u16 = 7878;

pub struct AppState {
    pub pool: rayon::ThreadPool,
    pub gpu: Option<GpuContext>,
    /// GPU-path bucket-count multiplier: [`DEFAULT_BUCKET_MULTIPLIER`]
    /// unless `autotune::calibrate` found a faster one on this machine at
    /// startup (see `main.rs`).
    pub bucket_multiplier: f64,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessage {
    Sort { array: Vec<f32> },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMessage<'a> {
    Progress {
        indices: &'a [usize],
        values: &'a [f32],
    },
    Done {
        elapsed_ms: f64,
        elements: usize,
        algorithm: &'a str,
        threads_used: usize,
        gpu_used: bool,
        gpu_device: Option<&'a str>,
        bucket_multiplier: Option<f64>,
        sorted: &'a [f32],
    },
    Error {
        message: &'a str,
    },
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(ws_handler))
        .with_state(state)
}

/// Axum/tungstenite default to a 16 MB max frame size, which a JSON array
/// of a few million `f32`s blows through easily (text encoding costs
/// several bytes per number). GSA is explicitly meant to be pushed with
/// large arrays for stress/benchmark purposes, so raise both limits well
/// past what even a very large single-message sort request needs.
const MAX_WS_MESSAGE_BYTES: usize = 512 * 1024 * 1024;

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.max_message_size(MAX_WS_MESSAGE_BYTES)
        .max_frame_size(MAX_WS_MESSAGE_BYTES)
        .on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    while let Some(Ok(msg)) = socket.recv().await {
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };

        let parsed: Result<ClientMessage, _> = serde_json::from_str(&text);
        let ClientMessage::Sort { array } = match parsed {
            Ok(m) => m,
            Err(e) => {
                let err = ServerMessage::Error {
                    message: &format!("invalid message: {e}"),
                };
                let _ = socket
                    .send(Message::Text(serde_json::to_string(&err).unwrap()))
                    .await;
                continue;
            }
        };

        let n = array.len();
        tracing::info!(elements = n, "starting GSA sort run");

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<SortEvent>();

        // GSA runs synchronously on the rayon pool + GPU, so it needs a
        // blocking thread; results (progress + final sorted array) come
        // back over the unbounded channel as they're produced.
        let state_for_blocking = state.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let mut data = array;
            gsa_sort_tuned(
                &mut data,
                &state_for_blocking.pool,
                state_for_blocking.gpu.as_ref(),
                state_for_blocking.bucket_multiplier,
                &tx,
            );
            data
        });

        while let Some(event) = rx.recv().await {
            match event {
                SortEvent::Progress { indices, values } => {
                    let frame = ServerMessage::Progress {
                        indices: &indices,
                        values: &values,
                    };
                    if socket
                        .send(Message::Text(serde_json::to_string(&frame).unwrap()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                SortEvent::Done(stats) => {
                    let sorted = handle.await.unwrap_or_default();
                    let frame = ServerMessage::Done {
                        elapsed_ms: stats.elapsed.as_secs_f64() * 1000.0,
                        elements: stats.elements,
                        algorithm: stats.algorithm,
                        threads_used: stats.threads_used,
                        gpu_used: stats.gpu_used,
                        gpu_device: stats.gpu_device.as_deref(),
                        bucket_multiplier: stats.bucket_multiplier,
                        sorted: &sorted,
                    };
                    let _ = socket
                        .send(Message::Text(serde_json::to_string(&frame).unwrap()))
                        .await;
                    tracing::info!(
                        elements = n,
                        elapsed_ms = stats.elapsed.as_secs_f64() * 1000.0,
                        algorithm = stats.algorithm,
                        gpu_used = stats.gpu_used,
                        "GSA sort run complete"
                    );
                    break;
                }
            }
        }
    }
}

/// Best-effort discovery of the machine's LAN-facing IPv4 address (the one
/// other devices on the network would use to reach this server). Falls
/// back to `127.0.0.1` if no non-loopback interface can be found, in
/// which case only the local machine will be able to connect.
pub fn discover_lan_ip() -> std::net::Ipv4Addr {
    match local_ip_address::local_ip() {
        Ok(std::net::IpAddr::V4(ip)) => ip,
        _ => std::net::Ipv4Addr::new(127, 0, 0, 1),
    }
}

pub async fn run(state: Arc<AppState>, port: u16) {
    let app = build_router(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind GSA WebSocket server");
    axum::serve(listener, app)
        .await
        .expect("GSA server crashed");
}
