#![deny(unused_imports)]

use colored::*;
use std::net::SocketAddr;
use std::path::Path;
use axum::{routing::post, Router, Json};
use tower_http::cors::CorsLayer;
use tower::Service;
use tokio::net::{TcpListener, UnixListener};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use hyper_util::service::TowerToHyperService;

mod basis;
mod engine;
mod memory_budget;
mod proof;
mod api;
mod sample;
mod expectation;
mod mid_circuit;
mod noise;
mod distribution_proof;
mod trajectory_proof;
mod tn;

#[tokio::main]
async fn main() {
    // --- 1. Read Connection Mode from Environment Variable ---
    // Default to "uds" if the variable is not explicitly set.
    let connection_mode = std::env::var("WQC_CONNECTION_MODE")
        .unwrap_or_else(|_| "uds".to_string())
        .to_lowercase();

    // --- 2. Build the Common Axum Routing Pipeline ---
    let app = Router::new()
        .route("/compute", post(api::handle_compute))
        .route("/verify", post(api::handle_verify))
        .route("/gates", axum::routing::get(api::get_supported_gates))
        .route("/sysinfo", axum::routing::get(api::get_system_info))
        .route("/health", axum::routing::get(|| async { Json(serde_json::json!({ "status": "UP" })) }))
        .layer(CorsLayer::permissive());

    // --- 3. Dynamic Listener Binding based on Configured Mode ---
    match connection_mode.as_str() {
        "uds" => {
            let socket_path = std::env::var("WQC_SOCKET_PATH")
                .unwrap_or_else(|_| "/var/run/wqc-core.sock".to_string())
                .to_lowercase();

            // Safety: Unlink the socket path if a stale file exists from a previous crash
            if Path::new(&socket_path).exists() {
                std::fs::remove_file(socket_path.clone()).unwrap_or_else(|e| {
                    println!("Warning: Failed to clear stale socket file: {}", e);
                });
            }

            print_core_banner(connection_mode.as_str(), &socket_path);

            // Bind to POSIX system native local stream pipe
            let listener = UnixListener::bind(socket_path).unwrap();
            let mut make_service = app.into_make_service();

            loop {
                let (stream, _remote_addr) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => continue,
                };
                let tower_service = match make_service.call(()).await {
                    Ok(svc) => svc,
                    Err(_) => continue,
                };
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let hyper_service = TowerToHyperService::new(tower_service);
                    let _ = auto::Builder::new(TokioExecutor::new())
                        .serve_connection(io, hyper_service)
                        .await;
                });
            }
        }
        _ => {
            // Default Fallback: Standard TCP Network Mode
            let port = std::env::var("WQC_CORE_TCP_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(3000);
            let addr = SocketAddr::from(([0, 0, 0, 0], port));

            let url = format!("http://{}", addr);
            print_core_banner(connection_mode.as_str(), &url);

            // Bind over the standard network stack listener
            let listener = TcpListener::bind(addr).await.unwrap();
            axum::serve(listener, app).await.unwrap();
        }
    }
}

fn print_core_banner(mode: &str, addr: &str) {
    println!("{}", "============================================================".bright_blue());
    print!("{}", "   ▄▄▄▄▄".bright_cyan());
    println!("  {}", "wqc-core ───".bright_white().bold());
    print!("{}", " ▄█  ".bright_cyan());
    print!("{}", "█".bright_magenta().bold()); // Core nucleus energy
    print!("{}", "  █▄".bright_cyan());
    println!("  {}", "Plonky3 zk-STARKs Engine (Mersenne31)".dimmed());
    println!("{}", " ▀█▄▄█▄▄█▀".bright_cyan());
    println!("{}", "============================================================".bright_blue());

    match mode {
        "uds" => {
            println!("  {} {:8} {}", "🟢", "STATUS".bold(), "Online & Ready".green());
            println!("  {} {:8} {}", "⚡", "MODE".bold(), "Unix Domain Socket (High-Performance IPC)".bright_magenta());
            println!("  {} {:8} {}", "📁", "SOCKET".bold(), addr.underline().bright_cyan());
        }
        _ => {
            println!("  {} {:8} {}", "🟢", "STATUS".bold(), "Online & Ready".green());
            println!("  {} {:8} {}", "🌐", "MODE".bold(), "TCP Network Listener".bright_yellow());
            println!("  {} {:8} {}", "🔗", "ENDPOINT".bold(), addr.underline().bright_cyan());
        }
    }

    let tn = crate::tn::tn_engine_status();
    let engine_line = if tn.requested == tn.active {
        match tn.active.as_str() {
            "webgpu" => "WEBGPU MPS".bright_magenta().bold().to_string(),
            _ => "CPU MPS".bright_cyan().bold().to_string(),
        }
    } else {
        format!(
            "{} {} {}",
            tn.requested.to_ascii_uppercase().bright_yellow(),
            "→".dimmed(),
            tn.active.to_ascii_uppercase().bright_red().bold()
        )
    };
    println!(
        "  {} {:8} {}  (χ≤{})",
        "🧠".bright_magenta(),
        "TN".bold(),
        engine_line,
        tn.mps_max_bond_dim
    );
    if let Some(note) = &tn.note {
        println!("  {} {:8} {}", " ", "↳".dimmed(), note.italic().bright_black());
    }
    println!();
}
