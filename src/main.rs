use colored::*;
use sysinfo::System;
use std::net::SocketAddr;
use axum::{routing::post, Router, Json};
use tower_http::cors::CorsLayer;

mod engine;
mod proof;
mod api;

#[tokio::main]
async fn main() {
    // --- 1. System Information Gathering ---
    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_gb = sys.total_memory() / 1024 / 1024 / 1024;
    let avail_gb = sys.available_memory() / 1024 / 1024 / 1024;

    // --- 2. Robust & Cyberpunk Startup Screen ---
    println!("{}", "=".repeat(60).bright_blue());

    let wqc_logo = r#"
██╗    ██╗  ██████╗   ██████╗
██║    ██║ ██╔═══██╗ ██╔════╝
██║ █╗ ██║ ██║   ██║ ██║
██║███╗██║ ██║▄▄ ██║ ██║
╚███╔███╔╝ ╚██████╔╝ ╚██████╗
 ╚══╝╚══╝   ╚══▀▀═╝   ╚═════╝ core-node
    "#;
    println!("{}", wqc_logo.bright_cyan().bold());

    println!("  {} {}", "VISION:".dimmed(), "\"We are the Computer.\"".italic().bright_magenta());
    println!("{}", "-".repeat(60).bright_blue());

    // Status Display
    println!(
        "  {}  {:12} {}",
        "●".green(), "Status:".bold(), "Online".green()
    );
    println!(
        "  {}  {:12} {} GB / {} GB (Available/Total)",
        "●".blue(), "Memory:".bold(), avail_gb, total_gb
    );
    println!(
        "  {}  {:12} {}",
        "●".magenta(), "Engine:".bold(), "Plonky3 zk-STARKs Engine Enabled".bright_green()
    );

    // Networking info
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!();
    println!("  {} {}", "➜".bright_yellow(), "API Endpoint:".bold());
    println!("    {}", format!("http://{}", addr).underline().bright_cyan());
    println!();
    println!("{}", "=".repeat(60).bright_blue());
    println!("{}", "System is ready for algebraic quantum verification (Mersenne31).".dimmed());
    println!();

    // --- 3. Start Server ---
    let app = Router::new()
        .route("/compute", post(api::handle_compute))
        .route("/verify", post(api::handle_verify))
        .route("/gates", axum::routing::get(api::get_supported_gates))
        .route("/health", axum::routing::get(|| async { Json(serde_json::json!({ "status": "UP" })) }))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
