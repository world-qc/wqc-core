use colored::*;
use std::net::SocketAddr;
use axum::{routing::post, Router, Json};
use tower_http::cors::CorsLayer;

mod engine;
mod proof;
mod api;

#[tokio::main]
async fn main() {
    // --- 1. Networking Setup ---
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));

    // --- 2. Print Minimal & Cyber Startup Banner ---
    print_core_banner(&addr);

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

fn print_core_banner(addr: &SocketAddr) {
    println!("{}", "============================================================".bright_blue());

    // Line 1: Cube top + Title
    print!("{}", "   ▄▄▄▄▄".bright_cyan());
    println!("  {}", "wqc-core ───".bright_white().bold());

    // Line 2: Cube center (Core nucleus in magenta) + Engine details
    print!("{}", " ▄█  ".bright_cyan());
    print!("{}", "█".bright_magenta().bold()); // Core nucleus energy
    print!("{}", "  █▄".bright_cyan());
    println!("  {}", "Plonky3 zk-STARKs Engine (Mersenne31)".dimmed());

    // Line 3: Cube bottom
    println!("{}", " ▀█▄▄█▄▄█▀".bright_cyan());

    println!("{}", "============================================================".bright_blue());

    // Status & Endpoint details
    println!(
        "  {} {:8} {}",
        "🟢", "STATUS".bold(), "Online & Ready".green()
    );
    println!(
        "  {} {:8} {}",
        "🔗", "URL".bold(), format!("http://{}", addr).underline().bright_cyan()
    );

    println!("{}", "============================================================".bright_blue());
    println!();
}
