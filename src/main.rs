mod engine;
mod proof;
mod api;

use axum::{routing::post, Router};
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() {
    println!("--- World Quantum Computer (WQC) Core Node ---");
    println!("Vision: 'We are the Computer.'");

    // 1. Setup API routes
    let app = Router::new()
        .route("/compute", post(api::handle_compute))
        // Add a health check to verify the server is alive
        .route("/health", axum::routing::get(|| async { "WQC Core is Online" }))
        .layer(CorsLayer::permissive());

    // 2. Define the address
    // 0.0.0.0 is used to allow connections from outside the Docker container
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("Listening on {}", addr);

    // 3. Start the server
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
