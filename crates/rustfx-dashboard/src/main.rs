//! rustFX payroll dashboard.
//!
//! Async patterns exercised here:
//! - `#[tokio::main]` multi-thread runtime
//! - shared `SqlitePool` cloned into Axum state
//! - `tokio::try_join!` for concurrent read queries
//! - sqlx query execution without compile-time `query!` macros
//!   so the crate builds without a live `DATABASE_URL`

mod db;
mod routes;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context;
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use crate::db::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite://rustfx-payroll.db?mode=rwc".to_string());
    let bind = std::env::var("RUSTFX_DASHBOARD_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());

    let pool = db::connect_and_seed(&database_url).await?;
    let state = AppState { pool };

    let ui_dir = resolve_ui_dir();
    tracing::info!(path = %ui_dir.display(), "serving static UI");

    let app = Router::new()
        .merge(routes::api_router(state))
        .fallback_service(ServeDir::new(ui_dir))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = bind.parse().context("invalid RUSTFX_DASHBOARD_BIND")?;
    tracing::info!(%addr, "payroll dashboard listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn resolve_ui_dir() -> PathBuf {
    let candidates = [
        PathBuf::from("ui"),
        PathBuf::from("../../ui"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ui"),
    ];
    candidates
        .into_iter()
        .find(|path| path.join("index.html").exists())
        .unwrap_or_else(|| PathBuf::from("ui"))
}
