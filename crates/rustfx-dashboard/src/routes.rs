use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use crate::db::{self, AppState};

#[derive(Debug, Deserialize)]
pub struct NewEmployee {
    name: String,
    salary: i64,
    manager_id: Option<i64>,
}

pub fn api_router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/dashboard", get(dashboard))
        .route("/api/employees", post(create_employee))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "service": "rustfx-dashboard" }))
}

async fn dashboard(State(state): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let snapshot = db::load_snapshot(&state.pool).await?;
    Ok(Json(snapshot))
}

async fn create_employee(
    State(state): State<AppState>,
    Json(body): Json<NewEmployee>,
) -> Result<impl IntoResponse, AppError> {
    if body.name.trim().is_empty() || body.salary <= 0 {
        return Err(AppError::bad_request("name and a positive salary are required"));
    }
    let id = db::insert_employee(&state.pool, body.name.trim(), body.salary, body.manager_id).await?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "id": id })),
    ))
}

struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        tracing::error!(error = %err, "dashboard error");
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let body = Json(serde_json::json!({ "error": self.message }));
        (self.status, body).into_response()
    }
}
