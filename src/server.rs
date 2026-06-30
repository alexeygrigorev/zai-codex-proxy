use axum::body::Body;
use axum::body::to_bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::Response;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;
use std::io;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tower_http::cors::{Any, CorsLayer};
use tracing::warn;

use crate::codex_wire::normalize::normalize_responses_request;
use crate::codex_wire::validate::validate_responses_request;

use crate::access::{AuthenticatedKey, require_admin};
use crate::codex_wire::schema::responses_wire::{
    ChatContent, ChatMessage, ChatRequest, ResponsesRequest,
};
use crate::config::{PersistedConfig, with_config, with_config_mut};
use crate::error::{ProviderError, ProviderErrorKind, ProxyError};
use crate::providers::zai::ZaiExecutionContext;
use crate::state::AppState;

pub fn build_router(state: AppState) -> Router {
    initialize_runtime_state(&state);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            header::HeaderName::from_static("x-api-key"),
            header::HeaderName::from_static("x-codex-proxy-key"),
            header::HeaderName::from_static("x-codex-proxy-session"),
        ]);

    Router::new()
        .route("/health", get(health_handler))
        .route("/favicon.ico", get(favicon_handler))
        .route("/v1/models", get(models_handler))
        .route("/models", get(models_handler))
        .route("/api/config", get(api_config_get).post(api_config_put))
        .route("/api/accounts", get(api_accounts_get))
        .route(
            "/api/access-keys",
            get(api_access_keys_get).post(api_access_keys_create),
        )
        .route("/api/access-keys/{id}", delete(api_access_keys_delete))
        .route("/api/usage/keys", get(api_usage_keys_get))
        .route("/api/usage/accounts", get(api_usage_accounts_get))
        .route("/api/usage/series", get(api_usage_series_get))
        .route("/v1/responses", post(responses_handler))
        .route("/responses", post(responses_handler))
        .layer(cors)
        .with_state(state)
}

fn initialize_runtime_state(state: &AppState) {
    let (health, accounts) = with_config(state.config(), |cfg| {
        (
            cfg.health.clone(),
            cfg.accounts.clone().into_iter().map(Into::into).collect(),
        )
    });
    state.accounts().configure_health(health);
    state.accounts().load_accounts(accounts);
    start_recovery_probe_loop(state);
}

fn start_recovery_probe_loop(state: &AppState) {
    if state.recovery_started_flag().swap(true, Ordering::AcqRel) {
        return;
    }

    let state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
            run_recovery_probe_pass(&state).await;
        }
    });
}

async fn run_recovery_probe_pass(state: &AppState) {
    for index in state.accounts().recovery_candidates() {
        if !state.accounts().begin_recovery_probe(index) {
            continue;
        }

        let Some((account, snapshot)) = state.accounts().get_account(index) else {
            state.accounts().finish_recovery_probe(index, false, None);
            continue;
        };

        let Some(target) = with_config(state.config(), |cfg| cfg.recovery_probe_target()) else {
            state.accounts().finish_recovery_probe(index, false, None);
            if !snapshot.alive {
                warn!(
                    "Recovery probe skipped for account index {} because no probe target is configured",
                    index
                );
            }
            continue;
        };

        let account_id = account.id.clone();
        let route = crate::account_pool::ResolvedRoute {
            requested_model: target.model.clone(),
            logical_model: target.model.clone(),
            upstream_model: target.model.clone(),
            account_index: index,
            account_id: account.id.clone(),
            cache_hit: false,
            cache_key: 0,
            preferred_target_index: 0,
            reasoning: Some(crate::config::EffectiveReasoningConfig {
                budget: 0,
                level: "LOW".into(),
                preset: Some("none".into()),
            }),
        };

        let normalized = crate::codex_wire::schema::responses_wire::ChatRequest {
            model: target.model.clone(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(ChatContent::Text("health check".into())),
                reasoning_content: None,
                thought_signature: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            }],
            tools: Vec::new(),
            tool_choice: Some("auto".to_string()),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: false,
            store: false,
            metadata: Default::default(),
            previous_response_id: None,
            include: Vec::new(),
        };
        let context = ZaiExecutionContext {
            route,
            account,
            config: state.config().clone(),
        };
        let result = state
            .zai()
            .handle_request(normalized, HeaderMap::new(), context)
            .await;
        let (success, error_reason) = match result {
            Ok(_) => (true, None),
            Err(err) => {
                let reason = format_proxy_error(&err);
                warn!(
                    "Recovery probe request failed for account {} reason={}",
                    account_id, reason
                );
                (false, Some(reason))
            }
        };
        state
            .accounts()
            .finish_recovery_probe(index, success, error_reason.as_deref());
    }
}

fn format_proxy_error(err: &ProxyError) -> String {
    let mut message = match err {
        ProxyError::Http(e) => format_reqwest_error(e),
        _ => format!(
            "{} ({}): {}",
            err.error_code(),
            err.status_code().as_u16(),
            err
        ),
    };

    message = message.replace('\n', "\\n").replace('\r', "\\r");
    message.truncate(4096);
    message
}

fn format_reqwest_error(err: &reqwest::Error) -> String {
    use std::error::Error;

    let mut details = Vec::new();
    if err.is_timeout() {
        details.push("timeout");
    }
    if err.is_connect() {
        details.push("connect");
    }
    if err.is_request() {
        details.push("request");
    }
    if err.is_body() {
        details.push("body");
    }
    if err.is_decode() {
        details.push("decode");
    }

    let mut message = String::new();
    message.push_str("http_error");
    if let Some(status) = err.status() {
        message.push_str(&format!(" ({}): {}", status.as_u16(), err));
    } else {
        message.push_str(&format!(" (500): {}", err));
    }

    if let Some(url) = err.url() {
        message.push_str(&format!(" url={url}"));
    }
    if !details.is_empty() {
        message.push_str(&format!(" kind={}", details.join(",")));
    }

    let mut source = err.source();
    let mut depth = 0usize;
    while let Some(next) = source {
        depth += 1;
        if depth > 8 {
            message.push_str(" caused_by=…");
            break;
        }
        message.push_str(&format!(" caused_by={next}"));
        source = next.source();
    }

    message
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn favicon_handler() -> StatusCode {
    StatusCode::NO_CONTENT
}

#[derive(Debug, Clone, serde::Serialize)]
struct PublicModelPricingDto {
    #[serde(skip_serializing_if = "Option::is_none")]
    input_per_mtoken: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_per_mtoken: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize, Default)]
struct PublicModelDto {
    id: String,
    object: &'static str,
    owned_by: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pricing: Option<PublicModelPricingDto>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct PublicModelsListDto {
    object: &'static str,
    data: Vec<PublicModelDto>,
}

fn project_public_pricing(
    pricing: &crate::config::ModelPricingConfig,
) -> Option<PublicModelPricingDto> {
    let projected = PublicModelPricingDto {
        input_per_mtoken: pricing.input_per_mtoken,
        output_per_mtoken: pricing.output_per_mtoken,
    };
    (projected.input_per_mtoken.is_some() || projected.output_per_mtoken.is_some())
        .then_some(projected)
}

fn project_public_model_metadata(
    metadata: &crate::config::ModelMetadataConfig,
) -> Option<PublicModelDto> {
    let projected = PublicModelDto {
        id: String::new(),
        object: "model",
        owned_by: "codex-proxy",
        context_window: metadata.context_window,
        max_output_tokens: metadata.max_output_tokens,
        pricing: metadata.pricing.as_ref().and_then(project_public_pricing),
    };
    (projected.context_window.is_some()
        || projected.max_output_tokens.is_some()
        || projected.pricing.is_some())
    .then_some(projected)
}

fn public_model_from_config(cfg: &crate::config::Config, served_model: &str) -> PublicModelDto {
    let targets = cfg
        .route_targets_for_model(served_model)
        .map(|(_, targets)| targets);
    let projected = targets
        .as_ref()
        .and_then(|targets| targets.first())
        .and_then(|target| cfg.model_metadata(&target.model))
        .and_then(project_public_model_metadata)
        .unwrap_or_default();

    PublicModelDto {
        id: served_model.to_string(),
        object: "model",
        owned_by: "codex-proxy",
        context_window: projected.context_window,
        max_output_tokens: projected.max_output_tokens,
        pricing: projected.pricing,
    }
}

fn build_public_models_response(cfg: &crate::config::Config) -> PublicModelsListDto {
    PublicModelsListDto {
        object: "list",
        data: cfg
            .models
            .served
            .iter()
            .map(|served_model| public_model_from_config(cfg, served_model))
            .collect(),
    }
}

async fn models_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<PublicModelsListDto>, ProxyError> {
    let _ = crate::access::authenticate_request(state.config(), &headers)?;
    let response = with_config(state.config(), build_public_models_response);
    Ok(Json(response))
}

async fn responses_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(data): Json<ResponsesRequest>,
) -> Result<Response<Body>, ProxyError> {
    let access_key = crate::access::authenticate_request(state.config(), &headers)?;
    validate_responses_request(&data).map_err(|err| ProxyError::Validation(err.to_string()))?;
    if !with_config(state.config(), |cfg| {
        cfg.is_served_model_allowed(&data.model)
    }) {
        return Err(ProxyError::Validation(format!(
            "Requested model '{}' is not in models.served",
            data.model
        )));
    }

    let request_bytes = serde_json::to_vec(&data)
        .map(|v| v.len() as u64)
        .unwrap_or(0);

    let session_key = resolve_session_key(&state, &headers, &data)?;
    let cache_key_override = Some(session_key.cache_key_override());
    let normalized = normalize_responses_request(&data);
    let auto_cfg = with_config(state.config(), |cfg| cfg.auto_compaction.clone());
    let retry_cfg = with_config(state.config(), |cfg| cfg.retry.clone());
    let mut attempt = 0u32;
    let mut current_request = data.clone();
    let mut current_normalized = normalized;
    let mut auto_compacted = false;
    let final_result = loop {
        let (context, result) = execute_provider_request_with_route_retry(
            &state,
            state.zai(),
            &current_request.model,
            &current_normalized,
            &headers,
            cache_key_override,
            &retry_cfg,
        )
        .await?;

        match result {
            Ok(resp) => break Ok((context, resp)),
            Err(err) => {
                if !auto_cfg.enabled
                    || attempt >= auto_cfg.max_attempts_per_request
                    || !is_context_length_error(&err)
                {
                    break Err((context, err));
                }

                attempt += 1;
                let compacted = auto_compact_request(
                    &state,
                    &current_request,
                    &auto_cfg,
                    cache_key_override,
                    &current_normalized.messages,
                )
                .await?;
                current_request = compacted;
                current_normalized = normalize_responses_request(&current_request);
                auto_compacted = true;
            }
        }
    };

    match final_result {
        Ok((context, resp)) => {
            let mut resp = finalize_response(&state, resp, &session_key).await;
            if auto_compacted {
                resp.headers_mut().insert(
                    header::HeaderName::from_static("x-codex-proxy-auto-compacted"),
                    header::HeaderValue::from_static("true"),
                );
            }
            record_and_apply_result(&state, &access_key, &context, request_bytes, Ok(resp))
        }
        Err((context, err)) => {
            record_and_apply_result(&state, &access_key, &context, request_bytes, Err(err))
        }
    }
}

fn resolve_response_route(
    state: &AppState,
    requested_model: &str,
    messages: &[ChatMessage],
    cache_key_override: Option<u64>,
) -> Result<crate::account_pool::ResolvedRoute, ProxyError> {
    let Some((logical_model, targets)) = with_config(state.config(), |cfg| {
        cfg.route_targets_for_model(requested_model)
    }) else {
        return Err(ProxyError::Validation(format!(
            "No preferred route targets configured for requested model '{}'",
            requested_model
        )));
    };
    let candidates = crate::account_pool::Router::build_candidates(
        requested_model,
        &logical_model,
        &targets,
        |target| {
            with_config(state.config(), |cfg| {
                cfg.resolve_reasoning(target.reasoning.as_ref())
            })
            .map_err(ProxyError::Config)
        },
    )?;
    crate::account_pool::Router::resolve_route(
        state.accounts(),
        state.routing(),
        &candidates,
        messages,
        cache_key_override,
    )
}

fn resolve_compaction_route(
    state: &AppState,
    messages: &[ChatMessage],
    cache_key_override: Option<u64>,
) -> Result<crate::account_pool::ResolvedRoute, ProxyError> {
    let targets = with_config(state.config(), |cfg| cfg.compaction_targets());
    if targets.is_empty() {
        return Err(ProxyError::Validation(
            "No compaction route targets configured".into(),
        ));
    }
    let candidates: Vec<crate::account_pool::RouteCandidate> =
        crate::account_pool::Router::build_candidates(
            "__compaction__",
            "__compaction__",
            &targets,
            |target| {
                with_config(state.config(), |cfg| {
                    cfg.resolve_reasoning(target.reasoning.as_ref())
                })
                .map_err(ProxyError::Config)
            },
        )?;
    crate::account_pool::Router::resolve_route(
        state.accounts(),
        state.routing(),
        &candidates,
        messages,
        cache_key_override,
    )
}

fn resolve_session_key(
    state: &AppState,
    headers: &HeaderMap,
    request: &ResponsesRequest,
) -> Result<crate::session::SessionKey, ProxyError> {
    let (header_name, metadata_key) = with_config(state.config(), |cfg| {
        (
            cfg.session.header_name.clone(),
            cfg.session.metadata_key.clone(),
        )
    });

    if let Some(key) = crate::session::extract_session_key_from_headers(headers, &header_name) {
        return Ok(key);
    }

    let metadata_json = request
        .metadata
        .as_ref()
        .and_then(|m| serde_json::to_value(m).ok());
    if let Some(key) =
        crate::session::extract_session_key_from_metadata(metadata_json.as_ref(), &metadata_key)
    {
        return Ok(key);
    }

    if let Some(prev) = request.previous_response_id.as_deref()
        && let Some(key) = state.sessions().get_by_previous_response_id(prev)
    {
        return Ok(key);
    }

    Ok(crate::session::SessionKey::generate())
}

async fn finalize_response(
    state: &AppState,
    response: Response<Body>,
    session_key: &crate::session::SessionKey,
) -> Response<Body> {
    let header_name = with_config(state.config(), |cfg| cfg.session.header_name.clone());
    let mut response = response;
    crate::session::attach_session_header(response.headers_mut(), &header_name, session_key);

    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !content_type.starts_with("application/json") {
        return response;
    }

    if let Some(len) = response
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        && len > 4 * 1024 * 1024
    {
        return response;
    }

    let (parts, body) = response.into_parts();
    let bytes = match to_bytes(body, 4 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return Response::from_parts(parts, Body::empty());
        }
    };

    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes)
        && let Some(id) = value.get("id").and_then(|v| v.as_str())
    {
        state
            .sessions()
            .record_response_id(id.to_string(), session_key.clone());
    }

    Response::from_parts(parts, Body::from(bytes))
}

fn is_context_length_error(error: &ProxyError) -> bool {
    let lower = match error {
        ProxyError::Provider(err) => err.message.to_ascii_lowercase(),
        ProxyError::Validation(msg) => msg.to_ascii_lowercase(),
        ProxyError::Http(err) => err.to_string().to_ascii_lowercase(),
        ProxyError::Auth(_) => return false,
        ProxyError::Config(_) => return false,
        ProxyError::NotImplemented(_) => return false,
        ProxyError::Internal(_) => return false,
    };
    let has_status = lower.contains("(400)") || lower.contains("(413)") || lower.contains(" 413");
    let has_context = lower.contains("context")
        || lower.contains("token")
        || lower.contains("prompt")
        || lower.contains("maximum")
        || lower.contains("too long");
    let has_signal = lower.contains("context length")
        || lower.contains("maximum context")
        || lower.contains("prompt is too long")
        || lower.contains("too many tokens")
        || lower.contains("token limit")
        || lower.contains("exceeds")
        || lower.contains("too large");

    (has_status && has_context) || has_signal
}

fn is_retryable_provider_error(error: &ProxyError) -> bool {
    match error {
        ProxyError::Provider(err) => {
            matches!(
                err.kind(),
                crate::error::ProviderErrorKind::Network | crate::error::ProviderErrorKind::Server
            ) || err.status == Some(StatusCode::TOO_MANY_REQUESTS)
                || err.message.to_ascii_lowercase().contains("429")
                || err.message.to_ascii_lowercase().contains("rate limit")
                || err
                    .message
                    .to_ascii_lowercase()
                    .contains("too many requests")
        }
        ProxyError::Http(err) => err.status().is_none_or(|status| {
            status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
        }),
        _ => false,
    }
}

fn retry_delay_ms(error: &ProxyError, attempt: u32, retry_cfg: &crate::config::RetryConfig) -> u64 {
    if let ProxyError::Provider(err) = error
        && let Some(retry_after) = err.retry_after
    {
        return (retry_after.as_millis() as u64).min(retry_cfg.max_delay_ms);
    }

    (retry_cfg.initial_delay_ms as f64 * retry_cfg.backoff_multiplier.powi((attempt - 1) as i32))
        .min(retry_cfg.max_delay_ms as f64) as u64
}

async fn execute_provider_request_with_route_retry(
    state: &AppState,
    provider: &crate::providers::zai::ZAIProvider,
    requested_model: &str,
    normalized: &ChatRequest,
    headers: &HeaderMap,
    cache_key_override: Option<u64>,
    retry_cfg: &crate::config::RetryConfig,
) -> Result<(ZaiExecutionContext, Result<Response<Body>, ProxyError>), ProxyError> {
    let max_attempts = retry_cfg.max_attempts.max(1);
    let mut attempt = 1u32;

    loop {
        let context = resolve_zai_execution_context(
            state,
            requested_model,
            &normalized.messages,
            cache_key_override,
        )?;
        let result = provider
            .handle_request(normalized.clone(), headers.clone(), context.clone())
            .await;

        match result {
            Ok(resp) => return Ok((context, Ok(resp))),
            Err(err) => {
                if !retry_cfg.enabled
                    || attempt >= max_attempts
                    || !is_retryable_provider_error(&err)
                {
                    return Ok((context, Err(err)));
                }

                apply_account_failure(state, &context, &err);
                let delay_ms = retry_delay_ms(&err, attempt, retry_cfg);
                attempt += 1;
                tracing::warn!(
                    "Z.AI request failed with retryable error on account {}. Retrying attempt {}/{} after {}ms: {}",
                    context.route.account_id,
                    attempt,
                    max_attempts,
                    delay_ms,
                    err
                );
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }
    }
}

fn resolve_zai_execution_context(
    state: &AppState,
    requested_model: &str,
    messages: &[ChatMessage],
    cache_key_override: Option<u64>,
) -> Result<ZaiExecutionContext, ProxyError> {
    let route = resolve_response_route(state, requested_model, messages, cache_key_override)?;
    let (account, _) = state
        .accounts()
        .get_account(route.account_index)
        .ok_or_else(|| ProxyError::Internal("Resolved account missing from pool".into()))?;
    Ok(ZaiExecutionContext {
        route,
        account,
        config: state.config().clone(),
    })
}

#[cfg(test)]
async fn execute_request_with_retry<F, Fut>(
    mut send: F,
    retry_cfg: &crate::config::RetryConfig,
) -> Result<Response<Body>, ProxyError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<Response<Body>, ProxyError>>,
{
    let max_attempts = retry_cfg.max_attempts.max(1);
    let mut attempt = 1u32;
    loop {
        let result = send().await;

        match result {
            Ok(resp) => return Ok(resp),
            Err(err) => {
                if !retry_cfg.enabled
                    || attempt >= max_attempts
                    || !is_retryable_provider_error(&err)
                {
                    return Err(err);
                }
                let delay_ms = retry_delay_ms(&err, attempt, retry_cfg);
                attempt += 1;
                tracing::warn!(
                    "Z.AI request failed with retryable error. Retrying attempt {}/{} after {}ms: {}",
                    attempt,
                    max_attempts,
                    delay_ms,
                    err
                );
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }
    }
}

async fn auto_compact_request(
    state: &AppState,
    request: &ResponsesRequest,
    cfg: &crate::config::AutoCompactionConfig,
    cache_key_override: Option<u64>,
    normalized_messages: &[ChatMessage],
) -> Result<ResponsesRequest, ProxyError> {
    use crate::codex_wire::schema::responses_wire::{Content, InputItem, ResponsesInput};

    let input = request
        .input
        .clone()
        .ok_or_else(|| ProxyError::Validation("Auto-compaction requires request.input".into()))?;
    let ResponsesInput::Items(items) = input else {
        return Err(ProxyError::Validation(
            "Auto-compaction requires request.input to be an items[] array".into(),
        ));
    };

    let tail = cfg.tail_items_to_keep.max(1);
    if items.len() <= tail {
        return Err(ProxyError::Provider(ProviderError::new(
            None,
            "Auto-compaction skipped: not enough input items to compact",
        )));
    }

    let prefix_items = items[..items.len() - tail].to_vec();
    let tail_items = items[items.len() - tail..].to_vec();

    let summary = run_summary_compaction(
        state,
        prefix_items,
        crate::config::AUTO_COMPACTION_SUMMARY_INSTRUCTIONS.to_string(),
        cache_key_override,
        normalized_messages,
    )
    .await?;

    let mut rewritten_items = Vec::with_capacity(1 + tail_items.len());
    rewritten_items.push(InputItem {
        item_type: "message".into(),
        id: None,
        call_id: None,
        role: Some("system".into()),
        author: None,
        recipient: None,
        name: None,
        content: Some(Content::Text(format!(
            "[codex-proxy auto-compaction summary]\n{}",
            summary
        ))),
        reasoning_content: None,
        thought_signature: None,
        thought: None,
        arguments: None,
        input: None,
        action: None,
        command: None,
        cwd: None,
        working_directory: None,
        changes: None,
        output: None,
        stdout: None,
        stderr: None,
        encrypted_content: None,
    });
    rewritten_items.extend(tail_items);

    let mut next = request.clone();
    next.input = Some(ResponsesInput::Items(rewritten_items));
    next.previous_response_id = None;
    Ok(next)
}

async fn run_summary_compaction(
    state: &AppState,
    prefix_items: Vec<crate::codex_wire::schema::responses_wire::InputItem>,
    instructions: String,
    cache_key_override: Option<u64>,
    normalized_messages: &[ChatMessage],
) -> Result<String, ProxyError> {
    let route = resolve_compaction_route(state, normalized_messages, cache_key_override)?;
    let (account, _) = state
        .accounts()
        .get_account(route.account_index)
        .ok_or_else(|| ProxyError::Internal("Resolved account missing from pool".into()))?;
    let context = ZaiExecutionContext {
        route: route.clone(),
        account,
        config: state.config().clone(),
    };

    let prompt = build_summary_prompt(&prefix_items, &instructions);
    let raw = ResponsesRequest {
        model: "__compaction__".into(),
        input: Some(crate::codex_wire::schema::responses_wire::ResponsesInput::Text(prompt)),
        messages: None,
        instructions: None,
        previous_response_id: None,
        store: Some(false),
        metadata: None,
        tools: None,
        tool_choice: None,
        temperature: Some(0.1),
        top_p: None,
        max_tokens: Some(4096),
        max_output_tokens: None,
        stream: Some(false),
        include: None,
    };
    let normalized = normalize_responses_request(&raw);
    let resp = state
        .zai()
        .handle_request(normalized, HeaderMap::new(), context)
        .await?;
    let bytes = to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .map_err(|e| ProxyError::Internal(format!("Failed to read summary response: {e}")))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| ProxyError::Provider(ProviderError::new(None, e.to_string())))?;
    extract_output_text(&value).ok_or_else(|| {
        ProxyError::Provider(ProviderError::new(
            None,
            "Summary compaction response did not include output text",
        ))
    })
}

fn build_summary_prompt(
    prefix_items: &[crate::codex_wire::schema::responses_wire::InputItem],
    instructions: &str,
) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(&mut out, "{instructions}");
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "Conversation:");
    for item in prefix_items {
        let role = item.role.as_deref().unwrap_or("unknown");
        let mut line = String::new();
        if let Some(content) = &item.content {
            match content {
                crate::codex_wire::schema::responses_wire::Content::Text(text) => {
                    line.push_str(text);
                }
                crate::codex_wire::schema::responses_wire::Content::Parts(parts) => {
                    for part in parts {
                        if let Some(text) = &part.text {
                            line.push_str(text);
                        }
                    }
                }
                _ => {}
            }
        }
        if line.is_empty() {
            continue;
        }
        let _ = writeln!(&mut out, "- {role}: {line}");
    }
    out
}

fn extract_output_text(value: &serde_json::Value) -> Option<String> {
    if let Some(text) = value.get("output_text").and_then(|v| v.as_str()) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    let output = value.get("output")?.as_array()?;
    for item in output {
        if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
            let mut parts = String::new();
            for part in content {
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    parts.push_str(text);
                }
            }
            let trimmed = parts.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod auto_compaction_tests {
    use super::*;
    use crate::config::{
        AccessControlConfig, AccessKeyConfig, AccessKeyRole, AccountConfig, AutoCompactionConfig,
        CompactionConfig, Config, ModelMetadataConfig, ModelPricingConfig, ModelsConfig,
        ReasoningConfig, RetryConfig, RoutingConfig, RoutingHealthConfig, ServerConfig,
        SessionConfig, TimeoutsConfig, ZaiProviderConfig,
    };
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use tower::util::ServiceExt;

    fn test_config() -> Config {
        Config {
            config_path: PathBuf::from("/tmp/config.json"),
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 8765,
                log_level: "INFO".into(),
            },
            zai: ZaiProviderConfig {
                api_url: "https://zai.example/chat".into(),
                models: vec!["real-routed-model".into()],
            },
            models: ModelsConfig {
                served: vec![
                    "logical-route-only".into(),
                    "served-without-metadata".into(),
                ],
            },
            model_metadata: HashMap::from([(
                "real-routed-model".into(),
                ModelMetadataConfig {
                    context_window: Some(200_000),
                    max_output_tokens: Some(16_384),
                    pricing: Some(ModelPricingConfig {
                        input_per_mtoken: Some(1.25),
                        output_per_mtoken: Some(5.0),
                    }),
                },
            )]),
            session: SessionConfig::default(),
            auto_compaction: AutoCompactionConfig::default(),
            routing: RoutingConfig {
                model_routes: HashMap::from([
                    (
                        "logical-route-only".into(),
                        vec![crate::config::ModelRouteStepConfig::Physical {
                            model: "real-routed-model".into(),
                            reasoning: None,
                        }],
                    ),
                    (
                        "served-without-metadata".into(),
                        vec![crate::config::ModelRouteStepConfig::Physical {
                            model: "unknown-upstream-model".into(),
                            reasoning: None,
                        }],
                    ),
                ]),
            },
            health: RoutingHealthConfig::default(),
            accounts: vec![AccountConfig {
                id: "route-only-a".into(),
                enabled: true,
                weight: 1,
                models: None,
                auth: crate::account_pool::AccountAuth::ApiKey {
                    api_key: "sk-test".into(),
                },
            }],
            access: AccessControlConfig {
                require_key: true,
                keys: vec![
                    AccessKeyConfig {
                        id: "api-key".into(),
                        key_sha256: crate::access::sha256_hex("api-secret"),
                        plaintext: None,
                        name: Some("API".into()),
                        enabled: true,
                        role: Some(AccessKeyRole::Api),
                        is_admin: false,
                    },
                    AccessKeyConfig {
                        id: "admin-key".into(),
                        key_sha256: crate::access::sha256_hex("admin-secret"),
                        plaintext: None,
                        name: Some("Admin".into()),
                        enabled: true,
                        role: Some(AccessKeyRole::Admin),
                        is_admin: false,
                    },
                ],
            },
            reasoning: ReasoningConfig::default(),
            timeouts: TimeoutsConfig {
                connect_seconds: 1,
                read_seconds: 1,
            },
            compaction: CompactionConfig {
                temperature: 0.1,
                preferred_targets: Vec::new(),
            },
            retry: RetryConfig::default(),
        }
    }

    fn base_test_state() -> AppState {
        let config = Arc::new(RwLock::new(test_config()));
        let state = AppState::new(config.clone());
        state
            .accounts()
            .configure_health(with_config(&config, |cfg| cfg.health.clone()));
        state.accounts().load_accounts(with_config(&config, |cfg| {
            cfg.accounts.clone().into_iter().map(Into::into).collect()
        }));
        state
    }

    fn test_context(state: &AppState) -> ZaiExecutionContext {
        let (account, _) = state.accounts().get_account(0).unwrap();
        ZaiExecutionContext {
            route: crate::account_pool::ResolvedRoute {
                requested_model: "logical-route-only".into(),
                logical_model: "logical-route-only".into(),
                upstream_model: "real-routed-model".into(),
                account_index: 0,
                account_id: account.id.clone(),
                cache_hit: false,
                cache_key: 0,
                preferred_target_index: 0,
                reasoning: None,
            },
            account,
            config: state.config().clone(),
        }
    }

    #[test]
    fn detects_context_length_provider_error() {
        let err = ProxyError::Provider(ProviderError::new(
            Some(StatusCode::BAD_REQUEST),
            "Responses request failed (400): This model's maximum context length is 128000 tokens",
        ));
        assert!(is_context_length_error(&err));
    }

    #[test]
    fn detects_retryable_provider_error() {
        let err = ProxyError::Provider(ProviderError::new(
            Some(StatusCode::TOO_MANY_REQUESTS),
            "upstream said slow down",
        ));
        assert!(is_retryable_provider_error(&err));

        let err = ProxyError::Provider(ProviderError::new(None, "rate limit exceeded"));
        assert!(is_retryable_provider_error(&err));

        let err = ProxyError::Provider(ProviderError::new(
            Some(StatusCode::BAD_GATEWAY),
            "temporary upstream failure",
        ));
        assert!(is_retryable_provider_error(&err));

        let err = ProxyError::Provider(ProviderError::new(
            Some(StatusCode::BAD_REQUEST),
            "context length exceeded",
        ));
        assert!(!is_retryable_provider_error(&err));

        let err = ProxyError::Provider(ProviderError::new(
            Some(StatusCode::UNAUTHORIZED),
            "bad api key",
        ));
        assert!(!is_retryable_provider_error(&err));
    }

    #[test]
    fn rate_limit_failure_marks_account_unhealthy() {
        let state = base_test_state();
        let context = test_context(&state);
        let err = ProxyError::Provider(
            ProviderError::new(Some(StatusCode::TOO_MANY_REQUESTS), "too many requests")
                .with_retry_after(Some(Duration::from_secs(2))),
        );

        apply_account_failure(&state, &context, &err);

        let (_, snapshot) = state.accounts().get_account(0).unwrap();
        assert!(!snapshot.alive);
        assert!(snapshot.recovery_probe_due);
        assert_eq!(snapshot.consecutive_failures, 1);
        assert!(snapshot.unhealthy_until.is_some());
    }

    #[test]
    fn retry_delay_prefers_retry_after_and_caps_to_config() {
        let retry_cfg = RetryConfig {
            enabled: true,
            max_attempts: 3,
            initial_delay_ms: 100,
            max_delay_ms: 250,
            backoff_multiplier: 2.0,
        };
        let err = ProxyError::Provider(
            ProviderError::new(Some(StatusCode::TOO_MANY_REQUESTS), "rate limit")
                .with_retry_after(Some(Duration::from_millis(500))),
        );
        assert_eq!(retry_delay_ms(&err, 1, &retry_cfg), 250);

        let err = ProxyError::Provider(ProviderError::new(
            Some(StatusCode::BAD_GATEWAY),
            "temporary upstream failure",
        ));
        assert_eq!(retry_delay_ms(&err, 2, &retry_cfg), 200);
    }

    #[derive(Clone)]
    struct FlakyProvider {
        calls: Arc<AtomicUsize>,
        failures_before_success: usize,
        failure_status: Option<StatusCode>,
        failure_message: String,
    }

    impl FlakyProvider {
        async fn send(&self) -> Result<Response<Body>, ProxyError> {
            let call = self.calls.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            if call <= self.failures_before_success {
                Err(ProxyError::Provider(ProviderError::new(
                    self.failure_status,
                    self.failure_message.clone(),
                )))
            } else {
                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::from("{}"))
                    .unwrap())
            }
        }
    }

    #[tokio::test]
    async fn retry_disabled_does_not_retry_rate_limits() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = FlakyProvider {
            calls: calls.clone(),
            failures_before_success: 1,
            failure_status: Some(StatusCode::TOO_MANY_REQUESTS),
            failure_message: "rate limit".into(),
        };

        let result = execute_request_with_retry(
            || provider.send(),
            &RetryConfig {
                enabled: false,
                max_attempts: 3,
                initial_delay_ms: 0,
                max_delay_ms: 0,
                backoff_multiplier: 2.0,
            },
        )
        .await;

        assert!(result.is_err());
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_succeeds_after_rate_limit_within_max_attempts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = FlakyProvider {
            calls: calls.clone(),
            failures_before_success: 1,
            failure_status: Some(StatusCode::TOO_MANY_REQUESTS),
            failure_message: "rate limit".into(),
        };

        let result = execute_request_with_retry(
            || provider.send(),
            &RetryConfig {
                enabled: true,
                max_attempts: 2,
                initial_delay_ms: 0,
                max_delay_ms: 0,
                backoff_multiplier: 2.0,
            },
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_succeeds_after_server_error_within_max_attempts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = FlakyProvider {
            calls: calls.clone(),
            failures_before_success: 1,
            failure_status: Some(StatusCode::BAD_GATEWAY),
            failure_message: "temporary upstream failure".into(),
        };

        let result = execute_request_with_retry(
            || provider.send(),
            &RetryConfig {
                enabled: true,
                max_attempts: 2,
                initial_delay_ms: 0,
                max_delay_ms: 0,
                backoff_multiplier: 2.0,
            },
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_succeeds_after_network_provider_error() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = FlakyProvider {
            calls: calls.clone(),
            failures_before_success: 1,
            failure_status: None,
            failure_message: "connection reset".into(),
        };

        let result = execute_request_with_retry(
            || provider.send(),
            &RetryConfig {
                enabled: true,
                max_attempts: 2,
                initial_delay_ms: 0,
                max_delay_ms: 0,
                backoff_multiplier: 2.0,
            },
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_does_not_retry_client_errors() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = FlakyProvider {
            calls: calls.clone(),
            failures_before_success: 1,
            failure_status: Some(StatusCode::BAD_REQUEST),
            failure_message: "bad request".into(),
        };

        let result = execute_request_with_retry(
            || provider.send(),
            &RetryConfig {
                enabled: true,
                max_attempts: 3,
                initial_delay_ms: 0,
                max_delay_ms: 0,
                backoff_multiplier: 2.0,
            },
        )
        .await;

        assert!(result.is_err());
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_stops_after_total_max_attempts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = FlakyProvider {
            calls: calls.clone(),
            failures_before_success: 10,
            failure_status: Some(StatusCode::TOO_MANY_REQUESTS),
            failure_message: "rate limit".into(),
        };

        let result = execute_request_with_retry(
            || provider.send(),
            &RetryConfig {
                enabled: true,
                max_attempts: 3,
                initial_delay_ms: 0,
                max_delay_ms: 0,
                backoff_multiplier: 2.0,
            },
        )
        .await;

        assert!(result.is_err());
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 3);
    }

    #[test]
    fn extracts_output_text_from_output_text_field() {
        let value = serde_json::json!({
            "output_text": " hello "
        });
        assert_eq!(extract_output_text(&value).as_deref(), Some("hello"));
    }

    #[test]
    fn extracts_output_text_from_output_array() {
        let value = serde_json::json!({
            "output": [
                {
                    "content": [{"type":"output_text","text":"abc"}]
                }
            ]
        });
        assert_eq!(extract_output_text(&value).as_deref(), Some("abc"));
    }

    #[test]
    fn builds_summary_prompt_from_message_items() {
        use crate::codex_wire::schema::responses_wire::{Content, InputItem};
        let items = vec![
            InputItem {
                item_type: "message".into(),
                id: None,
                call_id: None,
                role: Some("user".into()),
                author: None,
                recipient: None,
                name: None,
                content: Some(Content::Text("hi".into())),
                reasoning_content: None,
                thought_signature: None,
                thought: None,
                arguments: None,
                input: None,
                action: None,
                command: None,
                cwd: None,
                working_directory: None,
                changes: None,
                output: None,
                stdout: None,
                stderr: None,
                encrypted_content: None,
            },
            InputItem {
                item_type: "message".into(),
                id: None,
                call_id: None,
                role: Some("assistant".into()),
                author: None,
                recipient: None,
                name: None,
                content: Some(Content::Text("hello".into())),
                reasoning_content: None,
                thought_signature: None,
                thought: None,
                arguments: None,
                input: None,
                action: None,
                command: None,
                cwd: None,
                working_directory: None,
                changes: None,
                output: None,
                stdout: None,
                stderr: None,
                encrypted_content: None,
            },
        ];
        let prompt = build_summary_prompt(&items, "do a summary");
        assert!(prompt.contains("do a summary"));
        assert!(prompt.contains("- user: hi"));
        assert!(prompt.contains("- assistant: hello"));
    }

    #[test]
    fn recovery_probe_uses_zai_catalog_when_account_models_missing() {
        let state = base_test_state();
        let (account, _) = state.accounts().get_account(0).unwrap();
        let target = with_config(state.config(), |cfg| cfg.recovery_probe_target().unwrap());
        assert_eq!(target.model, "real-routed-model");

        let route = crate::account_pool::ResolvedRoute {
            requested_model: target.model.clone(),
            logical_model: target.model.clone(),
            upstream_model: target.model.clone(),
            account_index: 0,
            account_id: account.id.clone(),
            cache_hit: false,
            cache_key: 0,
            preferred_target_index: 0,
            reasoning: Some(crate::config::EffectiveReasoningConfig {
                budget: 0,
                level: "LOW".into(),
                preset: Some("none".into()),
            }),
        };
        let raw = ResponsesRequest {
            model: target.model.clone(),
            input: Some(
                crate::codex_wire::schema::responses_wire::ResponsesInput::Text(
                    "health check".into(),
                ),
            ),
            messages: None,
            instructions: None,
            previous_response_id: None,
            store: Some(false),
            metadata: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_output_tokens: None,
            stream: Some(false),
            include: None,
        };
        let normalized = normalize_responses_request(&raw);

        assert!(target.model == "real-routed-model" || target.model == "unknown-upstream-model");
        assert_eq!(route.upstream_model, target.model);
        assert_eq!(raw.model, target.model);
        assert_eq!(normalized.model, target.model);
        assert!(!raw.model.contains("__probe__"));
        assert!(!route.upstream_model.contains("__probe__"));
    }

    #[test]
    fn public_models_response_only_exposes_served_models_and_allowed_fields() {
        let cfg = test_config();
        let response = build_public_models_response(&cfg);
        let value = serde_json::to_value(&response).unwrap();
        let data = value
            .get("data")
            .and_then(|v| v.as_array())
            .expect("models list data");

        assert_eq!(data.len(), 2);
        assert_eq!(
            data[0].get("id").and_then(|v| v.as_str()),
            Some("logical-route-only")
        );
        assert_eq!(
            data[1].get("id").and_then(|v| v.as_str()),
            Some("served-without-metadata")
        );
        assert_eq!(
            data[0].get("object").and_then(|v| v.as_str()),
            Some("model")
        );
        assert_eq!(
            data[0].get("owned_by").and_then(|v| v.as_str()),
            Some("codex-proxy")
        );
        assert_eq!(
            data[0].get("context_window").and_then(|v| v.as_u64()),
            Some(200_000)
        );
        assert_eq!(
            data[0].get("max_output_tokens").and_then(|v| v.as_u64()),
            Some(16_384)
        );
        assert_eq!(
            data[0]
                .get("pricing")
                .and_then(|v| v.get("input_per_mtoken"))
                .and_then(|v| v.as_f64()),
            Some(1.25)
        );
        assert_eq!(
            data[0]
                .get("pricing")
                .and_then(|v| v.get("output_per_mtoken"))
                .and_then(|v| v.as_f64()),
            Some(5.0)
        );

        for entry in data {
            assert!(entry.get("logical_model").is_none());
            assert!(entry.get("routing_targets").is_none());
            assert!(entry.get("default_target").is_none());
            assert!(entry.get("default_target_metadata").is_none());
            assert!(entry.get("providers").is_none());
            assert!(entry.get("provider_metadata").is_none());
            assert!(entry.get("last_error").is_none());
            assert!(entry.get("metadata").is_none());
        }

        let missing_metadata = data
            .iter()
            .find(|entry| {
                entry.get("id").and_then(|v| v.as_str()) == Some("served-without-metadata")
            })
            .unwrap();
        assert!(missing_metadata.get("context_window").is_none());
        assert!(missing_metadata.get("max_output_tokens").is_none());
        assert!(missing_metadata.get("pricing").is_none());
        assert!(
            data.iter()
                .all(|entry| entry.get("id").and_then(|v| v.as_str())
                    != Some("not-served-discovered-model"))
        );
    }

    async fn perform_request(path: &str, key: Option<&str>) -> (StatusCode, serde_json::Value) {
        let state = base_test_state();
        let mut builder = Request::builder().uri(path).method("GET");
        if let Some(key) = key {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        let request = builder.body(Body::empty()).unwrap();
        let response = build_router(state).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| serde_json::json!({}));
        (status, body)
    }

    #[tokio::test]
    async fn non_admin_api_user_can_access_public_models_but_not_admin_config() {
        let (public_status, public_body) = perform_request("/v1/models", Some("api-secret")).await;
        assert_eq!(public_status, StatusCode::OK);
        let public_data = public_body
            .get("data")
            .and_then(|v| v.as_array())
            .expect("public models data");
        assert_eq!(public_data.len(), 2);
        assert!(
            public_data
                .iter()
                .all(|entry| entry.get("metadata").is_none())
        );
        assert!(
            public_data
                .iter()
                .all(|entry| entry.get("routing_targets").is_none())
        );
        assert!(
            public_data
                .iter()
                .all(|entry| entry.get("id").and_then(|v| v.as_str())
                    != Some("not-served-discovered-model"))
        );

        let (admin_status, admin_body) = perform_request("/api/config", Some("api-secret")).await;
        assert_eq!(admin_status, StatusCode::UNAUTHORIZED);
        assert!(
            admin_body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("role=admin")
        );
    }

    #[tokio::test]
    async fn admin_can_access_admin_endpoints() {
        let (config_status, _) = perform_request("/api/config", Some("admin-secret")).await;
        let (accounts_status, _) = perform_request("/api/accounts", Some("admin-secret")).await;
        let (usage_keys_status, _) = perform_request("/api/usage/keys", Some("admin-secret")).await;
        let (access_keys_status, _) =
            perform_request("/api/access-keys", Some("admin-secret")).await;
        assert_eq!(config_status, StatusCode::OK);
        assert_eq!(accounts_status, StatusCode::OK);
        assert_eq!(usage_keys_status, StatusCode::OK);
        assert_eq!(access_keys_status, StatusCode::OK);
    }
}

fn record_and_apply_result(
    state: &AppState,
    access_key: &Option<AuthenticatedKey>,
    context: &ZaiExecutionContext,
    request_bytes: u64,
    result: Result<Response<Body>, ProxyError>,
) -> Result<Response<Body>, ProxyError> {
    let status = match &result {
        Ok(resp) => resp.status(),
        Err(err) => err.status_code(),
    };
    let usage_handle = state
        .usage()
        .record_request_start_handle(crate::usage::RequestUsageStart {
            key_id: access_key.as_ref().map(|k| k.id.as_str()),
            account_id: &context.route.account_id,
            model: &context.route.upstream_model,
            status,
            cache_hit: context.route.cache_hit,
            request_bytes,
        });

    match result {
        Ok(response) => {
            let response = wrap_response_for_usage(state.clone(), usage_handle, response);
            state.accounts().mark_success(context.route.account_index);
            state.routing().bind_on_success(&context.route);
            Ok(response)
        }
        Err(error) => {
            apply_account_failure(state, context, &error);
            Err(error)
        }
    }
}

fn apply_account_failure(state: &AppState, context: &ZaiExecutionContext, error: &ProxyError) {
    let reason = format_proxy_error(error);
    if is_rate_limit_error(error) {
        state.accounts().mark_rate_limited(
            context.route.account_index,
            provider_retry_after(error),
            Some(reason.as_str()),
        );
        return;
    }

    match error.provider_kind() {
        Some(ProviderErrorKind::Client) => {
            state
                .accounts()
                .mark_nonfatal_failure(context.route.account_index, Some(reason.as_str()));
        }
        Some(ProviderErrorKind::Auth) => {
            state
                .accounts()
                .mark_failure(context.route.account_index, true, Some(reason.as_str()));
        }
        _ => {
            let is_auth_error = is_auth_failure(error);
            state.accounts().mark_failure(
                context.route.account_index,
                is_auth_error,
                Some(reason.as_str()),
            );
        }
    }
}

fn is_rate_limit_error(error: &ProxyError) -> bool {
    match error {
        ProxyError::Provider(err) => {
            err.status == Some(StatusCode::TOO_MANY_REQUESTS)
                || err.message.to_ascii_lowercase().contains("429")
                || err.message.to_ascii_lowercase().contains("rate limit")
                || err
                    .message
                    .to_ascii_lowercase()
                    .contains("too many requests")
        }
        ProxyError::Http(err) => err
            .status()
            .map(|status| status == reqwest::StatusCode::TOO_MANY_REQUESTS)
            .unwrap_or(false),
        _ => false,
    }
}

fn provider_retry_after(error: &ProxyError) -> Option<Duration> {
    match error {
        ProxyError::Provider(err) => err.retry_after,
        _ => None,
    }
}

struct UsageResponseTracker {
    state: AppState,
    handle: crate::usage::UsageHandle,
    response_bytes: u64,
}

impl Drop for UsageResponseTracker {
    fn drop(&mut self) {
        self.state
            .usage()
            .record_response_bytes(&self.handle, self.response_bytes);
    }
}

fn wrap_response_for_usage(
    state: AppState,
    handle: crate::usage::UsageHandle,
    response: Response<Body>,
) -> Response<Body> {
    let (parts, body) = response.into_parts();
    let mut tracker = UsageResponseTracker {
        state,
        handle,
        response_bytes: 0,
    };

    let mut data_stream = body.into_data_stream();
    let stream = async_stream::stream! {
        while let Some(chunk) = data_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    tracker.response_bytes += bytes.len() as u64;
                    yield Ok::<Bytes, io::Error>(bytes);
                }
                Err(err) => {
                    yield Err(io::Error::other(err.to_string()));
                    break;
                }
            }
        }
        drop(tracker);
    };

    Response::from_parts(parts, Body::from_stream(stream))
}

fn is_auth_failure(error: &ProxyError) -> bool {
    match error {
        ProxyError::Auth(_) => true,
        ProxyError::Http(err) => err
            .status()
            .map(|status| status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN)
            .unwrap_or(false),
        ProxyError::Provider(err) => matches!(err.kind(), ProviderErrorKind::Auth),
        _ => false,
    }
}

fn authenticate_admin(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<Option<AuthenticatedKey>, ProxyError> {
    let key = crate::access::authenticate_request(state.config(), headers)?;
    require_admin(key.as_ref())?;
    Ok(key)
}

async fn api_config_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    let (config_path, persisted) = with_config(state.config(), |cfg| {
        (cfg.config_path.display().to_string(), cfg.to_persisted())
    });
    let mut value = serde_json::to_value(persisted).unwrap_or_else(|_| json!({}));
    if let Some(accounts) = value.get_mut("accounts").and_then(|v| v.as_array_mut()) {
        for account in accounts {
            if let Some(auth) = account.get_mut("auth")
                && let Some(obj) = auth.as_object_mut()
                && obj.get("type").and_then(|v| v.as_str()) == Some("api_key")
                && let Some(api_key) = obj.get("api_key").and_then(|v| v.as_str())
            {
                let masked = if api_key.len() <= 8 {
                    "***".to_string()
                } else {
                    format!("{}...{}", &api_key[..4], &api_key[api_key.len() - 4..])
                };
                obj.insert("api_key".to_string(), serde_json::Value::String(masked));
            }
        }
    }
    Ok(Json(json!({ "config_path": config_path, "config": value })))
}

async fn api_config_put(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(data): Json<PersistedConfig>,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    let config_path = with_config(state.config(), |cfg| cfg.config_path.clone());
    let mut next = data.into_runtime(config_path.clone());
    next.validate().map_err(ProxyError::Config)?;
    next.save_to_path(&config_path)
        .map_err(ProxyError::Config)?;

    with_config_mut(state.config(), |cfg| {
        *cfg = next.clone();
    });
    state
        .sessions()
        .set_ttl(Duration::from_secs(next.session.response_id_ttl_seconds));
    state.accounts().configure_health(next.health.clone());
    state
        .accounts()
        .load_accounts(next.accounts.into_iter().map(Into::into).collect());
    state.routing().clear();

    api_config_get(State(state), headers).await
}

async fn api_accounts_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    Ok(Json(json!({
        "accounts": state.accounts().all_accounts_snapshot(),
        "sticky_bindings": state.routing().snapshot_size(),
    })))
}

async fn api_usage_keys_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    Ok(Json(json!({ "keys": state.usage().snapshot_keys() })))
}

async fn api_usage_accounts_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    Ok(Json(
        json!({ "accounts": state.usage().snapshot_accounts() }),
    ))
}

#[derive(serde::Deserialize)]
struct UsageSeriesQuery {
    #[serde(default)]
    bucket_seconds: Option<u64>,
    #[serde(default)]
    window_seconds: Option<u64>,
}

async fn api_usage_series_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<UsageSeriesQuery>,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;

    let bucket_seconds = query.bucket_seconds.unwrap_or(300);
    if bucket_seconds == 0 || bucket_seconds % 60 != 0 {
        return Err(ProxyError::Validation(
            "bucket_seconds must be a positive multiple of 60".into(),
        ));
    }
    let window_seconds = query.window_seconds.unwrap_or(24 * 60 * 60);
    if window_seconds == 0 {
        return Err(ProxyError::Validation(
            "window_seconds must be positive".into(),
        ));
    }

    let buckets = state
        .usage()
        .snapshot_series(bucket_seconds, window_seconds);
    Ok(Json(json!({
        "bucket_seconds": bucket_seconds,
        "window_seconds": window_seconds,
        "buckets": buckets,
    })))
}

#[derive(serde::Deserialize)]
struct AccessKeyCreateRequest {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub role: Option<crate::config::AccessKeyRole>,
}

async fn api_access_keys_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    let keys = with_config(state.config(), |cfg| cfg.access.keys.clone());
    Ok(Json(json!({ "keys": keys })))
}

async fn api_access_keys_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AccessKeyCreateRequest>,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    let plaintext = crate::access::generate_access_key();
    let sha = crate::access::sha256_hex(&plaintext);

    let key_id = req.id.unwrap_or_else(|| format!("key-{}", &sha[..12]));
    let enabled = req.enabled.unwrap_or(true);
    let role = req.role.unwrap_or(crate::config::AccessKeyRole::Api);
    let name = req.name;

    let config_path = with_config(state.config(), |cfg| cfg.config_path.clone());
    let next = with_config_mut(state.config(), |cfg| {
        if cfg.access.keys.iter().any(|k| k.id == key_id) {
            return Err(ProxyError::Validation(format!(
                "access key id '{}' already exists",
                key_id
            )));
        }
        cfg.access.keys.push(crate::config::AccessKeyConfig {
            id: key_id.clone(),
            key_sha256: sha.clone(),
            plaintext: None,
            name,
            enabled,
            role: Some(role),
            is_admin: false,
        });
        cfg.validate().map_err(ProxyError::Config)?;
        Ok(cfg.clone())
    })?;
    next.save_to_path(&config_path)
        .map_err(ProxyError::Config)?;
    Ok(Json(json!({
        "id": key_id,
        "plaintext": plaintext,
        "key_sha256": sha,
        "role": role,
    })))
}

async fn api_access_keys_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    let config_path = with_config(state.config(), |cfg| cfg.config_path.clone());
    let next = with_config_mut(state.config(), |cfg| {
        let before = cfg.access.keys.len();
        cfg.access.keys.retain(|k| k.id != id);
        if cfg.access.keys.len() == before {
            return Err(ProxyError::Validation(format!(
                "access key '{}' not found",
                id
            )));
        }
        cfg.validate().map_err(ProxyError::Config)?;
        Ok(cfg.clone())
    })?;
    next.save_to_path(&config_path)
        .map_err(ProxyError::Config)?;
    Ok(Json(json!({ "ok": true })))
}
