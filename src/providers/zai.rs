use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::account_pool::{Account, AccountAuth, ResolvedRoute};
use crate::codex_wire::providers::adapter::{ProviderBuildContext, build_zai_payload};
use crate::codex_wire::providers::zai::{custom_tool_names, extract_custom_input};
use crate::codex_wire::schema::responses_wire::ChatRequest;
use crate::codex_wire::schema::sse::{
    CustomToolCallItem, FunctionCallItem, LocalShellCallItem, MessageItem, OutputContentPart,
    OutputItem, ResponseObject, Usage,
};
use crate::config::{ConfigHandle, EffectiveReasoningConfig, with_config};
use crate::error::{ProviderError, ProxyError};

#[derive(Clone)]
pub struct ZAIProvider {
    client: reqwest::Client,
}

#[derive(Clone)]
pub struct ZaiExecutionContext {
    pub route: ResolvedRoute,
    pub account: Account,
    pub config: ConfigHandle,
}

impl ZaiExecutionContext {
    pub fn upstream_model(&self) -> &str {
        &self.route.upstream_model
    }

    pub fn reasoning(&self) -> Option<&EffectiveReasoningConfig> {
        self.route.reasoning.as_ref()
    }

    pub fn preferred_target_index(&self) -> usize {
        self.route.preferred_target_index
    }
}

impl Default for ZAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ZAIProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    fn resolve_endpoint_url(&self, context: &ZaiExecutionContext) -> String {
        with_config(&context.config, |cfg| cfg.zai.api_url.clone())
    }

    async fn post_json<T: serde::Serialize>(
        &self,
        payload: &T,
        auth: String,
        context: &ZaiExecutionContext,
    ) -> Result<reqwest::Response, ProxyError> {
        let endpoint_url = self.resolve_endpoint_url(context);
        self.client
            .post(endpoint_url)
            .header("Authorization", auth)
            .json(payload)
            .timeout(std::time::Duration::from_secs(with_config(
                &context.config,
                |cfg| cfg.timeouts.read_seconds,
            )))
            .send()
            .await
            .map_err(ProxyError::Http)
    }

    async fn execute_request(
        &self,
        req: &ChatRequest,
        context: &ZaiExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let auth = resolve_zai_auth(context)?;
        let mut ctx = ProviderBuildContext::new(context.upstream_model());
        ctx.thinking_enabled = context.reasoning().map(|r| r.budget > 0);
        let zai_req = build_zai_payload(req, &ctx);
        let resp = self.post_json(&zai_req, auth, context).await?;
        let status = resp.status();
        info!("Z.AI response status: {}", status);

        if !status.is_success() {
            let retry_after = parse_retry_after_header(resp.headers());
            let body = resp.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(ProxyError::Auth(format!(
                    "Z.AI request unauthorized ({}). Body: {}",
                    status, body
                )));
            }
            return Err(ProxyError::Provider(
                ProviderError::new(
                    Some(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY)),
                    format!("Z.AI request failed ({}): {}", status, body),
                )
                .with_retry_after(retry_after),
            ));
        }

        if req.stream {
            self.handle_stream_response(resp, req, context).await
        } else {
            self.handle_sync_response(resp, req).await
        }
    }

    async fn handle_stream_response(
        &self,
        resp: reqwest::Response,
        req: &ChatRequest,
        context: &ZaiExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let model = req.model.clone();
        let created_ts = now_seconds();
        let idle_timeout_seconds = with_config(&context.config, |cfg| cfg.timeouts.read_seconds);
        let sse_stream = crate::providers::zai_stream::stream_responses_sse(
            resp.bytes_stream(),
            &model,
            created_ts,
            req,
            idle_timeout_seconds,
        );
        let body = Body::from_stream(sse_stream);
        Ok(Response::builder()
            .status(200)
            .header("Content-Type", "text/event-stream; charset=utf-8")
            .header("Connection", "keep-alive")
            .body(body)
            .unwrap())
    }

    async fn handle_sync_response(
        &self,
        resp: reqwest::Response,
        req: &ChatRequest,
    ) -> Result<Response<Body>, ProxyError> {
        let body_bytes = resp.bytes().await?;
        let z_data: ZaiChatResponse = serde_json::from_slice(&body_bytes).map_err(|e| {
            ProxyError::Provider(ProviderError::new(
                None,
                format!("Failed to decode Z.AI response JSON: {e}"),
            ))
        })?;

        let out = map_zai_response_to_responses_api(&z_data, req);
        Ok(axum::Json(out).into_response())
    }

    pub async fn handle_request(
        &self,
        data: ChatRequest,
        _headers: HeaderMap,
        context: ZaiExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        self.execute_request(&data, &context).await
    }
}

fn parse_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    httpdate::parse_http_date(value)
        .ok()
        .and_then(|deadline| deadline.duration_since(SystemTime::now()).ok())
}

fn tool_namespace_by_name(req: &ChatRequest) -> HashMap<String, Option<String>> {
    let mut map = HashMap::new();
    for tool in &req.tools {
        if tool.tool_type != "function" {
            continue;
        }
        let name = tool
            .function
            .as_ref()
            .map(|function| function.name.clone())
            .or_else(|| tool.name.clone());
        let Some(name) = name else {
            continue;
        };
        let namespace = tool.namespace.clone();
        map.entry(name)
            .and_modify(|existing| {
                if *existing != namespace {
                    *existing = None;
                }
            })
            .or_insert(namespace);
    }
    map
}

fn resolve_zai_auth(context: &ZaiExecutionContext) -> Result<String, ProxyError> {
    match &context.account.auth {
        AccountAuth::ApiKey { api_key } if !api_key.is_empty() => Ok(format!("Bearer {api_key}")),
        _ => Err(ProxyError::Auth(
            "Missing Z.AI API key. Configure the account auth for this Z.AI account.".into(),
        )),
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiChatResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    created: Option<i64>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<ZaiChoice>,
    #[serde(default)]
    usage: Option<ZaiUsage>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiChoice {
    #[serde(default)]
    message: Option<ZaiChoiceMessage>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiChoiceMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ZaiToolCall>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ZaiToolCallFn>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiToolCallFn {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
}

fn map_zai_response_to_responses_api(z: &ZaiChatResponse, req: &ChatRequest) -> ResponseObject {
    let created = z.created.unwrap_or(0);
    let model = z.model.clone().unwrap_or_else(|| "unknown".into());
    let resp_id = format!("zai_{}", z.id.clone().unwrap_or_else(|| "unknown".into()));

    let namespace_by_name = tool_namespace_by_name(req);
    let custom_names = custom_tool_names(&req.tools);
    let mut output: Vec<OutputItem> = Vec::new();

    if let Some(choice) = z.choices.first()
        && let Some(msg) = &choice.message
    {
        for (idx, tc) in msg.tool_calls.iter().enumerate() {
            let call_id = tc
                .id
                .clone()
                .unwrap_or_else(|| format!("call_{}_{}", now_ms(), idx));
            let name = tc
                .function
                .as_ref()
                .and_then(|f| f.name.clone())
                .unwrap_or_default();
            let args = tc
                .function
                .as_ref()
                .and_then(|f| f.arguments.as_ref())
                .map(|a| serde_json::to_string(a).unwrap_or_default())
                .unwrap_or_else(|| "{}".into());

            let item = if custom_names.contains(&name) {
                OutputItem::CustomToolCall(CustomToolCallItem {
                    id: call_id.clone(),
                    status: "completed".into(),
                    name,
                    call_id: call_id.clone(),
                    input: extract_custom_input(&args),
                })
            } else if name == "shell" || name == "container.exec" || name == "shell_command" {
                let cmd = extract_command_from_args(&args);
                OutputItem::LocalShellCall(LocalShellCallItem {
                    id: call_id.clone(),
                    status: "completed".into(),
                    name,
                    arguments: args,
                    call_id: call_id.clone(),
                    action: crate::codex_wire::schema::sse::ShellAction {
                        action_type: "exec",
                        command: cmd,
                    },
                    thought_signature: None,
                })
            } else {
                OutputItem::FunctionCall(FunctionCallItem {
                    id: call_id.clone(),
                    status: "completed".into(),
                    namespace: namespace_by_name.get(&name).cloned().flatten(),
                    name,
                    arguments: args,
                    call_id: call_id.clone(),
                    thought_signature: None,
                })
            };
            output.push(item);
        }

        if let Some(content) = msg.content.as_deref()
            && !content.is_empty()
        {
            output.push(OutputItem::Message(MessageItem {
                id: format!("msg_{}", now_ms()),
                role: "assistant",
                status: "completed".into(),
                content: vec![OutputContentPart::OutputText {
                    text: content.to_string(),
                }],
            }));
        }
    }

    let usage = z.usage.as_ref().map(|u| {
        let prompt = u.prompt_tokens.unwrap_or(0);
        let completion = u.completion_tokens.unwrap_or(0);
        let total = u.total_tokens.unwrap_or(prompt + completion);
        Usage {
            input_tokens: prompt,
            output_tokens: completion,
            total_tokens: total,
            input_tokens_details: None,
            output_tokens_details: None,
        }
    });

    ResponseObject {
        id: resp_id,
        object: "response",
        created_at: created,
        completed_at: None,
        model,
        status: "completed".into(),
        temperature: 1.0,
        top_p: 1.0,
        tool_choice: "auto".into(),
        tools: Vec::new(),
        parallel_tool_calls: true,
        store: false,
        metadata: Default::default(),
        output,
        usage,
    }
}

fn extract_command_from_args(args: &str) -> Vec<String> {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(args);
    match parsed {
        Ok(serde_json::Value::Object(map)) => match map.get("command") {
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codex_wire::schema::responses_wire::{ChatRequest, FunctionDef, Tool};

    #[test]
    fn maps_zai_tool_call_back_to_original_namespace() {
        let request = ChatRequest {
            model: "glm-5.2".into(),
            messages: Vec::new(),
            tools: vec![Tool {
                tool_type: "function".into(),
                function: Some(FunctionDef {
                    name: "spawn_agent".into(),
                    description: None,
                    parameters: None,
                }),
                name: None,
                namespace: Some("multi_agent".into()),
                description: None,
                parameters: None,
                strict: None,
                web_search: None,
                tools: Vec::new(),
            }],
            tool_choice: Some("auto".into()),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: false,
            store: false,
            metadata: Default::default(),
            previous_response_id: None,
            include: Vec::new(),
        };
        let response = ZaiChatResponse {
            id: Some("abc".into()),
            created: Some(1),
            model: Some("glm-5.2".into()),
            choices: vec![ZaiChoice {
                message: Some(ZaiChoiceMessage {
                    content: None,
                    tool_calls: vec![ZaiToolCall {
                        id: Some("call_1".into()),
                        function: Some(ZaiToolCallFn {
                            name: Some("spawn_agent".into()),
                            arguments: Some(serde_json::json!({
                                "message": "write a haiku",
                                "task_name": "poet"
                            })),
                        }),
                    }],
                }),
            }],
            usage: None,
        };

        let mapped = map_zai_response_to_responses_api(&response, &request);

        assert_eq!(mapped.output.len(), 1);
        let OutputItem::FunctionCall(call) = &mapped.output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("multi_agent"));
        assert_eq!(call.name, "spawn_agent");
        assert_eq!(call.call_id, "call_1");
    }

    #[test]
    fn maps_zai_shell_tool_call_and_usage_to_responses_output() {
        let request = ChatRequest {
            model: "glm-5.2".into(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: Some("auto".into()),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: false,
            store: false,
            metadata: Default::default(),
            previous_response_id: None,
            include: Vec::new(),
        };
        let response = ZaiChatResponse {
            id: Some("abc".into()),
            created: Some(42),
            model: Some("glm-5.2".into()),
            choices: vec![ZaiChoice {
                message: Some(ZaiChoiceMessage {
                    content: Some("done".into()),
                    tool_calls: vec![ZaiToolCall {
                        id: Some("call_shell".into()),
                        function: Some(ZaiToolCallFn {
                            name: Some("container.exec".into()),
                            arguments: Some(serde_json::json!({
                                "command": ["echo", "hello"]
                            })),
                        }),
                    }],
                }),
            }],
            usage: Some(ZaiUsage {
                prompt_tokens: Some(5),
                completion_tokens: Some(7),
                total_tokens: None,
            }),
        };

        let mapped = map_zai_response_to_responses_api(&response, &request);

        assert_eq!(mapped.id, "zai_abc");
        assert_eq!(mapped.created_at, 42);
        assert_eq!(mapped.model, "glm-5.2");
        assert_eq!(mapped.output.len(), 2);
        let OutputItem::LocalShellCall(shell) = &mapped.output[0] else {
            panic!("expected shell call");
        };
        assert_eq!(shell.call_id, "call_shell");
        assert_eq!(shell.action.command, vec!["echo", "hello"]);
        let OutputItem::Message(message) = &mapped.output[1] else {
            panic!("expected assistant message");
        };
        assert_eq!(message.role, "assistant");
        let Some(OutputContentPart::OutputText { text }) = message.content.first() else {
            panic!("expected output text");
        };
        assert_eq!(text, "done");
        let usage = mapped.usage.unwrap();
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.total_tokens, 12);
    }

    #[test]
    fn maps_apply_patch_function_call_to_custom_tool_call() {
        let request = ChatRequest {
            model: "glm-5.2".into(),
            messages: Vec::new(),
            tools: vec![Tool {
                tool_type: "custom".into(),
                function: None,
                name: Some("apply_patch".into()),
                namespace: None,
                description: None,
                parameters: None,
                strict: None,
                web_search: None,
                tools: Vec::new(),
            }],
            tool_choice: Some("auto".into()),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: false,
            store: false,
            metadata: Default::default(),
            previous_response_id: None,
            include: Vec::new(),
        };
        let patch = "*** Begin Patch\n*** Update File: a.txt\n-x\n+y\n*** End Patch";
        let response = ZaiChatResponse {
            id: Some("abc".into()),
            created: Some(1),
            model: Some("glm-5.2".into()),
            choices: vec![ZaiChoice {
                message: Some(ZaiChoiceMessage {
                    content: None,
                    tool_calls: vec![ZaiToolCall {
                        id: Some("call_1".into()),
                        function: Some(ZaiToolCallFn {
                            name: Some("apply_patch".into()),
                            arguments: Some(serde_json::json!({ "input": patch })),
                        }),
                    }],
                }),
            }],
            usage: None,
        };

        let mapped = map_zai_response_to_responses_api(&response, &request);

        assert_eq!(mapped.output.len(), 1);
        let OutputItem::CustomToolCall(call) = &mapped.output[0] else {
            panic!("expected a custom_tool_call output item");
        };
        assert_eq!(call.name, "apply_patch");
        assert_eq!(call.call_id, "call_1");
        assert_eq!(call.status, "completed");
        assert_eq!(call.input, patch);
    }

    #[test]
    fn parses_retry_after_delta_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "2".parse().unwrap());

        assert_eq!(
            parse_retry_after_header(&headers),
            Some(Duration::from_secs(2))
        );
    }
}
