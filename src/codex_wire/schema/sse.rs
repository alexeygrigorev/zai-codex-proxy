use crate::codex_wire::schema::responses_wire::Tool;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub struct ResponseEvent<T: Serialize> {
    pub id: String,
    pub object: &'static str,
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub created_at: i64,
    pub sequence_number: u64,
    #[serde(flatten)]
    pub data: T,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseCreatedData {
    pub response: ResponseObject,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseCompletedData {
    pub response: ResponseObject,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseFailedData {
    pub response: FailedResponseObject,
}

#[derive(Clone, Debug, Serialize)]
pub struct FailedResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub status: &'static str,
    pub model: String,
    pub error: ResponseError,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseOutputItemAddedData {
    pub response_id: String,
    pub output_index: usize,
    pub item: OutputItem,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseOutputItemDoneData {
    pub response_id: String,
    pub output_index: usize,
    pub item: OutputItem,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseOutputTextDeltaData {
    pub response_id: String,
    pub item_id: String,
    pub output_index: i64,
    pub content_index: usize,
    pub delta: String,
}

/// Streaming delta for a function-call's `arguments` JSON fragment. Codex
/// incrementally parses these deltas to surface tool-call fields (for example
/// a `spawn_agent` `task_name`) before the call completes, which is what the
/// "Waiting for agents" panel relies on.
#[derive(Clone, Debug, Serialize)]
pub struct FunctionCallArgumentsDeltaData {
    pub response_id: String,
    pub item_id: String,
    pub output_index: i64,
    pub call_id: String,
    pub delta: String,
}

/// Terminal payload for a function-call's accumulated `arguments`.
#[derive(Clone, Debug, Serialize)]
pub struct FunctionCallArgumentsDoneData {
    pub response_id: String,
    pub item_id: String,
    pub output_index: i64,
    pub call_id: String,
    pub arguments: String,
}

/// Streaming delta for a custom tool call's freeform `input` payload
/// (e.g. `apply_patch`). Codex applies these incrementally like a native tool.
#[derive(Clone, Debug, Serialize)]
pub struct CustomToolCallInputDeltaData {
    pub response_id: String,
    pub item_id: String,
    pub output_index: i64,
    pub call_id: String,
    pub input_delta: String,
}

/// Terminal payload for a custom tool call's accumulated freeform `input`.
#[derive(Clone, Debug, Serialize)]
pub struct CustomToolCallInputDoneData {
    pub response_id: String,
    pub item_id: String,
    pub output_index: i64,
    pub call_id: String,
    pub input: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelsEtagData {
    pub etag: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ServerReasoningIncludedData {
    pub included: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RateLimitsData {
    pub primary: Option<Value>,
    pub secondary: Option<Value>,
    pub credits: CreditsData,
}

#[derive(Clone, Debug, Serialize)]
pub struct CreditsData {
    pub has_credits: bool,
    pub unlimited: bool,
    pub balance: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    pub model: String,
    pub status: String,

    pub temperature: f64,
    pub top_p: f64,
    pub tool_choice: String,
    pub tools: Vec<Tool>,
    pub parallel_tool_calls: bool,
    pub store: bool,
    pub metadata: BTreeMap<String, Value>,
    pub output: Vec<OutputItem>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Clone, Debug, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message(MessageItem),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningItem),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionCallItem),
    #[serde(rename = "local_shell_call")]
    LocalShellCall(LocalShellCallItem),
    // Codex registers file-editing as a "custom" (freeform, grammar) tool named
    // `apply_patch`. Its runtime expects patch results back as `custom_tool_call`
    // output items, so we mirror that shape exactly for a transparent round-trip.
    #[serde(rename = "custom_tool_call")]
    CustomToolCall(CustomToolCallItem),
}

#[derive(Clone, Debug, Serialize)]
pub struct CustomToolCallItem {
    pub id: String,
    pub status: String,
    pub name: String,
    pub call_id: String,
    // Freeform V4A patch payload (NOT JSON-encoded).
    pub input: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct MessageItem {
    pub id: String,
    pub role: &'static str,
    pub status: String,
    pub content: Vec<OutputContentPart>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReasoningItem {
    pub id: String,
    pub status: String,
    pub summary: Vec<SummaryPart>,
    pub content: Vec<ReasoningContentPart>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FunctionCallItem {
    pub id: String,
    pub status: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    pub arguments: String,
    pub call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalShellCallItem {
    pub id: String,
    pub status: String,
    pub name: String,
    pub arguments: String,
    pub call_id: String,
    pub action: ShellAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ShellAction {
    #[serde(rename = "type")]
    pub action_type: &'static str,
    pub command: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum OutputContentPart {
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum ReasoningContentPart {
    #[serde(rename = "reasoning_text")]
    ReasoningText { text: String },
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum SummaryPart {
    #[serde(rename = "summary_text")]
    SummaryText { text: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_failed_event_serializes_error_shape() {
        let evt = ResponseEvent {
            id: "evt_test".to_string(),
            object: "response.event",
            event_type: "response.failed",
            created_at: 123,
            sequence_number: 41,
            data: ResponseFailedData {
                response: FailedResponseObject {
                    id: "resp_test".to_string(),
                    object: "response",
                    created_at: 123,
                    status: "failed",
                    model: "glm-5-turbo".to_string(),
                    error: ResponseError {
                        code: "stream_timeout".to_string(),
                        message: "Upstream stream idle".to_string(),
                    },
                    metadata: BTreeMap::new(),
                },
            },
        };

        // Assert only load-bearing keys to keep this test resilient.
        let val = serde_json::to_value(evt).expect("event must serialize");
        assert_eq!(val["type"], "response.failed");
        assert_eq!(val["response"]["status"], "failed");
        assert_eq!(val["response"]["error"]["code"], "stream_timeout");
        assert_eq!(val["response"]["error"]["message"], "Upstream stream idle");
        assert_eq!(val["response"]["model"], "glm-5-turbo");
    }
}
