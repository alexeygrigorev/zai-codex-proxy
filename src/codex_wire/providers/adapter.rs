use crate::codex_wire::providers::zai::{ZaiChatRequest, build_zai_chat_request};
use crate::codex_wire::schema::responses_wire::ChatRequest;

#[derive(Debug, Clone)]
pub struct ProviderBuildContext {
    pub model: String,
    pub thinking_enabled: Option<bool>,
}

impl ProviderBuildContext {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            thinking_enabled: None,
        }
    }
}

pub fn build_zai_payload(request: &ChatRequest, ctx: &ProviderBuildContext) -> ZaiChatRequest {
    build_zai_chat_request(request, &ctx.model, ctx.thinking_enabled)
}
