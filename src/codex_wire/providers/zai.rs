use serde::Serialize;

use crate::codex_wire::schema::responses_wire::{
    ChatContent, ChatMessage, ChatRequest, FunctionDef, Tool, ToolCall,
};

#[derive(Clone, Debug, Serialize)]
pub struct ZaiChatRequest {
    pub model: String,
    pub messages: Vec<ZaiMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ZaiThinkingConfig>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ZaiMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ZaiThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
}

pub fn build_zai_chat_request(
    request: &ChatRequest,
    model: &str,
    thinking_enabled: Option<bool>,
) -> ZaiChatRequest {
    let compatible_tools = zai_compatible_tools(request.tools.clone());
    let tools = if compatible_tools.is_empty() {
        None
    } else {
        Some(compatible_tools.into_iter().map(transform_tool).collect())
    };
    ZaiChatRequest {
        model: model.to_string(),
        messages: request.messages.iter().map(to_zai_message).collect(),
        stream: request.stream,
        tools,
        tool_choice: request.tool_choice.clone(),
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        thinking: thinking_enabled.map(zai_thinking_config),
    }
}

pub fn zai_compatible_tools(tools: Vec<Tool>) -> Vec<Tool> {
    let mut out = Vec::new();
    for mut tool in tools {
        match tool.tool_type.as_str() {
            "function" => {
                if !ensure_function_schema(&mut tool) {
                    continue;
                }
                tool.tools.clear();
                out.push(tool);
            }
            "web_search" => {
                tool.tools.clear();
                if tool.web_search.is_none() {
                    tool.web_search = Some(default_web_search_config());
                }
                out.push(tool);
            }
            "namespace" => {
                let namespace = tool.name.clone();
                for mut child in std::mem::take(&mut tool.tools) {
                    if child.tool_type != "function" {
                        continue;
                    }
                    if !ensure_function_schema(&mut child) {
                        continue;
                    }
                    if child.namespace.is_none() {
                        child.namespace = namespace.clone();
                    }
                    if child.description.is_none() {
                        child.description = child
                            .function
                            .as_ref()
                            .and_then(|function| function.description.clone());
                    }
                    if let Some(ns_description) = tool.description.as_deref()
                        && !ns_description.trim().is_empty()
                    {
                        let child_description = child.description.clone().unwrap_or_default();
                        child.description = Some(if child_description.trim().is_empty() {
                            ns_description.to_string()
                        } else {
                            format!("{ns_description}\n\n{child_description}")
                        });
                    }
                    child.tools.clear();
                    out.push(child);
                }
            }
            "custom" => {
                // Codex freeform/custom tools (e.g. `apply_patch`) carry a grammar
                // definition that the Z.AI chat completions API cannot express. We
                // synthesize an equivalent `function` tool so the model can invoke
                // it, then translate the resulting call back into a
                // `custom_tool_call` on the response path (see `custom_tool_names`).
                let Some(name) = tool.name.clone() else {
                    continue;
                };
                let description = Some(custom_function_description(
                    &name,
                    tool.description.as_deref(),
                ));
                tool.tool_type = "function".into();
                tool.function = Some(FunctionDef {
                    name,
                    description,
                    parameters: Some(custom_function_parameters()),
                });
                tool.name = None;
                tool.description = None;
                tool.parameters = None;
                tool.strict = None;
                tool.namespace = None;
                tool.web_search = None;
                tool.tools.clear();
                out.push(tool);
            }
            _ => {}
        }
    }
    out
}

/// Names of every `custom` tool in a request. Tool calls targeting these names
/// must be emitted as `custom_tool_call` items (not `function_call`) so Codex's
/// runtime applies them like native tools.
pub fn custom_tool_names(tools: &[Tool]) -> std::collections::HashSet<String> {
    tools
        .iter()
        .filter(|tool| tool.tool_type == "custom")
        .filter_map(|tool| tool.name.clone())
        .collect()
}

/// Pull the freeform payload out of a synthesized function call's JSON
/// arguments (`{"input": "..."}`). Falls back to the raw arguments string when
/// the model did not produce valid JSON, so we never lose a patch.
pub fn extract_custom_input(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("input")
                .and_then(|input| input.as_str().map(|text| text.to_string()))
        })
        .unwrap_or_else(|| arguments.to_string())
}

const APPLY_PATCH_GUIDE: &str = "Edit files by applying a V4A patch. Pass the entire patch as the `input` string.\n\n\
Patch format:\n\
*** Begin Patch\n\
*** Add File: path/to/new_file\n\
+line to add\n\
*** Update File: path/to/existing_file\n\
 context line\n\
-removed line\n\
+added line\n\
*** Delete File: path/to/file\n\
*** End Patch\n\n\
For *** Update File, prefix unchanged context lines with a space, removed lines with `-`, and added lines with `+`. \
Use `@@` plus surrounding context to locate each change. A single patch may change many files.";

fn custom_function_description(name: &str, existing: Option<&str>) -> String {
    let guide = if name == "apply_patch" {
        APPLY_PATCH_GUIDE.to_string()
    } else {
        "Invoke this tool with the full freeform payload as the `input` string.".to_string()
    };
    match existing {
        Some(description) if !description.trim().is_empty() => {
            format!("{description}\n\n{guide}")
        }
        _ => guide,
    }
}

fn custom_function_parameters() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "The complete V4A patch (*** Begin Patch ... *** End Patch)."
            }
        },
        "required": ["input"],
        "additionalProperties": false
    })
}

fn ensure_function_schema(tool: &mut Tool) -> bool {
    if tool.tool_type != "function" {
        return true;
    }
    if tool.function.is_none()
        && let Some(name) = tool.name.clone()
    {
        tool.function = Some(crate::codex_wire::schema::responses_wire::FunctionDef {
            name,
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        });
    }
    tool.function.is_some()
}

fn default_web_search_config() -> crate::codex_wire::schema::responses_wire::WebSearchConfig {
    crate::codex_wire::schema::responses_wire::WebSearchConfig {
        enable: Some(true),
        search_engine: Some("search_pro_jina".into()),
    }
}

fn zai_thinking_config(enabled: bool) -> ZaiThinkingConfig {
    ZaiThinkingConfig {
        thinking_type: if enabled {
            "enabled".into()
        } else {
            "disabled".into()
        },
    }
}

fn to_zai_message(message: &ChatMessage) -> ZaiMessage {
    let role = if message.role == "developer" {
        "system".to_string()
    } else {
        message.role.clone()
    };
    let content = message.content.as_ref().map(chat_content_to_string);
    let tool_calls = if message.tool_calls.is_empty() {
        None
    } else {
        Some(message.tool_calls.clone())
    };
    ZaiMessage {
        role,
        content,
        tool_calls,
        tool_call_id: message.tool_call_id.clone(),
        name: message.name.clone(),
    }
}

fn chat_content_to_string(content: &ChatContent) -> String {
    match content {
        ChatContent::Text(text) => text.clone(),
        ChatContent::Parts(parts) => parts
            .iter()
            .map(|part| part.text.clone().unwrap_or_default())
            .collect::<Vec<_>>()
            .join(""),
    }
}

fn transform_tool(mut tool: Tool) -> Tool {
    if tool.tool_type == "function" {
        tool.strict = None;
        tool.name = None;
        tool.description = None;
        tool.parameters = None;
        tool.web_search = None;
    }
    tool.namespace = None;
    tool.tools.clear();
    if tool.tool_type == "web_search" && tool.web_search.is_none() {
        tool.web_search = Some(default_web_search_config());
    }
    tool
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codex_wire::schema::responses_wire::{
        ChatContent, FunctionDef, ToolCall, ToolCallFunction, WebSearchConfig,
    };

    #[test]
    fn build_zai_chat_request_serializes_web_search_tool() {
        let request = ChatRequest {
            model: "glm-5.2".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(ChatContent::Text("what changed today?".into())),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning_content: None,
                thought_signature: None,
            }],
            tools: vec![Tool {
                tool_type: "web_search".into(),
                function: None,
                name: None,
                namespace: Some("ignored".into()),
                description: None,
                parameters: None,
                strict: None,
                web_search: Some(WebSearchConfig {
                    enable: Some(true),
                    search_engine: Some("search-prime".into()),
                }),
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

        let payload = build_zai_chat_request(&request, "glm-5.2", Some(false));
        let value = serde_json::to_value(payload).unwrap();

        assert_eq!(value["tools"][0]["type"], "web_search");
        assert_eq!(value["tools"][0]["web_search"]["enable"], true);
        assert_eq!(
            value["tools"][0]["web_search"]["search_engine"],
            "search-prime"
        );
        assert!(value["tools"][0].get("namespace").is_none());
        assert!(value["tools"][0].get("tools").is_none());
    }

    #[test]
    fn build_zai_chat_request_maps_roles_tool_calls_and_thinking() {
        let request = ChatRequest {
            model: "codex-logical".into(),
            messages: vec![
                ChatMessage {
                    role: "developer".into(),
                    content: Some(ChatContent::Text("follow repo instructions".into())),
                    name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                    reasoning_content: None,
                    thought_signature: None,
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    name: None,
                    tool_call_id: None,
                    tool_calls: vec![ToolCall {
                        id: "call_1".into(),
                        call_type: "function".into(),
                        function: ToolCallFunction {
                            name: "container.exec".into(),
                            arguments: r#"{"command":["pwd"]}"#.into(),
                        },
                    }],
                    reasoning_content: None,
                    thought_signature: None,
                },
                ChatMessage {
                    role: "tool".into(),
                    content: Some(ChatContent::Text("/home/alexey/project".into())),
                    name: Some("container.exec".into()),
                    tool_call_id: Some("call_1".into()),
                    tool_calls: Vec::new(),
                    reasoning_content: None,
                    thought_signature: None,
                },
            ],
            tools: vec![Tool {
                tool_type: "function".into(),
                function: Some(FunctionDef {
                    name: "container.exec".into(),
                    description: Some("Run shell command".into()),
                    parameters: Some(serde_json::json!({"type": "object"})),
                }),
                name: None,
                namespace: Some("container".into()),
                description: None,
                parameters: None,
                strict: Some(true),
                web_search: None,
                tools: Vec::new(),
            }],
            tool_choice: Some("auto".into()),
            temperature: Some(0.2),
            top_p: Some(0.8),
            max_tokens: Some(1024),
            stream: true,
            store: false,
            metadata: Default::default(),
            previous_response_id: None,
            include: Vec::new(),
        };

        let payload = build_zai_chat_request(&request, "glm-5.2", Some(true));
        let value = serde_json::to_value(payload).unwrap();

        assert_eq!(value["model"], "glm-5.2");
        assert_eq!(value["stream"], true);
        assert_eq!(value["thinking"]["type"], "enabled");
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(value["messages"][1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            value["messages"][1]["tool_calls"][0]["function"]["name"],
            "container.exec"
        );
        assert_eq!(value["messages"][2]["role"], "tool");
        assert_eq!(value["messages"][2]["tool_call_id"], "call_1");
        assert_eq!(value["messages"][2]["name"], "container.exec");
        assert_eq!(value["tools"][0]["type"], "function");
        assert!(value["tools"][0].get("strict").is_none());
        assert!(value["tools"][0].get("namespace").is_none());
    }

    #[test]
    fn zai_compatible_tools_flattens_namespace_functions() {
        let tools = vec![Tool {
            tool_type: "namespace".into(),
            function: None,
            name: Some("multi_agent".into()),
            namespace: None,
            description: Some("Agent coordination".into()),
            parameters: None,
            strict: None,
            web_search: None,
            tools: vec![Tool {
                tool_type: "function".into(),
                function: Some(FunctionDef {
                    name: "spawn_agent".into(),
                    description: Some("Create an agent".into()),
                    parameters: None,
                }),
                name: None,
                namespace: None,
                description: None,
                parameters: None,
                strict: Some(true),
                web_search: None,
                tools: Vec::new(),
            }],
        }];

        let flattened = zai_compatible_tools(tools);

        assert_eq!(flattened.len(), 1);
        assert_eq!(flattened[0].tool_type, "function");
        assert_eq!(flattened[0].namespace.as_deref(), Some("multi_agent"));
        assert_eq!(
            flattened[0].description.as_deref(),
            Some("Agent coordination\n\nCreate an agent")
        );
        assert!(flattened[0].tools.is_empty());
    }

    #[test]
    fn flat_namespace_function_child_is_wrapped_for_zai() {
        let request = ChatRequest {
            model: "glm-5.2".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(ChatContent::Text("use a subagent".into())),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning_content: None,
                thought_signature: None,
            }],
            tools: vec![Tool {
                tool_type: "namespace".into(),
                function: None,
                name: Some("multi_agent".into()),
                namespace: None,
                description: Some("Agent coordination".into()),
                parameters: None,
                strict: None,
                web_search: None,
                tools: vec![Tool {
                    tool_type: "function".into(),
                    function: None,
                    name: Some("spawn_agent".into()),
                    namespace: None,
                    description: Some("Create an agent".into()),
                    parameters: Some(serde_json::json!({"type": "object"})),
                    strict: Some(true),
                    web_search: None,
                    tools: Vec::new(),
                }],
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

        let payload = build_zai_chat_request(&request, "glm-5.2", Some(false));
        let value = serde_json::to_value(payload).unwrap();

        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["function"]["name"], "spawn_agent");
        assert!(value["tools"][0].get("name").is_none());
        assert!(value["tools"][0].get("description").is_none());
        assert!(value["tools"][0].get("parameters").is_none());
        assert!(value["tools"][0].get("function").unwrap().is_object());
    }

    #[test]
    fn zai_compatible_tools_preserves_web_search_with_default_engine() {
        let tools = zai_compatible_tools(vec![Tool {
            tool_type: "web_search".into(),
            function: None,
            name: None,
            namespace: None,
            description: None,
            parameters: None,
            strict: None,
            web_search: None,
            tools: Vec::new(),
        }]);

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "web_search");
        let web_search = tools[0].web_search.as_ref().unwrap();
        assert_eq!(web_search.enable, Some(true));
        assert_eq!(web_search.search_engine.as_deref(), Some("search_pro_jina"));
        assert!(tools[0].tools.is_empty());
    }

    #[test]
    fn zai_compatible_tools_converts_custom_apply_patch_to_function() {
        let tools = vec![Tool {
            tool_type: "custom".into(),
            function: None,
            name: Some("apply_patch".into()),
            namespace: None,
            description: Some("Edit files.".into()),
            parameters: None,
            strict: None,
            web_search: None,
            tools: Vec::new(),
        }];

        let compat = zai_compatible_tools(tools);
        assert_eq!(compat.len(), 1);
        assert_eq!(compat[0].tool_type, "function");

        let function = compat[0]
            .function
            .as_ref()
            .expect("function def synthesized");
        assert_eq!(function.name, "apply_patch");
        let description = function.description.as_deref().expect("description set");
        assert!(description.contains("Edit files."));
        assert!(description.contains("*** Begin Patch"));

        let parameters = function
            .parameters
            .as_ref()
            .expect("parameters synthesized");
        assert_eq!(parameters["properties"]["input"]["type"], "string");
        assert_eq!(parameters["required"][0], "input");

        // Custom-only fields are cleared so Z.AI sees a clean function tool.
        assert!(compat[0].name.is_none());
        assert!(compat[0].web_search.is_none());
        assert!(compat[0].tools.is_empty());
    }

    #[test]
    fn custom_tool_names_collects_only_custom_tools() {
        let tools = vec![
            Tool {
                tool_type: "custom".into(),
                function: None,
                name: Some("apply_patch".into()),
                namespace: None,
                description: None,
                parameters: None,
                strict: None,
                web_search: None,
                tools: Vec::new(),
            },
            Tool {
                tool_type: "function".into(),
                function: None,
                name: Some("exec_command".into()),
                namespace: None,
                description: None,
                parameters: None,
                strict: None,
                web_search: None,
                tools: Vec::new(),
            },
        ];

        let names = custom_tool_names(&tools);
        assert!(names.contains("apply_patch"));
        assert!(!names.contains("exec_command"));
    }
}
