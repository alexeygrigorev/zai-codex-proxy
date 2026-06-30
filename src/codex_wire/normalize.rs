use crate::codex_wire::schema::responses_wire::{
    ChatContent, ChatMessage, ChatRequest, Content, InputItem, Instructions, ResponsesInput,
    ResponsesRequest, TextPart, Tool, ToolCall, ToolCallFunction,
};

pub fn normalize_responses_request(req: &ResponsesRequest) -> ChatRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    if let Some(instructions) = &req.instructions {
        let content = instructions_text(instructions);
        if !content.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: Some(ChatContent::Text(content)),
                reasoning_content: None,
                thought_signature: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            });
        }
    }

    if let Some(input) = &req.input {
        match input {
            ResponsesInput::Text(text) => {
                messages.push(ChatMessage {
                    role: "user".into(),
                    content: Some(ChatContent::Text(text.clone())),
                    reasoning_content: None,
                    thought_signature: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                });
            }
            ResponsesInput::Items(items) => {
                for item in items {
                    process_input_item(item, &mut messages);
                }
            }
        }
    } else if let Some(messages_input) = &req.messages {
        messages.extend(messages_input.clone());
    }

    let tools = req
        .tools
        .as_ref()
        .map(|tools| tools.iter().cloned().map(normalize_tool).collect())
        .unwrap_or_default();

    ChatRequest {
        model: req.model.clone(),
        messages,
        tools,
        tool_choice: req.tool_choice.clone(),
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stream: req.stream.unwrap_or(false),
        store: req.store.unwrap_or(false),
        metadata: req.metadata.clone().unwrap_or_default(),
        previous_response_id: req.previous_response_id.clone(),
        include: req.include.clone().unwrap_or_default(),
    }
}

fn instructions_text(instructions: &Instructions) -> String {
    match instructions {
        Instructions::Text(text) => text.clone(),
        Instructions::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                TextPart::Text(text) => text.clone(),
                TextPart::Obj { text } => text.clone(),
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

fn process_input_item(item: &InputItem, messages: &mut Vec<ChatMessage>) {
    let item_type = if item.item_type.is_empty() {
        "message"
    } else {
        item.item_type.as_str()
    };

    match item_type {
        "agent_message" | "agentMessage" => process_agent_message(item, messages),
        "message" => process_message(item, messages),
        "reasoning" => process_reasoning(item, messages),
        "function_call" | "commandExecution" | "local_shell_call" | "fileChange"
        | "custom_tool_call" | "web_search_call" => process_tool_call(item, messages),
        "function_call_output"
        | "commandExecutionOutput"
        | "fileChangeOutput"
        | "custom_tool_call_output" => process_tool_output(item, messages),
        _ => {}
    }
}

fn process_message(item: &InputItem, messages: &mut Vec<ChatMessage>) {
    let role = item.role.as_deref().unwrap_or("user");
    let role = if role == "developer" { "system" } else { role };
    let reasoning_content = item.reasoning_content.clone().filter(|s| !s.is_empty());
    let mut content = extract_content_text(item.content.as_ref());
    if content.is_empty() {
        content = item.encrypted_content.clone().unwrap_or_default();
    }

    if role == "assistant" || role == "model" {
        let idx = ensure_last_assistant(messages);
        let msg = &mut messages[idx];
        let current_content = chat_content_to_string(msg.content.as_ref());
        let merged = format!("{current_content}{content}");
        msg.content = Some(ChatContent::Text(merged));

        if let Some(rc) = reasoning_content {
            let current_rc = msg.reasoning_content.clone().unwrap_or_default();
            msg.reasoning_content = Some(format!("{current_rc}{rc}"));
        }
        if let Some(sig) = item.thought_signature.clone() {
            msg.thought_signature = Some(sig);
        }
        return;
    }

    messages.push(ChatMessage {
        role: role.to_string(),
        content: Some(ChatContent::Text(content)),
        reasoning_content,
        thought_signature: item.thought_signature.clone(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
    });
}

fn process_agent_message(item: &InputItem, messages: &mut Vec<ChatMessage>) {
    let content = item_content_text(item);
    if content.is_empty() {
        return;
    }

    let author = item.author.as_deref().unwrap_or("unknown");
    let recipient = item.recipient.as_deref().unwrap_or("unknown");
    let message_type = if recipient.starts_with(author) && recipient.len() > author.len() {
        "NEW_TASK"
    } else {
        "MESSAGE"
    };
    let task_name = if message_type == "NEW_TASK" {
        format!("Task name: {recipient}\n")
    } else {
        String::new()
    };
    let text =
        format!("Message Type: {message_type}\n{task_name}Sender: {author}\nPayload:\n{content}");

    messages.push(ChatMessage {
        role: "user".into(),
        content: Some(ChatContent::Text(text)),
        reasoning_content: None,
        thought_signature: item.thought_signature.clone(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
    });
}

fn process_reasoning(item: &InputItem, messages: &mut Vec<ChatMessage>) {
    let content = item_content_text(item);
    if content.is_empty() {
        return;
    }

    let idx = ensure_last_assistant(messages);
    let msg = &mut messages[idx];
    let current_rc = msg.reasoning_content.clone().unwrap_or_default();
    msg.reasoning_content = Some(format!("{current_rc}{content}"));
    if let Some(sig) = item.thought_signature.clone() {
        msg.thought_signature = Some(sig);
    }
}

fn item_content_text(item: &InputItem) -> String {
    let mut content = extract_content_text(item.content.as_ref());
    if content.is_empty() {
        content = item.encrypted_content.clone().unwrap_or_default();
    }
    content
}

fn process_tool_call(item: &InputItem, messages: &mut Vec<ChatMessage>) {
    let call_id = item
        .call_id
        .as_deref()
        .or(item.id.as_deref())
        .unwrap_or("call_unknown")
        .to_string();

    let item_type = if item.item_type.is_empty() {
        "function_call"
    } else {
        item.item_type.as_str()
    };

    let name = match item.name.as_deref() {
        Some(n) => match item_type {
            "commandExecution" => "run_shell_command",
            "local_shell_call" => "local_shell_command",
            "fileChange" => "write_file",
            "web_search_call" => "web_search",
            _ => n,
        }
        .to_string(),
        None => "unknown".to_string(),
    };

    let args = item
        .arguments
        .as_ref()
        .or(item.input.as_ref())
        .or(item.action.as_ref());

    let args_str = args_to_json_string(args, item_type, item);

    let idx = ensure_last_assistant(messages);
    let msg = &mut messages[idx];
    msg.tool_calls.push(ToolCall {
        id: call_id.clone(),
        call_type: "function".into(),
        function: ToolCallFunction {
            name,
            arguments: args_str,
        },
    });

    if msg.thought_signature.is_none() {
        msg.thought_signature = item.thought_signature.clone();
    }
    if let Some(thought) = item.thought.clone() {
        let current_rc = msg.reasoning_content.clone().unwrap_or_default();
        msg.reasoning_content = Some(format!("{current_rc}{thought}"));
    }
}

fn args_to_json_string(
    args: Option<&serde_json::Value>,
    item_type: &str,
    item: &InputItem,
) -> String {
    // Custom (freeform) tools carry their payload in `input` as a raw string.
    // Re-wrap it as a JSON object so the prior tool call is valid arguments for
    // the synthesized `apply_patch` function tool we send upstream.
    if item_type == "custom_tool_call" {
        let input = item.input.as_ref().and_then(|v| v.as_str()).unwrap_or("");
        return format!(
            "{{\"input\":{}}}",
            serde_json::to_string(input).unwrap_or_default()
        );
    }
    if let Some(serde_json::Value::String(s)) = args {
        return s.clone();
    }
    if let Some(v) = args {
        return serde_json::to_string(v).unwrap_or_default();
    }

    match item_type {
        "commandExecution" => {
            let command = item.command.as_deref().unwrap_or("");
            let dir_path = item.cwd.as_deref().unwrap_or(".");
            format!(
                "{{\"command\":{},\"dir_path\":{}}}",
                serde_json::to_string(command).unwrap_or_default(),
                serde_json::to_string(dir_path).unwrap_or_default()
            )
        }
        "local_shell_call" => {
            let command = item
                .action
                .as_ref()
                .and_then(|a| match a {
                    serde_json::Value::Object(m) => m.get("exec"),
                    _ => None,
                })
                .and_then(|exec| match exec {
                    serde_json::Value::Object(m) => m.get("command"),
                    _ => None,
                })
                .map(|v| serde_json::to_string(v).unwrap_or_default())
                .unwrap_or_else(|| "[]".into());
            format!("{{\"command\":{command}}}")
        }
        "fileChange" => {
            let path = item
                .changes
                .as_ref()
                .and_then(|c| c.first())
                .map(|c| c.path.as_str())
                .unwrap_or("unknown");
            format!(
                "{{\"file_path\":{}}}",
                serde_json::to_string(path).unwrap_or_default()
            )
        }
        _ => "{}".into(),
    }
}

fn process_tool_output(item: &InputItem, messages: &mut Vec<ChatMessage>) {
    let call_id = item
        .call_id
        .as_deref()
        .or(item.id.as_deref())
        .unwrap_or("call_unknown")
        .to_string();

    let output_raw = item
        .output
        .as_ref()
        .or(item.content.as_ref())
        .or(item.stdout.as_ref());

    let mut content = extract_content_text(output_raw);
    if content.is_empty()
        && let Some(stderr) = item.stderr.as_deref()
    {
        content = format!("Error: {stderr}");
    }

    messages.push(ChatMessage {
        role: "tool".into(),
        content: Some(ChatContent::Text(content)),
        reasoning_content: None,
        thought_signature: None,
        tool_calls: Vec::new(),
        tool_call_id: Some(call_id),
        name: None,
    });
}

fn extract_content_text(content: Option<&Content>) -> String {
    match content {
        None => String::new(),
        Some(Content::Text(text)) => text.clone(),
        Some(Content::Parts(parts)) => parts
            .iter()
            .map(|part| match part.part_type.as_str() {
                "input_text" | "text" | "output_text" => part.text.clone().unwrap_or_default(),
                "encrypted_content" => part.encrypted_content.clone().unwrap_or_default(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(Content::Json(_)) => String::new(),
    }
}

fn ensure_last_assistant(messages: &mut Vec<ChatMessage>) -> usize {
    if let Some(last) = messages.last()
        && last.role == "assistant"
    {
        return messages.len() - 1;
    }
    messages.push(ChatMessage {
        role: "assistant".into(),
        content: None,
        reasoning_content: None,
        thought_signature: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
    });
    messages.len() - 1
}

fn chat_content_to_string(content: Option<&ChatContent>) -> String {
    match content {
        None => String::new(),
        Some(ChatContent::Text(text)) => text.clone(),
        Some(ChatContent::Parts(parts)) => parts
            .iter()
            .map(|part| match part.part_type.as_str() {
                "input_text" | "text" | "output_text" => part.text.clone().unwrap_or_default(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

fn normalize_tool(mut tool: Tool) -> Tool {
    if tool.tool_type == "function"
        && tool.function.is_none()
        && let Some(name) = tool.name.clone()
    {
        tool.function = Some(crate::codex_wire::schema::responses_wire::FunctionDef {
            name,
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        });
    }
    tool
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codex_wire::schema::responses_wire::ContentPart;

    #[test]
    fn encrypted_agent_message_becomes_chat_text() {
        let req = ResponsesRequest {
            model: "test-model".into(),
            input: Some(ResponsesInput::Items(vec![InputItem {
                item_type: "agentMessage".into(),
                id: None,
                call_id: None,
                role: Some("user".into()),
                author: Some("/root".into()),
                recipient: Some("/root/worker".into()),
                name: None,
                content: Some(Content::Parts(vec![ContentPart {
                    part_type: "encrypted_content".into(),
                    text: None,
                    encrypted_content: Some("deliver this task".into()),
                    image_url: None,
                }])),
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
            }])),
            messages: None,
            instructions: None,
            previous_response_id: None,
            store: None,
            metadata: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_output_tokens: None,
            stream: None,
            include: None,
        };

        let normalized = normalize_responses_request(&req);

        assert_eq!(normalized.messages.len(), 1);
        assert_eq!(normalized.messages[0].role, "user");
        let Some(ChatContent::Text(content)) = normalized.messages[0].content.as_ref() else {
            panic!("expected text content");
        };
        assert_eq!(
            content,
            "Message Type: NEW_TASK\nTask name: /root/worker\nSender: /root\nPayload:\ndeliver this task"
        );
    }

    #[test]
    fn encrypted_reasoning_becomes_reasoning_text() {
        let req = ResponsesRequest {
            model: "test-model".into(),
            input: Some(ResponsesInput::Items(vec![InputItem {
                item_type: "reasoning".into(),
                id: None,
                call_id: None,
                role: None,
                author: None,
                recipient: None,
                name: None,
                content: None,
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
                encrypted_content: Some("encoded reasoning".into()),
            }])),
            messages: None,
            instructions: None,
            previous_response_id: None,
            store: None,
            metadata: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_output_tokens: None,
            stream: None,
            include: None,
        };

        let normalized = normalize_responses_request(&req);

        assert_eq!(normalized.messages.len(), 1);
        assert_eq!(normalized.messages[0].role, "assistant");
        assert_eq!(
            normalized.messages[0].reasoning_content,
            Some("encoded reasoning".into())
        );
    }

    #[test]
    fn flat_function_tool_is_wrapped_for_chat_completions() {
        let req = ResponsesRequest {
            model: "test-model".into(),
            input: Some(ResponsesInput::Text("use a tool".into())),
            messages: None,
            instructions: None,
            previous_response_id: None,
            store: None,
            metadata: None,
            tools: Some(vec![Tool {
                tool_type: "function".into(),
                function: None,
                name: Some("spawn_agent".into()),
                namespace: Some("multi_agent".into()),
                description: Some("Create an agent".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    }
                })),
                strict: None,
                web_search: None,
                tools: Vec::new(),
            }]),
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_output_tokens: None,
            stream: None,
            include: None,
        };

        let normalized = normalize_responses_request(&req);

        let tools = normalized.tools;
        assert_eq!(tools.len(), 1);
        let function = tools[0].function.as_ref().expect("expected function");
        assert_eq!(function.name, "spawn_agent");
        assert_eq!(function.description.as_deref(), Some("Create an agent"));
        assert!(function.parameters.is_some());
        assert_eq!(tools[0].namespace.as_deref(), Some("multi_agent"));
    }

    #[test]
    fn custom_tool_call_history_is_rewrapped_as_input_argument() {
        // On turn 2+, Codex echoes the prior `custom_tool_call` (a freeform V4A
        // patch in `input`) back in the request. We must re-wrap that payload as
        // `{"input": "..."}` so it is valid arguments for the synthesized
        // `apply_patch` function tool we send upstream.
        let patch = "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch";
        let req = ResponsesRequest {
            model: "glm-5.2".into(),
            input: Some(ResponsesInput::Items(vec![InputItem {
                item_type: "custom_tool_call".into(),
                id: None,
                call_id: Some("call_1".into()),
                role: None,
                author: None,
                recipient: None,
                name: Some("apply_patch".into()),
                content: None,
                reasoning_content: None,
                thought_signature: None,
                thought: None,
                arguments: None,
                input: Some(serde_json::Value::String(patch.into())),
                action: None,
                command: None,
                cwd: None,
                working_directory: None,
                changes: None,
                output: None,
                stdout: None,
                stderr: None,
                encrypted_content: None,
            }])),
            messages: None,
            instructions: None,
            previous_response_id: None,
            store: None,
            metadata: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_output_tokens: None,
            stream: None,
            include: None,
        };

        let normalized = normalize_responses_request(&req);

        // The custom_tool_call is attached to an assistant message as a tool call.
        let assistant = normalized
            .messages
            .iter()
            .find(|m| m.role == "assistant")
            .expect("assistant message with tool call");
        let call = assistant
            .tool_calls
            .first()
            .expect("tool call reattached to history");
        assert_eq!(call.function.name, "apply_patch");
        let parsed: serde_json::Value =
            serde_json::from_str(&call.function.arguments).expect("arguments are valid JSON");
        assert_eq!(parsed["input"], patch);
        assert_eq!(call.id, "call_1");
    }
}
