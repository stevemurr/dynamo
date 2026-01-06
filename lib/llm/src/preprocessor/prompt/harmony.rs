// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Harmony format prompt rendering for GPT-OSS models
//!
//! This module provides native Rust implementation of OpenAI's Harmony format
//! for GPT-OSS models. The Harmony format uses special tokens to structure
//! conversations, tool calls, and tool results.
//!
//! Reference: https://cookbook.openai.com/articles/openai-harmony

use anyhow::{Context, Result};
use openai_harmony::chat::{Author, Conversation, DeveloperContent, Message, Role, SystemContent, ToolDescription, ToolNamespaceConfig};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use serde_json::Value as JsonValue;
use std::sync::OnceLock;

static HARMONY_ENCODING: OnceLock<Result<HarmonyEncoding, String>> = OnceLock::new();

fn get_encoding() -> Result<&'static HarmonyEncoding> {
    let result = HARMONY_ENCODING.get_or_init(|| {
        load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .map_err(|e| e.to_string())
    });
    
    match result {
        Ok(enc) => Ok(enc),
        Err(e) => Err(anyhow::anyhow!("Failed to load Harmony encoding: {}", e)),
    }
}

/// GPT-OSS Harmony Prompt Formatter
///
/// Implements OAIPromptFormatter for GPT-OSS models using the Harmony format.
/// This formatter converts OpenAI-style chat messages to Harmony tokens,
/// properly handling tool calls and tool results.
#[derive(Debug, Default)]
pub struct HarmonyFormatter;

impl HarmonyFormatter {
    pub fn new() -> Self {
        Self
    }
}

impl super::OAIPromptFormatter for HarmonyFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn render(&self, req: &dyn super::OAIChatLikeRequest) -> Result<String> {
        let enc = get_encoding()?;
        
        // Convert OpenAI messages to Harmony messages
        let messages = convert_openai_to_harmony(req)?;
        
        // Create conversation from messages
        let conversation = Conversation::from_messages(messages);
        
        // Render to tokens for completion (appends assistant role)
        let tokens = enc.render_conversation_for_completion(&conversation, Role::Assistant, None)
            .context("Failed to render conversation to Harmony tokens")?;
        
        // Decode tokens back to string for the tokenizer using decode_utf8
        let prompt = enc.tokenizer().decode_utf8(&tokens)
            .context("Failed to decode Harmony tokens to string")?;
        
        tracing::debug!("Harmony formatted prompt length: {} chars", prompt.len());
        
        Ok(prompt)
    }
}

/// Convert OpenAI-format messages to Harmony Message objects
fn convert_openai_to_harmony(req: &dyn super::OAIChatLikeRequest) -> Result<Vec<Message>> {
    let messages_value = req.messages();
    let messages_json = serde_json::to_value(&messages_value)
        .context("Failed to convert messages to JSON")?;
    let messages_array = messages_json
        .as_array()
        .context("Messages is not an array")?;
    
    // Get tools if available for the developer message
    let tools = req.tools();
    let tools_array = tools
        .as_ref()
        .and_then(|t| serde_json::to_value(t).ok())
        .and_then(|v| v.as_array().cloned());
    
    let mut harmony_messages = Vec::with_capacity(messages_array.len());
    
    // Track tool call mappings: tool_call_id -> function_name
    // We need this to properly name tool result messages
    let mut tool_call_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    
    // First pass: collect tool call mappings from assistant messages
    for msg in messages_array {
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
            for tool_call in tool_calls {
                if let (Some(id), Some(name)) = (
                    tool_call.get("id").and_then(|i| i.as_str()),
                    tool_call.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str())
                ) {
                    tool_call_map.insert(id.to_string(), name.to_string());
                }
            }
        }
    }
    
    // Second pass: convert messages
    for msg in messages_array {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        
        let harmony_msg = match role {
            "system" => {
                let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                // SystemContent uses with_model_identity for setting instructions/identity
                let sys_content = if !content.is_empty() {
                    SystemContent::new().with_model_identity(content)
                } else {
                    SystemContent::new()
                };
                Message::from_role_and_content(Role::System, sys_content)
            },
            
            "developer" => {
                let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                let mut dev_content = DeveloperContent::new();
                if !content.is_empty() {
                    dev_content = dev_content.with_instructions(content);
                }
                // Add tools to developer message if available
                if let Some(ref tools) = tools_array {
                    if let Some(tool_namespace) = build_tool_namespace(tools) {
                        dev_content = dev_content.with_tools(tool_namespace);
                    }
                }
                Message::from_role_and_content(Role::Developer, dev_content)
            },
            
            "user" => {
                let content = get_message_content(msg);
                Message::from_role_and_content(Role::User, content.as_str())
            },
            
            "assistant" => {
                let content = get_message_content(msg);
                
                // Check for tool calls
                if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                    if !tool_calls.is_empty() {
                        // For assistant messages with tool calls, we need to render them
                        // in the commentary channel with the function recipient
                        let mut messages_for_tool_calls = Vec::new();
                        
                        // First, if there's regular content, add it
                        if !content.is_empty() {
                            messages_for_tool_calls.push(
                                Message::from_role_and_content(Role::Assistant, content.as_str())
                            );
                        }
                        
                        // Then add each tool call as a separate message in commentary channel
                        for tool_call in tool_calls {
                            if let Some(function) = tool_call.get("function") {
                                let name = function.get("name").and_then(|n| n.as_str()).unwrap_or("");
                                let arguments = function.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}");
                                
                                let tool_msg = Message::from_role_and_content(Role::Assistant, arguments)
                                    .with_channel("commentary")
                                    .with_recipient(&format!("functions.{}", name))
                                    .with_content_type("json");
                                
                                messages_for_tool_calls.push(tool_msg);
                            }
                        }
                        
                        // Add all messages and continue to next
                        harmony_messages.extend(messages_for_tool_calls);
                        continue;
                    }
                }
                
                // Regular assistant message without tool calls
                Message::from_role_and_content(Role::Assistant, content.as_str())
            },
            
            "tool" => {
                // This is the critical fix - tool results need special Harmony formatting
                let content = get_message_content(msg);
                let tool_call_id = msg.get("tool_call_id").and_then(|t| t.as_str()).unwrap_or("");
                
                // Look up the function name from our mapping
                let func_name = tool_call_map
                    .get(tool_call_id)
                    .cloned()
                    .unwrap_or_else(|| extract_function_name_from_id(tool_call_id));
                
                // Create tool message with proper author name and routing
                // Format: <|start|>functions.{name} to=assistant<|channel|>commentary<|message|>{content}<|end|>
                let author = Author::new(Role::Tool, format!("functions.{}", func_name));
                Message::from_author_and_content(author, content.as_str())
                    .with_channel("commentary")
                    .with_recipient("assistant")
            },
            
            _ => {
                tracing::warn!("Unknown message role '{}', skipping", role);
                continue;
            }
        };
        
        harmony_messages.push(harmony_msg);
    }
    
    Ok(harmony_messages)
}

/// Extract message content, handling both string and array formats
fn get_message_content(msg: &JsonValue) -> String {
    match msg.get("content") {
        Some(JsonValue::String(s)) => s.clone(),
        Some(JsonValue::Array(arr)) => {
            // Handle content array (multimodal messages)
            arr.iter()
                .filter_map(|part| {
                    if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                        part.get("text").and_then(|t| t.as_str()).map(String::from)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("")
        },
        _ => String::new(),
    }
}

/// Build tool namespace from OpenAI tool definitions
fn build_tool_namespace(tools: &[JsonValue]) -> Option<ToolNamespaceConfig> {
    let tool_descriptions: Vec<ToolDescription> = tools
        .iter()
        .filter_map(|tool| {
            let function = tool.get("function")?;
            let name = function.get("name")?.as_str()?;
            let description = function.get("description").and_then(|d| d.as_str()).unwrap_or("");
            let parameters = function.get("parameters").cloned();
            
            Some(ToolDescription::new(name, description, parameters))
        })
        .collect();
    
    if tool_descriptions.is_empty() {
        None
    } else {
        Some(ToolNamespaceConfig::new("functions", None, tool_descriptions))
    }
}

/// Extract function name from tool_call_id
/// This is a fallback when we can't find the mapping
fn extract_function_name_from_id(tool_call_id: &str) -> String {
    // Tool call IDs are usually like "call-1", "call-2", etc.
    // We can't really extract a function name from these, so return a default
    // The actual function name should come from the preceding assistant message
    tracing::warn!("Could not find function name for tool_call_id: {}", tool_call_id);
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_get_encoding() {
        let result = get_encoding();
        assert!(result.is_ok(), "Should load Harmony encoding");
    }

    #[test]
    fn test_get_message_content_string() {
        let msg = json!({"role": "user", "content": "Hello"});
        assert_eq!(get_message_content(&msg), "Hello");
    }

    #[test]
    fn test_get_message_content_array() {
        let msg = json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "World"}
            ]
        });
        assert_eq!(get_message_content(&msg), "Hello World");
    }

    #[test]
    fn test_build_tool_namespace() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        })];
        
        let namespace = build_tool_namespace(&tools);
        assert!(namespace.is_some());
    }
}
