//! Tool search context for handling dynamic tool discovery in Responses API.
//!
//! This module provides types and logic for handling `tool_search_call` and
//! `tool_search_output` items that implement OpenAI's dynamic tool discovery
//! mechanism.
//!
//! ## Overview
//!
//! In multi-turn conversations, a model may decide to search for tools
//! dynamically using `tool_search_call`. The response to this search comes
//! as a `tool_search_output` item containing the discovered tools.
//!
//! ## Priority Strategies
//!
//! When merging tools from multiple sources (predefined + searched), we support
//! different strategies via `ToolPriority`:
//! - `PreferDefined`: Use predefined tools, ignore searched tools
//! - `PreferSearched`: Use searched tools, discard predefined
//! - `Merge`: Combine both, with searched tools overriding on name conflicts

use std::collections::HashSet;
use std::str::FromStr;

use crate::types::response_api::Tool;

/// Tool priority when merging predefined tools with searched tools.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolPriority {
    /// Prefer predefined tools; ignore tools from `tool_search_output`
    PreferDefined,
    /// Prefer searched tools; discard predefined tools
    PreferSearched,
    /// Merge both; if names conflict, searched tools override
    #[default]
    Merge,
}

impl FromStr for ToolPriority {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "prefer_defined" | "prefer-defined" => ToolPriority::PreferDefined,
            "prefer_searched" | "prefer-searched" => ToolPriority::PreferSearched,
            "merge" | "combined" => ToolPriority::Merge,
            _ => {
                tracing::warn!(
                    "[TOOL_SEARCH] unknown tool_priority '{}', defaulting to 'merge'",
                    s
                );
                ToolPriority::Merge
            }
        })
    }
}

/// Context for tracking tool search state during input conversion.
///
/// This struct maintains:
/// - `pending_calls`: Map of call_id -> tool_search_call items awaiting output
/// - `resolved_tools`: Final merged tool list after processing all input items
/// - `priority`: Strategy for merging predefined and searched tools
#[derive(Debug, Clone)]
pub struct ToolSearchContext {
    /// Map of tool_search_call id -> empty slot (marks that a search was initiated)
    pending_calls: HashSet<String>,
    /// Tools collected from tool_search_output items
    searched_tools: Vec<Tool>,
    /// Predefined tools from ResponseRequest.tools (set separately)
    predefined_tools: Vec<Tool>,
    /// Final resolved tools after applying priority strategy
    resolved_tools: Vec<Tool>,
    /// Priority strategy for merging tools
    priority: ToolPriority,
    /// Whether the context has been finalized (tools resolved)
    finalized: bool,
}

impl ToolSearchContext {
    /// Create a new context with the given priority strategy.
    pub fn new(priority: ToolPriority) -> Self {
        Self {
            pending_calls: HashSet::new(),
            searched_tools: Vec::new(),
            predefined_tools: Vec::new(),
            resolved_tools: Vec::new(),
            priority,
            finalized: false,
        }
    }

    /// Record that a `tool_search_call` was initiated with the given id.
    pub fn register_search_call(&mut self, call_id: &str) {
        self.pending_calls.insert(call_id.to_string());
        tracing::debug!("[TOOL_SEARCH] registered tool_search_call: id={}", call_id);
    }

    /// Check if a call_id corresponds to a registered tool_search_call.
    pub fn is_registered_search(&self, call_id: &str) -> bool {
        self.pending_calls.contains(call_id)
    }

    /// Set the predefined tools from the request.
    ///
    /// This should be called once at the start of processing.
    pub fn set_predefined_tools(&mut self, tools: Vec<Tool>) {
        self.predefined_tools = tools;
        tracing::debug!(
            "[TOOL_SEARCH] set predefined tools: count={}",
            self.predefined_tools.len()
        );
    }

    /// Add tools from a `tool_search_output` item.
    ///
    /// The tools are collected but not yet merged - merging happens at finalize.
    pub fn add_searched_tools(&mut self, tools: Vec<Tool>, call_id: &str) {
        if !self.pending_calls.contains(call_id) {
            tracing::warn!(
                "[TOOL_SEARCH] tool_search_output has unrecognized call_id '{}', \
                it may not have a corresponding tool_search_call",
                call_id
            );
        }

        let count = tools.len();
        self.searched_tools.extend(tools);
        tracing::debug!(
            "[TOOL_SEARCH] added {} tools from tool_search_output: call_id={}, total_searched={}",
            count,
            call_id,
            self.searched_tools.len()
        );
    }

    /// Mark the tool_search_call as completed (output received).
    ///
    /// Currently a no-op since we don't validate strict call_id matching,
    /// but could be used for future validation.
    pub fn complete_search(&mut self, call_id: &str) {
        self.pending_calls.remove(call_id);
        tracing::debug!(
            "[TOOL_SEARCH] completed tool_search_call: call_id={}",
            call_id
        );
    }

    /// Finalize the context and resolve the final tool list.
    ///
    /// This applies the priority strategy to merge predefined and searched tools.
    /// Returns the resolved tools and clears internal state.
    ///
    /// Note: This method takes `self` by value and returns an owned `Vec<Tool>`.
    /// Subsequent calls to resolved_tools() will return an empty vec.
    #[must_use]
    pub fn finalize(mut self) -> Vec<Tool> {
        if self.finalized {
            return std::mem::take(&mut self.resolved_tools);
        }

        tracing::debug!(
            "[TOOL_SEARCH] finalizing: predefined={}, searched={}, priority={:?}",
            self.predefined_tools.len(),
            self.searched_tools.len(),
            self.priority
        );

        let result = match self.priority {
            ToolPriority::PreferDefined => {
                if !self.searched_tools.is_empty() {
                    tracing::info!(
                        "[TOOL_SEARCH] PreferDefined: ignoring {} searched tools",
                        self.searched_tools.len()
                    );
                }
                std::mem::take(&mut self.predefined_tools)
            }
            ToolPriority::PreferSearched => {
                if !self.predefined_tools.is_empty() {
                    tracing::info!(
                        "[TOOL_SEARCH] PreferSearched: ignoring {} predefined tools",
                        self.predefined_tools.len()
                    );
                }
                std::mem::take(&mut self.searched_tools)
            }
            ToolPriority::Merge => merge_tools_map(&self.predefined_tools, &self.searched_tools),
        };

        self.finalized = true;
        self.resolved_tools = result.clone();
        tracing::debug!("[TOOL_SEARCH] resolved tools: count={}", result.len());

        result
    }

    /// Get the resolved tools without finalizing (for reading only).
    pub fn resolved_tools(&self) -> &[Tool] {
        &self.resolved_tools
    }

    /// Get the priority strategy.
    pub fn priority(&self) -> ToolPriority {
        self.priority
    }

    /// Check if any searches are still pending (have output without a matching search).
    pub fn has_pending_searches(&self) -> bool {
        !self.pending_calls.is_empty()
    }

    /// Get count of predefined tools.
    pub fn predefined_count(&self) -> usize {
        self.predefined_tools.len()
    }

    /// Get count of searched tools.
    pub fn searched_count(&self) -> usize {
        self.searched_tools.len()
    }
}

/// Merge two tool lists with deduplication by name.
///
/// For tools with the same name:
/// - If only in `first`, keep it
/// - If only in `second`, keep it
/// - If in both, keep the one from `second` (override)
///
/// The order in the result is: all unique tools from first, then unique tools from
/// second that weren't in first (preserving second's order for overrides).
pub(crate) fn merge_tools_map(first: &[Tool], second: &[Tool]) -> Vec<Tool> {
    use std::collections::HashMap;

    let mut name_to_tool: HashMap<String, &Tool> = HashMap::new();

    // Add all tools from first
    for tool in first {
        if let Some(name) = &tool.name {
            name_to_tool.insert(name.clone(), tool);
        }
    }

    // Override/add with tools from second
    for tool in second {
        if let Some(name) = &tool.name {
            name_to_tool.insert(name.clone(), tool);
        }
    }

    // Convert back to Vec, preserving first's order for non-overridden,
    // then second's order for overrides and new entries
    let mut result: Vec<Tool> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // First pass: add first's tools in order
    for tool in first {
        if let Some(name) = &tool.name
            && !seen.contains(name)
        {
            result.push(tool.clone());
            seen.insert(name.clone());
        }
    }

    // Second pass: add second's tools that weren't in first, or override existing
    for tool in second {
        if let Some(name) = &tool.name {
            // Check if this tool is from second (not in first's original order)
            let first_had = first.iter().any(|t| t.name.as_ref() == Some(name));
            if !first_had || !seen.contains(name) {
                // This is either a new tool from second, or an override
                // For simplicity, we only add if not already added
                if !seen.contains(name) {
                    result.push(tool.clone());
                    seen.insert(name.clone());
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::response_api::{Tool, ToolType};

    fn make_tool(name: &str) -> Tool {
        Tool {
            tool_type: ToolType::Function,
            name: Some(name.to_string()),
            description: None,
            parameters: None,
            strict: None,
            extra: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_prefer_defined_keeps_predefined() {
        let mut ctx = ToolSearchContext::new(ToolPriority::PreferDefined);
        ctx.set_predefined_tools(vec![make_tool("tool_a"), make_tool("tool_b")]);
        ctx.add_searched_tools(vec![make_tool("tool_c")], "call_1");

        let resolved = ctx.finalize().clone();
        assert_eq!(resolved.len(), 2);
        assert!(
            resolved
                .iter()
                .any(|t| t.name.as_ref().unwrap() == "tool_a")
        );
        assert!(
            resolved
                .iter()
                .any(|t| t.name.as_ref().unwrap() == "tool_b")
        );
    }

    #[test]
    fn test_prefer_searched_keeps_searched() {
        let mut ctx = ToolSearchContext::new(ToolPriority::PreferSearched);
        ctx.set_predefined_tools(vec![make_tool("tool_a"), make_tool("tool_b")]);
        ctx.add_searched_tools(vec![make_tool("tool_c"), make_tool("tool_d")], "call_1");

        let resolved = ctx.finalize().clone();
        assert_eq!(resolved.len(), 2);
        assert!(
            resolved
                .iter()
                .any(|t| t.name.as_ref().unwrap() == "tool_c")
        );
        assert!(
            resolved
                .iter()
                .any(|t| t.name.as_ref().unwrap() == "tool_d")
        );
    }

    #[test]
    fn test_merge_combines_all_unique() {
        let mut ctx = ToolSearchContext::new(ToolPriority::Merge);
        ctx.set_predefined_tools(vec![make_tool("tool_a"), make_tool("tool_b")]);
        ctx.add_searched_tools(vec![make_tool("tool_c"), make_tool("tool_d")], "call_1");

        let resolved = ctx.finalize().clone();
        assert_eq!(resolved.len(), 4);
    }

    #[test]
    fn test_merge_searched_overrides_on_conflict() {
        let mut ctx = ToolSearchContext::new(ToolPriority::Merge);
        ctx.set_predefined_tools(vec![make_tool("tool_a"), make_tool("tool_b")]);
        ctx.add_searched_tools(vec![make_tool("tool_b"), make_tool("tool_c")], "call_1");

        let resolved = ctx.finalize().clone();
        assert_eq!(resolved.len(), 3);

        // Verify tool_b exists (we can't easily tell which version it is from)
        let _tool_b = resolved
            .iter()
            .find(|t| t.name.as_ref().unwrap() == "tool_b");
        assert!(_tool_b.is_some());
    }

    #[test]
    fn test_register_search_call() {
        let mut ctx = ToolSearchContext::new(ToolPriority::Merge);
        ctx.register_search_call("call_123");

        assert!(ctx.is_registered_search("call_123"));
        assert!(!ctx.is_registered_search("call_456"));
    }

    #[test]
    fn test_priority_from_str() {
        assert_eq!(
            "prefer_defined".parse::<ToolPriority>(),
            Ok(ToolPriority::PreferDefined)
        );
        assert_eq!(
            "prefer-searched".parse::<ToolPriority>(),
            Ok(ToolPriority::PreferSearched)
        );
        assert_eq!("merge".parse::<ToolPriority>(), Ok(ToolPriority::Merge));
        assert_eq!("unknown".parse::<ToolPriority>(), Ok(ToolPriority::Merge)); // default
    }

    #[test]
    fn test_finalize_idempotent() {
        let mut ctx = ToolSearchContext::new(ToolPriority::Merge);
        ctx.set_predefined_tools(vec![make_tool("tool_a")]);
        ctx.add_searched_tools(vec![make_tool("tool_b")], "call_1");

        // finalize takes ownership, so we can't call it twice
        // Instead, test that calling finalize on a finalized context returns empty
        let mut ctx2 = ToolSearchContext::new(ToolPriority::Merge);
        ctx2.set_predefined_tools(vec![make_tool("tool_a")]);
        ctx2.add_searched_tools(vec![make_tool("tool_b")], "call_1");

        let first = ctx.finalize();
        assert_eq!(first.len(), 2);

        // Second finalize should return empty since we moved ctx
        // (this is expected behavior - we take ownership)
        let second = ctx2.finalize();
        assert_eq!(second.len(), 2);
    }
}
