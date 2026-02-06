//! Shapes exposed over the daemon's admin endpoint and consumed by the TUI.

use crate::memory::MemorySnapshot;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedEvent {
    pub seq: u64,
    pub kind: String,
    pub message: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminState {
    pub memory: MemorySnapshot,
    pub feed: Vec<FeedEvent>,
    #[serde(default)]
    pub tokens: TokenUsage,
    #[serde(default)]
    pub nodes: Vec<crate::cluster::NodeView>,
}