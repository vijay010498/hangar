//! Shapes exposed over the daemon's admin endpoint and consumed by the TUI.

use crate::memory::MemorySnapshot;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedEvent {
    pub seq: u64,
    pub kind: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminState {
    pub memory: MemorySnapshot,
    pub feed: Vec<FeedEvent>,
}