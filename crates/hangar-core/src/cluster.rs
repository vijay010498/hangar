

use crate::memory::MemorySnapshot;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub serves: Vec<String>,
    pub memory: MemorySnapshot,
}

/// A node as seen by the coordinator / TUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeView {
    pub name: String,
    pub url: String,
    pub online: bool,
    #[serde(default)]
    pub serves: Vec<String>,
    pub memory: MemorySnapshot,
}

struct Entry {
    hb: Heartbeat,
    last_seen: u64,
}

#[derive(Default)]
pub struct ClusterRegistry {
    nodes: HashMap<String, Entry>,
}

impl ClusterRegistry {
    pub fn upsert(&mut self, hb: Heartbeat, now: u64) {
        self.nodes.insert(hb.name.clone(), Entry { hb, last_seen: now });
    }

    pub fn prune(&mut self, now: u64, drop_after: u64) {
        self.nodes
            .retain(|_, e| now.saturating_sub(e.last_seen) <= drop_after);
    }

    fn is_online(entry: &Entry, now: u64, timeout: u64) -> bool {
        now.saturating_sub(entry.last_seen) <= timeout
    }

    pub fn views(&self, now: u64, timeout: u64) -> Vec<NodeView> {
        let mut views: Vec<NodeView> = self
            .nodes
            .values()
            .map(|e| NodeView {
                name: e.hb.name.clone(),
                url: e.hb.url.clone(),
                online: Self::is_online(e, now, timeout),
                serves: e.hb.serves.clone(),
                memory: e.hb.memory.clone(),
            })
            .collect();
        views.sort_by(|a, b| a.name.cmp(&b.name));
        views
    }

    pub fn select_host(&self, model: &str, now: u64, timeout: u64) -> Option<String> {
        let capable: Vec<&Entry> = self
            .nodes
            .values()
            .filter(|e| Self::is_online(e, now, timeout))
            .filter(|e| {
                model.is_empty() || e.hb.serves.is_empty() || e.hb.serves.iter().any(|m| m == model)
            })
            .collect();
        if capable.is_empty() {
            return None;
        }
        if !model.is_empty() {
            if let Some(e) = capable.iter().find(|e| {
                e.hb.memory
                    .models
                    .iter()
                    .any(|m| m.name == model && m.resident)
            }) {
                return Some(e.hb.url.clone());
            }
        }
        capable
            .iter()
            .min_by_key(|e| e.hb.memory.resident_bytes)
            .map(|e| e.hb.url.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemorySnapshot, ModelView};

    fn snap(resident_bytes: u64, models: Vec<(&str, bool)>) -> MemorySnapshot {
        MemorySnapshot {
            ceiling_bytes: 40,
            resident_bytes,
            models: models
                .into_iter()
                .map(|(name, resident)| ModelView {
                    name: name.to_string(),
                    footprint_bytes: 0,
                    pinned: false,
                    resident,
                    last_used: 0,
                })
                .collect(),
        }
    }

    fn hb(name: &str, serves: &[&str], resident_bytes: u64, models: Vec<(&str, bool)>) -> Heartbeat {
        Heartbeat {
            name: name.to_string(),
            url: format!("http://{name}:11435"),
            serves: serves.iter().map(|s| s.to_string()).collect(),
            memory: snap(resident_bytes, models),
        }
    }

    #[test]
    fn empty_registry_selects_nothing() {
        let r = ClusterRegistry::default();
        assert_eq!(r.select_host("x", 100, 20), None);
    }

    #[test]
    fn prefers_host_with_model_resident() {
        let mut r = ClusterRegistry::default();
        r.upsert(hb("a", &["m"], 5, vec![("m", false)]), 100);
        r.upsert(hb("b", &["m"], 30, vec![("m", true)]), 100);
        assert_eq!(r.select_host("m", 100, 20).as_deref(), Some("http://b:11435"));
    }

    #[test]
    fn falls_back_to_least_loaded() {
        let mut r = ClusterRegistry::default();
        r.upsert(hb("a", &["m"], 20, vec![]), 100);
        r.upsert(hb("b", &["m"], 5, vec![]), 100);
        assert_eq!(r.select_host("m", 100, 20).as_deref(), Some("http://b:11435"));
    }

    #[test]
    fn only_capable_hosts_are_chosen() {
        let mut r = ClusterRegistry::default();
        r.upsert(hb("small", &["tiny"], 2, vec![]), 100);
        r.upsert(hb("big", &["tiny", "huge"], 20, vec![]), 100);
        assert_eq!(
            r.select_host("huge", 100, 20).as_deref(),
            Some("http://big:11435")
        );
        assert_eq!(r.select_host("missing", 100, 20), None);
    }

    #[test]
    fn stale_nodes_are_not_selected_and_prune() {
        let mut r = ClusterRegistry::default();
        r.upsert(hb("old", &["m"], 5, vec![]), 10);
        r.upsert(hb("new", &["m"], 5, vec![]), 100);
        assert_eq!(r.select_host("m", 100, 20).as_deref(), Some("http://new:11435"));
        assert_eq!(r.views(100, 20).iter().filter(|v| v.online).count(), 1);
        r.prune(100, 50);
        assert_eq!(r.views(100, 20).len(), 1);
    }
}