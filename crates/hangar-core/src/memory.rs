//! Memory policy: keep resident models within a hard ceiling by evicting the
//! least-recently-used non-pinned model, never oversubscribing. This module is
//! pure (no I/O); the daemon pairs it with the
//! runtime client to execute the load/unload decisions it returns.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Estimate used for a model whose real footprint hasn't been observed yet.
/// Reconciled to the true size from the runtime once it has loaded.
pub const DEFAULT_MODEL_FOOTPRINT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
struct Entry {
    footprint: u64,
    last_used: u64,
    pinned: bool,
    resident: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    Serve,
    /// Load it, first unloading these victims (in order) to make room.
    Load { victims: Vec<String> },
    CantFit,
}

pub struct MemoryManager {
    ceiling_bytes: u64,
    tick: u64,
    models: HashMap<String, Entry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelView {
    pub name: String,
    pub footprint_bytes: u64,
    pub pinned: bool,
    pub resident: bool,
    pub last_used: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub ceiling_bytes: u64,
    pub resident_bytes: u64,
    pub models: Vec<ModelView>,
}

impl MemoryManager {
    pub fn new(ceiling_bytes: u64, pinned: &[String]) -> Self {
        let models = pinned
            .iter()
            .map(|name| {
                (
                    name.clone(),
                    Entry {
                        footprint: 0,
                        last_used: 0,
                        pinned: true,
                        resident: false,
                    },
                )
            })
            .collect();
        Self {
            ceiling_bytes,
            tick: 0,
            models,
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    pub fn is_resident(&self, model: &str) -> bool {
        self.models.get(model).map(|e| e.resident).unwrap_or(false)
    }

    pub fn resident_bytes(&self) -> u64 {
        self.models
            .values()
            .filter(|e| e.resident)
            .map(|e| e.footprint)
            .sum()
    }

    pub fn learned_footprint(&self, model: &str) -> Option<u64> {
        self.models
            .get(model)
            .filter(|e| e.footprint > 0)
            .map(|e| e.footprint)
    }

    pub fn touch(&mut self, model: &str) {
        let t = self.next_tick();
        if let Some(e) = self.models.get_mut(model) {
            e.last_used = t;
        }
    }

    pub fn mark_unloaded(&mut self, model: &str) {
        if let Some(e) = self.models.get_mut(model) {
            e.resident = false;
        }
    }

    pub fn mark_loaded(&mut self, model: &str, footprint: u64) {
        let t = self.next_tick();
        let pinned = self.models.get(model).map(|e| e.pinned).unwrap_or(false);
        self.models.insert(
            model.to_string(),
            Entry {
                footprint,
                last_used: t,
                pinned,
                resident: true,
            },
        );
    }

    /// Sync state to the runtime's ground truth: models not in `resident` become
    /// non-resident; listed ones get their observed footprint. Pinned flags and
    /// LRU order are preserved.
    pub fn reconcile(&mut self, resident: &[(String, u64)]) {
        let names: HashSet<&str> = resident.iter().map(|(n, _)| n.as_str()).collect();
        for (name, e) in self.models.iter_mut() {
            if !names.contains(name.as_str()) {
                e.resident = false;
            }
        }
        for (name, size) in resident {
            match self.models.get_mut(name) {
                Some(e) => {
                    e.footprint = *size;
                    e.resident = true;
                }
                None => {
                    let t = self.next_tick();
                    self.models.insert(
                        name.clone(),
                        Entry {
                            footprint: *size,
                            last_used: t,
                            pinned: false,
                            resident: true,
                        },
                    );
                }
            }
        }
    }

    /// Decide how to admit `model`, using its learned footprint if known or
    /// `estimate` otherwise. Victims are chosen least-recently-used first and
    /// never include pinned models.
    pub fn plan_admission(&self, model: &str, estimate: u64) -> Admission {
        if self.is_resident(model) {
            return Admission::Serve;
        }
        let need = self.learned_footprint(model).unwrap_or(estimate);
        let mut free = self.ceiling_bytes.saturating_sub(self.resident_bytes());
        if need <= free {
            return Admission::Load {
                victims: Vec::new(),
            };
        }

        let mut candidates: Vec<(&String, &Entry)> = self
            .models
            .iter()
            .filter(|(_, e)| e.resident && !e.pinned)
            .collect();
        candidates.sort_by_key(|(_, e)| e.last_used);

        let mut victims = Vec::new();
        for (name, e) in candidates {
            if need <= free {
                break;
            }
            victims.push((*name).clone());
            free += e.footprint;
        }

        if need <= free {
            Admission::Load { victims }
        } else {
            Admission::CantFit
        }
    }

    /// Victims to evict (LRU non-pinned, excluding `protect`) to bring resident
    /// bytes back within the ceiling - used after a load whose real footprint
    /// turned out larger than the pre-load estimate.
    pub fn overflow_victims(&self, protect: &str) -> Vec<String> {
        let mut over = self.resident_bytes().saturating_sub(self.ceiling_bytes);
        if over == 0 {
            return Vec::new();
        }
        let mut candidates: Vec<(&String, &Entry)> = self
            .models
            .iter()
            .filter(|(n, e)| e.resident && !e.pinned && n.as_str() != protect)
            .collect();
        candidates.sort_by_key(|(_, e)| e.last_used);

        let mut victims = Vec::new();
        for (name, e) in candidates {
            if over == 0 {
                break;
            }
            victims.push((*name).clone());
            over = over.saturating_sub(e.footprint);
        }
        victims
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        let mut models: Vec<ModelView> = self
            .models
            .iter()
            .map(|(name, e)| ModelView {
                name: name.clone(),
                footprint_bytes: e.footprint,
                pinned: e.pinned,
                resident: e.resident,
                last_used: e.last_used,
            })
            .collect();
        models.sort_by(|a, b| a.name.cmp(&b.name));
        MemorySnapshot {
            ceiling_bytes: self.ceiling_bytes,
            resident_bytes: self.resident_bytes(),
            models,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn resident_model_just_serves() {
        let mut m = MemoryManager::new(40 * GB, &[]);
        m.mark_loaded("a", 5 * GB);
        assert_eq!(m.plan_admission("a", 5 * GB), Admission::Serve);
    }

    #[test]
    fn fits_without_eviction() {
        let mut m = MemoryManager::new(40 * GB, &[]);
        m.mark_loaded("a", 5 * GB);
        assert_eq!(
            m.plan_admission("b", 5 * GB),
            Admission::Load { victims: vec![] }
        );
    }

    #[test]
    fn evicts_lru_non_pinned() {
        let mut m = MemoryManager::new(10 * GB, &[]);
        m.mark_loaded("old", 6 * GB);
        m.mark_loaded("new", 3 * GB);
        assert_eq!(
            m.plan_admission("big", 5 * GB),
            Admission::Load {
                victims: vec!["old".to_string()]
            }
        );
    }

    #[test]
    fn touch_updates_lru_order() {
        let mut m = MemoryManager::new(10 * GB, &[]);
        m.mark_loaded("a", 6 * GB);
        m.mark_loaded("b", 3 * GB);
        // Without the touch, "a" would be the LRU victim; touching it makes "b"
        // the oldest, so a 3 GB admission evicts "b" instead.
        m.touch("a");
        assert_eq!(
            m.plan_admission("c", 3 * GB),
            Admission::Load {
                victims: vec!["b".to_string()]
            }
        );
    }

    #[test]
    fn pinned_is_never_a_victim() {
        let mut m = MemoryManager::new(10 * GB, &["pin".to_string()]);
        m.mark_loaded("pin", 8 * GB);
        assert_eq!(m.plan_admission("x", 5 * GB), Admission::CantFit);
    }

    #[test]
    fn model_bigger_than_ceiling_cant_fit() {
        let m = MemoryManager::new(10 * GB, &[]);
        assert_eq!(m.plan_admission("huge", 20 * GB), Admission::CantFit);
    }

    #[test]
    fn overflow_victims_trim_to_ceiling() {
        let mut m = MemoryManager::new(24 * GB, &[]);
        m.mark_loaded("chat", 10 * GB);
        m.mark_loaded("agent", 22 * GB); // 32 GB resident, 8 GB over
        assert_eq!(m.overflow_victims("agent"), vec!["chat".to_string()]);
    }

    #[test]
    fn reconcile_marks_absent_models_evicted() {
        let mut m = MemoryManager::new(40 * GB, &[]);
        m.mark_loaded("a", 5 * GB);
        m.mark_loaded("b", 5 * GB);
        m.reconcile(&[("a".to_string(), 6 * GB)]);
        assert!(m.is_resident("a"));
        assert!(!m.is_resident("b"));
        assert_eq!(m.resident_bytes(), 6 * GB);
    }
}
