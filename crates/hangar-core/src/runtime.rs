//! Ollama runtime client: the management surface the memory manager drives
//! (what's resident, load, unload). The chat/completions traffic itself is
//! proxied separately by the daemon; this is only for residency control.

use anyhow::{Context, Result};
use serde::Deserialize;

pub struct RuntimeClient {
    http: reqwest::Client,
    base: String,
}

#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub name: String,
    pub size_bytes: u64,
}

#[derive(Deserialize)]
struct PsResponse {
    #[serde(default)]
    models: Vec<PsModel>,
}

#[derive(Deserialize)]
struct PsModel {
    name: String,
    #[serde(default)]
    size: u64,
}

impl RuntimeClient {
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base: base.into().trim_end_matches('/').to_string(),
        }
    }

    /// Models currently resident in the runtime, with their footprints.
    pub async fn ps(&self) -> Result<Vec<LoadedModel>> {
        let parsed: PsResponse = self
            .http
            .get(format!("{}/api/ps", self.base))
            .send()
            .await
            .context("GET /api/ps")?
            .error_for_status()
            .context("/api/ps status")?
            .json()
            .await
            .context("decode /api/ps")?;
        Ok(parsed
            .models
            .into_iter()
            .map(|m| LoadedModel {
                name: m.name,
                size_bytes: m.size,
            })
            .collect())
    }

    pub async fn tags(&self) -> Result<Vec<LoadedModel>> {
        let parsed: PsResponse = self
            .http
            .get(format!("{}/api/tags", self.base))
            .send()
            .await
            .context("GET /api/tags")?
            .error_for_status()
            .context("/api/tags status")?
            .json()
            .await
            .context("decode /api/tags")?;
        Ok(parsed
            .models
            .into_iter()
            .map(|m| LoadedModel {
                name: m.name,
                size_bytes: m.size,
            })
            .collect())
    }

    pub async fn load(&self, model: &str) -> Result<()> {
        self.http
            .post(format!("{}/api/generate", self.base))
            .json(&serde_json::json!({
                "model": model,
                "prompt": "",
                "keep_alive": -1,
                "stream": false,
            }))
            .send()
            .await
            .with_context(|| format!("load {model}"))?
            .error_for_status()
            .with_context(|| format!("load {model} status"))?;
        Ok(())
    }

    /// Evict a model from the runtime immediately (keep_alive = 0).
    pub async fn unload(&self, model: &str) -> Result<()> {
        self.http
            .post(format!("{}/api/generate", self.base))
            .json(&serde_json::json!({
                "model": model,
                "keep_alive": 0,
                "stream": false,
            }))
            .send()
            .await
            .with_context(|| format!("unload {model}"))?
            .error_for_status()
            .with_context(|| format!("unload {model} status"))?;
        Ok(())
    }
}
