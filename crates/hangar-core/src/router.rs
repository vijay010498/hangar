//! Task routing: pick a model for each request from config-driven heuristics.

use crate::config::HangarConfig;

/// Chat requests whose rough token estimate exceeds this escalate to the agent tier.
pub const AGENT_TOKEN_THRESHOLD: usize = 8000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    Completions,
    ChatCompletions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskClass {
    Autocomplete,
    Chat,
    Agent,
}

impl TaskClass {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskClass::Autocomplete => "autocomplete",
            TaskClass::Chat => "chat",
            TaskClass::Agent => "agent",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub model: String,
    pub tier: String,
    pub reason: String,
}

pub fn classify(surface: Surface, has_tools: bool, approx_tokens: usize) -> (TaskClass, &'static str) {
    match surface {
        Surface::Completions => (
            TaskClass::Autocomplete,
            "completions endpoint (fill-in-the-middle)",
        ),
        Surface::ChatCompletions => {
            if has_tools {
                (TaskClass::Agent, "request carries tools")
            } else if approx_tokens > AGENT_TOKEN_THRESHOLD {
                (TaskClass::Agent, "token count over agent threshold")
            } else {
                (TaskClass::Chat, "chat endpoint, no tools")
            }
        }
    }
}

/// Selection order: explicit override -> endpoint/surface -> tools -> token count.
/// A requested model that names a configured tier selects that tier; any other
/// non-empty model is an explicit override and passes through unchanged.
pub fn route(
    cfg: &HangarConfig,
    surface: Surface,
    requested_model: Option<&str>,
    has_tools: bool,
    approx_tokens: usize,
) -> Decision {
    if let Some(raw) = requested_model {
        let m = raw.trim();
        if !m.is_empty() {
            if let Some(profile) = cfg.routing.get(m) {
                return Decision {
                    model: profile.model.clone(),
                    tier: m.to_string(),
                    reason: format!("explicit tier '{m}'"),
                };
            }
            return Decision {
                model: m.to_string(),
                tier: "override".to_string(),
                reason: format!("explicit model override '{m}'"),
            };
        }
    }

    let (class, why) = classify(surface, has_tools, approx_tokens);
    match cfg.routing.get(class.as_str()) {
        Some(profile) => Decision {
            model: profile.model.clone(),
            tier: class.as_str().to_string(),
            reason: why.to_string(),
        },
        None => Decision {
            model: requested_model.unwrap_or_default().to_string(),
            tier: class.as_str().to_string(),
            reason: format!(
                "no routing profile for tier '{}'; passing requested model through",
                class.as_str()
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HangarConfig;

    fn cfg() -> HangarConfig {
        toml::from_str(
            r#"
            [routing.autocomplete]
            model = "tiny"
            [routing.chat]
            model = "mid"
            [routing.agent]
            model = "big"
        "#,
        )
        .unwrap()
    }

    #[test]
    fn completions_routes_to_autocomplete() {
        let d = route(&cfg(), Surface::Completions, None, false, 0);
        assert_eq!(d.model, "tiny");
        assert_eq!(d.tier, "autocomplete");
    }

    #[test]
    fn chat_without_tools_routes_to_chat() {
        let d = route(&cfg(), Surface::ChatCompletions, None, false, 10);
        assert_eq!(d.model, "mid");
        assert_eq!(d.tier, "chat");
    }

    #[test]
    fn tools_route_to_agent() {
        let d = route(&cfg(), Surface::ChatCompletions, None, true, 10);
        assert_eq!(d.model, "big");
        assert_eq!(d.tier, "agent");
    }

    #[test]
    fn large_context_escalates_to_agent() {
        let d = route(&cfg(), Surface::ChatCompletions, None, false, AGENT_TOKEN_THRESHOLD + 1);
        assert_eq!(d.model, "big");
        assert_eq!(d.tier, "agent");
    }

    #[test]
    fn explicit_tier_alias_wins() {
        let d = route(&cfg(), Surface::ChatCompletions, Some("agent"), false, 0);
        assert_eq!(d.model, "big");
        assert_eq!(d.tier, "agent");
    }

    #[test]
    fn explicit_concrete_model_is_override() {
        let d = route(&cfg(), Surface::Completions, Some("llama3:70b"), false, 0);
        assert_eq!(d.model, "llama3:70b");
        assert_eq!(d.tier, "override");
    }
}