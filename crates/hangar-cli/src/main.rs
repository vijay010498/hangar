//! `hangar` - the agent CLI. A small tool-use loop over Hangar's chat API:
//! send the conversation + tool defs, run any tool calls locally (behind a
//! permission gate), feed results back, repeat until the model answers.
//! Non-streaming for a correct first cut; token-streaming is a later refinement.

mod tools;

use anyhow::{anyhow, bail, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, Write};

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

const MAX_STEPS: usize = 25;

const SYSTEM_PROMPT: &str = "You are Hangar, a local coding agent working in the user's project directory. \
Tools available: read_file, edit_file, bash, grep, glob. edit_file replaces an exact text block, or creates a file when old_string is empty. \
Always read a file before editing it so old_string matches exactly. Prefer small, targeted edits over rewriting whole files. \
Use bash to create directories, run generators, build, and test. Keep responses short; when the task is done, give a one or two sentence summary.";

#[derive(Parser)]
#[command(name = "hangar", about = "Hangar agent CLI - a local coding agent")]
struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:11435")]
    url: String,
    #[arg(long, default_value = "agent")]
    model: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl Message {
    fn system(s: &str) -> Self {
        Self::plain("system", s)
    }
    fn user(s: &str) -> Self {
        Self::plain("user", s)
    }
    fn plain(role: &str, content: &str) -> Self {
        Message {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
    fn tool(id: &str, content: String) -> Self {
        Message {
            role: "tool".into(),
            content: Some(content),
            tool_calls: None,
            tool_call_id: Some(id.into()),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct ToolCall {
    #[serde(default)]
    id: String,
    #[serde(rename = "type", default = "default_fn_type")]
    kind: String,
    function: FunctionCall,
}

fn default_fn_type() -> String {
    "function".into()
}

#[derive(Serialize, Deserialize, Clone)]
struct FunctionCall {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    tools: &'a [Value],
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = reqwest::Client::new();
    let endpoint = format!("{}/v1/chat/completions", cli.url.trim_end_matches('/'));
    let tool_defs = tools::tool_defs();
    let mut gate = tools::Gate::new(vec!["edit_file".into(), "bash".into()]);
    let mut messages = vec![Message::system(SYSTEM_PROMPT)];

    // Load the static project brief if present (the "project brain" - see §10.1).
    if let Ok(brief) = std::fs::read_to_string("HANGAR.md") {
        if !brief.trim().is_empty() {
            messages.push(Message::system(&format!(
                "Project brief (HANGAR.md):\n\n{brief}"
            )));
            println!("{DIM}loaded project brief: HANGAR.md{RESET}");
        }
    }

    println!(
        "{BOLD}hangar{RESET} agent - tier '{}', daemon {}",
        cli.model, cli.url
    );
    println!("{DIM}Type a request. Ctrl-C cancels a running turn; Ctrl-D or 'exit' quits.{RESET}");
    println!("{DIM}Edits and commands ask before they run (a = allow-all for this session).{RESET}\n");

    let stdin = std::io::stdin();
    loop {
        print!("{BOLD}hangar>{RESET} ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }
        messages.push(Message::user(line));

        tokio::select! {
            res = run_turn(&client, &endpoint, &cli.model, &tool_defs, &mut messages, &mut gate) => {
                if let Err(e) = res {
                    eprintln!("{RED}error: {e}{RESET}");
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\n{DIM}^C - cancelled this turn{RESET}");
            }
        }
    }
    Ok(())
}

async fn run_turn(
    client: &reqwest::Client,
    endpoint: &str,
    model: &str,
    tool_defs: &[Value],
    messages: &mut Vec<Message>,
    gate: &mut tools::Gate,
) -> Result<()> {
    for _ in 0..MAX_STEPS {
        let response = {
            let req = ChatRequest {
                model,
                messages: messages.as_slice(),
                tools: tool_defs,
                stream: false,
            };
            client.post(endpoint).json(&req).send().await?
        };

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!("daemon returned {status}: {body}");
        }

        let parsed: ChatResponse = response.json().await?;
        let mut msg = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no choices in response"))?
            .message;

        if let Some(calls) = msg.tool_calls.as_mut() {
            for (i, call) in calls.iter_mut().enumerate() {
                if call.id.trim().is_empty() {
                    call.id = format!("call_{i}");
                }
            }
        }

        if let Some(text) = msg.content.as_ref() {
            let text = text.trim();
            if !text.is_empty() {
                println!("{text}");
            }
        }

        let calls = msg.tool_calls.clone();
        messages.push(msg);

        match calls {
            Some(calls) if !calls.is_empty() => {
                for call in calls {
                    println!("{CYAN}> {}{RESET}", call.function.name);
                    let args: Value =
                        serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
                    let result = tools::execute(&call.function.name, &args, gate);
                    messages.push(Message::tool(&call.id, result));
                }
                continue;
            }
            _ => return Ok(()),
        }
    }
    println!("{DIM}(stopped after {MAX_STEPS} tool steps){RESET}");
    Ok(())
}
