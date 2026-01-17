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
use std::path::{Path, PathBuf};

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
    /// Model's context window in tokens (used to decide when to compact).
    #[arg(long, default_value_t = 32768)]
    context_window: usize,
    /// Compact the conversation once it passes this percent of the window.
    #[arg(long, default_value_t = 75)]
    compact_at: usize,
    /// Tier/model used to write compaction summaries (lighter than the agent).
    #[arg(long, default_value = "chat")]
    summary_model: String,
    /// Resume the most recent saved session.
    #[arg(long = "continue")]
    cont: bool,
    /// Resume a specific saved session by id.
    #[arg(long)]
    resume: Option<String>,
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
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize, Clone, Copy, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: usize,
}

const SESSIONS_DIR: &str = ".hangar/sessions";

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = reqwest::Client::new();
    let endpoint = format!("{}/v1/chat/completions", cli.url.trim_end_matches('/'));
    let tool_defs = tools::tool_defs();
    let mut gate = tools::Gate::new(vec!["edit_file".into(), "bash".into()]);
    let compact_limit = cli.context_window * cli.compact_at / 100;

    // Resume a saved session if asked, otherwise start fresh.
    let (mut messages, mut session_id) = match resume_target(&cli) {
        Some(id) => match load_session(&id) {
            Ok(msgs) => {
                println!("{DIM}resumed session {id} ({} messages){RESET}", msgs.len());
                (msgs, Some(id))
            }
            Err(e) => {
                eprintln!("{RED}could not resume: {e}{RESET}");
                (fresh_messages(), None)
            }
        },
        None => (fresh_messages(), None),
    };

    println!(
        "{BOLD}hangar{RESET} agent - tier '{}', daemon {}",
        cli.model, cli.url
    );
    println!("{DIM}Type a request, or /help for commands. Ctrl-C cancels a turn; Ctrl-D quits.{RESET}");
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
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }
        if let Some(cmd) = line.strip_prefix('/') {
            if handle_slash(cmd, &mut messages, &mut session_id) {
                break;
            }
            continue;
        }

        // The first real message names the session (title from the message text).
        if session_id.is_none() {
            let id = new_session_id(&line);
            println!("{DIM}[session {id} - resume later with: hangar --resume {id}]{RESET}");
            session_id = Some(id);
        }

        messages.push(Message::user(&line));

        tokio::select! {
            res = run_turn(&client, &endpoint, &cli.model, &tool_defs, &mut messages, &mut gate) => {
                match res {
                    Ok(prompt_tokens) => {
                        if compact_limit > 0 && prompt_tokens > compact_limit {
                            compact(&client, &endpoint, &cli.summary_model, &mut messages).await;
                        }
                    }
                    Err(e) => eprintln!("{RED}error: {e}{RESET}"),
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\n{DIM}^C - cancelled this turn{RESET}");
            }
        }
        if let Some(id) = &session_id {
            persist(id, &messages);
        }
    }

    if let Some(id) = &session_id {
        println!("{DIM}session saved - resume with: hangar --resume {id}{RESET}");
    }
    Ok(())
}

fn fresh_messages() -> Vec<Message> {
    let mut messages = vec![Message::system(SYSTEM_PROMPT)];
    // Load the static project brief if present (the "project brain").
    if let Ok(brief) = std::fs::read_to_string("HANGAR.md") {
        if !brief.trim().is_empty() {
            messages.push(Message::system(&format!(
                "Project brief (HANGAR.md):\n\n{brief}"
            )));
            println!("{DIM}loaded project brief: HANGAR.md{RESET}");
        }
    }
    messages
}

fn resume_target(cli: &Cli) -> Option<String> {
    if let Some(id) = &cli.resume {
        return Some(id.clone());
    }
    if cli.cont {
        return latest_session_id();
    }
    None
}

fn sessions_dir() -> &'static Path {
    Path::new(SESSIONS_DIR)
}

fn session_json(id: &str) -> PathBuf {
    sessions_dir().join(format!("{id}.json"))
}

fn session_md(id: &str) -> PathBuf {
    sessions_dir().join(format!("{id}.md"))
}

fn new_session_id(first_message: &str) -> String {
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    format!("{stamp}-{}", slug(first_message))
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for c in s.chars().take(60) {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "session".to_string()
    } else {
        trimmed
    }
}

fn persist(id: &str, messages: &[Message]) {
    let _ = std::fs::create_dir_all(sessions_dir());
    if let Ok(json) = serde_json::to_string_pretty(messages) {
        let _ = std::fs::write(session_json(id), json);
    }
    persist_session(&session_md(id), messages);
}

fn load_session(id: &str) -> Result<Vec<Message>> {
    let data = std::fs::read_to_string(session_json(id))
        .map_err(|e| anyhow!("no saved session '{id}' ({e})"))?;
    let messages: Vec<Message> = serde_json::from_str(&data)?;
    if messages.is_empty() {
        bail!("session '{id}' is empty");
    }
    Ok(messages)
}

fn latest_session_id() -> Option<String> {
    let mut newest: Option<(std::time::SystemTime, String)> = None;
    for entry in std::fs::read_dir(sessions_dir()).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let modified = entry.metadata().and_then(|m| m.modified()).ok();
        let stem = path.file_stem().and_then(|s| s.to_str()).map(String::from);
        if let (Some(m), Some(s)) = (modified, stem) {
            if newest.as_ref().map(|(t, _)| m > *t).unwrap_or(true) {
                newest = Some((m, s));
            }
        }
    }
    newest.map(|(_, id)| id)
}

fn list_sessions() -> Vec<String> {
    let mut ids = Vec::new();
    if let Ok(entries) = std::fs::read_dir(sessions_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    ids.push(stem.to_string());
                }
            }
        }
    }
    ids.sort();
    ids
}

fn print_sessions() {
    let ids = list_sessions();
    if ids.is_empty() {
        println!("{DIM}no saved sessions yet{RESET}");
        return;
    }
    println!("{DIM}saved sessions (resume with /resume <id>, or `hangar --resume <id>`):{RESET}");
    for id in ids {
        println!("  {id}");
    }
}

// Handle a /command typed at the prompt. Returns true if it asked to quit.
fn handle_slash(cmd: &str, messages: &mut Vec<Message>, session_id: &mut Option<String>) -> bool {
    let mut parts = cmd.split_whitespace();
    let name = parts.next().unwrap_or("");
    let arg = parts.next();
    match name {
        "" | "help" => {
            println!("{DIM}commands:{RESET}");
            println!("  /resume          list saved sessions");
            println!("  /resume <id>     load a saved session into this one");
            println!("  /sessions        list saved sessions");
            println!("  /help            show this");
            println!("  /quit            quit");
        }
        "sessions" => print_sessions(),
        "resume" => match arg {
            None => print_sessions(),
            Some(id) => match load_session(id) {
                Ok(loaded) => {
                    *messages = loaded;
                    *session_id = Some(id.to_string());
                    println!("{DIM}resumed session {id} ({} messages){RESET}", messages.len());
                }
                Err(e) => eprintln!("{RED}{e}{RESET}"),
            },
        },
        "quit" | "exit" => return true,
        other => eprintln!("{DIM}unknown command '/{other}' - try /help{RESET}"),
    }
    false
}

async fn run_turn(
    client: &reqwest::Client,
    endpoint: &str,
    model: &str,
    tool_defs: &[Value],
    messages: &mut Vec<Message>,
    gate: &mut tools::Gate,
) -> Result<usize> {
    let mut last_prompt = 0usize;
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
        if let Some(usage) = parsed.usage {
            last_prompt = usage.prompt_tokens;
        }
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
            _ => return Ok(last_prompt),
        }
    }
    println!("{DIM}(stopped after {MAX_STEPS} tool steps){RESET}");
    Ok(last_prompt)
}

const RECENT_TAIL: usize = 6;

// Summarize older turns into one message, keeping the system prefix and the
// most recent turns verbatim, so a long session stays within the context window.
async fn compact(
    client: &reqwest::Client,
    endpoint: &str,
    summary_model: &str,
    messages: &mut Vec<Message>,
) {
    let sys_end = messages.iter().take_while(|m| m.role == "system").count();
    if messages.len() < sys_end + RECENT_TAIL + 2 {
        return;
    }
    let mut tail_start = messages.len() - RECENT_TAIL;
    // Don't begin the kept tail on a tool result whose assistant call is above it.
    while tail_start > sys_end && messages[tail_start].role == "tool" {
        tail_start -= 1;
    }
    if tail_start <= sys_end {
        return;
    }

    let transcript = render_transcript(&messages[sys_end..tail_start]);
    let summary = match summarize(client, endpoint, summary_model, &transcript).await {
        Ok(s) if !s.is_empty() => s,
        _ => {
            eprintln!("{DIM}context compaction skipped (summary unavailable){RESET}");
            return;
        }
    };

    let removed = tail_start - sys_end;
    let mut rebuilt: Vec<Message> = messages[..sys_end].to_vec();
    rebuilt.push(Message::system(&format!(
        "Summary of earlier conversation:\n\n{summary}"
    )));
    rebuilt.extend_from_slice(&messages[tail_start..]);
    *messages = rebuilt;
    println!("{DIM}[context compacted: {removed} older messages -> summary]{RESET}");
}

async fn summarize(
    client: &reqwest::Client,
    endpoint: &str,
    model: &str,
    transcript: &str,
) -> Result<String> {
    let prompt = format!(
        "Summarize this coding-assistant conversation. Preserve: decisions made, files created or \
         changed, commands run and their results, and any open tasks or next steps. Be concise and \
         factual, with no preamble.\n\n---\n{transcript}\n---"
    );
    let msgs = vec![Message::user(&prompt)];
    let req = ChatRequest {
        model,
        messages: &msgs,
        tools: &[],
        stream: false,
    };
    let resp = client.post(endpoint).json(&req).send().await?;
    if !resp.status().is_success() {
        bail!("summary request failed: {}", resp.status());
    }
    let parsed: ChatResponse = resp.json().await?;
    let text = parsed
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .unwrap_or_default();
    Ok(text.trim().to_string())
}

fn render_transcript(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        match m.role.as_str() {
            "user" => {
                if let Some(c) = &m.content {
                    out.push_str(&format!("User: {c}\n"));
                }
            }
            "assistant" => {
                if let Some(c) = &m.content {
                    if !c.trim().is_empty() {
                        out.push_str(&format!("Assistant: {c}\n"));
                    }
                }
                if let Some(calls) = &m.tool_calls {
                    for call in calls {
                        out.push_str(&format!(
                            "Assistant called {}({})\n",
                            call.function.name,
                            truncate(&call.function.arguments, 200)
                        ));
                    }
                }
            }
            "tool" => {
                if let Some(c) = &m.content {
                    out.push_str(&format!("Tool result: {}\n", truncate(c, 400)));
                }
            }
            _ => {}
        }
    }
    out
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}...")
    }
}

// Persist the running session as readable Markdown, so it can be reopened or
// handed to a teammate (the compacted summary ends up at the top).
fn persist_session(path: &Path, messages: &[Message]) {
    let mut out = String::from("# Hangar session\n\n");
    for m in messages {
        match m.role.as_str() {
            "system" => {
                if let Some(c) = &m.content {
                    out.push_str(&format!("## system\n\n{c}\n\n"));
                }
            }
            "user" => {
                if let Some(c) = &m.content {
                    out.push_str(&format!("## user\n\n{c}\n\n"));
                }
            }
            "assistant" => {
                if let Some(c) = &m.content {
                    if !c.trim().is_empty() {
                        out.push_str(&format!("## assistant\n\n{c}\n\n"));
                    }
                }
                if let Some(calls) = &m.tool_calls {
                    for call in calls {
                        out.push_str(&format!(
                            "- called `{}` `{}`\n",
                            call.function.name, call.function.arguments
                        ));
                    }
                    out.push('\n');
                }
            }
            "tool" => {
                if let Some(c) = &m.content {
                    out.push_str(&format!("## tool result\n\n```\n{c}\n```\n\n"));
                }
            }
            _ => {}
        }
    }
    let _ = std::fs::write(path, out);
}
