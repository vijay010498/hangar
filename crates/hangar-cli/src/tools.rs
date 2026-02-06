//! The agent's five tools and the permission gate. Tool execution is local -
//! the model only requests a call; nothing here runs on the model/runtime side.

use serde_json::{json, Value};
use similar::{ChangeTag, TextDiff};
use std::io::Write;
use std::path::Path;

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

/// The permission gate: confirms before the tools named in `confirm_tools`
/// (edit_file, bash). A session-wide "allow all" is remembered once granted.
pub struct Gate {
    allow_all: bool,
    confirm_tools: Vec<String>,
}

impl Gate {
    pub fn new(confirm_tools: Vec<String>) -> Self {
        Self {
            allow_all: false,
            confirm_tools,
        }
    }

    fn needs_confirm(&self, tool: &str) -> bool {
        self.confirm_tools.iter().any(|t| t == tool)
    }

    fn confirm(&mut self, tool: &str, preview: &str) -> bool {
        if self.allow_all || !self.needs_confirm(tool) {
            return true;
        }
        println!("\n{preview}");
        print!("{BOLD}Apply {tool}? [y]es / [n]o / [a]llow-all this session: {RESET}");
        let _ = std::io::stdout().flush();
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            return false;
        }
        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => true,
            "a" | "all" => {
                self.allow_all = true;
                true
            }
            _ => false,
        }
    }
}

/// OpenAI-format tool definitions sent to the model.
pub fn tool_defs() -> Vec<Value> {
    vec![
        func(
            "read_file",
            "Read a file's full contents.",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "File path to read" } },
                "required": ["path"]
            }),
        ),
        func(
            "edit_file",
            "Replace an exact text block in a file (old_string must match exactly once). \
             If the file does not exist, or old_string is empty, create the file with new_string as its contents.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_string": { "type": "string", "description": "Exact text to replace; empty to create a new file" },
                    "new_string": { "type": "string", "description": "Replacement text, or the new file's contents" }
                },
                "required": ["path", "new_string"]
            }),
        ),
        func(
            "bash",
            "Run a bash command (for creating directories, running generators, building, testing).",
            json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        ),
        func(
            "grep",
            "Search files recursively for a regex pattern.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string", "description": "Directory to search (default .)" }
                },
                "required": ["pattern"]
            }),
        ),
        func(
            "glob",
            "List files matching a glob pattern, e.g. src/**/*.rs.",
            json!({
                "type": "object",
                "properties": { "pattern": { "type": "string" } },
                "required": ["pattern"]
            }),
        ),
    ]
}

fn func(name: &str, description: &str, parameters: Value) -> Value {
    json!({ "type": "function", "function": { "name": name, "description": description, "parameters": parameters } })
}

/// Execute a tool call. Always returns a string to feed back to the model -
/// errors are returned as text so the model can see them and retry.
// A single tool result must never dominate the model's context window. Results
// larger than this are truncated with a note, so one grep or read cannot blow
// up the conversation (and trip the daemon's body limit).
const MAX_TOOL_OUTPUT: usize = 60000;

pub fn execute(name: &str, args: &Value, gate: &mut Gate) -> String {
    let out = match name {
        "read_file" => read_file(args),
        "edit_file" => edit_file(args, gate),
        "bash" => bash(args, gate),
        "grep" => grep(args),
        "glob" => glob_tool(args),
        other => format!("error: unknown tool '{other}'"),
    };
    clamp_output(out)
}

fn clamp_output(mut out: String) -> String {
    if out.len() <= MAX_TOOL_OUTPUT {
        return out;
    }
    let mut end = MAX_TOOL_OUTPUT;
    while !out.is_char_boundary(end) {
        end -= 1;
    }
    out.truncate(end);
    out.push_str("\n... (output truncated - narrow the query or read a specific range)");
    out
}

fn read_file(args: &Value) -> String {
    let Some(path) = args.get("path").and_then(Value::as_str) else {
        return "error: missing 'path'".into();
    };
    match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => format!("error reading {path}: {e}"),
    }
}

fn edit_file(args: &Value, gate: &mut Gate) -> String {
    let path = args.get("path").and_then(Value::as_str).unwrap_or("");
    let old = args.get("old_string").and_then(Value::as_str).unwrap_or("");
    let new = args.get("new_string").and_then(Value::as_str).unwrap_or("");
    if path.is_empty() {
        return "error: missing 'path'".into();
    }

    let exists = Path::new(path).exists();
    let (current, updated, action) = if !exists || old.is_empty() {
        (String::new(), new.to_string(), "create")
    } else {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => return format!("error reading {path}: {e}"),
        };
        match content.matches(old).count() {
            0 => {
                return format!(
                    "error: old_string not found in {path}; re-read the file and match it exactly"
                )
            }
            1 => (content.clone(), content.replacen(old, new, 1), "edit"),
            n => return format!("error: old_string matches {n} times in {path}; include more surrounding context to make it unique"),
        }
    };

    let preview = render_diff(path, &current, &updated, action);
    if !gate.confirm("edit_file", &preview) {
        return "user declined the edit".into();
    }
    if let Some(parent) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(path, &updated) {
        Ok(()) => format!("{action} applied to {path}"),
        Err(e) => format!("error writing {path}: {e}"),
    }
}

fn render_diff(path: &str, old: &str, new: &str, action: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut out = format!("{BOLD}{action}: {path}{RESET}\n");
    for change in diff.iter_all_changes() {
        let (sign, color) = match change.tag() {
            ChangeTag::Delete => ("-", RED),
            ChangeTag::Insert => ("+", GREEN),
            ChangeTag::Equal => (" ", DIM),
        };
        out.push_str(&format!("{color}{sign} {}{RESET}", change.value()));
        if !change.value().ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

fn bash(args: &Value, gate: &mut Gate) -> String {
    let Some(command) = args.get("command").and_then(Value::as_str) else {
        return "error: missing 'command'".into();
    };
    let preview = format!("{BOLD}bash{RESET}\n{DIM}$ {command}{RESET}");
    if !gate.confirm("bash", &preview) {
        return "user declined the command".into();
    }
    match std::process::Command::new("bash")
        .arg("-c")
        .arg(command)
        .output()
    {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let code = out.status.code().unwrap_or(-1);
            let mut result = stdout.into_owned();
            if !stderr.trim().is_empty() {
                result.push_str(&format!("\n[stderr]\n{stderr}"));
            }
            result.push_str(&format!("\n[exit {code}]"));
            result
        }
        Err(e) => format!("error running command: {e}"),
    }
}

fn grep(args: &Value) -> String {
    let Some(pattern) = args.get("pattern").and_then(Value::as_str) else {
        return "error: missing 'pattern'".into();
    };
    let root = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let re = match regex::Regex::new(pattern) {
        Ok(r) => r,
        Err(e) => return format!("error: invalid regex: {e}"),
    };

    let walker = walkdir::WalkDir::new(root).into_iter().filter_entry(|e| {
        e.file_name()
            .to_str()
            .map(|n| n != ".git" && n != "target" && n != "node_modules")
            .unwrap_or(true)
    });

    let mut out = String::new();
    let mut count = 0;
    for entry in walker.filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(entry.path()) {
            for (i, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    let shown = if line.len() > 300 {
                        let mut end = 300;
                        while !line.is_char_boundary(end) {
                            end -= 1;
                        }
                        format!("{}... (line truncated)", &line[..end])
                    } else {
                        line.to_string()
                    };
                    out.push_str(&format!("{}:{}: {}\n", entry.path().display(), i + 1, shown));
                    count += 1;
                    if count >= 200 {
                        out.push_str("... (truncated at 200 matches)\n");
                        return out;
                    }
                }
            }
        }
    }
    if out.is_empty() {
        format!("no matches for /{pattern}/ under {root}")
    } else {
        out
    }
}

fn glob_tool(args: &Value) -> String {
    let Some(pattern) = args.get("pattern").and_then(Value::as_str) else {
        return "error: missing 'pattern'".into();
    };
    match glob::glob(pattern) {
        Ok(paths) => {
            let mut out = String::new();
            for (n, entry) in paths.filter_map(Result::ok).enumerate() {
                if n >= 500 {
                    out.push_str("... (truncated)\n");
                    break;
                }
                out.push_str(&format!("{}\n", entry.display()));
            }
            if out.is_empty() {
                format!("no files match {pattern}")
            } else {
                out
            }
        }
        Err(e) => format!("error: invalid glob: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("hangar-cli-{}-{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // A gate that requires no confirmation, so edit_file applies without stdin.
    fn open_gate() -> Gate {
        Gate::new(vec![])
    }

    #[test]
    fn edit_file_creates_new_file() {
        let dir = tmpdir("create");
        let path = dir.join("new.txt");
        let out = edit_file(
            &json!({ "path": path.to_str().unwrap(), "new_string": "hello\n" }),
            &mut open_gate(),
        );
        assert!(out.contains("create applied"), "{out}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_file_replaces_exact_block() {
        let dir = tmpdir("edit");
        let path = dir.join("f.txt");
        std::fs::write(&path, "line one\nline two\n").unwrap();
        let out = edit_file(
            &json!({ "path": path.to_str().unwrap(), "old_string": "line two", "new_string": "line 2" }),
            &mut open_gate(),
        );
        assert!(out.contains("edit applied"), "{out}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "line one\nline 2\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_file_reports_no_match() {
        let dir = tmpdir("nomatch");
        let path = dir.join("f.txt");
        std::fs::write(&path, "abc\n").unwrap();
        let out = edit_file(
            &json!({ "path": path.to_str().unwrap(), "old_string": "xyz", "new_string": "q" }),
            &mut open_gate(),
        );
        assert!(out.contains("not found"), "{out}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn grep_finds_matches() {
        let dir = tmpdir("grep");
        std::fs::write(dir.join("a.txt"), "foo\nbar\nfoobar\n").unwrap();
        let out = grep(&json!({ "pattern": "foo", "path": dir.to_str().unwrap() }));
        assert!(out.contains("a.txt"), "{out}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn glob_lists_matching_files() {
        let dir = tmpdir("glob");
        std::fs::write(dir.join("x.rs"), "").unwrap();
        std::fs::write(dir.join("y.txt"), "").unwrap();
        let out = glob_tool(&json!({ "pattern": format!("{}/*.rs", dir.to_str().unwrap()) }));
        assert!(out.contains("x.rs"), "{out}");
        assert!(!out.contains("y.txt"), "{out}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}