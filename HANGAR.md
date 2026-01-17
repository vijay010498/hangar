# Hangar

Standalone, offline-first local-AI coding platform. A Rust daemon (`hangard`) proxies
the chat-completions API to local model runtimes (Ollama) with task routing and a
memory manager; a ratatui TUI visualizes it; a `hangar` agent CLI is a terminal coding
agent. Full design in HANGAR-ARCHITECTURE.md.

## Layout (Cargo workspace)
- `crates/hangard` - daemon: chat API proxy, routing, memory manager, /admin endpoints
- `crates/hangar-cli` - the `hangar` agent CLI (5-tool loop, permission gate, compaction)
- `crates/hangar-tui` - ratatui dashboard (polls /admin/state)
- `crates/hangar-core` - shared: config, router, memory policy, runtime client, admin types
- `hangar.toml` - daemon config (routing tiers, memory ceiling, warm pool)

## Conventions
- Rust owns the control plane only; inference stays in the model runtime (Metal/CUDA).
- Comments: ASCII only, hyphen `-` not em/en dashes; comment the non-obvious "why",
  not every line; no "slice" references in code comments.
- Prefer small, targeted edits over rewriting whole files. Read a file before editing it.
- Keep the memory policy (`crates/hangar-core/src/memory.rs`) pure and unit-tested.

## Workflow
- Build: `cargo build`   Test: `cargo test`   Lint: `cargo clippy --workspace`
- Run daemon: `cargo run -p hangard` (needs Ollama running)
- Run TUI: `cargo run -p hangar-tui`
- Run agent: `cargo run -p hangar-cli`
- Always run `cargo test` before claiming something is done.

## House rules for the agent
- Do NOT run `git commit` or `git push`. Draft the commit message; the user runs it.
- Report outcomes honestly: if tests fail, say so with the output; never claim success unverified.