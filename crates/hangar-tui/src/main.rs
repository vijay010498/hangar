//! `hangar-tui` - a terminal dashboard for the daemon: RAM gauge against the
//! ceiling, resident models with footprint bars, and a live routing/eviction
//! feed. Polls the daemon's /admin/state endpoint once a second.

use anyhow::Result;
use clap::Parser;
use hangar_core::admin::AdminState;
use ratatui::{
    backend::{Backend, CrosstermBackend},
    crossterm::{
        event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    },
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Row, Table},
    Frame, Terminal,
};
use std::{
    io::stdout,
    sync::{Arc, Mutex},
    time::Duration,
};

const GB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Parser)]
#[command(name = "hangar-tui", about = "Hangar dashboard - live memory + routing view")]
struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:11435")]
    url: String,
}

#[derive(Default, Clone)]
struct UiState {
    admin: Option<AdminState>,
    out_per_sec: u64,
}

type Shared = Arc<Mutex<UiState>>;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let shared: Shared = Arc::new(Mutex::new(UiState::default()));

    // Background poller - keeps `shared` fresh without blocking the UI.
    {
        let shared = shared.clone();
        let url = format!("{}/admin/state", cli.url.trim_end_matches('/'));
        let client = reqwest::Client::new();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(1));
            let mut prev_out = 0u64;
            loop {
                ticker.tick().await;
                // On any failure drop to None so the UI shows "waiting for
                // hangard" rather than stale data; it reconnects next tick.
                let admin = match client.get(&url).send().await {
                    Ok(resp) => resp.json::<AdminState>().await.ok(),
                    Err(_) => None,
                };
                // Polls are 1s apart, so the output-token delta is ~tokens/sec.
                let out_per_sec = match &admin {
                    Some(a) => {
                        let now = a.tokens.output_tokens;
                        let rate = now.saturating_sub(prev_out);
                        prev_out = now;
                        rate
                    }
                    None => 0,
                };
                if let Ok(mut s) = shared.lock() {
                    *s = UiState { admin, out_per_sec };
                }
            }
        });
    }

    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(out))?;

    let result = run(&mut terminal, &shared, &cli.url);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn run<B: Backend>(terminal: &mut Terminal<B>, shared: &Shared, url: &str) -> Result<()> {
    loop {
        let ui = shared.lock().ok().map(|s| s.clone()).unwrap_or_default();
        terminal.draw(|f| draw(f, &ui, url))?;

        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    let quit = matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                        || (key.code == KeyCode::Char('c')
                            && key.modifiers.contains(KeyModifiers::CONTROL));
                    if quit {
                        return Ok(());
                    }
                }
            }
        }
    }
}

fn draw(f: &mut Frame, ui: &UiState, url: &str) {
    let state = ui.admin.as_ref();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(6),
            Constraint::Length(10),
        ])
        .split(f.area());

    draw_gauge(f, chunks[0], state, url);
    draw_stats(f, chunks[1], state, ui.out_per_sec);
    draw_models(f, chunks[2], state);
    draw_feed(f, chunks[3], state);
}

fn draw_stats(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    state: Option<&AdminState>,
    out_per_sec: u64,
) {
    let line = match state {
        Some(s) => format!(
            "  tokens   in {}   out {}   |   {} tok/s   |   {} requests",
            s.tokens.input_tokens, s.tokens.output_tokens, out_per_sec, s.tokens.requests
        ),
        None => "  tokens   -".to_string(),
    };
    f.render_widget(
        Paragraph::new(line).style(Style::default().fg(Color::Gray)),
        area,
    );
}

fn draw_gauge(f: &mut Frame, area: ratatui::layout::Rect, state: Option<&AdminState>, url: &str) {
    let (resident, ceiling) = state
        .map(|s| (s.memory.resident_bytes, s.memory.ceiling_bytes))
        .unwrap_or((0, 0));
    let unbounded = ceiling == 0 || ceiling == u64::MAX;
    let ratio = if unbounded {
        0.0
    } else {
        (resident as f64 / ceiling as f64).clamp(0.0, 1.0)
    };
    let label = match state {
        None => "waiting for hangard... (auto-reconnecting)".to_string(),
        Some(_) if unbounded => format!("{} resident (no ceiling)", gb(resident)),
        Some(_) => format!("{} / {}  ({:.0}%)", gb(resident), gb(ceiling), ratio * 100.0),
    };
    let color = if ratio > 0.9 {
        Color::Red
    } else if ratio > 0.7 {
        Color::Yellow
    } else {
        Color::Green
    };
    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Hangar - RAM ceiling  [{url}] ")),
        )
        .gauge_style(Style::default().fg(color))
        .ratio(ratio)
        .label(label);
    f.render_widget(gauge, area);
}

fn draw_models(f: &mut Frame, area: ratatui::layout::Rect, state: Option<&AdminState>) {
    let rows: Vec<Row> = state
        .map(|s| {
            s.memory
                .models
                .iter()
                .map(|m| {
                    let status = if m.resident { "●" } else { "○" };
                    let pin = if m.pinned { "pin" } else { "" };
                    let style = if m.resident {
                        Style::default().fg(Color::White)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };
                    Row::new(vec![
                        status.to_string(),
                        m.name.clone(),
                        gb(m.footprint_bytes),
                        pin.to_string(),
                        m.last_used.to_string(),
                    ])
                    .style(style)
                })
                .collect()
        })
        .unwrap_or_default();

    let table = Table::new(
        rows,
        [
            Constraint::Length(3),
            Constraint::Min(20),
            Constraint::Length(12),
            Constraint::Length(5),
            Constraint::Length(8),
        ],
    )
    .header(
        Row::new(vec!["", "model", "footprint", "pin", "lru"])
            .style(Style::default().add_modifier(Modifier::BOLD)),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Models  (● resident  ○ evicted) "),
    );
    f.render_widget(table, area);
}

fn draw_feed(f: &mut Frame, area: ratatui::layout::Rect, state: Option<&AdminState>) {
    let visible = area.height.saturating_sub(2) as usize;
    let items: Vec<ListItem> = state
        .map(|s| {
            s.feed
                .iter()
                .rev()
                .take(visible)
                .map(|e| {
                    let color = match e.kind.as_str() {
                        "evict" => Color::Red,
                        "load" => Color::Green,
                        "backpressure" => Color::Yellow,
                        "routed" => Color::Cyan,
                        "error" => Color::Magenta,
                        _ => Color::Gray,
                    };
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("{:>12} ", e.kind), Style::default().fg(color)),
                        Span::raw(e.message.clone()),
                    ]))
                })
                .collect()
        })
        .unwrap_or_default();

    let feed = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Routing / eviction feed   (q to quit) "),
    );
    f.render_widget(feed, area);
}

fn gb(bytes: u64) -> String {
    if bytes == u64::MAX {
        return "inf".to_string();
    }
    format!("{:.1} GB", bytes as f64 / GB)
}