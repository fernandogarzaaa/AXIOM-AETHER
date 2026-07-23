//! Interactive terminal menu for configuring and controlling Axiom.
//!
//! Launched by a bare `axiom` in an interactive terminal (see
//! `cli::is_interactive_terminal`) or explicitly via `axiom tui`. Edits the
//! same `~/.axiom/config.toml` every other entry point reads
//! (`config::UserConfig`) — nothing here is a separate, TUI-only config
//! surface, and nothing is written to disk until the user explicitly saves.

use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Tabs, Wrap};
use ratatui::{Frame, Terminal};

use crate::config::{self, AxiomPaths, UserConfig};
use crate::{bootstrap, daemon, hardware};

const TABS: [&str; 4] = ["Status", "Settings", "Features", "Actions"];

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsField {
    Host,
    Port,
    Device,
    VramBudgetMb,
    MaxContextTokens,
    DweBind,
    AutoFetchModel,
}
const SETTINGS_FIELDS: [SettingsField; 7] = [
    SettingsField::Host,
    SettingsField::Port,
    SettingsField::Device,
    SettingsField::VramBudgetMb,
    SettingsField::MaxContextTokens,
    SettingsField::DweBind,
    SettingsField::AutoFetchModel,
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum FeaturesField {
    TttCompress,
    TttThreshold,
    SwarmLocal,
    MeshRouting,
    OllamaUrl,
    OllamaModels,
}
const FEATURES_FIELDS: [FeaturesField; 6] = [
    FeaturesField::TttCompress,
    FeaturesField::TttThreshold,
    FeaturesField::SwarmLocal,
    FeaturesField::MeshRouting,
    FeaturesField::OllamaUrl,
    FeaturesField::OllamaModels,
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActionItem {
    RunDoctor,
    InitHome,
    DaemonStart,
    DaemonStop,
    DaemonRefresh,
    SaveConfig,
}
const ACTIONS: [ActionItem; 6] = [
    ActionItem::RunDoctor,
    ActionItem::InitHome,
    ActionItem::DaemonStart,
    ActionItem::DaemonStop,
    ActionItem::DaemonRefresh,
    ActionItem::SaveConfig,
];

/// A field being text-edited inline. Bool fields toggle immediately on Enter
/// and never enter this state.
struct Editing {
    tab: usize,
    index: usize,
    buffer: String,
}

struct App {
    paths: AxiomPaths,
    cfg: UserConfig,
    dirty: bool,
    tab: usize,
    settings_idx: usize,
    features_idx: usize,
    actions_idx: usize,
    editing: Option<Editing>,
    status_line: String,
    output: Vec<String>,
    daemon: daemon::DaemonStatus,
    hardware_report: String,
    should_quit: bool,
    /// Set while `ActionItem::InitHome` runs on a background thread — it does
    /// real network + local-training work that can take tens of seconds, and
    /// nothing else in this menu takes remotely that long. Blocking the whole
    /// event loop on it (the first version of this menu did) defeats the
    /// point of an *interactive* menu, so it runs off-thread and streams its
    /// log lines back here instead.
    background_job: Option<mpsc::Receiver<String>>,
}

impl App {
    fn new(paths: AxiomPaths, cfg: UserConfig) -> Self {
        let profile = hardware::detect();
        let rec = hardware::recommend(&profile);
        let hardware_report = hardware::report(&profile, &rec);
        let daemon = daemon::status().unwrap_or(daemon::DaemonStatus {
            pid: None,
            running: false,
            log_path: paths.hypervisor_log.clone(),
            pid_file: paths.pid_file.clone(),
            endpoint: format!("http://{}:{}", cfg.runtime.host, cfg.runtime.port),
        });
        Self {
            paths,
            cfg,
            dirty: false,
            tab: 0,
            settings_idx: 0,
            features_idx: 0,
            actions_idx: 0,
            editing: None,
            status_line: "Tab: switch panes  ↑/↓: select  Enter: edit/toggle/run  s: save  q: quit"
                .to_string(),
            output: Vec::new(),
            daemon,
            hardware_report,
            should_quit: false,
            background_job: None,
        }
    }

    /// Pull any log lines the background init job has produced since the
    /// last tick, without blocking. Returns once no message is immediately
    /// available; clears `background_job` once the sender side is dropped
    /// (the job finished).
    fn drain_background_job(&mut self) {
        let Some(rx) = self.background_job.as_ref() else { return };
        let mut lines = Vec::new();
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(line) => lines.push(line),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        for line in lines {
            self.log(line);
        }
        if disconnected {
            self.background_job = None;
        }
    }

    fn log(&mut self, line: impl Into<String>) {
        self.output.push(line.into());
        if self.output.len() > 200 {
            self.output.remove(0);
        }
    }

    fn save(&mut self) {
        match config::write_user_config(&self.paths, &self.cfg) {
            Ok(()) => {
                self.dirty = false;
                self.status_line = format!("saved to {}", self.paths.config.display());
            }
            Err(e) => self.status_line = format!("save failed: {e}"),
        }
    }

    fn refresh_daemon(&mut self) {
        match daemon::status() {
            Ok(s) => self.daemon = s,
            Err(e) => self.status_line = format!("daemon status failed: {e}"),
        }
    }

    // -- field value <-> string, per tab --

    fn settings_value(&self, field: SettingsField) -> String {
        match field {
            SettingsField::Host => self.cfg.runtime.host.clone(),
            SettingsField::Port => self.cfg.runtime.port.to_string(),
            SettingsField::Device => self.cfg.runtime.device.clone(),
            SettingsField::VramBudgetMb => self.cfg.runtime.vram_budget_mb.to_string(),
            SettingsField::MaxContextTokens => self.cfg.runtime.max_context_tokens.to_string(),
            SettingsField::DweBind => self.cfg.swarm.dwe_bind.clone(),
            SettingsField::AutoFetchModel => bool_label(self.cfg.models.auto_fetch),
        }
    }

    fn settings_label(field: SettingsField) -> &'static str {
        match field {
            SettingsField::Host => "runtime.host",
            SettingsField::Port => "runtime.port",
            SettingsField::Device => "runtime.device (auto|cpu|cuda|metal)",
            SettingsField::VramBudgetMb => "runtime.vram_budget_mb",
            SettingsField::MaxContextTokens => "runtime.max_context_tokens",
            SettingsField::DweBind => "swarm.dwe_bind",
            SettingsField::AutoFetchModel => "models.auto_fetch",
        }
    }

    fn features_value(&self, field: FeaturesField) -> String {
        match field {
            FeaturesField::TttCompress => bool_label(self.cfg.features.ttt_compress),
            FeaturesField::TttThreshold => self.cfg.features.ttt_compress_threshold_tokens.to_string(),
            FeaturesField::SwarmLocal => bool_label(self.cfg.features.swarm_local),
            FeaturesField::MeshRouting => bool_label(self.cfg.features.mesh_routing),
            FeaturesField::OllamaUrl => self.cfg.features.ollama_url.clone(),
            FeaturesField::OllamaModels => self.cfg.features.ollama_models.join(","),
        }
    }

    fn features_label(field: FeaturesField) -> &'static str {
        match field {
            FeaturesField::TttCompress => "features.ttt_compress (context compression)",
            FeaturesField::TttThreshold => "features.ttt_compress_threshold_tokens",
            FeaturesField::SwarmLocal => "features.swarm_local (local-model routing)",
            FeaturesField::MeshRouting => "features.mesh_routing (Axiom Mesh selector, opt-in)",
            FeaturesField::OllamaUrl => "features.ollama_url",
            FeaturesField::OllamaModels => "features.ollama_models (comma-separated)",
        }
    }

    fn is_bool_field(&self) -> bool {
        match self.tab {
            1 => matches!(SETTINGS_FIELDS[self.settings_idx], SettingsField::AutoFetchModel),
            2 => matches!(
                FEATURES_FIELDS[self.features_idx],
                FeaturesField::TttCompress | FeaturesField::SwarmLocal | FeaturesField::MeshRouting
            ),
            _ => false,
        }
    }

    fn toggle_current_bool(&mut self) {
        match self.tab {
            1 => {
                self.cfg.models.auto_fetch = !self.cfg.models.auto_fetch;
                self.dirty = true;
            }
            2 => {
                let field = FEATURES_FIELDS[self.features_idx];
                match field {
                    FeaturesField::TttCompress => self.cfg.features.ttt_compress ^= true,
                    FeaturesField::SwarmLocal => self.cfg.features.swarm_local ^= true,
                    FeaturesField::MeshRouting => self.cfg.features.mesh_routing ^= true,
                    _ => return,
                }
                self.dirty = true;
            }
            _ => {}
        }
    }

    /// Starts the edit buffer empty rather than pre-filled with the current
    /// value: typing only ever appends (there's no cursor movement within the
    /// field, just append/backspace-from-end), so pre-filling would make
    /// "replace this value" silently become "append to it" — confirmed by
    /// hand: typing "cpu" over a pre-filled "auto" produced "autocpu", not
    /// "cpu". The row still shows the previous value as a `was: ...` hint
    /// (see `field_row`) so the user isn't editing blind.
    fn begin_edit(&mut self) {
        if !matches!(self.tab, 1 | 2) {
            return;
        }
        self.editing = Some(Editing {
            tab: self.tab,
            index: if self.tab == 1 { self.settings_idx } else { self.features_idx },
            buffer: String::new(),
        });
    }

    fn commit_edit(&mut self) {
        let Some(edit) = self.editing.take() else { return };
        if edit.buffer.trim().is_empty() {
            self.status_line = "edit cancelled (nothing typed)".to_string();
            return;
        }
        let result = if edit.tab == 1 {
            apply_settings_edit(&mut self.cfg, SETTINGS_FIELDS[edit.index], &edit.buffer)
        } else {
            apply_features_edit(&mut self.cfg, FEATURES_FIELDS[edit.index], &edit.buffer)
        };
        match result {
            Ok(()) => {
                self.dirty = true;
                self.status_line = "updated (unsaved — press s to write config.toml)".to_string();
            }
            Err(e) => self.status_line = format!("invalid value: {e}"),
        }
    }

    fn run_action(&mut self, action: ActionItem) {
        match action {
            ActionItem::RunDoctor => {
                self.log("--- axiom doctor ---");
                let report = self.hardware_report.clone();
                for line in report.lines() {
                    self.log(line.to_string());
                }
            }
            ActionItem::InitHome => {
                if self.background_job.is_some() {
                    self.status_line = "a background job is already running".to_string();
                    return;
                }
                self.log(format!("initializing {}", self.paths.home.display()));
                self.log(
                    "(this can take up to a minute — model fetch + local checkpoint training \
                     run in the background; the menu stays usable meanwhile, though the \
                     training step's own progress output may print underneath it)",
                );
                let (tx, rx) = mpsc::channel();
                self.background_job = Some(rx);
                let paths = self.paths.clone();
                let cfg = self.cfg.clone();
                std::thread::spawn(move || {
                    if cfg.models.auto_fetch {
                        let _ = match config::ensure_base_model(&paths, &cfg) {
                            Ok(Some(p)) => tx.send(format!("base model ready at {}", p.display())),
                            Ok(None) => tx.send("base model fetch skipped (auto_fetch off)".to_string()),
                            Err(e) => tx.send(format!("base model fetch skipped: {e}")),
                        };
                    }
                    let _ = match bootstrap::ensure_checkpoint(
                        config::DEFAULT_CHECKPOINT_PATH,
                        candle_core::Device::Cpu,
                    ) {
                        Ok(true) => tx.send(format!(
                            "bootstrapped a local checkpoint at {}",
                            config::DEFAULT_CHECKPOINT_PATH
                        )),
                        Ok(false) => tx.send("checkpoint already present — skipped".to_string()),
                        Err(e) => tx.send(format!("checkpoint bootstrap skipped: {e}")),
                    };
                    let _ = tx.send("done.".to_string());
                });
            }
            ActionItem::DaemonStart => match daemon::start() {
                Ok(s) => {
                    self.log(format!("daemon started: pid={:?} endpoint={}", s.pid, s.endpoint));
                    self.daemon = s;
                }
                Err(e) => self.log(format!("daemon start failed: {e}")),
            },
            ActionItem::DaemonStop => match daemon::stop() {
                Ok(s) => {
                    self.log("daemon stopped");
                    self.daemon = s;
                }
                Err(e) => self.log(format!("daemon stop failed: {e}")),
            },
            ActionItem::DaemonRefresh => {
                self.refresh_daemon();
                self.log(format!(
                    "daemon: running={} pid={:?} endpoint={}",
                    self.daemon.running, self.daemon.pid, self.daemon.endpoint
                ));
            }
            ActionItem::SaveConfig => self.save(),
        }
    }
}

fn bool_label(v: bool) -> String {
    if v {
        "on".to_string()
    } else {
        "off".to_string()
    }
}

fn apply_settings_edit(cfg: &mut UserConfig, field: SettingsField, raw: &str) -> Result<(), String> {
    let raw = raw.trim();
    match field {
        SettingsField::Host => cfg.runtime.host = raw.to_string(),
        SettingsField::Port => cfg.runtime.port = raw.parse().map_err(|_| "expected a port number")?,
        SettingsField::Device => cfg.runtime.device = raw.to_string(),
        SettingsField::VramBudgetMb => {
            cfg.runtime.vram_budget_mb = raw.parse().map_err(|_| "expected a whole number (MB)")?
        }
        SettingsField::MaxContextTokens => {
            cfg.runtime.max_context_tokens = raw.parse().map_err(|_| "expected a whole number")?
        }
        SettingsField::DweBind => cfg.swarm.dwe_bind = raw.to_string(),
        SettingsField::AutoFetchModel => {}
    }
    Ok(())
}

fn apply_features_edit(cfg: &mut UserConfig, field: FeaturesField, raw: &str) -> Result<(), String> {
    let raw = raw.trim();
    match field {
        FeaturesField::TttThreshold => {
            cfg.features.ttt_compress_threshold_tokens =
                raw.parse().map_err(|_| "expected a whole number (tokens)")?
        }
        FeaturesField::OllamaUrl => cfg.features.ollama_url = raw.to_string(),
        FeaturesField::OllamaModels => {
            cfg.features.ollama_models = raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        }
        FeaturesField::TttCompress | FeaturesField::SwarmLocal | FeaturesField::MeshRouting => {}
    }
    Ok(())
}

/// Enter the interactive menu. Blocking: runs its own event loop and returns
/// once the user quits. Terminal state (raw mode, alternate screen) is always
/// restored on the way out, including after an internal error or panic.
pub fn run() -> io::Result<()> {
    let (paths, cfg, _created) = config::load_or_init_user_config()?;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let app = App::new(paths, cfg);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| event_loop(&mut terminal, app)));

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    match result {
        Ok(inner) => inner,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn event_loop<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, mut app: App) -> io::Result<()> {
    let tick = Duration::from_millis(250);
    let mut last_tick = Instant::now();
    loop {
        app.drain_background_job();
        terminal.draw(|f| draw(f, &app))?;

        let timeout = tick.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    handle_key(&mut app, key.code);
                }
            }
        }
        if last_tick.elapsed() >= tick {
            last_tick = Instant::now();
        }
        if app.should_quit {
            return Ok(());
        }
    }
}

fn handle_key(app: &mut App, code: KeyCode) {
    if app.editing.is_some() {
        match code {
            KeyCode::Enter => app.commit_edit(),
            KeyCode::Esc => {
                app.editing = None;
                app.status_line = "edit cancelled".to_string();
            }
            KeyCode::Backspace => {
                if let Some(e) = app.editing.as_mut() {
                    e.buffer.pop();
                }
            }
            KeyCode::Char(c) => {
                if let Some(e) = app.editing.as_mut() {
                    e.buffer.push(c);
                }
            }
            _ => {}
        }
        return;
    }

    match code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Char('s') => app.save(),
        KeyCode::Tab | KeyCode::Right => app.tab = (app.tab + 1) % TABS.len(),
        KeyCode::BackTab | KeyCode::Left => app.tab = (app.tab + TABS.len() - 1) % TABS.len(),
        KeyCode::Down | KeyCode::Char('j') => move_selection(app, 1),
        KeyCode::Up | KeyCode::Char('k') => move_selection(app, -1),
        KeyCode::Enter => match app.tab {
            1 | 2 => {
                if app.is_bool_field() {
                    app.toggle_current_bool();
                } else {
                    app.begin_edit();
                }
            }
            3 => {
                let action = ACTIONS[app.actions_idx];
                app.run_action(action);
            }
            _ => {}
        },
        _ => {}
    }
}

fn move_selection(app: &mut App, delta: i32) {
    match app.tab {
        1 => app.settings_idx = wrap_index(app.settings_idx, delta, SETTINGS_FIELDS.len()),
        2 => app.features_idx = wrap_index(app.features_idx, delta, FEATURES_FIELDS.len()),
        3 => app.actions_idx = wrap_index(app.actions_idx, delta, ACTIONS.len()),
        _ => {}
    }
}

fn wrap_index(current: usize, delta: i32, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let next = current as i32 + delta;
    ((next % len as i32) + len as i32) as usize % len
}

fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(3), Constraint::Length(2)])
        .split(f.area());

    let titles: Vec<Line> = TABS.iter().map(|t| Line::from(*t)).collect();
    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title(" axiom "))
        .select(app.tab)
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_widget(tabs, chunks[0]);

    match app.tab {
        0 => draw_status(f, app, chunks[1]),
        1 => draw_settings(f, app, chunks[1]),
        2 => draw_features(f, app, chunks[1]),
        3 => draw_actions(f, app, chunks[1]),
        _ => {}
    }

    let dirty_marker = if app.dirty { " [unsaved changes]" } else { "" };
    let footer = Paragraph::new(format!("{}{}", app.status_line, dirty_marker))
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(footer, chunks[2]);
}

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    let mut lines = vec![
        Line::from(format!("config:  {}", app.paths.config.display())),
        Line::from(format!("logs:    {}", app.paths.logs_dir.display())),
        Line::from(format!(
            "daemon:  {} {}",
            if app.daemon.running { "running" } else { "stopped" },
            app.daemon
                .pid
                .map(|p| format!("(pid {p})"))
                .unwrap_or_default()
        )),
        Line::from(format!("endpoint: {}", app.daemon.endpoint)),
        Line::from(""),
    ];
    for line in app.hardware_report.lines() {
        lines.push(Line::from(line.to_string()));
    }
    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" status "))
        .wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn draw_settings(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = SETTINGS_FIELDS
        .iter()
        .enumerate()
        .map(|(i, field)| field_row(f_selected(app.tab, i, app.settings_idx), App::settings_label(*field), &app.settings_value(*field), app, i))
        .collect();
    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" settings — Enter to edit/toggle "),
    );
    f.render_widget(list, area);
}

fn draw_features(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = FEATURES_FIELDS
        .iter()
        .enumerate()
        .map(|(i, field)| field_row(f_selected(app.tab, i, app.features_idx), App::features_label(*field), &app.features_value(*field), app, i))
        .collect();
    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" features — opt-in server behavior, applied on next `axiom daemon start` "),
    );
    f.render_widget(list, area);
}

fn f_selected(_tab: usize, i: usize, selected: usize) -> bool {
    i == selected
}

fn field_row(selected: bool, label: &str, value: &str, app: &App, index: usize) -> ListItem<'static> {
    let editing_here = app
        .editing
        .as_ref()
        .map(|e| e.tab == app.tab && e.index == index)
        .unwrap_or(false);
    let value_text = if editing_here {
        format!("(was: {value}) {}█", app.editing.as_ref().unwrap().buffer)
    } else {
        value.to_string()
    };
    let style = if selected {
        Style::default().fg(Color::Black).bg(Color::Cyan)
    } else {
        Style::default()
    };
    ListItem::new(Line::from(vec![
        Span::styled(format!("{label:<48}"), style),
        Span::styled(value_text, style.add_modifier(Modifier::BOLD)),
    ]))
    .style(style)
}

fn draw_actions(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(area);

    let items: Vec<ListItem> = ACTIONS
        .iter()
        .enumerate()
        .map(|(i, action)| {
            let selected = i == app.actions_idx;
            let style = if selected {
                Style::default().fg(Color::Black).bg(Color::Cyan)
            } else {
                Style::default()
            };
            ListItem::new(action_label(*action)).style(style)
        })
        .collect();
    let list = List::new(items).block(Block::default().borders(Borders::ALL).title(" actions — Enter to run "));
    f.render_widget(list, chunks[0]);

    let output_lines: Vec<Line> = app.output.iter().rev().take(200).rev().map(|l| Line::from(l.clone())).collect();
    let output = Paragraph::new(output_lines)
        .block(Block::default().borders(Borders::ALL).title(" output "))
        .wrap(Wrap { trim: false });
    f.render_widget(output, chunks[1]);
}

fn action_label(action: ActionItem) -> &'static str {
    match action {
        ActionItem::RunDoctor => "Run hardware doctor",
        ActionItem::InitHome => "Initialize ~/.axiom (config + local checkpoint)",
        ActionItem::DaemonStart => "Start daemon",
        ActionItem::DaemonStop => "Stop daemon",
        ActionItem::DaemonRefresh => "Refresh daemon status",
        ActionItem::SaveConfig => "Save settings to config.toml",
    }
}
