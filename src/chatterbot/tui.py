"""Streamer-only terminal UI.

Five tabs (the Moderation tab is only included when MOD_MODE_ENABLED):

  - Chatters: searchable list; right pane shows aliases, notes (with source
    previews), recent SE events, and recent messages (with reply context).
  - Live Topics: rolling history of channel-wide topic snapshots.
  - Events: chronological StreamElements feed.
  - Live: rolling tail of the latest channel chat (auto-refresh every 2s).
  - Moderation (when on): incidents with mark-reviewed / dismiss / reopen
    actions. Same advisory-only contract as the dashboard.

The TUI shares the SQLite DB with the bot and the dashboard via WAL mode.

Things deliberately not in the TUI (use the dashboard instead):
  - "Ask Qwen about this user" RAG (long streaming prose UX is rough in a
    terminal).
  - Per-note edit (the rendering is Static — there's no per-note picker).
  - Open-issue-on-GitHub (terminals can't reliably open URLs).
  - Settings credential editing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from .config import Settings
from .diagnose import build_diagnostic_bundle, default_bundle_filename
from .llm.ollama_client import OllamaClient
from .repo import (
    ChatterRepo,
    ChatterRow,
    Event,
    IncidentRow,
    Message,
    Note,
    NoteWithSources,
    User,
    UserEventSummary,
)


# ---------------- modal screens ----------------


class AddNoteScreen(ModalScreen[str | None]):
    """Modal for the 'n' binding — manually add a profile note."""

    DEFAULT_CSS = """
    AddNoteScreen { align: center middle; }
    #dialog { width: 60; height: auto; padding: 1 2; background: $surface;
              border: round $primary; }
    #dialog Label { content-align: left middle; }
    #dialog Input { margin: 1 0; }
    #buttons { height: auto; align: right middle; }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, viewer_name: str):
        super().__init__()
        self.viewer_name = viewer_name

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(f"[b]+ add note[/b] for {self.viewer_name}")
            yield Input(
                placeholder="e.g. Owns a black cat named Loki",
                id="note-input",
                max_length=500,
            )
            with Horizontal(id="buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Save", id="save-btn", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#note-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            text = self.query_one("#note-input", Input).value.strip()
            self.dismiss(text or None)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------- main app ----------------


class ChatterbotTUI(App):
    CSS = """
    #chatters_layout { height: 1fr; }
    #left_pane { width: 38%; min-width: 32; border-right: solid $primary; }
    #right_pane { width: 1fr; padding: 0 1; }
    #search { dock: top; }
    #user_table { height: 1fr; }
    #notes_box  { height: auto; min-height: 6; border: round $primary; padding: 0 1; }
    #events_box { height: auto; min-height: 4; border: round $accent; padding: 0 1; margin-top: 1; }
    #msgs_box   { height: 1fr; border: round $secondary; padding: 0 1; margin-top: 1; }
    #topics_list { padding: 1; }
    #events_list { padding: 1; }
    #live_feed   { padding: 1; }
    #incidents_table { height: 60%; }
    #incident_detail { height: 1fr; border: round $warning; padding: 0 1; margin-top: 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("slash", "focus_search", "Search"),
        Binding("f", "forget_user", "Forget"),
        Binding("o", "toggle_opt_out", "Opt-out"),
        Binding("d", "delete_note", "Del note"),
        Binding("n", "add_note", "+ note"),
        Binding("D", "diagnose", "Bundle"),
        Binding("m", "incident_status('reviewed')", "Reviewed", show=False),
        Binding("x", "incident_status('dismissed')", "Dismiss", show=False),
        Binding("R", "incident_status('open')", "Reopen", show=False),
        Binding("1", "switch_tab('chatters')", "Chatters"),
        Binding("2", "switch_tab('topics')", "Topics"),
        Binding("3", "switch_tab('events')", "Events"),
        Binding("4", "switch_tab('live')", "Live"),
        Binding("5", "switch_tab('moderation')", "Mod", show=False),
    ]

    selected_user_id: reactive[str | None] = reactive(None)
    selected_incident_id: reactive[int | None] = reactive(None)

    def __init__(self, repo: ChatterRepo, settings: Settings, llm: OllamaClient | None = None):
        super().__init__()
        self.repo = repo
        self.settings = settings
        self.llm = llm
        self._search_query = ""

    # ---------------- compose ----------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="chatters", id="tabs"):
            with TabPane("Chatters", id="chatters"):
                yield Input(placeholder="search by name…", id="search")
                with Horizontal(id="chatters_layout"):
                    with Vertical(id="left_pane"):
                        yield DataTable(
                            id="user_table", cursor_type="row", zebra_stripes=True
                        )
                    with Vertical(id="right_pane"):
                        yield Static("Select a chatter.", id="notes_box")
                        yield Static("", id="events_box")
                        yield Static("", id="msgs_box")
            with TabPane("Live Topics", id="topics"):
                yield Static("Loading topic snapshots…", id="topics_list")
            with TabPane("Events", id="events"):
                yield Static("Loading events…", id="events_list")
            with TabPane("Live", id="live"):
                yield Static("Connecting…", id="live_feed")
            if self.settings.mod_mode_enabled:
                with TabPane("Moderation", id="moderation"):
                    yield DataTable(
                        id="incidents_table", cursor_type="row", zebra_stripes=True
                    )
                    yield Static("Select an incident.", id="incident_detail")
        yield Footer()

    # ---------------- lifecycle ----------------

    def on_mount(self) -> None:
        table = self.query_one("#user_table", DataTable)
        table.add_columns("name", "last_seen", "msgs", "notes", "opt-out")
        if self.settings.mod_mode_enabled:
            inc = self.query_one("#incidents_table", DataTable)
            inc.add_columns("when", "sev", "user", "categories", "status")
        self.refresh_chatters()
        self.refresh_topics()
        self.refresh_events()
        self.refresh_live()
        if self.settings.mod_mode_enabled:
            self.refresh_incidents()
        self.set_interval(10.0, self.refresh_chatters)
        self.set_interval(15.0, self.refresh_topics)
        self.set_interval(15.0, self.refresh_events)
        self.set_interval(2.0, self.refresh_live)
        if self.settings.mod_mode_enabled:
            self.set_interval(15.0, self.refresh_incidents)

    # ---------------- generic actions ----------------

    def action_refresh(self) -> None:
        self.refresh_chatters()
        self.refresh_topics()
        self.refresh_events()
        self.refresh_live()
        if self.settings.mod_mode_enabled:
            self.refresh_incidents()
        if self.selected_user_id:
            self._render_user_detail(self.selected_user_id)

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    def action_switch_tab(self, tab_id: str) -> None:
        if tab_id == "moderation" and not self.settings.mod_mode_enabled:
            return
        self.query_one(TabbedContent).active = tab_id

    def action_diagnose(self) -> None:
        out = Path(default_bundle_filename())
        try:
            path = build_diagnostic_bundle(out, self.settings)
        except Exception as e:
            self.notify(f"diagnose failed: {e}", severity="error")
            return
        self.notify(f"saved {path}  ({path.stat().st_size:,} bytes)", timeout=8.0)

    # ---------------- chatters tab ----------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search":
            return
        self._search_query = event.value.strip()
        self.refresh_chatters()

    def refresh_chatters(self) -> None:
        table = self.query_one("#user_table", DataTable)
        previously_selected = self.selected_user_id
        rows = self.repo.list_chatters(query=self._search_query, limit=200)
        table.clear()
        for r in rows:
            table.add_row(
                r.user.name,
                _short_ts(r.user.last_seen),
                str(r.msg_count),
                str(r.note_count),
                "yes" if r.user.opt_out else "",
                key=r.user.twitch_id,
            )
        if previously_selected and any(r.user.twitch_id == previously_selected for r in rows):
            try:
                table.move_cursor(row=_row_index(rows, previously_selected))
            except Exception:
                pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        key = str(event.row_key.value) if event.row_key else None
        if not key:
            return
        if event.data_table.id == "user_table":
            self.selected_user_id = key
            self._render_user_detail(key)
        elif event.data_table.id == "incidents_table":
            try:
                self.selected_incident_id = int(key)
            except ValueError:
                return
            self._render_incident_detail(int(key))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = str(event.row_key.value) if event.row_key else None
        if not key:
            return
        if event.data_table.id == "user_table":
            self.selected_user_id = key
            self._render_user_detail(key)
        elif event.data_table.id == "incidents_table":
            try:
                self.selected_incident_id = int(key)
            except ValueError:
                return
            self._render_incident_detail(int(key))

    def action_forget_user(self) -> None:
        uid = self.selected_user_id
        if not uid:
            return
        self.repo.forget_user(uid)
        self.selected_user_id = None
        self.query_one("#notes_box", Static).update("Select a chatter.")
        self.query_one("#events_box", Static).update("")
        self.query_one("#msgs_box", Static).update("")
        self.refresh_chatters()
        self.notify("forgotten.", severity="warning")

    def action_toggle_opt_out(self) -> None:
        uid = self.selected_user_id
        if not uid:
            return
        user = self.repo.get_user(uid)
        if not user:
            return
        self.repo.set_opt_out(uid, not user.opt_out)
        self.refresh_chatters()
        self._render_user_detail(uid)

    def action_delete_note(self) -> None:
        uid = self.selected_user_id
        if not uid:
            return
        notes = self.repo.get_notes(uid)
        if not notes:
            return
        self.repo.delete_note(notes[0].id)
        self._render_user_detail(uid)
        self.refresh_chatters()
        self.notify("deleted most-recent note.")

    def action_add_note(self) -> None:
        uid = self.selected_user_id
        if not uid:
            self.notify("select a chatter first.", severity="warning")
            return
        user = self.repo.get_user(uid)
        if not user:
            return

        def on_dismiss(text: str | None) -> None:
            if not text:
                return
            self._add_note_with_embedding(uid, text)

        self.push_screen(AddNoteScreen(user.name), on_dismiss)

    @work(exclusive=False)
    async def _add_note_with_embedding(self, user_id: str, text: str) -> None:
        """Embed (if LLM available) then store. Manual notes have no source link."""
        embedding: list[float] | None = None
        if self.llm is not None:
            try:
                embedding = await self.llm.embed(text[:500])
            except Exception:
                embedding = None
        # repo is sync; fine on the worker thread.
        self.repo.add_note(user_id, text[:500], embedding)
        self.call_from_thread(self._render_user_detail, user_id)
        self.call_from_thread(self.refresh_chatters)
        self.call_from_thread(self.notify, "note added.")

    def _render_user_detail(self, twitch_id: str) -> None:
        user = self.repo.get_user(twitch_id)
        notes_box = self.query_one("#notes_box", Static)
        events_box = self.query_one("#events_box", Static)
        msgs_box = self.query_one("#msgs_box", Static)
        if not user:
            notes_box.update("[user gone]")
            events_box.update("")
            msgs_box.update("")
            return
        nws = self.repo.get_notes_with_sources(twitch_id)
        aliases = self.repo.get_user_aliases(twitch_id)
        notes_box.update(_format_notes(user, nws, aliases))
        summary = self.repo.get_user_event_summary(twitch_id)
        recent_events = self.repo.get_user_events(twitch_id, limit=8)
        events_box.update(_format_events(summary, recent_events))
        recent_msgs = self.repo.get_messages(twitch_id, limit=12)
        msgs_box.update(_format_messages(recent_msgs))

    # ---------------- topics tab ----------------

    def refresh_topics(self) -> None:
        snapshots = self.repo.list_topic_snapshots(limit=20)
        widget = self.query_one("#topics_list", Static)
        if not snapshots:
            widget.update(
                f"[dim]No topic snapshots yet. They appear every "
                f"{self.settings.topics_interval_minutes} minutes.[/dim]"
            )
            return
        chunks: list[str] = []
        for s in snapshots:
            header = f"[b]{_short_ts(s.ts)}[/b]"
            chunks.append(f"{header}\n{s.summary}")
        widget.update("\n\n──\n\n".join(chunks))

    # ---------------- events tab ----------------

    def refresh_events(self) -> None:
        events = self.repo.list_events(limit=200)
        widget = self.query_one("#events_list", Static)
        if not events:
            widget.update("[dim]No StreamElements events yet.[/dim]")
            return
        lines: list[str] = []
        for e in events:
            amt = ""
            if e.amount is not None:
                amt = f"{e.amount:g}"
                if e.currency:
                    amt += f" {e.currency}"
            msg = f" — {e.message}" if e.message else ""
            lines.append(
                f"{_short_ts(e.ts)}  [b]{e.type:<5}[/b] "
                f"{e.twitch_name:<20} {amt}{msg}"
            )
        widget.update("\n".join(lines))

    # ---------------- live tab ----------------

    def refresh_live(self) -> None:
        try:
            widget = self.query_one("#live_feed", Static)
        except Exception:
            return
        msgs = self.repo.recent_global_messages(limit=30)
        if not msgs:
            widget.update("[dim]No messages yet.[/dim]")
            return
        lines = []
        for m in reversed(msgs):  # display oldest-first within the visible window
            prefix = ""
            if m.reply_parent_body:
                snip = m.reply_parent_body[:80]
                prefix = (
                    f"  [dim]↩ replying to {m.reply_parent_login or '?'}: "
                    f"\"{snip}\"[/dim]\n"
                )
            lines.append(
                f"{prefix}[dim]{_short_ts(m.ts)}[/dim]  "
                f"[b]{m.name}[/b]  {m.content}"
            )
        widget.update("\n".join(lines))

    # ---------------- moderation tab ----------------

    def refresh_incidents(self) -> None:
        if not self.settings.mod_mode_enabled:
            return
        try:
            table = self.query_one("#incidents_table", DataTable)
        except Exception:
            return
        rows = self.repo.list_incidents(status=None, limit=200)
        table.clear()
        for r in rows:
            table.add_row(
                _short_ts(r.incident.ts),
                str(r.incident.severity),
                r.user_name or "?",
                ",".join(r.incident.categories) or "—",
                r.incident.status,
                key=str(r.incident.id),
            )

    def _render_incident_detail(self, incident_id: int) -> None:
        row = self.repo.get_incident(incident_id)
        widget = self.query_one("#incident_detail", Static)
        if not row:
            widget.update("[user gone]")
            return
        widget.update(_format_incident(row))

    def action_incident_status(self, status: str) -> None:
        if not self.settings.mod_mode_enabled:
            return
        if status not in ("open", "reviewed", "dismissed"):
            return
        if self.selected_incident_id is None:
            return
        try:
            self.repo.update_incident_status(self.selected_incident_id, status)
        except Exception as e:
            self.notify(f"failed: {e}", severity="error")
            return
        self.refresh_incidents()
        self._render_incident_detail(self.selected_incident_id)
        self.notify(f"incident → {status}.")


# ---------------- formatting helpers ----------------


def _short_ts(ts: str | None) -> str:
    if not ts:
        return ""
    return ts.replace("T", " ")[:16]


def _row_index(rows: list[ChatterRow], twitch_id: str) -> int:
    for i, r in enumerate(rows):
        if r.user.twitch_id == twitch_id:
            return i
    return 0


def _format_notes(
    user: User, notes_with_sources: list[NoteWithSources], aliases: list,
) -> str:
    header_lines = [
        f"[b]{user.name}[/b]  [dim]({user.twitch_id})[/dim]",
        f"first seen: {_short_ts(user.first_seen)}    last seen: {_short_ts(user.last_seen)}",
    ]
    prior = [a.name for a in aliases if a.name != user.name]
    if prior:
        header_lines.append(
            f"[dim]previously: {', '.join(prior)}[/dim]"
        )
    if user.opt_out:
        header_lines.append("[red]OPTED OUT — no new notes will be created[/red]")
    if not notes_with_sources:
        body = "\n[dim]No notes yet.   Press 'n' to add one manually.[/dim]"
    else:
        body_lines = ["", "[b]Notes[/b]"]
        for nws in notes_with_sources:
            n = nws.note
            sources = nws.sources
            tag = (
                f"[dim]({len(sources)} src)[/dim]"
                if sources else "[dim](manual)[/dim]"
            )
            body_lines.append(
                f"  • {n.text}  {tag}  [dim]{_short_ts(n.ts)}[/dim]"
            )
            # Show first cited source inline (if any) so the streamer can see
            # where the fact came from at a glance.
            if sources:
                m = sources[0]
                snippet = m.content[:90]
                body_lines.append(
                    f"      [dim]from [{m.id}] {snippet}[/dim]"
                )
        body = "\n".join(body_lines)
    return "\n".join(header_lines) + "\n" + body


def _format_events(summary: UserEventSummary, recent: list[Event]) -> str:
    if (
        summary.total_tip_amount == 0.0
        and summary.total_bits == 0
        and summary.sub_months == 0
        and not recent
    ):
        return "[dim]No StreamElements events.[/dim]"
    lines = ["[b]StreamElements[/b]"]
    if summary.total_tip_amount:
        cur = summary.tip_currency or ""
        last = f" (last {_short_ts(summary.last_tip_ts)})" if summary.last_tip_ts else ""
        lines.append(f"  tips: {summary.total_tip_amount:.2f} {cur}{last}")
    if summary.total_bits:
        lines.append(f"  bits: {summary.total_bits}")
    if summary.sub_months:
        tier = summary.sub_tier or ""
        last = f" (last {_short_ts(summary.last_sub_ts)})" if summary.last_sub_ts else ""
        lines.append(f"  sub: {summary.sub_months:g} months tier={tier}{last}")
    if recent:
        lines.append("  [dim]recent:[/dim]")
        for e in recent[:5]:
            amt = ""
            if e.amount is not None:
                amt = f" {e.amount:g}"
                if e.currency:
                    amt += f" {e.currency}"
            msg = f" — {e.message}" if e.message else ""
            lines.append(f"    {_short_ts(e.ts)} {e.type}{amt}{msg}")
    return "\n".join(lines)


def _format_messages(msgs: list[Message]) -> str:
    if not msgs:
        return "[dim]No messages yet.[/dim]"
    lines = ["[b]Recent messages[/b]"]
    for m in msgs:
        if m.reply_parent_body:
            snip = m.reply_parent_body[:60]
            lines.append(
                f"  [dim]↩ {m.reply_parent_login or '?'}: \"{snip}\"[/dim]"
            )
        lines.append(f"  [dim]{_short_ts(m.ts)}[/dim]  {m.content}")
    return "\n".join(lines)


def _format_incident(row: IncidentRow) -> str:
    sev_word = {1: "minor", 2: "warning", 3: "serious"}.get(
        row.incident.severity, "?"
    )
    sev_color = {1: "yellow", 2: "orange3", 3: "red"}.get(
        row.incident.severity, "white"
    )
    lines = [
        f"[b]{row.user_name or '?'}[/b]  "
        f"[{sev_color}]{sev_word}[/{sev_color}]  "
        f"[dim]({row.incident.status})[/dim]",
        f"[dim]{_short_ts(row.incident.ts)} · "
        f"{','.join(row.incident.categories) or '—'}[/dim]",
    ]
    if row.message_reply_parent_body:
        snip = row.message_reply_parent_body[:80]
        lines.append(
            f"[dim]↩ replying to {row.message_reply_parent_login or '?'}: "
            f"\"{snip}\"[/dim]"
        )
    if row.message_content:
        lines.append(f'[red]"{row.message_content}"[/red]')
    if row.incident.rationale:
        lines.append(f"[dim]🤖 {row.incident.rationale}[/dim]")
    lines.append("")
    lines.append(
        "[dim]m=mark reviewed   x=dismiss   R=reopen[/dim]"
    )
    return "\n".join(lines)


def run_tui(
    repo: ChatterRepo, settings: Settings, llm: OllamaClient | None = None
) -> None:
    app = ChatterbotTUI(repo, settings, llm=llm)
    app.run()
