"""Streamer-only terminal UI.

Three tabs:
  - Chatters: searchable list of recent chatters; selecting one shows their
    notes plus a side panel with their StreamElements event history and a
    stack of recent messages. Notes can be deleted with `d`.
  - Topics:   rolling history of channel-wide topic snapshots, newest first.
  - Events:   chronological list of StreamElements events.

Hard rule: this is the only TUI surface for notes / events / topic snapshots.
None of this data ever feeds a chat-facing LLM prompt.

The TUI shares the SQLite DB with the bot and the dashboard via WAL mode.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Static,
    TabbedContent,
    TabPane,
)

from .config import Settings
from .repo import ChatterRepo, ChatterRow, Event, Message, Note, User, UserEventSummary


class ChatterbotTUI(App):
    CSS = """
    #chatters_layout { height: 1fr; }
    #left_pane { width: 38%; min-width: 32; border-right: solid $primary; }
    #right_pane { width: 1fr; padding: 0 1; }
    #search { dock: top; }
    #user_table { height: 1fr; }
    #notes_box  { height: 8; border: round $primary; padding: 0 1; }
    #events_box { height: 8; border: round $accent; padding: 0 1; margin-top: 1; }
    #msgs_box   { height: 1fr; border: round $secondary; padding: 0 1; margin-top: 1; }
    #topics_list { padding: 1; }
    #events_list { padding: 1; }
    .muted { color: $text-muted; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("slash", "focus_search", "Search"),
        Binding("f", "forget_user", "Forget"),
        Binding("o", "toggle_opt_out", "Opt-out"),
        Binding("d", "delete_note", "Del note"),
        Binding("1", "switch_tab('chatters')", "Chatters"),
        Binding("2", "switch_tab('topics')", "Topics"),
        Binding("3", "switch_tab('events')", "Events"),
    ]

    selected_user_id: reactive[str | None] = reactive(None)

    def __init__(self, repo: ChatterRepo, settings: Settings):
        super().__init__()
        self.repo = repo
        self.settings = settings
        self._search_query = ""

    # ---------------- compose ----------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="chatters", id="tabs"):
            with TabPane("Chatters", id="chatters"):
                yield Input(placeholder="search by name…", id="search")
                with Horizontal(id="chatters_layout"):
                    with Vertical(id="left_pane"):
                        yield DataTable(id="user_table", cursor_type="row", zebra_stripes=True)
                    with Vertical(id="right_pane"):
                        yield Static("Select a chatter.", id="notes_box")
                        yield Static("", id="events_box")
                        yield Static("", id="msgs_box")
            with TabPane("Live Topics", id="topics"):
                yield Static("Loading topic snapshots…", id="topics_list")
            with TabPane("Events", id="events"):
                yield Static("Loading events…", id="events_list")
        yield Footer()

    # ---------------- lifecycle ----------------

    def on_mount(self) -> None:
        table = self.query_one("#user_table", DataTable)
        table.add_columns("name", "last_seen", "msgs", "notes", "opt-out")
        self.refresh_chatters()
        self.refresh_topics()
        self.refresh_events()
        self.set_interval(10.0, self.refresh_chatters)
        self.set_interval(15.0, self.refresh_topics)
        self.set_interval(15.0, self.refresh_events)

    # ---------------- actions ----------------

    def action_refresh(self) -> None:
        self.refresh_chatters()
        self.refresh_topics()
        self.refresh_events()
        if self.selected_user_id:
            self._render_user_detail(self.selected_user_id)

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

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
        # Deletes the most-recent note for the selected user. Quick affordance
        # for "this note is wrong / privacy violation". Use the dashboard for
        # finer-grained editing.
        uid = self.selected_user_id
        if not uid:
            return
        notes = self.repo.get_notes(uid)
        if not notes:
            return
        self.repo.delete_note(notes[0].id)
        self._render_user_detail(uid)
        self.refresh_chatters()

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
        uid = str(event.row_key.value) if event.row_key else None
        if uid:
            self.selected_user_id = uid
            self._render_user_detail(uid)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        uid = str(event.row_key.value) if event.row_key else None
        if uid:
            self.selected_user_id = uid
            self._render_user_detail(uid)

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
        notes = self.repo.get_notes(twitch_id)
        notes_box.update(_format_notes(user, notes))
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


def _format_notes(user: User, notes: list[Note]) -> str:
    header_lines = [
        f"[b]{user.name}[/b]  [dim]({user.twitch_id})[/dim]",
        f"first seen: {_short_ts(user.first_seen)}    last seen: {_short_ts(user.last_seen)}",
    ]
    if user.opt_out:
        header_lines.append("[red]OPTED OUT — no new notes will be created[/red]")
    if not notes:
        body = "\n[dim]No notes yet.[/dim]"
    else:
        body_lines = ["", "[b]Notes[/b]"]
        for n in notes:
            body_lines.append(f"  • {n.text}  [dim]{_short_ts(n.ts)}[/dim]")
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
        lines.append(f"  [dim]{_short_ts(m.ts)}[/dim]  {m.content}")
    return "\n".join(lines)


def run_tui(repo: ChatterRepo, settings: Settings) -> None:
    app = ChatterbotTUI(repo, settings)
    app.run()
