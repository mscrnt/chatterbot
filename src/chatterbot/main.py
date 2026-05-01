"""CLI entrypoint.

Usage:
  chatterbot bot         # silent listener: TwitchIO + summarizer + SE + topics loop
  chatterbot tui         # streamer-only Textual viewer
  chatterbot dashboard   # streamer-only FastAPI dashboard
  chatterbot diagnose    # write a privacy-safe .cbreport bundle for bug reports

The first three are independent processes — they share the SQLite DB via WAL
so they can run simultaneously. `diagnose` is a one-shot.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .bot import ChatterListener
from .config import Settings, get_settings
from .diagnose import build_diagnostic_bundle, default_bundle_filename
from .discord_bot import DiscordListener
from .llm.ollama_client import OllamaClient
from .helix_sync import HelixSyncService
from .logging_setup import setup_logging
from .moderator import Moderator
from .obs import OBSStatusService
from .repo import ChatterRepo
from .streamelements import StreamElementsListener
from .summarizer import Summarizer
from .tui import run_tui
from .youtube import YouTubeListener

logger = logging.getLogger(__name__)


_PID_FILE = Path("data/.bot.pid")


def _write_pid_file() -> None:
    try:
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(__import__("os").getpid()))
    except OSError:
        logger.warning("could not write %s — restart-bot button won't find this process", _PID_FILE)


def _clear_pid_file() -> None:
    try:
        _PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# Poll cadence for the lifecycle flags. Short enough that streamer
# clicks feel responsive (<5s perceived); long enough that an idle
# bot isn't hammering the DB on a tight loop.
_LIFECYCLE_POLL_SECONDS = 3


def _apply_reload(reconfigurables: list, new_settings: Settings) -> None:
    """Fan out a fresh Settings snapshot to every long-lived service
    that holds its own reference. Two paths per service:

      - If the service exposes `reconfigure(settings)`, call it. Use
        this when a service has nested children with their own
        settings (Summarizer holds a Threader; ChatterListener holds
        the same Summarizer) so the call can propagate.

      - Otherwise duck-type-swap `s.settings = new_settings`. Most
        reload-class settings are read live via
        `getattr(self.settings, key, default)` at use time, so simply
        swapping the reference makes the next loop iteration pick up
        the new value — no explicit method needed.

    Failures are logged but don't propagate; one broken service's
    reconfigure shouldn't block the rest from picking up settings."""
    for s in reconfigurables:
        try:
            if hasattr(s, "reconfigure"):
                s.reconfigure(new_settings)
            else:
                s.settings = new_settings
        except Exception:
            logger.exception(
                "lifecycle reload: %s.reconfigure failed; continuing",
                type(s).__name__,
            )


async def _lifecycle_poller(
    repo: ChatterRepo,
    shutdown_event: asyncio.Event,
    reconfigurables: list,
) -> None:
    """Watch the dashboard-driven lifecycle flags and act when they
    advance. The dashboard writes `bot_restart_at` / `bot_reload_at`
    to app_settings as ISO timestamps; we remember the values we saw
    at boot and trigger the corresponding action whenever the
    persisted value moves past the one in memory.

      - Restart: set `shutdown_event` so run_bot's outer asyncio.wait
        breaks out and the existing finally-block cancels every task
        cleanly. The supervisor (`make all`, docker compose restart-
        policy, systemd) brings it back with the new settings.

      - Reload: pull a fresh Settings (env + DB overrides) and fan
        out to every service via `_apply_reload`. No process bounce,
        no in-flight LLM calls dropped. Settings classified as
        'restart' in settings_meta won't actually take effect via
        reload (the Twitch IRC client / LLM client are still using
        old creds); the restart-impact badge in the UI tells the
        streamer that.

    This indirection (vs os.kill) is what makes both buttons work
    uniformly across topologies — same-process, separate containers,
    and separate hosts. Both sides only need to share the SQLite
    file."""
    from .config import get_settings as _get_settings
    last_restart = (await asyncio.to_thread(
        repo.get_app_setting, "bot_restart_at",
    )) or ""
    last_reload = (await asyncio.to_thread(
        repo.get_app_setting, "bot_reload_at",
    )) or ""
    while True:
        try:
            await asyncio.sleep(_LIFECYCLE_POLL_SECONDS)
            cur_restart = (await asyncio.to_thread(
                repo.get_app_setting, "bot_restart_at",
            )) or ""
            if cur_restart and cur_restart != last_restart:
                logger.info(
                    "lifecycle: restart requested at %s — signalling "
                    "clean shutdown (supervisor will respawn)",
                    cur_restart,
                )
                shutdown_event.set()
                return

            cur_reload = (await asyncio.to_thread(
                repo.get_app_setting, "bot_reload_at",
            )) or ""
            if cur_reload and cur_reload != last_reload:
                logger.info(
                    "lifecycle: reload requested at %s — fanning out "
                    "to %d service(s)", cur_reload, len(reconfigurables),
                )
                try:
                    new_settings = await asyncio.to_thread(_get_settings)
                except Exception:
                    logger.exception(
                        "lifecycle reload: failed to load fresh "
                        "Settings; skipping this reload tick",
                    )
                else:
                    await asyncio.to_thread(
                        _apply_reload, reconfigurables, new_settings,
                    )
                last_reload = cur_reload
            last_restart = cur_restart
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("lifecycle_poller: iteration failed")


async def run_bot(settings: Settings) -> None:
    if not settings.twitch_oauth_token or not settings.twitch_channel:
        logger.error("missing TWITCH_OAUTH_TOKEN or TWITCH_CHANNEL — refusing to start")
        return

    _write_pid_file()

    repo = ChatterRepo(settings.db_path, embed_dim=settings.ollama_embed_dim)
    # `make_llm_client` returns the configured provider (Ollama by
    # default, Claude / OpenAI via `LLM_PROVIDER` setting). Embeddings
    # always go to a local Ollama regardless of provider.
    from .llm.providers import make_llm_client
    llm = make_llm_client(settings)

    if not await llm.health_check():
        logger.warning(
            "ollama health check failed at %s — starting anyway",
            settings.ollama_base_url,
        )

    # Personal-dataset capture unlock — opt-in. No-op when the toggle
    # is off, the wrapped DEK is missing, or the passphrase env var
    # isn't set. Capture is silently off in any of those cases; the
    # bot runs as if the feature didn't exist.
    from .dataset import try_unlock_at_startup
    try_unlock_at_startup(repo, llm)

    # OBS is the authoritative "are we streaming?" signal. The bot process
    # owns its own poller (the dashboard has a separate one for the nav)
    # so YouTube + other quota-sensitive pollers can pause when offline.
    obs = OBSStatusService(settings)
    # Twitch Helix poller — separate instance from the dashboard's,
    # since the bot is a different process. Feeds the summarizer's
    # topic-snapshot call with live channel context (game / title /
    # tags / viewer tier / uptime). OBS-aware pause keeps Helix
    # quiet while we're not streaming.
    from .twitch import TwitchService
    twitch_status = TwitchService(settings, obs=obs)
    summarizer = Summarizer(repo, llm, settings, twitch_status=twitch_status)
    listener = ChatterListener(settings, repo, summarizer)
    se = StreamElementsListener(repo, settings)

    # One-time backfill of any pre-existing topic_snapshots into the
    # topic_threads index. Idempotent — only operates on snapshots that
    # don't yet have thread members. Non-fatal: bot keeps running on error.
    try:
        n = await summarizer._threader.backfill()
        if n:
            logger.info("threader: backfilled %d existing snapshots into threads", n)
    except Exception:
        logger.exception("threader: backfill failed — clustering will catch up on the next snapshot")

    tasks: list[asyncio.Task] = [
        asyncio.create_task(listener.start(), name="twitch_listener"),
        asyncio.create_task(summarizer.idle_loop(), name="summarizer_idle"),
        asyncio.create_task(summarizer.topics_loop(), name="summarizer_topics"),
    ]
    if settings.message_embed_interval_seconds > 0:
        tasks.append(
            asyncio.create_task(summarizer.embed_loop(), name="message_embedder")
        )
    if settings.obs_enabled:
        tasks.append(asyncio.create_task(obs.poll_loop(), name="obs_poll"))
    # Helix poller — gated on credentials being present. The
    # poll_loop self-terminates on missing oauth token / channel, so
    # always-on-task with no extra guard is safe.
    if settings.twitch_oauth_token and settings.twitch_channel:
        tasks.append(
            asyncio.create_task(twitch_status.poll_loop(), name="twitch_poll"),
        )

        # Stream-start session reset. On not-streaming → streaming,
        # wipe per-session ephemeral state (currently-addressed insight
        # cards, engaging-subjects blocklist) so the new stream starts
        # with a clean slate. The audit history table preserves the
        # transitions so /audit still tells the full story.
        def _on_stream_start_reset() -> None:
            try:
                cleared = repo.reset_session_addressed_states()
                if cleared:
                    logger.info(
                        "session-reset: cleared %d 'addressed' insight "
                        "states for new stream", cleared,
                    )
            except Exception:
                logger.exception(
                    "session-reset: failed to clear addressed states"
                )
            # Engaging-subjects blocklist is owned by the dashboard
            # process via app_settings (cross-process safe). Wipe it
            # so the new stream's subject extractor starts fresh.
            try:
                repo.set_app_setting(
                    "engaging_subjects_blocklist", "[]",
                )
                logger.info("session-reset: cleared engaging-subjects blocklist")
            except Exception:
                logger.exception(
                    "session-reset: failed to clear subjects blocklist"
                )

        # End-of-stream recap: watch OBS state, fire LLM debrief on
        # streaming → not-streaming transition. No-op if OBS is disabled.
        tasks.append(asyncio.create_task(
            summarizer.recap_loop(obs, on_stream_start=_on_stream_start_reset),
            name="recap_loop",
        ))
    if settings.streamelements_enabled:
        tasks.append(asyncio.create_task(se.run(), name="streamelements"))
    if settings.mod_mode_enabled:
        moderator = Moderator(repo, llm, settings)
        tasks.append(asyncio.create_task(moderator.review_loop(), name="moderator"))
        logger.info("moderation mode ENABLED — advisory-only classifier active")
    # Helix roster sync — VIPs / mods / subs / followers polled
    # periodically into the users table. No-op without scopes or
    # without TWITCH_OAUTH_TOKEN. OBS-aware: pauses while offline.
    helix_sync = HelixSyncService(settings, repo, obs=obs)
    if helix_sync.configured:
        tasks.append(asyncio.create_task(helix_sync.run(), name="helix_sync"))

    # Cross-platform listeners. Both no-op when disabled or unconfigured.
    # YouTube takes the OBS poller so it can skip its 100-unit search.list
    # while we're not actually streaming (OBS-confirmed offline).
    yt = None
    if settings.youtube_enabled:
        yt = YouTubeListener(settings, repo, summarizer, obs=obs)
        tasks.append(asyncio.create_task(yt.start(), name="youtube_listener"))
    dc = None
    if settings.discord_enabled:
        dc = DiscordListener(settings, repo, summarizer)
        tasks.append(asyncio.create_task(dc.start(), name="discord_listener"))

    # Reconfigurables — every long-lived service that holds a
    # Settings reference. The lifecycle_poller's reload path fans a
    # fresh Settings to each on `bot_reload_at` advance. Order
    # matters mildly: parents before children, since a parent's
    # reconfigure() may propagate to its children (Summarizer →
    # Threader, etc.). Top-level only — nested services that don't
    # need explicit reconfigure() get reached via attribute hop.
    reconfigurables: list = [obs, twitch_status, summarizer, listener, se]
    if settings.mod_mode_enabled:
        # `moderator` is only defined inside the if branch above; pull
        # it from the local scope when present.
        try:
            reconfigurables.append(moderator)  # type: ignore[name-defined]
        except NameError:
            pass
    if helix_sync.configured:
        reconfigurables.append(helix_sync)
    if yt is not None:
        reconfigurables.append(yt)
    if dc is not None:
        reconfigurables.append(dc)

    # Lifecycle poller — watches the dashboard's `bot_restart_at`
    # and `bot_reload_at` flags in app_settings. Restart sets
    # `shutdown_event` so the outer await below breaks out, the
    # finally-block cancels every task cleanly, and run_bot returns
    # for the supervisor to respawn. Reload fans a fresh Settings
    # snapshot to every service in `reconfigurables` in place — no
    # process bounce, no in-flight LLM calls dropped.
    #
    # Why DB-flag instead of SIGTERM: works the same in-container
    # and bare-metal because there's no cross-PID-namespace
    # signaling. Both sides only need to share the SQLite file.
    shutdown_event = asyncio.Event()
    lifecycle_task = asyncio.create_task(
        _lifecycle_poller(repo, shutdown_event, reconfigurables),
        name="lifecycle_poller",
    )
    tasks.append(lifecycle_task)

    shutdown_waiter = asyncio.create_task(
        shutdown_event.wait(), name="shutdown_waiter",
    )
    try:
        # asyncio.wait with FIRST_COMPLETED so a service crashing
        # OR the lifecycle poller setting `shutdown_event` both
        # exit the bot cleanly. Without FIRST_COMPLETED, a single
        # task crash would leave the rest running silently.
        await asyncio.wait(
            [*tasks, shutdown_waiter],
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        pass
    finally:
        shutdown_waiter.cancel()
        for t in tasks:
            t.cancel()
        await asyncio.gather(
            *tasks, shutdown_waiter, return_exceptions=True,
        )
        repo.close()
        _clear_pid_file()


def run_dashboard(settings: Settings) -> None:
    import uvicorn

    from .web.app import create_app

    repo = ChatterRepo(settings.db_path, embed_dim=settings.ollama_embed_dim)
    app = create_app(repo, settings)
    if not settings.dashboard_basic_auth_enabled and settings.dashboard_host not in (
        "127.0.0.1",
        "localhost",
    ):
        logger.warning(
            "dashboard bound to %s without basic auth — anyone on this network can read it",
            settings.dashboard_host,
        )
    try:
        uvicorn.run(
            app,
            host=settings.dashboard_host,
            port=settings.dashboard_port,
            log_level="info",
        )
    finally:
        repo.close()


def run_diagnose(
    settings: Settings,
    with_recent_activity: bool,
    anonymize_recent_activity: bool = False,
) -> Path:
    out = Path(default_bundle_filename())
    path = build_diagnostic_bundle(
        out, settings,
        with_recent_activity=with_recent_activity,
        anonymize_recent_activity=anonymize_recent_activity,
    )
    print(f"wrote {path}  ({path.stat().st_size:,} bytes)")
    if with_recent_activity:
        if anonymize_recent_activity:
            print(
                "(includes recent_activity.json with anonymised "
                "<USER_NNN> tokens — no real chatter names)"
            )
        else:
            print("(includes opt-in recent_activity.json — usernames + per-user message counts)")
    else:
        print("(minimal mode — no chat content, no usernames, no secrets)")
    return path


def main() -> None:
    # Pre-dispatch for the `dataset` subcommand. argparse can't cleanly
    # mix a positional `mode` with subparsers (the first positional
    # always tries to match a subparser choice first), so we peel
    # `chatterbot dataset ...` off before the legacy parser ever sees
    # it. This keeps `chatterbot bot/tui/dashboard/diagnose` working
    # exactly as before.
    if len(sys.argv) >= 2 and sys.argv[1] == "dataset":
        from .dataset import cli as dataset_cli
        sub_parser = argparse.ArgumentParser(prog="chatterbot dataset")
        sub_subs = sub_parser.add_subparsers(dest="dataset_cmd", required=True)
        dataset_cli.register_subcommands(sub_subs)
        args = sub_parser.parse_args(sys.argv[2:])
        settings = get_settings()
        setup_logging("dataset")
        sys.exit(dataset_cli.dispatch(args, settings))

    parser = argparse.ArgumentParser(prog="chatterbot")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("bot", "tui", "dashboard", "diagnose"),
        default=None,
        help="bot = silent listener; tui = Textual viewer; "
             "dashboard = FastAPI viewer; diagnose = write a .cbreport bundle. "
             "Use `chatterbot dataset --help` for the opt-in dataset capture "
             "subcommands.",
    )
    parser.add_argument(
        "--with-recent-activity",
        action="store_true",
        help="(diagnose only) include usernames + per-user message counts "
             "from the last 24h. Off by default for privacy.",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="(diagnose only, with --with-recent-activity) replace chatter "
             "usernames with stable <USER_NNN> tokens via the dataset "
             "redactor. Lets you share activity SHAPE without identities.",
    )

    args = parser.parse_args()

    settings = get_settings()
    mode = args.mode or settings.run_mode

    setup_logging(mode)

    if mode == "bot":
        try:
            asyncio.run(run_bot(settings))
        except KeyboardInterrupt:
            pass
    elif mode == "tui":
        repo = ChatterRepo(settings.db_path, embed_dim=settings.ollama_embed_dim)
        # Optional LLM client so manual notes added via the TUI can be embedded
        # for RAG. Failures inside the TUI are non-fatal — note still saves.
        from .llm.providers import make_llm_client
        llm = make_llm_client(settings)
        try:
            run_tui(repo, settings, llm=llm)
        finally:
            repo.close()
    elif mode == "dashboard":
        run_dashboard(settings)
    elif mode == "diagnose":
        run_diagnose(
            settings,
            with_recent_activity=args.with_recent_activity,
            anonymize_recent_activity=(
                args.anonymize and args.with_recent_activity
            ),
        )
    else:
        parser.error(f"unknown mode: {mode!r}")


if __name__ == "__main__":
    main()
