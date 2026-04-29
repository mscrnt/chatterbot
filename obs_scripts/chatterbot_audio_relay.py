"""chatterbot audio relay — OBS script that coordinates audio_client.py
subprocesses, one per enabled audio source.

Why this is a coordinator and not a direct hook
-----------------------------------------------
obspython's SWIG bindings can't marshal a Python callable into the C
callback type `obs_source_audio_capture_t`, so attaching directly to
OBS audio sources from Python isn't possible. Instead, this script
spawns one or more `audio_client.py` instances as subprocesses; each
opens a system audio device via PortAudio (sounddevice) and POSTs PCM
chunks to chatterbot's `/audio/ingest`.

Two slots are exposed by default:

  * **Mic** — any input device on your system.
  * **System / window audio** — on Windows, any output device opened
    via WASAPI loopback. Lets you transcribe browsers, games, etc.
    without a virtual cable. (Linux/macOS: no loopback in PortAudio,
    so the slot is hidden / disabled there.)

Both feed the same dashboard endpoint. The transcript pipeline doesn't
care which client the audio came from — utterances are just buffered,
VAD-filtered, and embedded.

Setup
-----
1. From the chatterbot repo root, run once to bootstrap the Windows
   venv that audio_client.py uses:

       obs_scripts\\audio_client.bat --list-devices

   This creates `.venv-win` with numpy + sounddevice. Linux/macOS:
   the regular `.venv` from `uv sync --extra whisper` is reused.

2. In OBS: Tools → Scripts → + → pick this file
   (`chatterbot_audio_relay.py`).

3. Configure dashboard URL, pick devices, hit **Start capture**.
   Subprocesses persist across OBS scene changes; they only stop on
   "Stop capture" or when you remove the script.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import Any

import obspython as obs


# ---------------- module state ------------------------------------------

# These globals hold the script's runtime state. OBS reloads the module
# on script changes, so we rely on script_load / script_unload to
# (re)initialise them.

_state: dict[str, Any] = {
    "url": "http://127.0.0.1:8765",
    "auth_user": "",
    "auth_pass": "",
    "mic_enabled": False,
    "mic_device": "",         # stored as the device index, stringified
    "sys_enabled": False,
    "sys_device": "",
    "running": False,
    "procs": {},              # slot_name -> subprocess.Popen
    "devices": {              # cached from --list-devices-json
        "input_devices": [],
        "loopback_devices": [],
        "platform": sys.platform,
    },
    "last_status": "idle",
    # ---- auto-recover state ----
    # When `auto_recover` is on we (a) start capture immediately on
    # script_load if at least one slot is enabled, and (b) periodically
    # health-check the dashboard URL — if it transitions from
    # unreachable → reachable while we're supposed to be running, the
    # health thread sets `recover_pending` and script_tick acts on it
    # (subprocess.Popen stays on the OBS main thread, no shared-state
    # races). Default ON because the streamer's complaint is exactly
    # this workflow — restart chatterbot, capture's gone.
    "auto_recover": True,
    "health_interval_s": 10,
    "dashboard_reachable": None,   # tri-state: None / True / False
    "recover_pending": False,
    "_health_stop": None,          # threading.Event set on script_unload
    "_health_thread": None,
    # ---- watchdog backoff state ----
    # Exponential backoff between subprocess restarts to stop the
    # "audio_client crashes immediately → restart 30Hz" log flood.
    # Per-slot since the two slots fail independently. Each entry:
    #   {"next_attempt": float, "delay_s": float, "consecutive": int}
    # `next_attempt` is monotonic seconds from time.time(). After N
    # consecutive immediate exits we cap delay at 60 s and stop
    # bumping it; the subprocess can still recover if the underlying
    # cause clears (device returns, dashboard wakes up, etc).
    "watchdog_backoff": {},
    "watchdog_max_consecutive": 30,  # surface giving-up status after this
}


# ---------------- path discovery ----------------------------------------


def _repo_root() -> str:
    """The OBS script lives at <repo>/obs_scripts/chatterbot_audio_relay.py.
    Walk one directory up to reach the repo root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _scripts_dir() -> str:
    return os.path.join(_repo_root(), "obs_scripts")


def _venv_python() -> str | None:
    """Path to the venv Python that has sounddevice installed.
    Windows: `.venv-win\\Scripts\\python.exe`.
    POSIX:   `.venv/bin/python`. Returns None if neither exists yet."""
    if sys.platform == "win32":
        candidate = os.path.join(_repo_root(), ".venv-win", "Scripts", "python.exe")
    else:
        candidate = os.path.join(_repo_root(), ".venv", "bin", "python")
    return candidate if os.path.exists(candidate) else None


def _bootstrap_cmd() -> list[str]:
    """Command that bootstraps the venv on first use.
    Windows: invoke the .bat (creates .venv-win, installs deps).
    POSIX:   tell the user to run `uv sync --extra whisper`."""
    if sys.platform == "win32":
        return [os.path.join(_scripts_dir(), "audio_client.bat"), "--list-devices-json"]
    return ["uv", "run", "--extra", "whisper", "python",
            os.path.join(_scripts_dir(), "audio_client.py"),
            "--list-devices-json"]


# ---------------- subprocess plumbing -----------------------------------


def _drain_stream(slot: str, stream, label: str) -> None:
    """Read lines from a subprocess pipe and forward them to the OBS
    script log so the audio_client.py output is visible in real time
    (otherwise it sits buffered in PIPE forever and only surfaces if the
    process dies). Runs as a daemon thread; exits when the pipe closes."""
    try:
        for raw in iter(stream.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:
                line = repr(raw)
            if not line:
                continue
            print(f"[{slot}{':err' if label == 'stderr' else ''}] {line}")
    except (ValueError, OSError):
        # ValueError = stream closed mid-read; OSError = broken pipe.
        # Either way, the subprocess has gone away — script_tick will
        # pick up the exit + decide whether to restart.
        pass


def _start_log_drainers(slot: str, proc: subprocess.Popen) -> None:
    """Spin up the two daemon threads that relay stdout/stderr."""
    if proc.stdout is not None:
        t = threading.Thread(
            target=_drain_stream, args=(slot, proc.stdout, "stdout"),
            daemon=True, name=f"chatterbot-drain-{slot}-out",
        )
        t.start()
    if proc.stderr is not None:
        t = threading.Thread(
            target=_drain_stream, args=(slot, proc.stderr, "stderr"),
            daemon=True, name=f"chatterbot-drain-{slot}-err",
        )
        t.start()


def _popen_kwargs() -> dict:
    """Keep subprocesses headless on Windows (no console flash) and
    detached enough that OBS's stdout doesn't get spammed."""
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    return kwargs


def _run_subprocess(cmd: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess | None:
    """Wrap subprocess.run with the OBS-friendly creationflags. Returns
    None on OSError so callers can fall through to a status update."""
    kwargs: dict[str, Any] = {
        "capture_output": True, "text": True, "timeout": timeout,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.run(cmd, **kwargs)
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[chatterbot] subprocess failed: {e}")
        return None


def _refresh_devices() -> None:
    """Run `audio_client.py --list-devices-json` and cache the result.
    Bootstraps the venv on first call. Logs to OBS's script console on
    failure but never raises — the properties UI keeps working."""
    py = _venv_python()
    if py is None:
        # Bootstrap. The .bat / uv invocation produces non-JSON chatter
        # before the JSON itself, so we discard its stdout and re-invoke
        # the now-existing venv directly afterwards. Synchronous: the
        # user clicked "Refresh", a 30-60 s wait is acceptable.
        print("[chatterbot] bootstrapping audio venv (first run, may take a minute)…")
        r = _run_subprocess(_bootstrap_cmd(), timeout=300)
        if r is None or r.returncode != 0:
            err = (r.stderr if r else "").strip()
            print(f"[chatterbot] bootstrap failed:\n{err or '(no stderr)'}")
            _state["last_status"] = "bootstrap failed — check OBS log"
            return
        py = _venv_python()
        if py is None:
            _state["last_status"] = "venv missing after bootstrap (unexpected)"
            return
        print("[chatterbot] bootstrap done")

    r = _run_subprocess(
        [py, os.path.join(_scripts_dir(), "audio_client.py"), "--list-devices-json"],
        timeout=20,
    )
    if r is None or r.returncode != 0:
        err = (r.stderr if r else "").strip()
        print(f"[chatterbot] device list failed:\n{err or '(no stderr)'}")
        _state["last_status"] = "device list failed"
        return
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        print(f"[chatterbot] device list parse error: {e}\nstdout: {r.stdout[:500]}")
        _state["last_status"] = "device list parse error"
        return

    _state["devices"] = {
        "platform": data.get("platform", sys.platform),
        "input_devices": data.get("input_devices", []) or [],
        "loopback_devices": data.get("loopback_devices", []) or [],
    }
    _state["last_status"] = (
        f"{len(_state['devices']['input_devices'])} mic / "
        f"{len(_state['devices']['loopback_devices'])} loopback devices"
    )
    print(f"[chatterbot] {_state['last_status']}")


def _spawn_for_slot(slot: str, device_idx: str, loopback: bool) -> None:
    """Launch one audio_client subprocess for the given slot."""
    py = _venv_python()
    if py is None:
        print(f"[chatterbot] cannot start {slot}: venv not bootstrapped yet")
        return
    cmd = [
        # `-u` forces unbuffered stdout/stderr — without it Python
        # block-buffers when its output is a pipe, and the drainer
        # threads see nothing until a buffer fills (often never, since
        # audio_client.py prints sparingly).
        py, "-u", os.path.join(_scripts_dir(), "audio_client.py"),
        "--device", str(device_idx),
        "--url", _state["url"] or "http://127.0.0.1:8765",
    ]
    if loopback:
        cmd.append("--loopback")
    if _state["auth_user"] and _state["auth_pass"]:
        cmd.extend(["--auth-user", _state["auth_user"],
                    "--auth-pass", _state["auth_pass"]])
    try:
        proc = subprocess.Popen(cmd, **_popen_kwargs())
        _state["procs"][slot] = proc
        _start_log_drainers(slot, proc)
        print(f"[chatterbot] started {slot} (pid={proc.pid}, device={device_idx}{', loopback' if loopback else ''})")
    except OSError as e:
        print(f"[chatterbot] failed to start {slot}: {e}")


def _kill_slot(slot: str) -> None:
    proc = _state["procs"].pop(slot, None)
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    except OSError:
        pass
    print(f"[chatterbot] stopped {slot}")


def _start_all() -> None:
    if _state["running"]:
        return
    # Manual start = clean slate. Wipe any per-slot backoff so we don't
    # honour a 60s cooldown left over from earlier crashes.
    _state["watchdog_backoff"].clear()
    if _state["mic_enabled"] and _state["mic_device"]:
        _watchdog_mark_started("mic")
        _spawn_for_slot("mic", _state["mic_device"], loopback=False)
    if _state["sys_enabled"] and _state["sys_device"]:
        # System slot is loopback on Windows; on other platforms the
        # loopback list is empty so the user simply can't enable it.
        _watchdog_mark_started("system")
        _spawn_for_slot("system", _state["sys_device"],
                        loopback=(sys.platform == "win32"))
    if _state["procs"]:
        _state["running"] = True
        _state["last_status"] = f"capturing — {len(_state['procs'])} source(s)"
    else:
        _state["last_status"] = "no slots enabled — nothing to capture"
    print(f"[chatterbot] start: {_state['last_status']}")


def _stop_all() -> None:
    for slot in list(_state["procs"].keys()):
        _kill_slot(slot)
    _state["running"] = False
    _state["last_status"] = "stopped"
    print("[chatterbot] stop: all slots terminated")


# ---------------- auto-recover health-check loop ------------------------


def _health_check_once() -> bool:
    """Return True if the dashboard's /healthz responds 2xx within a
    short timeout, False otherwise. Conservative: any error counts as
    unreachable (network down, dashboard restart in progress, etc)."""
    import urllib.request, urllib.error
    base = (_state.get("url") or "http://127.0.0.1:8765").rstrip("/")
    headers = {}
    if _state.get("auth_user") and _state.get("auth_pass"):
        import base64
        tok = base64.b64encode(
            f"{_state['auth_user']}:{_state['auth_pass']}".encode("utf-8")
        ).decode("ascii")
        headers["Authorization"] = f"Basic {tok}"
    try:
        req = urllib.request.Request(f"{base}/healthz", headers=headers)
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False
    except Exception:
        return False


def _health_loop(stop_event) -> None:
    """Background thread: poll /healthz every `health_interval_s`. On
    unreachable→reachable transition, AND if we're not currently
    running but at least one slot is enabled AND auto_recover is on,
    set `recover_pending` so script_tick (on the OBS main thread)
    starts capture safely. Calling subprocess.Popen from this thread
    works on POSIX but is iffier inside OBS's Qt event loop — defer
    via the tick flag for portability."""
    last_reachable: bool | None = None
    while not stop_event.is_set():
        reachable = _health_check_once()
        prev = last_reachable
        last_reachable = reachable
        _state["dashboard_reachable"] = reachable

        if (
            _state.get("auto_recover")
            and reachable
            and not _state.get("running")
            and (
                (_state.get("mic_enabled") and _state.get("mic_device"))
                or (_state.get("sys_enabled") and _state.get("sys_device"))
            )
        ):
            # Two trigger conditions:
            #   - First successful check ever (prev is None) — covers
            #     the "OBS started before chatterbot was up" case.
            #   - Transition False → True — covers the "chatterbot
            #     was restarted" case.
            if prev is None or prev is False:
                _state["recover_pending"] = True
                print(
                    "[chatterbot] dashboard reachable — queueing capture "
                    "auto-start (transition: %s → True)" % str(prev)
                )

        # Wait `health_interval_s` but wake up early on unload.
        stop_event.wait(timeout=float(_state.get("health_interval_s", 10)))


def _start_health_thread() -> None:
    """Spin up the health-check thread. Idempotent — if a thread is
    already running we leave it alone (script reloads share state)."""
    if _state.get("_health_thread") and _state["_health_thread"].is_alive():
        return
    stop_event = threading.Event()
    t = threading.Thread(
        target=_health_loop, args=(stop_event,),
        daemon=True, name="chatterbot-health-check",
    )
    _state["_health_stop"] = stop_event
    _state["_health_thread"] = t
    t.start()


def _stop_health_thread() -> None:
    stop_event = _state.get("_health_stop")
    if stop_event is not None:
        stop_event.set()
    t = _state.get("_health_thread")
    if t is not None and t.is_alive():
        t.join(timeout=2)
    _state["_health_stop"] = None
    _state["_health_thread"] = None


# ---------------- OBS script callbacks ----------------------------------


def script_description() -> str:
    return (
        "<b>chatterbot audio relay</b><br>"
        "Captures audio from your mic and (on Windows) any output device "
        "via WASAPI loopback — letting chatterbot transcribe browser, "
        "game, and system audio alongside your mic. Spawns "
        "<code>audio_client.py</code> subprocesses; manage start/stop "
        "from the buttons below."
    )


def script_defaults(settings) -> None:
    obs.obs_data_set_default_string(settings, "url", "http://127.0.0.1:8765")
    obs.obs_data_set_default_bool(settings, "mic_enabled", False)
    obs.obs_data_set_default_bool(settings, "sys_enabled", False)
    # Auto-recover defaults to ON since the streamer's complaint is
    # exactly this workflow — chatterbot restart wipes capture.
    obs.obs_data_set_default_bool(settings, "auto_recover", True)


def script_load(settings) -> None:
    # Pull saved settings up front so script_load can decide whether
    # to auto-start. script_update runs AFTER script_load on initial
    # load, so without this the auto-start branch sees the bare
    # _state defaults and skips.
    if settings is not None:
        try:
            script_update(settings)
        except Exception:
            pass

    # Don't bootstrap-on-load: pip install can take 30+ seconds and would
    # hang the OBS UI. If the venv already exists, do a quick device list
    # so the dropdowns are populated. Otherwise the user clicks "Refresh
    # devices" to trigger bootstrap explicitly (status text explains).
    if _venv_python() is not None:
        print("[chatterbot] script loaded — refreshing device list")
        _refresh_devices()
    else:
        _state["last_status"] = (
            "venv not found — click Refresh devices to bootstrap "
            "(first run installs sounddevice)"
        )
        print(f"[chatterbot] {_state['last_status']}")

    # Auto-start capture if the streamer left it enabled. Two cases
    # this covers:
    #   1. OBS itself was restarted (whole machine, scene-collection
    #      change, etc.) — auto_recover keeps capture going.
    #   2. The chatterbot dashboard was restarted while OBS stayed up.
    #      Audio_client subprocesses are robust to this on their own
    #      (Sender thread retries forever) so usually no action is
    #      needed; but if capture was manually stopped and the user
    #      forgot to restart it, the health thread will catch the
    #      dashboard becoming reachable and trigger auto-start.
    if (
        _state.get("auto_recover")
        and (
            (_state.get("mic_enabled") and _state.get("mic_device"))
            or (_state.get("sys_enabled") and _state.get("sys_device"))
        )
    ):
        print("[chatterbot] auto_recover on — starting capture on script_load")
        _start_all()

    # Health-check thread runs regardless of auto_recover so the
    # status pane shows whether the dashboard is reachable. Auto-recover
    # gates the recovery action, not the polling.
    _start_health_thread()


def script_unload() -> None:
    _stop_health_thread()
    _stop_all()


def script_update(settings) -> None:
    _state["url"] = obs.obs_data_get_string(settings, "url") or "http://127.0.0.1:8765"
    _state["auth_user"] = obs.obs_data_get_string(settings, "auth_user") or ""
    _state["auth_pass"] = obs.obs_data_get_string(settings, "auth_pass") or ""
    _state["mic_enabled"] = obs.obs_data_get_bool(settings, "mic_enabled")
    _state["mic_device"] = obs.obs_data_get_string(settings, "mic_device") or ""
    _state["sys_enabled"] = obs.obs_data_get_bool(settings, "sys_enabled")
    _state["sys_device"] = obs.obs_data_get_string(settings, "sys_device") or ""
    _state["auto_recover"] = obs.obs_data_get_bool(settings, "auto_recover")


def _on_refresh_clicked(_props, _prop) -> bool:
    _refresh_devices()
    # Returning True tells OBS to refresh the property pane so the
    # repopulated dropdowns show up immediately.
    return True


def _on_start_clicked(_props, _prop) -> bool:
    _start_all()
    return True


def _on_stop_clicked(_props, _prop) -> bool:
    _stop_all()
    return True


def _populate_device_list(prop, devices: list[dict]) -> None:
    obs.obs_property_list_clear(prop)
    if not devices:
        obs.obs_property_list_add_string(prop, "(no devices found — click Refresh)", "")
        return
    for d in devices:
        label = f"[{d['index']}] {d['name']}"
        if d.get("api"):
            label += f" · {d['api']}"
        if d.get("default"):
            label += " (default)"
        obs.obs_property_list_add_string(prop, label, str(d["index"]))


def script_properties():
    props = obs.obs_properties_create()

    obs.obs_properties_add_text(
        props, "url", "Dashboard URL", obs.OBS_TEXT_DEFAULT,
    )
    obs.obs_properties_add_text(
        props, "auth_user", "Basic auth user (optional)", obs.OBS_TEXT_DEFAULT,
    )
    obs.obs_properties_add_text(
        props, "auth_pass", "Basic auth pass (optional)", obs.OBS_TEXT_PASSWORD,
    )

    obs.obs_properties_add_button(
        props, "refresh_btn", "🔄  Refresh devices", _on_refresh_clicked,
    )

    # Auto-recover toggle. When on:
    #   1. Capture auto-starts when OBS loads / reloads this script.
    #   2. A background thread polls the dashboard's /healthz; if it
    #      transitions from unreachable→reachable while we're stopped
    #      but slots are configured, capture restarts itself.
    auto = obs.obs_properties_add_bool(
        props, "auto_recover",
        "Auto-recover capture (start on load + restart when "
        "dashboard returns)",
    )
    obs.obs_property_set_long_description(
        auto,
        "Recommended on. When checked, capture restarts itself "
        "automatically if you restart chatterbot, restart OBS, or "
        "the dashboard temporarily disconnects. The audio_client "
        "subprocesses already retry POSTs forever once running, "
        "so this primarily covers the 'capture got stopped and I "
        "forgot to restart it' case."
    )

    # --- mic slot ---------------------------------------------------------
    obs.obs_properties_add_bool(props, "mic_enabled", "Capture from mic")
    mic_list = obs.obs_properties_add_list(
        props, "mic_device", "  └ Mic device",
        obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_STRING,
    )
    _populate_device_list(mic_list, _state["devices"]["input_devices"])

    # --- system slot ------------------------------------------------------
    sys_label = (
        "Capture from system / window audio (WASAPI loopback)"
        if sys.platform == "win32"
        else "Capture from system audio  ·  loopback unsupported on this platform"
    )
    obs.obs_properties_add_bool(props, "sys_enabled", sys_label)
    sys_list = obs.obs_properties_add_list(
        props, "sys_device", "  └ Output device",
        obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_STRING,
    )
    _populate_device_list(sys_list, _state["devices"]["loopback_devices"])

    # --- start / stop -----------------------------------------------------
    obs.obs_properties_add_button(props, "start_btn", "▶  Start capture", _on_start_clicked)
    obs.obs_properties_add_button(props, "stop_btn",  "■  Stop capture",  _on_stop_clicked)

    # Status line — a read-only text input with the current state.
    reach = _state.get("dashboard_reachable")
    reach_label = (
        "checking…" if reach is None else
        "✅ reachable" if reach else
        "❌ unreachable"
    )
    status = obs.obs_properties_add_text(
        props, "_status", "Status", obs.OBS_TEXT_INFO,
    )
    obs.obs_property_set_long_description(
        status,
        f"Platform: {_state['devices']['platform']}\n"
        f"Last status: {_state['last_status']}\n"
        f"Active subprocesses: {', '.join(_state['procs'].keys()) or 'none'}\n"
        f"Dashboard ({_state.get('url', '')}): {reach_label}\n"
        f"Auto-recover: {'on' if _state.get('auto_recover') else 'off'}",
    )

    return props


def _watchdog_can_restart(slot: str) -> bool:
    """Honour the per-slot exponential-backoff schedule. Returns True
    when enough time has elapsed since the last failed restart attempt
    that the watchdog may try again."""
    import time
    rec = _state["watchdog_backoff"].get(slot)
    if rec is None:
        return True
    return time.time() >= rec.get("next_attempt", 0.0)


def _watchdog_record_exit(slot: str) -> float:
    """Bookkeeping after a subprocess exits. If it ran a long time
    ('we got useful audio out of it') reset the backoff; if it died
    quickly, bump the delay exponentially. Returns the new delay so
    the caller can log it."""
    import time
    now = time.time()
    rec = _state["watchdog_backoff"].setdefault(
        slot, {"next_attempt": 0.0, "delay_s": 0.0, "consecutive": 0,
               "started_at": now},
    )
    runtime = now - rec.get("started_at", now)
    if runtime >= 30.0:
        # Subprocess was healthy; reset.
        rec["delay_s"] = 0.0
        rec["consecutive"] = 0
    else:
        # Crashed fast — exponential backoff capped at 60s.
        rec["consecutive"] = int(rec.get("consecutive", 0)) + 1
        prev = float(rec.get("delay_s", 0.0))
        rec["delay_s"] = min(60.0, max(2.0, prev * 2 if prev > 0 else 2.0))
    rec["next_attempt"] = now + rec["delay_s"]
    return rec["delay_s"]


def _watchdog_mark_started(slot: str) -> None:
    """Stamp `started_at` so a long-running subprocess can reset
    the backoff when it finally exits."""
    import time
    rec = _state["watchdog_backoff"].setdefault(
        slot, {"next_attempt": 0.0, "delay_s": 0.0, "consecutive": 0},
    )
    rec["started_at"] = time.time()


def script_tick(_seconds: float) -> None:
    """Watchdog — restart subprocesses that have died unexpectedly,
    and act on auto-recover requests from the health thread. Ticks
    happen ~30x/second; we early-out aggressively to keep this cheap."""
    # Auto-recover trigger from the health thread. Performed on the
    # OBS main thread so subprocess.Popen behaves consistently across
    # platforms (Qt event loop quirks on Windows).
    if _state.get("recover_pending"):
        _state["recover_pending"] = False
        if (
            _state.get("auto_recover")
            and not _state.get("running")
            and (
                (_state.get("mic_enabled") and _state.get("mic_device"))
                or (_state.get("sys_enabled") and _state.get("sys_device"))
            )
        ):
            print("[chatterbot] auto-recover: dashboard back, starting capture")
            _start_all()

    if not _state["running"] or not _state["procs"]:
        return
    for slot in list(_state["procs"].keys()):
        proc = _state["procs"][slot]
        rc = proc.poll()
        if rc is None:
            continue
        # The drainer threads have already relayed stderr to the OBS
        # log line-by-line; we just need to announce the exit code.
        delay = _watchdog_record_exit(slot)
        rec = _state["watchdog_backoff"].get(slot, {})
        consecutive = int(rec.get("consecutive", 0))
        max_cons = int(_state.get("watchdog_max_consecutive", 30))
        if consecutive >= max_cons:
            print(
                f"[chatterbot] {slot} subprocess exited rc={rc} "
                f"(consecutive={consecutive} ≥ cap, giving up — "
                f"check the device + Refresh devices to retry)"
            )
            # Drop from procs but DON'T respawn. Manual Start will
            # reset bookkeeping; auto-recover will too on the next
            # dashboard-reachable transition.
            del _state["procs"][slot]
            _state["last_status"] = (
                f"{slot} stopped after {consecutive} consecutive "
                f"crashes — fix underlying issue, then click Start"
            )
            continue
        if delay > 0:
            print(
                f"[chatterbot] {slot} subprocess exited rc={rc} "
                f"(consecutive={consecutive}, restarting in {delay:.0f}s)"
            )
        else:
            print(f"[chatterbot] {slot} subprocess exited rc={rc} (restarting)")
        del _state["procs"][slot]
        # Don't restart immediately — let the backoff clock run. The
        # next tick will see the slot missing AND running=True and
        # respawn after `_watchdog_can_restart(slot)` returns True.

    # Re-spawn any expected-but-missing slot whose backoff cooldown
    # has elapsed. This unifies the "just died" and "delayed restart"
    # cases — script_tick is the single restart entry point.
    if _state["running"]:
        if (
            _state["mic_enabled"] and _state["mic_device"]
            and "mic" not in _state["procs"]
            and _watchdog_can_restart("mic")
        ):
            _watchdog_mark_started("mic")
            _spawn_for_slot("mic", _state["mic_device"], loopback=False)
        if (
            _state["sys_enabled"] and _state["sys_device"]
            and "system" not in _state["procs"]
            and _watchdog_can_restart("system")
        ):
            _watchdog_mark_started("system")
            _spawn_for_slot("system", _state["sys_device"],
                            loopback=(sys.platform == "win32"))
