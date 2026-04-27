"""chatterbot audio client — captures from a system audio input device and
POSTs PCM chunks to chatterbot's `/audio/ingest` endpoint.

Replaces the original OBS-script approach. obspython's SWIG bindings
can't marshal a Python callable into `obs_source_audio_capture_t`
(the C callback type for `obs_source_add_audio_capture_callback`), so
hooking OBS's audio pipeline directly from a Python script isn't
feasible. This standalone script uses sounddevice (PortAudio) instead
to capture from any system audio input device — your mic, a Stereo
Mix loopback, a virtual cable, etc.

Runs alongside OBS / the dashboard. On Windows / macOS / Linux.

Setup
-----
1. Install the deps (one-time):

       uv sync --extra whisper

   (`sounddevice` is bundled in the whisper extra.)

2. List available input devices:

       uv run python obs_scripts/audio_client.py --list-devices

3. Start the client, picking your mic:

       uv run python obs_scripts/audio_client.py --device "Microphone"
       # or by index:
       uv run python obs_scripts/audio_client.py --device 1

   The default URL is `http://127.0.0.1:8765` — change with `--url`.

The client captures at 16 kHz mono float32 (whisper's required input
format) so no server-side resampling. PCM chunks are 1 second each,
sent over plain HTTP POST. Stays on localhost by default.

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import base64
import queue
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

try:
    import sounddevice as sd
except ImportError:
    print(
        "sounddevice is not installed. Run `uv sync --extra whisper` "
        "from the chatterbot repo root, then re-run this client.",
        file=sys.stderr,
    )
    sys.exit(2)
except OSError as e:
    # PortAudio is bundled with the wheel on Windows / macOS but Linux
    # needs the system package (libportaudio2 on Debian/Ubuntu,
    # portaudio on Arch/Fedora). This client is meant to run on the
    # streamer's host machine — typically Windows alongside OBS — so
    # most users won't hit this.
    print(
        f"sounddevice failed to load: {e}\n"
        "On Linux, install PortAudio:\n"
        "  Debian/Ubuntu: sudo apt install libportaudio2\n"
        "  Arch:          sudo pacman -S portaudio\n"
        "  Fedora:        sudo dnf install portaudio\n"
        "On Windows / macOS this should work out of the box — "
        "if you're seeing this there, something's wrong with the install.",
        file=sys.stderr,
    )
    sys.exit(2)

import numpy as np


TARGET_SR = 16000   # whisper's input rate
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SECONDS)


# ---------------- worker thread ------------------------------------------


class Sender(threading.Thread):
    """Drains a queue of float32 PCM chunks, POSTs each to /audio/ingest.
    Failures are logged once per error type to avoid spamming."""

    def __init__(
        self, url: str, queue_: queue.Queue,
        auth_user: str | None, auth_pass: str | None,
    ):
        super().__init__(daemon=True, name="chatterbot-audio-sender")
        self.url = url.rstrip("/") + "/audio/ingest"
        self.q = queue_
        self.auth_user = auth_user
        self.auth_pass = auth_pass
        self._stop = threading.Event()
        self._last_err: str | None = None
        self._sent = 0

    def stop(self):
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass

    def run(self):
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(TARGET_SR),
        }
        if self.auth_user and self.auth_pass:
            tok = base64.b64encode(
                f"{self.auth_user}:{self.auth_pass}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {tok}"
        while not self._stop.is_set():
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if payload is None:
                break
            req = urllib.request.Request(
                self.url, data=payload, method="POST", headers=headers,
            )
            try:
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    resp.read()
                self._sent += 1
                if self._sent in (1, 5, 30):
                    print(
                        f"[audio_client] sent chunk #{self._sent} "
                        f"to {self.url}"
                    )
                self._last_err = None
            except urllib.error.URLError as e:
                err = f"URLError: {e}"
                if err != self._last_err:
                    print(f"[audio_client] POST failed: {err}", file=sys.stderr)
                    self._last_err = err
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if err != self._last_err:
                    print(f"[audio_client] worker error: {err}", file=sys.stderr)
                    self._last_err = err


# ---------------- audio capture ------------------------------------------


def _resolve_device(spec: str | None) -> int | None:
    """Resolve a --device flag into a sounddevice device index. Accepts
    None (use default), an integer string, or a substring of the device
    name. Returns None to use the default; raises SystemExit on no match."""
    if spec is None or spec == "":
        return None
    try:
        return int(spec)
    except ValueError:
        pass
    needle = spec.lower()
    matches = []
    for idx, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) <= 0:
            continue
        if needle in info["name"].lower():
            matches.append((idx, info["name"]))
    if not matches:
        print(f"[audio_client] no input device matches {spec!r}", file=sys.stderr)
        print("  available input devices:", file=sys.stderr)
        for idx, info in enumerate(sd.query_devices()):
            if info.get("max_input_channels", 0) > 0:
                print(f"    [{idx}] {info['name']}", file=sys.stderr)
        raise SystemExit(2)
    if len(matches) > 1:
        # First match wins, but warn so the streamer can disambiguate
        # if needed.
        print(
            f"[audio_client] multiple devices match {spec!r}, picking first:",
            file=sys.stderr,
        )
        for idx, name in matches:
            print(f"    [{idx}] {name}", file=sys.stderr)
    return matches[0][0]


def _list_devices():
    print("Available audio input devices:")
    for idx, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) <= 0:
            continue
        default = " (default)" if idx == sd.default.device[0] else ""
        print(
            f"  [{idx}] {info['name']}{default} "
            f"— {info['max_input_channels']} ch, "
            f"{int(info['default_samplerate'])} Hz"
        )


def _list_devices_json():
    """Machine-readable device listing — input devices for mic capture
    and WASAPI output devices for loopback (Windows browser/window audio).

    Output format:

        {
          "platform": "win32" | "linux" | "darwin",
          "input_devices":  [{"index": N, "name": "...", "api": "MME",
                              "channels": K, "sr": 48000, "default": bool}, ...],
          "loopback_devices": [{"index": N, "name": "Speakers (Realtek)",
                                "api": "Windows WASAPI", "channels": 2,
                                "sr": 48000, "default": bool}, ...]
        }

    `loopback_devices` is empty on non-Windows platforms (PortAudio
    loopback support is Windows-only via WASAPI). The OBS coordinator
    consumes this to populate its device dropdowns.
    """
    import json
    import sys as _sys

    apis = sd.query_hostapis()
    api_name_for = {i: a.get("name", "?") for i, a in enumerate(apis)}
    wasapi_idx = next(
        (i for i, a in enumerate(apis) if "WASAPI" in a.get("name", "")),
        None,
    )

    default_in = sd.default.device[0] if sd.default.device else -1
    default_out = sd.default.device[1] if sd.default.device else -1

    inputs: list[dict] = []
    loopback: list[dict] = []

    for idx, info in enumerate(sd.query_devices()):
        host_api_idx = info.get("hostapi", 0)
        api_name = api_name_for.get(host_api_idx, "?")
        common = {
            "index": idx,
            "name": info["name"],
            "api": api_name,
            "channels_in": int(info.get("max_input_channels", 0)),
            "channels_out": int(info.get("max_output_channels", 0)),
            "sr": int(info.get("default_samplerate") or 48000),
        }
        if common["channels_in"] > 0:
            entry = dict(common)
            entry["channels"] = common["channels_in"]
            entry["default"] = (idx == default_in)
            inputs.append(entry)
        # WASAPI loopback exposes any output device as a capture source.
        # PortAudio doesn't expose loopback for other host APIs, so skip
        # them — would just confuse the dropdown.
        if (
            wasapi_idx is not None
            and host_api_idx == wasapi_idx
            and common["channels_out"] > 0
        ):
            entry = dict(common)
            entry["channels"] = common["channels_out"]
            entry["default"] = (idx == default_out)
            loopback.append(entry)

    out = {
        "platform": _sys.platform,
        "input_devices": inputs,
        "loopback_devices": loopback,
    }
    print(json.dumps(out, indent=2))


def _ensure_soundcard():
    """Lazy-import soundcard; pip-install it if missing. We don't bundle
    it in the .bat's initial REQS for backward compatibility with users
    whose .venv-win was created before loopback existed — auto-installing
    on first --loopback invocation keeps that path self-healing."""
    try:
        import soundcard as sc  # noqa: F401
        return sc
    except ImportError:
        print(
            "[audio_client] soundcard not installed; pip-installing now…",
            file=sys.stderr,
        )
        import subprocess as _sp
        _sp.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "soundcard"],
            stdout=_sp.PIPE, stderr=_sp.PIPE,
        )
        import soundcard as sc  # noqa: F811
        return sc


def _run_loopback(args):
    """Capture audio playing on an OUTPUT device (browser, game, system
    sounds) via WASAPI loopback. Uses the `soundcard` library because
    sounddevice / PortAudio doesn't expose loopback flags. Windows-only
    in practice — soundcard supports loopback on Windows; on macOS it's
    limited and on Linux it depends on PulseAudio routing.

    The OBS coordinator passes a sounddevice device index (since that's
    what the JSON listing emits). We look up that device's name via
    sounddevice and find the matching loopback mic in soundcard's list,
    so the dropdown UX in OBS remains unified.
    """
    if sys.platform != "win32":
        print(
            "[audio_client] --loopback is only reliably supported on "
            "Windows. Falling back to standard input capture.",
            file=sys.stderr,
        )
        return _run(args)

    sc = _ensure_soundcard()

    # Resolve the target device name from the sounddevice index the
    # coordinator passed. soundcard mics are matched by substring.
    target_name: str | None = None
    if args.device not in (None, ""):
        try:
            sd_idx = int(args.device)
            target_name = str(sd.query_devices(sd_idx)["name"])
        except (ValueError, sd.PortAudioError):
            target_name = str(args.device)

    mics = sc.all_microphones(include_loopback=True)
    loopback_mic = None
    if target_name:
        needle = target_name.lower()
        loopback_mic = next(
            (m for m in mics if needle in m.name.lower()), None,
        )
    if loopback_mic is None:
        # Fall back to default speaker's loopback so the user gets *some*
        # audio rather than a hard error; warn so it's debuggable.
        speaker = sc.default_speaker()
        loopback_mic = sc.get_microphone(str(speaker.name), include_loopback=True)
        print(
            f"[audio_client] no loopback mic matched {target_name!r}; "
            f"using default speaker loopback ({speaker.name!r}). "
            f"Available loopback mics:",
            file=sys.stderr,
        )
        for m in mics:
            if getattr(m, "isloopback", False):
                print(f"    {m.name!r}", file=sys.stderr)

    print(f"[audio_client] using device: {loopback_mic.name} (loopback)")
    print(f"[audio_client] capturing at {TARGET_SR} Hz")
    print(f"[audio_client] target: {args.url}")

    q: queue.Queue = queue.Queue(maxsize=64)
    sender = Sender(args.url, q, args.auth_user, args.auth_pass)
    sender.start()

    # Block size for soundcard.recorder — 100 ms gives smooth stream.
    block_frames = int(TARGET_SR * 0.1)
    buf = np.empty(0, dtype=np.float32)
    try:
        # soundcard's recorder is a context manager; samplerate is set
        # at the API level so it does its own resampling. channels=2
        # so we can mix to mono ourselves (more consistent than letting
        # the library pick).
        with loopback_mic.recorder(
            samplerate=TARGET_SR, channels=2, blocksize=block_frames,
        ) as rec:
            print("[audio_client] capturing — Ctrl+C to stop")
            while True:
                data = rec.record(numframes=block_frames)
                if data.ndim == 2 and data.shape[1] > 1:
                    mono = data.mean(axis=1).astype(np.float32, copy=False)
                else:
                    mono = data.reshape(-1).astype(np.float32, copy=False)
                buf = np.concatenate([buf, mono]) if buf.shape[0] else mono.copy()
                while buf.shape[0] >= CHUNK_SAMPLES:
                    chunk = buf[:CHUNK_SAMPLES]
                    buf = buf[CHUNK_SAMPLES:]
                    try:
                        q.put_nowait(chunk.tobytes())
                    except queue.Full:
                        try:
                            q.get_nowait()
                            q.put_nowait(chunk.tobytes())
                        except queue.Empty:
                            pass
    except KeyboardInterrupt:
        print("\n[audio_client] stopping")
    finally:
        sender.stop()
        sender.join(timeout=2)


def _run(args):
    device = _resolve_device(args.device)
    info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
    # Capture at the device's native sample rate, then resample to
    # 16 kHz in Python before sending. PortAudio's per-API resampling
    # quality varies; doing it ourselves with numpy is consistent
    # cross-platform and good enough for whisper (which runs its own
    # mel-spec conversion downstream).
    native_sr = int(info.get("default_samplerate") or 48000)
    cap_channels = max(1, min(2, int(info.get("max_input_channels", 1))))
    print(f"[audio_client] using device: {info['name']} (input)")
    print(f"[audio_client] capturing at {native_sr} Hz, resampling to {TARGET_SR} Hz")
    print(f"[audio_client] target: {args.url}")

    q: queue.Queue = queue.Queue(maxsize=64)
    sender = Sender(args.url, q, args.auth_user, args.auth_pass)
    sender.start()

    # Buffer of post-resample 16 kHz mono audio.
    buf = np.empty(0, dtype=np.float32)
    # Resample ratio. >1.0 means downsampling.
    sr_ratio = native_sr / TARGET_SR

    def callback(indata, frames, _time_info, status):
        nonlocal buf
        if status:
            # Buffer overflow / underrun on the device side. Print but
            # keep going; we'd rather drop a frame than abort the stream.
            print(f"[audio_client] sounddevice status: {status}", file=sys.stderr)
        # Mix to mono.
        if indata.shape[1] > 1:
            mono = indata.mean(axis=1).astype(np.float32, copy=False)
        else:
            mono = indata[:, 0].astype(np.float32, copy=False)
        # Linear-interp downsample to 16 kHz. Whisper does its own
        # bandlimit filtering so a cheap resampler is fine.
        if native_sr != TARGET_SR:
            n_out = int(mono.shape[0] / sr_ratio)
            if n_out <= 0:
                return
            idx = (np.arange(n_out) * sr_ratio).astype(np.int64)
            idx = np.clip(idx, 0, mono.shape[0] - 1)
            mono = mono[idx]
        buf = np.concatenate([buf, mono]) if buf.shape[0] else mono.copy()
        # Flush full 1-second chunks while the buffer is large enough.
        while buf.shape[0] >= CHUNK_SAMPLES:
            chunk = buf[:CHUNK_SAMPLES]
            buf = buf[CHUNK_SAMPLES:]
            try:
                q.put_nowait(chunk.tobytes())
            except queue.Full:
                # Drop oldest if backed up — keep the audio stream alive.
                try:
                    q.get_nowait()
                    q.put_nowait(chunk.tobytes())
                except queue.Empty:
                    pass

    try:
        with sd.InputStream(
            samplerate=native_sr,
            channels=cap_channels,
            device=device,
            dtype="float32",
            callback=callback,
            blocksize=int(native_sr * 0.1),  # 100 ms blocks from device
        ):
            print("[audio_client] capturing — Ctrl+C to stop")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[audio_client] stopping")
    finally:
        sender.stop()
        sender.join(timeout=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url", default="http://127.0.0.1:8765",
        help="chatterbot dashboard URL (default: http://127.0.0.1:8765)",
    )
    p.add_argument(
        "--device", default=None,
        help="audio input device — index, or substring of the name. "
             "Use --list-devices to see options. Default: system default input.",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="print available input devices and exit",
    )
    p.add_argument(
        "--list-devices-json", action="store_true",
        help="print devices as JSON (machine-readable, for the OBS coordinator)",
    )
    p.add_argument(
        "--loopback", action="store_true",
        help="open the device as a WASAPI loopback (Windows only) — "
             "captures whatever audio the OUTPUT device is playing "
             "(browsers, games, system sounds). Pass --device pointing "
             "to a WASAPI output device.",
    )
    p.add_argument(
        "--auth-user", default=None,
        help="basic auth user — only set if your dashboard has DASHBOARD_BASIC_AUTH_USER",
    )
    p.add_argument(
        "--auth-pass", default=None,
        help="basic auth password",
    )
    args = p.parse_args()
    if args.list_devices:
        _list_devices()
        return 0
    if args.list_devices_json:
        _list_devices_json()
        return 0
    if args.loopback:
        return _run_loopback(args) or 0
    return _run(args) or 0


if __name__ == "__main__":
    sys.exit(main() or 0)
