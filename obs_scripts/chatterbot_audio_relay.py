"""DEPRECATED — this OBS script approach doesn't work.

obspython's SWIG bindings can't marshal a Python callable into the C
callback type `obs_source_audio_capture_t` that
`obs_source_add_audio_capture_callback` requires. Attempting to attach
raises:

    TypeError: in method 'obs_source_add_audio_capture_callback',
        argument 2 of type 'obs_source_audio_capture_t'

There's no workaround inside a Python script — the binding gap is
structural. Use the standalone client instead:

    obs_scripts/audio_client.py

It captures from any system audio input device (your mic, a virtual
cable, a Stereo Mix loopback) via sounddevice and POSTs the same PCM
chunks to chatterbot's `/audio/ingest` endpoint. Same dashboard
pipeline, different audio plumbing.

If you have this script enabled in OBS, remove it from Tools → Scripts
and run the standalone client alongside OBS instead.
"""

import obspython as obs


def script_description():
    return (
        "<b>chatterbot audio relay (deprecated)</b><br>"
        "OBS Python can't expose audio frames to scripts due to a SWIG "
        "binding limitation. Use <code>obs_scripts/audio_client.py</code> "
        "instead — a standalone Python client that captures from any "
        "system audio input device and feeds chatterbot's whisper "
        "pipeline. Remove this script from OBS to clear the warning."
    )


def script_load(_settings):
    print(
        "[chatterbot_audio_relay] DEPRECATED — this OBS-script approach "
        "doesn't work. Run obs_scripts/audio_client.py alongside OBS instead."
    )


def script_unload():
    pass
