# Personal training dataset

An opt-in encrypted log of every LLM call + streamer dashboard action.
The captured stream is the raw material for fine-tuning a
streamer-personalised model later — assistant, moderation classifier,
ASR, whatever the streamer wants to train.

**Off by default.** Streamers who don't run setup see zero behaviour
change. The optional dependency (`uv sync --extra dataset`) is required
only when capture is enabled.

## Contents

- [Why](#why)
- [Threat model + crypto](#threat-model--crypto)
- [Quickstart](#quickstart)
- [What gets captured](#what-gets-captured)
- [Storage shape](#storage-shape)
- [Retention + compaction](#retention--compaction)
- [Export](#export)
- [Redaction](#redaction)
- [Dashboard surface](#dashboard-surface)

---

## Why

LLM calls in chatterbot already do useful work — they extract notes,
classify subjects, generate talking points, summarise transcripts. The
streamer's reactions to that work (dismiss / addressed / corrected /
rejected) are implicit human-validation labels. Together they form a
preference dataset most teams pay to collect.

Capture turns this into something exportable:

- Every `generate_structured(...)` call → one `LLM_CALL` event with
  prompt + response + model_id + latency + call_site
- Every dashboard mutation (insight state change, note CRUD,
  engaging-subject reject) → one `STREAMER_ACTION` event
- Periodic `CONTEXT_SNAPSHOT` events bundle recent messages +
  transcripts + active threads so the bundle is self-contained for
  fine-tuning

## Threat model + crypto

Defended:

- **Casual disk theft / cloud-backup leak** — DEK never on disk in
  plaintext.
- **Streamer screenshots their data dir on stream** — file is
  ciphertext + metadata; reveals only file size + timestamp.
- **Cleanup of one chatter's data** — opt-out at capture time, not
  export time, so opted-out chatters never enter the encrypted store
  in the first place.

Out of scope:

- Live process memory dump (DEK is in process RAM while capture is
  on; inherent to local encryption).
- Streamer forgets passphrase + loses recovery string (data is
  permanently unrecoverable — encryption working as intended).

### Two-tier envelope

```
passphrase ─Argon2id─► KEK ─AES-GCM-wrap─► DEK ─AES-GCM─► event blob
```

- **DEK** (32 bytes, random) generated once at `chatterbot dataset
  setup`. Used directly for AES-GCM on every event blob.
- **KEK** derived from the streamer's passphrase via Argon2id with
  per-install salt + cost params. Wraps the DEK with AES-GCM.

Implementation:
[`dataset/cipher.py`](../src/chatterbot/dataset/cipher.py).

Each event uses a fresh 12-byte nonce and binds the row's timestamp
into AES-GCM's associated-data field — an attacker can't reorder
rows by swapping ciphertexts between timestamps without the decrypt
failing.

## Quickstart

```bash
# 1. install the optional extra
uv sync --extra dataset

# 2. one-time setup — generate + wrap a fresh DEK with a passphrase
chatterbot dataset setup
# (prints a base32 recovery string; save it offline)

# 3. enable capture
chatterbot dataset enable

# 4. unlock the DEK at process startup via env var
export CHATTERBOT_DATASET_PASSPHRASE='your-passphrase'
chatterbot bot &
chatterbot dashboard &

# ...later, after some streams:

# 5. read-only status (never decrypts)
chatterbot dataset info

# 6. export an encrypted bundle
chatterbot dataset export --out my-bundle.cbds

# 7. verify the bundle decrypts cleanly + matches its manifest
chatterbot dataset verify my-bundle.cbds
```

See [`dataset/cli.py`](../src/chatterbot/dataset/cli.py) for the full
CLI surface.

## What gets captured

Three event kinds:

### `LLM_CALL`

Fired at the LLM provider layer (Ollama / Anthropic / OpenAI). Wraps
every `generate_structured(...)` call.

```python
# Provider's generate_structured wrapper (pseudocode):
try:
    raw = await self.generate(...)
    return response_model.model_validate_json(raw)
finally:
    await record_llm_call_safe(
        self._dataset_repo,
        call_site=call_site,
        model_id=...,
        provider="ollama",
        prompt=prompt,
        response_text=raw,
        ...
    )
```

Real implementation:
[`llm/ollama_client.py:generate_structured`](../src/chatterbot/llm/ollama_client.py#L166-L237)
(plus equivalents in
[`llm/providers.py`](../src/chatterbot/llm/providers.py) for Anthropic
+ OpenAI).

Failures are recorded too — `error="ValidationError: ..."` is a
useful negative signal for prompt iteration.

### `STREAMER_ACTION`

Fired at the repo / insights mutation chokepoints when the streamer
takes an action that's a human-validation signal:

- `repo.set_insight_state` — dismiss / addressed / snooze / pin
- `repo.add_note` / `update_note` / `delete_note` — note CRUD
- `insights.reject_subject` — engaging-subject negative supervision
- `insights.clear_subject_blocklist` — regime-reset signal

Each event records `(action_kind, item_key, action, note)` so a
downstream consumer can pair the action with the LLM call that
produced the underlying insight.

### `CONTEXT_SNAPSHOT`

Periodic — every ~5 min — snapshot of the recent message window +
recent transcripts + active threads + channel context. Lets future
fine-tune bundles be self-contained without needing the full
`chatters.db` attached.

Background loop in
[`dataset/loops.py:context_snapshot_loop`](../src/chatterbot/dataset/loops.py#L151-L181).

## Storage shape

Hybrid index + shards. The `dataset_events` table in `chatters.db`
holds one row per encrypted record:

```sql
CREATE TABLE dataset_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    event_kind     TEXT NOT NULL,
    shard_path     TEXT NOT NULL,
    byte_offset    INTEGER NOT NULL,
    byte_length    INTEGER NOT NULL,
    schema_version INTEGER NOT NULL
);
```

Encrypted ciphertext lives in `data/dataset/shards/YYYY-MM-DD__NNNN.cbds.bin`
files — append-only, length-prefixed records. Shards roll at 50 MiB or
24h. Reads are one `pread` + decrypt per event.

Implementation:
[`dataset/storage.py`](../src/chatterbot/dataset/storage.py).

### Hot-path opt-in gate

When capture is OFF, `record_llm_call_safe` is a single attribute
read on the repo:

```python
# Pseudocode for the hot-path gate:
def record_llm_call_safe(repo, ...):
    if repo is None:
        return                              # no capture wired
    if not repo.dataset_capture_enabled():
        return                              # toggle off
    if repo.dataset_dek() is None:
        warn_once_about_missing_passphrase()
        return                              # no DEK loaded
    # ...actually write the event
```

Real implementation:
[`dataset/capture.py:record_llm_call_safe`](../src/chatterbot/dataset/capture.py#L202-L264)
(the safe wrapper) +
[`record_llm_call`](../src/chatterbot/dataset/capture.py#L267-L356)
(the encrypt-and-write path).

### Schema version

Every row stamps the schema version code knew about at write time
(`CAPTURE_SCHEMA_VERSION`). Reader code dispatches on this; new
fields with sensible defaults don't need a bump (forward-additive).

## Retention + compaction

Two app_settings knobs:

- `dataset_retention_days` (default 30, 0 = forever)
- `dataset_retention_max_mb` (default 5000, 0 = unbounded)

A daily background loop runs `compact(repo)`:

1. Prune events older than `retention_days`.
2. If still over `retention_max_mb`, drop oldest events until under
   the cap.
3. Delete shard files that have no index rows pointing at them
   (orphans). The most-recently-modified shard is always spared
   (it's likely the active append target).

Oldest-first policy is intentional — recent data is what the
streamer cares about; an old burst pushing them over the cap should
not lose this week's events.

Idempotent — running `compact` twice produces no extra deletions.

Implementation:
[`dataset/retention.py:compact`](../src/chatterbot/dataset/retention.py#L106-L200).

## Export

`chatterbot dataset export --out my-bundle.cbds` produces a single
tar file containing:

- `manifest.json` — cleartext: schema_version, date range, event
  counts, KDF params, fingerprint, ts. Lets a fine-tune service
  inspect the bundle's shape without decrypting.
- `payload.bin` — `AES-GCM(bundle_dek, zstd(NDJSON of events))`
- `bundle_dek.wrapped` — bundle DEK wrapped under the streamer's
  passphrase-derived KEK
- `payload.nonce` — AES-GCM nonce for the payload

The bundle uses a fresh DEK (separate from the streamer's long-lived
DEK) so the streamer can hand the bundle to a fine-tuning service
without giving up their primary key.

Browser-side export lives at `/dataset/export` in the dashboard;
returns the same `.cbds` bundle as a download.

Implementation:
[`dataset/cli.py:cmd_export`](../src/chatterbot/dataset/cli.py#L253-L411).

## Redaction

The `--redact-users` flag (or the equivalent checkbox on
`/dataset/export`) anonymises chatter usernames in the bundle.
Before serialising, every event is run through
[`dataset/redactor.py`](../src/chatterbot/dataset/redactor.py):

1. **Collect** every user_id the events declare via
   `referenced_user_ids` + nested `snapshot.messages.user_id`.
2. **Sweep** every text field (prompt / response_text / note /
   snapshot message content) for `@(\w{2,})` mentions and resolve
   them against the `user_aliases` table. Catches chatters mentioned
   only in prose, not just in declared metadata.
3. **Build a plan** — each user gets a stable per-bundle token
   (`<USER_001>`, `<USER_002>`, etc).
4. **Apply** with word-boundary case-insensitive matching. So
   `@alice` becomes `@<USER_001>` but `alicelike` stays untouched.

Manifest stamps `redacted: true` and the strategy
(`user_names_with_at_mentions`) so a downstream consumer's policy
can refuse-to-train on un-redacted data.

## Dashboard surface

`/dataset` (visible in the nav only when capture is enabled OR a DEK
is configured):

- **Status panel** — configured / enabled / unlocked tri-state,
  fingerprint, event counts by kind, encrypted bytes.
- **Setup form** (when not yet configured) — passphrase + confirm,
  flashes the recovery string once.
- **Export form** (when configured + has events) — passphrase, optional
  date range, redact-users checkbox.
- **Recent events** — last 20 rows from the index, just metadata
  (never decrypts).

Implementation:
[`web/app.py`](../src/chatterbot/web/app.py) +
[`templates/dataset.html`](../src/chatterbot/web/templates/dataset.html).

The status panel surfaces the most common debug case: capture
enabled but DEK not loaded → yellow warning pointing the streamer at
`CHATTERBOT_DATASET_PASSPHRASE`.
