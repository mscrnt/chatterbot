# Development

Setting up a dev environment, running tests, contributing changes,
filing bugs, and the project's style conventions.

## Contents

- [Dev environment](#dev-environment)
- [Tests](#tests)
- [GitHub Actions CI](#github-actions-ci)
- [Code style](#code-style)
- [Filing bugs](#filing-bugs)
- [Pull requests](#pull-requests)
- [Project structure](#project-structure)

---

## Dev environment

```bash
git clone https://github.com/mscrnt/chatterbot
cd chatterbot
uv sync --extra dataset --extra dev
```

That gets you the base install + `cryptography` / `zstandard` (for
dataset capture) + `pytest` / `pytest-asyncio` / `ruff` (for tests +
lint). If you'll touch the whisper pipeline, add `--extra whisper`
too.

Run any tool through uv to use the project's venv:

```bash
uv run pytest
uv run ruff check src/ tests/
uv run python -m chatterbot dashboard
```

Or activate the venv once per shell:

```bash
source .venv/bin/activate
pytest
```

## Tests

The suite has two tiers, both runnable with `pytest`:

- **Unit tier** — factory-fed throwaway DBs, mock LLM at the
  Protocol layer, no external services. Runs in <60s on CI.
- **Replay tier** (optional) — opt-in via env var. Tests marked
  `@pytest.mark.replay` open a real chatter DB read-only and
  exercise heavier query helpers. Skipped silently when no DB is
  configured.

Run the unit tier:

```bash
uv run pytest
```

The full suite is currently 211 tests in ~50s. See
[`tests/`](../tests/) for the layout — one test file per
module-under-test.

### Mock LLM client

We mock at the LLMProvider Protocol layer, not the HTTP layer.
[`tests/_mock_llm.py`](../tests/_mock_llm.py) implements the same
surface `OllamaClient` / `AnthropicClient` / `OpenAIClient` share,
with FIFO-queued canned responses keyed on
`(call_site, response_model)`:

```python
# Pseudocode for queueing a canned LLM response in a test:
mock_llm = MockLLMClient()
mock_llm.queue_response(
    call_site="insights.engaging_subjects",
    response=EngagingSubjectsResponse(subjects=[...]),
)
# ...drive production code through it...
assert mock_llm.calls[0].call_site == "insights.engaging_subjects"
```

Real implementation: [`tests/_mock_llm.py`](../tests/_mock_llm.py).

### Test fixtures

Shared fixtures in [`tests/conftest.py`](../tests/conftest.py):

- `tmp_repo` — fresh `ChatterRepo` against a throwaway sqlite DB
- `unlocked_repo` — `tmp_repo` with a DEK loaded + capture enabled
- `mock_llm` — fresh `MockLLMClient` per test
- `make_user`, `make_message` — factories for seeding rows

### Call-site coverage test

`test_call_sites.py` AST-walks the production tree and verifies
every `generate_structured(...)` call passes a `call_site=` argument
matching the expected registry. Catches the regression where someone
adds a new prompt site but forgets to wire the dataset capture
correctly.

```python
# Pseudocode for the call-site assertion:
sites = collect_call_sites_via_ast()
assert {cs for (_, _, cs) in sites if cs} == EXPECTED_CALL_SITES
```

Real implementation:
[`tests/dataset/test_call_sites.py`](../tests/dataset/test_call_sites.py).

## GitHub Actions CI

[`.github/workflows/tests.yml`](../.github/workflows/tests.yml) runs
on every push to main + every PR. Python 3.12 on ubuntu-latest:

```yaml
# Steps in the workflow (pseudocode):
- uses: actions/checkout@v4
- uses: astral-sh/setup-uv@v5 (with caching)
- uv python install 3.12
- uv sync --extra dataset --extra dev
- uv run pytest
```

The unit suite is the only thing CI runs — no integration tests, no
deployments. Replay tests only fire when a fixture DB is configured,
which CI never has. ~50s end-to-end including the uv install.

Stale runs on the same branch cancel automatically (concurrency
group on the ref).

## Code style

A few conventions worth knowing about:

### Comments

- Default to writing **no comments**. Identifier names should carry
  meaning.
- Add comments only when the *why* is non-obvious — a hidden
  constraint, a subtle invariant, a workaround for a specific bug,
  behavior that would surprise a reader.
- Don't explain the *what* (the code already does that).
- One short line max for inline comments. Multi-paragraph docstrings
  are reserved for module-level + public-API.

### Imports

- Lazy-import optional extras. The `dataset` module pulls in
  `cryptography` + `zstandard`; we defer those imports to inside
  the functions that need them so a base install (without the
  extra) can still import the module + see "feature off" cleanly.

### Migrations

- Additive only. No destructive schema changes after data lands.
- Pattern: `PRAGMA table_info` + `ALTER TABLE` inside the
  `_init_schema` block in [`repo.py`](../src/chatterbot/repo.py).

### Commits

- The project doesn't use a `Co-Authored-By` footer.
- Subject line ≤ 70 chars; body explains the *why* and any user-
  visible behavior change.
- Prefer one focused commit per slice over many tiny commits.

### Tests

- Every test file should be runnable in isolation
  (`pytest tests/insights/test_engaging_subjects.py`).
- Pin invariants, not implementation details. Tests that assert on
  the SHAPE of a response are good; tests that assert on internal
  cluster IDs are brittle.
- Skip tests that need external services unless an env var
  configures them in (see the replay-tier pattern).

## Filing bugs

The dashboard's Diagnostics tab generates a privacy-safe `.cbreport`
zip:

1. Visit `/settings → Diagnostics`
2. (Optional) check "Anonymize chatter names" + "Include 24h
   activity" if the bug is timing-related
3. Click **Download bundle** — produces `chatterbot-diagnose-<ts>.cbreport`
4. Click **Open new GitHub issue** — opens a pre-filled issue
   template
5. Drag the `.cbreport` into the issue body to attach it

The bundle includes log tails, system info, DB row counts, app
settings (with secrets masked), insight-state shape, and dataset-
capture status. It excludes chat content, usernames (unless
"Include 24h activity" was checked), and any secrets.

See [`diagnose.py`](../src/chatterbot/diagnose.py) for what's in /
out of the bundle.

## Pull requests

Smaller is better. The commits in this repo's history follow a
"slice" pattern — each commit is one shippable feature with its
own tests. Look at recent commits for examples of the cadence
("Dataset capture: opt-in encrypted training-dataset (slice 1)" —
self-contained PR with cipher + storage + tests + CI).

Before submitting:

1. Run `uv run pytest` locally and confirm all tests pass
2. Run `uv run ruff check src/ tests/` for lint
3. If you added a new LLM call site, add it to
   `EXPECTED_CALL_SITES` in
   [`tests/dataset/test_call_sites.py`](../tests/dataset/test_call_sites.py)
4. If you touched a streamer-personality prompt, consider whether it
   should go in the editable registry in
   [`llm/prompts.py`](../src/chatterbot/llm/prompts.py) — see
   [prompts.md](prompts.md)
5. If you added new database columns or tables, follow the additive
   migration pattern (no destructive changes) — see
   [`repo.py`](../src/chatterbot/repo.py)

PR description should explain the *why* — what problem you saw,
what you tried, what trade-offs you weighed. The codebase's commit
log is full of these as templates.

## Project structure

```
chatterbot/
├── docs/                       # this directory
├── src/chatterbot/
│   ├── bot.py                  # Twitch listener (the bot process)
│   ├── tui.py                  # Textual streamer viewer
│   ├── main.py                 # CLI dispatch
│   ├── config.py               # Settings + EDITABLE_SETTING_KEYS
│   ├── repo.py                 # SQLite access (the only DB layer)
│   ├── summarizer.py           # per-chatter notes + profile + topics
│   ├── moderator.py            # opt-in advisory mod classifier
│   ├── insights.py             # talking points / subjects / etc.
│   ├── transcript.py           # whisper pipeline + group summaries
│   ├── threader.py             # topic-thread clustering
│   ├── twitch.py               # Helix poller + status
│   ├── helix_sync.py           # roster sync (mods / vips / subs)
│   ├── obs.py                  # OBS websocket poller
│   ├── streamelements.py       # SE WebSocket listener
│   ├── youtube.py              # YT live-chat poller
│   ├── discord_bot.py          # Discord listener (WIP)
│   ├── eventbus.py             # cross-process notify
│   ├── spam.py                 # ingest-time spam scoring
│   ├── latency.py              # internal-bus latency telemetry
│   ├── diagnose.py             # .cbreport bundle builder
│   ├── llm/
│   │   ├── ollama_client.py    # Ollama HTTP wrapper
│   │   ├── providers.py        # Anthropic + OpenAI; LLMProvider Protocol
│   │   ├── schemas.py          # every pydantic structured-output model
│   │   └── prompts.py          # streamer-customizable prompt registry
│   ├── dataset/
│   │   ├── cipher.py           # Argon2id + AES-GCM
│   │   ├── storage.py          # append-only encrypted shard writer
│   │   ├── capture.py          # record_llm_call + record_streamer_action
│   │   ├── retention.py        # daily compaction loop
│   │   ├── redactor.py         # export-time anonymisation
│   │   ├── loops.py            # context-snapshot + retention background tasks
│   │   └── cli.py              # `chatterbot dataset` subcommands
│   └── web/
│       ├── app.py              # FastAPI routes (one big file by design)
│       ├── auth.py             # optional basic auth middleware
│       ├── settings_meta.py    # /settings field metadata
│       ├── insight_rag.py      # per-user RAG endpoint
│       ├── rag.py              # search endpoints
│       ├── thread_rag.py       # thread-explain SSE
│       ├── topic_rag.py        # topic-thread RAG
│       ├── static/             # tailwind output + small scripts
│       └── templates/          # Jinja2 + HTMX views
├── tests/                      # pytest suite (mirrors src/ layout)
├── obs_scripts/
│   └── audio_client.py         # OBS Python script — audio relay
├── scripts/                    # backfill + perf benchmark scripts
├── data/                       # gitignored; runtime state (db, logs, etc.)
├── pyproject.toml              # uv / pip-installable
├── docker-compose.yml          # bot + dashboard as separate services
└── README.md                   # quickstart + doc links
```

The `web/app.py` is intentionally one big file — keeps every route in
one searchable place rather than scattered across blueprint
files. The route count is ~70+ but each is short and the file is
sectioned with comment banners.
