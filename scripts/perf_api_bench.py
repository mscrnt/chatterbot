"""Sequential API benchmark for the chatterbot dashboard.

Hits each route N times sequentially (so latency stats reflect
single-user clicking, not max throughput), reports min / median / p95
/ max in ms. Run against a live dashboard:

    BENCH_URL=http://127.0.0.1:8765 .venv/bin/python scripts/perf_api_bench.py

The endpoints are grouped so the report shows hot click-paths separate
from polling endpoints. Sample-count is per endpoint; defaults are
small to keep total run time under a minute on a busy dashboard.
"""

from __future__ import annotations

import os
import statistics
import time

import httpx


BASE = os.environ.get("BENCH_URL", "http://127.0.0.1:8765").rstrip("/")
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "8"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "1"))


# (group, label, path) tuples — order matters only for grouping.
# Hot click-paths first; polling / partial endpoints second.
ENDPOINTS: list[tuple[str, str, str]] = [
    ("baseline",     "/health",                                "/health"),
    ("page",         "/  (insights, full page)",                "/"),
    ("page",         "/insights?view=engagement",                "/insights?view=engagement"),
    ("page",         "/insights?view=topics",                    "/insights?view=topics"),
    ("page",         "/chatters",                                "/chatters"),
    ("page",         "/settings",                                "/settings"),
    ("page",         "/search",                                  "/search"),
    ("page",         "/live",                                    "/live"),
    ("page",         "/audit",                                   "/audit"),
    ("page",         "/stats",                                    "/stats"),
    ("page",         "/stats/wordcloud",                          "/stats/wordcloud"),
    # Partials that re-render via HTMX on every SSE / interval tick.
    ("partial",      "/insights?partial=1&view=engagement&window=7d",
                                                                 "/insights?partial=1&view=engagement&window=7d"),
    ("partial",      "/insights?partial=1&view=topics",          "/insights?partial=1&view=topics"),
    ("partial",      "/transcript?view=summary&limit=15",        "/transcript?view=summary&limit=15"),
    ("partial",      "/transcript?view=log&limit=15",            "/transcript?view=log&limit=15"),
]


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:7.1f}ms"


def bench_one(client: httpx.Client, path: str) -> tuple[list[float], int, int]:
    """Returns (sample_seconds, last_status, last_size_bytes). Warmup
    runs are not included in the sample list — they exist so the
    server's caches / lazy imports are paid before we start timing."""
    samples: list[float] = []
    status = 0
    size = 0
    for i in range(WARMUP + SAMPLES):
        t0 = time.perf_counter()
        resp = client.get(path)
        dt = time.perf_counter() - t0
        if i >= WARMUP:
            samples.append(dt)
        status = resp.status_code
        size = len(resp.content)
    return samples, status, size


def main() -> None:
    print(f"# chatterbot API benchmark · base={BASE} · samples={SAMPLES} · warmup={WARMUP}\n")
    print(
        f"{'group':<10} {'endpoint':<55} "
        f"{'status':<7} {'size(KB)':<10} "
        f"{'min':>9} {'median':>9} {'p95':>9} {'max':>9}"
    )
    print("-" * 130)

    with httpx.Client(base_url=BASE, timeout=30.0, follow_redirects=False) as client:
        # Verify the dashboard is up before slamming it.
        try:
            client.get("/health")
        except Exception as e:
            print(f"ERROR: dashboard not reachable at {BASE}: {e}")
            return

        results = []
        for group, label, path in ENDPOINTS:
            try:
                samples, status, size = bench_one(client, path)
            except httpx.HTTPError as e:
                print(f"{group:<10} {label:<55} ERR: {e}")
                continue
            samples_sorted = sorted(samples)
            mn = min(samples)
            md = statistics.median(samples)
            p95 = samples_sorted[max(0, int(len(samples_sorted) * 0.95) - 1)]
            mx = max(samples)
            print(
                f"{group:<10} {label:<55} "
                f"{status:<7} {size/1024:<10.1f} "
                f"{_fmt_ms(mn):>9} {_fmt_ms(md):>9} {_fmt_ms(p95):>9} {_fmt_ms(mx):>9}"
            )
            results.append((group, label, status, size, mn, md, p95, mx))

    # Top offenders
    print("\n# top 5 by p95 (slowest)")
    results.sort(key=lambda r: r[6], reverse=True)
    for group, label, status, size, mn, md, p95, mx in results[:5]:
        print(f"  {group:<10} {label:<55} p95={_fmt_ms(p95)} median={_fmt_ms(md)}")


if __name__ == "__main__":
    main()
