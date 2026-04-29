"""Click-latency benchmark via Playwright.

Measures the gap between clicking a UI element and the page being
interactive again. Captures both:
  - server response time (HTMX swap completes, DOM updated)
  - client hydration (Alpine.js + listener wiring + first paint)

Run against a live dashboard:

    BENCH_URL=http://127.0.0.1:8765 .venv/bin/python scripts/perf_playwright_bench.py

We use Chromium headless; first run downloads ~150MB if missing.
"""

from __future__ import annotations

import os
import statistics
import time
from contextlib import contextmanager

from playwright.sync_api import sync_playwright, Page, TimeoutError as PWTimeout


BASE = os.environ.get("BENCH_URL", "http://127.0.0.1:8765").rstrip("/")
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "5"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "1"))


@contextmanager
def _timer():
    """Yield a callable that returns elapsed seconds when called."""
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0


def _nav_load(page: Page, url: str) -> dict[str, float]:
    """Time a full navigation. Returns {response, dom_loaded, load,
    first_paint, ttfp_after_idle}. `ttfp_after_idle` is the time to
    'page is settled' — when networkidle fires."""
    out: dict[str, float] = {}
    t0 = time.perf_counter()
    resp = page.goto(url, wait_until="commit")
    out["response"] = time.perf_counter() - t0
    page.wait_for_load_state("domcontentloaded")
    out["dom_loaded"] = time.perf_counter() - t0
    page.wait_for_load_state("load")
    out["load"] = time.perf_counter() - t0
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
        out["networkidle"] = time.perf_counter() - t0
    except PWTimeout:
        out["networkidle"] = float("nan")
    return out


def _click_swap(page: Page, selector: str, target_text_or_selector: str) -> float:
    """Click `selector`, wait until the page contains
    `target_text_or_selector` (str = text to wait for, or starts with
    'sel:' = CSS selector). Returns elapsed seconds."""
    t0 = time.perf_counter()
    page.click(selector)
    if target_text_or_selector.startswith("sel:"):
        page.wait_for_selector(target_text_or_selector[4:], state="visible", timeout=15_000)
    else:
        page.wait_for_selector(
            f"text={target_text_or_selector}", state="visible", timeout=15_000,
        )
    return time.perf_counter() - t0


def _stats(samples: list[float]) -> str:
    if not samples:
        return "(no data)"
    s = sorted(samples)
    p95 = s[max(0, int(len(s) * 0.95) - 1)]
    return (
        f"min={min(samples)*1000:6.0f}ms · "
        f"med={statistics.median(samples)*1000:6.0f}ms · "
        f"p95={p95*1000:6.0f}ms · "
        f"max={max(samples)*1000:6.0f}ms"
    )


def main() -> None:
    print(f"# Playwright click-latency benchmark · base={BASE} · samples={SAMPLES}\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()

        # ---- 1. Cold navigation timings ----
        print("## Cold navigation (page load → settled)\n")
        nav_targets = [
            ("/  (insights home)", BASE + "/"),
            ("/insights?view=topics", BASE + "/insights?view=topics"),
            ("/chatters", BASE + "/chatters"),
            ("/settings", BASE + "/settings"),
            ("/live", BASE + "/live"),
            ("/stats", BASE + "/stats"),
        ]
        for label, url in nav_targets:
            samples_resp: list[float] = []
            samples_load: list[float] = []
            samples_idle: list[float] = []
            for i in range(WARMUP + SAMPLES):
                t = _nav_load(page, url)
                if i >= WARMUP:
                    samples_resp.append(t["response"])
                    samples_load.append(t["load"])
                    samples_idle.append(t["networkidle"])
            print(f"  {label}")
            print(f"    response:    {_stats(samples_resp)}")
            print(f"    load event:  {_stats(samples_load)}")
            print(f"    networkidle: {_stats(samples_idle)}")
            print()

        # ---- 2. Click latency on already-loaded /insights ----
        print("## Click latency on /insights\n")
        page.goto(BASE + "/", wait_until="networkidle")

        # Pull the visible nav tab strip's elements once for stable selectors.
        # The dashboard's nav uses /insights, /chatters, /settings, /live, /search
        click_scenarios = [
            (
                "click 'Chatters' nav link → page shows chatter list",
                "a[href='/chatters']",
                "sel:#chatters-list, [data-chatters-list], h1:has-text('Chatters')",
            ),
            (
                "click 'Settings' nav link → settings page renders",
                "a[href='/settings']",
                "sel:form[action='/settings']",
            ),
            (
                "click 'Search' nav link → search page renders",
                "a[href='/search']",
                "sel:input[name='q']",
            ),
            (
                "click 'Live' nav link → live page renders",
                "a[href='/live']",
                "sel:[data-live-feed], h1:has-text('Live')",
            ),
        ]
        for label, click_sel, wait_for in click_scenarios:
            samples: list[float] = []
            # Each iteration: navigate home then click → measure
            for i in range(WARMUP + SAMPLES):
                page.goto(BASE + "/", wait_until="networkidle")
                try:
                    dt = _click_swap(page, click_sel, wait_for)
                except PWTimeout as e:
                    print(f"  {label}: timeout — {e}")
                    break
                except Exception as e:
                    print(f"  {label}: error — {e}")
                    break
                if i >= WARMUP:
                    samples.append(dt)
            if samples:
                print(f"  {label}\n    {_stats(samples)}\n")

        # ---- 3. Settings tab switching (the new tabbed UI) ----
        print("## Settings tab switching\n")
        page.goto(BASE + "/settings", wait_until="networkidle")
        # The 5 tab labels live in the nav element rendered by settings.html
        tabs = ["AI brain", "Voice & screen", "Insights", "Advanced", "Connections"]
        for tab_label in tabs:
            samples: list[float] = []
            for i in range(WARMUP + SAMPLES):
                # Reset by clicking Connections first (the default tab)
                page.click(f"button[role='tab']:has-text('Connections')")
                page.wait_for_timeout(50)
                t0 = time.perf_counter()
                page.click(f"button[role='tab']:has-text('{tab_label}')")
                # Wait for any field from the tab to be visible — the
                # tab body uses x-show so we just need the cloak to
                # clear. Use a small idle wait.
                try:
                    page.wait_for_function(
                        "document.querySelectorAll('section[x-cloak]').length === "
                        "document.querySelectorAll('section').length - "
                        "document.querySelectorAll('section:not([x-cloak])').length",
                        timeout=2000,
                    )
                except PWTimeout:
                    pass
                dt = time.perf_counter() - t0
                if i >= WARMUP:
                    samples.append(dt)
            if samples:
                print(f"  switch → {tab_label!r}: {_stats(samples)}")

        browser.close()


if __name__ == "__main__":
    main()
