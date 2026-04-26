"""Capture dashboard screenshots at 360 / 768 / 1280px wide.

Used to spot-check the responsive dark-mode layout. Run with the dashboard
already serving on http://127.0.0.1:8765 (e.g. `make dashboard` against
data/demo.db). Dark mode is forced via localStorage on each context.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright


OUT = Path("screenshots")
BASE = "http://127.0.0.1:8765"

# Each tuple: (slug, viewport-w, viewport-h, path)
SHOTS = [
    ("chatters_360",  360, 740,  "/"),
    ("user_768",      768, 1024, "/users/11005"),
    ("user_1280",     1280, 900, "/users/11005"),
    # Plus a few extras for self-check, not required by the brief.
    ("topics_768",    768, 1024, "/topics"),
    ("events_1280",   1280, 900, "/events"),
    ("live_360",      360, 740,  "/live"),
    ("settings_768",  768, 1024, "/settings"),
    ("modal_forget_768",   768, 1024, "/users/11005?_modal=forget"),
    ("modal_shortcuts_768", 768, 1024, "/?_modal=shortcuts"),
]


async def force_dark_mode(page) -> None:
    await page.add_init_script(
        """
        try {
          localStorage.setItem('chatterbot.theme', 'dark');
        } catch (_) {}
        """
    )


async def main() -> int:
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            for slug, w, h, path in SHOTS:
                ctx = await browser.new_context(
                    viewport={"width": w, "height": h},
                    color_scheme="dark",
                )
                page = await ctx.new_page()
                await force_dark_mode(page)
                trigger_modal = path.endswith("?_modal=forget") or path.endswith("?_modal=shortcuts")
                clean_path = path.split("?_modal=")[0]
                try:
                    await page.goto(BASE + clean_path, wait_until="networkidle", timeout=15000)
                except Exception as e:
                    print(f"!!! {clean_path} failed: {e}")
                    await ctx.close()
                    continue
                # Trigger modal openings via the hover/click on a test target.
                if path.endswith("?_modal=forget"):
                    # Click the "forget" button on the user detail header.
                    try:
                        await page.get_by_role("button", name="forget").click()
                        await page.wait_for_selector("#modal-root > *", timeout=4000)
                    except Exception as e:
                        print(f"   could not open forget modal: {e}")
                elif path.endswith("?_modal=shortcuts"):
                    # Trigger the shortcuts modal via the keyboard.
                    try:
                        await page.keyboard.press("Shift+Slash")  # ?
                        await page.wait_for_selector("#modal-root > *", timeout=4000)
                    except Exception as e:
                        print(f"   could not open shortcuts modal: {e}")
                out_path = OUT / f"{slug}.png"
                await page.screenshot(path=str(out_path), full_page=True)
                print(f"   {out_path}")
                await ctx.close()
        finally:
            await browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
