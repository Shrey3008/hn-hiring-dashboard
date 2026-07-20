"""Capture README screenshots from the running app.

Dev-only tooling — Playwright is deliberately NOT in requirements.txt, since
the deployed app has no use for a browser engine.

    .venv/bin/playwright install chromium     # once
    streamlit run app.py                      # in another shell
    python scripts/capture_screenshots.py

Writes PNGs to docs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
URL = "http://localhost:8501"

VIEWPORT = {"width": 1440, "height": 950}


def scroll_to(page, y: int) -> None:
    """Scroll Streamlit's inner main container, not the window."""
    page.evaluate(
        """(y) => {
            const cands = [...document.querySelectorAll('*')]
              .filter(e => e.scrollHeight > e.clientHeight + 200 && e.clientHeight > 300);
            (cands[0] || document.scrollingElement).scrollTop = y;
        }""",
        y,
    )
    page.wait_for_timeout(1200)


def main() -> int:
    DOCS.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, color_scheme="dark")

        try:
            page.goto(URL, timeout=20_000)
        except Exception:
            print(f"Could not reach {URL} — is `streamlit run app.py` running?")
            return 1

        # Streamlit streams content in; wait for the last section to exist.
        page.wait_for_selector("text=Browse job posts", timeout=60_000)
        page.wait_for_timeout(3500)

        shots = [
            ("01-overview.png", 0, "KPI row + NL query + hiring volume"),
            ("02-work-mode.png", 780, "remote vs hybrid vs onsite"),
            ("03-tech-trend.png", 1500, "technology trend"),
        ]
        for name, y, label in shots:
            scroll_to(page, y)
            page.screenshot(path=str(DOCS / name))
            print(f"  wrote docs/{name}  ({label})")

        browser.close()

    print(f"\n{len(list(DOCS.glob('*.png')))} screenshots in {DOCS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
