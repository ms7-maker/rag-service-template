"""
Рендер веб-страниц в Chromium (Playwright), очистка HTML → текст для RAG.

Каждый URL → data/<имя>_clean.txt (например data/pourdebeau_clean.txt).

Первый запуск браузера:
  playwright install chromium

Запуск из assistant_api:
  python render_and_clean.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from data_paths import DATA_DIR, clean_output_path, stem_from_url

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MIN_TEXT_LEN = 100


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


_CHROME_CLASS_NAMES = frozenset(
    {
        "menu",
        "footer",
        "header",
        "nav",
        "navigation",
        "sidebar",
        "popup",
        "cookie",
        "banner",
        "advert",
        "site-header",
        "page-header",
        "main-header",
        "site-footer",
        "main-nav",
        "cookie-banner",
        "cookie-notice",
    }
)


def _is_chrome_by_class_or_id(tag: Tag) -> bool:
    if tag.attrs is None:
        return False
    tag_id = (tag.get("id") or "").strip().lower()
    if tag_id in _CHROME_CLASS_NAMES:
        return True
    for raw in tag.get("class") or []:
        cl = raw.strip().lower()
        if not cl:
            continue
        if cl in _CHROME_CLASS_NAMES:
            return True
        if cl.startswith(("cookie-", "banner-", "popup-", "advert-")):
            return True
    return False


def _clean_html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "header", "footer", "nav", "iframe", "noscript", "svg"]):
        tag.decompose()

    for tag in list(soup.find_all(True)):
        if isinstance(tag, Tag) and _is_chrome_by_class_or_id(tag):
            tag.decompose()

    clean_html = soup.prettify()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return clean_html, text


def _playwright_user_agent() -> str:
    return os.getenv(
        "HTTP_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    ).strip()


def _dismiss_cookie_banner(page) -> None:
    for sel in (
        "button >> text=/принять|соглас|accept/i",
        "[data-testid*='cookie'] button",
        "[class*='cookie'] button",
    ):
        try:
            page.locator(sel).first.click(timeout=2000)
            return
        except Exception:
            continue


def _wait_for_spa_catalog(page, extra_ms: int) -> None:
    page.wait_for_timeout(min(extra_ms, 4000))
    try:
        page.wait_for_load_state("networkidle", timeout=45000)
    except Exception:
        pass
    try:
        page.wait_for_function(
            "() => { const l = document.querySelector('#__layout'); "
            "return l && l.innerText.trim().length > 80; }",
            timeout=45000,
        )
    except Exception:
        page.wait_for_timeout(8000)
    _dismiss_cookie_banner(page)
    for sel in ("text=₽", "a[href*='product']", "[class*='product-card']", "[class*='ProductCard']"):
        try:
            page.wait_for_selector(sel, timeout=20000)
            break
        except Exception:
            continue
    for _ in range(4):
        page.evaluate("window.scrollBy(0, Math.max(window.innerHeight, 900))")
        page.wait_for_timeout(1200)


def _fetch_page(url: str, *, headless: bool, wait_ms: int, nav_timeout: int) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            ctx = browser.new_context(
                user_agent=_playwright_user_agent(),
                locale="ru-RU",
                viewport={"width": 1280, "height": 900},
            )
            page = ctx.new_page()
            page.set_default_timeout(nav_timeout)
            try:
                page.goto(url, wait_until="networkidle", timeout=nav_timeout)
            except Exception:
                page.goto(url, wait_until="load", timeout=nav_timeout)
            _wait_for_spa_catalog(page, wait_ms)
            for attempt in range(4):
                try:
                    return page.content()
                except Exception:
                    if attempt >= 3:
                        raise
                    page.wait_for_timeout(2000)
            return page.content()
        finally:
            browser.close()


def _scrape_url(
    url: str,
    *,
    headless: bool,
    wait_ms: int,
    nav_timeout: int,
    html_out: Path | None = None,
) -> str:
    print(f"URL: {url}")
    html = _fetch_page(url, headless=headless, wait_ms=wait_ms, nav_timeout=nav_timeout)
    clean_html, text = _clean_html_to_text(html)
    if html_out is not None:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(clean_html, encoding="utf-8")
        print(f"HTML: {html_out} ({len(clean_html)} символов)")
    print(f"Текст: {len(text)} символов")
    return text


def _scrape_and_save(
    url: str,
    *,
    headless: bool,
    wait_ms: int,
    nav_timeout: int,
    scraped_dir: Path,
) -> bool:
    stem = stem_from_url(url)
    text = _scrape_url(
        url,
        headless=headless,
        wait_ms=wait_ms,
        nav_timeout=nav_timeout,
        html_out=scraped_dir / f"{stem}_clean.html",
    )
    if len(text) < MIN_TEXT_LEN:
        print(f"Предупреждение: мало текста для {url}", file=sys.stderr)
        return False

    out_path = clean_output_path(stem)
    out_path.write_text(text, encoding="utf-8")
    print(f"Сохранено: {out_path} ({len(text)} символов)")
    return True


def main() -> int:
    load_dotenv(ROOT / ".env")

    main_url = os.getenv("SCRAPER_PAGE_URL", "").strip()
    second_url = os.getenv("SCRAPER_PAGE_URL_2", "").strip()

    if not main_url and not second_url:
        print(
            "Задайте SCRAPER_PAGE_URL и/или SCRAPER_PAGE_URL_2 в .env (см. .env_exsample)",
            file=sys.stderr,
        )
        return 1

    headless = _env_bool("PLAYWRIGHT_HEADLESS", True)
    wait_ms = int(os.getenv("PLAYWRIGHT_WAIT_MS", "5000").strip() or "5000")
    nav_timeout = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "60000").strip() or "60000")

    print(f"headless={headless}, wait={wait_ms}ms, timeout={nav_timeout}ms")
    print(f"Выход: {DATA_DIR}/*_clean.txt")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scraped_dir = DATA_DIR / "scraped"

    saved = 0
    urls = [u for u in (main_url, second_url) if u]

    for i, url in enumerate(urls):
        if i > 0:
            print("---")
        if _scrape_and_save(
            url,
            headless=headless,
            wait_ms=wait_ms,
            nav_timeout=nav_timeout,
            scraped_dir=scraped_dir,
        ):
            saved += 1

    if saved == 0:
        print("Не удалось получить достаточно текста ни с одного URL", file=sys.stderr)
        return 1

    print(f"\nГотово: сохранено страниц {saved}")
    print("Для пересборки индекса удалите папку faiss_db и снова запустите app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
