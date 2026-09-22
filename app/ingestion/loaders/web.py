"""Web intake for the ingestion pipeline.

Two source kinds:
  * ``supabase``        — pull rows from Supabase tables via the PostgREST API
                          (``/rest/v1/<table>?select=*``) and render each row
                          to plain text.
  * ``page``            — fetch a server-rendered web page and parse it with the
                          same BeautifulSoup loader used for local HTML files.

The rendered text flows through the shared chunker + embeddings unchanged, so
web sources behave exactly like local documents.
"""
import re
from typing import Any, Iterator

import logfire
import requests

from app.config import settings
from app.ingestion.loaders.html import parse_html_content

_TIMEOUT = 25
_RETRIES = 2
_UA = {"User-Agent": "Mozilla/5.0 (compatible; VantageRAG/1.0 +https://kartikworks.co.in)"}

# Row fields that carry no retrieval value for the "about me" corpus.
_EXCLUDE_KEYS = {
    "id",
    "created_at",
    "updated_at",
    "media_path",
    "media_url",
    "sort_order",
}
_TITLE_CANDIDATES = ("title", "name", "role", "company", "project", "heading")


def _key_label(key: str) -> str:
    return re.sub(r"[_]+", " ", key).strip().title()


def fetch_page(url: str, timeout: int = _TIMEOUT) -> str:
    """GET a page and return its parsed readable text (HTML parser shared with
    the file loaders). Raises on non-2xx."""
    with logfire.span("🌐 Fetch Page", url=url):
        resp = requests.get(url, headers=_UA, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
        text = parse_html_content(resp.text)
        if not text.strip():
            logfire.warning(f"No readable text extracted from {url} (SPA shell?).")
        return text


def fetch_json(url: str, timeout: int = _TIMEOUT) -> list[dict[str, Any]]:
    """GET a URL and parse the body as a JSON array/list of records."""
    with logfire.span("🌐 Fetch JSON", url=url):
        resp = requests.get(url, headers=_UA, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []


def fetch_records(
    base_url: str | None = None,
    anon_key: str | None = None,
    table: str = "projects",
    limit: int | None = 100,
    select: str = "*",
    timeout: int = _TIMEOUT,
) -> list[dict[str, Any]]:
    """Fetch rows from one Supabase/public JSON table.

    Uses the anon key with ``apikey`` + ``Authorization: Bearer`` headers
    (standard PostgREST). Non-2xx responses raise with the server message."""
    if not base_url:
        raise ValueError("SUPABASE_URL is not configured.")
    url = f"{base_url.rstrip('/')}/rest/v1/{table}"
    params: dict[str, Any] = {"select": select}
    if limit:
        params["limit"] = limit
    headers = {"Accept": "application/json"}
    if anon_key:
        headers["apikey"] = anon_key
        headers["Authorization"] = f"Bearer {anon_key}"
    else:
        raise ValueError("SUPABASE_ANON_KEY is not configured.")

    with logfire.span("🌐 Fetch Supabase Records", table=table):
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Supabase {table} -> HTTP {resp.status_code}: {resp.text[:200]}")
        rows = resp.json()
        return rows if isinstance(rows, list) else []


def _row_to_text(row: dict[str, Any]) -> str:
    title_key = next((c for c in _TITLE_CANDIDATES if row.get(c)), None)
    title = str(row[title_key]).strip() if title_key else None
    blocks: list[str] = []
    for key, value in row.items():
        if key in _EXCLUDE_KEYS or key == title_key:
            continue
        if value is None:
            continue
        if isinstance(value, list):
            value = ", ".join(str(x) for x in value)
        elif isinstance(value, (dict, tuple)):
            continue
        text = str(value).strip()
        if not text:
            continue
        blocks.append(f"{_key_label(key)}: {text}")
    if not blocks and title:
        return title
    if title:
        return f"{title} — " + " | ".join(blocks)
    return " | ".join(blocks)


def records_to_text(rows: list[dict[str, Any]]) -> str:
    """Render Supabase rows to readable paragraphs the chunker can consume."""
    parts: list[str] = []
    for row in rows:
        parts.append(_row_to_text(row))
    return "\n\n".join(p for p in parts if p)


def table_text(base_url: str, anon_key: str, table: str, limit: int | None = 100) -> str:
    """One-shot convenience: fetch a table and render it to text."""
    with logfire.span("Supabase Table → Text", table=table):
        rows = fetch_records(base_url, anon_key, table=table, limit=limit)
        logfire.info(f"Supabase '{table}': {len(rows)} rows.")
        return records_to_text(rows)


def iter_configured_sources() -> Iterator[dict[str, Any]]:
    """Yield configured web sources:
        * one entry per Supabase table (SUPABASE_TABLES)
        * each entry in WEB_SOURCES_JSON (type page|json)
    """
    if settings.SUPABASE_URL and settings.SUPABASE_ANON_KEY:
        for table in settings.supabase_tables:
            yield {
                "kind": "supabase",
                "source_type": "supabase",
                "title": table,
                "url": settings.SUPABASE_URL,
                "table": table,
            }
    sources = settings.web_sources
    for entry in sources:
        yield {
            "kind": entry.get("type", "page"),
            "source_type": entry.get("source_type", "web"),
            "title": entry.get("title") or entry.get("url", "web"),
            "url": entry.get("url", ""),
            "table": entry.get("table"),
        }
    if not (settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY) and not sources:
        logfire.info("No web sources configured (SUPABASE_URL / WEB_SOURCES_JSON empty).")


def source_text(src: dict[str, Any]) -> str:
    if src["kind"] == "supabase":
        return table_text(src["url"], settings.SUPABASE_ANON_KEY, src["table"])
    if src["kind"] == "json":
        return records_to_text(fetch_json(src["url"]))
    return fetch_page(src["url"])