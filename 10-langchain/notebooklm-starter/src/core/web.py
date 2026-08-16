"""Thin wrapper around the Firecrawl SDK: web search, single-page scrape, site crawl."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from firecrawl import Firecrawl
from firecrawl.v2.types import ScrapeOptions

_client: Firecrawl | None = None

# A line that is *only* a markdown link/image (optionally a list item) — the shape of a
# nav menu or footer link. A run of several of these in a row is boilerplate, not content;
# a couple in a row can legitimately appear in an article (e.g. a short "see also" list).
_LINK_ONLY_LINE_RE = re.compile(r"^-?\s*!?\[[^\]]*\]\([^)]+\)\s*$")
_MIN_LINK_CLUSTER = 4


def _clean_markdown(text: str) -> str:
    """Strip nav/footer link clusters and reCAPTCHA notices picked up alongside the content."""
    lines = [line for line in text.splitlines() if "recaptcha" not in line.lower()]

    kept: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if _LINK_ONLY_LINE_RE.match(lines[i].strip()):
            j, link_count = i, 0
            while j < n and (not lines[j].strip() or _LINK_ONLY_LINE_RE.match(lines[j].strip())):
                if lines[j].strip():
                    link_count += 1
                j += 1
            if link_count >= _MIN_LINK_CLUSTER:
                i = j
                continue
        kept.append(lines[i])
        i += 1

    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _get_client() -> Firecrawl:
    global _client
    if _client is None:
        _client = Firecrawl(api_key=os.environ["FIRECRAWL_API_KEY"])
    return _client


@dataclass
class WebResult:
    """One hit from a web search — no content yet, just enough to judge relevance."""

    url: str
    title: str
    description: str


@dataclass
class WebPage:
    """Scraped page content, ready to be indexed as a source."""

    url: str
    title: str
    content: str


def web_search(query: str, limit: int = 5) -> list[WebResult]:
    data = _get_client().search(query, limit=limit)
    return [
        WebResult(url=r.url, title=r.title or r.url, description=r.description or "")
        for r in (data.web or [])
    ]


def web_scrape(url: str) -> WebPage:
    doc = _get_client().scrape(url, formats=["markdown"])
    title = (doc.metadata.title if doc.metadata else None) or url
    return WebPage(url=url, title=title, content=_clean_markdown(doc.markdown or ""))


def web_crawl(url: str, limit: int = 5) -> list[WebPage]:
    job = _get_client().crawl(url, limit=limit, scrape_options=ScrapeOptions(formats=["markdown"]))
    pages = []
    for doc in job.data or []:
        page_url = (doc.metadata.url if doc.metadata else None) or url
        title = (doc.metadata.title if doc.metadata else None) or page_url
        pages.append(WebPage(url=page_url, title=title, content=_clean_markdown(doc.markdown or "")))
    return pages
