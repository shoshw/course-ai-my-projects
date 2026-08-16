"""The web research agent: finds, evaluates, and indexes internet sources on a topic.

Given a topic, the agent runs several differently-phrased searches (via Firecrawl) to
cover it from multiple angles, picks the results that look like genuinely good sources,
scrapes them, and adds each one to the SourceStore — where it's chunked and embedded
exactly like an uploaded document.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.globals import set_debug
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from core.store import SourceStore, store
from core.web import WebPage, web_crawl, web_scrape, web_search

if os.getenv("NOTEBOOKLM_DEBUG") == "1":
    set_debug(True)

_MAX_PAGE_CHARS = 6000

SYSTEM_PROMPT = (
    "You are the web research assistant for NotebookLM. Given a topic, find good "
    "sources on the open web and add them to the user's source library so the chat "
    "agent can search and cite them later.\n\n"
    "Process:\n"
    "1. Come up with 3-4 differently phrased search queries that approach the topic "
    "from different angles (e.g. overview, recent developments, comparisons, technical "
    "detail) so results aren't all the same page.\n"
    "2. Call `search_web` for each query.\n"
    "3. From all the results, pick the ones that look like genuinely high-quality, "
    "relevant, non-duplicate sources — skip thin or spammy pages, prefer reputable "
    "domains, and don't pick the same URL or domain twice.\n"
    "4. For each selected result, call `scrape_url` to fetch its content. If a result "
    "looks like a docs/wiki site where several pages would be useful, call `crawl_site` "
    "on it instead.\n"
    "5. For every page worth keeping, call `add_source` with that page's url (and, "
    "optionally, a nicer name). You never need to retype or paste a page's text — "
    "`add_source` looks up and indexes the exact content you already fetched.\n"
    "6. Reply with a short summary of which sources you added and why you picked them.\n\n"
    "Be selective rather than exhaustive — aim for roughly 3-6 well-chosen sources per "
    "topic, not every result you found."
)


def _format_results(results, query: str) -> str:
    if not results:
        return f"No results for query {query!r}."
    return "\n".join(f"- {r.title} — {r.url}\n  {r.description}" for r in results)


def _format_page(page) -> str:
    content = page.content[:_MAX_PAGE_CHARS]
    if len(page.content) > _MAX_PAGE_CHARS:
        content += "\n[content truncated]"
    return f"Title: {page.title}\nURL: {page.url}\n\n{content}"


def make_tools_(store: SourceStore) -> list:
    """Build the research tools bound to a given SourceStore.

    `scrape_url`/`crawl_site` cache the full page text they fetch, keyed by url; `add_source`
    then indexes that cached text by url rather than trusting the model to retype it. An LLM
    tool call is a poor transport for a multi-KB page — it tends to paraphrase or truncate,
    which would silently index a summary instead of the source itself.
    """

    fetched: dict[str, WebPage] = {}

    @tool
    def search_web(query: str) -> str:
        """Search the web for a query and return matching results (title, url, description).

        Call this once per query phrasing to cover a topic from different angles.
        """
        try:
            return _format_results(web_search(query), query)
        except Exception as exc:  # Firecrawl API errors shouldn't crash the run.
            return f"Search failed for query {query!r}: {exc}"

    @tool
    def scrape_url(url: str) -> str:
        """Scrape a single web page and return its title and content.

        Use this on a specific result you've decided is worth indexing.
        """
        try:
            page = web_scrape(url)
        except Exception as exc:  # e.g. DNS failure, paywall, timeout — try another source.
            return f"Scrape failed for {url}: {exc}"
        fetched[page.url] = page
        return _format_page(page)

    @tool
    def crawl_site(url: str) -> str:
        """Crawl a small site starting at url and return each page's title and content.

        Use this instead of `scrape_url` when a single result looks like a docs/wiki
        site where multiple pages would be useful, rather than just the one page.
        """
        try:
            pages = web_crawl(url)
        except Exception as exc:
            return f"Crawl failed for {url}: {exc}"
        if not pages:
            return f"No pages found crawling {url}."
        for page in pages:
            fetched[page.url] = page
        return "\n\n---\n\n".join(_format_page(p) for p in pages)

    @tool
    def add_source(url: str, name: str | None = None) -> str:
        """Index a page you've already fetched (via `scrape_url` or `crawl_site`) as a source.

        Pass the exact url you fetched; its full content is indexed automatically. Call this
        once per page you've decided to keep.
        """
        page = fetched.get(url)
        if page is None:
            return f"No fetched content for {url!r} — call scrape_url or crawl_site on it first."
        try:
            source = store.add(name=name or page.title, content=page.content)
        except Exception as exc:
            return f"Failed to add source for {url}: {exc}"
        return f"Added source id={source.id} name={source.name!r} ({len(source.content)} chars) from {url}."

    return [search_web, scrape_url, crawl_site, add_source]


agent = create_agent(
    model="openai:gpt-4o-mini",
    system_prompt=SYSTEM_PROMPT,
    tools=make_tools_(store),
)


@dataclass
class ResearchResult:
    summary: str
    source_ids: list[str] = field(default_factory=list)


def _added_source_ids(messages) -> list[str]:
    ids: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "add_source":
            text = str(message.content)
            if text.startswith("Added source id="):
                ids.append(text.split("id=", 1)[1].split(" ", 1)[0])
    return ids


def research(topic: str) -> ResearchResult:
    state = agent.invoke({"messages": [{"role": "user", "content": topic}]})
    return ResearchResult(
        summary=state["messages"][-1].content,
        source_ids=_added_source_ids(state["messages"]),
    )
