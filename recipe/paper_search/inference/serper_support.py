"""Serper (Google) search support for paper retrieval during inference only."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Optional

import httpx

from recipe.paper_search.http_retry import httpx_request_with_retry
from recipe.paper_search.inference.inference_date_utils import parse_year_month_str
from recipe.paper_search.utils import Paper, PaperSearchClient

_SERPER_SEARCH_URL = os.getenv("SERPER_SEARCH_URL", "https://google.serper.dev/search")
_ARXIV_URL_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([a-z.-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?(?:[/?#].*)?$",
    re.IGNORECASE,
)


class ApiKeyPool:
    """Rotate and retire bad Serper API keys."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = list(keys)
        self.current_index = 0
        self._lock = threading.Lock()

    def get_next_key(self) -> Optional[str]:
        with self._lock:
            if not self.keys:
                return None
            key = self.keys[self.current_index % len(self.keys)]
            self.current_index = (self.current_index + 1) % len(self.keys)
            return key

    def remove_key(self, key: str) -> None:
        with self._lock:
            if key not in self.keys:
                return
            self.keys.remove(key)
            if self.keys:
                self.current_index %= len(self.keys)
            else:
                self.current_index = 0

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.keys)


def _normalize_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip(), flags=re.IGNORECASE)


def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    """Best-effort arXiv id from a search result link."""
    if not url:
        return None
    match = _ARXIV_URL_PATTERN.search(url.strip())
    if not match:
        return None
    return _normalize_arxiv_id(match.group(1))


def build_google_search_query(
    query: str,
    *,
    from_month: Optional[str] = None,
    to_month: Optional[str] = None,
) -> str:
    """Build Serper ``q`` string with arXiv site filter and optional date bounds."""
    parts = [query.strip(), "site:arxiv.org"]
    if from_month:
        y, m = parse_year_month_str(from_month)
        parts.append(f"after:{y:04d}-{m:02d}-01")
    if to_month:
        y, m = parse_year_month_str(to_month)
        next_y, next_m = (y + 1, 1) if m == 12 else (y, m + 1)
        parts.append(f"before:{next_y:04d}-{next_m:02d}-01")
    return " ".join(part for part in parts if part)


def serper_api_keys_from_env() -> list[str]:
    """Load comma-separated keys from ``PAPER_SEARCH_SERPER_API_KEYS`` or single ``SERPER_API_KEY``."""
    raw = os.getenv("PAPER_SEARCH_SERPER_API_KEYS", "").strip()
    if raw:
        return [k.strip() for k in raw.split(",") if k.strip()]
    single = os.getenv("SERPER_API_KEY", "").strip()
    return [single] if single else []


async def search_google_via_serper(
    paper_client: PaperSearchClient,
    *,
    key_pool: ApiKeyPool,
    query: str,
    limit: int,
    from_month: Optional[str] = None,
    to_month: Optional[str] = None,
    fields: str = "title,abstract,year,authors,externalIds",
) -> list[Paper]:
    """Return hydrated ``Paper`` rows via Serper organic links + local ``/paper/{id}`` API."""
    initial_keys = key_pool.snapshot()
    if not initial_keys:
        raise ValueError("Serper search requires non-empty PAPER_SEARCH_SERPER_API_KEYS or SERPER_API_KEY")
    if limit <= 0:
        return []
    if limit > 10:
        raise ValueError("Serper supports at most num=10 organic hits per request")

    search_query = build_google_search_query(query, from_month=from_month, to_month=to_month)
    payload = {"q": search_query, "num": limit, "page": 1}
    attempted: set[str] = set()
    last_exc: Optional[BaseException] = None
    resp: Optional[httpx.Response] = None
    max_attempts = len(initial_keys)

    while len(attempted) < max_attempts:
        api_key = key_pool.get_next_key()
        if not api_key:
            break
        attempted.add(api_key)
        try:
            resp = await httpx_request_with_retry(
                paper_client.client,
                "POST",
                _SERPER_SEARCH_URL,
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                content=json.dumps(payload),
                max_retries=2,
                retry_status_codes={429, 500, 502, 503, 504},
                retry_exceptions=(httpx.RequestError, httpx.TimeoutException),
            )
            resp.raise_for_status()
            break
        except Exception as exc:
            last_exc = exc
            key_pool.remove_key(api_key)
            resp = None

    if resp is None:
        if last_exc is not None:
            raise RuntimeError("All Serper API keys failed") from last_exc
        raise ValueError("Serper API key pool is empty")

    result = resp.json()
    organic = result.get("organic")
    if not isinstance(organic, list):
        return []

    paper_ids: list[str] = []
    seen: set[str] = set()
    for item in organic:
        if not isinstance(item, dict):
            continue
        pid = extract_arxiv_id_from_url(str(item.get("link") or ""))
        if not pid or pid in seen:
            continue
        seen.add(pid)
        paper_ids.append(pid)

    tasks = [paper_client.get_paper(pid, fields=fields) for pid in paper_ids]
    papers = await asyncio.gather(*tasks) if tasks else []
    return [p for p in papers if p is not None]
