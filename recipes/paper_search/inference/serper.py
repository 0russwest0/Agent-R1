"""Serper Google search support for paper search inference."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Optional

import httpx

from recipes.paper_search.env.http_retry import httpx_request_with_retry
from recipes.paper_search.env.paper_client import Paper, PaperSearchClient
from recipes.paper_search.inference.date_utils import parse_year_month_str

ARXIV_URL_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([a-z.-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?(?:[/?#].*)?$",
    re.IGNORECASE,
)


class ApiKeyPool:
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
            self.current_index = self.current_index % len(self.keys) if self.keys else 0

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.keys)


def serper_api_keys_from_env() -> list[str]:
    raw = os.getenv("PAPER_SEARCH_SERPER_API_KEYS", "").strip()
    if raw:
        return [key.strip() for key in raw.split(",") if key.strip()]
    single = os.getenv("SERPER_API_KEY", "").strip()
    return [single] if single else []


def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    match = ARXIV_URL_PATTERN.search(url.strip()) if url else None
    if not match:
        return None
    return re.sub(r"v\d+$", "", match.group(1).strip(), flags=re.IGNORECASE)


def build_google_search_query(query: str, *, from_month: Optional[str] = None, to_month: Optional[str] = None) -> str:
    parts = [query.strip(), "site:arxiv.org"]
    if from_month:
        year, month = parse_year_month_str(from_month)
        parts.append(f"after:{year:04d}-{month:02d}-01")
    if to_month:
        year, month = parse_year_month_str(to_month)
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        parts.append(f"before:{next_year:04d}-{next_month:02d}-01")
    return " ".join(part for part in parts if part)


async def search_google_via_serper(
    paper_client: PaperSearchClient,
    *,
    key_pool: ApiKeyPool,
    query: str,
    limit: int,
    from_month: Optional[str] = None,
    to_month: Optional[str] = None,
    fields: str = "title,abstract,year,authors,externalIds",
    search_url: str = "https://google.serper.dev/search",
) -> list[Paper]:
    initial_keys = key_pool.snapshot()
    if not initial_keys:
        raise ValueError("Serper search requires non-empty PAPER_SEARCH_SERPER_API_KEYS or SERPER_API_KEY")
    if limit <= 0:
        return []
    if limit > 10:
        raise ValueError("Serper supports at most num=10 organic hits per request")

    payload = {"q": build_google_search_query(query, from_month=from_month, to_month=to_month), "num": limit, "page": 1}
    attempted: set[str] = set()
    last_exc: Optional[BaseException] = None
    resp: Optional[httpx.Response] = None

    while len(attempted) < len(initial_keys):
        api_key = key_pool.get_next_key()
        if not api_key:
            break
        attempted.add(api_key)
        try:
            resp = await httpx_request_with_retry(
                paper_client.client,
                "POST",
                search_url,
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
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

    organic = resp.json().get("organic")
    if not isinstance(organic, list):
        return []

    paper_ids: list[str] = []
    seen: set[str] = set()
    for item in organic:
        if not isinstance(item, dict):
            continue
        paper_id = extract_arxiv_id_from_url(str(item.get("link") or ""))
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            paper_ids.append(paper_id)

    tasks = [paper_client.get_paper(paper_id, fields=fields) for paper_id in paper_ids]
    papers = await asyncio.gather(*tasks) if tasks else []
    return [paper for paper in papers if paper is not None]
