import asyncio
import logging
import os
from functools import total_ordering
from typing import Any, Optional

import httpx
from pydantic import BaseModel
from sortedcontainers import SortedList

from .http_retry import httpx_request_with_retry

logger = logging.getLogger(__file__)
DEFAULT_PAPER_FIELDS = "title,abstract,year,authors,externalIds"


def _format_authors(authors: Any) -> str:
    if not authors:
        return ""
    if isinstance(authors, str):
        return authors
    if isinstance(authors, list):
        names: list[str] = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name")
                if name:
                    names.append(str(name))
            elif author:
                names.append(str(author))
        return ", ".join(names)
    return str(authors)


class Paper(BaseModel):
    paper_id: str
    raw_paper_id: str = ""
    arxiv_id: str = ""
    title: str
    abstract: str
    authors: str = ""
    year: Optional[int] = None
    score: float = 0.0


@total_ordering
class PaperPoolEntry(BaseModel):
    paper: Paper
    source: str
    origin: str
    score: float
    expand: bool = False

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PaperPoolEntry):
            return NotImplemented
        if self.score != other.score:
            return self.score < other.score
        return self.paper.paper_id < other.paper.paper_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PaperPoolEntry):
            return NotImplemented
        return self.score == other.score and self.paper.paper_id == other.paper.paper_id

    def __hash__(self) -> int:
        return hash(self.paper.paper_id)


class PaperPool:
    def __init__(self, max_size: int = 20, threshold: float = 0.0, max_abstract_words: int = 400):
        self.papers: dict[str, PaperPoolEntry] = {}
        self.ranked_papers: SortedList[PaperPoolEntry] = SortedList()
        self.max_size = max_size
        self.threshold = threshold
        self.max_abstract_words = max_abstract_words

    def add_paper(self, paper: Paper, source: str, origin: str, score: float) -> None:
        if paper.paper_id in self.papers:
            return

        paper_pool_entry = PaperPoolEntry(paper=paper, source=source, origin=origin, score=score)
        self.papers[paper.paper_id] = paper_pool_entry
        self.ranked_papers.add(paper_pool_entry)

    def get_paper(self, paper_id: str) -> Optional[PaperPoolEntry]:
        return self.papers.get(paper_id)

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self.papers

    @property
    def paper_list(self) -> str:
        if not self.papers:
            return "No papers in the pool."

        expanded_entries = [e for e in self.ranked_papers if e.expand and e.score >= self.threshold]
        unexpanded_entries = [e for e in self.ranked_papers if not e.expand and e.score >= self.threshold]

        expanded_entries.reverse()
        unexpanded_entries.reverse()

        half_size = self.max_size // 2
        top_expanded = expanded_entries[:half_size]
        top_unexpanded = unexpanded_entries[:half_size]

        display_entries = top_expanded + top_unexpanded
        display_entries.sort(key=lambda x: x.score, reverse=True)

        if not display_entries:
            return "No relevant papers found above threshold."

        description = (
            "Paper Pool Status:\n"
            "- [EXP]: Paper has been expanded already.\n"
            "- [NEW]: New paper found via search or expansion.\n"
            "- Format: [paper_id] (score) [STATUS] Title\n"
        )

        lines = [description]
        for entry in display_entries:
            paper = entry.paper
            status_tag = "[EXP]" if entry.expand else "[NEW]"
            abstract = paper.abstract
            words = abstract.split()
            if len(words) > self.max_abstract_words:
                abstract = " ".join(words[: self.max_abstract_words]) + "..."

            entry_str = f"[{paper.paper_id}] ({entry.score:.2f}) {status_tag} {paper.title}\nAbstract: {abstract}"
            lines.append(entry_str)

        return "\n\n".join(lines)


class PaperSearchClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        *,
        max_concurrency: Optional[int] = 16,
        max_detail_concurrency: Optional[int] = 16,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
    ):
        self.base_url = (
            base_url
            or os.getenv("PAPER_SEARCH_BASE_URL")
            or "http://localhost:4000"
        ).rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0, pool=60.0),
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=128),
        )
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None
        self._detail_semaphore = (
            asyncio.Semaphore(max_detail_concurrency) if max_detail_concurrency and max_detail_concurrency > 0 else None
        )
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

    async def _request(
        self, method: str, url: str, *, semaphore: Optional[asyncio.Semaphore] = None, **kwargs: Any
    ) -> httpx.Response:
        sem = self._semaphore if semaphore is None else semaphore
        total_attempts = self.max_retries + 1
        try:
            resp = await httpx_request_with_retry(
                self.client,
                method,
                url,
                semaphore=sem,
                max_retries=self.max_retries,
                retry_status_codes={503},
                retry_exceptions=(httpx.RequestError, httpx.TimeoutException),
                initial_backoff=self.initial_backoff,
                max_backoff=self.max_backoff,
                **kwargs,
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            logger.warning(
                "Request failed after %d retries (%d total attempts): %s %s. Last error: %r",
                self.max_retries,
                total_attempts,
                method,
                url,
                exc,
            )
            raise

        if resp.status_code == 503:
            logger.warning(
                "Request returned 503 after %d retries (%d total attempts): %s %s",
                self.max_retries,
                total_attempts,
                method,
                url,
            )
        return resp

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "PaperSearchClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @staticmethod
    def _record_to_paper(data: dict[str, Any]) -> Paper:
        raw_paper_id = str(data.get("paperId") or data.get("paper_id") or "")
        external_ids = data.get("externalIds") or {}
        arxiv_id = str(data.get("arxiv_id") or external_ids.get("ArXiv") or "")
        paper_id = arxiv_id or raw_paper_id
        return Paper(
            paper_id=paper_id,
            raw_paper_id=raw_paper_id,
            arxiv_id=arxiv_id,
            title=str(data.get("title") or ""),
            abstract=str(data.get("abstract") or ""),
            authors=_format_authors(data.get("authors")),
            year=data.get("year"),
            score=float(data.get("score", 0.0) or 0.0),
        )

    @staticmethod
    def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = payload.get("data")
        return data if isinstance(data, list) else []

    async def search(
        self,
        query: str,
        limit: int = 10,
        *,
        year: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> list[Paper]:
        params: dict[str, Any] = {"query": query, "limit": limit}
        if year:
            params["year"] = year
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count
        if fields:
            params["fields"] = fields

        resp = await self._request("GET", "/paper/search", params=params)
        resp.raise_for_status()
        payload = resp.json()
        return [self._record_to_paper(item) for item in self._extract_items(payload)]

    async def get_paper(self, paper_id: str, fields: str = DEFAULT_PAPER_FIELDS) -> Optional[Paper]:
        params = {"fields": fields} if fields else None
        resp = await self._request("GET", f"/paper/{paper_id}", params=params, semaphore=self._detail_semaphore)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not data:
            return None
        return self._record_to_paper(data)

    async def get_citations(
        self, paper_id: str, limit: int = 50, fields: str = DEFAULT_PAPER_FIELDS
    ) -> list[Paper]:
        params: dict[str, Any] = {"limit": limit}
        if fields:
            params["fields"] = fields
        resp = await self._request("GET", f"/paper/{paper_id}/citations", params=params)
        resp.raise_for_status()
        payload = resp.json()
        items = self._extract_items(payload)
        papers: list[Paper] = []
        for item in items:
            citing_paper = item.get("citingPaper")
            if isinstance(citing_paper, dict):
                papers.append(self._record_to_paper(citing_paper))
        return papers

    async def get_references(
        self, paper_id: str, limit: int = 50, fields: str = DEFAULT_PAPER_FIELDS
    ) -> list[Paper]:
        if limit < 0:
            limit = 99

        params: dict[str, Any] = {"limit": limit}
        if fields:
            params["fields"] = fields
        resp = await self._request("GET", f"/paper/{paper_id}/references", params=params)
        resp.raise_for_status()
        payload = resp.json()
        items = self._extract_items(payload)
        papers: list[Paper] = []
        for item in items:
            cited_paper = item.get("citedPaper")
            if isinstance(cited_paper, dict):
                papers.append(self._record_to_paper(cited_paper))
        return papers


class SelectorClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 30.0,
        *,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
    ):
        self.base_url = (base_url or os.getenv("PAPERSEARCH_SELECTOR_BASE_URL") or "http://localhost:8000").rstrip("/")
        self.model_name = (
            model_name
            or os.getenv("PAPERSEARCH_SELECTOR_MODEL_NAME")
            or os.getenv("PAPERSEARCH_SELECTOR_MODEL_PATH")
            or ""
        )
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0, pool=60.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "SelectorClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def classify(self, prompt: str) -> float:
        payload: dict[str, Any] = {"input": prompt, "activation": False}
        if self.model_name:
            payload["model"] = self.model_name

        resp = await httpx_request_with_retry(
            self.client,
            "POST",
            "/classify",
            json=payload,
            max_retries=self.max_retries,
            retry_status_codes={429, 500, 502, 503, 504},
            retry_exceptions=(httpx.RequestError, httpx.TimeoutException),
            initial_backoff=self.initial_backoff,
            max_backoff=self.max_backoff,
        )
        resp.raise_for_status()
        data = resp.json().get("data")
        if not isinstance(data, list) or not data:
            raise ValueError(f"Invalid selector response payload: {resp.text[:512]}")

        last_item = data[-1]
        probs = last_item.get("probs") if isinstance(last_item, dict) else None
        if not isinstance(probs, list) or not probs:
            raise ValueError(f"Invalid selector response probs: {resp.text[:512]}")

        return float(probs[-1])
