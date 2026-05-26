"""Paper retrieval for inference: local hybrid API and/or Serper Google search (inference-only)."""

from __future__ import annotations

from typing import Optional

from recipe.paper_search.utils import Paper, PaperSearchClient

from recipe.paper_search.inference.serper_support import ApiKeyPool, serper_api_keys_from_env, search_google_via_serper


class InferencePaperClient:
    """Delegates ``get_paper`` / citations / references to ``PaperSearchClient``; ``search`` may use Serper."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        search_source: str = "local_db",
        paper_from_month: Optional[str] = None,
        paper_to_month: Optional[str] = None,
        timeout: float = 30.0,
        serper_keys: Optional[list[str]] = None,
    ) -> None:
        """Initialize client.

        Args:
            base_url: Paper Search HTTP API (hydration + local search), same as ``PAPER_SEARCH_BASE_URL``.
            search_source: ``local_db`` or ``google`` (Serper + hydration via same API).
            paper_from_month: Optional ``YYYY-MM`` lower bound (Serper ``after:``).
            paper_to_month: Optional ``YYYY-MM`` upper bound (Serper ``before:``).
            timeout: HTTP timeout seconds.
            serper_keys: Optional explicit key list; default reads env (see ``serper_api_keys_from_env``).
        """
        src = (search_source or "local_db").strip().lower()
        if src not in {"local_db", "google"}:
            raise ValueError("search_source must be 'local_db' or 'google'")
        self.search_source = src
        self.paper_from_month = paper_from_month
        self.paper_to_month = paper_to_month
        self._paper = PaperSearchClient(base_url=base_url, timeout=timeout)
        self._serper_pool: Optional[ApiKeyPool] = None
        if self.search_source == "google":
            keys = serper_keys if serper_keys is not None else serper_api_keys_from_env()
            if not keys:
                raise ValueError(
                    "search_source=google requires Serper API keys: set "
                    "PAPER_SEARCH_SERPER_API_KEYS (comma-separated) or SERPER_API_KEY"
                )
            self._serper_pool = ApiKeyPool(keys)

    async def search(
        self,
        query: str,
        limit: int = 10,
        *,
        year: Optional[str] = None,
    ) -> list[Paper]:
        """Run local ``/paper/search`` or Serper + ``get_paper`` per arXiv id."""
        if self.search_source == "google":
            assert self._serper_pool is not None
            cap = min(limit, 10)
            return await search_google_via_serper(
                self._paper,
                key_pool=self._serper_pool,
                query=query,
                limit=cap,
                from_month=self.paper_from_month,
                to_month=self.paper_to_month,
            )
        return await self._paper.search(query=query, limit=limit, year=year)

    async def get_paper(self, paper_id: str, fields: str = "title,abstract,year,authors,externalIds") -> Optional[Paper]:
        """Fetch a single paper record."""
        return await self._paper.get_paper(paper_id, fields=fields)

    async def get_citations(
        self, paper_id: str, limit: int = 50, fields: str = "title,abstract,year,authors,externalIds"
    ) -> list[Paper]:
        """List citing papers from the hybrid API."""
        return await self._paper.get_citations(paper_id, limit=limit, fields=fields)

    async def get_references(
        self, paper_id: str, limit: int = 50, fields: str = "title,abstract,year,authors,externalIds"
    ) -> list[Paper]:
        """List referenced papers from the hybrid API."""
        return await self._paper.get_references(paper_id, limit=limit, fields=fields)

    async def close(self) -> None:
        """Close HTTP client."""
        await self._paper.close()
