import logging
import os
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, List, Optional
import threading

import faiss
import torch
from FlagEmbedding import FlagAutoModel
import numpy as np

# Retrieval corpus root: defaults to <repo>/data/corpus/hotpotqa_corpus
# (index.bin + hpqa_corpus.jsonl). Override with HOTPOTQA_CORPUS_DATA_ROOT.
_STEPPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HOTPOTQA_CORPUS_DATA_ROOT = _STEPPO_ROOT / "data" / "corpus" / "hotpotqa_corpus"


def resolve_hotpotqa_corpus_data_root(corpus_data_dir: Optional[str] = None) -> Path:
    """Resolve the directory that stores HotpotQA retrieval files.

    Args:
        corpus_data_dir: Optional explicit directory from recipe YAML.

    Returns:
        Absolute path containing ``index.bin`` and ``hpqa_corpus.jsonl``.
    """
    raw = (
        corpus_data_dir
        or os.environ.get("HOTPOTQA_CORPUS_DATA_ROOT")
        or str(_DEFAULT_HOTPOTQA_CORPUS_DATA_ROOT)
    )
    return Path(raw).expanduser().resolve()


HOTPOTQA_CORPUS_DATA_ROOT = resolve_hotpotqa_corpus_data_root()
# Backward-compatible alias for older smoke tests/imports. New code should use HOTPOTQA_CORPUS_DATA_ROOT.
HOTPOTQA_DATA_ROOT = HOTPOTQA_CORPUS_DATA_ROOT
HOTPOTQA_INDEX_BIN = HOTPOTQA_CORPUS_DATA_ROOT / "index.bin"
# Passage text for decoding search hits; must match hpqa_corpus.jsonl used when building the index.
HOTPOTQA_CORPUS_JSONL = HOTPOTQA_CORPUS_DATA_ROOT / "hpqa_corpus.jsonl"
# hpqa_corpus.npy is only produced by process_hotpotqa for embedding cache; not loaded at runtime.

# Default BGE checkpoint (local dir or Hugging Face hub id). Override via YAML `embedding_model_name` or
# `HOTPOTQA_EMBEDDING_MODEL` for portability.
DEFAULT_HOTPOTQA_EMBEDDING_MODEL = (
    os.environ.get("HOTPOTQA_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5").strip()
    or "BAAI/bge-large-en-v1.5"
)

logger = logging.getLogger(__name__)


def default_hotpotqa_embedding_device() -> str:
    """Read desired device string from env (e.g. cuda:N); see `normalize_embedding_device` for actual resolution."""
    return os.environ.get("HOTPOTQA_EMBEDDING_DEVICE", "cpu").strip() or "cpu"


def normalize_embedding_device(requested: str) -> str:
    """
    Resolve configured device to a FlagEmbedding-compatible `devices` string.

    Ray AgentFlowWorker processes often see **no GPUs** (no CUDA device assigned or visibility
    masked). Passing `cuda:*` then raises "no CUDA GPUs are available" — we fall back to `cpu`
    with a warning.
    """
    dev = (requested or "cpu").strip() or "cpu"
    if dev.lower() == "cpu":
        return "cpu"
    dl = dev.lower()
    if dl.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "embedding device %r requested but torch.cuda.is_available() is False; "
                "using cpu (typical for Ray agent workers without GPU allocation).",
                dev,
            )
            return "cpu"
        if ":" in dev:
            try:
                idx = int(dev.split(":")[-1])
            except ValueError:
                return dev
            if idx < 0 or idx >= torch.cuda.device_count():
                logger.warning(
                    "embedding device %r invalid for torch.cuda.device_count()=%s; using cpu.",
                    dev,
                    torch.cuda.device_count(),
                )
                return "cpu"
        return dev
    return dev


def resolve_hotpotqa_embedding_devices(
    embedding_devices: Optional[str],
    agent_flow_worker_index: Optional[int],
) -> Optional[str]:
    """Pick BGE `devices` for HotpotQAAgentFlow before constructing `HotpotQASearchToolLegacy`.

    Args:
        embedding_devices: Non-empty value from YAML ``embedding_devices``; if set, it wins.
        agent_flow_worker_index: Index of the Ray ``AgentFlowWorker`` (0 .. N-1).

    Returns:
        Device string for FlagEmbedding, or ``None`` to use ``HotpotQASearchToolLegacy`` defaults
        (``HOTPOTQA_EMBEDDING_DEVICE`` env, then ``cpu``).

    If environment variable ``HOTPOTQA_EMBEDDING_PER_WORKER_GPU`` is truthy (``1``/``true``/``yes``)
    and ``agent_flow_worker_index`` is not ``None``, uses ``cuda:{index}`` (after
    :func:`normalize_embedding_device`), so each of 4 workers can colocate BGE on its training GPU.
    """
    if embedding_devices is not None:
        s = str(embedding_devices).strip()
        if s and s.lower() != "null":
            return s
    flag = os.environ.get("HOTPOTQA_EMBEDDING_PER_WORKER_GPU", "").strip().lower()
    if flag in ("1", "true", "yes", "on") and agent_flow_worker_index is not None:
        raw = f"cuda:{int(agent_flow_worker_index)}"
        return normalize_embedding_device(raw)
    return None


@dataclass
class Passage:
    pid: int
    title: str
    text: str
    score: float = 0.0


@dataclass
class PassagePool:
    passages: List[Passage] = field(default_factory=list)

    def has_passage(self, pid: int) -> bool:
        return any(p.pid == pid for p in self.passages)

    def add_passage(self, passage: Passage) -> None:
        # Dedupe by passage text; pid is the index in-pool (avoids fixed 0~4 ids across searches dropping hits).
        if any(p.text == passage.text for p in self.passages):
            return
        pid = len(self.passages)
        self.passages.append(
            Passage(pid=pid, title=passage.title, text=passage.text, score=passage.score)
        )

    @property
    def passage_list(self) -> str:
        if not self.passages:
            return "None"
        lines = []
        for i, p in enumerate(self.passages, start=1):
            snippet = p.text[:512].replace("\n", " ")
            lines.append(f"[{i}] (id={p.pid}) {p.title}: {snippet}")
        return "\n".join(lines)


class HotpotQASearchToolLegacy:
    """
    Local HotpotQA FAISS retrieval; behavior matches the production-validated implementation in
    `Agent-R1-legacy/agent_r1/tool/tools/search_tool.py` (SearchTool).

    Extensions vs. upstream (success-path semantics unchanged):
    - Data layout: `HOTPOTQA_CORPUS_DATA_ROOT` + `index.bin` / `hpqa_corpus.jsonl`
    - Process-wide shared index/corpus/model to avoid reloading every trajectory
    - `_format_results` bounds-checks ids (legacy direct indexing could raise IndexError)
    - If `encode_queries` returns torch.Tensor, converts via `.cpu().numpy()` for FAISS CPU index I/O

    Upstream often uses `FlagAutoModel.from_finetuned(..., devices="cpu")`; CPU encode is recommended
    for training. If `HOTPOTQA_EMBEDDING_DEVICE=cuda:*`, call from the same thread (see hotpotqa_agent_flow).
    """

    _shared_lock = threading.RLock()
    _shared_key: Optional[str] = None
    _shared_index: Optional[faiss.Index] = None
    _shared_corpus: Optional[list[str]] = None
    _shared_model: Optional[FlagAutoModel] = None

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_HOTPOTQA_EMBEDDING_MODEL,
        query_instruction: str = "Represent this sentence for searching relevant passages: ",
        embedding_devices: Optional[str] = None,
        corpus_data_dir: Optional[str] = None,
    ) -> None:
        self.data_dir = resolve_hotpotqa_corpus_data_root(corpus_data_dir)
        self.index_path = self.data_dir / "index.bin"
        self.corpus_path = self.data_dir / "hpqa_corpus.jsonl"
        self.embedding_model_name = embedding_model_name
        self.query_instruction = query_instruction
        raw = (embedding_devices if embedding_devices is not None else default_hotpotqa_embedding_device()).strip() or "cpu"
        self.embedding_devices = normalize_embedding_device(raw)
        if self.embedding_devices != raw:
            logger.info("HotpotQASearchToolLegacy: effective embedding_devices=%r (from requested %r)", self.embedding_devices, raw)

        self._index: Optional[faiss.Index] = None
        self._corpus: list[str] = []
        self._model: Optional[FlagAutoModel] = None
        self._ensure_loaded()

    def __enter__(self):
        self._ensure_loaded()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ensure_loaded(self) -> None:
        cache_key = f"{self.data_dir}|{self.embedding_devices}|{self.embedding_model_name}"
        with self.__class__._shared_lock:
            if (
                self.__class__._shared_key != cache_key
                or self.__class__._shared_index is None
                or self.__class__._shared_corpus is None
                or self.__class__._shared_model is None
            ):
                if not self.index_path.exists():
                    raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
                if not self.corpus_path.exists():
                    raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

                logger.info("HotpotQASearchToolLegacy: loading FAISS index from %s", self.index_path)
                index = faiss.read_index(str(self.index_path))
                logger.info(
                    "HotpotQASearchToolLegacy: loading corpus jsonl from %s (may take several minutes)",
                    self.corpus_path,
                )
                corpus: list[str] = []
                with self.corpus_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        title = str(rec.get("title", ""))
                        text = str(rec.get("text", ""))
                        corpus.append(f"{title} {text}".strip())

                logger.info(
                    "HotpotQASearchToolLegacy: loading FlagEmbedding model=%s devices=%s",
                    self.embedding_model_name,
                    self.embedding_devices,
                )
                model = FlagAutoModel.from_finetuned(
                    self.embedding_model_name,
                    query_instruction_for_retrieval=self.query_instruction,
                    devices=self.embedding_devices,
                )
                self.__class__._shared_key = cache_key
                self.__class__._shared_index = index
                self.__class__._shared_corpus = corpus
                self.__class__._shared_model = model

                if int(index.ntotal) != len(corpus):
                    logger.warning(
                        "FAISS index.ntotal (%s) != hpqa_corpus.jsonl rows (%s). "
                        "Ids from search may be out of range and passages will be empty; "
                        "rebuild index.bin with the same jsonl or fix the corpus file.",
                        int(index.ntotal),
                        len(corpus),
                    )

            self._index = self.__class__._shared_index
            self._corpus = self.__class__._shared_corpus or []
            self._model = self.__class__._shared_model

    def close(self) -> None:
        # Keep shared model/index alive for whole training process.
        # This matches legacy behavior where SearchTool is initialized once and reused.
        self._index = self.__class__._shared_index
        self._corpus = self.__class__._shared_corpus or []
        self._model = self.__class__._shared_model

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            query = str(args["query"])
            embeddings = self._encode_queries([query])
            assert self._index is not None
            _, ids = self._index.search(embeddings, 5)
            result_str = self._format_results(ids[0])
            return {"content": result_str, "success": True}
        except Exception as e:
            return {"content": str(e), "success": False}

    def batch_execute(self, args_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Same contract as Agent-R1-legacy `SearchTool.batch_execute`: one encode + search; on failure each row gets str(e)."""
        if not args_list:
            return []
        try:
            queries = [str(x["query"]) for x in args_list]
            embeddings = self._encode_queries(queries)
            assert self._index is not None
            _, ids = self._index.search(embeddings, 5)
            results_str = [self._format_results(ids[i]) for i in range(len(ids))]
            return [{"content": result_str, "success": True} for result_str in results_str]
        except Exception as e:
            logger.warning(
                "HotpotQASearchToolLegacy.batch_execute failed (%s queries): %s",
                len(args_list),
                e,
                exc_info=True,
            )
            return [{"content": str(e), "success": False} for _ in args_list]

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        self._ensure_loaded()
        with self.__class__._shared_lock:
            assert self._model is not None
            out = self._model.encode_queries(queries)
        # FAISS CPU Index::search expects host float32 ndarray; BGE on GPU may return torch.Tensor.
        if torch.is_tensor(out):
            out = out.detach().float().cpu().numpy()
        arr = np.asarray(out, dtype=np.float32)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr

    def _format_results(self, results) -> str:
        results_list: list[str] = []
        row_ids = [int(x) for x in np.asarray(results, dtype=np.int64).reshape(-1)]
        for result in row_ids:
            if result < 0 or result >= len(self._corpus):
                continue
            results_list.append(self._corpus[result])
        if (
            not results_list
            and self._corpus
            and row_ids
            and max(row_ids) >= len(self._corpus)
        ):
            logger.warning(
                "FAISS returned ids %s but corpus length is %s; dropping all hits.",
                row_ids[:10],
                len(self._corpus),
            )
        return json.dumps({"results": results_list}, ensure_ascii=False)


def parse_legacy_tool_result(content: str) -> list[Passage]:
    """Parse legacy `{"results":[...]}` tool content into Passage list."""
    passages: list[Passage] = []
    try:
        payload = json.loads(content)
        results = payload.get("results", [])
        for idx, text in enumerate(results):
            text_str = str(text)
            passages.append(Passage(pid=idx, title="", text=text_str, score=0.0))
    except Exception:
        return []
    return passages
