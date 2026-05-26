from __future__ import annotations

import logging
import os
import time
from typing import Any
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from recipe.webshop.env.data import load_product_index
from recipe.webshop.env.engine import WebShopEngine
from recipe.webshop.env.full_data import load_full_product_index
from recipe.webshop.env.schemas import ResetRequest, ResetResponse, StepRequest, StepResponse


TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def _log_level() -> int:
    name = os.getenv("WEBSHOP_ENV_APP_LOG_LEVEL", os.getenv("WEBSHOP_ENV_LOG_LEVEL", "INFO")).upper()
    return getattr(logging, name, logging.INFO)


logging.basicConfig(
    level=_log_level(),
    format="%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s",
)
logger = logging.getLogger("webshop.env")
logger.setLevel(_log_level())


def _action_type(action: str) -> str:
    return (action or "").strip().split("[", 1)[0].lower()


def _click_result_count(actions: list[str]) -> int:
    return sum(
        1
        for action in actions
        if action.startswith("click[") and action not in {"click[Next >]", "click[< Prev]"}
    )


def _should_log_step(action: str) -> bool:
    return _env_flag("WEBSHOP_ENV_LOG_STEPS") or (
        _env_flag("WEBSHOP_ENV_LOG_SEARCH") and _action_type(action) == "search"
    )


def _safe_info(info: dict[str, Any]) -> dict[str, Any]:
    actions = info.get("available_actions") or []
    if not isinstance(actions, list):
        actions = []
    return {
        "error": info.get("error"),
        "success": info.get("success"),
        "task_score": info.get("task_score"),
        "final_reward": info.get("final_reward"),
        "selected_asin": info.get("selected_asin"),
        "target_asin": info.get("target_asin"),
        "available_actions": len(actions),
        "click_results": _click_result_count(actions),
    }


@lru_cache(maxsize=1)
def get_engine() -> WebShopEngine:
    dataset_mode = os.getenv("WEBSHOP_DATASET_MODE", "small").lower()
    data_dir = os.getenv("WEBSHOP_DATA_DIR", "webshop_data")
    index_dir = os.getenv("WEBSHOP_INDEX_DIR", "data/webshop/index")
    search_top_k = int(os.getenv("WEBSHOP_SEARCH_TOP_K", "50" if dataset_mode == "full" else "10"))
    if dataset_mode == "full":
        return WebShopEngine(
            load_full_product_index(index_dir=index_dir, search_top_k=search_top_k),
            search_top_k=search_top_k,
        )
    return WebShopEngine(load_product_index(data_dir=data_dir, index_dir=index_dir), search_top_k=search_top_k)


app = FastAPI(title="WebShop Environment", version="0.2.0")


@app.get("/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "pid": os.getpid(),
        "dataset_mode": os.getenv("WEBSHOP_DATASET_MODE", "small").lower(),
        "num_products": engine.index.num_products,
        "num_goals": len(engine.index.goals),
        "search_top_k": engine.search_top_k,
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    try:
        observation, state, info = get_engine().reset(req.goal_index)
    except Exception as exc:
        logger.exception("reset failed goal_index=%s", req.goal_index)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if _env_flag("WEBSHOP_ENV_LOG_RESETS") or _env_flag("WEBSHOP_ENV_LOG_STEPS"):
        actions = info.get("available_actions") or []
        logger.info(
            "reset goal_index=%s available_actions=%s instruction=%r",
            req.goal_index,
            len(actions) if isinstance(actions, list) else 0,
            info.get("instruction"),
        )
    return ResetResponse(observation=observation, env_state=state, info=info)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    start_time = time.perf_counter()
    try:
        response = get_engine().step(req.goal_index, req.env_state, req.action)
    except Exception as exc:
        logger.exception("step failed goal_index=%s action=%r", req.goal_index, req.action)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if _should_log_step(req.action):
        info = _safe_info(response.info or {})
        logger.info(
            (
                "step goal_index=%s action=%r page=%s reward=%.4f done=%s "
                "error=%s success=%s task_score=%s final_reward=%s selected_asin=%s "
                "target_asin=%s available_actions=%s click_results=%s latency_ms=%.1f"
            ),
            req.goal_index,
            req.action,
            response.env_state.page_type,
            response.reward,
            response.done,
            info["error"],
            info["success"],
            info["task_score"],
            info["final_reward"],
            info["selected_asin"],
            info["target_asin"],
            info["available_actions"],
            info["click_results"],
            (time.perf_counter() - start_time) * 1000,
        )
    return response
