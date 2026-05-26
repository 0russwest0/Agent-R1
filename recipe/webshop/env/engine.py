from __future__ import annotations

import re
from typing import Any

from recipe.webshop.env.data import ProductIndex, normalize_options, normalize_text, product_attributes_text, product_price
from recipe.webshop.env.schemas import EnvState, StepResponse


ACTION_RE = re.compile(r"^\s*(search|click)\[(.*)\]\s*$", re.IGNORECASE | re.DOTALL)
PRODUCT_WINDOW = 10
NEXT_PAGE = "next >"
PREV_PAGE = "< prev"


def parse_action(action: str) -> tuple[str, str] | None:
    match = ACTION_RE.match(action or "")
    if not match:
        return None
    return match.group(1).lower(), normalize_text(match.group(2))


def _short(text: Any, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def render_home(goal: dict[str, Any]) -> str:
    return (
        "Amazon Shopping Game\n"
        f"Instruction: {goal['instruction']}\n"
        "You may search for products with search[query]."
    )


def _page_results(results: list[dict[str, Any]], page_num: int) -> list[dict[str, Any]]:
    page_num = max(0, int(page_num or 0))
    return results[page_num * PRODUCT_WINDOW : (page_num + 1) * PRODUCT_WINDOW]


def render_search_results(goal: dict[str, Any], query: str, results: list[dict[str, Any]], page_num: int = 0) -> str:
    lines = [
        "Search Results",
        f"Instruction: {goal['instruction']}",
        f"Query: {query}",
        f"Page: {page_num + 1}",
        "Clickable products:",
    ]
    if not results:
        lines.append("No results found.")
        return "\n".join(lines)
    for i, result in enumerate(_page_results(results, page_num), start=page_num * PRODUCT_WINDOW + 1):
        item = result["item"]
        lines.append(
            f"[{i}] {item.get('asin')} | {_short(item.get('name'), 180)} | "
            f"{item.get('pricing') or '$100.00'} | {item.get('query') or item.get('category')}"
        )
    lines.append("Click a product ASIN, or search again.")
    return "\n".join(lines)


def render_item(goal: dict[str, Any], item: dict[str, Any], selected_options: dict[str, str]) -> str:
    options = normalize_options(item.get("customization_options"))
    lines = [
        "Product Page",
        f"Instruction: {goal['instruction']}",
        f"ASIN: {item.get('asin')}",
        f"Title: {_short(item.get('name'), 260)}",
        f"Price: {item.get('pricing') or '$100.00'}",
        f"Category: {item.get('product_category') or item.get('category')}",
    ]
    small_description = item.get("small_description")
    if isinstance(small_description, list) and small_description:
        lines.append("Summary:")
        for bullet in small_description[:3]:
            lines.append(f"- {_short(bullet, 220)}")
    elif small_description:
        lines.append(f"Summary: {_short(small_description, 260)}")

    if options:
        lines.append("Options:")
        for option_name, choices in options.items():
            current = selected_options.get(option_name, "not selected")
            values = ", ".join(choice["value"] for choice in choices[:30])
            if len(choices) > 30:
                values += f", ... ({len(choices) - 30} more)"
            lines.append(f"- {option_name} (selected: {current}): {values}")
    else:
        lines.append("Options: none")

    lines.append("Clickable controls: Description, Features, Reviews, Buy Now, Back to Search")
    return "\n".join(lines)


def render_subpage(goal: dict[str, Any], item: dict[str, Any], subpage: str) -> str:
    subpage_norm = normalize_text(subpage)
    lines = [
        subpage.title(),
        f"Instruction: {goal['instruction']}",
        f"ASIN: {item.get('asin')}",
        f"Title: {_short(item.get('name'), 260)}",
    ]
    if subpage_norm == "description":
        lines.append(_short(item.get("full_description") or item.get("small_description") or "No description.", 2000))
    elif subpage_norm == "features":
        small_description = item.get("small_description")
        if isinstance(small_description, list):
            lines.extend(f"- {_short(x, 260)}" for x in small_description[:10])
        else:
            lines.append(_short(small_description or "No features.", 2000))
    elif subpage_norm == "reviews":
        lines.append(f"Average rating: {item.get('average_rating') or 'N.A.'}")
        lines.append(f"Total reviews: {item.get('total_reviews') or 'N.A.'}")
    lines.append("Clickable controls: Back to Item, Back to Search, Buy Now")
    return "\n".join(lines)


def _option_lookup(item: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for option_name, choices in normalize_options(item.get("customization_options")).items():
        for choice in choices:
            lookup[choice["value"]] = option_name
    return lookup


def _get_product(index: Any, asin: str | None) -> dict[str, Any] | None:
    if hasattr(index, "get_product"):
        return index.get_product(asin or "")
    return getattr(index, "asin_to_product", {}).get(asin or "")


def _has_product(index: Any, asin: str | None) -> bool:
    if hasattr(index, "has_product"):
        return bool(index.has_product(asin or ""))
    return (asin or "") in getattr(index, "asin_to_product", {})


def _fuzzy_score(left: str, right: str) -> int:
    try:
        from thefuzz import fuzz

        return int(fuzz.token_set_ratio(left, right))
    except Exception:
        left_set = set(normalize_text(left).split())
        right_set = set(normalize_text(right).split())
        if not left_set or not right_set:
            return 0
        return int(100 * len(left_set & right_set) / len(right_set))


def _normalize_color(value: str) -> str:
    color_set = [
        "beige", "black", "blue", "brown", "burgundy", "camel", "charcoal", "cream", "dark", "gold",
        "gray", "green", "grey", "ivory", "khaki", "light", "navy", "orange", "pink", "purple",
        "red", "silver", "tan", "taupe", "teal", "white", "wine", "yellow",
    ]
    value = normalize_text(value)
    for color in color_set:
        if color in value:
            return color
    return value


def _title_nouns(text: str) -> list[str]:
    text = normalize_text(text)
    try:
        import spacy

        if not hasattr(_title_nouns, "_nlp"):
            _title_nouns._nlp = spacy.load("en_core_web_sm")  # type: ignore[attr-defined]
        doc = _title_nouns._nlp(text)  # type: ignore[attr-defined]
        nouns = [token.text.lower() for token in doc if token.pos_ in ("PNOUN", "NOUN", "PROPN")]
        return nouns or text.split()
    except Exception:
        return text.split()


def _full_goal_options(goal_options: Any) -> list[str]:
    if isinstance(goal_options, dict):
        return [normalize_text(v) for v in goal_options.values() if normalize_text(v)]
    if isinstance(goal_options, list):
        out = []
        for option in goal_options:
            if isinstance(option, dict):
                value = option.get("value")
            else:
                value = option
            value = normalize_text(value)
            if value:
                out.append(value)
        return out
    return []


def compute_full_reward(index: Any, goal: dict[str, Any], state: EnvState) -> tuple[float, dict[str, Any]]:
    item = _get_product(index, state.asin)
    if not item:
        return 0.0, {"success": False, "reason": "no_product_selected", "task_score": 0.0}

    selected_asin = str(item.get("asin") or "")
    target_asin = str(goal.get("asin") or "")
    query_match = normalize_text(item.get("query")) == normalize_text(goal.get("query"))
    purchased_category = [normalize_text(x) for x in str(item.get("product_category") or "").split("›")]
    goal_category = [normalize_text(x) for x in str(goal.get("product_category") or "").split("›")]
    category_match = len(set(purchased_category) & set(goal_category)) >= 2

    purchased_nouns = set(_title_nouns(str(item.get("name") or item.get("Title") or "")))
    desired_nouns = _title_nouns(str(goal.get("name") or ""))
    title_score = (len(purchased_nouns & set(desired_nouns)) / len(desired_nouns)) if desired_nouns else 0.2
    type_match = query_match or category_match or title_score > 0.2
    r_type = 1.0 if type_match else 0.5
    if title_score < 0.1:
        r_type = 0.1
    if title_score == 0.0:
        r_type = 0.0

    searchable_text = product_attributes_text(item, {"attributes": item.get("Attributes") or []})
    purchased_attrs = [normalize_text(x) for x in item.get("Attributes") or []]
    goal_attrs = [normalize_text(x) for x in goal.get("attributes") or goal.get("instruction_attributes") or []]
    num_attr_matches = 0
    matched_attributes: list[str] = []
    for goal_attr in goal_attrs:
        matched = any(_fuzzy_score(attr, goal_attr) > 85 for attr in purchased_attrs)
        if not matched and goal_attr in searchable_text:
            matched = True
        if matched:
            num_attr_matches += 1
            matched_attributes.append(goal_attr)

    goal_options = [_normalize_color(x) for x in _full_goal_options(goal.get("goal_options"))]
    selected_option_values = [_normalize_color(v) for v in state.selected_options.values()]
    num_option_matches = 0
    option_hits: dict[str, bool] = {}
    for goal_option in goal_options:
        hit = any(_fuzzy_score(selected, goal_option) > 85 for selected in selected_option_values)
        option_hits[goal_option] = hit
        if hit:
            num_option_matches += 1

    price = float(item.get("_price") or product_price(item))
    price_upper = float(goal.get("price_upper") or 1000000.0)
    r_price = 1.0 if price <= price_upper else 0.0
    denom = len(goal_attrs) + len(goal_options) + 1
    task_score = ((num_attr_matches + num_option_matches + r_price) / denom) * r_type if denom else 0.0
    task_score = max(0.0, min(1.0, task_score))
    success = task_score >= 0.999
    return (1.0 if success else 0.0), {
        "success": success,
        "task_score": task_score,
        "final_reward": 1.0 if success else 0.0,
        "target_asin": target_asin,
        "selected_asin": selected_asin,
        "product_match": 1.0 if selected_asin == target_asin else 0.0,
        "r_type": r_type,
        "r_attr": (num_attr_matches / len(goal_attrs)) if goal_attrs else 1.0,
        "r_option": (num_option_matches / len(goal_options)) if goal_options else 1.0,
        "r_price": r_price,
        "query_match": query_match,
        "category_match": category_match,
        "title_score": title_score,
        "matched_attributes": matched_attributes,
        "option_hits": option_hits,
        "selected_options": state.selected_options,
        "goal_options": goal_options,
    }


def compute_reward(index: Any, goal: dict[str, Any], state: EnvState) -> tuple[float, dict[str, Any]]:
    if goal.get("reward_mode") == "webshop_full":
        return compute_full_reward(index, goal, state)

    item = _get_product(index, state.asin)
    if not item:
        return 0.0, {"success": False, "reason": "no_product_selected"}

    target_asin = str(goal.get("asin") or "")
    selected_asin = str(item.get("asin") or "")
    product_match = 1.0 if selected_asin == target_asin else 0.0

    attr_record = index.attrs.get(selected_asin) or {}
    searchable_text = product_attributes_text(item, attr_record)
    target_attrs = [normalize_text(x) for x in goal.get("instruction_attributes") or goal.get("attributes") or []]
    attr_hits = [attr for attr in target_attrs if attr and attr in searchable_text]
    attr_score = (len(attr_hits) / len(target_attrs)) if target_attrs else 1.0

    goal_options = {normalize_text(k): normalize_text(v) for k, v in (goal.get("goal_options") or {}).items()}
    selected_options = {normalize_text(k): normalize_text(v) for k, v in state.selected_options.items()}
    option_hits = {
        name: selected_options.get(name) == value
        for name, value in goal_options.items()
    }
    option_score = (sum(option_hits.values()) / len(goal_options)) if goal_options else 1.0

    price = product_price(item)
    price_upper = float(goal.get("price_upper") or 1e9)
    price_score = 1.0 if price <= price_upper else 0.0

    score = 0.35 * product_match + 0.30 * attr_score + 0.25 * option_score + 0.10 * price_score
    if selected_asin != target_asin:
        score *= 0.5
    score = max(0.0, min(1.0, score))
    return score, {
        "success": score >= 0.999,
        "target_asin": target_asin,
        "selected_asin": selected_asin,
        "product_match": product_match,
        "attr_score": attr_score,
        "option_score": option_score,
        "price_score": price_score,
        "matched_attributes": attr_hits,
        "option_hits": option_hits,
        "selected_options": selected_options,
        "goal_options": goal_options,
    }


class WebShopEngine:
    def __init__(self, index: ProductIndex, *, search_top_k: int = 10):
        self.index = index
        self.search_top_k = search_top_k

    def reset(self, goal_index: int) -> tuple[str, EnvState, dict[str, Any]]:
        goal = self.index.goal(goal_index)
        state = EnvState(page_type="home")
        info = {
            "goal_index": goal_index,
            "asin": goal.get("asin"),
            "instruction": goal.get("instruction"),
            "available_actions": self.available_actions(goal, state),
        }
        return render_home(goal), state, info

    def step(self, goal_index: int, state: EnvState, action: str) -> StepResponse:
        goal = self.index.goal(goal_index)
        parsed = parse_action(action)
        if parsed is None:
            return StepResponse(
                observation=self._render_current(goal, state),
                env_state=state,
                reward=0.0,
                done=False,
                info={
                    "error": "invalid_action_format",
                    "expected": "search[...] or click[...]",
                    "available_actions": self.available_actions(goal, state),
                },
            )

        action_name, value = parsed
        new_state = state.model_copy(deep=True)
        new_state.last_action = action

        if action_name == "search":
            new_state.page_type = "search_results"
            new_state.query = value
            new_state.page_num = 0
            new_state.asin = None
            new_state.subpage = None
            observation = self._render_current(goal, new_state)
            return StepResponse(
                observation=observation,
                env_state=new_state,
                reward=0.0,
                done=False,
                info={"available_actions": self.available_actions(goal, new_state)},
            )

        if value == "buy now":
            new_state.page_type = "done"
            reward, reward_info = compute_reward(self.index, goal, new_state)
            reward_info.update({"final_reward": reward, "goal_index": goal_index})
            reward_info["available_actions"] = self.available_actions(goal, new_state)
            return StepResponse(
                observation="Episode complete.",
                env_state=new_state,
                reward=reward,
                done=True,
                info=reward_info,
            )

        if value == "back to search":
            new_state.page_type = "search_results" if new_state.query else "home"
            new_state.subpage = None
            observation = self._render_current(goal, new_state)
            return StepResponse(
                observation=observation,
                env_state=new_state,
                reward=0.0,
                done=False,
                info={"available_actions": self.available_actions(goal, new_state)},
            )

        if value == "back to item":
            if new_state.asin:
                new_state.page_type = "item"
                new_state.subpage = None
            observation = self._render_current(goal, new_state)
            return StepResponse(
                observation=observation,
                env_state=new_state,
                reward=0.0,
                done=False,
                info={"available_actions": self.available_actions(goal, new_state)},
            )

        if value == NEXT_PAGE and new_state.page_type == "search_results":
            new_state.page_num += 1
            observation = self._render_current(goal, new_state)
            return StepResponse(
                observation=observation,
                env_state=new_state,
                reward=0.0,
                done=False,
                info={"available_actions": self.available_actions(goal, new_state)},
            )

        if value == PREV_PAGE and new_state.page_type == "search_results":
            new_state.page_num = max(0, new_state.page_num - 1)
            observation = self._render_current(goal, new_state)
            return StepResponse(
                observation=observation,
                env_state=new_state,
                reward=0.0,
                done=False,
                info={"available_actions": self.available_actions(goal, new_state)},
            )

        asin = value.upper()
        if _has_product(self.index, asin):
            new_state.page_type = "item"
            new_state.asin = asin
            new_state.subpage = None
            new_state.selected_options = {}
            observation = self._render_current(goal, new_state)
            return StepResponse(
                observation=observation,
                env_state=new_state,
                reward=0.0,
                done=False,
                info={"available_actions": self.available_actions(goal, new_state)},
            )

        if value in {"description", "features", "reviews"} and new_state.asin:
            new_state.page_type = "subpage"
            new_state.subpage = value
            observation = self._render_current(goal, new_state)
            return StepResponse(
                observation=observation,
                env_state=new_state,
                reward=0.0,
                done=False,
                info={"available_actions": self.available_actions(goal, new_state)},
            )

        if new_state.asin:
            item = _get_product(self.index, new_state.asin)
            option_name = _option_lookup(item or {}).get(value)
            if option_name:
                new_state.selected_options[option_name] = value
                new_state.page_type = "item"
                new_state.subpage = None
                observation = self._render_current(goal, new_state)
                return StepResponse(
                    observation=observation,
                    env_state=new_state,
                    reward=0.0,
                    done=False,
                    info={"available_actions": self.available_actions(goal, new_state)},
                )

        return StepResponse(
            observation=self._render_current(goal, state),
            env_state=state,
            reward=0.0,
            done=False,
            info={
                "error": "click_target_not_available",
                "target": value,
                "available_actions": self.available_actions(goal, state),
            },
        )

    def available_actions(self, goal: dict[str, Any], state: EnvState) -> list[str]:
        if state.page_type == "done":
            return []

        actions = ["search[<your query>]"]

        if state.page_type == "search_results":
            results = self.index.search(state.query, top_k=self.search_top_k)
            for result in _page_results(results, state.page_num):
                asin = str(result["item"].get("asin") or "").strip()
                if asin:
                    actions.append(f"click[{asin}]")
            if state.page_num > 0:
                actions.append("click[< Prev]")
            if (state.page_num + 1) * PRODUCT_WINDOW < len(results):
                actions.append("click[Next >]")
            return actions

        if state.page_type == "item" and _has_product(self.index, state.asin):
            item = _get_product(self.index, state.asin) or {}
            for target in ("Description", "Features", "Reviews"):
                actions.append(f"click[{target}]")
            for choices in normalize_options(item.get("customization_options")).values():
                for choice in choices:
                    actions.append(f"click[{choice['value']}]")
            actions.append("click[Back to Search]")
            actions.append("click[Buy Now]")
            return actions

        if state.page_type == "subpage" and _has_product(self.index, state.asin):
            actions.append("click[Back to Item]")
            actions.append("click[Back to Search]")
            actions.append("click[Buy Now]")
            return actions

        return actions

    def _render_current(self, goal: dict[str, Any], state: EnvState) -> str:
        if state.page_type == "home":
            return render_home(goal)
        if state.page_type == "search_results":
            results = self.index.search(state.query, top_k=self.search_top_k)
            return render_search_results(goal, state.query, results, state.page_num)
        if state.page_type == "item" and _has_product(self.index, state.asin):
            return render_item(goal, _get_product(self.index, state.asin) or {}, state.selected_options)
        if state.page_type == "subpage" and _has_product(self.index, state.asin):
            return render_subpage(goal, _get_product(self.index, state.asin) or {}, state.subpage or "")
        return render_home(goal)
