"""SingleTurnRAGAgent — semantic-search-only baseline.

Uses the user's initial query for a single semantic search and recommends
the top-ranked products. No conversation, no filtering.

Purpose: shows the cost of not eliciting constraints — pure retrieval
without any constraint awareness.
"""

from __future__ import annotations

import json

from simulation.agents.base import BaseAgent
from shared.types import Action, RECOMMEND_ACTION_NAME


class SingleTurnRAGAgent(BaseAgent):
    """One search, top-K recommend --- no conversation, no filtering."""

    requires_llm = False

    def __init__(self, num_recommendations: int = 3) -> None:
        self.num_recs = num_recommendations
        self._step = 0

    def reset(self, env=None) -> None:
        self._step = 0

    def catalog_tool_names(self) -> tuple[str, ...]:
        return ("search_products",)

    def act(self, observation: str) -> Action:
        if self._step == 0:
            self._step = 1
            return Action("search_products", {"query": observation, "top_k": self.num_recs})

        try:
            items = json.loads(observation)
            product_ids = []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        for key in ("name", "product_name", "car_id", "id"):
                            if key in item:
                                product_ids.append(item[key])
                                break
        except (json.JSONDecodeError, TypeError):
            product_ids = []

        return Action(RECOMMEND_ACTION_NAME, {"product_ids": product_ids[: self.num_recs]})
