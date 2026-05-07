"""RandomAgent --- simplest possible baseline.

Searches the catalog with the user's initial query, picks random products
from the results, and recommends immediately.  No conversation, no LLM.
"""

from __future__ import annotations

import json
import random

from simulation.agents.base import BaseAgent
from shared.types import Action, RECOMMEND_ACTION_NAME


class RandomAgent(BaseAgent):
    """Search once, recommend random products from the results."""

    requires_llm = False

    def __init__(self, num_recommendations: int = 3):
        self.num_recs = num_recommendations
        self._step = 0

    def reset(self, env=None) -> None:
        self._step = 0

    def catalog_tool_names(self) -> tuple[str, ...]:
        return ("search_products",)

    def act(self, observation: str) -> Action:
        if self._step == 0:
            self._step = 1
            return Action("search_products", {"query": observation, "top_k": 15})

        try:
            items = json.loads(observation)
            if isinstance(items, list):
                product_ids = []
                for item in items:
                    if isinstance(item, dict):
                        for key in ("name", "product_name", "car_id", "id"):
                            if key in item:
                                product_ids.append(item[key])
                                break
            else:
                product_ids = []
        except (json.JSONDecodeError, TypeError):
            product_ids = []

        selected = (
            random.sample(product_ids, min(self.num_recs, len(product_ids)))
            if product_ids
            else []
        )
        return Action(RECOMMEND_ACTION_NAME, {"product_ids": selected})
