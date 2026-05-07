"""FilterOracleAgent --- ceiling baseline with oracle constraint access.

Has direct access to the task's ground-truth hard constraints.
Uses GenericFilter to apply all constraints domain-agnostically, then
recommends the top products.  No conversation at all.
"""

from __future__ import annotations

import json

from simulation.agents.base import BaseAgent
from shared.types import Action, RECOMMEND_ACTION_NAME, DECLARE_INFEASIBLE_ACTION_NAME
from shared.filter import GenericFilter


class FilterOracleAgent(BaseAgent):
    """Oracle: reads ground-truth constraints, filters, recommends."""

    requires_llm = False

    def __init__(self, num_recommendations: int = 3) -> None:
        self.num_recs = num_recommendations
        self._step = 0
        self._candidate_ids: list = []

    def reset(self, env=None) -> None:
        self._step = 0
        self._candidate_ids = []
        if env is not None:
            task = env.current_task
            constraints = task["user_profile"]["hard_constraints"]
            registry = env.config.constraints
            pool = GenericFilter.apply(env.catalog, constraints, registry)
            self._candidate_ids = pool.index.tolist()

    def catalog_tool_names(self) -> tuple[str, ...]:
        return ("search_products",)

    def act(self, observation: str) -> Action:
        if not self._candidate_ids:
            return Action(
                DECLARE_INFEASIBLE_ACTION_NAME,
                {"reason": "No products in the catalog satisfy all constraints."},
            )

        if self._step == 0 and len(self._candidate_ids) > self.num_recs:
            self._step = 1
            return Action("search_products", {"query": observation, "top_k": 50})

        if self._step == 1:
            try:
                items = json.loads(observation)
                search_ids = []
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            for key in ("name", "product_name", "car_id", "id"):
                                if key in item:
                                    search_ids.append(item[key])
                                    break
            except (json.JSONDecodeError, TypeError):
                search_ids = []

            cand_set = set(str(c) for c in self._candidate_ids)
            ranked = [pid for pid in search_ids if str(pid) in cand_set]
            remaining = [pid for pid in self._candidate_ids if str(pid) not in set(str(r) for r in ranked)]
            selected = (ranked + remaining)[: self.num_recs]
            return Action(RECOMMEND_ACTION_NAME, {"product_ids": selected})

        selected = self._candidate_ids[: self.num_recs]
        return Action(RECOMMEND_ACTION_NAME, {"product_ids": selected})
