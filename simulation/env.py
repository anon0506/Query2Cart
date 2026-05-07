"""Domain-agnostic environment for conversational recommendation.

Replaces the laptop-specific ``Query2Cart`` with a config-driven environment
that uses ``GenericFilter`` for all constraint checking and pool computation.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from shared.config import DomainConfig
from shared.filter import GenericFilter, hard_constraint_value_missing
from shared.types import Action, EnvResponse
from shared.scoring import (
    compute_preference_weights,
    compute_pool_stats,
    compute_product_utility,
)

from simulation.tools import (
    DomainCatalogContext,
    DomainToolSpec,
    build_domain_tools,
    get_domain_tool_map,
    get_domain_tool_schemas,
)
from simulation.user import DomainSimulatedUser
from simulation.metrics import compute_continuous_ranking_metrics

logger = logging.getLogger(__name__)

RESPOND_ACTION_NAME = "respond_to_user"
RECOMMEND_ACTION_NAME = "recommend_products"
DECLARE_INFEASIBLE_ACTION_NAME = "declare_infeasible"


class DomainEnv:
    """Config-driven environment for any product domain."""

    def __init__(
        self,
        config: DomainConfig,
        catalog: pd.DataFrame,
        tasks: list[dict],
        *,
        tools: list[DomainToolSpec] | None = None,
        user: DomainSimulatedUser | None = None,
        user_model: str = "gpt-4.1-mini",
        max_turns: int = 20,
        track_elicitation: bool = False,
        elicitation_model: str | None = None,
        desc_attrs: dict[str, dict] | None = None,
    ):
        self.config = config
        self.catalog = catalog
        self.tasks = tasks
        self.user_model = user_model
        self.max_turns = max_turns
        self.track_elicitation = track_elicitation
        self.elicitation_model = elicitation_model
        self.desc_attrs = desc_attrs or {}

        self._tools = tools or build_domain_tools(config)
        self._tools_map = get_domain_tool_map(self._tools)
        self._ctx = DomainCatalogContext(catalog, config)

        self._user = user
        self.current_task: dict = {}
        self.turn_count: int = 0
        self._recommended: bool = False

    def reset(self, task_index: int | None = None) -> EnvResponse:
        if task_index is None:
            raise ValueError("task_index is required")

        self.current_task = self.tasks[task_index]
        self.turn_count = 0
        self._recommended = False

        if self._user is None:
            self._user = DomainSimulatedUser(self.config, model=self.user_model)
        self._user.reset(self.current_task)

        return EnvResponse(
            observation=self.current_task["initial_query"],
            reward=0.0,
            done=False,
            info={"source": "user", "turn": 0},
        )

    def step(self, action: Action) -> EnvResponse:
        if action.name == RESPOND_ACTION_NAME:
            content = action.kwargs.get("content", "")
            self.turn_count += 1
            assert self._user is not None
            user_response, user_cost = self._user.step(content)

            if self.turn_count >= self.max_turns and not self._recommended:
                forced = (
                    user_response
                    + "\n\n[SYSTEM: Turn limit reached. You must now call "
                    "recommend_products with your best guess.]"
                )
                return EnvResponse(
                    observation=forced,
                    reward=0.0,
                    done=False,
                    truncated=True,
                    info={
                        "source": "user",
                        "turn": self.turn_count,
                        "reason": "max_turns_force_recommend",
                        "user_cost": user_cost,
                    },
                )

            return EnvResponse(
                observation=user_response,
                reward=0.0,
                done=False,
                info={"source": "user", "turn": self.turn_count, "user_cost": user_cost},
            )

        if action.name == RECOMMEND_ACTION_NAME:
            id_col = self.config.id_column
            param_name = id_col + "s" if not id_col.endswith("s") else id_col
            product_ids = (
                action.kwargs.get(param_name)
                or action.kwargs.get("product_ids")
                or action.kwargs.get("product_names")
                or action.kwargs.get(id_col)
                or []
            )
            if not isinstance(product_ids, list):
                product_ids = [product_ids]
            self._recommended = True
            reward, eval_info = self._evaluate(product_ids)
            return EnvResponse(
                observation=f"Recommendation submitted: {product_ids}",
                reward=reward,
                done=True,
                info=eval_info,
            )

        if action.name == DECLARE_INFEASIBLE_ACTION_NAME:
            reason = action.kwargs.get("reason", "")
            self._recommended = True
            reward, eval_info = self._evaluate_infeasibility(reason)
            return EnvResponse(
                observation="Infeasibility declaration submitted.",
                reward=reward,
                done=True,
                info=eval_info,
            )

        if action.name in self._tools_map:
            try:
                result = self._tools_map[action.name].invoke(self._ctx, **action.kwargs)
            except Exception as exc:
                logger.warning("Tool %s error: %s", action.name, exc)
                result = f"Error: {exc}"
            return EnvResponse(
                observation=result,
                reward=0.0,
                done=False,
                info={"source": action.name},
            )

        return EnvResponse(
            observation=f"Unknown action: '{action.name}'. Available: {list(self._tools_map.keys())}",
            reward=0.0,
            done=False,
            info={"source": "error"},
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return get_domain_tool_schemas(self._tools)

    # -- evaluation ------------------------------------------------------------

    def _coerce_id(self, pid: Any) -> Any:
        """Cast *pid* to the catalog index dtype so lookups match."""
        idx_dtype = self.catalog.index.dtype
        try:
            if pd.api.types.is_integer_dtype(idx_dtype):
                return int(pid)
            if pd.api.types.is_float_dtype(idx_dtype):
                return float(pid)
        except (ValueError, TypeError):
            pass
        return pid

    # -- description-constraint helpers -----------------------------------------

    def _check_desc_violations(
        self, pid: Any, desc_constraints: dict[str, Any],
    ) -> list[str]:
        """Check a product against description-based constraints."""
        pid_str = str(pid)
        product_attrs = self.desc_attrs.get(pid_str, self.desc_attrs.get(pid, {}))
        violations: list[str] = []
        for key, required_val in desc_constraints.items():
            attr_name = key[len("desc_"):]
            actual = product_attrs.get(attr_name)
            if hard_constraint_value_missing(actual):
                violations.append(
                    f"desc attr '{attr_name}' missing or NA for '{pid}'"
                )
                continue
            actual_str = str(actual).lower().strip()
            required_str = str(required_val).lower().strip()
            if required_str in ("true", "false"):
                if actual_str != required_str:
                    violations.append(
                        f"desc_{attr_name}={actual} violates required {required_val}"
                    )
                continue
            try:
                if not (float(actual) >= float(required_val)):
                    violations.append(
                        f"desc_{attr_name}={actual} violates >={required_val}"
                    )
                continue
            except (ValueError, TypeError):
                pass
            if actual_str != required_str:
                violations.append(
                    f"desc_{attr_name}={actual} violates required {required_val}"
                )
        return violations

    def _apply_desc_filter(
        self, pool: pd.DataFrame, desc_constraints: dict[str, Any],
    ) -> pd.DataFrame:
        """Filter a pool by description-based constraints."""
        id_col = self.config.id_column
        keep = []
        for idx in pool.index:
            pid = pool.at[idx, id_col] if id_col in pool.columns else idx
            if not self._check_desc_violations(pid, desc_constraints):
                keep.append(idx)
        return pool.loc[pool.index.isin(keep)]

    # -- evaluation ------------------------------------------------------------

    def _evaluate(self, product_ids: list) -> tuple[float, dict[str, Any]]:
        task = self.current_task
        all_constraints = task["user_profile"]["hard_constraints"]
        preferences = task["user_profile"].get("attribute_preferences", [])
        registry = self.config.constraints

        structured_constraints = {
            k: v for k, v in all_constraints.items() if not k.startswith("desc_")
        }
        desc_constraints = {
            k: v for k, v in all_constraints.items() if k.startswith("desc_")
        }

        product_ids = [self._coerce_id(pid) for pid in product_ids]

        violations: dict[str, list[str]] = {}
        for pid in product_ids:
            if pid not in self.catalog.index:
                violations[pid] = [f"product '{pid}' not found in catalog"]
                continue
            v = GenericFilter.check_violations(
                self.catalog.loc[pid], structured_constraints, registry,
            )
            if desc_constraints and self.desc_attrs:
                v.extend(self._check_desc_violations(pid, desc_constraints))
            if v:
                violations[pid] = v

        n_recs = max(len(product_ids), 1)
        csr = 1.0 - len(violations) / n_recs

        satisfying_pids = [
            pid for pid in product_ids
            if pid not in violations and pid in self.catalog.index
        ]

        if not satisfying_pids:
            reward = 0.0
            preference_utility = 0.0
            best_pid = None
            constraints_satisfied = False
        else:
            constraints_satisfied = True
            pool = GenericFilter.apply(self.catalog, structured_constraints, registry)
            if desc_constraints and self.desc_attrs:
                pool = self._apply_desc_filter(pool, desc_constraints)
            pool_stats = compute_pool_stats(pool, preferences)
            weights = compute_preference_weights(preferences)

            pool_utilities: dict[str, float] = {}
            for pid in pool.index:
                u = compute_product_utility(pool.loc[pid], preferences, pool_stats, weights)
                pool_utilities[pid] = max(0.0, u)

            min_pool_utility = min(pool_utilities.values()) if pool_utilities else 0.0
            max_pool_utility = max(pool_utilities.values()) if pool_utilities else 1.0

            best_pid = max(
                satisfying_pids,
                key=lambda p: pool_utilities.get(p, 0.0),
            )
            best_utility = pool_utilities.get(best_pid, 0.0)

            if preferences and max_pool_utility > min_pool_utility:
                preference_utility = (best_utility - min_pool_utility) / (max_pool_utility - min_pool_utility)
            elif preferences:
                preference_utility = 1.0
            else:
                preference_utility = 1.0

            reward = 0.5 + 0.5 * preference_utility

        info: dict[str, Any] = {
            "source": "evaluation",
            "reward": reward,
            "preference_utility": preference_utility,
            "constraints_satisfied": constraints_satisfied,
            "best_product": best_pid,
            "recommended_products": product_ids,
            "constraint_satisfaction_rate": csr,
            "violations": violations,
            "conversation_turns": self.turn_count,
        }

        if satisfying_pids:
            best_util_val = max(pool_utilities.values()) if pool_utilities else 1.0
            ranking = compute_continuous_ranking_metrics(product_ids, pool_utilities, best_util_val)
            info.update(ranking)
        else:
            info.update({
                "ndcg@1": 0.0, "ndcg@3": 0.0, "ndcg@5": 0.0, "ndcg@n": 0.0,
                "graded_precision": 0.0, "graded_recall": 0.0, "graded_f1": 0.0,
                "num_recommendations": len(product_ids),
            })

        if self._user:
            self._attach_user_info(info)

        return reward, info

    def _attach_user_info(self, info: dict[str, Any]) -> None:
        assert self._user is not None
        if self.track_elicitation:
            self._user.compute_elicitation(model=self.elicitation_model)
            info["elicitation_completeness"] = self._user.get_elicitation_completeness()
            info["preference_elicitation"] = self._user.get_preference_elicitation()
            info["revealed_constraints"] = sorted(self._user.revealed_constraints)
            info["revealed_preferences"] = sorted(self._user.revealed_preferences)
        info["user_simulator_cost"] = self._user.get_total_cost()

    def _evaluate_infeasibility(self, reason: str) -> tuple[float, dict[str, Any]]:
        task = self.current_task
        task_infeasible = task.get("infeasible", False)
        reward = 1.0 if task_infeasible else 0.0
        match_type = "infeasible_correct" if task_infeasible else "infeasible_false_positive"

        info: dict[str, Any] = {
            "source": "evaluation",
            "match_type": match_type,
            "reward": reward,
            "preference_utility": 0.0,
            "constraints_satisfied": False,
            "best_product": None,
            "recommended_products": [],
            "constraint_satisfaction_rate": 0.0,
            "violations": {},
            "conversation_turns": self.turn_count,
            "infeasibility_reason": reason,
            "task_infeasible": task_infeasible,
        }
        if self._user:
            self._attach_user_info(info)
        return reward, info
