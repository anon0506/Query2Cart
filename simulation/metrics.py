"""Evaluation metrics for Query2Cart.

Primary metrics:
    reward           – I(constraints_ok) * (0.5 + 0.5 * preference_utility)
    success_rate     – fraction of tasks with reward > 0 (= constraints satisfied)
    pref_utility     – average preference utility for constraint-satisfying recs
    csr              – constraint satisfaction rate of recommended products

Ranking metrics (``compute_continuous_ranking_metrics``, pool utilities as relevance):
    ndcg@k           – Normalized Discounted Cumulative Gain at k (k=1,3,5,n)
    graded_precision – average relevance of recommended items (normalized)
    graded_recall    – fraction of total pool utility captured by recommendations
    graded_f1        – harmonic mean of graded precision and recall

Secondary metrics:
    avg_turns        – mean conversation turns before recommendation
    elicitation      – fraction of hard constraints the agent discovered
"""

from __future__ import annotations

import math
import statistics
from typing import Any


# ── helpers ──────────────────────────────────────────────────────────────

def _dcg(relevances: list[float]) -> float:
    """Discounted Cumulative Gain."""
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def _deduplicate(items: list[str]) -> list[str]:
    """Remove duplicates preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


# ── ranking metrics (per-task) ───────────────────────────────────────────

def compute_continuous_ranking_metrics(
    recommended: list[str],
    pool_utilities: dict[str, float],
    best_utility: float | None = None,
) -> dict[str, float]:
    """Compute NDCG@k and graded F1 using continuous utility scores.

    Args:
        recommended: ordered list of recommended product IDs (duplicates removed).
        pool_utilities: {product_id: utility} for ALL constraint-satisfying products.
        best_utility: maximum utility for normalization.

    Returns dict with ndcg@{1,3,5,n}, graded_precision, graded_recall,
    graded_f1, num_recommendations.
    """
    recommended = _deduplicate(recommended)
    n = len(recommended)

    rec_rels = [pool_utilities.get(pid, 0.0) for pid in recommended]
    ideal_rels = sorted(pool_utilities.values(), reverse=True)
    total_pool_utility = sum(ideal_rels)
    max_rel = best_utility if best_utility is not None else (ideal_rels[0] if ideal_rels else 1.0)

    # NDCG@k
    ndcg: dict[str, float] = {}
    for k in (1, 3, 5):
        actual = rec_rels[:k] + [0.0] * max(0, k - n)
        ideal = ideal_rels[:k] + [0.0] * max(0, k - len(ideal_rels))
        idcg = _dcg(ideal)
        ndcg[f"ndcg@{k}"] = _safe_div(_dcg(actual), idcg)

    if n > 0:
        ideal_at_n = ideal_rels[:n] + [0.0] * max(0, n - len(ideal_rels))
        idcg_n = _dcg(ideal_at_n)
        ndcg["ndcg@n"] = _safe_div(_dcg(rec_rels), idcg_n)
    else:
        ndcg["ndcg@n"] = 0.0

    # Graded precision / recall / F1
    captured = sum(rec_rels)
    graded_precision = _safe_div(captured, n * max_rel)
    graded_recall = _safe_div(captured, total_pool_utility)

    if graded_precision + graded_recall > 0:
        graded_f1 = 2 * graded_precision * graded_recall / (graded_precision + graded_recall)
    else:
        graded_f1 = 0.0

    return {
        **ndcg,
        "graded_precision": graded_precision,
        "graded_recall": graded_recall,
        "graded_f1": graded_f1,
        "num_recommendations": n,
    }


# ── single-run aggregation ──────────────────────────────────────────────

def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from a list of per-task result dicts."""
    if not results:
        return {}

    metrics: dict[str, Any] = {"overall": _agg_group(results)}

    for diff in ("small", "medium", "large", "oc_feasible", "oc_infeasible"):
        group = [r for r in results if r.get("difficulty") == diff]
        if group:
            metrics[diff] = _agg_group(group)

    return metrics


def _agg_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)

    def _avg(key: str, nested: str | None = None) -> float:
        if nested:
            vals = [r.get(nested, {}).get(key, 0.0) for r in results]
        else:
            vals = [r.get(key, 0.0) for r in results]
        return sum(vals) / n

    avg_elic = _avg("elicitation_completeness", "info")
    avg_pref_elic = _avg("preference_elicitation", "info")

    agg: dict[str, Any] = {
        "n": n,
        "avg_reward": _avg("reward"),
        "success_rate": sum(1 for r in results if r.get("reward", 0) > 0) / n,
        "avg_preference_utility": _avg("preference_utility", "info"),
        "avg_csr": _avg("constraint_satisfaction_rate", "info"),
        "avg_turns": _avg("conversation_turns"),
        "avg_elicitation": avg_elic,
        "avg_pref_elicitation": avg_pref_elic,
        "avg_combined_elicitation": (avg_elic + avg_pref_elic) / 2 if (avg_elic + avg_pref_elic) > 0 else 0.0,
    }

    # Ranking metrics (only for tasks where they were computed)
    for key in ("ndcg@1", "ndcg@3", "ndcg@5", "ndcg@n",
                "graded_precision", "graded_recall", "graded_f1"):
        vals = [r.get("info", {}).get(key) for r in results]
        vals = [v for v in vals if v is not None]
        if vals:
            agg[f"avg_{key}"] = sum(vals) / len(vals)

    num_recs = [r.get("info", {}).get("num_recommendations") for r in results]
    num_recs = [v for v in num_recs if v is not None]
    if num_recs:
        agg["avg_num_recommendations"] = sum(num_recs) / len(num_recs)

    return agg


# ── multi-trial task aggregation ─────────────────────────────────────────

def aggregate_multi_trial_tasks(
    trial_data: dict[str, list[dict]],
    save_individual: bool = False,
) -> list[dict]:
    """Average metrics across trials per task, return flat task list."""
    aggregated_tasks = []

    for task_id, trials in trial_data.items():
        if not trials:
            continue

        first = trials[0]

        rewards = [t["reward"] for t in trials]
        turns = [t["conversation_turns"] for t in trials]
        costs = [t.get("total_cost", 0.0) for t in trials]

        # Merge info: average numeric fields, carry non-numeric from first trial
        info_merged = _merge_trial_info(trials)

        task_result: dict[str, Any] = {
            "task_id": task_id,
            "difficulty": first.get("difficulty"),
            "initial_query": first.get("initial_query"),
            "reward": statistics.mean(rewards),
            "conversation_turns": statistics.mean(turns),
            "total_cost": statistics.mean(costs),
            "num_trials": len(trials),
            "info": info_merged,
        }

        if len(trials) > 1:
            task_result["reward_std"] = statistics.stdev(rewards)
            task_result["conversation_turns_std"] = statistics.stdev(turns)
            task_result["total_cost_std"] = statistics.stdev(costs)

        if save_individual:
            task_result["trials"] = trials

        best_trial = max(trials, key=lambda x: x["reward"])
        task_result["recommended_products"] = best_trial.get("recommended_products", [])

        aggregated_tasks.append(task_result)

    return aggregated_tasks


def _merge_trial_info(trials: list[dict]) -> dict[str, Any]:
    """Merge info dicts across trials: average numerics, keep non-numerics from first."""
    all_keys: set[str] = set()
    for t in trials:
        info = t.get("info")
        if isinstance(info, dict):
            all_keys.update(info.keys())

    if not all_keys:
        return {}

    first_info = trials[0].get("info", {})
    merged: dict[str, Any] = {}

    for key in all_keys:
        numeric_vals = []
        first_val = None
        for t in trials:
            val = t.get("info", {}).get(key)
            if first_val is None and val is not None:
                first_val = val
            if isinstance(val, (int, float)):
                numeric_vals.append(val)

        if numeric_vals:
            merged[key] = statistics.mean(numeric_vals)
            if len(numeric_vals) > 1:
                merged[f"{key}_std"] = statistics.stdev(numeric_vals)
        elif first_val is not None:
            # Carry forward non-numeric values (reason, match_type, etc.)
            merged[key] = first_info.get(key, first_val)

    return merged


# ── multi-trial seed aggregation (mean ± std across seeds) ───────────────

_MULTI_TRIAL_KEYS = [
    "success_rate", "avg_reward", "avg_csr", "avg_turns",
    "avg_elicitation", "avg_preference_utility",
    "avg_ndcg@3", "avg_graded_f1", "avg_num_recommendations",
]


def aggregate_multi_trial(
    all_trial_results: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate metrics across multiple independent evaluation runs (seeds).

    Args:
        all_trial_results: list of length ``num_seeds``, where each element is
            a flat list of per-task result dicts from one full evaluation run.

    Returns:
        Dict with ``overall`` and per-difficulty keys containing
        ``mean_<metric>`` and ``std_<metric>`` for key metrics.
    """
    if not all_trial_results:
        return {}

    n_seeds = len(all_trial_results)
    per_seed_metrics = [aggregate_results(trial) for trial in all_trial_results]

    groups: set[str] = set()
    for sm in per_seed_metrics:
        groups.update(sm.keys())

    combined: dict[str, Any] = {"n_seeds": n_seeds}

    for group in sorted(groups):
        seed_group_metrics = [sm[group] for sm in per_seed_metrics if group in sm]
        if not seed_group_metrics:
            continue

        group_agg: dict[str, Any] = {"n": seed_group_metrics[0].get("n", 0)}
        for key in _MULTI_TRIAL_KEYS:
            vals = [m.get(key) for m in seed_group_metrics if m.get(key) is not None]
            if len(vals) >= 2:
                group_agg[f"mean_{key}"] = statistics.mean(vals)
                group_agg[f"std_{key}"] = statistics.stdev(vals)
            elif len(vals) == 1:
                group_agg[f"mean_{key}"] = vals[0]
                group_agg[f"std_{key}"] = 0.0

        group_agg["per_seed"] = seed_group_metrics
        combined[group] = group_agg

    return combined


# ── printing ─────────────────────────────────────────────────────────────

_DIFFICULTY_ORDER = ("overall", "small", "medium", "large", "oc_feasible", "oc_infeasible")


def _difficulty_label(key: str) -> str:
    return "Overall" if key == "overall" else key.replace("_", "-").title()


def print_results_table(metrics: dict[str, Any], agent_name: str = "") -> None:
    """Pretty-print aggregated metrics."""
    if agent_name:
        print(f"\n  Agent: {agent_name}")
    header = (
        f"{'Difficulty':<18} {'N':>4} {'Reward':>7} {'Success':>8} "
        f"{'Util':>6} {'CSR':>6} {'C.Elic':>6} {'P.Elic':>6} {'Turns':>6}"
    )
    print(header)
    print("─" * len(header))
    for key in _DIFFICULTY_ORDER:
        m = metrics.get(key)
        if m is None:
            continue
        print(
            f"{_difficulty_label(key):<18} {m['n']:>4} "
            f"{m['avg_reward']:>7.3f} "
            f"{m['success_rate']:>7.1%} "
            f"{m.get('avg_preference_utility', 0):>5.1%} "
            f"{m['avg_csr']:>5.1%} "
            f"{m.get('avg_elicitation', 0):>5.1%} "
            f"{m.get('avg_pref_elicitation', 0):>5.1%} "
            f"{m['avg_turns']:>6.1f}"
        )


def print_multi_trial_table(
    metrics: dict[str, Any],
    agent_name: str = "",
) -> None:
    """Pretty-print multi-trial metrics with mean ± std."""
    n_seeds = metrics.get("n_seeds", 0)
    if agent_name:
        print(f"\n  {agent_name} ({n_seeds} seeds)")
    header = (
        f"{'Difficulty':<18} {'N':>4} {'Success':>14} "
        f"{'Reward':>14} {'CSR':>14} {'Elic':>14} {'Turns':>14}"
    )
    print(header)
    print("─" * len(header))
    for key in _DIFFICULTY_ORDER:
        m = metrics.get(key)
        if m is None:
            continue

        def _fmt(metric_name: str) -> str:
            mean = m.get(f"mean_{metric_name}")
            std = m.get(f"std_{metric_name}")
            if mean is None:
                return f"{'—':>14}"
            if std is not None and std > 0:
                return f"{mean:>6.1%}±{std:.1%}"
            return f"{mean:>6.1%}      "

        print(
            f"{_difficulty_label(key):<18} {m.get('n', 0):>4} "
            f"{_fmt('success_rate')} "
            f"{_fmt('avg_reward')} "
            f"{_fmt('avg_csr')} "
            f"{_fmt('avg_elicitation')} "
            f"{_fmt('avg_turns')}"
        )


def print_ranking_table(metrics: dict[str, Any], agent_name: str = "") -> None:
    """Pretty-print ranking metrics (NDCG, graded P/R/F1)."""
    overall = metrics.get("overall", {})
    if "avg_ndcg@1" not in overall:
        return
    if agent_name:
        print(f"\n  Ranking metrics — {agent_name}")
    header = (
        f"{'Difficulty':<18} {'N':>4} {'NDCG@1':>7} {'NDCG@3':>7} "
        f"{'NDCG@5':>7} {'NDCG@n':>7} {'G.Prec':>7} {'G.Rec':>7} "
        f"{'G.F1':>7} {'#Rec':>5}"
    )
    print(header)
    print("─" * len(header))
    for key in _DIFFICULTY_ORDER:
        m = metrics.get(key)
        if m is None:
            continue
        print(
            f"{_difficulty_label(key):<18} {m['n']:>4} "
            f"{m.get('avg_ndcg@1', 0):>7.3f} "
            f"{m.get('avg_ndcg@3', 0):>7.3f} "
            f"{m.get('avg_ndcg@5', 0):>7.3f} "
            f"{m.get('avg_ndcg@n', 0):>7.3f} "
            f"{m.get('avg_graded_precision', 0):>7.3f} "
            f"{m.get('avg_graded_recall', 0):>7.3f} "
            f"{m.get('avg_graded_f1', 0):>7.3f} "
            f"{m.get('avg_num_recommendations', 0):>5.1f}"
        )
