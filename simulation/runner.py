"""Domain-agnostic simulation runner.

Two modes:

1. **Single task (verbose)** --- ``simulate_single_task()``
   Runs one task with rich console output showing every dialogue turn,
   tool call, and the final evaluation breakdown.

2. **Batch mode** --- ``simulate_all_tasks()``
   Runs all tasks with a progress bar, saves results incrementally to a
   JSONL file, and prints a summary table at the end.

Both accept an optional ``agent`` (or ``agent_fn``) argument.  When
omitted, a default ``ToolCallingAgent`` is created automatically.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from shared.config import DomainConfig
from shared.types import Action, EnvResponse

from simulation.env import DomainEnv
from simulation.tools import build_domain_tools
from simulation.user import DomainSimulatedUser
from simulation.metrics import aggregate_results, print_results_table, print_ranking_table

logger = logging.getLogger(__name__)


def _make_default_agent(config: DomainConfig, model: str = "gpt-4.1"):
    from simulation.agents.tool_calling import ToolCallingAgent
    return ToolCallingAgent(config=config, model=model)


# ── Pretty-printing helpers ──────────────────────────────────────────────

def _print_header(text: str, char: str = "═", width: int = 80) -> None:
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _print_task_header(task: dict, config: DomainConfig) -> None:
    tid = task.get("task_id", "?")
    diff = task.get("difficulty", "?")
    constraints = task["user_profile"]["hard_constraints"]
    preferences = task["user_profile"].get("attribute_preferences", [])
    plan = task["user_profile"]["constraint_revelation_plan"]
    use_case = task["user_profile"].get("use_case", "")
    pool_size = task.get("metadata", {}).get("filtered_pool_size", "?")

    _print_header(f"📋 TASK: {tid}  |  Difficulty: {diff}  |  Pool: {pool_size}")

    print(f"\n  💬 Initial query: \"{task['initial_query']}\"")
    if use_case:
        print(f"  🎯 Use case: {use_case}")

    # Constraints with revelation mode
    proactive = set(plan.get("proactive", []))
    responsive_keys = set()
    for keys in plan.get("responsive", {}).values():
        responsive_keys.update(keys if isinstance(keys, list) else [keys])
    reactive_keys = set()
    for keys in plan.get("reactive", {}).values():
        reactive_keys.update(keys if isinstance(keys, list) else [keys])

    print(f"\n  🔒 Hard Constraints ({len(constraints)}):")
    for key, val in constraints.items():
        label = config.get_constraint_label(key, val) or f"{key}: {val}"
        if key in proactive:
            mode = "🟢 PROACTIVE"
        elif key in responsive_keys:
            mode = "🟡 RESPONSIVE"
        elif key in reactive_keys:
            mode = "🔴 REACTIVE"
        else:
            mode = "⚪ UNKNOWN"
        print(f"     {mode}  {label}")

    if preferences:
        sorted_prefs = sorted(preferences, key=lambda p: p.get("priority", 99))
        print(f"\n  ⭐ Preferences ({len(preferences)}):")
        for p in sorted_prefs:
            attr = p["attribute"].replace("_", " ").title()
            direction = p.get("direction", "?")
            priority = p.get("priority", "?")
            mode = p.get("revelation_mode", "responsive")
            icon = "🟢" if mode == "proactive" else "🟡" if mode == "responsive" else "🔵"
            print(f"     {icon} P{priority}: {attr} → {direction}")

    print(f"\n{'─' * 80}")
    print("  🎬 CONVERSATION START")
    print(f"{'─' * 80}")


def _print_user_turn(text: str, turn: int) -> None:
    label = "Opening" if turn == 0 else f"Turn {turn}"
    print(f"\n  👤 User ({label}):")
    for line in text.split("\n"):
        print(f"     {line}")


def _print_agent_turn(text: str) -> None:
    print(f"\n  🤖 Agent:")
    for line in text.split("\n"):
        print(f"     {line}")


def _print_tool_call(name: str, arguments: str) -> None:
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        args = arguments
    args_str = json.dumps(args, indent=None, default=str) if isinstance(args, dict) else str(args)
    if len(args_str) > 120:
        args_str = args_str[:117] + "..."
    print(f"\n  🔧 Tool Call: {name}")
    print(f"     Args: {args_str}")


def _print_tool_result(name: str, result: str) -> None:
    if name in ("filter_products", "search_products"):
        try:
            items = json.loads(result)
            if isinstance(items, list):
                print(f"     📦 Result: {len(items)} {name.split('_')[0]} results")
                for item in items[:3]:
                    if isinstance(item, dict):
                        preview_keys = list(item.keys())[:4]
                        preview = {k: item[k] for k in preview_keys}
                        print(f"        • {preview}")
                if len(items) > 3:
                    print(f"        ... and {len(items) - 3} more")
                return
        except (json.JSONDecodeError, TypeError):
            pass

    if name == "get_product_details":
        lines = result.split("\n")
        print(f"     📄 Product Details:")
        for line in lines[:8]:
            print(f"        {line}")
        if len(lines) > 8:
            print(f"        ... ({len(lines) - 8} more fields)")
        return

    if name == "recommend_products":
        print(f"     ✅ Recommendation submitted!")
        return

    if name == "declare_infeasible":
        print(f"     ❌ Infeasibility declared!")
        return

    truncated = result[:200] + "..." if len(result) > 200 else result
    print(f"     📄 Result: {truncated}")


def _print_evaluation(info: dict[str, Any], config: DomainConfig) -> None:
    reward = info.get("reward", 0)
    csr = info.get("constraint_satisfaction_rate", 0)
    pref_util = info.get("preference_utility", 0)
    violations = info.get("violations", {})
    recs = info.get("recommended_products", [])
    turns = info.get("conversation_turns", 0)
    best = info.get("best_product")

    _print_header("📊 EVALUATION", char="─")

    # Reward with color indicator
    if reward >= 0.9:
        icon = "🏆"
    elif reward >= 0.5:
        icon = "✅"
    elif reward > 0:
        icon = "⚠️"
    else:
        icon = "❌"

    print(f"\n  {icon} Reward: {reward:.4f}")
    print(f"  📈 Preference Utility: {pref_util:.4f}")
    print(f"  🎯 Constraint Satisfaction: {csr:.1%}")
    print(f"  💬 Conversation Turns: {turns}")
    print(f"  📦 Products Recommended: {len(recs)}")
    if best:
        print(f"  ⭐ Best Product: {best}")

    if violations:
        print(f"\n  ⚠️  Violations:")
        for pid, v_list in violations.items():
            print(f"     {pid}:")
            for v in v_list:
                print(f"       ✗ {v}")

    # Ranking metrics
    ndcg1 = info.get("ndcg@1")
    if ndcg1 is not None:
        print(f"\n  📊 Ranking: NDCG@1={ndcg1:.3f}  NDCG@3={info.get('ndcg@3', 0):.3f}  "
              f"F1={info.get('graded_f1', 0):.3f}")

    # Elicitation
    elic = info.get("elicitation_completeness")
    if elic is not None:
        pref_elic = info.get("preference_elicitation", 0)
        print(f"  🔍 Elicitation: constraints={elic:.1%}  preferences={pref_elic:.1%}")

    # Cost
    user_cost = info.get("user_simulator_cost", 0)
    print(f"\n  💰 User Simulator Cost: ${user_cost:.4f}")


def _print_batch_line(entry: dict, idx: int, total: int, elapsed: float) -> None:
    tid = entry.get("task_id", "?")
    diff = entry.get("difficulty", "?")
    reward = entry.get("reward", 0)
    turns = entry.get("conversation_turns", 0)
    pref_util = entry.get("info", {}).get("preference_utility", 0)

    if reward >= 0.9:
        icon = "🏆"
    elif reward >= 0.5:
        icon = "✅"
    elif reward > 0:
        icon = "⚠️"
    else:
        icon = "❌"

    avg_per_task = elapsed / max(idx + 1, 1)
    remaining = avg_per_task * (total - idx - 1)
    eta = f"{remaining / 60:.1f}m" if remaining > 60 else f"{remaining:.0f}s"

    progress = f"[{idx + 1}/{total}]"
    print(
        f"  {icon} {progress:>8s}  {tid:<20s}  {diff:<14s}  "
        f"reward={reward:.3f}  util={pref_util:.3f}  turns={turns:>2d}  "
        f"ETA {eta}"
    )


# ── Single task runner (verbose) ─────────────────────────────────────────

def _infer_domain_dir(config: DomainConfig) -> Path | None:
    """Derive the domain directory from the config's catalog_path."""
    if config.catalog_path:
        p = Path(config.catalog_path)
        if p.is_absolute():
            return p.parent
        return p.parent if p.parent != Path(".") else None
    return None


def simulate_single_task(
    task: dict,
    catalog: pd.DataFrame,
    config: DomainConfig,
    *,
    agent=None,
    agent_model: str = "gpt-4.1",
    user_model: str = "gpt-4.1-mini",
    max_turns: int = 20,
    max_steps: int = 50,
    verbose: bool = True,
    track_elicitation: bool = True,
    domain_dir: str | Path | None = None,
    desc_attrs: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Run a single task with rich console output.

    Args:
        agent: A ConversationalAgent instance.  If *None*, a default
            ``ToolCallingAgent`` is created using *agent_model*.
    """
    if domain_dir is None:
        domain_dir = _infer_domain_dir(config)
    tools = build_domain_tools(config, domain_dir=domain_dir)
    user = DomainSimulatedUser(config, model=user_model)
    env = DomainEnv(
        config, catalog, [task],
        tools=tools,
        user=user,
        user_model=user_model,
        max_turns=max_turns,
        track_elicitation=track_elicitation,
        desc_attrs=desc_attrs,
    )
    if agent is None:
        agent = _make_default_agent(config, model=agent_model)

    if verbose:
        _print_task_header(task, config)

    obs = env.reset(task_index=0)
    agent.reset(env)

    if verbose:
        _print_user_turn(obs.observation, turn=0)

    incoming: dict[str, Any] | None = {"role": "user", "content": obs.observation}
    total_agent_cost = 0.0
    step = 0
    final_info: dict[str, Any] = {}

    while step < max_steps:
        try:
            agent_msg = agent.act(incoming)
            incoming = None
        except Exception as exc:
            logger.error("Agent LLM error: %s", exc)
            if verbose:
                print(f"\n  ❌ Agent Error: {exc}")
            return _build_error_result(task, str(exc))

        total_agent_cost += agent_msg.cost

        if agent_msg.tool_calls:
            for tc in agent_msg.tool_calls:
                if verbose:
                    _print_tool_call(tc.name, tc.arguments)

                try:
                    kwargs = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    kwargs = {}

                env_response = env.step(Action(name=tc.name, kwargs=kwargs))

                if verbose:
                    _print_tool_result(tc.name, env_response.observation)

                agent.add_to_history({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": env_response.observation,
                })

                if env_response.done:
                    final_info = env_response.info
                    if verbose:
                        _print_evaluation(final_info, config)
                    return _build_result(task, env, env_response, total_agent_cost)

            step += 1
            continue

        if agent_msg.content:
            if verbose:
                _print_agent_turn(agent_msg.content)

            env_response = env.step(
                Action("respond_to_user", {"content": agent_msg.content})
            )

            if verbose:
                _print_user_turn(env_response.observation, turn=env.turn_count)

            incoming = {"role": "user", "content": env_response.observation}
            step += 1

    # Emergency: max steps hit
    if verbose:
        print(f"\n  ⏰ Max steps ({max_steps}) reached — forcing recommendation")
    env_response = env.step(Action("recommend_products", {"product_ids": []}))
    final_info = env_response.info
    if verbose:
        _print_evaluation(final_info, config)
    return _build_result(task, env, env_response, total_agent_cost)


def _build_result(
    task: dict,
    env: DomainEnv,
    env_response: Any,
    agent_cost: float,
) -> dict[str, Any]:
    info = env_response.info
    info["agent_cost"] = agent_cost
    return {
        "task_id": task.get("task_id"),
        "difficulty": task.get("difficulty"),
        "initial_query": task.get("initial_query"),
        "reward": env_response.reward,
        "recommended_products": info.get("recommended_products", []),
        "conversation_turns": env.turn_count,
        "total_cost": agent_cost + info.get("user_simulator_cost", 0),
        "info": info,
    }


def _build_error_result(task: dict, error: str) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "difficulty": task.get("difficulty"),
        "initial_query": task.get("initial_query"),
        "reward": 0.0,
        "recommended_products": [],
        "conversation_turns": 0,
        "total_cost": 0.0,
        "info": {"error": error},
    }


# ── Batch runner (incremental save + progress) ───────────────────────────

def _run_one_task(
    task_index: int,
    tasks: list[dict],
    catalog: pd.DataFrame,
    config: DomainConfig,
    agent_fn: Callable,
    user_model: str,
    max_turns: int,
    track_elicitation: bool,
    save_transcripts: bool,
    domain_dir: str | Path | None = None,
    desc_attrs: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Run a single task (thread-safe entry point for batch execution)."""
    task = tasks[task_index]
    tools = build_domain_tools(config, domain_dir=domain_dir)
    user = DomainSimulatedUser(config, model=user_model)
    env = DomainEnv(
        config, catalog, tasks,
        tools=tools,
        user=user,
        user_model=user_model,
        max_turns=max_turns,
        track_elicitation=track_elicitation,
        desc_attrs=desc_attrs,
    )
    agent = agent_fn()

    obs = env.reset(task_index=task_index)
    agent.reset(env)

    incoming: dict[str, Any] | None = {"role": "user", "content": obs.observation}
    total_agent_cost = 0.0
    messages: list[dict[str, Any]] = []
    step = 0
    max_steps = max_turns * 3

    while step < max_steps:
        try:
            agent_msg = agent.act(incoming)
            incoming = None
        except Exception as exc:
            return _build_error_result(task, str(exc))

        total_agent_cost += agent_msg.cost

        if agent_msg.tool_calls:
            for tc in agent_msg.tool_calls:
                try:
                    kwargs = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    kwargs = {}

                env_response = env.step(Action(name=tc.name, kwargs=kwargs))

                if save_transcripts:
                    messages.append({"type": "tool_call", "tool": tc.name, "args": kwargs})
                    messages.append({"type": "tool_result", "tool": tc.name,
                                     "content": env_response.observation[:500]})

                agent.add_to_history({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": env_response.observation,
                })

                if env_response.done:
                    result = _build_result(task, env, env_response, total_agent_cost)
                    if save_transcripts:
                        result["messages"] = messages
                    return result
            step += 1
            continue

        if agent_msg.content:
            env_response = env.step(
                Action("respond_to_user", {"content": agent_msg.content})
            )

            if save_transcripts:
                messages.append({"type": "agent", "content": agent_msg.content})
                messages.append({"type": "user", "content": env_response.observation})

            incoming = {"role": "user", "content": env_response.observation}
            step += 1

    env_response = env.step(Action("recommend_products", {"product_ids": []}))
    result = _build_result(task, env, env_response, total_agent_cost)
    if save_transcripts:
        result["messages"] = messages
    return result


def simulate_all_tasks(
    tasks: list[dict],
    catalog: pd.DataFrame,
    config: DomainConfig,
    *,
    agent_fn: Callable | None = None,
    agent_model: str = "gpt-4.1",
    user_model: str = "gpt-4.1-mini",
    max_turns: int = 20,
    max_concurrency: int = 1,
    output_path: str | Path | None = None,
    save_transcripts: bool = False,
    track_elicitation: bool = False,
    domain_dir: str | Path | None = None,
    desc_attrs: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    """Run all tasks with progress display and incremental JSONL saves.

    Args:
        tasks: List of task dicts (from pilot or full generation).
        catalog: Product catalog DataFrame (indexed by id_column).
        config: DomainConfig for this domain.
        agent_fn: Callable returning a fresh ConversationalAgent per task.
            If *None*, a default ``ToolCallingAgent`` is used.
        agent_model: LLM model for the default agent (ignored when *agent_fn* is set).
        user_model: LLM model for the simulated user.
        max_turns: Max conversation turns per task.
        max_concurrency: Parallel tasks (1 = sequential).
        output_path: Path to JSONL file for incremental saves.
        save_transcripts: Include conversation messages in results.
        track_elicitation: Run post-conversation elicitation analysis.
        desc_attrs: Description attributes for desc_* constraint evaluation.
    """
    if agent_fn is None:
        agent_fn = lambda: _make_default_agent(config, model=agent_model)
    if domain_dir is None:
        domain_dir = _infer_domain_dir(config)

    n = len(tasks)
    if n == 0:
        print("  No tasks to simulate.")
        return []

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear previous results
        output_path.write_text("")

    _print_header(
        f"🚀 BATCH SIMULATION: {n} tasks  |  "
        f"Agent: {agent_model}  |  User: {user_model}  |  "
        f"Concurrency: {max_concurrency}"
    )
    if output_path:
        print(f"  📁 Incremental results: {output_path}")
        print(f"     (tail -f {output_path} | python -m json.tool  to watch)")
    print()

    results: list[dict[str, Any] | None] = [None] * n
    start_time = time.time()
    completed = 0

    def _on_done(entry: dict, idx: int) -> None:
        nonlocal completed
        completed += 1
        elapsed = time.time() - start_time
        _print_batch_line(entry, completed - 1, n, elapsed)

        if output_path:
            with open(output_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    if max_concurrency > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = {
                pool.submit(
                    _run_one_task,
                    i, tasks, catalog, config,
                    agent_fn, user_model, max_turns,
                    track_elicitation, save_transcripts,
                    domain_dir, desc_attrs,
                ): i
                for i in range(n)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    entry = future.result()
                except Exception as exc:
                    entry = _build_error_result(tasks[idx], str(exc))
                results[idx] = entry
                _on_done(entry, idx)
    else:
        for i in range(n):
            try:
                entry = _run_one_task(
                    i, tasks, catalog, config,
                    agent_fn, user_model, max_turns,
                    track_elicitation, save_transcripts,
                    domain_dir, desc_attrs,
                )
            except Exception as exc:
                entry = _build_error_result(tasks[i], str(exc))
            results[i] = entry
            _on_done(entry, i)

    elapsed = time.time() - start_time
    final_results = [r for r in results if r is not None]

    # Summary
    _print_header("📊 SIMULATION SUMMARY")
    print(f"\n  ⏱️  Total time: {elapsed / 60:.1f}m ({elapsed / max(n, 1):.1f}s per task)")
    total_cost = sum(r.get("total_cost", 0) for r in final_results)
    print(f"  💰 Total cost: ${total_cost:.4f}")

    if final_results:
        metrics = aggregate_results(final_results)
        print()
        print_results_table(metrics, agent_name=f"{config.item_noun} simulation")
        print()
        print_ranking_table(metrics, agent_name=f"{config.item_noun} simulation")

    if output_path:
        print(f"\n  📁 Full results saved to: {output_path}")

    return final_results
