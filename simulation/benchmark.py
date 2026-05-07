"""Benchmark --- main entry point for evaluating agents on Query2Cart.

Usage::

    from simulation.benchmark import Benchmark

    bench = Benchmark.load("games")
    results = bench.run(agent_model="gpt-4.1")
    bench.report(results)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import random

import numpy as np

from shared.config import DomainConfig
from simulation.runner import simulate_single_task, simulate_all_tasks
from simulation.metrics import aggregate_results, print_results_table, print_ranking_table


DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DOMAINS = ("games", "beauty", "cars")


def _find_file(domain_dir: Path, name: str) -> Path | None:
    """Look for *name* in ``extras/``, then root of *domain_dir*."""
    for candidate in (domain_dir / "extras" / name, domain_dir / name):
        if candidate.is_file():
            return candidate
    return None


def _load_desc_attrs(domain_dir: Path) -> tuple[dict[str, dict], dict | None]:
    """Load description_attributes.json from *domain_dir* (extras/ first)."""
    path = _find_file(domain_dir, "description_attributes.json")
    if path is None:
        return {}, None
    from generation.desc_extraction import load_desc_attrs
    return load_desc_attrs(path)


def _load_embeddings(domain_dir: Path) -> tuple[np.ndarray | None, list[str] | None]:
    """Load precomputed embedding vectors and their product-ID index."""
    emb_path = _find_file(domain_dir, "all_embedding_text_embeddings.npy")
    ids_path = _find_file(domain_dir, "embedding_product_ids.json")
    if emb_path is None or ids_path is None:
        return None, None
    embeddings = np.load(str(emb_path))
    with open(ids_path) as f:
        product_ids = json.load(f)
    return embeddings, product_ids


class Benchmark:
    """Load a domain and run agents against it."""

    def __init__(
        self,
        config: DomainConfig,
        catalog: pd.DataFrame,
        tasks: list[dict],
        domain_dir: str | Path | None = None,
        desc_attrs: dict[str, dict] | None = None,
        desc_catalog: dict | None = None,
        embeddings: np.ndarray | None = None,
        embedding_ids: list[str] | None = None,
    ):
        self.config = config
        self.catalog = catalog
        self.tasks = tasks
        self.domain_dir = domain_dir
        self.desc_attrs = desc_attrs or {}
        self.desc_catalog = desc_catalog
        self.embeddings = embeddings
        self.embedding_ids = embedding_ids

    @classmethod
    def _load_from_dir(cls, domain_dir: Path) -> "Benchmark":
        """Load all runtime artifacts from a single directory."""
        config_path = domain_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"No config.json in {domain_dir}. "
                f"This domain may need task generation first."
            )

        config = DomainConfig.load(str(config_path))

        catalog_path = domain_dir / "catalogue.parquet"
        if not catalog_path.is_file():
            catalog_path = domain_dir / "cleaned_catalogue.parquet"
        catalog = pd.read_parquet(catalog_path)

        id_col = config.id_column
        if id_col in catalog.columns:
            catalog = catalog.set_index(id_col, drop=False)

        with open(domain_dir / "tasks.json") as f:
            tasks = json.load(f)

        desc_attrs, desc_catalog = _load_desc_attrs(domain_dir)
        embeddings, embedding_ids = _load_embeddings(domain_dir)

        return cls(
            config, catalog, tasks,
            domain_dir=str(domain_dir),
            desc_attrs=desc_attrs,
            desc_catalog=desc_catalog,
            embeddings=embeddings,
            embedding_ids=embedding_ids,
        )

    @classmethod
    def load(cls, domain: str) -> "Benchmark":
        """Load a domain by short name (from ``datasets/``) or directory path.

        Examples::

            Benchmark.load("games")                   # datasets/games/
            Benchmark.load("datasets/games")           # relative path
            Benchmark.load("/abs/path/to/my_domain")   # absolute path
        """
        candidate = Path(domain)
        if candidate.is_dir():
            return cls._load_from_dir(candidate)

        domain_dir = DATASETS_DIR / domain
        if domain_dir.is_dir():
            return cls._load_from_dir(domain_dir)

        available = [d.name for d in DATASETS_DIR.iterdir() if d.is_dir()]
        raise ValueError(
            f"Unknown domain '{domain}'. Available: {available}"
        )

    @classmethod
    def from_files(
        cls,
        config_path: str,
        catalog_path: str,
        tasks_path: str,
    ) -> "Benchmark":
        """Load from custom file paths."""
        config = DomainConfig.load(config_path)
        catalog = pd.read_parquet(catalog_path)

        id_col = config.id_column
        if id_col in catalog.columns:
            catalog = catalog.set_index(id_col, drop=False)

        with open(tasks_path) as f:
            tasks = json.load(f)

        domain_dir = Path(config_path).parent
        desc_attrs, desc_catalog = _load_desc_attrs(domain_dir)
        embeddings, embedding_ids = _load_embeddings(domain_dir)

        return cls(
            config, catalog, tasks,
            domain_dir=str(domain_dir),
            desc_attrs=desc_attrs,
            desc_catalog=desc_catalog,
            embeddings=embeddings,
            embedding_ids=embedding_ids,
        )

    def _sample_tasks(
        self,
        tasks: list[dict],
        *,
        n_tasks: int | None = None,
        difficulty_counts: dict[str, int] | None = None,
        seed: int | None = None,
    ) -> list[dict]:
        """Return a (possibly sampled) subset of *tasks*.

        Priority: *difficulty_counts* > *n_tasks* > all tasks.
        """
        if difficulty_counts is not None:
            rng = random.Random(seed)
            by_diff: dict[str, list[dict]] = {}
            for t in tasks:
                by_diff.setdefault(t.get("difficulty", "unknown"), []).append(t)
            selected: list[dict] = []
            for diff, count in difficulty_counts.items():
                pool = by_diff.get(diff, [])
                if not pool:
                    continue
                selected.extend(rng.sample(pool, min(count, len(pool))))
            return selected

        if n_tasks is not None and n_tasks < len(tasks):
            rng = random.Random(seed)
            return rng.sample(tasks, n_tasks)

        return tasks

    def run(
        self,
        *,
        agent=None,
        agent_fn: Callable | None = None,
        agent_model: str = "gpt-4.1",
        user_model: str = "gpt-4.1-mini",
        max_turns: int = 20,
        max_concurrency: int = 1,
        task_ids: list[str] | None = None,
        n_tasks: int | None = None,
        difficulty_counts: dict[str, int] | None = None,
        seed: int | None = None,
        output_path: str | None = None,
        save_transcripts: bool = False,
        track_elicitation: bool = False,
    ) -> list[dict[str, Any]]:
        """Run agents on all (or selected) tasks.

        Args:
            agent: A single ConversationalAgent instance (for sequential runs).
            agent_fn: Callable returning a fresh agent per task (for concurrent
                runs or to ensure clean state).  Takes precedence over *agent*.
            agent_model: LLM model name for the default agent (used when neither
                *agent* nor *agent_fn* is provided).
            user_model: LLM model for the simulated user.
            max_turns: Maximum conversation turns per task.
            max_concurrency: Number of tasks to run in parallel.
            task_ids: Subset of task IDs to run.  ``None`` = all.
            n_tasks: Run this many tasks chosen at random.  Ignored when
                *task_ids* or *difficulty_counts* are set.
            difficulty_counts: ``{"small": 5, "large": 10}`` — sample exactly
                this many tasks per difficulty bucket.
            seed: Random seed for task sampling reproducibility.
            output_path: JSONL path for incremental result saves.  Defaults
                to ``results/<domain>_results.jsonl``.
            save_transcripts: Include message transcripts in results.
            track_elicitation: Compute constraint elicitation metrics.
        """
        tasks = self.tasks
        if task_ids is not None:
            id_set = set(task_ids)
            tasks = [t for t in self.tasks if t.get("task_id") in id_set]
        else:
            tasks = self._sample_tasks(
                tasks,
                n_tasks=n_tasks,
                difficulty_counts=difficulty_counts,
                seed=seed,
            )

        if output_path is None:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = str(
                RESULTS_DIR / f"{self.config.name}_results.jsonl"
            )

        if agent_fn is None and agent is not None:
            _agent = agent
            agent_fn = lambda: _agent

        return simulate_all_tasks(
            tasks,
            self.catalog,
            self.config,
            agent_fn=agent_fn,
            agent_model=agent_model,
            user_model=user_model,
            max_turns=max_turns,
            max_concurrency=max_concurrency,
            output_path=output_path,
            save_transcripts=save_transcripts,
            track_elicitation=track_elicitation,
            domain_dir=self.domain_dir,
            desc_attrs=self.desc_attrs,
        )

    def run_single(
        self,
        task_id: str,
        *,
        agent=None,
        agent_model: str = "gpt-4.1",
        user_model: str = "gpt-4.1-mini",
        max_turns: int = 20,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run a single task with verbose output (for debugging)."""
        task = None
        for t in self.tasks:
            if t.get("task_id") == task_id:
                task = t
                break
        if task is None:
            raise ValueError(f"Task '{task_id}' not found")

        return simulate_single_task(
            task,
            self.catalog,
            self.config,
            agent=agent,
            agent_model=agent_model,
            user_model=user_model,
            max_turns=max_turns,
            verbose=verbose,
            domain_dir=self.domain_dir,
            desc_attrs=self.desc_attrs,
        )

    def report(self, results: list[dict]) -> None:
        """Print aggregate metrics summary."""
        if not results:
            print("No results to report.")
            return
        metrics = aggregate_results(results)
        print_results_table(metrics, agent_name=self.config.item_noun)
        print()
        print_ranking_table(metrics, agent_name=self.config.item_noun)

    @property
    def task_ids(self) -> list[str]:
        return [t.get("task_id", str(i)) for i, t in enumerate(self.tasks)]

    @property
    def domain_name(self) -> str:
        return self.config.item_noun
