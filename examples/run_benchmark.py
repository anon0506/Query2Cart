"""Evaluate an agent on a pre-built Query2Cart domain.

Usage:
    # Load by name (from datasets/)
    python examples/run_benchmark.py

    # Load by directory path (e.g. after generation)
    python examples/run_benchmark.py ./datasets/games
"""

from simulation.benchmark import Benchmark

bench = Benchmark.load("games")
print(f"Loaded {len(bench.tasks)} tasks for '{bench.domain_name}'")
print(f"  Description attributes: {len(bench.desc_attrs)} products")
print(f"  Embeddings: {'yes' if bench.embeddings is not None else 'no'}")

# Run a random subset: 10 tasks total
results = bench.run(
    agent_model="gpt-4.1",
    user_model="gpt-4.1-mini",
    max_turns=20,
    n_tasks=20,
    seed=42,
)

# Or run with per-difficulty buckets:
# results = bench.run(
#     agent_model="gpt-4.1",
#     difficulty_counts={"small": 3, "medium": 4, "large": 3},
#     seed=42,
# )

# Results are saved automatically to results/<domain>_results.jsonl
bench.report(results)
