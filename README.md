# Query2Cart

A benchmark for evaluating conversational recommendation agents on multi-turn constraint elicitation and preference learning.

Query2Cart measures how well an AI agent can guide a simulated user through a product search conversation -- asking the right questions, discovering hidden constraints, learning soft preferences, and ultimately recommending products that match what the user actually wants. It ships with four ready-made domains and a generation pipeline to create benchmarks from any product catalog.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pre-built Domains](#pre-built-domains)
- [Running the Benchmark](#running-the-benchmark)
- [Baseline Agents](#baseline-agents)
- [Building a Custom Agent](#building-a-custom-agent)
- [Architecture](#architecture)
- [Task Anatomy](#task-anatomy)
- [Difficulty Tiers](#difficulty-tiers)
- [Metrics](#metrics)
- [Generating a New Domain](#generating-a-new-domain)
- [Catalog embeddings](#catalog-embeddings)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [License](#license)

---

## Key Features

- **Domain-agnostic** -- constraints, tools, prompts, and evaluation are all generated from a single JSON config. No hardcoded domain logic.
- **Realistic conversations** -- an LLM-based simulated user reveals constraints gradually (proactively, responsively, reactively, or contextually) based on behavioral profiles.
- **Calibrated difficulty** -- five difficulty tiers from wide-open searches to over-constrained infeasible tasks, each with controlled pool sizes and constraint counts.
- **Continuous evaluation** -- goes beyond binary hit/miss with utility-weighted ranking metrics (NDCG, graded F1).
- **Pluggable agents** -- bring your own agent by implementing a simple interface. Supports both LLM-based conversational agents and rule-based baselines.
- **End-to-end generation** -- create a complete benchmark from any product catalog (parquet file) using the LLM-powered generation pipeline.
- **Cost tracking** -- per-task LLM cost aggregation for both agent and simulated user.
- **Multi-trial support** -- run multiple seeds and get mean +/- std across trials.

---

## Installation

**Requirements:** Python >= 3.10

```bash
# Core (running benchmarks)
pip install -e .

# With generation pipeline (creating new domains)
pip install -e ".[generation]"
```

Version pins for core libraries (`pandas`, `numpy`, `pydantic`, `litellm`, `scikit-learn`, `tqdm`, and optional generation extras) are listed in [`pyproject.toml`](pyproject.toml).

### LLM Configuration

Query2Cart uses [LiteLLM](https://docs.litellm.ai/) for LLM calls. Set the appropriate API key for your provider:

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Quick Start

After [installation](#installation) and API keys are set:

```bash
git clone <repo-url> && cd query2cart
pip install -e .

python examples/run_benchmark.py
```

The example evaluates a subset of tasks on the `games` domain and prints a summary. For scripted use (other domains, sampling, saving results), see [Running the Benchmark](#running-the-benchmark).

Example summary (columns are documented under [Metrics](#metrics)):

```
Difficulty              N  Reward  Success   Util    CSR  C.Elic  P.Elic  Turns
────────────────────────────────────────────────────────────────────────────────
Overall                20   0.612   75.0%   74.3%  82.1%  68.5%   55.2%    8.3
Small                   6   0.734   83.3%   80.1%  91.7%  72.0%   60.1%    7.1
Medium                  6   0.618   75.0%   73.2%  83.3%  65.4%   52.8%    8.5
Large                   4   0.542   75.0%   68.5%  75.0%  70.3%   55.0%    9.2
Oc-Feasible             2   0.500   50.0%   65.0%  50.0%  64.0%   48.5%   10.0
Oc-Infeasible           2   0.500   50.0%     —      —    71.0%     —      6.5
```

---

## Pre-built Domains

Four domains ship ready to use in `datasets/`:

| Domain | Products | Tasks | Constraints | Pref-Eligible Attributes | Description |
|---|---|---|---|---|---|
| **Laptops** | 8,000+ | 220 | 10 | 8 | Consumer laptops with specs, pricing, and features |
| **Beauty** | 8,000+ | 250 | 11 | 3 | Skincare and beauty products |
| **Cars** | 4,000+ | 150 | 13 | 3 | Used car listings with detailed specs |
| **Games** | 50,000+ | 250 | 11 | 5 | Video games across platforms and genres |

Each domain directory contains:

```
datasets/<domain>/
  config.json           # Domain specification (attributes, constraints, triggers, difficulty)
  tasks.json            # Benchmark tasks with user profiles and revelation plans
  catalogue.parquet     # Product catalog
  tools.py              # Domain-specific tool implementations (auto-generated)
```

Semantic search also needs [catalog embedding files](#catalog-embeddings) (often under `extras/`).

---

## Running the Benchmark

### Basic Usage

```python
from simulation.benchmark import Benchmark

bench = Benchmark.load("games")  # or "laptops", "beauty", "cars"
results = bench.run(
    agent_model="gpt-4.1",
    user_model="gpt-4.1-mini",
    max_turns=20,
    n_tasks=20,
    seed=42,
)
bench.report(results)
```

### Per-Difficulty Sampling

Control how many tasks to sample from each difficulty tier:

```python
results = bench.run(
    agent_model="gpt-4.1",
    difficulty_counts={"small": 10, "medium": 10, "large": 5, "oc_feasible": 3, "oc_infeasible": 2},
    seed=42,
)
```

### Loading from a Custom Directory

```python
bench = Benchmark.load("./my_generated_benchmark")
```

### Results

Results are saved automatically as JSONL to `results/`. Each line is a per-task result dict containing reward, conversation turns, recommended products, constraint satisfaction details, and ranking metrics.

---

## Baseline Agents

Four baseline agents are included for comparison:

| Agent | Type | Conversation | Description |
|---|---|---|---|
| **ToolCallingAgent** | LLM-based | Multi-turn | Default agent. Uses tools to search, filter, and verify before recommending. |
| **RandomAgent** | Rule-based | None | Searches once, recommends random products. No conversation. |
| **SingleTurnRAGAgent** | Rule-based | None | One semantic search on the initial query, recommends top-K. |
| **FilterOracleAgent** | Oracle | None | Has direct access to ground-truth constraints. Upper-bound baseline. |

[`examples/run_benchmark.py`](examples/run_benchmark.py) uses the default **ToolCallingAgent** when you pass `agent_model` to `Benchmark.run()` (same pattern as above).

---

## Building a Custom Agent

Implement either `BaseAgent` (for non-LLM agents) or `ConversationalAgent` (for LLM-based agents) and pass it to the benchmark.

### ConversationalAgent (LLM-based)

```python
from simulation.agents.base import ConversationalAgent
from shared.types import AgentMessage, ToolCallInfo
from shared.llm import completion

class MyAgent(ConversationalAgent):
    def __init__(self, model: str = "gpt-4.1"):
        super().__init__()
        self.model = model

    def reset(self, env=None) -> None:
        self._messages = [
            {"role": "system", "content": "You are a helpful shopping assistant."}
        ]
        if env is not None:
            self.tool_schemas = env.get_tool_schemas()

    def act(self, message: dict | None = None) -> AgentMessage:
        if message is not None:
            self._messages.append(message)

        response = completion(
            model=self.model,
            messages=self._messages,
            tools=self.tool_schemas,
            temperature=0.0,
        )

        choice = response.choices[0].message
        # Build assistant message for history
        assistant_msg = {"role": "assistant"}
        if choice.content:
            assistant_msg["content"] = choice.content
        if choice.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in choice.tool_calls
            ]
        self._messages.append(assistant_msg)

        tool_calls = None
        if choice.tool_calls:
            tool_calls = [
                ToolCallInfo(id=tc.id, name=tc.function.name,
                             arguments=tc.function.arguments)
                for tc in choice.tool_calls
            ]
        return AgentMessage(content=choice.content, tool_calls=tool_calls)

# Run it
bench = Benchmark.load("games")
results = bench.run(agent_fn=lambda: MyAgent(model="gpt-4.1"))
bench.report(results)
```

### Agent Interface

Your agent interacts with the environment through a loop:

1. **`reset(env)`** -- called once per task. Use `env.get_tool_schemas()` to get available tools.
2. **`act(message)`** -- called each turn. Receives the latest user message (or tool result). Return an `AgentMessage` with text content and/or tool calls.

**Terminal actions** (returned as tool calls by name):
- `recommend_products` -- recommend a set of product IDs (triggers evaluation)
- `declare_infeasible` -- declare that no products match the user's requirements
- `respond_to_user` -- send a text response to the user (continues conversation)

**Catalog tools** (auto-generated per domain):
- `filter_<domain>` -- structured filtering with sort and limit
- `search_<domain>` -- keyword (TF-IDF) and semantic (embedding) search
- `get_<domain>_details` -- full product details by ID
- `get_catalog_stats` -- summary statistics of the catalog

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │            Generation Pipeline           │
                    │  catalog.parquet ──► config.json          │
                    │                  ──► tasks.json           │
                    │                  ──► embeddings           │
                    └────────────────────┬─────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────┐
                    │           Simulation Environment          │
                    │                                           │
                    │  ┌─────────┐    ┌──────────────────┐     │
                    │  │  Agent  │◄──►│   DomainEnv      │     │
                    │  │ (yours) │    │  ┌────────────┐  │     │
                    │  └─────────┘    │  │ Catalog    │  │     │
                    │       │         │  │ Tools      │  │     │
                    │       │         │  │ Simulated  │  │     │
                    │       ▼         │  │ User (LLM) │  │     │
                    │   Tool calls    │  └────────────┘  │     │
                    │   & responses   └──────────────────┘     │
                    │       │                                   │
                    │       ▼                                   │
                    │  ┌──────────────────────────────────┐    │
                    │  │  Evaluation (per recommendation)  │    │
                    │  │  • Hard constraint checking        │    │
                    │  │  • Preference utility scoring      │    │
                    │  │  • NDCG / Graded P-R-F1            │    │
                    │  │  • Elicitation completeness         │    │
                    │  └──────────────────────────────────┘    │
                    └──────────────────────────────────────────┘
```

### Conversation Flow

1. The **simulated user** opens with an initial query (e.g., *"I need a lightweight laptop for travel under $1000"*).
2. The **agent** can respond to the user, call catalog tools, or make a recommendation.
3. When the agent responds, the **simulated user** replies based on its behavioral profile, gradually revealing constraints and preferences.
4. Constraints are revealed through four modes:
   - **Proactive** -- included in the initial query
   - **Responsive** -- revealed when the agent asks the right question (trigger-based)
   - **Reactive** -- revealed when a recommendation violates the constraint
   - **Contextual** -- revealed naturally during conversation
5. The conversation continues until the agent calls `recommend_products` or `declare_infeasible`, or the turn limit is reached.
6. The environment evaluates the recommendation against ground-truth constraints and preferences.

---

## Task Anatomy

Each task in `tasks.json` defines a complete evaluation scenario:

```json
{
  "task_id": "task_0042",
  "difficulty": "medium",
  "initial_query": "Looking for a gaming laptop with good cooling and at least 32GB RAM",
  "user_profile": {
    "expertise": "intermediate",
    "hard_constraints": {
      "price_max_usd": 1500,
      "ram_min_gb": 32,
      "gpu_type": "dedicated"
    },
    "use_case": "gaming and streaming",
    "attribute_preferences": [
      {
        "attribute": "weight_kg",
        "direction": "minimize",
        "priority": 1,
        "revelation_mode": "responsive",
        "trigger": "agent_asks_portability"
      },
      {
        "attribute": "user_rating",
        "direction": "maximize",
        "priority": 2,
        "revelation_mode": "contextual"
      }
    ],
    "constraint_revelation_plan": {
      "proactive": ["price_max_usd", "ram_min_gb"],
      "responsive": {
        "agent_asks_gpu": ["gpu_type"]
      }
    },
    "behavioral_profile": {
      "patience_turns": 9,
      "response_verbosity": "brief"
    }
  }
}
```

- **Hard constraints** are binary pass/fail requirements the user has in mind.
- **Attribute preferences** are soft "nice to have" dimensions with priority ordering.
- **The revelation plan** controls when and how each constraint surfaces in conversation.
- **The behavioral profile** shapes the simulated user's patience and communication style.

---

## Difficulty Tiers

Tasks are organized into five difficulty tiers based on how constrained the search space is:

| Tier | Pool Size | Min Constraints | Description |
|---|---|---|---|
| **Small** | 20 -- 80 | 3 | Wide search space. Many products match. |
| **Medium** | 10 -- 50 | 4 | Moderate filtering needed. |
| **Large** | 5 -- 25 | 5 | Narrow search. Strong constraints. |
| **OC-Feasible** | 1 -- 3 | 4 | Over-constrained but solvable. Very few matches. |
| **OC-Infeasible** | 0 | 4 | No products satisfy all constraints. Agent should call `declare_infeasible`. |

Pool size = number of products in the catalog that satisfy all hard constraints for that task.

---

## Metrics

### Primary Metrics

| Metric | Formula / Description | Range |
|---|---|---|
| **Reward** | `I(all hard constraints met) * (0.5 + 0.5 * preference_utility)` | [0, 1] |
| **Success Rate** | Fraction of tasks where reward > 0 (i.e., all hard constraints satisfied) | [0, 1] |
| **Preference Utility** | Weighted score of how well recommendations match the user's soft preferences (geometric decay by priority) | [0, 1] |
| **Constraint Satisfaction Rate (CSR)** | Fraction of recommended products that satisfy all hard constraints | [0, 1] |
| **Constraint Elicitation** | Fraction of the user's hard constraints the agent successfully discovered during conversation | [0, 1] |
| **Preference Elicitation** | Fraction of the user's soft preferences the agent discovered | [0, 1] |
| **Avg Turns** | Mean number of conversation turns before the agent makes a recommendation | integer |

### Ranking Metrics

| Metric | Description | Range |
|---|---|---|
| **NDCG@k** (k=1,3,5,n) | Normalized Discounted Cumulative Gain using continuous utility scores as relevance | [0, 1] |
| **Graded Precision** | Average relevance of recommended items, normalized by the best possible utility | [0, 1] |
| **Graded Recall** | Fraction of total pool utility captured by the agent's recommendations | [0, 1] |
| **Graded F1** | Harmonic mean of graded precision and graded recall | [0, 1] |

### How Reward Works

The reward function decomposes into two independent components:

1. **Hard constraint gate** -- a binary indicator that is 1 only if every recommended product satisfies all of the user's hard constraints (price ceiling, minimum RAM, required brand, etc.). A single violation zeros out the entire reward.

2. **Preference utility** -- a continuous score in [0, 1] measuring how well the recommendation matches the user's soft preferences (e.g., "lighter is better", "higher rating preferred"). Preferences are priority-weighted using geometric decay: priority 1 gets 2x the weight of priority 2, which gets 2x the weight of priority 3.

```
reward = I(all hard constraints met) * (0.5 + 0.5 * preference_utility)
```

A recommendation that satisfies constraints but ignores preferences scores 0.5. One that perfectly nails both scores 1.0. One that violates any constraint scores 0.0.

All metrics are reported both overall and broken down by difficulty tier.

---

## Generating a New Domain

Create a benchmark from any product catalog using the generation pipeline.

### Prerequisites

- A product catalog as a Parquet file
- LLM API access (for column classification, config generation, and task creation)

```bash
pip install -e ".[generation]"
```

### Usage

```python
from generation.pipeline import Pipeline

pipeline = Pipeline(
    catalog="path/to/your/catalog.parquet",
    domain="brief description of your product domain",
    item_noun="product",       # e.g., "laptop", "car", "game"
    output_dir="./my_benchmark",
)

# Run stage by stage for review and control:
pipeline.profile()                  # Analyze catalog columns
pipeline.triage()                   # Classify column roles (review checkpoints/triage_result.json)
pipeline.configure()                # Generate domain config (review config.json)
pipeline.generate_coherence()       # Create coherence rules
pipeline.calibrate()                # Validate difficulty calibration
pipeline.extract_descriptions()     # Extract attributes from text fields
pipeline.generate_tasks(n_tasks=250)  # Produce tasks.json

# Or run everything at once:
# pipeline.run()
```

### Generation Pipeline Stages

| Stage | What It Does | Output |
|---|---|---|
| **Profile** | Statistical analysis of every catalog column (cardinality, nulls, distributions) | In-memory profile |
| **Triage** | LLM classifies columns as ID, hard filter, soft preference, set filter, embedding text, or drop | `checkpoints/triage_result.json` |
| **Configure** | Generates the full `DomainConfig` (attributes, constraints, triggers, difficulty brackets, prompts) | `config.json` |
| **Coherence** | LLM produces domain-specific coherence rules (e.g., "diesel cars don't have battery range") | `checkpoints/coherence.py` |
| **Calibrate** | Validates that difficulty brackets produce viable pool sizes | Warnings/errors |
| **Extract Descriptions** | Three-phase LLM extraction of structured attributes from product text fields | `extras/description_attributes.json` (+ embeddings; see [below](#catalog-embeddings)) |
| **Generate Tasks** | Samples constraint sets, finds matching product pools, generates user profiles and revelation plans | `tasks.json` |

Each stage is checkpointed -- you can stop, review, edit intermediate outputs, and resume.

---

## Catalog embeddings

Place **`all_embedding_text_embeddings.npy`** and **`embedding_product_ids.json`** in **`datasets/<domain>/extras/`** (or the domain root; `extras/` is preferred). They must match that domain’s catalog.

**Source:** Run **`pipeline.extract_descriptions()`** or **`pipeline.run()`** (uses your embedding API; same setup as [LLM Configuration](#llm-configuration)), **or** download pre-built files from Kaggle *(URL TBD)* and unpack into `extras/`.

---

## Project Structure

```
query2cart/
├── datasets/                      # Pre-built benchmark domains
│   ├── laptops/
│   ├── beauty/
│   ├── cars/
│   └── games/
│       ├── config.json            # Domain specification
│       ├── tasks.json             # Benchmark tasks
│       ├── catalogue.parquet      # Product catalog
│       └── tools.py               # Auto-generated domain tools
├── examples/
│   ├── run_benchmark.py           # Run evaluation on a pre-built domain
│   ├── custom_agent.py            # Plug in your own agent
│   └── generate_domain.py         # Create a new domain from a catalog
├── generation/                    # Task generation pipeline
│   ├── pipeline.py                # Orchestrates the multi-stage pipeline
│   ├── profile.py                 # Statistical column profiling
│   ├── triage.py                  # LLM-based column role classification
│   ├── configure.py               # DomainConfig generation
│   ├── calibrate.py               # Difficulty calibration validation
│   ├── desc_extraction.py         # LLM-based attribute extraction from text
│   ├── generate_tasks.py          # Task sampling and user profile generation
│   ├── catalog_hygiene.py         # Catalog cleaning and normalization
│   ├── tool_generator.py          # Domain tool code generation
│   ├── onboarding.py              # Interactive setup wizard
│   └── prompts/                   # LLM prompt templates
├── simulation/                    # Evaluation environment
│   ├── benchmark.py               # Main entry point (load + run + report)
│   ├── env.py                     # Conversation environment (step/reset)
│   ├── runner.py                  # Single-task and batch simulation
│   ├── tools.py                   # Domain-agnostic tool generation
│   ├── user.py                    # LLM-based simulated user
│   ├── metrics.py                 # All evaluation metrics
│   └── agents/
│       ├── base.py                # BaseAgent and ConversationalAgent interfaces
│       ├── tool_calling.py        # Default LLM agent
│       ├── random.py              # Random baseline
│       ├── single_turn_rag.py     # Semantic search baseline
│       └── filter_oracle.py       # Oracle ceiling baseline
├── shared/                        # Core infrastructure
│   ├── config.py                  # DomainConfig schema (Pydantic)
│   ├── types.py                   # Message and result types
│   ├── filter.py                  # Generic constraint filtering engine
│   ├── scoring.py                 # Preference utility scoring
│   └── llm.py                     # Centralized LLM interface
├── pyproject.toml
└── requirements.txt
```

---

## Configuration Reference

The `DomainConfig` (`config.json`) is the single source of truth for a domain. Key sections:

| Section | Purpose |
|---|---|
| `attributes` | Catalog column definitions (type, unit, filterable, preference-eligible) |
| `constraints` | User-facing hard constraints with operators, sampling values, and display templates |
| `triggers` | Conversational triggers that unlock responsive constraint revelation |
| `difficulty` | Pool size ranges and minimum constraint counts per difficulty tier |
| `prompt_fragments` | Domain-specific text for LLM prompts (domain description, query rules, expertise levels) |

### Supported Constraint Operators

| Operator | Description | Example |
|---|---|---|
| `lte` / `gte` | Less/greater than or equal | Price <= $1000 |
| `eq` / `neq` | Equals / not equals | Brand = "Apple" |
| `in_set` | Value is one of a set | Category in {Action, RPG} |
| `contains` | Collection contains value | Ports contains "USB-C" |
| `contains_all` | Collection contains all values | Tags contains all {wireless, bluetooth} |
| `contains_any` | Collection contains any value | Genres contains any {action, adventure} |
| `not_contains` | Collection does not contain | Ingredients not contains "paraben" |
| `substring` | Text contains substring | Name contains "Pro" |
| `boolean` | True/false flag | Is_organic = true |
| `range` | Value within range | Year in [2020, 2024] |

---

## License

See [LICENSE](LICENSE) for details.
