"""Pipeline --- orchestrates the 5-stage task generation process.

Usage::

    from generation.pipeline import Pipeline

    p = Pipeline(
        catalog="path/to/catalog.parquet",
        domain="cooking recipes",
        item_noun="recipe",
        output_dir="./datasets/recipes",
    )
    p.run()   # all stages, checkpointed

    # Or stage by stage:
    p.profile()
    p.triage()         # review output_dir/checkpoints/triage_result.json
    p.configure()      # review output_dir/config.json
    p.calibrate()      # check diagnostics
    p.generate_tasks() # produces output_dir/tasks.json

Directory layout after a full run::

    output_dir/
        config.json                    # ← needed at runtime
        tasks.json                     # ← needed at runtime
        catalogue.parquet              # ← needed at runtime (cleaned)
        extras/
            description_attributes.json    # ← desc constraint eval
            all_embedding_text_embeddings.npy
            embedding_product_ids.json
            tools.py                       # ← domain-specific agent tools
        checkpoints/
            profile.pkl
            triage_result.json
            coherence.py
            _checkpoint.json
            _desc_phase0_guidelines.txt
            _desc_phase1_raw.json
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_cached(primary: Path, fallback: Path | None) -> Path | None:
    """Return *primary* if it exists, else *fallback* if it exists, else None."""
    if primary.exists():
        return primary
    if fallback is not None and fallback.exists():
        return fallback
    return None


def _distribute_tasks(n_tasks: int, config) -> dict[str, int]:
    """Distribute *n_tasks* across difficulty tiers proportionally to their config target_count."""
    raw = {name: bracket.target_count for name, bracket in config.difficulty.items()}
    total_raw = sum(raw.values())
    if total_raw == 0:
        names = list(raw.keys())
        per = n_tasks // len(names) if names else 0
        return {n: per for n in names}
    counts = {name: max(1, round(n_tasks * cnt / total_raw)) for name, cnt in raw.items()}
    # Adjust rounding so the total matches n_tasks exactly
    diff = n_tasks - sum(counts.values())
    if diff != 0:
        ordered = sorted(counts, key=lambda d: raw[d], reverse=(diff > 0))
        for d in ordered:
            if diff == 0:
                break
            step = 1 if diff > 0 else -1
            counts[d] += step
            diff -= step
    return counts


class Pipeline:
    """Checkpointed pipeline from raw catalog to benchmark tasks."""

    STEP_NAMES = (
        "TRIAGE",
        "CONFIGURE",
        "COHERENCE",
        "EXTRACT_DESCRIPTIONS",
        "EXTRACT_DESCRIPTIONS_DISCOVERY",
        "EXTRACT_DESCRIPTIONS_EXTRACTION",
        "EXTRACT_DESCRIPTIONS_NORMALIZATION",
        "GENERATE_TASKS",
    )

    def __init__(
        self,
        catalog: str | pd.DataFrame,
        domain: str,
        item_noun: str,
        output_dir: str | Path,
        *,
        model: str = "o4-mini",
        model_map: dict[str, str] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.extras_dir = self.output_dir / "extras"
        self.extras_dir.mkdir(parents=True, exist_ok=True)
        self.domain = domain
        self.item_noun = item_noun
        self.model = model
        self.model_map: dict[str, str] = model_map or {}

        if isinstance(catalog, (str, Path)):
            self._catalog_path = Path(catalog)
            self._catalog: pd.DataFrame | None = None
        else:
            self._catalog_path = None
            self._catalog = catalog

    @property
    def catalog(self) -> pd.DataFrame:
        if self._catalog is None:
            assert self._catalog_path is not None
            self._catalog = pd.read_parquet(self._catalog_path)
        return self._catalog

    def run(self, *, skip_descriptions: bool = False) -> None:
        """Run all stages sequentially.

        Stages: profile → triage → configure → generate_coherence
        → calibrate → extract_descriptions → generate_tasks.
        """
        self.profile()
        self.triage()
        self.configure()
        self.generate_coherence()
        self.calibrate()
        if not skip_descriptions:
            self.extract_descriptions()
        self.generate_tasks()

    def profile(self) -> Any:
        """Stage 1: Analyze catalog columns (types, distributions, quality)."""
        from generation.profile import profile_dataset

        cache = self.checkpoint_dir / "profile.pkl"
        hit = _resolve_cached(cache, self.output_dir / "profile.pkl")
        if hit is not None:
            print(f"  Loading cached profile from {hit}")
            with open(hit, "rb") as f:
                result = pickle.load(f)
            if hit != cache:
                shutil.copy2(hit, cache)
            return result

        print("  Stage 1: Profiling catalog...")
        from generation.profile import profile_dataset_from_path

        if self._catalog_path is not None:
            result = profile_dataset_from_path(
                str(self._catalog_path),
                max_rows=200_000,
            )
        else:
            result = profile_dataset(self.catalog)
        with open(cache, "wb") as f:
            pickle.dump(result, f)
        print(f"  Profile saved to {cache}")
        return result

    def triage(self) -> Any:
        """Stage 2: Classify columns into roles (hard_filter, soft_pref, etc.)."""
        from generation.triage import TriageResult, triage_columns

        cache = self.checkpoint_dir / "triage_result.json"
        hit = _resolve_cached(cache, self.output_dir / "triage_result.json")
        profile_data = self.profile()

        if hit is not None:
            print(f"  Loading cached triage from {hit}")
            result = TriageResult.load(str(hit))
            if hit != cache:
                shutil.copy2(hit, cache)
            return result

        print("  Stage 2: Triaging columns...")
        result = triage_columns(
            profile_data,
            domain_description=self.domain,
            item_noun=self.item_noun,
            model=self._model_for("TRIAGE"),
        )
        result.save(str(cache))
        print(f"  Triage saved to {cache}")
        print(f"  >>> Review {cache} before proceeding to configure()")
        return result

    def configure(self, *, force: bool = False) -> Any:
        """Stage 3: Generate full domain config from triage results.

        Skips generation when ``config.json`` already exists (set
        *force=True* to regenerate).
        """
        from shared.config import DomainConfig
        from generation.configure import generate_domain_config

        config_path = self.output_dir / "config.json"

        if config_path.exists() and not force:
            print(f"  Loading cached config from {config_path}")
            return DomainConfig.load(str(config_path))

        print("  Stage 3: Generating domain config...")
        triage_result = self.triage()
        profile_data = self.profile()

        catalog_path = (
            str(self._catalog_path) if self._catalog_path is not None else ""
        )
        config = generate_domain_config(
            triage_result,
            profile_data,
            catalog_path=catalog_path,
            catalog=self.catalog,
            model=self._model_for("CONFIGURE"),
        )
        config.save(str(config_path))
        print(f"  Config saved to {config_path}")
        print(f"  >>> Review {config_path} before proceeding to calibrate()")
        return config

    def generate_coherence(self, *, force: bool = False) -> None:
        """Generate coherence rules (checkpointed to ``checkpoints/coherence.py``).

        Requires config.json to exist. Skips if the checkpoint already
        exists (set *force=True* to regenerate).
        """
        from generation.configure import generate_coherence_module_for_domain

        cache = self.checkpoint_dir / "coherence.py"
        hit = _resolve_cached(cache, self.output_dir / "coherence.py")

        if hit is not None and not force:
            print(f"  Loading cached coherence rules from {hit}")
            if hit != cache:
                shutil.copy2(hit, cache)
            return

        config = self._load_config()
        print("  Generating coherence rules...")
        generate_coherence_module_for_domain(
            config, self.checkpoint_dir, model=self._model_for("COHERENCE"),
        )
        print(f"  Coherence rules saved to {cache}")

    def calibrate(self) -> Any:
        """Stage 4: Empirical validation of config against catalog."""
        from generation.calibrate import calibration_sweep

        config = self._load_config()
        catalog = self._load_cleaned_catalog(config)

        print("  Stage 4: Calibrating...")
        report = calibration_sweep(catalog, config)
        print(f"  Calibration complete. Zero-pool rate: {report.zero_pool_rate:.1%}")
        return report

    def extract_descriptions(self, **kwargs: Any) -> tuple[dict, dict | None]:
        """Extract description attributes from catalog embedding text.

        Intermediate files (phase 0 guidelines, phase 1 raw) go to
        ``checkpoints/``; the final ``description_attributes.json`` is
        copied to ``extras/`` so the benchmark runner can find it.
        """
        from generation.desc_extraction import extract_description_attributes

        config = self._load_config()
        catalog = self._load_cleaned_catalog(config)

        desc_base = self._model_for("EXTRACT_DESCRIPTIONS")
        desc_model_kwargs: dict[str, str] = {
            "phase0_model": self.model_map.get("EXTRACT_DESCRIPTIONS_DISCOVERY", desc_base),
            "extraction_model": self.model_map.get("EXTRACT_DESCRIPTIONS_EXTRACTION", desc_base),
            "normalize_model": self.model_map.get("EXTRACT_DESCRIPTIONS_NORMALIZATION", desc_base),
        }
        merged = {**desc_model_kwargs, **kwargs}

        desc_attrs, desc_catalog = extract_description_attributes(
            catalog, config, self.checkpoint_dir, **merged,
        )

        checkpoint_file = self.checkpoint_dir / "description_attributes.json"
        final_file = self.extras_dir / "description_attributes.json"
        if checkpoint_file.exists():
            shutil.copy2(checkpoint_file, final_file)
            print(f"  Copied description_attributes.json → {final_file}")

        for emb_name in (
            "all_embedding_text_embeddings.npy",
            "embedding_product_ids.json",
        ):
            src = self.checkpoint_dir / emb_name
            if src.exists():
                shutil.copy2(src, self.extras_dir / emb_name)

        return desc_attrs, desc_catalog

    def generate_tasks(self, n_tasks: int = 250, *, max_workers: int = 8) -> list[dict]:
        """Stage 5: Generate benchmark tasks across difficulty tiers.

        *n_tasks* is distributed proportionally across difficulty tiers
        defined in the config (respecting their relative ``target_count``
        ratios). *max_workers* controls concurrent LLM profile generations
        (see :func:`generation.generate_tasks.generate_all_tasks`).
        """
        config = self._load_config()
        catalog = self._load_cleaned_catalog(config)

        id_col = config.id_column
        if id_col in catalog.columns:
            catalog = catalog.set_index(id_col, drop=False)

        from generation.generate_tasks import generate_all_tasks

        desc_attrs: dict = {}
        desc_catalog: dict | None = None
        desc_path = _resolve_cached(
            self.extras_dir / "description_attributes.json",
            _resolve_cached(
                self.checkpoint_dir / "description_attributes.json",
                self.output_dir / "description_attributes.json",
            ),
        )
        if desc_path is not None:
            from generation.desc_extraction import load_desc_attrs
            desc_attrs, desc_catalog = load_desc_attrs(desc_path)
            print(f"  Loaded description attributes ({len(desc_attrs)} products)")

        checkpoint_path = str(self.checkpoint_dir / "_checkpoint.json")
        old_ckpt = self.output_dir / "_checkpoint.json"
        if old_ckpt.exists() and not (self.checkpoint_dir / "_checkpoint.json").exists():
            shutil.copy2(old_ckpt, checkpoint_path)

        # Distribute n_tasks across difficulty tiers proportionally
        target_counts = _distribute_tasks(n_tasks, config)

        print("  Stage 5: Generating tasks...")
        tasks = generate_all_tasks(
            catalog,
            config,
            target_counts=target_counts,
            model=self._model_for("GENERATE_TASKS"),
            checkpoint_path=checkpoint_path,
            desc_attrs=desc_attrs,
            desc_catalog=desc_catalog,
            max_workers=max_workers,
        )

        tasks_path = self.output_dir / "tasks.json"
        with open(tasks_path, "w") as f:
            json.dump(tasks, f, indent=2, default=str)
        print(f"  {len(tasks)} tasks saved to {tasks_path}")
        return tasks

    # -- helpers ---------------------------------------------------------------

    def _model_for(self, step: str) -> str:
        """Return the model override for *step* if present in model_map, else self.model."""
        return self.model_map.get(step, self.model)

    def _load_config(self):
        """Load config.json and wire up coherence from checkpoints/."""
        from shared.config import DomainConfig

        config_path = self.output_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError("Run configure() first")

        print(f"  Loading config from {config_path}")
        config = DomainConfig.load(str(config_path))

        primary_coh = self.checkpoint_dir / "coherence.py"
        fallback_coh = self.output_dir / "coherence.py"
        coherence_path = _resolve_cached(primary_coh, fallback_coh)
        if coherence_path is not None:
            print(f"  Loading coherence rules from {coherence_path}")
            if coherence_path != primary_coh:
                print(
                    f"    (fallback: no file at {primary_coh}; "
                    f"using {coherence_path})",
                )
            config.load_coherence_from_file(str(coherence_path))
        elif config.coherence_module:
            print(
                f"  No coherence.py under {self.checkpoint_dir.name}/ or "
                f"{self.output_dir.name}/; using coherence_module "
                f"{config.coherence_module!r}",
            )
        else:
            print(
                "  No coherence.py or coherence_module; constraint coherence checks are disabled",
            )

        return config

    def _load_cleaned_catalog(self, config=None) -> pd.DataFrame:
        """Load the cleaned catalogue (falls back to raw)."""
        cleaned = self.output_dir / "catalogue.parquet"
        if not cleaned.exists():
            cleaned = self.output_dir / "cleaned_catalogue.parquet"
        if cleaned.exists():
            catalog = pd.read_parquet(cleaned)
        else:
            catalog = self.catalog

        if config is not None:
            id_col = config.id_column
            if id_col in catalog.columns:
                catalog = catalog.set_index(id_col, drop=False)
        return catalog
