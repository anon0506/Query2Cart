"""Load per-domain onboarding settings from ``onboarding_manifest.json``.

Each domain folder (e.g. ``domain/sephora``) should contain this JSON next to its
catalog. Required keys describe the catalog and LLM-facing domain context;
optional keys override default artifact filenames under the same folder.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

MANIFEST_NAME = "onboarding_manifest.json"

_REQUIRED = frozenset({"catalog_file", "domain_description", "item_noun", "source_name"})

_ARTIFACT_DEFAULTS: dict[str, str] = {
    "cleaned_catalogue_file": "cleaned_catalogue.parquet",
    "profile_cache": "profile.pkl",
    "triage_cache": "triage_result.json",
    "config_cache": "config.json",
    "coherence_module_file": "coherence.py",
    "pilot_tasks_file": "pilot_tasks.json",
}


@dataclass(frozen=True)
class OnboardingArtifacts:
    cleaned_catalog_parquet: Path
    profile: Path
    triage: Path
    config: Path
    coherence_py: Path
    pilot_tasks: Path


def load_onboarding_manifest(domain_dir: Path | str) -> dict[str, Any]:
    domain_dir = Path(domain_dir)
    path = domain_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {MANIFEST_NAME} in {domain_dir.resolve()}. "
            "Add catalog_file, domain_description, item_noun, source_name."
        )
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    missing = _REQUIRED - data.keys()
    if missing:
        raise ValueError(f"{path}: missing required keys {sorted(missing)}")
    return data


def resolved_catalog_path(
    domain_dir: Path | str, manifest: Mapping[str, Any] | None = None
) -> Path:
    domain_dir = Path(domain_dir)
    if manifest is None:
        manifest = load_onboarding_manifest(domain_dir)
    rel = manifest["catalog_file"]
    p = Path(rel)
    if not p.is_absolute():
        p = domain_dir / p
    return p.resolve()


def slim_catalog_parquet_path(domain_dir: Path | str, item_noun: str) -> Path:
    """Path to ``{sanitized_item_noun}_slim.parquet`` under ``domain_dir``.

    Non-alphanumeric runs in ``item_noun`` become single underscores so the name is
    safe on all platforms (e.g. ``beauty product`` → ``beauty_product_slim.parquet``).
    """
    domain_dir = Path(domain_dir).resolve()
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", str(item_noun).strip()).strip("_").lower()
    if not stem:
        stem = "catalog"
    return domain_dir / f"{stem}_slim.parquet"


def resolve_domain_artifacts(
    domain_dir: Path | str, manifest: Mapping[str, Any] | None = None
) -> OnboardingArtifacts:
    domain_dir = Path(domain_dir)
    if manifest is None:
        manifest = load_onboarding_manifest(domain_dir)

    def _p(key: str) -> Path:
        filename = manifest.get(key, _ARTIFACT_DEFAULTS[key])
        return domain_dir / str(filename)

    return OnboardingArtifacts(
        cleaned_catalog_parquet=_p("cleaned_catalogue_file"),
        profile=_p("profile_cache"),
        triage=_p("triage_cache"),
        config=_p("config_cache"),
        coherence_py=_p("coherence_module_file"),
        pilot_tasks=_p("pilot_tasks_file"),
    )


def publish_cleaned_catalogue_alias(parquet_path: Path | str) -> Path:
    """Ensure ``cleaned_catalogue`` exists beside the parquet file (symlink or copy).

    Some tooling expects an extensionless name; we mirror ``cleaned_catalogue.parquet``.
    Returns the path to the alias file.
    """
    parquet_path = Path(parquet_path)
    stem_alias = parquet_path.with_name("cleaned_catalogue")
    if not parquet_path.is_file():
        return stem_alias
    try:
        if stem_alias.exists() or stem_alias.is_symlink():
            stem_alias.unlink()
        stem_alias.symlink_to(parquet_path.name)
    except OSError:
        shutil.copy2(parquet_path, stem_alias)
    return stem_alias
