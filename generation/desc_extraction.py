"""Domain-agnostic description attribute extraction.

Extracts structured key-value attributes from the ``all_embedding_text``
column in a catalog using an LLM.  Produces
``description_attributes.json`` for use by task generation.

Three-phase architecture with per-phase checkpointing:

  Phase 0: Schema discovery — cluster products by embedding similarity,
           sample diverse products, iteratively build naming guidelines
           across batches.  Saved as ``_desc_phase0_guidelines.txt``.
  Phase 1: Guided extraction — extract attributes from every product using
           Phase 0 guidelines.  Saved as ``_desc_phase1_raw.json``.
  Phase 2: Key normalization — two-tier: primary keys via cumulative
           frequency cutoff, then tail key classification.
           Final output as ``description_attributes.json``.

Each phase checks for its checkpoint and skips if found (unless forced).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from shared.config import DomainConfig
from shared.llm import call_embedding, call_llm, call_llm_json, format_prompt_template, resolve_prompt_path

logger = logging.getLogger(__name__)

_GARBAGE_VALUES = frozenset({
    "n/a", "unknown", "none", "null", "na", "", "not available",
    "not specified", "unspecified", "other", "-", "--", "n.a.",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_usable(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    s = str(val).strip()
    return len(s) > 0 and s.lower() not in _GARBAGE_VALUES


def _clean_entry(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "product_id" and _is_usable(v)}


def _build_structured_fields(config: DomainConfig) -> set[str]:
    fields: set[str] = set()
    for a in config.attributes:
        if not a.embedding_field:
            fields.add(a.name)
    if config.id_column:
        fields.add(config.id_column)
    return fields


def _load_prompt(name: str, **kwargs: Any) -> str:
    path = resolve_prompt_path(name)
    with open(path) as f:
        template = f.read()
    return format_prompt_template(template, **kwargs)


def _build_products_block(
    batch: pd.DataFrame,
    id_col: str,
) -> tuple[str, list[str]]:
    """Build text block from the all_embedding_text column."""
    lines: list[str] = []
    included_pids: list[str] = []
    for _, row in batch.iterrows():
        pid = str(row.get(id_col, row.name))
        text = row.get("all_embedding_text")
        if not _is_usable(text):
            continue
        lines.append(f"[{pid}]\n{str(text).strip()}")
        included_pids.append(pid)
    return "\n\n".join(lines), included_pids


def _parse_phase1_llm_list(
    result: list[Any],
    included_pids: list[str],
) -> dict[str, dict]:
    """Map Phase‑1 LLM array entries onto catalog product ids.

    Uses each object's ``product_id`` when it matches an id from this batch
    (so row order in the JSON does not matter). Entries without a valid
    ``product_id`` are paired with the remaining expected ids in *prompt*
    order — the same contract as the per‑batch prompt listing.
    """
    included_set = set(included_pids)
    out: dict[str, dict] = {}
    unmapped: list[dict] = []

    for entry in result:
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        pid_raw = row.pop("product_id", None)
        pid: str | None = None
        if pid_raw is not None:
            ps = str(pid_raw).strip()
            if ps in included_set:
                pid = ps
        if pid is not None:
            cleaned = _clean_entry(row)
            if cleaned:
                out[pid] = cleaned
            continue
        unmapped.append(row)

    remaining = [p for p in included_pids if p not in out]
    for j, row in enumerate(unmapped):
        if j >= len(remaining):
            break
        cleaned = _clean_entry(row)
        if cleaned:
            out[remaining[j]] = cleaned
    return out


def _phase1_one_batch_llm(
    prompt: str,
    included_pids: list[str],
    model: str,
) -> dict[str, dict]:
    try:
        result = call_llm_json(
            prompt,
            model,
            temperature=0.1,
            max_tokens=len(included_pids) * 300,
        )
        if isinstance(result, list):
            return _parse_phase1_llm_list(result, included_pids)
    except Exception:
        logger.debug("Phase 1 batch failed", exc_info=True)
    return {}


# ---------------------------------------------------------------------------
# Auto-generate catalog embeddings
# ---------------------------------------------------------------------------

def _ensure_embeddings(
    catalog: pd.DataFrame,
    config: DomainConfig,
    output_dir: Path,
    *,
    batch_size: int = 512,
    model: str = "text-embedding-3-large",
) -> None:
    """Build and save catalog embeddings if the files don't already exist."""
    npy_path = output_dir / "all_embedding_text_embeddings.npy"
    ids_path = output_dir / "embedding_product_ids.json"

    if npy_path.exists() and ids_path.exists():
        return

    id_col = config.id_column or "product_id"

    product_ids: list[str] = []
    texts: list[str] = []
    for _, row in catalog.iterrows():
        pid = str(row.get(id_col, row.name))
        product_ids.append(pid)
        text = row.get("all_embedding_text")
        texts.append(str(text).strip() if _is_usable(text) else "")

    nonempty_indices = [i for i, t in enumerate(texts) if t]
    nonempty_texts = [texts[i] for i in nonempty_indices]

    if not nonempty_texts:
        print("  No products with embedding text — skipping embedding generation")
        return

    print(
        f"  Generating embeddings for {len(nonempty_texts)}/{len(texts)} "
        f"products (model={model}, batch_size={batch_size})...",
    )

    all_embeddings: list[list[float]] = []
    for start in tqdm(
        range(0, len(nonempty_texts), batch_size),
        desc="Embedding batches",
        unit="batch",
    ):
        batch = nonempty_texts[start : start + batch_size]
        all_embeddings.extend(call_embedding(batch, model=model))

    dim = len(all_embeddings[0])
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for j, idx in enumerate(nonempty_indices):
        vec = np.asarray(all_embeddings[j], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        matrix[idx] = vec

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, matrix)
    with open(ids_path, "w") as f:
        json.dump(product_ids, f)

    n_nonzero = int((np.linalg.norm(matrix, axis=1) > 0).sum())
    print(
        f"  Saved embeddings {matrix.shape} to {output_dir.name}/ "
        f"({n_nonzero} non-zero rows)",
    )


# ---------------------------------------------------------------------------
# Embedding-based diverse sampling
# ---------------------------------------------------------------------------

def _cluster_sample(
    catalog: pd.DataFrame,
    config: DomainConfig,
    embeddings_dir: Path,
    sample_size: int,
    n_batches: int,
) -> list[pd.DataFrame]:
    """Pick maximally diverse products via fine-grained clustering.

    Creates ``sample_size`` k-means clusters over the embedding space and
    selects the product closest to each centroid.  The resulting samples
    are then split into ``n_batches`` equal batches for LLM calls.
    """
    npy_path = embeddings_dir / "all_embedding_text_embeddings.npy"
    ids_path = embeddings_dir / "embedding_product_ids.json"
    id_col = config.id_column or "product_id"

    eligible = catalog[catalog["all_embedding_text"].apply(_is_usable)].copy()

    if npy_path.exists() and ids_path.exists():
        embeddings = np.load(npy_path)
        with open(ids_path) as f:
            emb_ids = json.load(f)

        id_to_idx = {str(pid): i for i, pid in enumerate(emb_ids)}

        eligible_indices: list = []
        emb_row_indices: list[int] = []
        for df_idx, row in tqdm(
            eligible.iterrows(),
            total=len(eligible),
            desc="Aligning IDs to embeddings",
            unit="row",
        ):
            pid = str(row[id_col])
            if pid in id_to_idx:
                emb_idx = id_to_idx[pid]
                if np.linalg.norm(embeddings[emb_idx]) > 0:
                    eligible_indices.append(df_idx)
                    emb_row_indices.append(emb_idx)

        n_clusters = min(sample_size, len(eligible_indices))
        if n_clusters >= n_batches:
            from sklearn.cluster import KMeans
            from scipy.spatial.distance import cdist

            valid_embeddings = embeddings[emb_row_indices]
            n_init = 10
            best_km: KMeans | None = None
            for seed_offset in tqdm(
                range(n_init),
                desc="KMeans (n_init retries)",
                unit="run",
            ):
                km_run = KMeans(
                    n_clusters=n_clusters,
                    n_init=1,
                    random_state=42 + seed_offset,
                )
                km_run.fit(valid_embeddings)
                if best_km is None or km_run.inertia_ < best_km.inertia_:
                    best_km = km_run
            assert best_km is not None
            km = best_km

            distances = cdist(km.cluster_centers_, valid_embeddings)
            representative_indices = distances.argmin(axis=1)
            unique_reps = list(dict.fromkeys(representative_indices))

            sampled_df_indices = [eligible_indices[i] for i in unique_reps]
            sampled = eligible.loc[sampled_df_indices]

            per_batch = len(sampled) // n_batches
            batches: list[pd.DataFrame] = []
            for b in range(n_batches):
                start = b * per_batch
                end = start + per_batch if b < n_batches - 1 else len(sampled)
                batches.append(sampled.iloc[start:end])

            print(
                f"  Clustered {len(eligible_indices)} products into "
                f"{n_clusters} micro-clusters, picked 1 representative each "
                f"-> {len(sampled)} samples in {n_batches} batches: "
                f"{[len(b) for b in batches]}",
            )
            return batches

    print("  Embeddings not available — falling back to random sampling")
    total = min(sample_size, len(eligible))
    sampled = eligible.sample(n=total, random_state=42)
    per_batch = total // n_batches
    return [
        sampled.iloc[i:i + per_batch]
        for i in range(0, total, per_batch)
    ]


# ---------------------------------------------------------------------------
# Phase 0: Iterative schema discovery
# ---------------------------------------------------------------------------

def run_phase0(
    catalog: pd.DataFrame,
    config: DomainConfig,
    output_path: Path,
    *,
    embeddings_dir: Path | None = None,
    model: str = "gpt-4.1",
    sample_size: int = 200,
    n_batches: int = 4,
    force: bool = False,
) -> str:
    """Cluster products, sample diverse batches, iteratively build guidelines.

    Each batch receives guidelines accumulated from prior batches so that
    later batches focus on discovering NEW attributes.

    Returns the final guideline text.  Saves to ``output_path``.
    """
    if output_path.exists() and not force:
        guidelines = output_path.read_text().strip()
        print(
            f"Phase 0: loaded cached guidelines from {output_path.name} "
            f"({guidelines.count(chr(10)) + 1} lines)",
        )
        return guidelines

    structured_fields = _build_structured_fields(config)
    id_col = config.id_column or catalog.index.name or "product_id"
    domain = config.prompt_fragments.domain_description or "product"
    item = config.prompt_fragments.item_noun or "product"
    fields_str = ", ".join(sorted(structured_fields))

    if embeddings_dir is None:
        embeddings_dir = output_path.parent

    batches = _cluster_sample(
        catalog, config, embeddings_dir,
        sample_size=sample_size,
        n_batches=n_batches,
    )

    total_sampled = sum(len(b) for b in batches)
    print(
        f"Phase 0: {total_sampled} diverse samples across "
        f"{len(batches)} batches\n",
    )

    key_counts: Counter[str] = Counter()
    sample_values: dict[str, list[str]] = {}
    guidelines = ""

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1

        products_block, included_pids = _build_products_block(batch, id_col)
        if not included_pids:
            continue

        prior_guidelines_block = ""
        if guidelines:
            prior_guidelines_block = (
                "\nATTRIBUTES ALREADY DISCOVERED (follow these names where "
                "applicable, but also look for any NEW attributes not yet "
                "covered by the guidelines below):\n"
                f"{guidelines}\n"
            )

        prompt = _load_prompt(
            "desc_phase0_extraction.txt",
            item=item,
            structured_fields=fields_str,
            prior_guidelines_block=prior_guidelines_block,
            domain=domain,
            n_products=str(len(included_pids)),
            products_block=products_block,
        )

        print(
            f"  Batch {batch_num}/{len(batches)}: "
            f"{len(included_pids)} products",
            end="",
        )
        try:
            result = call_llm_json(
                prompt, model, temperature=0.1,
                max_tokens=len(included_pids) * 300,
            )
            if isinstance(result, list):
                batch_entries = [e for e in result if isinstance(e, dict)]
                for entry in batch_entries:
                    cleaned = _clean_entry(entry)
                    for k, v in cleaned.items():
                        key_counts[k] += 1
                        sample_values.setdefault(k, [])
                        if len(sample_values[k]) < 12:
                            sample_values[k].append(str(v))
                print(f" -> {len(batch_entries)} extracted")
            else:
                print(" -> unexpected response format")
        except Exception:
            logger.debug("Phase 0 batch %d failed", batch_num, exc_info=True)
            print(" -> failed")
            continue

        # Synthesize / refine guidelines from all keys accumulated so far
        key_summary_lines: list[str] = []
        for k, c in key_counts.most_common():
            vals = ", ".join(sample_values.get(k, [])[:10])
            key_summary_lines.append(f"  {k} (n={c}): {vals}")
        key_summary = "\n".join(key_summary_lines)

        n_sampled = sum(len(b) for b in batches[: batch_idx + 1])
        guideline_prompt = _load_prompt(
            "desc_phase0_guidelines.txt",
            domain=domain,
            n_sampled=str(n_sampled),
            n_batches_done=str(batch_num),
            key_summary=key_summary,
            structured_fields=fields_str,
        )

        try:
            guidelines = call_llm(
                guideline_prompt, model, temperature=0.2, max_tokens=3000,
            )
            print(
                f"  -> Guidelines updated "
                f"({guidelines.count(chr(10)) + 1} lines)",
            )
        except Exception:
            logger.warning(
                "Phase 0 guideline synthesis failed for batch %d",
                batch_num, exc_info=True,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(guidelines)
    n_lines = guidelines.count("\n") + 1 if guidelines else 0
    print(f"\nPhase 0: saved {n_lines}-line guideline to {output_path.name}")
    return guidelines


# ---------------------------------------------------------------------------
# Phase 1: Guided extraction
# ---------------------------------------------------------------------------

def run_phase1(
    catalog: pd.DataFrame,
    config: DomainConfig,
    output_path: Path,
    *,
    model: str = "gpt-4.1-mini",
    batch_size: int = 20,
    guidelines: str = "",
    phase1_parallel_workers: int = 1,
    force: bool = False,
) -> dict[str, dict]:
    """Extract attributes from every product using Phase 0 guidelines.

    Returns ``{product_id: {attr: value}}``.  Saves to ``output_path``.
    """
    if output_path.exists() and not force:
        with open(output_path) as f:
            raw_attrs = json.load(f)
        print(
            f"Phase 1: loaded {len(raw_attrs)} products from cache "
            f"({output_path.name})",
        )
        return raw_attrs

    structured_fields = _build_structured_fields(config)
    id_col = config.id_column or catalog.index.name or "product_id"
    domain = config.prompt_fragments.domain_description or "product"
    item = config.prompt_fragments.item_noun or "product"
    fields_str = ", ".join(sorted(structured_fields))

    eligible = catalog[catalog["all_embedding_text"].apply(_is_usable)]
    print(
        f"Phase 1: {len(eligible)}/{len(catalog)} products have embedding text",
    )

    guidelines_block = ""
    if guidelines:
        guidelines_block = (
            "ATTRIBUTE NAMING GUIDELINES (follow these strictly):\n"
            f"{guidelines}"
        )

    raw_attrs: dict[str, dict] = {}
    n_batches = (len(eligible) + batch_size - 1) // batch_size
    workers = max(1, int(phase1_parallel_workers))

    batch_specs: list[tuple[str | None, list[str]]] = []
    for i in range(0, len(eligible), batch_size):
        batch = eligible.iloc[i:i + batch_size]
        products_block, included_pids = _build_products_block(batch, id_col)
        if not included_pids:
            batch_specs.append((None, []))
            continue
        prompt = _load_prompt(
            "desc_phase1_extraction.txt",
            item=item,
            structured_fields=fields_str,
            guidelines_block=guidelines_block,
            domain=domain,
            n_products=str(len(included_pids)),
            products_block=products_block,
        )
        batch_specs.append((prompt, included_pids))

    if workers > 1:
        print(
            f"Phase 1: parallel workers={workers} "
            f"({sum(1 for p, _ in batch_specs if p is not None)} LLM batches)",
        )

    pbar = tqdm(total=n_batches, desc="Phase 1 — extracting", unit="batch")
    if workers <= 1:
        for prompt, included_pids in batch_specs:
            if prompt is None:
                pbar.update(1)
                continue
            raw_attrs.update(_phase1_one_batch_llm(prompt, included_pids, model))
            pbar.update(1)
            pbar.set_postfix(products=len(raw_attrs))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_phase1_one_batch_llm, prompt, included_pids, model)
                for prompt, included_pids in batch_specs
                if prompt is not None
            ]
            for fut in as_completed(futures):
                raw_attrs.update(fut.result())
                pbar.update(1)
                pbar.set_postfix(products=len(raw_attrs))
    pbar.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(raw_attrs, f, indent=2)
    print(f"Phase 1: saved {len(raw_attrs)} products to {output_path.name}")
    return raw_attrs


# ---------------------------------------------------------------------------
# Phase 2: Two-tier normalization + coverage + final output
# ---------------------------------------------------------------------------

def run_phase2(
    raw_attrs: dict[str, dict],
    catalog: pd.DataFrame,
    config: DomainConfig,
    output_path: Path,
    *,
    model: str = "gpt-4.1-mini",
    cumulative_freq_cutoff: float = 0.95,
    min_coverage_pct: float = 3.0,
    max_single_value_dominance: float = 0.98,
    force: bool = False,
) -> tuple[dict[str, dict], dict | None]:
    """Two-tier normalization, coverage filtering, final output.

    Tier 1: keys covering the top ``cumulative_freq_cutoff`` of all
    occurrences go through full LLM normalization.
    Tier 2: remaining tail keys are classified into existing groups.

    Returns ``(desc_attrs, desc_catalog)`` ready for task generation.
    """
    if output_path.exists() and not force:
        print(f"Phase 2: loaded from cache ({output_path.name})")
        return load_desc_attrs(output_path)

    structured_fields = _build_structured_fields(config)
    domain = config.prompt_fragments.domain_description or "product"
    fields_str = ", ".join(sorted(structured_fields))

    # -- Count all raw keys --
    key_counts: Counter[str] = Counter()
    for attrs in raw_attrs.values():
        key_counts.update(attrs.keys())

    total_keys = len(key_counts)
    total_occurrences = sum(key_counts.values())
    print(
        f"Phase 2: {total_keys} unique raw keys, "
        f"{total_occurrences} total occurrences",
    )

    # -- Tier 1: cumulative frequency cutoff --
    sorted_keys = sorted(key_counts.items(), key=lambda x: -x[1])
    cumulative = 0
    tier1_keys: dict[str, int] = {}

    for k, c in sorted_keys:
        tier1_keys[k] = c
        cumulative += c
        if cumulative >= total_occurrences * cumulative_freq_cutoff:
            break

    tier2_keys: dict[str, int] = {
        k: c for k, c in key_counts.items() if k not in tier1_keys
    }
    tier1_pct = cumulative / total_occurrences * 100 if total_occurrences else 0
    print(
        f"  Tier 1: {len(tier1_keys)} keys "
        f"(covering {cumulative}/{total_occurrences} = {tier1_pct:.1f}%)",
    )
    print(f"  Tier 2: {len(tier2_keys)} tail keys")

    taxonomy: dict[str, dict] = {}
    key_to_canonical: dict[str, str] = {}

    # -- Tier 1: full LLM normalization --
    if tier1_keys:
        payload_lines: list[str] = []
        for k, c in sorted(tier1_keys.items(), key=lambda x: -x[1]):
            payload_lines.append(f"  {k}: {c}")
        payload = "\n".join(payload_lines)

        prompt = _load_prompt(
            "desc_phase2_normalization.txt",
            domain=domain,
            n_keys=str(len(tier1_keys)),
            structured_fields=fields_str,
            payload=payload,
        )

        data = call_llm_json(prompt, model, temperature=0.1, max_tokens=8000)
        groups = data.get("groups", [])

        for group in groups:
            if not isinstance(group, dict):
                continue
            canonical = group.get("canonical", "")
            if not canonical:
                continue
            for rk in group.get("raw_keys", []):
                key_to_canonical[rk] = canonical
            taxonomy[canonical] = {
                "type": group.get("type", "categorical"),
                "description": group.get("description", ""),
                "raw_keys": list(group.get("raw_keys", [])),
            }

        print(
            f"  Tier 1 -> {len(taxonomy)} canonical attributes, "
            f"{len(key_to_canonical)} raw keys mapped",
        )

    # -- Tier 2: classify tail keys into existing groups --
    if tier2_keys and taxonomy:
        canonical_groups_desc = "\n".join(
            f"  {name} ({info['type']}): {info['description']} "
            f"[raw: {', '.join(info['raw_keys'])}]"
            for name, info in taxonomy.items()
        )
        tail_keys_str = "\n".join(
            f"  {k}: {c}"
            for k, c in sorted(tier2_keys.items(), key=lambda x: -x[1])
        )

        prompt = _load_prompt(
            "desc_phase2_classify_tail.txt",
            canonical_groups=canonical_groups_desc,
            n_tail_keys=str(len(tier2_keys)),
            tail_keys=tail_keys_str,
        )

        try:
            data = call_llm_json(
                prompt, model, temperature=0.1, max_tokens=4000,
            )
            assignments = data.get("assignments", [])
            n_assigned = 0
            for a in assignments:
                if not isinstance(a, dict):
                    continue
                rk = a.get("raw_key", "")
                cn = a.get("canonical", "")
                if rk and cn and cn in taxonomy:
                    key_to_canonical[rk] = cn
                    taxonomy[cn]["raw_keys"].append(rk)
                    n_assigned += 1
            print(
                f"  Tier 2 -> {n_assigned} tail keys assigned "
                f"to existing groups",
            )
        except Exception:
            logger.debug("Tier 2 classification failed", exc_info=True)
            print("  Tier 2 -> classification failed, skipping tail keys")

    # -- Apply mapping --
    if key_to_canonical:
        normalized: dict[str, dict] = {}
        for pid, attrs in raw_attrs.items():
            mapped: dict[str, Any] = {}
            for raw_key, val in attrs.items():
                if not _is_usable(val):
                    continue
                canonical = key_to_canonical.get(raw_key)
                if canonical and canonical not in mapped:
                    mapped[canonical] = val
            if mapped:
                normalized[pid] = mapped
    else:
        normalized = {
            pid: {k: v for k, v in attrs.items() if _is_usable(v)}
            for pid, attrs in raw_attrs.items()
            if any(_is_usable(v) for v in attrs.values())
        }

    # -- Coverage stats --
    total = len(normalized)
    all_attrs: set[str] = set()
    for attrs in normalized.values():
        all_attrs.update(attrs.keys())

    min_products = max(20, int(len(catalog) * 0.005))
    all_stats: dict[str, dict] = {}

    for attr in all_attrs:
        values = [
            prod_attrs[attr]
            for prod_attrs in normalized.values()
            if attr in prod_attrs and _is_usable(prod_attrs[attr])
        ]
        count = len(values)
        pct = count / total * 100 if total else 0
        vc = Counter(str(v).lower().strip() for v in values)
        top_values = vc.most_common(15)

        is_boolean = (
            bool(values)
            and all(
                str(v).lower().strip() in ("true", "false") for v in values
            )
        )
        if is_boolean:
            attr_type = "boolean"
        else:
            try:
                _ = [float(v) for v in values]
                attr_type = "numeric"
            except (ValueError, TypeError):
                attr_type = "categorical"

        top_share = (
            top_values[0][1] / count if count > 0 and top_values else 0
        )
        dominance_limit = (
            1.0 if is_boolean else max_single_value_dominance
        )
        passes = (
            pct >= min_coverage_pct
            and count >= min_products
            and top_share < dominance_limit
        )

        all_stats[attr] = {
            "count": count,
            "pct": round(pct, 1),
            "type": attr_type,
            "top_values": [
                {"value": v, "count": c} for v, c in top_values
            ],
            "passes_filter": passes,
        }

    usable_stats = {k: v for k, v in all_stats.items() if v["passes_filter"]}
    usable_attrs = sorted(usable_stats.keys())

    # -- Build final output --
    filtered_products: dict[str, dict] = {}
    for pid, attrs in normalized.items():
        filtered = {
            k: v for k, v in attrs.items()
            if k in usable_stats and _is_usable(v)
        }
        if filtered:
            filtered_products[pid] = filtered

    final = {
        "metadata": {
            "total_products": len(normalized),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "method": "iterative_discovery_with_two_tier_normalization",
            "attributes": usable_attrs,
            "total_discovered_attributes": len(all_stats),
            "usable_attributes": len(usable_attrs),
            "filter_criteria": {
                "min_coverage_pct": min_coverage_pct,
                "min_product_count": min_products,
                "max_single_value_dominance": max_single_value_dominance,
                "cumulative_freq_cutoff": cumulative_freq_cutoff,
            },
        },
        "taxonomy": {
            name: {
                "type": info.get("type", "unknown"),
                "description": info.get("description", ""),
                "raw_keys": info.get("raw_keys", []),
            }
            for name, info in taxonomy.items()
            if name in usable_stats
        },
        "coverage_stats": usable_stats,
        "all_discovered_stats": all_stats,
        "products": filtered_products,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)

    # -- Summary --
    n_usable = len(usable_stats)
    print(f"\n{'=' * 70}")
    print("  DESCRIPTION ATTRIBUTE EXTRACTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Products processed: {len(normalized)}")
    print(f"  Total discovered:   {len(all_stats)}")
    print(f"  Usable (pass filter): {n_usable}")
    if usable_stats:
        print("\n  Usable attributes:")
        for attr in sorted(
            usable_stats, key=lambda a: -usable_stats[a]["count"],
        ):
            s = usable_stats[attr]
            print(
                f"    {attr:35s}: {s['count']:5d} "
                f"({s['pct']:5.1f}%) [{s['type']}]",
            )
            top = s.get("top_values", [])[:5]
            if top and s["type"] != "boolean":
                vals = ", ".join(
                    f"{v['value']}({v['count']})" for v in top
                )
                print(f"      top: {vals}")
    print(f"{'=' * 70}\n")

    return load_desc_attrs(output_path)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_desc_attrs(path: str | Path) -> tuple[dict[str, dict], dict | None]:
    """Load description_attributes.json -> (product_attrs, desc_catalog)."""
    path = Path(path)
    if not path.exists():
        logger.warning(
            "Description attributes not found at %s — skipping", path,
        )
        return {}, None

    with open(path) as f:
        data = json.load(f)

    attrs = data.get("products", {})
    metadata = data.get("metadata", {})
    usable_attrs = metadata.get("attributes", [])
    coverage_stats = data.get("coverage_stats", {})
    taxonomy = data.get("taxonomy", {})

    desc_catalog = {
        "attributes": usable_attrs,
        "coverage_stats": coverage_stats,
        "taxonomy": taxonomy,
    }

    logger.info(
        "Loaded description attributes for %d products (%d usable attrs: %s)",
        len(attrs), len(usable_attrs),
        ", ".join(usable_attrs[:10]),
    )
    return attrs, desc_catalog


# ---------------------------------------------------------------------------
# Build all_embedding_text from config
# ---------------------------------------------------------------------------

def _build_embedding_text_column(
    catalog: pd.DataFrame,
    config: DomainConfig,
) -> pd.DataFrame:
    """Construct ``all_embedding_text`` from embedding-field attributes."""
    embedding_attrs = [a for a in config.attributes if a.embedding_field]
    if not embedding_attrs:
        return catalog

    present = [a for a in embedding_attrs if a.name in catalog.columns]
    if not present:
        logger.warning(
            "Embedding attributes %s not in catalog columns — skipping",
            [a.name for a in embedding_attrs],
        )
        return catalog

    print(f"  Building 'all_embedding_text' from {len(present)} column(s): "
          f"{[a.name for a in present]}")

    def _row_text(row: pd.Series) -> str:
        lines: list[str] = []
        for a in present:
            val = row.get(a.name)
            if val is None:
                continue
            if isinstance(val, float) and pd.isna(val):
                continue
            s = str(val).strip()
            if not s:
                continue
            lines.append(f"{a.name}: {s}")
        return "\n".join(lines)

    catalog = catalog.copy()
    catalog["all_embedding_text"] = catalog.apply(_row_text, axis=1)
    n_ok = (catalog["all_embedding_text"].str.len() > 0).sum()
    print(f"  {n_ok}/{len(catalog)} products have embedding text")
    return catalog


# ---------------------------------------------------------------------------
# Convenience: run all phases
# ---------------------------------------------------------------------------

def extract_description_attributes(
    catalog: pd.DataFrame,
    config: DomainConfig,
    output_dir: str | Path,
    *,
    phase0_model: str = "gpt-4.1",
    extraction_model: str = "gpt-4.1-mini",
    normalize_model: str = "gpt-4.1-mini",
    batch_size: int = 20,
    phase1_parallel_workers: int = 1,
    phase0_sample_size: int = 200,
    phase0_n_batches: int = 4,
    cumulative_freq_cutoff: float = 0.95,
    min_coverage_pct: float = 3.0,
    max_single_value_dominance: float = 0.98,
    force_phase0: bool = False,
    force_phase1: bool = False,
    force_phase2: bool = False,
) -> tuple[dict[str, dict], dict | None]:
    """Run the full extraction pipeline (phases 0-1-2) with per-phase caching.

    All tunable parameters are exposed here so they can be set from the
    notebook cell that calls this function.

    Args:
        catalog: Must contain an ``all_embedding_text`` column.
        output_dir: Directory for all checkpoint files.
        phase0_model: Model for schema discovery (stronger = better).
        extraction_model: Model for Phase 1 bulk extraction.
        normalize_model: Model for Phase 2 normalization.
        batch_size: Products per LLM call in Phase 1.
        phase1_parallel_workers: Concurrent Phase‑1 LLM batch calls (``1`` =
            sequential).  Raise with care — too high may trigger rate limits.
        phase0_sample_size: Total products to sample in Phase 0.
        phase0_n_batches: Number of embedding clusters / iterative batches
            in Phase 0.  Each batch sees a different product region.
        cumulative_freq_cutoff: Fraction of total key occurrences that
            Tier 1 normalization covers (rest goes to Tier 2).
        min_coverage_pct: Minimum % of products an attribute must cover.
        max_single_value_dominance: Drop non-boolean attributes where one
            value exceeds this share.  Booleans are exempt.
        force_phase0/1/2: Re-run that phase even if its checkpoint exists.

    Returns:
        ``(desc_attrs, desc_catalog)`` ready for ``generate_all_tasks``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase0_path = output_dir / "_desc_phase0_guidelines.txt"
    phase1_path = output_dir / "_desc_phase1_raw.json"
    final_path = output_dir / "description_attributes.json"

    # The pipeline copies the final file to extras/; if it exists there but
    # not in checkpoints, use it as the phase 2 cache.
    extras_path = output_dir.parent / "extras" / "description_attributes.json"
    if not final_path.exists() and extras_path.is_file():
        import shutil
        shutil.copy2(extras_path, final_path)
        logger.info("Restored phase 2 cache from %s", extras_path)

    if "all_embedding_text" not in catalog.columns:
        catalog = _build_embedding_text_column(catalog, config)
        if "all_embedding_text" not in catalog.columns:
            print("ERROR: no embedding_field attributes in config — "
                  "cannot build all_embedding_text.")
            return {}, None

    n_with_text = catalog["all_embedding_text"].apply(_is_usable).sum()
    print(
        f"Catalog: {n_with_text}/{len(catalog)} products have "
        f"embedding text\n",
    )

    if n_with_text == 0:
        print(
            "No products with embedding text — skipping description "
            "extraction",
        )
        return {}, None

    _ensure_embeddings(catalog, config, output_dir)

    # Phase 0
    guidelines = run_phase0(
        catalog, config, phase0_path,
        embeddings_dir=output_dir,
        model=phase0_model,
        sample_size=phase0_sample_size,
        n_batches=phase0_n_batches,
        force=force_phase0,
    )
    print()

    # Phase 1
    raw_attrs = run_phase1(
        catalog, config, phase1_path,
        model=extraction_model, batch_size=batch_size,
        guidelines=guidelines, force=force_phase1,
        phase1_parallel_workers=phase1_parallel_workers,
    )
    if not raw_attrs:
        print("Phase 1 produced no results — aborting")
        return {}, None
    print()

    # Phase 2
    return run_phase2(
        raw_attrs, catalog, config, final_path,
        model=normalize_model,
        cumulative_freq_cutoff=cumulative_freq_cutoff,
        min_coverage_pct=min_coverage_pct,
        max_single_value_dominance=max_single_value_dominance,
        force=force_phase2,
    )
