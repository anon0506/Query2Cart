"""Domain configuration schema.

A single ``DomainConfig`` instance fully describes everything the generic pipeline
needs to know about a product domain: its catalog attributes, constraint algebra,
difficulty calibration, conversational triggers, and LLM prompt fragments.

Domain-specific coherence rules live in a separate Python module referenced by
``coherence_module`` (e.g. ``domains.laptops.coherence``).
"""

from __future__ import annotations

import importlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AttrType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    SET_VALUED = "set_valued"
    BOOLEAN = "boolean"
    ORDINAL = "ordinal"
    TEXT = "text"


class ConstraintOp(str, Enum):
    LTE = "lte"
    GTE = "gte"
    EQ = "eq"
    EQ_ANY = "eq_any"
    NEQ = "neq"
    IN_SET = "in_set"
    CONTAINS = "contains"
    CONTAINS_ALL = "contains_all"
    CONTAINS_ANY = "contains_any"
    NOT_CONTAINS = "not_contains"
    NOT_CONTAINS_ANY = "not_contains_any"
    NOT_CONTAINS_ALL = "not_contains_all"
    SUBSTRING = "substring"
    BOOLEAN = "boolean"
    RANGE = "range"


# ---------------------------------------------------------------------------
# Attribute & constraint specs
# ---------------------------------------------------------------------------

class AttributeSpec(BaseModel):
    """One column in the cleaned catalog."""
    name: str
    display_name: str
    attr_type: AttrType
    unit: str | None = None
    filterable: bool = False
    required: bool = False
    coverage_threshold: float = 0.0
    vocabulary: list[str] | None = None
    embedding_field: bool = False
    preference_eligible: bool = False
    preference_direction: str | None = None  # minimize | maximize | match
    popularity_proxy: bool = False


class ConstraintSpec(BaseModel):
    """One user-facing constraint the pipeline can sample and filter on."""
    name: str                           # key in task JSON, e.g. "price_max_usd"
    attribute: str                      # catalog column it operates on
    operator: ConstraintOp
    display_template: str               # "Maximum price: ${value}"
    value_type: str = "float"           # float | int | str | list[str] | bool
    sampling_values: list[Any] | None = None
    sampling_probability: float = 0.5
    always_include: bool = False
    desc_constraint: bool = False
    filter_key: str | None = None       # key used during filtering if different from name
    demotable: bool = False             # can be softened to a preference per-task
    soft_direction: str | None = None   # minimize | maximize | match (when demoted)

    def get_filter_key(self) -> str:
        return self.filter_key or self.name


class TriggerSpec(BaseModel):
    """A conversational topic that can unlock responsive/contextual revelations."""
    name: str                           # e.g. "agent_asks_budget"
    keywords: list[str]                 # detection keywords
    description: str = ""               # human-readable topic description
    unlocks_constraints: list[str] = Field(default_factory=list)
    unlocks_preferences: list[str] = Field(default_factory=list)


class ViolationTrigger(BaseModel):
    """A reactive trigger fired when a recommendation violates a constraint."""
    name: str                           # e.g. "shown_expensive_product"
    description: str
    constraint_name: str                # which constraint this reacts to


class PreferenceAttributeSpec(BaseModel):
    """Declares a valid soft-preference attribute for task generation."""
    attribute: str                      # catalog column name
    directions: list[str]               # subset of [minimize, maximize, match]
    catalog_field: str | None = None    # if different from attribute

    def get_catalog_field(self) -> str:
        return self.catalog_field or self.attribute


# ---------------------------------------------------------------------------
# Difficulty calibration
# ---------------------------------------------------------------------------

class DifficultyBracket(BaseModel):
    pool_range: tuple[int, int]
    min_constraints: int = 3
    desc_constraint_probability: float = 0.3
    target_count: int = 100


# ---------------------------------------------------------------------------
# Prompt fragments for LLM-generated profiles
# ---------------------------------------------------------------------------

class PromptFragments(BaseModel):
    """Domain-specific text injected into generic LLM prompt templates."""
    # Human-readable product domain; required for all LLM-assisted steps (triage, triggers, task profiles).
    domain_description: str = ""
    query_rules: str = ""

    @field_validator("query_rules", mode="before")
    @classmethod
    def _coerce_query_rules(cls, v: Any) -> str:
        if isinstance(v, list):
            return "\n".join(str(x).strip() for x in v if x is not None and str(x).strip())
        return v
    expertise_levels: dict[str, str] = Field(default_factory=dict)
    use_case_description: str = ""
    item_noun: str = "product"
    item_noun_plural: str = "products"
    system_persona: str = ""            # e.g. "a laptop shopping assistant"
    embedding_template: str = "{name}. {description}"


# ---------------------------------------------------------------------------
# Top-level domain config
# ---------------------------------------------------------------------------

class DomainConfig(BaseModel):
    """Complete specification of a product domain for the benchmark pipeline."""
    name: str
    item_noun: str
    item_noun_plural: str
    catalog_path: str
    id_column: str

    attributes: list[AttributeSpec]
    constraints: list[ConstraintSpec]
    triggers: list[TriggerSpec]
    violation_triggers: list[ViolationTrigger] = Field(default_factory=list)
    preference_attributes: list[PreferenceAttributeSpec] = Field(default_factory=list)

    difficulty: dict[str, DifficultyBracket]
    prompt_fragments: PromptFragments = Field(default_factory=PromptFragments)

    coherence_module: str | None = None

    # -- Derived lookups (populated by model_validator) --

    _constraint_by_name: dict[str, ConstraintSpec] = {}
    _attribute_by_name: dict[str, AttributeSpec] = {}
    _trigger_by_name: dict[str, TriggerSpec] = {}

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def _build_indexes(self) -> "DomainConfig":
        object.__setattr__(
            self, "_constraint_by_name",
            {c.name: c for c in self.constraints},
        )
        object.__setattr__(
            self, "_attribute_by_name",
            {a.name: a for a in self.attributes},
        )
        object.__setattr__(
            self, "_trigger_by_name",
            {t.name: t for t in self.triggers},
        )
        return self

    # -- Convenience accessors --

    def get_constraint(self, name: str) -> ConstraintSpec:
        return self._constraint_by_name[name]

    def get_attribute(self, name: str) -> AttributeSpec:
        return self._attribute_by_name[name]

    def get_trigger(self, name: str) -> TriggerSpec:
        return self._trigger_by_name[name]

    def get_filterable_constraints(self) -> list[ConstraintSpec]:
        return [c for c in self.constraints if not c.desc_constraint]

    def get_trigger_description_map(self) -> dict[str, str]:
        return {t.name: t.description or ", ".join(t.keywords[:5]) for t in self.triggers}

    def get_constraint_label(self, name: str, value: Any) -> str | None:
        spec = self._constraint_by_name.get(name)
        if spec is None:
            return None
        try:
            return spec.display_template.format(value=value)
        except (KeyError, IndexError):
            return f"{spec.display_template}: {value}"

    # -- Coherence --

    _coherence_fn: Any = None

    def load_coherence_from_file(self, path: str | None) -> None:
        """Load ``is_coherent`` from a file path instead of a dotted module."""
        if path is None:
            return
        from pathlib import Path as _P
        p = _P(path)
        if not p.is_file():
            return
        spec = importlib.util.spec_from_file_location(
            f"coherence_{p.stem}", str(p),
        )
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "is_coherent", None)
        if fn is not None:
            object.__setattr__(self, "_coherence_fn", fn)

    def is_coherent(self, constraints: dict[str, Any]) -> bool:
        if self._coherence_fn is not None:
            return self._coherence_fn(constraints)
        if self.coherence_module is None:
            return True
        try:
            mod = importlib.import_module(self.coherence_module)
            return mod.is_coherent(constraints)
        except (ImportError, ModuleNotFoundError):
            return True

    # -- Serialisation helpers --

    def save(self, path: str) -> None:
        from pathlib import Path
        import json
        Path(path).write_text(json.dumps(self.model_dump(), indent=2, default=str))

    @classmethod
    def load(cls, path: str) -> "DomainConfig":
        from pathlib import Path
        import json
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)
