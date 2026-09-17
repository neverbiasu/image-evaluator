"""Immutable domain contracts for describing evaluation metrics.

These contracts describe metrics only. Benchmarks, datasets, and judges are
separate domain concepts and intentionally have no runtime abstraction here.
"""

from dataclasses import dataclass
from typing import Literal

ScoreDirection = Literal["higher_is_better", "lower_is_better"]


def _normalize_terms(field_name: str, values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be an iterable of strings.")
    try:
        normalized = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            f"{field_name} must be an iterable of strings."
        ) from exc
    return normalized


def _validate_terms(
    field_name: str, values: tuple[str, ...], *, allow_empty: bool = False
) -> None:
    if not values and not allow_empty:
        raise ValueError(f"{field_name} must contain at least one value.")
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in values
    ):
        raise ValueError(
            f"{field_name} values must be non-empty strings without "
            "surrounding whitespace."
        )
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique.")


@dataclass(frozen=True, slots=True)
class InputContract:
    """Named input roles required or optionally accepted by a metric.

    Roles are stable descriptive strings such as ``image``,
    ``reference_image``, or ``prompt``. They describe inputs; they do not
    prescribe a CLI shape or load any data.
    """

    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "required", _normalize_terms("required", self.required)
        )
        object.__setattr__(
            self, "optional", _normalize_terms("optional", self.optional)
        )
        _validate_terms("required", self.required)
        _validate_terms("optional", self.optional, allow_empty=True)
        overlap = set(self.required) & set(self.optional)
        if overlap:
            raise ValueError(
                "Input roles cannot be both required and optional: "
                f"{sorted(overlap)}"
            )


@dataclass(frozen=True, slots=True)
class ImplementationRef:
    """Traceable implementation identity without importing its backend."""

    backend: str
    protocol: str
    model: str | None = None
    backend_version: str | None = None
    model_revision: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("backend", "protocol"):
            value = getattr(self, field_name)
            if not value or value != value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string without "
                    "surrounding whitespace."
                )
        for field_name in ("model", "backend_version", "model_revision"):
            value = getattr(self, field_name)
            if value is not None and (
                not value or value != value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be None or a non-empty string "
                    "without surrounding whitespace."
                )


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Discoverable contract for one evaluation metric.

    ``tasks`` and ``objectives`` are intentionally open string vocabularies.
    They support navigation and filtering, not runtime dispatch or automatic
    metric selection.
    """

    id: str
    display_name: str
    tasks: tuple[str, ...]
    objectives: tuple[str, ...]
    inputs: InputContract
    score_direction: ScoreDirection
    implementation: ImplementationRef
    aggregation: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    docs_path: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("id", "display_name"):
            value = getattr(self, field_name)
            if not value or value != value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string without "
                    "surrounding whitespace."
                )
        if self.id != self.id.lower():
            raise ValueError("Metric id must be lowercase.")
        for field_name in (
            "tasks",
            "objectives",
            "aggregation",
            "dependencies",
            "citations",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_terms(field_name, getattr(self, field_name)),
            )
        _validate_terms("tasks", self.tasks)
        _validate_terms("objectives", self.objectives)
        for field_name in ("aggregation", "dependencies", "citations"):
            _validate_terms(
                field_name, getattr(self, field_name), allow_empty=True
            )
        if self.score_direction not in {
            "higher_is_better",
            "lower_is_better",
        }:
            raise ValueError(
                "score_direction must be 'higher_is_better' or "
                "'lower_is_better'."
            )
        if self.docs_path is not None and (
            not self.docs_path or self.docs_path != self.docs_path.strip()
        ):
            raise ValueError(
                "docs_path must be None or a non-empty string without "
                "surrounding whitespace."
            )
