"""EvaluationResult data structure and structured outcome presentation."""

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from image_evaluator.specifications import MetricSpec


def _normalize_json_value(val: Any) -> Any:
    """Normalize floats and structures for RFC 8259 compliance."""
    if isinstance(val, float):
        if not math.isfinite(val):
            return None
        return val
    if isinstance(val, Mapping):
        return {str(k): _normalize_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_normalize_json_value(v) for v in val]
    return val


def _freeze_value(value: Any) -> Any:
    """Detach and recursively freeze supported result metadata containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _thaw_value(value: Any) -> Any:
    """Return detached mutable containers for backwards-compatible output."""
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw_value(item) for item in value}
    return value


@dataclass(frozen=True, slots=True)
class EvaluationResult(Mapping[str, Any]):
    """Structured evaluation outcome containing scores and metadata contracts.

    Implements Mapping protocol for dict-like score access (e.g.
    ``result['ssim']``), provides ``to_dict()`` for backwards compatibility,
    ``to_json()`` for RFC 8259 compliance, and exposes Registry specifications.
    """

    scores: Mapping[str, Any]
    specs: Mapping[str, MetricSpec]
    inputs: Mapping[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", _freeze_value(self.scores))
        object.__setattr__(self, "specs", _freeze_value(self.specs))
        object.__setattr__(self, "inputs", _freeze_value(self.inputs))

        for k, v in self.scores.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"Metric name must be a str, got {type(k).__name__}."
                )
        if self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be non-negative.")

    def __getitem__(self, key: str) -> Any:
        return self.scores[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.scores)

    def __len__(self) -> int:
        return len(self.scores)

    def __contains__(self, key: object) -> bool:
        return key in self.scores

    def get(self, key: str, default: Any = None) -> Any:
        return self.scores.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the scores mapping."""
        return _thaw_value(self.scores)

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize scores, specifications, and input metadata to JSON."""
        payload = {
            "scores": {
                k: _normalize_json_value(v) for k, v in self.scores.items()
            },
            "specs": {
                k: {
                    "id": spec.id,
                    "display_name": spec.display_name,
                    "score_direction": spec.score_direction,
                    "tasks": list(spec.tasks),
                    "objectives": list(spec.objectives),
                    "docs_path": spec.docs_path,
                }
                for k, spec in self.specs.items()
            },
            "inputs": _normalize_json_value(self.inputs),
            "duration_seconds": round(self.duration_seconds, 4),
        }
        return json.dumps(payload, indent=indent, allow_nan=False)

    def __repr__(self) -> str:
        entries = []
        for k, v in self.scores.items():
            if isinstance(v, float) and math.isfinite(v):
                entries.append(f"{k}={v:.4f}")
            else:
                entries.append(f"{k}={v}")
        summary = ", ".join(entries)
        return f"EvaluationResult({summary})"
