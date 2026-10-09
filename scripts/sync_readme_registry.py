#!/usr/bin/env python3
"""Sync MetricRegistry specifications into README.md.

Supports:
    python scripts/sync_readme_registry.py --check
    python scripts/sync_readme_registry.py --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path so we can import without installation
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from image_evaluator.registry import list_metrics  # noqa: E402
from image_evaluator.specifications import MetricSpec  # noqa: E402

TABLE_START_MARKER = "<!-- BEGIN METRICS_TABLE -->"
TABLE_END_MARKER = "<!-- END METRICS_TABLE -->"
TUTORIAL_START_MARKER = "<!-- BEGIN TUTORIAL_METRICS -->"
TUTORIAL_END_MARKER = "<!-- END TUTORIAL_METRICS -->"


def generate_metrics_table(metrics: tuple[MetricSpec, ...]) -> str:
    """Generate Markdown table for registered metrics."""
    header = (
        "| Metric ID | Display Name | Tasks | Objectives | "
        "Required Inputs | Direction | Technical Details |"
    )
    separator = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    lines = [header, separator]
    for m in metrics:
        tasks = ", ".join(f"`{t}`" for t in m.tasks)
        objs = ", ".join(f"`{o}`" for o in m.objectives)
        reqs = ", ".join(f"`{r}`" for r in m.inputs.required)
        direction = (
            "Higher is better"
            if m.score_direction == "higher_is_better"
            else "Lower is better"
        )
        doc = f"[{m.docs_path}]({m.docs_path})" if m.docs_path else "N/A"
        row = (
            f"| `{m.id}` | {m.display_name} | {tasks} | {objs} | "
            f"{reqs} | {direction} | {doc} |"
        )
        lines.append(row)
    return "\n".join(lines)


def generate_tutorial_metrics(metrics: tuple[MetricSpec, ...]) -> str:
    """Generate tutorial lines for --metrics and --prompt."""
    all_ids = " ".join(f"`{m.id}`" for m in metrics)
    prompt_ids = " ".join(
        f"`{m.id}`" for m in metrics if "prompt" in m.inputs.required
    )
    lines = [
        f"- `--metrics` (required): One or more of {all_ids}.",
        "- `--image` (required): Path to an image file or directory "
        "(must be a directory when `fid` or `kid` is selected).",
        (
            f"- `--prompt`: Required when {prompt_ids} is selected; "
            "prohibited otherwise."
        ),
    ]
    return "\n".join(lines)


def replace_block(
    content: str, start_marker: str, end_marker: str, new_body: str
) -> str:
    """Replace content between markers, preserving markers."""
    start_count = content.count(start_marker)
    if start_count != 1:
        raise ValueError(
            f"Expected start marker {start_marker!r} to appear exactly once, "
            f"found {start_count} occurrences."
        )
    end_count = content.count(end_marker)
    if end_count != 1:
        raise ValueError(
            f"Expected end marker {end_marker!r} to appear exactly once, "
            f"found {end_count} occurrences."
        )

    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker, start_pos + len(start_marker))
    if end_pos == -1:
        raise ValueError(
            f"End marker {end_marker!r} appears before start marker "
            f"{start_marker!r}."
        )

    before = content[: start_pos + len(start_marker)]
    after = content[end_pos:]
    return f"{before}\n{new_body}\n{after}"


def sync_readme(readme_text: str) -> str:
    """Generate synchronized README content from MetricRegistry."""
    metrics = list_metrics()
    table_md = generate_metrics_table(metrics)
    tutorial_md = generate_tutorial_metrics(metrics)

    updated = replace_block(
        readme_text, TABLE_START_MARKER, TABLE_END_MARKER, table_md
    )
    updated = replace_block(
        updated, TUTORIAL_START_MARKER, TUTORIAL_END_MARKER, tutorial_md
    )
    return updated


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via temporary file and replace."""
    import os
    import tempfile

    parent = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=parent,
        prefix=f".{path.name}.tmp_",
        delete=False,
    ) as tf:
        tf.write(text)
        temp_name = tf.name

    os.replace(temp_name, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync MetricRegistry specifications into README.md."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--check",
        action="store_true",
        help="Check whether README.md is in sync with MetricRegistry.",
    )
    mode_group.add_argument(
        "--write",
        action="store_true",
        help="Update README.md in-place with MetricRegistry specifications.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=REPO_ROOT / "README.md",
        help="Path to README.md file (default: repository root README.md).",
    )
    args = parser.parse_args()

    readme_path: Path = args.readme
    if not readme_path.is_file():
        print(f"Error: README not found at {readme_path}", file=sys.stderr)
        return 1

    original = readme_path.read_text(encoding="utf-8")
    try:
        updated = sync_readme(original)
    except ValueError as exc:
        print(f"Error syncing README: {exc}", file=sys.stderr)
        return 1

    if args.check:
        if original != updated:
            print(
                "README.md is out of sync with MetricRegistry!\n"
                "Please run: python scripts/sync_readme_registry.py --write",
                file=sys.stderr,
            )
            return 1
        print("README.md is synchronized with MetricRegistry.")
        return 0

    if args.write:
        if original != updated:
            _atomic_write_text(readme_path, updated)
            print(f"Updated {readme_path} with MetricRegistry specifications.")
        else:
            print("README.md is already up to date.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
