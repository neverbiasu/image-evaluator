#!/usr/bin/env python3
"""Verify git tag matches pyproject.toml package version and Changelog state.

Usage:
    python scripts/verify_tag_version.py <tag_name>
    python scripts/verify_tag_version.py --env GITHUB_REF_NAME
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from datetime import date
from pathlib import Path

from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent


def extract_changelog_section(
    changelog_text: str, version_str: str
) -> tuple[str, list[str]] | None:
    """Extract header line and body lines for a version in CHANGELOG.md."""
    header_pattern = re.compile(
        rf"^##\s+(?:\[{re.escape(version_str)}\]|{re.escape(version_str)})"
        r"(?:[\s\-:]+(.*))?$",
        re.IGNORECASE,
    )
    in_section = False
    header_line = ""
    section_lines: list[str] = []

    for line in changelog_text.splitlines():
        if in_section:
            if line.startswith("## ") and not line.startswith("### "):
                break
            section_lines.append(line)
        else:
            if header_pattern.match(line.strip()):
                in_section = True
                header_line = line.strip()

    if not in_section:
        return None
    return header_line, section_lines


def validate_changelog_section(
    header_line: str,
    section_lines: list[str],
    version_str: str,
) -> tuple[bool, str]:
    """Validate that the changelog section represents a finalized release."""
    header_match = re.match(
        rf"^##\s+(?:\[{re.escape(version_str)}\]|{re.escape(version_str)})"
        r"(?:[\s\-:]+(.*))?$",
        header_line,
        re.IGNORECASE,
    )
    suffix = (header_match.group(1) or "").strip() if header_match else ""

    if not suffix or re.search(r"\bunreleased\b", suffix, re.IGNORECASE):
        return (
            False,
            f"Changelog section for version {version_str} is marked as "
            f"Unreleased or lacks a release date: {header_line!r}.",
        )

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", suffix):
        return (
            False,
            f"Changelog section for version {version_str} must contain "
            f"exactly a single release date (YYYY-MM-DD) without "
            f"trailing text: {header_line!r}.",
        )

    try:
        date.fromisoformat(suffix)
    except ValueError as exc:
        return (
            False,
            f"Changelog section for version {version_str} contains an "
            f"invalid calendar date ({exc}): {suffix!r}.",
        )

    full_text = "\n".join([header_line] + section_lines)
    prohibited_checks = [
        (r"\bnot\s+yet\s+released\b", "contains 'Not yet released'"),
        (r"\brelease\s+candidate\b", "contains 'Release Candidate' wording"),
        (
            rf"\b{re.escape(version_str)}[-_]?rc\b",
            f"contains '{version_str}-rc' wording",
        ),
    ]

    for pattern, reason in prohibited_checks:
        if re.search(pattern, full_text, re.IGNORECASE):
            return (
                False,
                f"Changelog section for version {version_str} {reason}.",
            )

    return (
        True,
        f"Changelog section for version {version_str} is valid and dated.",
    )


def verify_release_contract(
    raw_tag: str,
    pyproject_path: Path | None = None,
    changelog_path: Path | None = None,
) -> tuple[bool, str]:
    """Verify tag matches pyproject.toml and changelog release state."""
    if not raw_tag:
        return False, "No tag specified."

    clean_tag = raw_tag.strip()
    if clean_tag.startswith("refs/tags/"):
        clean_tag = clean_tag.removeprefix("refs/tags/")
    tag_version_str = clean_tag.removeprefix("v")

    pyproject_file = pyproject_path or (REPO_ROOT / "pyproject.toml")
    if not pyproject_file.is_file():
        return False, f"pyproject.toml not found at {pyproject_file}."

    with open(pyproject_file, "rb") as f:
        package_version_str = tomllib.load(f)["tool"]["poetry"]["version"]

    try:
        tag_v = Version(tag_version_str)
        pkg_v = Version(package_version_str)
    except Exception as exc:
        return False, f"Error parsing version: {exc}"

    if tag_v != pkg_v:
        return (
            False,
            f"Git tag {raw_tag!r} (version {tag_version_str}) does not "
            f"match pyproject.toml version {package_version_str!r}.",
        )

    # For final release tags, enforce Changelog release state gating
    if not tag_v.is_prerelease:
        changelog_file = changelog_path or (REPO_ROOT / "CHANGELOG.md")
        if not changelog_file.is_file():
            return False, f"CHANGELOG.md not found at {changelog_file}."

        with open(changelog_file, "r", encoding="utf-8") as f:
            changelog_content = f.read()

        section = extract_changelog_section(
            changelog_content, package_version_str
        )
        if section is None:
            return (
                False,
                f"Missing Changelog section for version {package_version_str} "
                f"in {changelog_file.name}.",
            )

        header_line, section_lines = section
        valid, reason = validate_changelog_section(
            header_line, section_lines, package_version_str
        )
        if not valid:
            return False, reason

    return (
        True,
        f"Git tag {raw_tag!r} matches pyproject.toml version "
        f"{package_version_str!r} and Changelog release gate passed.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify git tag matches pyproject.toml and Changelog."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("tag", nargs="?", help="Git tag (e.g. v0.7.0)")
    group.add_argument(
        "--env",
        dest="env_var",
        help="Environment variable containing git tag name.",
    )
    args = parser.parse_args()

    raw_tag = os.environ.get(args.env_var) if args.env_var else args.tag
    valid, message = verify_release_contract(raw_tag or "")
    if not valid:
        print(f"Error: {message}", file=sys.stderr)
        return 1

    print(f"Success: {message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
