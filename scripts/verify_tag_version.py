#!/usr/bin/env python3
"""Verify git tag matches pyproject.toml package version.

Usage:
    python scripts/verify_tag_version.py <tag_name>
    python scripts/verify_tag_version.py --env GITHUB_REF_NAME
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify git tag matches pyproject.toml version."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("tag", nargs="?", help="Git tag (e.g. v0.6.1)")
    group.add_argument(
        "--env",
        dest="env_var",
        help="Environment variable containing git tag name.",
    )
    args = parser.parse_args()

    raw_tag = os.environ.get(args.env_var) if args.env_var else args.tag
    if not raw_tag:
        print("Error: No tag specified.", file=sys.stderr)
        return 1

    tag_version_str = raw_tag.removeprefix("v")

    pyproject_file = REPO_ROOT / "pyproject.toml"
    with open(pyproject_file, "rb") as f:
        package_version_str = tomllib.load(f)["tool"]["poetry"]["version"]

    try:
        tag_v = Version(tag_version_str)
        pkg_v = Version(package_version_str)
    except Exception as exc:
        print(f"Error parsing version: {exc}", file=sys.stderr)
        return 1

    if tag_v != pkg_v:
        print(
            f"Error: Git tag {raw_tag!r} (version {tag_version_str}) does not "
            f"match pyproject.toml version {package_version_str!r}.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Success: Git tag {raw_tag!r} matches pyproject.toml version "
        f"{package_version_str!r}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
