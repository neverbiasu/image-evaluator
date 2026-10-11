"""Focused regression and contract tests for tag version and Changelog gating.

Verifies that scripts/verify_tag_version.py enforces:
- Exact version matching between git tag and pyproject.toml
- Final release Changelog section presence and ISO date format
- Rejection of 'Unreleased', 'Not yet released', and RC wording in the section
- Isolation from historical sections (historical notes do not trigger false
  positives)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.verify_tag_version import (
    extract_changelog_section,
    validate_changelog_section,
    verify_release_contract,
)


def _create_fixture_files(
    tmp_path: Path,
    version: str,
    changelog_content: str,
) -> tuple[Path, Path]:
    """Helper to create dummy pyproject.toml and CHANGELOG.md files."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f'[tool.poetry]\nname = "image-evaluator"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(changelog_content, encoding="utf-8")
    return pyproject, changelog


def test_matching_tag_version_and_dated_changelog_passes(
    tmp_path: Path,
) -> None:
    """Verify matching tag, package version, and dated Changelog passes."""
    content = """# Changelog

## [0.7.0] - 2026-10-09

### Changed
- Production release notes.
"""
    pyproject, changelog = _create_fixture_files(tmp_path, "0.7.0", content)
    ok, msg = verify_release_contract(
        "v0.7.0",
        pyproject_path=pyproject,
        changelog_path=changelog,
    )
    assert ok is True
    assert "release gate passed" in msg


def test_matching_tag_version_with_unreleased_fails(tmp_path: Path) -> None:
    """Verify tag with 'Unreleased' in Changelog section fails."""
    content = """# Changelog

## [0.7.0] - Unreleased

### Changed
- Still unreleased.
"""
    pyproject, changelog = _create_fixture_files(tmp_path, "0.7.0", content)
    ok, msg = verify_release_contract(
        "v0.7.0",
        pyproject_path=pyproject,
        changelog_path=changelog,
    )
    assert ok is False
    assert "Unreleased" in msg


def test_matching_tag_version_with_rc_wording_fails(tmp_path: Path) -> None:
    """Verify dated section containing RC or 'Not yet released' fails."""
    content = """# Changelog

## [0.7.0] - 2026-10-09

*Release Candidate build (0.7.0-rc). Not yet released.*

### Changed
- Work in progress.
"""
    pyproject, changelog = _create_fixture_files(tmp_path, "0.7.0", content)
    ok, msg = verify_release_contract(
        "v0.7.0",
        pyproject_path=pyproject,
        changelog_path=changelog,
    )
    assert ok is False
    assert "Not yet released" in msg or "Release Candidate" in msg


def test_missing_matching_changelog_section_fails(tmp_path: Path) -> None:
    """Verify failure when matching Changelog version section is missing."""
    content = """# Changelog

## [0.6.1] - 2026-09-20

### Fixed
- Previous release.
"""
    pyproject, changelog = _create_fixture_files(tmp_path, "0.7.0", content)
    ok, msg = verify_release_contract(
        "v0.7.0",
        pyproject_path=pyproject,
        changelog_path=changelog,
    )
    assert ok is False
    assert "Missing Changelog section for version 0.7.0" in msg


def test_mismatched_tag_package_version_fails(tmp_path: Path) -> None:
    """Verify failure when tag does not match pyproject.toml version."""
    content = """# Changelog

## [0.7.0] - 2026-10-09

### Changed
- Release notes.
"""
    pyproject, changelog = _create_fixture_files(tmp_path, "0.7.0", content)
    ok, msg = verify_release_contract(
        "v0.7.1",
        pyproject_path=pyproject,
        changelog_path=changelog,
    )
    assert ok is False
    assert "does not match pyproject.toml version" in msg


def test_unrelated_historical_rc_text_does_not_fail_current(
    tmp_path: Path,
) -> None:
    """Verify historical RC notes do not fail current release."""
    content = """# Changelog

## [0.7.0] - 2026-10-09

### Changed
- Official clean release.

## [0.6.1] - 2026-09-20

*Release Candidate build (0.6.1-rc). Not yet released.*

### Fixed
- Old fix.
"""
    pyproject, changelog = _create_fixture_files(tmp_path, "0.7.0", content)
    ok, msg = verify_release_contract(
        "v0.7.0",
        pyproject_path=pyproject,
        changelog_path=changelog,
    )
    assert ok is True
    assert "release gate passed" in msg


def test_extract_changelog_section_bounds() -> None:
    """Verify extract_changelog_section isolates target version section."""
    text = """# Header

## [1.0.0] - 2026-10-10
Section 1.0.0 content line 1
Section 1.0.0 content line 2

## [0.9.0] - 2026-09-01
Section 0.9.0 content line
"""
    extracted = extract_changelog_section(text, "1.0.0")
    assert extracted is not None
    hdr, lines = extracted
    assert hdr == "## [1.0.0] - 2026-10-10"
    assert len(lines) == 3
    assert "Section 0.9.0" not in "\n".join(lines)


def test_validate_changelog_section_requires_iso_date() -> None:
    """Verify section header must have a valid YYYY-MM-DD date."""
    valid, msg = validate_changelog_section(
        "## [1.0.0] - Sometime Soon",
        ["### Notes"],
        "1.0.0",
    )
    assert valid is False
    assert "without trailing text" in msg


def test_validate_changelog_section_invalid_month_fails() -> None:
    """Verify section header with invalid month 2026-13-01 fails."""
    valid, msg = validate_changelog_section(
        "## [1.0.0] - 2026-13-01",
        ["### Notes"],
        "1.0.0",
    )
    assert valid is False
    assert "invalid calendar date" in msg


def test_validate_changelog_section_invalid_day_fails() -> None:
    """Verify section header with invalid day 2026-02-31 fails."""
    valid, msg = validate_changelog_section(
        "## [1.0.0] - 2026-02-31",
        ["### Notes"],
        "1.0.0",
    )
    assert valid is False
    assert "invalid calendar date" in msg


def test_validate_changelog_section_trailing_text_fails() -> None:
    """Verify section header with trailing text fails."""
    valid, msg = validate_changelog_section(
        "## [1.0.0] - 2026-10-09 (final)",
        ["### Notes"],
        "1.0.0",
    )
    assert valid is False
    assert "without trailing text" in msg


def test_validate_changelog_section_valid_leap_day_passes() -> None:
    """Verify section header with valid leap day 2028-02-29 passes."""
    valid, msg = validate_changelog_section(
        "## [1.0.0] - 2028-02-29",
        ["### Notes"],
        "1.0.0",
    )
    assert valid is True
    assert "valid and dated" in msg


def test_verify_release_contract_with_invalid_calendar_date_fails(
    tmp_path: Path,
) -> None:
    """Verify contract verification fails on invalid calendar date."""
    content = """# Changelog

## [0.7.0] - 2026-02-31

### Changed
- Some notes.
"""
    pyproject, changelog = _create_fixture_files(tmp_path, "0.7.0", content)
    ok, msg = verify_release_contract(
        "v0.7.0",
        pyproject_path=pyproject,
        changelog_path=changelog,
    )
    assert ok is False
    assert "invalid calendar date" in msg


def test_cli_execution_with_real_repo_files() -> None:
    """Verify verify_tag_version.py passes as subprocess on v0.7.0."""
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "verify_tag_version.py"
    )
    cmd = [sys.executable, str(script_path), "v0.7.0"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "release gate passed" in result.stdout
