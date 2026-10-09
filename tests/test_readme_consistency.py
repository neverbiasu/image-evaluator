"""README consistency checks against package metadata and the Registry."""

import subprocess
import sys
import tomllib
from pathlib import Path

import image_evaluator
from image_evaluator.registry import list_metrics
from scripts.sync_readme_registry import (
    TABLE_END_MARKER,
    TABLE_START_MARKER,
    TUTORIAL_END_MARKER,
    TUTORIAL_START_MARKER,
    sync_readme,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"


def _read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _read_pyproject_version() -> str:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return str(tomllib.load(f)["tool"]["poetry"]["version"])


def test_package_version_matches_pyproject():
    """Verify package __version__ matches single truth in pyproject.toml."""
    assert image_evaluator.__version__ == _read_pyproject_version()


def test_readme_does_not_hardcode_version():
    """README installation instructions must not pin a hardcoded version."""
    readme = _read_readme()
    assert "python -m pip install image-evaluator" in readme
    assert "image-evaluator==" not in readme


def test_readme_displays_pypi_badge():
    """README must display dynamic PyPI version badge."""
    readme = _read_readme()
    assert "https://img.shields.io/pypi/v/image-evaluator.svg" in readme
    assert "https://pypi.org/project/image-evaluator/" in readme


def test_readme_contains_required_sync_markers():
    """README must contain both pairs of sync boundary markers."""
    readme = _read_readme()
    assert TABLE_START_MARKER in readme
    assert TABLE_END_MARKER in readme
    assert TUTORIAL_START_MARKER in readme
    assert TUTORIAL_END_MARKER in readme


def test_readme_in_sync_with_registry():
    """Running sync_readme on README must produce no diff."""
    readme = _read_readme()
    synced = sync_readme(readme)
    assert readme == synced, (
        "README.md is out of sync with MetricRegistry. "
        "Run: python scripts/sync_readme_registry.py --write"
    )


def test_sync_script_check_mode_succeeds_when_in_sync():
    """Command 'python scripts/sync_readme_registry.py --check' exits 0."""
    script_path = str(REPO_ROOT / "scripts/sync_readme_registry.py")
    res = subprocess.run(
        [sys.executable, script_path, "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "README.md is synchronized with MetricRegistry." in res.stdout


def test_sync_script_check_mode_fails_on_drift(tmp_path):
    """Negative test: check mode returns non-zero when drift is introduced."""
    drift_readme = tmp_path / "README.md"
    drift_content = (
        f"# Sample\n"
        f"{TABLE_START_MARKER}\nstale table\n{TABLE_END_MARKER}\n"
        f"{TUTORIAL_START_MARKER}\nstale tutorial\n{TUTORIAL_END_MARKER}\n"
    )
    drift_readme.write_text(drift_content, encoding="utf-8")

    script_path = str(REPO_ROOT / "scripts/sync_readme_registry.py")
    res = subprocess.run(
        [
            sys.executable,
            script_path,
            "--check",
            "--readme",
            str(drift_readme),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "README.md is out of sync with MetricRegistry!" in res.stderr
    assert "python scripts/sync_readme_registry.py --write" in res.stderr


def test_readme_table_covers_all_registered_metrics():
    """Every MetricSpec in the Registry must appear in the README table."""
    readme = _read_readme()
    for spec in list_metrics():
        assert f"`{spec.id}`" in readme
        assert spec.display_name in readme


def test_readme_discloses_known_limitations():
    readme = _read_readme()
    assert "#### Known Limitations" in readme
    assert "complete Windows support remains unverified" in readme
    assert "trained primarily on English text" in readme


def test_resolve_version_prefers_pyproject_over_installed_metadata():
    """In-tree pyproject overrides different installed metadata."""
    from unittest.mock import patch

    with patch("importlib.metadata.version", return_value="0.1.0-stale"):
        resolved = image_evaluator._resolve_version()
        assert resolved == _read_pyproject_version()
        assert resolved != "0.1.0-stale"


def test_resolve_version_raises_key_error_on_missing_version(tmp_path):
    """Missing poetry.version key in adjacent pyproject raises KeyError."""
    from unittest.mock import patch

    import pytest

    fake_pkg_dir = tmp_path / "fake_pkg"
    fake_pkg_dir.mkdir()
    fake_pyproject = tmp_path / "pyproject.toml"
    fake_pyproject.write_text(
        "[tool.poetry]\nname = 'test'\n", encoding="utf-8"
    )

    fake_init = fake_pkg_dir / "__init__.py"
    fake_init.touch()

    with patch.object(image_evaluator, "__file__", str(fake_init)):
        expected_msg = r"Missing \['tool'\]\['poetry'\]\['version'\]"
        with pytest.raises(KeyError, match=expected_msg):
            image_evaluator._resolve_version()


def test_sync_script_fails_on_duplicate_markers(tmp_path):
    """Negative test: duplicate markers in README must raise error."""
    dup_readme = tmp_path / "README.md"
    dup_readme.write_text(
        f"{TABLE_START_MARKER}\n{TABLE_START_MARKER}\n{TABLE_END_MARKER}\n"
        f"{TUTORIAL_START_MARKER}\n{TUTORIAL_END_MARKER}\n",
        encoding="utf-8",
    )
    script_path = str(REPO_ROOT / "scripts/sync_readme_registry.py")
    res = subprocess.run(
        [
            sys.executable,
            script_path,
            "--check",
            "--readme",
            str(dup_readme),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "Expected start marker" in res.stderr
    assert "found 2 occurrences" in res.stderr


def test_sync_script_detects_drift_when_sixteenth_metric_added():
    """If a 16th metric is registered, current README is flagged as drift."""
    from unittest.mock import patch

    from image_evaluator.specifications import (
        ImplementationRef,
        InputContract,
        MetricSpec,
    )

    base_metrics = list(list_metrics())
    dummy_spec = MetricSpec(
        id="dummy_future_metric",
        display_name="Dummy Future Metric",
        tasks=("image_to_image",),
        objectives=("quality",),
        score_direction="higher_is_better",
        inputs=InputContract(required=("image", "prompt")),
        implementation=ImplementationRef(
            backend="dummy",
            protocol="dummy",
        ),
        docs_path="docs/metrics/dummy.md",
    )
    extended_metrics = tuple(base_metrics + [dummy_spec])
    assert len(extended_metrics) == 16

    with patch(
        "scripts.sync_readme_registry.list_metrics",
        return_value=extended_metrics,
    ):
        readme = _read_readme()
        synced = sync_readme(readme)
        # README must drift because dummy_future_metric is absent
        assert readme != synced
        assert "`dummy_future_metric`" in synced
        assert "`dummy_future_metric`" not in readme
