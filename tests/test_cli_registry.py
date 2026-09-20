"""Tests for CLI metric list/show and task/objective filtering (M8-04)."""

import json
import subprocess
import sys

import pytest

from image_evaluator.main import cli, main


def test_cli_list_text_default(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify default text list outputs all 15 registered metrics."""
    res = main(["list"])
    out = capsys.readouterr().out
    assert isinstance(res, list)
    assert len(res) == 15
    assert "Available Evaluation Metrics (15 total):" in out
    assert "directional_clip" in out
    assert "clip_i" in out
    assert "dino_similarity" in out
    assert "hpsv2" in out
    assert "image_reward" in out
    assert "vqascore" in out
    assert "ssim" in out
    assert "clip" in out


def test_cli_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --format json outputs RFC 8259 compliant JSON array."""
    res = main(["list", "--format", "json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 15
    metric_ids = {m["id"] for m in data}
    assert "directional_clip" in metric_ids
    assert "clip_i" in metric_ids
    assert "dino_similarity" in metric_ids
    assert "hpsv2" in metric_ids
    assert "image_reward" in metric_ids
    assert "vqascore" in metric_ids
    assert "ssim" in metric_ids
    assert res == data


def test_cli_list_filter_task(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify task filtering in list command."""
    res = main(["list", "--task", "image_editing"])
    out = capsys.readouterr().out
    assert "task='image_editing'" in out
    assert "directional_clip" in out
    for item in res:
        assert "image_editing" in item["tasks"]


def test_cli_list_filter_objective(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify objective filtering in list command."""
    res = main(["list", "--objective", "pixel_fidelity"])
    out = capsys.readouterr().out
    assert "objective='pixel_fidelity'" in out
    assert len(res) > 0
    for item in res:
        assert "pixel_fidelity" in item["objectives"]


def test_cli_list_filter_combined(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify combined task and objective filtering in list command."""
    res = main(
        [
            "list",
            "--task",
            "image_reconstruction",
            "--objective",
            "pixel_fidelity",
        ]
    )
    out = capsys.readouterr().out
    assert "task='image_reconstruction'" in out
    assert "objective='pixel_fidelity'" in out
    assert len(res) > 0
    for item in res:
        assert "image_reconstruction" in item["tasks"]
        assert "pixel_fidelity" in item["objectives"]


def test_cli_list_filter_no_match(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify clean message when filter yields no matches."""
    res = main(["list", "--task", "nonexistent_task"])
    out = capsys.readouterr().out
    assert len(res) == 0
    assert "No metrics found matching task='nonexistent_task'." in out


def test_cli_show_text(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify text inspection of a single metric."""
    res = main(["show", "directional_clip"])
    out = capsys.readouterr().out
    assert res["id"] == "directional_clip"
    assert "Metric: directional_clip" in out
    assert "Directional CLIP" in out
    assert "higher_is_better" in out
    assert "transformers" in out
    assert "docs/directional-clip.md" in out


def test_cli_show_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify JSON inspection of a single metric."""
    res = main(["show", "ssim", "--format", "json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["id"] == "ssim"
    assert data["score_direction"] == "higher_is_better"
    assert "image" in data["inputs"]["required"]
    assert "reference_image" in data["inputs"]["required"]
    assert res == data


def test_cli_show_unknown_metric_exits_1_no_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify unknown metric in show exits with 1 and 0 traceback."""
    exit_code = cli(["show", "nonexistent_metric"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "error: Unknown metric 'nonexistent_metric'." in err
    assert "Available metrics:" in err
    assert "Traceback" not in err


def test_cli_flag_aliases_parity(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --list-metrics and --show-metric flag forms match subcommand."""
    res_list = main(["--list-metrics", "--format", "json"])
    capsys.readouterr()
    assert len(res_list) == 15

    res_show = main(["--show-metric", "lpips", "--format", "json"])
    capsys.readouterr()
    assert res_show["id"] == "lpips"


def test_cli_metrics_subcommand_alias(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify 'metrics list' and 'metrics show' work as aliases."""
    res_list = main(["metrics", "list", "--format", "json"])
    capsys.readouterr()
    assert len(res_list) == 15

    res_show = main(["metrics", "show", "clip", "--format", "json"])
    capsys.readouterr()
    assert res_show["id"] == "clip"


def test_cli_registry_zero_heavy_import_isolation() -> None:
    """Verify CLI list and show commands do not load heavy frameworks."""
    code = (
        "import sys; "
        "from image_evaluator.main import main; "
        "main(['list', '--format', 'json']); "
        "assert 'torch' not in sys.modules, 'torch loaded during list'; "
        "assert 'cleanfid' not in sys.modules, 'cleanfid loaded during list'; "
        "assert 'clip' not in sys.modules, 'clip loaded during list'; "
        "main(['show', 'directional_clip', '--format', 'json']); "
        "assert 'torch' not in sys.modules, 'torch loaded during show'; "
        "assert 'cleanfid' not in sys.modules, 'cleanfid loaded during show'; "
        "assert 'clip' not in sys.modules, 'clip loaded during show';"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.returncode == 0


def test_cli_unknown_metric_after_registry_reload() -> None:
    """Reloading Registry does not leak its internal exception through CLI."""
    code = """
import importlib
import image_evaluator.registry as registry
from image_evaluator.main import cli

importlib.reload(registry)
assert cli(["show", "missing_after_reload"]) == 1
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Unknown metric 'missing_after_reload'" in completed.stderr
    assert "Traceback" not in completed.stderr
