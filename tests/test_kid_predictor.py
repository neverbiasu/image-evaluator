import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from image_evaluator.kid_predictor import KIDPredictor, KIDResult


def _create_dummy_image(path: str, size: tuple[int, int] = (32, 32)) -> None:
    """Create a minimal RGB dummy image on disk."""
    arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_init_defaults_to_cpu_and_seed_zero() -> None:
    """Verify that KIDPredictor defaults to CPU, seed 0, and fixes protocol."""
    pred = KIDPredictor()
    assert pred.device == torch.device("cpu")
    assert pred.seed == 0
    assert pred.BACKEND == "clean-fid"
    assert pred.VERSION == "0.1.35"
    assert pred.MODE == "clean"
    assert pred.MODEL_NAME == "inception_v3"
    assert pred.NUM_SUBSETS == 100
    assert pred.MAX_SUBSET_SIZE == 1000
    assert pred.MIN_SAMPLES == 2
    assert pred.num_workers == 0
    assert pred.batch_size == 32


def test_init_explicit_device_and_seed_passing() -> None:
    """Verify explicit device and seed injection without silent fallback."""
    pred_cuda = KIDPredictor(device="cuda", seed=42)
    assert pred_cuda.device == torch.device("cuda")
    assert pred_cuda.seed == 42

    pred_none = KIDPredictor(device=None, seed=None)
    assert pred_none.device == torch.device("cpu")
    assert pred_none.seed is None


def test_delayed_import_cleanfid() -> None:
    """Verify cleanfid is not imported at module import time."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import image_evaluator.kid_predictor; "
            "assert 'cleanfid' not in sys.modules, "
            "'cleanfid was eagerly imported!'"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Stderr: {res.stderr}"


def test_non_directory_inputs_raise_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify non-directory paths raise ValueError."""
    pred = KIDPredictor()
    file_path = tmp_path / "dummy.png"
    _create_dummy_image(str(file_path))

    dir_path = tmp_path / "dummy_dir"
    dir_path.mkdir()
    _create_dummy_image(str(dir_path / "img1.png"))
    _create_dummy_image(str(dir_path / "img2.png"))

    with pytest.raises(ValueError, match="Expected directory path, got file"):
        pred.evaluate_folder_kid(str(file_path), str(dir_path))

    with pytest.raises(ValueError, match="Expected directory path, got file"):
        pred.evaluate_folder_kid(str(dir_path), str(file_path))


def test_non_existent_directory_raises_file_not_found(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify non-existent paths raise FileNotFoundError."""
    pred = KIDPredictor()
    valid_dir = tmp_path / "valid_dir"
    valid_dir.mkdir()
    _create_dummy_image(str(valid_dir / "img1.png"))
    _create_dummy_image(str(valid_dir / "img2.png"))

    non_existent = str(tmp_path / "does_not_exist")

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        pred.evaluate_folder_kid(non_existent, str(valid_dir))

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        pred.evaluate_folder_kid(str(valid_dir), non_existent)


def test_fewer_than_two_samples_raises_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify directories with fewer than 2 valid images fail."""
    pred = KIDPredictor()

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    one_img_dir = tmp_path / "one_img_dir"
    one_img_dir.mkdir()
    _create_dummy_image(str(one_img_dir / "img1.png"))

    valid_dir = tmp_path / "valid_dir"
    valid_dir.mkdir()
    _create_dummy_image(str(valid_dir / "img1.png"))
    _create_dummy_image(str(valid_dir / "img2.png"))

    # Empty reference directory
    with pytest.raises(ValueError, match="at least 2 images are required"):
        pred.evaluate_folder_kid(str(empty_dir), str(valid_dir))

    # Single image reference directory
    with pytest.raises(ValueError, match="at least 2 images are required"):
        pred.evaluate_folder_kid(str(one_img_dir), str(valid_dir))

    # Single image generated directory
    with pytest.raises(ValueError, match="at least 2 images are required"):
        pred.evaluate_folder_kid(str(valid_dir), str(one_img_dir))


def test_corrupted_image_raises_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify corrupted image file causes batch failure."""
    pred = KIDPredictor()

    ref_dir = tmp_path / "ref_dir"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_dir"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    corrupt_file = gen_dir / "corrupt.png"
    corrupt_file.write_bytes(b"not an image content")

    with pytest.raises(ValueError, match="Unreadable image file"):
        pred.evaluate_folder_kid(str(ref_dir), str(gen_dir))


def test_corrupted_npy_file_raises_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify corrupted npy file causes batch failure."""
    pred = KIDPredictor()

    ref_dir = tmp_path / "ref_npy"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_npy"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    bad_npy = gen_dir / "corrupt.npy"
    bad_npy.write_bytes(b"broken npy")

    with pytest.raises(ValueError, match="Unreadable numpy file"):
        pred.evaluate_folder_kid(str(ref_dir), str(gen_dir))


def test_different_sample_counts_and_directory_roles(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify unequal sample counts work and directory roles are bound."""
    pred = KIDPredictor(seed=123)

    ref_dir = tmp_path / "ref_dir"
    ref_dir.mkdir()
    for i in range(2):
        _create_dummy_image(str(ref_dir / f"ref_{i}.png"))

    gen_dir = tmp_path / "gen_dir"
    gen_dir.mkdir()
    for i in range(5):
        _create_dummy_image(str(gen_dir / f"gen_{i}.png"))

    with patch("cleanfid.fid.compute_kid", return_value=0.0123) as mock_kid:
        res = pred.evaluate_folder_kid(str(ref_dir), str(gen_dir))

        mock_kid.assert_called_once_with(
            fdir1=str(ref_dir),
            fdir2=str(gen_dir),
            mode="clean",
            num_workers=0,
            batch_size=32,
            device=torch.device("cpu"),
            verbose=False,
        )

        assert isinstance(res, KIDResult)
        assert res.kid == pytest.approx(0.0123)
        assert res.score == pytest.approx(0.0123)
        assert res["kid"] == pytest.approx(0.0123)
        assert res.backend == "clean-fid"
        assert res.version == "0.1.35"
        assert res.mode == "clean"
        assert res.model == "inception_v3"
        assert res.device == "cpu"
        assert res.num_subsets == 100
        assert res.max_subset_size == 1000
        assert res.seed == 123
        assert res.Nref == 2
        assert res.Ngen == 5


def test_negative_kid_value_accepted_as_valid(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify negative KID is accepted as valid float."""
    pred = KIDPredictor()

    ref_dir = tmp_path / "ref_neg"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_neg"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch("cleanfid.fid.compute_kid", return_value=-0.0025):
        res = pred.evaluate_folder_kid(str(ref_dir), str(gen_dir))
        assert res.kid == pytest.approx(-0.0025)
        assert res.score == pytest.approx(-0.0025)
        assert "kid=-0.002500" in repr(res)


def test_non_finite_return_value_raises_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify NaN or Inf from backend raises ValueError."""
    pred = KIDPredictor()

    ref_dir = tmp_path / "ref_nan"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_nan"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch("cleanfid.fid.compute_kid", return_value=float("nan")):
        with pytest.raises(ValueError, match="non-finite value"):
            pred.evaluate_folder_kid(str(ref_dir), str(gen_dir))

    with patch("cleanfid.fid.compute_kid", return_value=float("inf")):
        with pytest.raises(ValueError, match="non-finite value"):
            pred.evaluate_folder_kid(str(ref_dir), str(gen_dir))


def test_recursive_discovery_and_non_image_filtering(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify recursive scanning discovers images and ignores other files."""
    pred = KIDPredictor()

    ref_dir = tmp_path / "ref_nested"
    sub1 = ref_dir / "sub1"
    sub2 = ref_dir / "sub2" / "deep"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    _create_dummy_image(str(sub1 / "a.jpg"))
    _create_dummy_image(str(sub2 / "b.webp"))
    _create_dummy_image(str(ref_dir / "c.png"))
    (ref_dir / "notes.txt").write_text("ignore me")

    gen_dir = tmp_path / "gen_nested"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "g1.bmp"))
    _create_dummy_image(str(gen_dir / "g2.tiff"))

    with patch("cleanfid.fid.compute_kid", return_value=0.05):
        res = pred.evaluate_folder_kid(str(ref_dir), str(gen_dir))
        assert res.Nref == 3
        assert res.Ngen == 2


def test_seed_determinism(tmp_path: pytest.TempPathFactory) -> None:
    """Verify seed ensures reproducible random subset sampling."""
    ref_dir = tmp_path / "ref_seed"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_seed"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    # Side effect simulating numpy random choice dependency
    def mock_compute_kid(*args, **kwargs):
        return float(np.random.rand())

    with patch("cleanfid.fid.compute_kid", side_effect=mock_compute_kid):
        pred_a = KIDPredictor(seed=42)
        score_a1 = pred_a.evaluate_folder_kid(
            str(ref_dir), str(gen_dir)
        ).kid

        pred_b = KIDPredictor(seed=42)
        score_a2 = pred_b.evaluate_folder_kid(
            str(ref_dir), str(gen_dir)
        ).kid

        assert score_a1 == score_a2

        pred_c = KIDPredictor(seed=99)
        score_diff = pred_c.evaluate_folder_kid(
            str(ref_dir), str(gen_dir)
        ).kid

        assert score_a1 != score_diff


def test_seed_does_not_pollute_global_random_state(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify evaluating with a seed restores previous global random state."""
    ref_dir = tmp_path / "ref_state"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_state"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    np.random.seed(12345)
    expected_next_rand = np.random.rand()
    # Reset to identical state
    np.random.seed(12345)

    pred = KIDPredictor(seed=999)
    with patch("cleanfid.fid.compute_kid", return_value=0.01):
        pred.evaluate_folder_kid(str(ref_dir), str(gen_dir))

    actual_next_rand = np.random.rand()
    assert actual_next_rand == expected_next_rand


def test_cleanfid_version_mismatch_fails(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify mismatched clean-fid version raises RuntimeError."""
    pred = KIDPredictor()

    ref_dir = tmp_path / "ref_ver"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_ver"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch("importlib.metadata.version", return_value="0.2.0"):
        with pytest.raises(RuntimeError, match="clean-fid version mismatch"):
            pred.evaluate_folder_kid(str(ref_dir), str(gen_dir))


def test_cleanfid_missing_metadata_raises_import_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify missing clean-fid package raises ImportError."""
    pred = KIDPredictor()

    ref_dir = tmp_path / "ref_meta"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_meta"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch(
        "importlib.metadata.version", side_effect=Exception("not found")
    ):
        with pytest.raises(ImportError, match="clean-fid is not installed"):
            pred.evaluate_folder_kid(str(ref_dir), str(gen_dir))


# Mathematical Verification of Polynomial Kernel & Unbiased MMD Formula
def _independent_kernel_distance(
    feats1: np.ndarray,
    feats2: np.ndarray,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
    seed: int | None = None,
) -> float:
    """Independent implementation of polynomial kernel MMD for verification."""
    if seed is not None:
        np.random.seed(seed)

    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0.0

    for _ in range(num_subsets):
        idx2 = np.random.choice(feats2.shape[0], m, replace=False)
        idx1 = np.random.choice(feats1.shape[0], m, replace=False)
        x = feats2[idx2]
        y = feats1[idx1]

        # Polynomial kernel: k(x, y) = (x^T y / d + 1)^3
        k_xx = (x @ x.T / n + 1.0) ** 3
        k_yy = (y @ y.T / n + 1.0) ** 3
        k_xy = (x @ y.T / n + 1.0) ** 3

        mmd2_subset = (
            (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
            + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
            - 2.0 * k_xy.sum() / (m * m)
        )
        t += mmd2_subset

    return float(t / num_subsets)


def test_kernel_distance_math_against_independent_formula() -> None:
    """Verify cleanfid kernel_distance matches independent MMD calculation."""
    from cleanfid.fid import kernel_distance

    rng = np.random.RandomState(42)
    feats1 = rng.randn(30, 8)
    feats2 = rng.randn(25, 8) + 0.8

    # Compare with identical seed
    np.random.seed(999)
    cleanfid_score = kernel_distance(
        feats1, feats2, num_subsets=50, max_subset_size=20
    )

    np.random.seed(999)
    independent_score = _independent_kernel_distance(
        feats1, feats2, num_subsets=50, max_subset_size=20, seed=999
    )

    assert cleanfid_score == pytest.approx(independent_score, abs=1e-6)


def test_kernel_distance_separates_different_distributions() -> None:
    """Verify KID separates identical from different feature distributions."""
    from cleanfid.fid import kernel_distance

    rng = np.random.RandomState(777)
    f1 = rng.randn(150, 2048)
    f2 = rng.randn(150, 2048)
    f_diff = rng.randn(150, 2048) + 2.0

    score_same = kernel_distance(
        f1, f2, num_subsets=50, max_subset_size=100
    )
    score_diff = kernel_distance(
        f1, f_diff, num_subsets=50, max_subset_size=100
    )

    # Identical distributions fluctuate closely around 0.0
    assert abs(score_same) < 0.1
    # Distinct distributions yield significantly higher KID
    assert score_diff > 10.0
    assert score_diff > abs(score_same) * 50
