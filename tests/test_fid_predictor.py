import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from scipy import linalg

from image_evaluator.fid_predictor import FIDPredictor, FIDResult


def _create_dummy_image(path: str, size: tuple[int, int] = (32, 32)) -> None:
    """Create a minimal RGB dummy image on disk."""
    arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_init_defaults_to_cpu() -> None:
    """Verify that FIDPredictor defaults to CPU and fixes protocol."""
    pred = FIDPredictor()
    assert pred.device == torch.device("cpu")
    assert pred.BACKEND == "clean-fid"
    assert pred.VERSION == "0.1.35"
    assert pred.MODE == "clean"
    assert pred.MODEL_NAME == "inception_v3"
    assert pred.num_workers == 0
    assert pred.batch_size == 32


def test_init_explicit_device_passing() -> None:
    """Verify explicit device injection without silent fallback."""
    pred_cuda = FIDPredictor(device="cuda")
    assert pred_cuda.device == torch.device("cuda")

    pred_cpu = FIDPredictor(device=torch.device("cpu"))
    assert pred_cpu.device == torch.device("cpu")


def test_delayed_import_cleanfid() -> None:
    """Verify cleanfid is not imported at module import time."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import image_evaluator.fid_predictor; "
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
    pred = FIDPredictor()
    file_path = tmp_path / "dummy.png"
    _create_dummy_image(str(file_path))

    dir_path = tmp_path / "dummy_dir"
    dir_path.mkdir()
    _create_dummy_image(str(dir_path / "img1.png"))
    _create_dummy_image(str(dir_path / "img2.png"))

    with pytest.raises(ValueError, match="Expected directory path, got file"):
        pred.evaluate_folder_fid(str(file_path), str(dir_path))

    with pytest.raises(ValueError, match="Expected directory path, got file"):
        pred.evaluate_folder_fid(str(dir_path), str(file_path))


def test_non_existent_directory_raises_file_not_found(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify non-existent paths raise FileNotFoundError."""
    pred = FIDPredictor()
    valid_dir = tmp_path / "valid_dir"
    valid_dir.mkdir()
    _create_dummy_image(str(valid_dir / "img1.png"))
    _create_dummy_image(str(valid_dir / "img2.png"))

    non_existent = str(tmp_path / "does_not_exist")

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        pred.evaluate_folder_fid(non_existent, str(valid_dir))

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        pred.evaluate_folder_fid(str(valid_dir), non_existent)


def test_fewer_than_two_samples_raises_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify directories with fewer than 2 valid images fail."""
    pred = FIDPredictor()

    dir_empty = tmp_path / "empty"
    dir_empty.mkdir()

    dir_single = tmp_path / "single"
    dir_single.mkdir()
    _create_dummy_image(str(dir_single / "img1.png"))

    dir_valid = tmp_path / "valid"
    dir_valid.mkdir()
    _create_dummy_image(str(dir_valid / "img1.png"))
    _create_dummy_image(str(dir_valid / "img2.png"))

    with pytest.raises(ValueError, match="at least 2 images are required"):
        pred.evaluate_folder_fid(str(dir_empty), str(dir_valid))

    with pytest.raises(ValueError, match="at least 2 images are required"):
        pred.evaluate_folder_fid(str(dir_single), str(dir_valid))

    with pytest.raises(ValueError, match="at least 2 images are required"):
        pred.evaluate_folder_fid(str(dir_valid), str(dir_empty))

    with pytest.raises(ValueError, match="at least 2 images are required"):
        pred.evaluate_folder_fid(str(dir_valid), str(dir_single))


def test_unreadable_or_corrupt_image_raises_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify corrupted image files fail explicitly."""
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    corrupt_file = gen_dir / "corrupt.png"
    with open(corrupt_file, "wb") as f:
        f.write(b"corrupt non-image binary payload")

    with pytest.raises(ValueError, match="Unreadable image file"):
        pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))


def test_unreadable_npy_file_raises_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify corrupted npy files fail explicitly."""
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref_npy"
    ref_dir.mkdir()
    corrupt_npy = ref_dir / "corrupt.npy"
    with open(corrupt_npy, "wb") as f:
        f.write(b"corrupt numpy file")
    _create_dummy_image(str(ref_dir / "img1.png"))

    gen_dir = tmp_path / "gen_npy"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with pytest.raises(ValueError, match="Unreadable numpy file"):
        pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))


def test_different_sample_counts_and_directory_roles(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify unequal sample counts succeed and map directory roles."""
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref_counts"
    ref_dir.mkdir()
    for i in range(2):
        _create_dummy_image(str(ref_dir / f"ref_{i}.png"))

    gen_dir = tmp_path / "gen_counts"
    gen_dir.mkdir()
    for i in range(5):
        _create_dummy_image(str(gen_dir / f"gen_{i}.png"))

    with patch("cleanfid.fid.compute_fid", return_value=15.42) as mock_compute:
        res = pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))

        mock_compute.assert_called_once_with(
            fdir1=str(ref_dir),
            fdir2=str(gen_dir),
            mode="clean",
            model_name="inception_v3",
            num_workers=0,
            batch_size=32,
            device=pred.device,
            verbose=False,
        )

        assert res.fid == pytest.approx(15.42)
        assert res["fid"] == pytest.approx(15.42)
        assert res["score"] == pytest.approx(15.42)
        assert res["Nref"] == 2
        assert res["Ngen"] == 5
        assert res["backend"] == "clean-fid"
        assert res["version"] == "0.1.35"
        assert res["mode"] == "clean"
        assert res["model"] == "inception_v3"
        assert res["device"] == "cpu"
        assert isinstance(res, FIDResult)
        assert isinstance(res, dict)


def test_recursive_discovery_and_non_image_filtering(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify recursive discovery finds nested images and ignores non-images.
    """
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref_recursive"
    sub_dir = ref_dir / "nested" / "deep"
    sub_dir.mkdir(parents=True)

    _create_dummy_image(str(ref_dir / "root.jpg"))
    _create_dummy_image(str(sub_dir / "deep.PNG"))
    _create_dummy_image(str(sub_dir / "extra.webp"))

    # Non-image files that should be ignored in image count
    with open(ref_dir / "notes.txt", "w") as f:
        f.write("text file")
    with open(sub_dir / "data.csv", "w") as f:
        f.write("a,b,c")

    gen_dir = tmp_path / "gen_recursive"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "g1.bmp"))
    _create_dummy_image(str(gen_dir / "g2.jpeg"))

    with patch("cleanfid.fid.compute_fid", return_value=8.10) as mock_compute:
        res = pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))
        assert mock_compute.called
        assert res.Nref == 3
        assert res.Ngen == 2


def test_non_finite_return_value_raises_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify non-finite return values (NaN, Inf) fail explicitly."""
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref_nan"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_nan"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch("cleanfid.fid.compute_fid", return_value=float("nan")):
        with pytest.raises(ValueError, match="non-finite value"):
            pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))

    with patch("cleanfid.fid.compute_fid", return_value=float("inf")):
        with pytest.raises(ValueError, match="non-finite value"):
            pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))


def test_backend_exception_propagates(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify exceptions from upstream cleanfid propagate transparently."""
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref_exc"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_exc"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch(
        "cleanfid.fid.compute_fid",
        side_effect=RuntimeError("CUDA out of memory"),
    ):
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))


def test_cleanfid_version_match_succeeds(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify exact version match clean-fid 0.1.35 proceeds normally."""
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref_match"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_match"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch("importlib.metadata.version", return_value="0.1.35"):
        with patch("cleanfid.fid.compute_fid", return_value=3.14) as mock_fid:
            res = pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))
            mock_fid.assert_called_once()
            assert res.fid == pytest.approx(3.14)
            assert res.version == "0.1.35"


def test_cleanfid_version_mismatch_fails_and_does_not_call_backend(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify wrong version raises RuntimeError without calling backend."""
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref_mismatch"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_mismatch"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch("importlib.metadata.version", return_value="0.2.0"):
        with patch("cleanfid.fid.compute_fid") as mock_fid:
            with pytest.raises(
                RuntimeError, match="clean-fid version mismatch"
            ):
                pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))
            mock_fid.assert_not_called()


def test_cleanfid_missing_metadata_raises_import_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify missing clean-fid package raises informative ImportError."""
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref_missing"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_missing"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch(
        "importlib.metadata.version",
        side_effect=Exception("Package clean-fid not found"),
    ):
        with patch("cleanfid.fid.compute_fid") as mock_fid:
            with pytest.raises(
                ImportError, match="clean-fid is not installed"
            ):
                pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))
            mock_fid.assert_not_called()


def test_sample_sensitivity_warning_emitted(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify sample size sensitivity warning is emitted on evaluation."""
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref_warn"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_warn"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch("cleanfid.fid.compute_fid", return_value=5.0):
        with pytest.warns(
            UserWarning, match="FID is sensitive to sample size"
        ):
            pred.evaluate_folder_fid(str(ref_dir), str(gen_dir))


def test_evaluate_fid_alias(tmp_path: pytest.TempPathFactory) -> None:
    """Verify evaluate_fid is a valid alias for evaluate_folder_fid."""
    pred = FIDPredictor()

    ref_dir = tmp_path / "ref_alias"
    ref_dir.mkdir()
    _create_dummy_image(str(ref_dir / "img1.png"))
    _create_dummy_image(str(ref_dir / "img2.png"))

    gen_dir = tmp_path / "gen_alias"
    gen_dir.mkdir()
    _create_dummy_image(str(gen_dir / "img1.png"))
    _create_dummy_image(str(gen_dir / "img2.png"))

    with patch.object(
        pred, "evaluate_folder_fid", return_value="mock_result"
    ) as mock_fn:
        res = pred.evaluate_fid(str(ref_dir), str(gen_dir))
        mock_fn.assert_called_once_with(str(ref_dir), str(gen_dir))
        assert res == "mock_result"


# Mathematical Tests against Independent Formula (Requirement 8)
def _independent_frechet_distance(
    feats1: np.ndarray, feats2: np.ndarray, eps: float = 1e-6
) -> float:
    """Independent implementation of Fréchet Distance for verification."""
    m1 = np.mean(feats1, axis=0)
    m2 = np.mean(feats2, axis=0)
    s1 = np.cov(feats1, rowvar=False)
    s2 = np.cov(feats2, rowvar=False)

    diff = m1 - m2

    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(
        diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * tr_covmean
    )


def test_frechet_distance_math_against_independent_formula() -> None:
    """Verify cleanfid frechet_distance matches independent math formula."""
    from cleanfid.fid import frechet_distance

    rng = np.random.RandomState(42)
    feats1 = rng.randn(20, 5)
    feats2 = rng.randn(25, 5) + 0.5

    mu1 = np.mean(feats1, axis=0)
    sigma1 = np.cov(feats1, rowvar=False)
    mu2 = np.mean(feats2, axis=0)
    sigma2 = np.cov(feats2, rowvar=False)

    clean_fid_score = frechet_distance(mu1, sigma1, mu2, sigma2)
    independent_score = _independent_frechet_distance(feats1, feats2)

    assert clean_fid_score == pytest.approx(independent_score, abs=1e-5)
    assert clean_fid_score > 0.0


def test_frechet_distance_identical_features_zero() -> None:
    """Verify self-distance between identical distributions is zero."""
    from cleanfid.fid import frechet_distance

    rng = np.random.RandomState(123)
    feats = rng.randn(30, 4)

    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)

    dist = frechet_distance(mu, sigma, mu, sigma)
    assert dist == pytest.approx(0.0, abs=1e-5)


def test_frechet_distance_symmetry() -> None:
    """Verify frechet_distance is symmetric under argument permutation."""
    from cleanfid.fid import frechet_distance

    rng = np.random.RandomState(999)
    f1 = rng.randn(15, 3)
    f2 = rng.randn(18, 3) * 1.5 + 2.0

    mu1, sig1 = np.mean(f1, axis=0), np.cov(f1, rowvar=False)
    mu2, sig2 = np.mean(f2, axis=0), np.cov(f2, rowvar=False)

    d12 = frechet_distance(mu1, sig1, mu2, sig2)
    d21 = frechet_distance(mu2, sig2, mu1, sig1)

    assert d12 == pytest.approx(d21, abs=1e-5)
