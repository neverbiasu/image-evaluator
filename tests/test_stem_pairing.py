"""Unit tests for flat stem-pairing helpers."""

import pytest

from image_evaluator._stem_pairing import (
    IMAGE_EXTS,
    allowed_exts_for_flag,
    collect_flat_dir,
    pair_dirs,
)


def test_collect_flat_dir_with_images_and_sidecars(tmp_path):
    # Setup images and sidecars
    (tmp_path / "img1.png").touch()
    (tmp_path / "img1.json").write_text('{"prompt": "hello"}')
    (tmp_path / "img2.jpg").touch()
    (tmp_path / "img2.yaml").write_text("prompt: hello")
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / ".hidden_file").touch()

    mapping = collect_flat_dir(str(tmp_path), IMAGE_EXTS)
    assert set(mapping.keys()) == {"img1", "img2"}
    assert mapping["img1"] == str(tmp_path / "img1.png")
    assert mapping["img2"] == str(tmp_path / "img2.jpg")


def test_collect_flat_dir_unsupported_file_raises_value_error(tmp_path):
    (tmp_path / "img1.png").touch()
    (tmp_path / "data.csv").write_text("1,2,3")

    with pytest.raises(ValueError, match="Unsupported file"):
        collect_flat_dir(str(tmp_path), IMAGE_EXTS)


def test_collect_flat_dir_empty_dir_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No visible files"):
        collect_flat_dir(str(tmp_path), IMAGE_EXTS)


def test_collect_flat_dir_no_matching_extension_raises_value_error(tmp_path):
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / "config.yaml").write_text("info: 1")

    with pytest.raises(
        ValueError, match="No supported files matching extensions"
    ):
        collect_flat_dir(str(tmp_path), IMAGE_EXTS)


def test_collect_flat_dir_duplicate_stem_raises_value_error(tmp_path):
    (tmp_path / "sample.png").touch()
    (tmp_path / "sample.jpg").touch()

    with pytest.raises(ValueError, match="Duplicate stem 'sample'"):
        collect_flat_dir(str(tmp_path), IMAGE_EXTS)


def test_collect_flat_dir_subdirectory_raises_value_error(tmp_path):
    (tmp_path / "img1.png").touch()
    sub = tmp_path / "subdir"
    sub.mkdir()

    with pytest.raises(
        ValueError, match="Unsupported entry \\(not a flat file\\)"
    ):
        collect_flat_dir(str(tmp_path), IMAGE_EXTS)


def test_pair_dirs_successful_with_sidecars(tmp_path):
    dir_a = tmp_path / "real"
    dir_b = tmp_path / "fake"
    dir_a.mkdir()
    dir_b.mkdir()

    # dir_a has png + sidecar json
    (dir_a / "c01.png").touch()
    (dir_a / "c01.json").write_text("{}")
    (dir_a / "c02.png").touch()
    (dir_a / "c02.json").write_text("{}")

    # dir_b has jpg + sidecar json + readme.md
    (dir_b / "c01.jpg").touch()
    (dir_b / "c01.json").write_text("{}")
    (dir_b / "c02.webp").touch()
    (dir_b / "notes.yml").write_text("test")

    pairs = pair_dirs(str(dir_a), str(dir_b), IMAGE_EXTS, IMAGE_EXTS)
    assert len(pairs) == 2
    assert pairs[0] == (str(dir_a / "c01.png"), str(dir_b / "c01.jpg"))
    assert pairs[1] == (str(dir_a / "c02.png"), str(dir_b / "c02.webp"))


def test_pair_dirs_mismatched_stems_raises_file_not_found(tmp_path):
    dir_a = tmp_path / "real"
    dir_b = tmp_path / "fake"
    dir_a.mkdir()
    dir_b.mkdir()

    (dir_a / "c01.png").touch()
    (dir_a / "c02.png").touch()

    (dir_b / "c01.png").touch()
    (dir_b / "c03.png").touch()

    with pytest.raises(FileNotFoundError, match="Stem mismatch"):
        pair_dirs(str(dir_a), str(dir_b), IMAGE_EXTS, IMAGE_EXTS)


def test_collect_flat_dir_text_directory_preserves_txt_with_sidecars(tmp_path):
    from image_evaluator._stem_pairing import TEXT_EXTS

    (tmp_path / "t01.txt").write_text("prompt one")
    (tmp_path / "t01.json").write_text('{"seed": 1}')
    (tmp_path / "t02.txt").write_text("prompt two")
    (tmp_path / "t02.yaml").write_text("seed: 2")
    (tmp_path / "t02.yml").write_text("seed: 2")

    mapping = collect_flat_dir(str(tmp_path), TEXT_EXTS)
    assert set(mapping.keys()) == {"t01", "t02"}
    assert mapping["t01"] == str(tmp_path / "t01.txt")
    assert mapping["t02"] == str(tmp_path / "t02.txt")


def test_collect_flat_dir_text_directory_rejects_unsupported_files(tmp_path):
    from image_evaluator._stem_pairing import TEXT_EXTS

    (tmp_path / "t01.txt").write_text("prompt")
    (tmp_path / "t01.png").touch()

    with pytest.raises(ValueError, match="Unsupported file"):
        collect_flat_dir(str(tmp_path), TEXT_EXTS)


def test_collect_flat_dir_image_directory_rejects_md_and_csv(tmp_path):
    (tmp_path / "img1.png").touch()
    (tmp_path / "notes.md").write_text("documentation")

    with pytest.raises(ValueError, match="Unsupported file"):
        collect_flat_dir(str(tmp_path), IMAGE_EXTS)

    (tmp_path / "notes.md").unlink()
    (tmp_path / "data.csv").write_text("1,2,3")

    with pytest.raises(ValueError, match="Unsupported file"):
        collect_flat_dir(str(tmp_path), IMAGE_EXTS)

    (tmp_path / "data.csv").unlink()
    (tmp_path / "prompt.txt").write_text("prompt")

    with pytest.raises(ValueError, match="Unsupported file"):
        collect_flat_dir(str(tmp_path), IMAGE_EXTS)


def test_allowed_exts_for_flag():
    assert allowed_exts_for_flag("img") == IMAGE_EXTS
    assert allowed_exts_for_flag("txt") == frozenset({".txt"})
    with pytest.raises(TypeError, match="Got unexpected modality"):
        allowed_exts_for_flag("video")
