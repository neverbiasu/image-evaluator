import pytest

from image_evaluator.clip_score_predictor import DummyDataset


def _mk(d, names):
    for n in names:
        (d / n).write_text("x")


def test_paired_by_stem_ignore_ext(tmp_path):
    i = tmp_path / "img"
    t = tmp_path / "txt"
    i.mkdir()
    t.mkdir()
    _mk(i, ["b.png", "a.jpg"])
    _mk(t, ["a.txt", "b.txt"])
    ds = DummyDataset(str(i), str(t), "img", "txt")
    assert len(ds) == 2
    assert ds.real_folder[0].endswith("a.jpg")
    assert ds.fake_folder[0].endswith("a.txt")


def test_missing_stem_rejected(tmp_path):
    i = tmp_path / "i"
    t = tmp_path / "t"
    i.mkdir()
    t.mkdir()
    _mk(i, ["a.png"])
    _mk(t, ["b.txt"])
    with pytest.raises(FileNotFoundError):
        DummyDataset(str(i), str(t), "img", "txt")


def test_duplicate_stem_rejected(tmp_path):
    i = tmp_path / "i"
    t = tmp_path / "t"
    i.mkdir()
    t.mkdir()
    _mk(i, ["a.png", "a.jpg"])
    _mk(t, ["a.txt"])
    with pytest.raises(ValueError, match="Duplicate stem"):
        DummyDataset(str(i), str(t), "img", "txt")


def test_unsupported_visible_rejected(tmp_path):
    i = tmp_path / "i"
    t = tmp_path / "t"
    i.mkdir()
    t.mkdir()
    _mk(i, ["a.png", "note.csv"])
    _mk(t, ["a.txt"])
    with pytest.raises(ValueError, match="Unsupported"):
        DummyDataset(str(i), str(t), "img", "txt")


def test_case_sensitive_stems(tmp_path):
    i = tmp_path / "i"
    t = tmp_path / "t"
    i.mkdir()
    t.mkdir()
    _mk(i, ["A.png"])
    _mk(t, ["a.txt"])
    with pytest.raises(FileNotFoundError):
        DummyDataset(str(i), str(t), "img", "txt")


def test_unpaired_dir_plus_prompt(tmp_path):
    i = tmp_path / "i"
    i.mkdir()
    _mk(i, ["b.png", "a.png"])
    ds = DummyDataset(str(i), "one prompt", "img", "txt")
    assert len(ds) == 2
    assert ds.fake_folder == "one prompt"
