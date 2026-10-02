"""Flat stem-pairing helpers for M2-03 (Boss-approved contract)."""

import os
import os.path as osp

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
TEXT_EXTS = frozenset({".txt"})
SIDECAR_EXTS = frozenset({".json", ".yaml", ".yml"})


def _visible_names(folder):
    return [n for n in os.listdir(folder) if not n.startswith(".")]


def _stem(filename):
    return osp.splitext(filename)[0]


def collect_flat_dir(folder, allowed_exts):
    """Validate a flat dir; return {stem: full path} sorted by stem."""
    names = _visible_names(folder)
    if not names:
        raise FileNotFoundError(f"No visible files in {folder}")
    mapping = {}
    for name in names:
        full = osp.join(folder, name)
        if not osp.isfile(full):
            raise ValueError(f"Unsupported entry (not a flat file): {full}")
        ext = osp.splitext(name)[1].lower()
        if ext in SIDECAR_EXTS:
            continue
        if ext not in allowed_exts:
            raise ValueError(f"Unsupported file: {full}")
        if not os.access(full, os.R_OK):
            raise OSError(f"Unreadable file: {full}")
        stem = _stem(name)
        if stem in mapping:
            raise ValueError(f"Duplicate stem {stem!r} in {folder}")
        mapping[stem] = full
    if not mapping:
        exts_str = sorted(allowed_exts)
        raise ValueError(
            f"No supported files matching extensions {exts_str} "
            f"found in {folder}"
        )
    return mapping


def pair_dirs(real_dir, fake_dir, real_exts, fake_exts):
    """Stem-match two flat dirs; raise on any mismatch."""
    real_map = collect_flat_dir(real_dir, real_exts)
    fake_map = collect_flat_dir(fake_dir, fake_exts)
    real_stems = set(real_map)
    fake_stems = set(fake_map)
    if real_stems != fake_stems:
        missing = sorted(real_stems ^ fake_stems)
        raise FileNotFoundError(f"Stem mismatch: {missing}")
    stems = sorted(real_stems)
    return [(real_map[s], fake_map[s]) for s in stems]


def allowed_exts_for_flag(flag):
    if flag == "img":
        return IMAGE_EXTS
    if flag == "txt":
        return TEXT_EXTS
    raise TypeError(f"Got unexpected modality: {flag}")
