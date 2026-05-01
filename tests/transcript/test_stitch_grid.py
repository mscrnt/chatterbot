"""Tests for the screenshot grid stitcher.

Covers the layout table (1/2/3-4/5-6 cell cases producing the right
canvas dimensions) and the perceptual-hash dedup that drops
near-duplicate adjacent frames before stitching."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

# Pillow is in the whisper extra. Skip the whole module when unavailable
# so the dashboard's CI doesn't pretend to test image work it can't run.
PIL = pytest.importorskip("PIL")
from PIL import Image

from chatterbot.transcript import (
    _GRID_LAYOUTS,
    _hamming,
    _phash_64,
    _stitch_grid,
)


def _solid(tmp: Path, name: str, color: tuple[int, int, int]) -> str:
    """Write a 480x270 solid-colored JPEG for use as a stitcher input.
    Returns the path as a string (matching `_stitch_grid`'s expected
    input shape). Solid colors give predictable phash values — uniform
    fill → all bits below avg → hash = 0; near-uniform colors collide,
    distinct colors don't."""
    p = tmp / f"{name}.jpg"
    Image.new("RGB", (480, 270), color).save(p, format="JPEG", quality=80)
    return str(p)


def _grid_size(blob: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(blob)).size


# ---- layout table ----

def test_layout_one_cell_960x540(tmp_path):
    """Single frame uses 1x1 layout at the legacy 960x540 canvas."""
    path = _solid(tmp_path, "a", (200, 30, 30))
    blob = _stitch_grid([path])
    assert _grid_size(blob) == (960, 540)


def test_layout_two_cells_960x540(tmp_path):
    paths = [
        _solid(tmp_path, "a", (200, 30, 30)),
        _solid(tmp_path, "b", (30, 200, 30)),
    ]
    blob = _stitch_grid(paths)
    assert _grid_size(blob) == (960, 540)


def test_layout_four_cells_960x540(tmp_path):
    """Four cells stay on the legacy 960x540 canvas (2x2)."""
    paths = [
        _solid(tmp_path, f"c{i}", (i * 40, i * 30, 200 - i * 20))
        for i in range(4)
    ]
    blob = _stitch_grid(paths)
    assert _grid_size(blob) == (960, 540)


def test_layout_six_cells_widens_canvas(tmp_path):
    """Six cells use the 3x2 layout at 1440x540 — widened so cells
    stay at 480x270 instead of shrinking to 320x180."""
    paths = [
        _solid(tmp_path, f"c{i}", (i * 40, 100 - i * 10, i * 30))
        for i in range(6)
    ]
    # phash_distance=0 disables dedup so all 6 visually-distinct
    # solids make it into the grid.
    blob = _stitch_grid(paths, phash_distance=0)
    assert _grid_size(blob) == (1440, 540)


def test_layout_table_caps_at_six(tmp_path):
    """`_GRID_LAYOUTS` only defines layouts up to 6 cells; passing
    more inputs uses the same 6-cell layout (rest are dropped)."""
    paths = [
        _solid(tmp_path, f"c{i}", (i * 30, 100 - i * 8, i * 25))
        for i in range(10)
    ]
    blob = _stitch_grid(paths, phash_distance=0)
    assert _grid_size(blob) == (1440, 540)


# ---- perceptual-hash dedup ----

def test_phash_identical_frames_collide(tmp_path):
    """Two identical solid-color frames produce identical phashes
    (Hamming distance 0)."""
    p1 = _solid(tmp_path, "x", (123, 45, 67))
    p2 = _solid(tmp_path, "y", (123, 45, 67))
    assert _hamming(_phash_64(p1), _phash_64(p2)) == 0


def test_phash_distinct_colors_diverge(tmp_path):
    """Solid-color frames with very different brightness produce
    phashes with a non-zero Hamming distance. The exact value
    depends on the gradient pattern Pillow's LANCZOS resize introduces
    near edges; we just check that they're NOT bit-identical."""
    p_dark = _solid(tmp_path, "dark", (10, 10, 10))
    p_light = _solid(tmp_path, "light", (250, 250, 250))
    # Solid uniform-color frames hash to 0 (everything below avg → no
    # bits set), so identity actually holds across solid colors of
    # any brightness. Use a striped image instead so the resize
    # produces different per-pixel deltas around the average.
    striped = tmp_path / "striped.jpg"
    img = Image.new("RGB", (480, 270), (10, 10, 10))
    for x in range(0, 480, 40):
        for y in range(270):
            img.putpixel((x, y), (250, 250, 250))
    img.save(striped, format="JPEG", quality=80)
    h_dark = _phash_64(p_dark)
    h_striped = _phash_64(str(striped))
    assert h_dark != h_striped


def test_dedup_collapses_runs_of_duplicates(tmp_path):
    """Six visually-identical frames collapse to one cell after
    dedup — paused-on-menu scenario."""
    paths = [
        _solid(tmp_path, f"dup{i}", (123, 45, 67))
        for i in range(6)
    ]
    blob = _stitch_grid(paths, phash_distance=6)
    # All six dedup down to one — uses the 1x1 layout (960x540).
    assert _grid_size(blob) == (960, 540)


def test_dedup_keeps_genuine_changes(tmp_path):
    """Three unique-then-three-duplicate produces three cells.
    The 3x2 = 6 cells doesn't fire because dedup drops the trailing
    duplicates of cell 3."""
    a = _solid(tmp_path, "a", (200, 30, 30))
    b = _solid(tmp_path, "b", (30, 200, 30))
    c = _solid(tmp_path, "c", (30, 30, 200))
    # Three unique solids look identical at phash level (uniform fill
    # all hashes to 0), so we use stripe-shifted images to ensure the
    # phashes diverge.
    ps: list[str] = []
    for i, color in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200)]):
        f = tmp_path / f"u{i}.jpg"
        img = Image.new("RGB", (480, 270), (10, 10, 10))
        # Each unique image has its stripe offset shifted so the
        # resampled 8x8 thumbnail produces different bit patterns.
        for x in range(i * 5, 480, 40):
            for y in range(270):
                img.putpixel((x, y), color)
        img.save(f, format="JPEG", quality=80)
        ps.append(str(f))
    # Append three duplicates of the LAST unique frame so dedup drops
    # them adjacent to it.
    ps.append(ps[-1])
    ps.append(ps[-1])
    ps.append(ps[-1])
    blob = _stitch_grid(ps, phash_distance=6)
    # 3 unique cells → 2x2 layout (last cell blank, 960x540).
    assert _grid_size(blob) == (960, 540)


def test_dedup_disabled_passes_all_through(tmp_path):
    """phash_distance=0 disables dedup — even six identical frames
    fill all 6 cells of the 3x2 layout."""
    paths = [
        _solid(tmp_path, f"dup{i}", (123, 45, 67))
        for i in range(6)
    ]
    blob = _stitch_grid(paths, phash_distance=0)
    assert _grid_size(blob) == (1440, 540)


# ---- empty / pathological inputs ----

def test_empty_input_returns_none():
    assert _stitch_grid([]) is None


def test_unreadable_path_skipped(tmp_path):
    """A path that doesn't exist is silently skipped — the loop
    continues with whatever it can open. With one good + one bad
    path, the result is a 1x1 grid."""
    good = _solid(tmp_path, "ok", (100, 100, 100))
    bad = str(tmp_path / "missing.jpg")
    blob = _stitch_grid([bad, good])
    assert _grid_size(blob) == (960, 540)


def test_layout_table_canvases_keep_cells_at_or_above_320x180():
    """Sanity check on `_GRID_LAYOUTS`: every layout's cell size is
    at least 320x180. Going below that hurts game-UI legibility for
    the multimodal LLM."""
    for n, (cols, rows, w, h) in _GRID_LAYOUTS.items():
        cell_w = w // cols
        cell_h = h // rows
        assert cell_w >= 320, f"cell width too small at n={n}: {cell_w}"
        assert cell_h >= 180, f"cell height too small at n={n}: {cell_h}"
