"""The batched (GPU) and convex-hull (CPU) diameter paths must agree exactly."""

import numpy as np
import pytest
import torch
from scipy import ndimage

from fastrad.features import shape_utils as su


@pytest.mark.parametrize("seed", range(8))
def test_batched_diameters_match_hull_path(seed):
    rng = np.random.default_rng(seed)
    shape = tuple(int(x) for x in rng.integers(4, 36, 3))
    field = ndimage.gaussian_filter(rng.standard_normal(shape), rng.uniform(0.8, 3))
    mask = field > np.quantile(field, rng.uniform(0.3, 0.9))
    if mask.sum() < 2:
        pytest.skip("degenerate mask")
    mm = torch.nn.functional.pad(
        torch.from_numpy(mask.astype(np.float32)), (1, 1, 1, 1, 1, 1)
    )
    spacing = tuple(float(x) for x in rng.uniform(0.4, 5, 3))

    captured = {}
    original = su._max_2d_diameters

    def capture(points):
        captured["P"] = points
        return original(points)

    su._max_2d_diameters = capture
    try:
        ref = su.calculate_mesh_features(mm, spacing)
    finally:
        su._max_2d_diameters = original

    P = torch.from_numpy(captured["P"])
    H = torch.round(P / torch.tensor(spacing, dtype=torch.float64) * 2).to(torch.int64)
    got = su._max_2d_diameters_t(H, P)
    expected = [
        ref["Maximum2DDiameterSlice"],
        ref["Maximum2DDiameterColumn"],
        ref["Maximum2DDiameterRow"],
    ]
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)
