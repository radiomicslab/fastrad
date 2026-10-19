"""
Feature-level parity with PyRadiomics on synthetic CT-like cases.

Runs without downloaded data. The ROIs are irregular, so they exercise
marching-cubes configurations (and texture edge cases) that spheres and the
IBSI phantom do not. Tolerance: 1e-6 relative (absolute below magnitude 1);
in practice agreement is ~1e-11 or better.
"""

import warnings

import numpy as np
import pytest
import torch

radiomics = pytest.importorskip("radiomics")
from radiomics import featureextractor, setVerbosity

from fastrad import FeatureExtractor, FeatureSettings
from tests.synthetic import compare, make_case, to_fastrad, to_sitk

CONFIG = {
    "setting": {"binWidth": 25.0, "label": 1},
    "imageType": {"Original": {}},
    "featureClass": {
        c: None
        for c in ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
    },
}
TOL = 1e-6


@pytest.fixture(scope="module")
def pyrad():
    setVerbosity(40)
    return featureextractor.RadiomicsFeatureExtractor(CONFIG)


@pytest.mark.parametrize("seed", [0, 2, 3])
def test_all_features_match_pyradiomics(pyrad, seed):
    img, mask, spacing = make_case(
        shape=(48, 128, 128), roi_box=(20, 48, 48), seed=seed
    )
    si, sm = to_sitk(img, mask, spacing)
    fi, fm = to_fastrad(img, mask, spacing)
    expected = pyrad.execute(si, sm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = FeatureExtractor(FeatureSettings.from_dict(CONFIG, device="cpu")).extract(
            fi, fm
        )
    rows = compare(got, expected)
    assert len(rows) == 107
    bad = [(k, a, b) for k, a, b, d in rows if d > TOL * max(1.0, abs(b))]
    assert not bad, f"features outside tolerance: {bad}"


@pytest.mark.parametrize("batch_voxels", [1, 10**9])
def test_glszm_slab_labelling_matches_per_level(monkeypatch, batch_voxels):
    """The batched slab labelling (used on GPU) must find exactly the same zones as per-level labelling."""
    from fastrad.features import glszm

    img, mask, _ = make_case(shape=(40, 96, 96), roi_box=(16, 40, 40), seed=5)
    levels = np.floor(img / 25.0).astype(np.int64)
    levels = np.where(mask > 0, levels - levels[mask > 0].min() + 1, 0)
    t = torch.from_numpy(levels)
    n = int(t.max())
    monkeypatch.setattr(
        glszm, "_LABEL_BATCH_VOXELS", batch_voxels
    )  # 1 => one level per call
    a = glszm._label_zones(t, n, use_slabs=True)
    b = glszm._label_zones(t, n, use_slabs=False)

    def zones(lab):
        ids, counts = torch.unique(lab[lab > 0], return_counts=True)
        level_of = torch.zeros(int(lab.max()) + 1, dtype=torch.int64)
        level_of[lab[lab > 0]] = t[lab > 0]
        return sorted(zip(level_of[ids].tolist(), counts.tolist()))

    assert zones(a) == zones(b)


def test_result_is_independent_of_volume_size():
    """Cropping must make features independent of how much empty volume surrounds the ROI."""
    img, mask, spacing = make_case(shape=(40, 96, 96), roi_box=(16, 40, 40), seed=1)
    big_img = np.pad(img, ((5, 9), (30, 11), (7, 40)), constant_values=-1000)
    big_mask = np.pad(mask, ((5, 9), (30, 11), (7, 40)))
    ex = FeatureExtractor(FeatureSettings(device="cpu"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ex.extract(*to_fastrad(img, mask, spacing))
        b = ex.extract(*to_fastrad(big_img, big_mask, spacing))
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-12), k


def test_default_device_auto_runs():
    img, mask, spacing = make_case(shape=(30, 64, 64), roi_box=(12, 24, 24), seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = FeatureExtractor(FeatureSettings()).extract(
            *to_fastrad(img, mask, spacing)
        )
    assert len(out) > 100 and all(
        np.isfinite(v) for k, v in out.items() if "mcc" not in k
    )
