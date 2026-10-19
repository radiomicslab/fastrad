"""
Synthetic, CT-like test cases for parity testing against PyRadiomics.

They need no downloaded data, so they can run in CI. A case is a chest-like
volume (soft tissue, two lung fields, noise) with an irregular, tumour-like ROI
whose intensities are heterogeneous. Irregular boundaries matter: they
exercise marching-cubes configurations that a sphere or the IBSI phantom
rarely hit.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

LUNG1_SPACING = (3.0, 0.9765625, 0.9765625)  # (z, y, x) mm, as LUNG1-001


def make_case(
    shape: tuple[int, int, int] = (134, 512, 512),
    roi_box: tuple[int, int, int] = (24, 64, 64),
    seed: int = 0,
    spacing: tuple[float, float, float] = LUNG1_SPACING,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Returns (image HU float32 (z,y,x), mask uint8 {0,1}, spacing (z,y,x))."""
    rng = np.random.default_rng(seed)
    D, H, W = shape
    _z, y, x = np.ogrid[:D, :H, :W]

    img = np.full(shape, -1000.0, dtype=np.float32)  # air
    body = ((y - H / 2) / (0.42 * H)) ** 2 + ((x - W / 2) / (0.47 * W)) ** 2 <= 1
    img[np.broadcast_to(body, shape)] = 40.0  # soft tissue
    for cx in (0.32 * W, 0.68 * W):  # lungs
        lung = ((y - 0.48 * H) / (0.28 * H)) ** 2 + ((x - cx) / (0.15 * W)) ** 2 <= 1
        img[np.broadcast_to(lung, shape)] = -850.0
    img += rng.normal(0, 20, shape).astype(np.float32)

    # Irregular tumour: thresholded smooth noise, largest component kept
    bz, by, bx = roi_box
    field = ndimage.gaussian_filter(
        rng.standard_normal(roi_box), sigma=rng.uniform(1.8, 3.0)
    )
    blob = field > np.quantile(field, 0.72)
    lab, n = ndimage.label(blob)
    blob = lab == (np.argmax(ndimage.sum(blob, lab, range(1, n + 1))) + 1)
    blob = ndimage.binary_fill_holes(blob)

    z0 = (D - bz) // 2 + int(rng.integers(-D // 8, D // 8 + 1))
    y0 = int(0.48 * H) - by // 2 + int(rng.integers(-8, 9))
    x0 = int(0.32 * W) - bx // 2 + int(rng.integers(-8, 9))
    mask = np.zeros(shape, dtype=np.uint8)
    mask[z0 : z0 + bz, y0 : y0 + by, x0 : x0 + bx] = blob

    # Heterogeneous tumour intensities (necrotic core to enhancing rim)
    tumour_hu = ndimage.gaussian_filter(rng.standard_normal(roi_box), 2.0)
    tumour_hu = -20 + 110 * (tumour_hu - tumour_hu.min()) / (np.ptp(tumour_hu) + 1e-9)
    tumour_hu += rng.normal(0, 15, roi_box)
    region = img[z0 : z0 + bz, y0 : y0 + by, x0 : x0 + bx]
    region[blob] = tumour_hu[blob]

    img = np.round(img)  # CT values are integers
    return img.astype(np.float32), mask, spacing


def to_sitk(img: np.ndarray, mask: np.ndarray, spacing):
    import SimpleITK as sitk

    si, sm = sitk.GetImageFromArray(img), sitk.GetImageFromArray(mask)
    for s in (si, sm):
        s.SetSpacing(tuple(float(v) for v in spacing[::-1]))  # SimpleITK is (x, y, z)
    return si, sm


def to_fastrad(img: np.ndarray, mask: np.ndarray, spacing):
    import torch

    from fastrad import Mask, MedicalImage

    sp = tuple(float(v) for v in spacing)
    return (
        MedicalImage(torch.from_numpy(img), spacing=sp),
        Mask(torch.from_numpy(mask.astype(np.float32)), spacing=sp),
    )


# ---- fastrad key -> PyRadiomics key -------------------------------------
_SPECIAL = {
    ("firstorder", "10th_percentile"): "10Percentile",
    ("firstorder", "90th_percentile"): "90Percentile",
}


def pyrad_key(fastrad_key: str) -> str | None:
    """'glcm:joint_average' -> 'original_glcm_JointAverage' (case-insensitive match used)."""
    cls, name = fastrad_key.split(":")
    if (cls, name) == ("firstorder", "standard_deviation"):
        return None  # fastrad-only convenience feature
    name = _SPECIAL.get((cls, name), name)
    return f"original_{cls}_{name.replace('_', '').lower()}"


def compare(fr: dict, pr: dict) -> list[tuple[str, float, float, float]]:
    """Returns (key, fastrad, pyradiomics, abs diff) for every comparable feature."""
    pr_lower = {
        k.lower(): float(v) for k, v in pr.items() if not k.startswith("diagnostics")
    }
    rows = []
    for k, v in fr.items():
        pk = pyrad_key(k)
        if pk is None:
            continue
        if pk not in pr_lower:
            continue  # e.g. PyRadiomics' deprecated shape features, disabled by default
        p = pr_lower[pk]
        rows.append((k, float(v), p, abs(float(v) - p)))
    return rows
