"""
Wall-clock breakdown of ONE extraction, step by step, on CPU or GPU.

Unlike torch.profiler, this also counts work outside PyTorch (CuPy labelling,
NumPy/SciPy shape diameters, Python overhead), because every step is timed with
a device synchronisation before and after.

usage:
  uv run python benchmarks/profile_extraction.py IMAGE MASK [--device cuda] [--reps 10]
"""

import argparse
import statistics
import sys
import time
import warnings
from pathlib import Path

import SimpleITK as sitk
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage
from fastrad.extractor import _FEATURE_MAP, _crop_to_roi
from fastrad.image import binning_cache, get_binned_image
from fastrad.logger import logger

logger.remove()
ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("mask")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument(
    "--config",
    default=str(Path(__file__).resolve().parent.parent / "pyradiomics_config.yaml"),
)
a = ap.parse_args()
torch.set_num_threads(1)

si, sm = sitk.ReadImage(a.image), sitk.ReadImage(a.mask)
sp = si.GetSpacing()[::-1]
img = MedicalImage(torch.from_numpy(sitk.GetArrayFromImage(si)).float(), sp)
msk = Mask(torch.from_numpy(sitk.GetArrayFromImage(sm)).float(), sp)
dev = a.device


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def tm(fn):
    fn()
    sync()
    ts = []
    for _ in range(a.reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


settings = FeatureSettings.from_yaml(a.config, device=dev)
settings.spacing = sp
full = FeatureExtractor(settings)
rows = [("FULL extract() call", tm(lambda: full.extract(img, msk)))]
rows.append(("  crop on host", tm(lambda: _crop_to_roi(img.tensor, msk.tensor))))
ci, cm, n = _crop_to_roi(img.tensor, msk.tensor)
rows.append(("  host->device transfer", tm(lambda: (ci.to(dev), cm.to(dev)))))
ci, cm = ci.to(dev), cm.to(dev)
rows.append(
    (
        "  grey-level binning (x1)",
        tm(lambda: get_binned_image(ci, cm, settings.bin_width)),
    )
)
for cls in settings.feature_classes:

    def run(c=cls):
        with binning_cache():
            return _FEATURE_MAP[c](image_tensor=ci, mask_tensor=cm, settings=settings)

    rows.append((f"  {cls}", tm(run)))

from fastrad.features import glszm

print(
    f"\nDevice: {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}), "
    f"ROI {n} voxels, crop {tuple(ci.shape)}, GLSZM labeller: {glszm.LAST_LABEL_BACKEND}"
)
print(f"{'step':32s} {'median ms':>10s}")
for name, ms in rows:
    print(f"{name:32s} {ms:10.2f}")
print(
    "\nClass rows include their own binning; they are timed separately, so they do not add up exactly to the full call."
)
