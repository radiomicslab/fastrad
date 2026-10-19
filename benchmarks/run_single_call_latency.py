"""
Full single-call extraction latency on ONE image/mask, all feature classes in
one call, for: PyRadiomics (1 thread), fastrad CPU (1 thread), fastrad GPU.
This is the like-for-like end-to-end comparison (no per-class summing).

usage (from repo root):
  uv run python benchmarks/run_single_call_latency.py IMAGE.nrrd MASK.nrrd [pyradiomics_config.yaml] [--reps 20]
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "1"  # must be set before numpy/torch/radiomics import

import argparse
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("mask")
ap.add_argument("config", nargs="?", default="pyradiomics_config.yaml")
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
a = ap.parse_args()

try:
    from loguru import logger

    logger.remove()
except Exception:
    pass


def bench(fn, sync=None):
    for _ in range(a.warmup):
        fn()
        if sync:
            sync()
    t = []
    for _ in range(a.reps):
        if sync:
            sync()
        t0 = time.perf_counter()
        out = fn()
        if sync:
            sync()
        t.append(time.perf_counter() - t0)
    t.sort()
    q1, q3 = t[len(t) // 4], t[(3 * len(t)) // 4]
    return statistics.median(t), q3 - q1, out


def n_features(d):
    return sum(1 for k in d if not str(k).startswith("diagnostics"))


# --- PyRadiomics: one execute() with every class enabled in the config ---
from radiomics import featureextractor, setVerbosity

setVerbosity(40)
import SimpleITK as sitk

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(
    1
)  # SimpleITK has its own thread pool
pr = featureextractor.RadiomicsFeatureExtractor(a.config)
pr_img, pr_msk = (
    sitk.ReadImage(a.image),
    sitk.ReadImage(a.mask),
)  # load once, like fastrad
pr_med, pr_iqr, pr_out = bench(lambda: pr.execute(pr_img, pr_msk))

# --- fastrad (same loader/settings as the cohort benchmark) ---
import torch

torch.set_num_threads(1)
from run_cohort_throughput import load_fastrad_case, make_fastrad_extractor

img, msk = load_fastrad_case(a.image, a.mask)

ex_cpu = make_fastrad_extractor("cpu", a.config)
cpu_med, cpu_iqr, cpu_out = bench(lambda: ex_cpu.extract(img, msk))

rows = [
    ("PyRadiomics, 1 thread", pr_med, pr_iqr, n_features(pr_out)),
    ("fastrad CPU, 1 thread", cpu_med, cpu_iqr, n_features(cpu_out)),
]

if torch.cuda.is_available():
    ex_gpu = make_fastrad_extractor("cuda", a.config)
    gpu_med, gpu_iqr, gpu_out = bench(
        lambda: ex_gpu.extract(img, msk), sync=torch.cuda.synchronize
    )
    rows.append(("fastrad GPU", gpu_med, gpu_iqr, n_features(gpu_out)))

print(
    f"\nFull single-call extraction, {a.reps} reps after {a.warmup} warm-up "
    f"(image/mask loaded once for both tools; file I/O excluded)"
)
print(
    f"{'config':24s} {'median s':>10s} {'IQR s':>8s} {'features':>9s} {'speedup vs PyRad':>17s}"
)
for name, med, iqr, n in rows:
    print(f"{name:24s} {med:10.4f} {iqr:8.4f} {n:9d} {pr_med / med:16.2f}x")
print(
    "\nCheck: the 'features' column should be the same 107 (+ fastrad's extra standard_deviation) for all rows."
)
