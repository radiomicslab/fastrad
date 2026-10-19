"""
Runtime benchmark (Tables 4, 5 and the multi-threading comparison).

End-to-end figures are ONE full extraction call per timing (all feature
classes in a single PyRadiomics execute() / fastrad extract()), never a sum of
per-class calls: every execute() repeats PyRadiomics' fixed preprocessing, so
summing per-class calls counts that overhead once per class.

Per-class times are still reported (each class timed in its own call, fixed
overhead included) for the per-class breakdown only, never summed.

Single-thread runs: OMP/MKL/OpenBLAS threads are pinned to 1 before any
numerical library is imported. The PyRadiomics multi-thread run executes in a
subprocess with OMP_NUM_THREADS=N, because those variables cannot be changed
after import.
"""

import os
import sys

_PIN = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
if "--pyrad-mt-worker" not in sys.argv:
    for _v in _PIN:
        os.environ[_v] = "1"

import csv
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor, setVerbosity

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage

warnings.filterwarnings("ignore")
setVerbosity(40)
try:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
except Exception:
    pass

import argparse as _argparse

_ap = _argparse.ArgumentParser(description=__doc__)
_ap.add_argument(
    "--image", help="image file (default: tests/fixtures/tcia/lung1_image.nrrd)"
)
_ap.add_argument(
    "--mask", help="mask file (default: tests/fixtures/tcia/lung1_label.nrrd)"
)
_ap.add_argument(
    "--synthetic",
    action="store_true",
    help="use a synthetic LUNG1-sized case (smoke test)",
)
_ap.add_argument("--reps", type=int, default=20)
_ap.add_argument("--scaling-reps", type=int, default=10)
_ap.add_argument("--warmup", type=int, default=3)
_ARGS, _ = _ap.parse_known_args()

N_REPETITIONS = _ARGS.reps
N_SCALING_REPETITIONS = _ARGS.scaling_reps
N_WARMUP = _ARGS.warmup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "pyradiomics_config.yaml"
IMG_PATH = (
    Path(_ARGS.image)
    if _ARGS.image
    else PROJECT_ROOT / "tests" / "fixtures" / "tcia" / "lung1_image.nrrd"
)
MASK_PATH = (
    Path(_ARGS.mask)
    if _ARGS.mask
    else PROJECT_ROOT / "tests" / "fixtures" / "tcia" / "lung1_label.nrrd"
)
if _ARGS.synthetic:
    sys.path.insert(0, str(PROJECT_ROOT))
    import tempfile

    from tests.synthetic import make_case, to_sitk

    _tmp = Path(tempfile.mkdtemp())
    _si, _sm = to_sitk(*make_case())
    IMG_PATH, MASK_PATH = _tmp / "synthetic_image.nrrd", _tmp / "synthetic_label.nrrd"
    sitk.WriteImage(_si, str(IMG_PATH))
    sitk.WriteImage(_sm, str(MASK_PATH))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def create_spherical_mask(image_tensor, radius_mm, spacing):
    D, H, W = image_tensor.shape
    cz, cy, cx = D // 2, H // 2, W // 2
    rz = max(1, int(radius_mm / spacing[0]))
    ry = max(1, int(radius_mm / spacing[1]))
    rx = max(1, int(radius_mm / spacing[2]))
    z, y, x = torch.meshgrid(
        torch.arange(D, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    d2 = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    m = torch.zeros_like(image_tensor, dtype=torch.float32)
    m[d2 <= 1.0] = 1.0
    return m


def stats(times):
    a = np.asarray(times)
    return {
        "median": float(np.median(a)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "mean": float(a.mean()),
        "sd": float(a.std()),
    }


def time_calls(fn, reps, sync=None):
    for _ in range(N_WARMUP):
        fn()
        if sync:
            sync()
    out, times = None, []
    for _ in range(reps):
        if sync:
            sync()
        t0 = time.perf_counter()
        out = fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return times, out


def n_features(d):
    return sum(1 for k in d if not str(k).startswith("diagnostics"))


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --- PyRadiomics ----------------------------------------------------------


def pyrad_extractor(cls=None):
    ex = featureextractor.RadiomicsFeatureExtractor(str(CONFIG_PATH))
    if cls is not None:
        ex.disableAllFeatures()
        ex.enableFeatureClassByName(cls)
    return ex


def time_pyrad_full(sitk_img, sitk_mask, reps=N_REPETITIONS):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    ex = pyrad_extractor()
    return time_calls(lambda: ex.execute(sitk_img, sitk_mask), reps)


def time_pyrad_class(sitk_img, sitk_mask, cls, reps=N_REPETITIONS):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    ex = pyrad_extractor(cls)
    return time_calls(lambda: ex.execute(sitk_img, sitk_mask), reps)[0]


# --- fastrad --------------------------------------------------------------


def fastrad_extractor(device, cls=None):
    if cls is None:
        s = FeatureSettings.from_yaml(CONFIG_PATH, device=device)
    else:
        s = FeatureSettings.from_yaml(CONFIG_PATH, feature_classes=[cls], device=device)
    return FeatureExtractor(s)


def time_fastrad_full(img, mask, use_gpu, reps=N_REPETITIONS, threads=1):
    torch.set_num_threads(threads)
    ex = fastrad_extractor("cuda" if use_gpu else "cpu")
    return time_calls(
        lambda: ex.extract(img, mask), reps, sync=cuda_sync if use_gpu else None
    )


def time_fastrad_class(img, mask, cls, use_gpu, reps=N_REPETITIONS):
    torch.set_num_threads(1)
    ex = fastrad_extractor("cuda" if use_gpu else "cpu", cls)
    return time_calls(
        lambda: ex.extract(img, mask), reps, sync=cuda_sync if use_gpu else None
    )[0]


# --- PyRadiomics multi-thread run in a subprocess --------------------------


def _pyrad_mt_worker(threads, reps):
    """Runs inside the subprocess (env already set). Prints JSON list of times."""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    img, msk = sitk.ReadImage(str(IMG_PATH)), sitk.ReadImage(str(MASK_PATH))
    ex = pyrad_extractor()
    times, _ = time_calls(lambda: ex.execute(img, msk), reps)
    print("JSON:" + json.dumps(times))


def time_pyrad_full_multithread(threads, reps):
    env = dict(os.environ)
    for v in _PIN:
        env[v] = str(threads)
    r = subprocess.run(
        [
            sys.executable,
            __file__,
            "--pyrad-mt-worker",
            str(threads),
            str(reps),
            "--image",
            str(IMG_PATH),
            "--mask",
            str(MASK_PATH),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(l for l in r.stdout.splitlines() if l.startswith("JSON:"))
    return json.loads(line[5:])


# --------------------------------------------------------------------------
# report sections
# --------------------------------------------------------------------------


def fmt(st):
    return f"{st['median']:.4f} [{st['iqr']:.4f}]"


def end_to_end_section(title, sitk_img, sitk_mask, fr_img, fr_mask):
    pr_t, pr_out = time_pyrad_full(sitk_img, sitk_mask)
    fc_t, fc_out = time_fastrad_full(fr_img, fr_mask, use_gpu=False)
    rows = [
        ("PyRadiomics, 1 thread", stats(pr_t), n_features(pr_out)),
        ("fastrad CPU, 1 thread", stats(fc_t), n_features(fc_out)),
    ]
    if torch.cuda.is_available():
        fg_t, fg_out = time_fastrad_full(fr_img, fr_mask, use_gpu=True)
        rows.append(("fastrad GPU", stats(fg_t), n_features(fg_out)))
    base = rows[0][1]["median"]
    md = [
        f"#### {title}: end-to-end, one full extraction call\n",
        "| Configuration | Median [IQR] (s) | Features returned | Speedup vs PyRadiomics |",
        "|---|---|---|---|",
    ]
    for name, st, n in rows:
        md.append(f"| {name} | {fmt(st)} | {n} | {base / st['median']:.2f}x |")
    return "\n".join(md) + "\n", {name: st for name, st, _ in rows}


def per_class_section(title, sitk_img, sitk_mask, fr_img, fr_mask, classes):
    md = [
        (
            f"#### {title}: per class (each class timed in its own call; fixed per-call "
            f"overhead included, so rows must NOT be summed)\n"
        ),
        "| Feature Class | PyRadiomics Median [IQR] (s) | fastrad CPU Median [IQR] (s) | CPU Speedup | fastrad GPU Median [IQR] (s) | GPU Speedup |",
        "|---|---|---|---|---|---|",
    ]
    for cls in classes:
        p = stats(time_pyrad_class(sitk_img, sitk_mask, cls))
        c = stats(time_fastrad_class(fr_img, fr_mask, cls, use_gpu=False))
        if torch.cuda.is_available():
            g = stats(time_fastrad_class(fr_img, fr_mask, cls, use_gpu=True))
            g_str, g_spd = fmt(g), f"{p['median'] / g['median']:.2f}x"
        else:
            g_str, g_spd = "N/A", "N/A"
        md.append(
            f"| {cls} | {fmt(p)} | {fmt(c)} | {p['median'] / c['median']:.2f}x | {g_str} | {g_spd} |"
        )
    return "\n".join(md) + "\n"


def run():
    print(
        f"Runtime benchmark: {N_REPETITIONS} reps ({N_SCALING_REPETITIONS} for scaling), "
        f"{N_WARMUP} warm-up, single full-extraction calls for all totals..."
    )
    if not (IMG_PATH.exists() and MASK_PATH.exists()):
        return "Error: TCIA nrrd files not found."
    out_dir = PROJECT_ROOT / "benchmarks" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_sitk = sitk.ReadImage(str(IMG_PATH))
    mask_sitk = sitk.ReadImage(str(MASK_PATH))
    img_t = torch.from_numpy(sitk.GetArrayFromImage(img_sitk)).float()
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    mask_t = torch.from_numpy(mask_arr).float()
    spacing = img_sitk.GetSpacing()[::-1]  # (z, y, x)
    fr_img = MedicalImage(img_t, spacing=spacing)
    fr_mask = Mask(mask_t, spacing=spacing)
    classes = FeatureSettings.from_yaml(CONFIG_PATH).feature_classes

    n_clin = int((mask_arr > 0).sum())
    labels = sorted(int(v) for v in np.unique(mask_arr))
    synth_t = create_spherical_mask(img_t, 15.0, spacing)
    synth_sitk = sitk.GetImageFromArray(synth_t.numpy().astype(np.uint8))
    synth_sitk.CopyInformation(img_sitk)
    fr_synth = Mask(synth_t, spacing=spacing)
    n_synth = int(synth_t.sum().item())

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    md = [
        "## Section 3: Runtime Performance\n",
        "> - Totals and speedups: ONE full extraction call per timing (all classes), never summed per-class calls.",
        (
            f"> - {N_REPETITIONS} repetitions after {N_WARMUP} warm-up; median [IQR]; `time.perf_counter()`; "
            "`torch.cuda.synchronize()` before and after every timed GPU call."
        ),
        "> - Image and mask loaded once before timing (file I/O excluded).",
        "> - Single-thread: OMP/MKL/OpenBLAS pinned to 1, `torch.set_num_threads(1)`, SimpleITK 1 thread.",
        (
            f"> - Image: {IMG_PATH.name}, shape {tuple(mask_arr.shape)} (z,y,x), spacing "
            f"{tuple(round(s, 4) for s in spacing)} mm; clinical mask {n_clin} voxels, label values {labels}."
        ),
        (
            f"> - Synthetic 15 mm sphere: {n_synth} voxels. GPU: {gpu_name}. "
            f"PyTorch {torch.__version__}.\n"
        ),
    ]

    print("  -> end-to-end, synthetic 15 mm sphere...")
    s_md, _ = end_to_end_section(
        f"Synthetic 15 mm sphere ({n_synth} voxels)",
        img_sitk,
        synth_sitk,
        fr_img,
        fr_synth,
    )
    print("  -> end-to-end, clinical LUNG1-001...")
    c_md, clin_rows = end_to_end_section(
        f"Clinical case {MASK_PATH.name} ({n_clin} voxels)",
        img_sitk,
        mask_sitk,
        fr_img,
        fr_mask,
    )
    md += ["### 3.1 End-to-end extraction time\n", s_md, c_md]

    print("  -> per-class breakdown...")
    md += [
        "### 3.2 Per-class breakdown\n",
        per_class_section(
            f"Synthetic 15 mm sphere ({n_synth} voxels)",
            img_sitk,
            synth_sitk,
            fr_img,
            fr_synth,
            classes,
        ),
        per_class_section(
            f"Clinical case {MASK_PATH.name} ({n_clin} voxels)",
            img_sitk,
            mask_sitk,
            fr_img,
            fr_mask,
            classes,
        ),
    ]

    print("  -> ROI size scaling sweep (single full calls)...")
    md += [
        "### 3.3 ROI size scaling (one full extraction call per timing)\n",
        "| Radius (mm) | Voxel Count | PyRadiomics Median [IQR] (s) | fastrad CPU Median [IQR] (s) | CPU Speedup | fastrad GPU Median [IQR] (s) | GPU Speedup |",
        "|---|---|---|---|---|---|---|",
    ]
    scaling_records = []
    for r in [5, 10, 15, 20, 25, 30]:
        t_mask = create_spherical_mask(img_t, float(r), spacing)
        n_vox = int(t_mask.sum().item())
        s_mask = sitk.GetImageFromArray(t_mask.numpy().astype(np.uint8))
        s_mask.CopyInformation(img_sitk)
        f_mask = Mask(t_mask, spacing=spacing)

        p = stats(time_pyrad_full(img_sitk, s_mask, N_SCALING_REPETITIONS)[0])
        c = stats(time_fastrad_full(fr_img, f_mask, False, N_SCALING_REPETITIONS)[0])
        if torch.cuda.is_available():
            g = stats(time_fastrad_full(fr_img, f_mask, True, N_SCALING_REPETITIONS)[0])
            g_str, g_spd, g_med = (
                fmt(g),
                f"{p['median'] / g['median']:.2f}x",
                g["median"],
            )
        else:
            g_str, g_spd, g_med = "N/A", "N/A", float("nan")
        md.append(
            f"| {r} | {n_vox} | {fmt(p)} | {fmt(c)} | {p['median'] / c['median']:.2f}x | {g_str} | {g_spd} |"
        )
        scaling_records.append(
            {
                "radius_mm": r,
                "voxel_count": n_vox,
                "pyradiomics_median": p["median"],
                "pyradiomics_iqr": p["iqr"],
                "fastrad_cpu_median": c["median"],
                "fastrad_cpu_iqr": c["iqr"],
                "fastrad_gpu_median": g_med,
            }
        )
    md.append("")

    print("  -> multi-threading comparison (PyRadiomics N threads in a subprocess)...")
    max_t = os.cpu_count() or 4
    pr_mt = stats(time_pyrad_full_multithread(max_t, N_SCALING_REPETITIONS))
    pr_st = clin_rows["PyRadiomics, 1 thread"]
    fc_st = clin_rows["fastrad CPU, 1 thread"]
    md += [
        "### 3.4 Multi-threading (clinical case, one full extraction call)\n",
        f"- PyRadiomics, 1 thread: {fmt(pr_st)} s",
        (
            f"- PyRadiomics, {max_t} threads: {fmt(pr_mt)} s "
            f"(threading speedup {pr_st['median'] / pr_mt['median']:.2f}x)"
        ),
        f"- fastrad CPU, 1 thread: {fmt(fc_st)} s",
        (
            f"\n=> fastrad CPU 1 thread vs PyRadiomics {max_t} threads: "
            f"{pr_mt['median'] / fc_st['median']:.2f}x\n"
        ),
    ]

    with open(out_dir / "runtime_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(scaling_records[0].keys()))
        w.writeheader()
        w.writerows(scaling_records)
    report = "\n".join(md)
    (out_dir / "runtime_report.md").write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    if "--pyrad-mt-worker" in sys.argv:
        i = sys.argv.index("--pyrad-mt-worker")
        _ap.add_argument("--pyrad-mt-worker", nargs=2)
        _pyrad_mt_worker(int(sys.argv[i + 1]), int(sys.argv[i + 2]))
    else:
        print(run())
