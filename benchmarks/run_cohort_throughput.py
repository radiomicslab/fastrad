#!/usr/bin/env python3
"""
Cohort-throughput benchmark: PyRadiomics (N CPU processes) vs fastrad on one
GPU, sequential (K=1) and concurrent (K workers sharing the GPU, K chosen
automatically from free VRAM). Addresses Reviewer 2, Question 5.

Usage
-----
    python benchmarks/run_cohort_throughput.py \\
        --manifest cohort.csv \\
        --pyradiomics-config pyradiomics_config.yaml \\
        --cpu-workers 32 --repeats 3 --out results/cohort_throughput

    # Small fixture cohort: repeat each case to get >= 2 tasks per CPU worker,
    # and also run fastrad on CPU with the same process counts as PyRadiomics
    python benchmarks/run_cohort_throughput.py --replicate 15 --fastrad-cpu-workers 16,32

cohort.csv columns:
    case_id, fr_image, fr_mask, pr_image, pr_mask
  fr_*  = inputs for fastrad (e.g. DICOM series dir / RTSTRUCT-derived mask dir)
  pr_*  = inputs for PyRadiomics (any SimpleITK-readable file, e.g. .nrrd)
Both pairs must describe the same image and the same binary mask.

Outputs  <out>.json  (everything) and <out>.csv  (one row per configuration).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "pyradiomics_config.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "benchmarks" / "outputs"


# --------------------------------------------------------------------------
# Adapters -- match fastrad codebase I/O and FeatureExtractor API
# --------------------------------------------------------------------------

def load_fastrad_case(image_path: str, mask_path: str):
    """
    Loads image and mask into fastrad MedicalImage and Mask objects.
    Supports DICOM directories, NIfTI files (.nii, .nii.gz), and NRRD /
    SimpleITK-readable files.
    """
    import numpy as np
    import SimpleITK as sitk
    import torch

    from fastrad import Mask, MedicalImage

    img_p = Path(image_path)
    mask_p = Path(mask_path)

    if img_p.is_dir():
        img = MedicalImage.from_dicom(img_p)
    elif str(img_p).endswith((".nii", ".nii.gz")):
        img = MedicalImage.from_nifti(img_p)
    else:
        sitk_img = sitk.ReadImage(str(img_p))
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        sp = sitk_img.GetSpacing()
        spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
        img = MedicalImage(torch.from_numpy(arr), spacing=spacing)

    if mask_p.is_dir():
        mask = Mask.from_dicom(mask_p)
    elif str(mask_p).endswith((".nii", ".nii.gz")):
        mask = Mask.from_nifti(mask_p)
    else:
        sitk_mask = sitk.ReadImage(str(mask_p))
        arr = sitk.GetArrayFromImage(sitk_mask).astype(np.float32)
        sp = sitk_mask.GetSpacing()
        spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
        mask = Mask(torch.from_numpy(arr), spacing=spacing)

    return img, mask


def make_fastrad_extractor(device: str, pyradiomics_config: str):
    """
    Creates a fastrad FeatureExtractor with extraction settings matching the
    parity benchmark and pyradiomics_config.yaml exactly.
    """
    from fastrad import FeatureExtractor, FeatureSettings
    settings = FeatureSettings.from_yaml(pyradiomics_config, device=device)
    return FeatureExtractor(settings)


# --------------------------------------------------------------------------
# Worker-side state (one copy per worker process)
# --------------------------------------------------------------------------
_W: dict = {}


def _to_float_dict(d: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        try:
            val = float(v.item() if hasattr(v, "item") else v)
            out[str(k)] = val
        except (TypeError, ValueError):
            pass  # PyRadiomics diagnostics_* entries etc.
    return out


def _quiet_logs():
    """Silence per-case warnings in workers (they flood the log, not the result)."""
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    except Exception:
        pass


def _single_thread_env():
    """Must run before numpy/scipy/torch are imported in a spawned worker."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def _fastrad_init(device: str, cfg: str, warm_img: str, warm_mask: str):
    _single_thread_env()
    _quiet_logs()
    import torch
    torch.set_num_threads(1)  # one core per worker; avoid CPU oversubscription
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _W["torch"] = torch
    _W["device"] = device
    _W["ex"] = make_fastrad_extractor(device, cfg)
    img, msk = load_fastrad_case(warm_img, warm_mask)
    _W["ex"].extract(img, msk)          # warm-up: CUDA context, kernels, CuPy
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _fastrad_task(case_id: str, img_path: str, mask_path: str):
    torch = _W["torch"]
    t0 = time.perf_counter()
    img, msk = load_fastrad_case(img_path, mask_path)          # I/O included
    feats = _W["ex"].extract(img, msk)
    if _W["device"] == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    return case_id, time.perf_counter() - t0, os.getpid(), _to_float_dict(feats)


def _load_pyrad_input(path: str):
    if os.path.isdir(path):
        import SimpleITK as sitk
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        return reader.Execute()
    return path


def _pyrad_init(cfg: str, warm_img: str, warm_mask: str):
    _single_thread_env()   # set before radiomics/numpy import (spawned process)
    _quiet_logs()
    from radiomics import featureextractor, setVerbosity
    setVerbosity(40)
    _W["ex"] = featureextractor.RadiomicsFeatureExtractor(cfg)
    img_in = _load_pyrad_input(warm_img)
    mask_in = _load_pyrad_input(warm_mask)
    _W["ex"].execute(img_in, mask_in)


def _pyrad_task(case_id: str, img_path: str, mask_path: str):
    t0 = time.perf_counter()
    img_in = _load_pyrad_input(img_path)
    mask_in = _load_pyrad_input(mask_path)
    _W["ex"].execute(img_in, mask_in)
    return case_id, time.perf_counter() - t0, os.getpid(), {}  # features not needed here


def _ready(_):
    time.sleep(0.2)
    return os.getpid()


# --------------------------------------------------------------------------
# Profiling pass: per-case peak VRAM, CUDA context cost, reference features
# --------------------------------------------------------------------------

def _profile(cases: list[dict], cfg: str) -> dict:
    """Profiles each UNIQUE input once; replicated case_ids reuse its result."""
    _quiet_logs()
    import torch
    torch.set_num_threads(1)

    if not torch.cuda.is_available():
        ex = make_fastrad_extractor("cpu", cfg)
        peaks, ref = {}, {}
        for c in cases:
            img, msk = load_fastrad_case(c["fr_image"], c["fr_mask"])
            feats = ex.extract(img, msk)
            peaks[c["case_id"]] = {
                "max_reserved": 0,
                "max_allocated": 0,
            }
            ref[c["case_id"]] = _to_float_dict(feats)
        return {
            "total_bytes": 0,
            "free_bytes_before_any_fastrad": 0,
            "ctx_bytes": 0,
            "peaks": peaks,
            "reference_features": ref,
            "cuda_available": False,
        }

    torch.cuda.init()
    torch.cuda.synchronize()
    free_after_ctx, total = torch.cuda.mem_get_info()
    # Context overhead as seen by the driver = used memory not owned by the allocator.
    ctx = (total - free_after_ctx) - torch.cuda.memory_reserved()
    ex = make_fastrad_extractor("cuda", cfg)
    peaks, ref, seen = {}, {}, {}
    for c in cases:
        key = (c["fr_image"], c["fr_mask"])
        if key not in seen:
            img, msk = load_fastrad_case(c["fr_image"], c["fr_mask"])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            feats = ex.extract(img, msk)
            torch.cuda.synchronize()
            seen[key] = (
                {"max_reserved": torch.cuda.max_memory_reserved(),
                 "max_allocated": torch.cuda.max_memory_allocated()},
                _to_float_dict(feats),
            )
        peaks[c["case_id"]], ref[c["case_id"]] = seen[key]
    return {
        "total_bytes": total,
        "free_bytes_before_any_fastrad": free_after_ctx + ctx,  # what K workers can share
        "ctx_bytes": ctx,
        "peaks": peaks,
        "reference_features": ref,
        "cuda_available": True,
    }


def auto_workers(prof: dict, k_max: int, alpha: float = 0.9) -> int:
    peaks = prof["peaks"].values()
    m_peak = max((p["max_reserved"] for p in peaks), default=0)
    per_worker = prof["ctx_bytes"] + m_peak
    if per_worker <= 0:
        return 1
    k = math.floor(alpha * prof["free_bytes_before_any_fastrad"] / per_worker)
    return max(1, min(k_max, k))


# --------------------------------------------------------------------------
# Timed cohort run
# --------------------------------------------------------------------------

def run_cohort(kind: str, workers: int, cases: list[dict], cfg: str, retries: int = 2):
    """Runs one timed cohort pass; retries if a worker dies (e.g. during CUDA init)."""
    from concurrent.futures.process import BrokenProcessPool
    for attempt in range(retries + 1):
        try:
            return _run_cohort_once(kind, workers, cases, cfg)
        except BrokenProcessPool as e:
            print(f"  ! worker pool broke ({kind}, workers={workers}, attempt {attempt + 1}): {e}")
            time.sleep(10)
    raise RuntimeError(f"{kind} with {workers} workers failed {retries + 1} times")


def _run_cohort_once(kind: str, workers: int, cases: list[dict], cfg: str):
    time.sleep(3)  # let processes from the previous pool release GPU memory
    ctx = mp.get_context("spawn")
    first = cases[0]
    if kind == "pyradiomics":
        init, initargs, task = _pyrad_init, (cfg, first["pr_image"], first["pr_mask"]), _pyrad_task
        args = [(c["case_id"], c["pr_image"], c["pr_mask"]) for c in cases]
    else:
        device = "cuda" if kind == "fastrad_gpu" else "cpu"
        init, initargs, task = _fastrad_init, (device, cfg, first["fr_image"], first["fr_mask"]), _fastrad_task
        args = [(c["case_id"], c["fr_image"], c["fr_mask"]) for c in cases]

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=init, initargs=initargs) as pool:
        # Force every worker to start and finish its warm-up before timing.
        pids = set()
        attempts = 0
        while len(pids) < workers and attempts < 10:
            pids |= set(pool.map(_ready, range(workers * 2)))
            attempts += 1
        t0 = time.perf_counter()
        futs = [pool.submit(task, *a) for a in args]
        results = [f.result() for f in as_completed(futs)]
        wall = time.perf_counter() - t0

    per_case = {cid: dt for cid, dt, _, _ in results}
    feats = {cid: f for cid, _, _, f in results}
    return {
        "wall_s": wall,
        "scans_per_min": 60.0 * len(cases) / wall,
        "per_case_s": per_case,
        "workers_used": len({pid for _, _, pid, _ in results}),
    }, feats


def max_abs_diff(a: dict, b: dict) -> float:
    worst = 0.0
    for cid, feats_a in a.items():
        feats_b = b.get(cid, {})
        for k, v in feats_a.items():
            if k in feats_b:
                d = abs(v - feats_b[k])
                if not math.isnan(d):
                    worst = max(worst, d)
    return worst


def discover_cases(project_root: Path) -> list[dict]:
    """Auto-discovers cases when no manifest is explicitly passed."""
    # 1. Check for cohort.csv in project root or benchmarks/outputs
    for candidate in [
        project_root / "cohort.csv",
        project_root / "benchmarks" / "outputs" / "cohort.csv",
    ]:
        if candidate.exists():
            with open(candidate, newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))

    # 2. Check for multicase directory
    multicase_dir = project_root / "tests" / "fixtures" / "tcia" / "multicase"
    cases = []
    if multicase_dir.exists():
        for case_dir in sorted(multicase_dir.iterdir()):
            img_path = case_dir / "image.nrrd"
            mask_path = case_dir / "label.nrrd"
            if img_path.exists() and mask_path.exists():
                cases.append({
                    "case_id": case_dir.name,
                    "fr_image": str(img_path),
                    "fr_mask": str(mask_path),
                    "pr_image": str(img_path),
                    "pr_mask": str(mask_path),
                })
    if cases:
        return cases

    # 3. Check for single lung1 case for synthetic fallback
    single_img = project_root / "tests" / "fixtures" / "tcia" / "lung1_image.nrrd"
    single_mask = project_root / "tests" / "fixtures" / "tcia" / "lung1_label.nrrd"
    if single_img.exists() and single_mask.exists():
        print("WARNING: No multi-case cohort found. Falling back to 20 synthetic "
              "re-extractions of LUNG1-001 to validate throughput mechanics.")
        return [{
            "case_id": f"LUNG1-001-rep{i}",
            "fr_image": str(single_img),
            "fr_mask": str(single_mask),
            "pr_image": str(single_img),
            "pr_mask": str(single_mask),
        } for i in range(20)]

    return []


def run(
    manifest: str | Path | None = None,
    pyradiomics_config: str | Path | None = None,
    cpu_workers: int = 32,
    k_list: str = "1,2,3,4",
    k_max: int | None = None,
    alpha: float = 0.9,
    repeats: int = 3,
    replicate: int = 1,
    fastrad_cpu_workers: str = "",
    out: str | Path | None = None,
) -> str:
    """
    Executes the cohort throughput benchmark comparing PyRadiomics CPU
    multiprocessing vs fastrad (sequential and concurrent).
    """
    cfg_path = Path(pyradiomics_config or DEFAULT_CONFIG).resolve()
    if not cfg_path.exists():
        msg = f"Error: pyradiomics configuration file not found at {cfg_path}"
        print(msg)
        return msg

    out_prefix = str(out or (DEFAULT_OUT_DIR / "cohort_throughput"))
    max_k = k_max if k_max is not None else (os.cpu_count() or 4)

    if manifest is not None:
        manifest_path = Path(manifest).resolve()
        if not manifest_path.exists():
            msg = f"Error: manifest file not found at {manifest_path}"
            print(msg)
            return msg
        manifest_dir = manifest_path.parent
        with open(manifest_path, newline="", encoding="utf-8") as fh:
            cases = list(csv.DictReader(fh))
        for c in cases:
            for k in ("fr_image", "fr_mask", "pr_image", "pr_mask"):
                val = c.get(k, "")
                if val and not os.path.isabs(val) and not os.path.exists(val):
                    alt = manifest_dir / val
                    if alt.exists():
                        c[k] = str(alt)
    else:
        cases = discover_cases(PROJECT_ROOT)

    if not cases:
        msg = (
            "Error: no cases found (no manifest supplied and fixtures directory "
            "tests/fixtures/tcia/ is empty). Cannot run cohort throughput benchmark."
        )
        print(msg)
        return msg

    if replicate > 1:
        base_cases = cases
        cases = [
            {**c, "case_id": f"{c['case_id']}_r{r:02d}"}
            for r in range(replicate)
            for c in base_cases
        ]
        print(f"Replicated {len(base_cases)} unique cases x{replicate} "
              f"= {len(cases)} tasks")

    print(f"{len(cases)} cases")
    if len(cases) < 2 * cpu_workers:
        print(f"WARNING: fewer than 2 cases per CPU worker ({len(cases)} cases, "
              f"{cpu_workers} workers); the CPU baseline will be dominated by the slowest case.")

    # Profiling runs in its own process so this process never holds a CUDA context.
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as p:
        prof = p.submit(_profile, cases, str(cfg_path)).result()

    has_cuda = prof.get("cuda_available", False)
    if has_cuda:
        k_auto = auto_workers(prof, max_k, alpha)
        gib = 1024 ** 3
        print(f"ctx={prof['ctx_bytes']/gib:.2f} GiB  "
              f"peak={max(v['max_reserved'] for v in prof['peaks'].values())/gib:.2f} GiB  "
              f"free={prof['free_bytes_before_any_fastrad']/gib:.2f} GiB  ->  K_auto={k_auto}")

        configs = [("pyradiomics", cpu_workers)]
        ks = sorted({int(k) for k in k_list.split(",") if int(k) <= k_auto} | {1, k_auto})
        configs += [("fastrad_gpu", k) for k in ks]
        configs += [("fastrad_cpu", int(w)) for w in fastrad_cpu_workers.split(",") if w.strip()]
    else:
        k_auto = 1
        print("NOTICE: CUDA is not available on this system. "
              "Running PyRadiomics CPU baseline vs fastrad CPU sequential baseline.")
        configs = [("pyradiomics", cpu_workers), ("fastrad_cpu", 1)]
        configs += [("fastrad_cpu", int(w)) for w in fastrad_cpu_workers.split(",") if w.strip()]

    out_json = out_prefix + ".json"
    out_csv = out_prefix + ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    prof_out = {k: v for k, v in prof.items() if k != "reference_features"}

    def save(summary):
        pyrad = [s["median_scans_per_min"] for s in summary if s["config"] == "pyradiomics"]
        base = pyrad[0] if pyrad and pyrad[0] > 0 else 1.0
        for s in summary:
            s["ratio_vs_pyradiomics"] = s["median_scans_per_min"] / base
            s["scans_per_min_per_worker"] = s["median_scans_per_min"] / s["workers"]
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump({"n_cases": len(cases), "k_auto": k_auto, "alpha": alpha,
                       "profile": prof_out, "summary": summary}, fh, indent=2)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["config", "workers", "median_scans_per_min", "ratio_vs_pyradiomics",
                         "scans_per_min_per_worker", "max_abs_feature_diff_vs_gpu_reference"])
            for s in summary:
                d = s["max_abs_feature_diff_vs_sequential"]
                wr.writerow([s["config"], s["workers"], f"{s['median_scans_per_min']:.1f}",
                             f"{s['ratio_vs_pyradiomics']:.2f}",
                             f"{s['scans_per_min_per_worker']:.1f}",
                             f"{d:.2e}" if d is not None else "N/A"])

    summary = []
    for kind, w in configs:
        runs, max_diff = [], 0.0
        for r in range(repeats):
            res, feats = run_cohort(kind, w, cases, str(cfg_path))
            if kind.startswith("fastrad"):
                max_diff = max(max_diff, max_abs_diff(prof["reference_features"], feats))
            runs.append(res)
            print(f"{kind:12s} workers={w:3d} rep={r}  {res['scans_per_min']:.1f} scans/min")
        spm = [x["scans_per_min"] for x in runs]
        summary.append({
            "config": kind,
            "workers": w,
            "k_auto": w == k_auto and kind == "fastrad_gpu",
            "median_scans_per_min": statistics.median(spm),
            "min_scans_per_min": min(spm),
            "max_scans_per_min": max(spm),
            # per-task wall time inside a worker (load + extract), median over all tasks/reps
            "median_case_s": statistics.median(
                t for x in runs for t in x["per_case_s"].values()),
            "max_abs_feature_diff_vs_sequential": max_diff if kind.startswith("fastrad") else None,
            "runs": runs,
        })
        save(summary)  # partial results survive a later crash

    save(summary)
    print(f"wrote {out_json} / {out_csv}")

    # Copy to project root if expected by Makefile target $(COHORT_CSV)
    root_csv = PROJECT_ROOT / "cohort_throughput.csv"
    if Path(out_csv).resolve() != root_csv.resolve():
        try:
            import shutil
            shutil.copyfile(out_csv, root_csv)
        except OSError:
            pass

    lines = [f"Cohort Throughput Benchmark ({len(cases)} cases, {repeats} repeats):"]
    for s in summary:
        diff_info = (
            f", max |diff| vs seq = {s['max_abs_feature_diff_vs_sequential']:.2e}"
            if s["max_abs_feature_diff_vs_sequential"] is not None
            else ""
        )
        lines.append(
            f"  - {s['config']} ({s['workers']} workers): "
            f"{s['median_scans_per_min']:.1f} scans/min "
            f"({s['ratio_vs_pyradiomics']:.2f}x vs PyRadiomics, "
            f"{s['scans_per_min_per_worker']:.1f}/worker, "
            f"median case {s['median_case_s']:.2f} s{diff_info})"
        )
    lines.append(f"Output saved to {out_json} and {out_csv}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Cohort-throughput benchmark: PyRadiomics vs fastrad (sequential & concurrent GPU)"
    )
    ap.add_argument("--manifest", default=None, help="Path to cohort manifest CSV")
    ap.add_argument(
        "--pyradiomics-config",
        default=str(DEFAULT_CONFIG),
        help="Path to pyradiomics YAML config file",
    )
    ap.add_argument("--cpu-workers", type=int, default=32)
    ap.add_argument("--k-list", default="1,2,3,4", help="extra K values besides K_auto")
    ap.add_argument("--replicate", type=int, default=1,
                    help="repeat each case N times to form a larger cohort")
    ap.add_argument("--fastrad-cpu-workers", default="",
                    help="comma-separated process counts for fastrad on CPU, e.g. '16,32' "
                         "(same-hardware comparison against PyRadiomics)")
    ap.add_argument("--k-max", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT_DIR / "cohort_throughput"),
        help="Output path prefix without extension",
    )
    a = ap.parse_args()

    res = run(
        manifest=a.manifest,
        pyradiomics_config=a.pyradiomics_config,
        cpu_workers=a.cpu_workers,
        k_list=a.k_list,
        k_max=a.k_max,
        alpha=a.alpha,
        repeats=a.repeats,
        replicate=a.replicate,
        fastrad_cpu_workers=a.fastrad_cpu_workers,
        out=a.out,
    )
    print(res)


if __name__ == "__main__":
    main()