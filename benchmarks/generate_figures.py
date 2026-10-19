"""
Generate the five figures requested by reviewers (A1-A5).

*** THIS SCRIPT WAS WRITTEN WITHOUT ACCESS TO A GPU OR THE REAL TCIA/RIDER
    DATA AND HAS NEVER BEEN RUN. Treat it as a scaffold, not a verified
    tool. Run it locally, inspect every figure, and fix anything that
    looks wrong before using the output in the manuscript. ***

Expects the following CSVs to already exist in benchmarks/outputs/
(produced by the other benchmark scripts):
    - runtime_table.csv        (run_runtime_performance.py)
    - parity_table.csv         (run_numerical_parity.py)
    - rider_icc_per_feature.csv (run_reproducibility_stability.py)
    - perturbation_stability.csv (run_reproducibility_stability.py)
    - memory_scaling.csv       (run_memory_efficiency.py)

Produces PNG figures in benchmarks/figures/:
    A1_architecture.png   - static schematic (hand-drawn, not data-driven)
    A2_gpu_scaling.png    - GPU speedup vs. ROI voxel count
    A3_icc_scatter.png    - per-feature ICC, fastrad vs. PyRadiomics
    A4_parity_scatter.png - Bland-Altman style parity plot
    A5_memory_scaling.png - CPU RAM (and GPU VRAM, if available) vs. ROI size
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "benchmarks" / "figures"


def _require(csv_name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / csv_name
    if not path.exists():
        print(
            f"ERROR: {path} does not exist. Run the benchmark script that "
            f"produces it before calling generate_figures.py.",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_csv(path)


def fig_a1_architecture():
    """
    A1: system architecture diagram.

    This is NOT data-driven -- it is a static schematic of the pipeline
    described in the paper's Methods section (DICOM -> tensor -> device
    routing -> discretisation/masking -> per-class feature modules ->
    aggregation). Drawing this well by hand (or in a vector tool) will
    almost certainly look better than this quick matplotlib version;
    treat this as a placeholder to replace, not a final figure.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    stages = [
        "DICOM\ningestion",
        "Tensor\nconversion",
        "Device\nrouting\n(CPU/CUDA)",
        "Discretisation\n+ ROI mask",
        "Per-class\nfeature modules\n(8x)",
        "Result\naggregation",
    ]
    n = len(stages)
    for i, label in enumerate(stages):
        x = i * 1.6
        ax.add_patch(
            plt.Rectangle(
                (x, 0), 1.3, 1, fill=True, facecolor="#cfe8ff", edgecolor="black"
            )
        )
        ax.text(x + 0.65, 0.5, label, ha="center", va="center", fontsize=8)
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + 1.5, 0.5),
                xytext=(x + 1.3, 0.5),
                arrowprops={"arrowstyle": "->"},
            )
    ax.set_xlim(-0.3, n * 1.6)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")
    ax.set_title("fastrad pipeline (schematic -- replace with a proper diagram)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "A1_architecture.png", dpi=200)
    plt.close(fig)


def fig_a2_gpu_scaling():
    """A2: end-to-end extraction time vs. ROI voxel count (single full calls), from runtime_table.csv."""
    df = _require("runtime_table.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        df["voxel_count"],
        df["pyradiomics_median"],
        marker="o",
        label="PyRadiomics (1 thread)",
    )
    ax.plot(
        df["voxel_count"],
        df["fastrad_cpu_median"],
        marker="s",
        label="fastrad CPU (1 thread)",
    )
    if not df["fastrad_gpu_median"].isna().all():
        ax.plot(
            df["voxel_count"], df["fastrad_gpu_median"], marker="^", label="fastrad GPU"
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ROI voxel count (log scale)")
    ax.set_ylabel("Extraction time per scan (s, log scale)")
    ax.set_title("Single-scan extraction time vs. ROI size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "A2_gpu_scaling.png", dpi=200)
    plt.close(fig)


def fig_a3_icc_scatter():
    """A3: per-feature ICC scatter, fastrad vs. PyRadiomics, with identity line."""
    df = _require("rider_icc_per_feature.csv")
    if df.empty:
        print("WARNING: rider_icc_per_feature.csv is empty. Skipping A3.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["pyradiomics_icc"], df["fastrad_icc"], alpha=0.5, s=15)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="identity")
    ax.set_xlabel("PyRadiomics ICC(2,1)")
    ax.set_ylabel("fastrad ICC(2,1)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    n_subjects = (
        df["n_subjects"].iloc[0] if "n_subjects" in df.columns and len(df) else "?"
    )
    ax.set_title(
        f"Per-feature ICC(2,1): fastrad vs. PyRadiomics (N={n_subjects} pairs)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "A3_icc_scatter.png", dpi=200)
    plt.close(fig)


def fig_a4_parity_scatter():
    """A4: Bland-Altman style plot of paired feature differences."""
    df = _require("parity_table.csv")
    if df.empty:
        print("WARNING: parity_table.csv is empty. Skipping A4.")
        return

    mean_val = (df["fastrad_val"] + df["pyradiomics_val"]) / 2.0
    diff_val = df["fastrad_val"] - df["pyradiomics_val"]

    # Use signed log-scale magnitude for mean_val on the x-axis since
    # radiomic features span many orders of magnitude; guard against
    # exactly-zero means.
    eps = 1e-12
    x = np.sign(mean_val) * np.log10(np.abs(mean_val) + eps)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = df["feature_class"].astype("category").cat.codes
    sc = ax.scatter(x, diff_val, c=colors, cmap="tab10", alpha=0.6, s=15)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("sign(mean) * log10(|mean(fastrad, PyRadiomics)| + eps)")
    ax.set_ylabel("fastrad - PyRadiomics (absolute difference)")
    ax.set_title("Feature-level parity (Bland-Altman style)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "A4_parity_scatter.png", dpi=200)
    plt.close(fig)


def fig_a5_memory_scaling():
    """A5: CPU RAM (and GPU VRAM if available) vs. ROI size."""
    df = _require("memory_scaling.csv")
    if df.empty:
        print("WARNING: memory_scaling.csv is empty. Skipping A5.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        df["voxel_count"],
        df["pyradiomics_ram_mb"],
        marker="o",
        label="PyRadiomics CPU RAM",
    )
    ax.plot(
        df["voxel_count"], df["fastrad_cpu_ram_mb"], marker="s", label="fastrad CPU RAM"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ROI voxel count (log scale)")
    ax.set_ylabel("Peak RAM (MB, log scale)")
    ax.set_title("CPU RAM scaling vs. ROI size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "A5_memory_scaling.png", dpi=200)
    plt.close(fig)


def fig_a6_cohort_throughput():
    """A6: cohort throughput vs. concurrent GPU workers, with CPU baselines, from cohort_throughput.csv."""
    df = _require("cohort_throughput.csv")
    gpu = df[df["config"] == "fastrad_gpu"].sort_values("workers")
    pyr = df[df["config"] == "pyradiomics"]
    cpu = df[df["config"] == "fastrad_cpu"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        gpu["workers"],
        gpu["median_scans_per_min"],
        marker="o",
        label="fastrad, 1 GPU, K concurrent workers",
    )
    for _, r in pyr.iterrows():
        ax.axhline(
            r["median_scans_per_min"],
            color="C1",
            linestyle="--",
            label=f"PyRadiomics, {int(r['workers'])} CPU processes",
        )
    for i, (_, r) in enumerate(cpu.iterrows()):
        ax.axhline(
            r["median_scans_per_min"],
            color="C2",
            linestyle=[":", "-."][i % 2],
            label=f"fastrad CPU, {int(r['workers'])} processes",
        )
    ax.set_xlabel("Concurrent GPU workers K")
    ax.set_ylabel("Cohort throughput (scans/min)")
    ax.set_title("Cohort throughput")
    ax.set_ylim(bottom=0)
    ax.set_xticks(gpu["workers"].tolist())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "A6_cohort_throughput.png", dpi=200)
    plt.close(fig)


def run():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating A1 (architecture schematic)...")
    fig_a1_architecture()
    print("Generating A2 (GPU scaling)...")
    fig_a2_gpu_scaling()
    print("Generating A3 (ICC scatter)...")
    fig_a3_icc_scatter()
    print("Generating A4 (parity Bland-Altman)...")
    fig_a4_parity_scatter()
    print("Generating A5 (memory scaling)...")
    fig_a5_memory_scaling()
    print("Generating A6 (cohort throughput)...")
    if (OUTPUT_DIR / "cohort_throughput.csv").exists():
        fig_a6_cohort_throughput()
    else:
        print("  cohort_throughput.csv not found; skipping A6.")
    print(f"Done. Check {FIGURES_DIR} and inspect every figure before use.")


if __name__ == "__main__":
    run()
