import csv
import subprocess
import sys
from pathlib import Path

import SimpleITK as sitk
import torch

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


def create_spherical_mask(
    image_shape, radius_mm: float, spacing: tuple[float, float, float]
):
    D, H, W = image_shape
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    r_z = max(1, int(radius_mm / spacing[0]))
    r_y = max(1, int(radius_mm / spacing[1]))
    r_x = max(1, int(radius_mm / spacing[2]))

    z, y, x = torch.meshgrid(
        torch.arange(D, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    dist_sq = (
        ((z - center_z) / r_z) ** 2
        + ((y - center_y) / r_y) ** 2
        + ((x - center_x) / r_x) ** 2
    )
    mask = torch.zeros(image_shape, dtype=torch.float32)
    mask[dist_sq <= 1.0] = 1.0
    return mask


def measure_peak_ram_subprocess(lib_name: str, radius: float) -> float:
    # NOTE: each branch below imports ONLY the library it needs.
    # A previous version of this script unconditionally imported both
    # `radiomics` and `fastrad` (plus torch/SimpleITK) at the top of every
    # subprocess regardless of which branch ran, which meant both the
    # "pyrad" and "fastrad" peak-RSS measurements were dominated by the
    # same shared import overhead (torch alone can be several hundred MB
    # resident) and could not detect the real differential memory usage
    # between the two libraries. Keeping the imports branch-local ensures
    # each subprocess's peak RSS reflects only the library under test.
    if lib_name == "pyrad":
        script = f'''
import os
import sys
import gc
import resource
import warnings
import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor

def create_spherical_mask(image_shape, radius_mm: float, spacing: tuple):
    D, H, W = image_shape
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    r_z = max(1, int(radius_mm / spacing[0]))
    r_y = max(1, int(radius_mm / spacing[1]))
    r_x = max(1, int(radius_mm / spacing[2]))

    z, y, x = torch.meshgrid(
        torch.arange(D, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    dist_sq = ((z - center_z) / r_z)**2 + ((y - center_y) / r_y)**2 + ((x - center_x) / r_x)**2
    mask = torch.zeros(image_shape, dtype=torch.float32)
    mask[dist_sq <= 1.0] = 1.0
    return mask

project_root = "{Path(__file__).parent.parent.absolute()}"
config_path = os.path.join(project_root, "pyradiomics_config.yaml")
img_path = os.path.join(project_root, "tests", "fixtures", "tcia", "lung1_image.nrrd")
img_sitk = sitk.ReadImage(str(img_path))
img_t = torch.from_numpy(sitk.GetArrayFromImage(img_sitk)).float()
spacing = img_sitk.GetSpacing()[::-1]
mask_t = create_spherical_mask(img_t.shape, {radius}, spacing)

gc.collect()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pyrad_ext = featureextractor.RadiomicsFeatureExtractor(config_path)
    m_sitk = sitk.GetImageFromArray(mask_t.numpy().astype(np.uint8))
    m_sitk.CopyInformation(img_sitk)
    pyrad_ext.execute(img_sitk, m_sitk)

maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    print(maxrss / (1024 * 1024))
else:
    print(maxrss / 1024)
'''
    else:
        script = f'''
import os
import sys
import gc
import resource
import numpy as np
import torch
import SimpleITK as sitk
from fastrad import MedicalImage, Mask, FeatureSettings, FeatureExtractor

def create_spherical_mask(image_shape, radius_mm: float, spacing: tuple):
    D, H, W = image_shape
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    r_z = max(1, int(radius_mm / spacing[0]))
    r_y = max(1, int(radius_mm / spacing[1]))
    r_x = max(1, int(radius_mm / spacing[2]))

    z, y, x = torch.meshgrid(
        torch.arange(D, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    dist_sq = ((z - center_z) / r_z)**2 + ((y - center_y) / r_y)**2 + ((x - center_x) / r_x)**2
    mask = torch.zeros(image_shape, dtype=torch.float32)
    mask[dist_sq <= 1.0] = 1.0
    return mask

project_root = "{Path(__file__).parent.parent.absolute()}"
config_path = os.path.join(project_root, "pyradiomics_config.yaml")
img_path = os.path.join(project_root, "tests", "fixtures", "tcia", "lung1_image.nrrd")
img_sitk = sitk.ReadImage(str(img_path))
img_t = torch.from_numpy(sitk.GetArrayFromImage(img_sitk)).float()
spacing = img_sitk.GetSpacing()[::-1]

fastrad_settings = FeatureSettings.from_yaml(config_path, device="cpu")
mask_t = create_spherical_mask(img_t.shape, {radius}, spacing)

gc.collect()

fastrad_img = MedicalImage(img_t, spacing=spacing)
f_mask = Mask(mask_t, spacing=spacing)
f_ext = FeatureExtractor(fastrad_settings)
f_ext.extract(fastrad_img, f_mask)

maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    print(maxrss / (1024 * 1024))
else:
    print(maxrss / 1024)
'''
    res = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0


def run():
    print("Running Memory Efficiency Benchmark...")
    project_root = Path(__file__).parent.parent
    img_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_image.nrrd"

    if not img_path.exists():
        return "Error: TCIA image not found."

    output_dir = project_root / "benchmarks" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("## Section 4: Memory Efficiency\n")

    # 4.1 Peak RAM vs ROI size
    print("  -> Profiling CPU RAM scaling against ROI size (4.1)...")
    md.append("### 4.1 Peak RAM vs ROI Size (CPU)\n")
    md.append(
        "| Radius (mm) | Voxel Count | PyRadiomics RAM (MB) | fastrad CPU RAM (MB) | Memory Overhead Ratio |"
    )
    md.append("|---|---|---|---|---|")

    img_sitk = sitk.ReadImage(str(img_path))
    img_t = torch.from_numpy(sitk.GetArrayFromImage(img_sitk)).float()
    spacing = img_sitk.GetSpacing()[::-1]

    mem_records = []

    for r in [5, 10, 15, 20, 25, 30]:
        mask_t = create_spherical_mask(img_t.shape, r, spacing)
        n_voxels = int(mask_t.sum().item())

        pyrad_mem = measure_peak_ram_subprocess("pyrad", r)
        fastrad_mem = measure_peak_ram_subprocess("fastrad", r)

        ratio = (fastrad_mem / pyrad_mem) if pyrad_mem > 0 else 0
        md.append(
            f"| {r} | {n_voxels} | {pyrad_mem:.2f} | {fastrad_mem:.2f} | {ratio:.2f}x |"
        )

        mem_records.append(
            {
                "radius_mm": r,
                "voxel_count": n_voxels,
                "pyradiomics_ram_mb": pyrad_mem,
                "fastrad_cpu_ram_mb": fastrad_mem,
                "overhead_ratio": ratio,
            }
        )

    md.append("\n")

    # Save CSV
    csv_path = output_dir / "memory_scaling.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "radius_mm",
                "voxel_count",
                "pyradiomics_ram_mb",
                "fastrad_cpu_ram_mb",
                "overhead_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(mem_records)

    # 4.2 GPU VRAM Profile
    print("  -> Profiling GPU VRAM per feature class (4.2)...")
    md.append("### 4.2 GPU VRAM Profile\n")
    if torch.cuda.is_available():
        # Inputs stay on the CPU, as in normal use: the extractor crops on the
        # host and moves only the ROI region to the GPU. Reported: peak memory
        # allocated by PyTorch during extract(), plus the driver-level used
        # memory (includes the CUDA context and the CuPy pool).
        config_path = project_root / "pyradiomics_config.yaml"
        classes = FeatureSettings.from_yaml(config_path, device="cuda").feature_classes
        mask_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_label.nrrd"
        inputs = [
            (
                "Synthetic 15 mm sphere",
                create_spherical_mask(img_t.shape, 15.0, spacing),
            )
        ]
        if mask_path.exists():
            inputs.append(
                (
                    "Clinical LUNG1-001 GTV-1",
                    torch.from_numpy(
                        sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
                    ).float(),
                )
            )
        fastrad_img = MedicalImage(img_t, spacing=spacing)

        def profile_vram(f_mask, classes_to_run):
            f_ext = FeatureExtractor(
                FeatureSettings.from_yaml(
                    config_path, feature_classes=classes_to_run, device="cuda"
                )
            )
            f_ext.extract(fastrad_img, f_mask)  # warm-up (CUDA context, kernels, CuPy)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            f_ext.extract(fastrad_img, f_mask)
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - base) / 1024**2
            free, total = torch.cuda.mem_get_info()
            return peak, (total - free) / 1024**2

        md.append(
            "| Input | Feature Class | Peak VRAM allocated by extract() (MB) | Device memory in use (MB) |"
        )
        md.append("|---|---|---|---|")
        for name, m in inputs:
            f_mask = Mask(m, spacing=spacing)
            n_vox = int((m > 0).sum())
            for cls in classes:
                peak, used = profile_vram(f_mask, [cls])
                md.append(f"| {name} ({n_vox} vox) | {cls} | {peak:.2f} | {used:.0f} |")
            peak, used = profile_vram(f_mask, classes)
            md.append(
                f"| {name} ({n_vox} vox) | **FULL PIPELINE** | **{peak:.2f}** | {used:.0f} |"
            )
    else:
        md.append("*GPU VRAM profiling skipped (CUDA not available).*")

    md.append("\n\n")
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
