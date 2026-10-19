import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


def map_fastrad_to_pyrad_key(fastrad_k: str) -> tuple[str, str]:
    """Maps fastrad 'class:feature' to pyradiomics (class, feature_clean_key)."""
    c_name, f_name = fastrad_k.split(":")

    special_map = {
        ("firstorder", "10th_percentile"): ("firstorder", "10percentile"),
        ("firstorder", "90th_percentile"): ("firstorder", "90percentile"),
        ("firstorder", "interquartile_range"): ("firstorder", "interquartilerange"),
        ("firstorder", "mean_absolute_deviation"): (
            "firstorder",
            "meanabsolutedeviation",
        ),
        ("firstorder", "robust_mean_absolute_deviation"): (
            "firstorder",
            "robustmeanabsolutedeviation",
        ),
        ("firstorder", "root_mean_squared"): ("firstorder", "rootmeansquared"),
        ("firstorder", "total_energy"): ("firstorder", "totalenergy"),
        ("shape", "voxel_volume"): ("shape", "voxelvolume"),
        ("shape", "mesh_volume"): ("shape", "meshvolume"),
        ("shape", "surface_area"): ("shape", "surfacearea"),
        ("shape", "surface_volume_ratio"): ("shape", "surfacevolumeratio"),
        ("shape", "compactness_1"): ("shape", "compactness1"),
        ("shape", "compactness_2"): ("shape", "compactness2"),
        ("shape", "spherical_disproportion"): ("shape", "sphericaldisproportion"),
        ("shape", "maximum_3d_diameter"): ("shape", "maximum3ddiameter"),
        ("shape", "maximum_2d_diameter_slice"): ("shape", "maximum2ddiameterslice"),
        ("shape", "maximum_2d_diameter_column"): ("shape", "maximum2ddiametercolumn"),
        ("shape", "maximum_2d_diameter_row"): ("shape", "maximum2ddiameterrow"),
        ("shape", "major_axis_length"): ("shape", "majoraxislength"),
        ("shape", "minor_axis_length"): ("shape", "minoraxislength"),
        ("shape", "least_axis_length"): ("shape", "leastaxislength"),
    }

    if (c_name, f_name) in special_map:
        return special_map[(c_name, f_name)]

    return c_name, f_name.replace("_", "").lower()


def run():
    print("Running Numerical Parity Benchmark with PyRadiomics on Clinical Data...")
    project_root = Path(__file__).parent.parent
    img_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_image.nrrd"
    mask_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_label.nrrd"

    if not (img_path.exists() and mask_path.exists()):
        return "Error: TCIA clinical nrrd files not found."

    output_dir = project_root / "benchmarks" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # PyRadiomics
    sitk_image = sitk.ReadImage(str(img_path))
    sitk_mask = sitk.ReadImage(str(mask_path))

    # fastrad
    image_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_image)).float()
    mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask)).float()
    spacing = sitk_image.GetSpacing()[::-1]  # (Z, Y, X)

    fastrad_image = MedicalImage(image_tensor, spacing=spacing)
    fastrad_mask = Mask(mask_tensor, spacing=spacing)

    config_path = project_root / "pyradiomics_config.yaml"
    fastrad_settings = FeatureSettings.from_yaml(config_path, device="cpu")
    feature_classes = fastrad_settings.feature_classes

    pyrad_extractor = featureextractor.RadiomicsFeatureExtractor(str(config_path))
    fastrad_extractor = FeatureExtractor(fastrad_settings)

    fastrad_features = fastrad_extractor.extract(fastrad_image, fastrad_mask)

    pyrad_features = {}
    for cls in feature_classes:
        pyrad_extractor.disableAllFeatures()
        pyrad_extractor.enableFeatureClassByName(cls)
        res = pyrad_extractor.execute(sitk_image, sitk_mask)
        for k, v in res.items():
            if not k.startswith("original_"):
                continue
            parts = k.split("_")
            c_name = parts[1].lower()
            f_name = "_".join(parts[2:]).lower().replace("_", "")
            key = f"{c_name}_{f_name}"
            if hasattr(v, "item"):
                v = v.item()
            pyrad_features[key] = float(getattr(v, "real", v))

    # Compute differences and per-feature entries
    diff_stats = {}
    raw_feature_rows = []

    for k, fastrad_val in fastrad_features.items():
        if k == "firstorder:standard_deviation":
            # standard_deviation is an internal fastrad convenience feature, not in pyradiomics
            continue

        c_name, pyrad_f_name = map_fastrad_to_pyrad_key(k)
        pyrad_key = f"{c_name}_{pyrad_f_name}"

        if pyrad_key in pyrad_features:
            pyrad_val = pyrad_features[pyrad_key]
            abs_diff = abs(fastrad_val - pyrad_val)

            is_zero_ref = abs(pyrad_val) < 1e-12
            rel_diff = (abs_diff / abs(pyrad_val)) if not is_zero_ref else abs_diff

            if c_name not in diff_stats:
                diff_stats[c_name] = {
                    "diffs": [],
                    "within": 0,
                    "outside": 0,
                    "outside_list": [],
                    "zero_ref_count": 0,
                }

            diff_stats[c_name]["diffs"].append(abs_diff)
            if is_zero_ref:
                diff_stats[c_name]["zero_ref_count"] += 1

            if abs_diff <= 1e-4:
                diff_stats[c_name]["within"] += 1
            else:
                diff_stats[c_name]["outside"] += 1
                diff_stats[c_name]["outside_list"].append(
                    (k, abs_diff, fastrad_val, pyrad_val)
                )

            raw_feature_rows.append(
                {
                    "feature_class": c_name,
                    "fastrad_key": k,
                    "pyradiomics_key": pyrad_key,
                    "fastrad_val": fastrad_val,
                    "pyradiomics_val": pyrad_val,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "is_zero_reference": is_zero_ref,
                }
            )

    # Save raw CSV
    csv_path = output_dir / "parity_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_class",
                "fastrad_key",
                "pyradiomics_key",
                "fastrad_val",
                "pyradiomics_val",
                "abs_diff",
                "rel_diff",
                "is_zero_reference",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_feature_rows)

    # Generate Markdown Table
    md = []
    md.append("## Section 2: Numerical Parity with PyRadiomics (TCIA Clinical Image)\n")
    md.append("### 2.1 Feature-level Agreement Table\n")
    md.append(
        "| Feature Class | Validated Features | Mean Abs Diff | Median Abs Diff | IQR Abs Diff | Max Abs Diff | Within 1e-4 | Zero-Ref Features |"
    )
    md.append("|---|---|---|---|---|---|---|---|")

    total_validated = 0
    for cls in feature_classes:
        if cls in diff_stats:
            stats = diff_stats[cls]
            diffs = np.array(stats["diffs"])
            mean_diff = float(np.mean(diffs))
            med_diff = float(np.median(diffs))
            iqr_diff = float(np.percentile(diffs, 75) - np.percentile(diffs, 25))
            max_diff = float(np.max(diffs))
            within = stats["within"]
            outside = stats["outside"]
            n_features = len(diffs)
            total_validated += n_features
            z_count = stats["zero_ref_count"]

            md.append(
                f"| {cls} | {n_features} | {mean_diff:.2e} | {med_diff:.2e} | {iqr_diff:.2e} | {max_diff:.2e} | {within}/{n_features} | {z_count} |"
            )

    md.append(
        f"\n**Total Validated 3D Features Across 7 Classes**: {total_validated}\n"
    )
    md.append("\n**Outlier Analysis:**\n")
    has_outliers = False
    for cls in feature_classes:
        if cls in diff_stats and diff_stats[cls]["outside"] > 0:
            has_outliers = True
            for out_k, out_diff, f_val, p_val in diff_stats[cls]["outside_list"]:
                md.append(
                    f"- `{out_k}`: diff={out_diff:.2e} (fastrad={f_val:.4e}, PyRadiomics={p_val:.4e})"
                )

    if not has_outliers:
        md.append(
            "All features across all classes are strictly within the designated `1e-4` parity tolerance. "
            "Features with reference values of zero are evaluated using absolute error.\n"
        )

    return "\n".join(md) + "\n\n"


if __name__ == "__main__":
    print(run())
