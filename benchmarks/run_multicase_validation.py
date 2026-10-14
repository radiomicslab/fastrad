"""
Multi-case numerical parity and runtime validation.

*** THIS SCRIPT WAS WRITTEN WITHOUT ACCESS TO A GPU OR REAL TCIA DATA AND
    HAS NEVER BEEN RUN. It is a scaffold following the same pattern as
    run_numerical_parity.py, extended to loop over multiple cases instead
    of the single hardcoded LUNG1-001 case. You must supply real
    additional cases (see CASES below) before this produces any real
    output. ***

Addresses the reviewer criticism (raised by 3 of 4 reviewers) that all
numerical parity and runtime results in this study come from a single
clinical case. This script repeats the same fastrad-vs-PyRadiomics parity
comparison used in run_numerical_parity.py across multiple cases and
reports per-case and aggregate agreement.

Expected data layout (you must populate this yourself):
    tests/fixtures/tcia/multicase/<case_id>/image.nrrd
    tests/fixtures/tcia/multicase/<case_id>/label.nrrd

Each <case_id> directory should contain one clinical CT image and its
corresponding binary segmentation mask, in NRRD format, matching the
convention already used for LUNG1-001 in run_numerical_parity.py.
"""
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage

# Re-use the same key-mapping logic as run_numerical_parity.py so the two
# scripts cannot silently drift apart. If you edit the mapping in one
# place, edit it in both, or better, factor this out into a shared module
# (see review checklist -- this duplication already exists between
# run_numerical_parity.py and run_reproducibility_stability.py).
from run_numerical_parity import map_fastrad_to_pyrad_key  # noqa: E402


# List of case IDs to validate, each expected to have its own
# image.nrrd / label.nrrd under tests/fixtures/tcia/multicase/<case_id>/.
# THIS LIST IS EMPTY BY DEFAULT -- populate it with real TCIA cases
# (ideally spanning different nodule sizes and, if feasible, a non-lung
# anatomical site) before running this script.
CASES: list[str] = [
    # Pre-populated from list_tcia_patients.py output (all 15 checked
    # candidates had CT+RTSTRUCT). IMPORTANT: after running
    # download_tcia_sample.py, check its printed "Label check" line for
    # each case -- remove any case here that didn't print "OK" (i.e. its
    # mask wasn't exactly [0, 1]) before running this script.
    "LUNG1-002",
    "LUNG1-003",
    "LUNG1-004",
    "LUNG1-005",
    "LUNG1-006",
]


def validate_case(case_id: str, img_path: Path, mask_path: Path, config_path: Path) -> dict:
    sitk_image = sitk.ReadImage(str(img_path))
    sitk_mask = sitk.ReadImage(str(mask_path))

    image_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_image)).float()
    mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask)).float()
    spacing = sitk_image.GetSpacing()[::-1]

    fastrad_image = MedicalImage(image_tensor, spacing=spacing)
    fastrad_mask = Mask(mask_tensor, spacing=spacing)

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
            pyrad_features[key] = float(getattr(v.item() if hasattr(v, "item") else v, "real", v.item() if hasattr(v, "item") else v))

    abs_diffs = []
    n_within = 0
    n_total = 0
    outliers = []  # (fastrad_key, pyrad_key, fastrad_val, pyrad_val, diff)
    for k, fastrad_val in fastrad_features.items():
        if k == "firstorder:standard_deviation":
            continue
        c_name, pyrad_f_name = map_fastrad_to_pyrad_key(k)
        pyrad_key = f"{c_name}_{pyrad_f_name}"
        if pyrad_key in pyrad_features:
            diff = abs(fastrad_val - pyrad_features[pyrad_key])
            abs_diffs.append(diff)
            n_total += 1
            if diff <= 1e-4:
                n_within += 1
            else:
                outliers.append((k, pyrad_key, fastrad_val, pyrad_features[pyrad_key], diff))

    if outliers:
        print(f"    OUTLIERS for {case_id}:")
        for fk, pk, fv, pv, d in sorted(outliers, key=lambda x: -x[4]):
            print(f"      {fk} (pyrad: {pk}): fastrad={fv:.6f}, pyradiomics={pv:.6f}, diff={d:.6f}")

    n_voxels = int(mask_tensor.sum().item())
    return {
        "case_id": case_id,
        "voxel_count": n_voxels,
        "n_features_compared": n_total,
        "n_within_tolerance": n_within,
        "mean_abs_diff": float(np.mean(abs_diffs)) if abs_diffs else float("nan"),
        "max_abs_diff": float(np.max(abs_diffs)) if abs_diffs else float("nan"),
    }


def run():
    print("Running Multi-Case Numerical Parity Validation...")
    project_root = Path(__file__).parent.parent
    config_path = project_root / "pyradiomics_config.yaml"
    multicase_dir = project_root / "tests" / "fixtures" / "tcia" / "multicase"
    output_dir = project_root / "benchmarks" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not CASES:
        msg = ("No cases configured in CASES list at the top of this script. "
               "This script cannot run until you add real TCIA case IDs and "
               "populate tests/fixtures/tcia/multicase/<case_id>/{image,label}.nrrd "
               "for each one. See the module docstring for the expected layout.")
        print(msg)
        return msg

    records = []
    for case_id in CASES:
        case_dir = multicase_dir / case_id
        img_path = case_dir / "image.nrrd"
        mask_path = case_dir / "label.nrrd"
        if not (img_path.exists() and mask_path.exists()):
            print(f"  -> SKIPPING {case_id}: files not found at {case_dir}")
            continue
        print(f"  -> Validating case {case_id}...")
        records.append(validate_case(case_id, img_path, mask_path, config_path))

    if not records:
        msg = "No cases were actually validated (all files missing). See warnings above."
        print(msg)
        return msg

    csv_path = output_dir / "multicase_validation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    all_within = [r["n_within_tolerance"] == r["n_features_compared"] for r in records]
    summary = (
        f"Validated {len(records)} cases. "
        f"All features within 1e-4 tolerance on {sum(all_within)}/{len(records)} cases. "
        f"See {csv_path} for per-case detail."
    )
    print(summary)
    return summary


if __name__ == "__main__":
    print(run())
