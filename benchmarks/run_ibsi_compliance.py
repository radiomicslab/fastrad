import json
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import torch
from nibabel.spatialimages import SpatialImage

from fastrad.extractor import FeatureExtractor
from fastrad.image import Mask, MedicalImage
from fastrad.settings import FeatureSettings


def run():
    print("Running IBSI Compliance Benchmark...")
    project_root = Path(__file__).parent.parent
    fixture_dir = project_root / "tests" / "fixtures" / "ibsi"

    img_path = fixture_dir / "phantom.nii.gz"
    mask_path = fixture_dir / "mask.nii.gz"
    ref_path = fixture_dir / "reference.json"

    if not (img_path.exists() and mask_path.exists() and ref_path.exists()):
        return "Error: IBSI phantom dataset or reference not found. Run tests/test_ibsi.py first."

    img_ni = cast(SpatialImage, nib.load(str(img_path)))
    mask_ni = cast(SpatialImage, nib.load(str(mask_path)))

    img_data = np.transpose(img_ni.get_fdata(), (2, 1, 0))
    mask_data = np.transpose(mask_ni.get_fdata(), (2, 1, 0))
    zooms = img_ni.header.get_zooms()
    spacing = (zooms[2], zooms[1], zooms[0])

    image = MedicalImage(torch.from_numpy(img_data).float(), spacing=spacing)
    mask = Mask(torch.from_numpy(mask_data).float(), spacing=spacing)

    config_path = project_root / "pyradiomics_config.yaml"
    settings = FeatureSettings.from_yaml(config_path, device="cpu")

    features = FeatureExtractor(settings).extract(image, mask)
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_features = json.load(f)

    # We want to produce a table with four columns: feature name, IBSI reference value, fastrad value, absolute relative deviation (%). Every feature class, every feature.

    md = []
    md.append("## Section 1: IBSI Compliance\n")
    md.append("### 1.1 Digital Phantom Numerical Table\n")
    md.append(
        "| Feature Class | Feature Name | IBSI Ref Value | fastrad Value | Abs Relative Deviation (%) |"
    )
    md.append("|---|---|---|---|---|")

    for k, fastrad_val in sorted(features.items()):
        if k in ref_features:
            ref_val = ref_features[k]

            # Avoid divide by zero
            if abs(ref_val) < 1e-10:
                rel_dev = abs(fastrad_val - ref_val) * 100.0  # Just absolute deviation
            else:
                rel_dev = abs((fastrad_val - ref_val) / ref_val) * 100.0

            # fastrad keys are always "{class}:{feature_name}" (e.g. "firstorder:10th_percentile").
            # Splitting on "_" is wrong here because many feature *names* themselves contain
            # underscores (10th_percentile, large_dependence_high_gray_level_emphasis, etc.),
            # which previously collapsed several distinct features down to the same truncated
            # display label (e.g. multiple "*_emphasis" features all showing as "large | emphasis").
            class_name, feature_name = k.split(":", 1)

            # Formatting floats
            # E-notation is cleaner for very small or very large
            ref_str = f"{ref_val:.4e}"
            fastrad_str = f"{fastrad_val:.4e}"
            rel_dev_str = f"{rel_dev:.2e}%"

            md.append(
                f"| {class_name} | {feature_name} | {ref_str} | {fastrad_str} | {rel_dev_str} |"
            )

    return "\n".join(md) + "\n\n"


if __name__ == "__main__":
    print(run())
