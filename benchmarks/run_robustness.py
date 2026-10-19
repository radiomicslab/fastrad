import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


def create_edge_case(t_shape, case_type):
    mask = torch.zeros(t_shape, dtype=torch.float32)
    img = torch.rand(t_shape, dtype=torch.float32) * 1000
    spacing = (1.0, 1.0, 1.0)

    if case_type == "empty":
        pass
    elif case_type == "single":
        mask[t_shape[0] // 2, t_shape[1] // 2, t_shape[2] // 2] = 1.0
    elif case_type == "small":
        z, y, x = t_shape[0] // 2, t_shape[1] // 2, t_shape[2] // 2
        mask[z, y, x : x + 2] = 1.0
        mask[z, y + 1, x : x + 2] = 1.0
    elif case_type == "non-isotropic":
        mask[
            t_shape[0] // 2 - 1 : t_shape[0] // 2 + 2,
            t_shape[1] // 2 - 1 : t_shape[1] // 2 + 2,
            t_shape[2] // 2 - 1 : t_shape[2] // 2 + 2,
        ] = 1.0
        spacing = (5.0, 0.5, 0.5)
    elif case_type == "homogeneous":
        img = torch.ones(t_shape, dtype=torch.float32) * 100.0
        mask[
            t_shape[0] // 2 - 2 : t_shape[0] // 2 + 2,
            t_shape[1] // 2 - 2 : t_shape[1] // 2 + 2,
            t_shape[2] // 2 - 2 : t_shape[2] // 2 + 2,
        ] = 1.0
    elif case_type == "single-slice-2d":
        mask[
            t_shape[0] // 2,
            t_shape[1] // 4 : 3 * t_shape[1] // 4,
            t_shape[2] // 4 : 3 * t_shape[2] // 4,
        ] = 1.0
    elif case_type == "shape-mismatch":
        img = torch.rand((10, 10, 10), dtype=torch.float32)
        mask = torch.zeros((10, 12, 10), dtype=torch.float32)
        mask[5, 5, 5] = 1.0
        return img, mask, spacing

    return img, mask, spacing


def extract_safe(extractor_func, *args):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = extractor_func(*args)
            if len(w) > 0:
                warning_msgs = [str(warn.message) for warn in w]
                return f"Graceful Completion (Warnings: {len(warning_msgs)})"
            return "Graceful Completion"
    except Exception as e:
        err_type = type(e).__name__
        return f"{err_type}"


def _is_exception_like(observed: str) -> bool:
    """True if the observed behaviour string looks like a raised exception."""
    return not observed.startswith("Graceful")


def check_verified(expected: str, observed: str) -> str:
    """
    Honest pass/fail check: does the *category* of observed behaviour
    (exception vs. graceful completion) match what was expected for this
    edge case? This replaces a previous version of this script that
    unconditionally wrote "Yes" for every row regardless of what actually
    happened.
    """
    expected_is_exception = ("Error" in expected) or ("Exception" in expected)
    observed_is_exception = _is_exception_like(observed)
    return "Yes" if (expected_is_exception == observed_is_exception) else "Review"


def run():
    print("Running Expanded Robustness Edge Cases Benchmark (Table 8)...")
    md = []
    md.append("## Section 6: Robustness\n")
    md.append("### 6.1 Expanded Edge Case Matrix\n")
    md.append(
        "| Edge Case | Expected Behaviour | fastrad Behaviour | PyRadiomics Behaviour | Robustness Verified |"
    )
    md.append("|---|---|---|---|---|")

    cases = {
        "Empty Mask": ("empty", "ValueError / Exception"),
        "Single-voxel ROI": ("single", "Graceful / Warning"),
        "Very Small ROI (< 8 voxels)": ("small", "Graceful Completion"),
        "Homogeneous / Flat Intensity ROI": (
            "homogeneous",
            "Graceful Completion (zero variance)",
        ),
        "Single-slice 2D ROI in 3D Volume": ("single-slice-2d", "Graceful Completion"),
        "Non-isotropic Spacing": ("non-isotropic", "UserWarning"),
        "Shape Mismatch (Image != Mask)": ("shape-mismatch", "ValueError / Exception"),
    }

    project_root = Path(__file__).parent.parent
    config_path = project_root / "pyradiomics_config.yaml"
    pyrad_ext = featureextractor.RadiomicsFeatureExtractor(str(config_path))
    fastrad_settings = FeatureSettings.from_yaml(config_path, device="cpu")

    for case_name, (case_id, expected) in cases.items():
        img_t, mask_t, spacing = create_edge_case((10, 10, 10), case_id)

        # Fastrad
        f_img = MedicalImage(img_t, spacing=spacing)
        f_mask = Mask(mask_t, spacing=spacing)
        f_ext = FeatureExtractor(fastrad_settings)
        f_res = extract_safe(f_ext.extract, f_img, f_mask)

        # PyRadiomics
        try:
            s_img = sitk.GetImageFromArray(img_t.numpy())
            s_img.SetSpacing((spacing[2], spacing[1], spacing[0]))
            s_mask = sitk.GetImageFromArray(mask_t.numpy().astype(np.uint8))
            s_mask.SetSpacing((spacing[2], spacing[1], spacing[0]))
            p_res = extract_safe(pyrad_ext.execute, s_img, s_mask)
        except Exception as e:
            p_res = type(e).__name__

        verified = check_verified(expected, f_res)
        md.append(f"| {case_name} | {expected} | {f_res} | {p_res} | {verified} |")

    md.append("\n\n")
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
