import os

import pytest
import SimpleITK as sitk
from radiomics import featureextractor

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


# Map Pyradiomics camel case to fastrad snake case
def to_fastrad_key(pyrad_key):
    # original_firstorder_Energy -> firstorder:energy
    # Strip "original_"
    if not pyrad_key.startswith("original_"):
        return None

    parts = pyrad_key.split("_")
    if len(parts) < 3:
        return None

    f_class = parts[1]
    f_name = "_".join(parts[2:])

    # Manual mappings for specifics
    mapping = {
        "10Percentile": "10th_percentile",
        "90Percentile": "90th_percentile",
        "InterquartileRange": "interquartile_range",
        "MeanAbsoluteDeviation": "mean_absolute_deviation",
        "RobustMeanAbsoluteDeviation": "robust_mean_absolute_deviation",
        "RootMeanSquared": "root_mean_squared",
        "TotalEnergy": "total_energy",
    }

    if f_name in mapping:
        snake_name = mapping[f_name]
    else:
        # Simple lowercase for single words
        snake_name = f_name.lower()

    return f"{f_class}:{snake_name}"


def test_firstorder_cpu():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    import sys

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    img_dir = os.path.join(base_dir, "fixtures", "image")
    mask_dir = os.path.join(base_dir, "fixtures", "mask")

    # Generate fixtures if they don't exist
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        import make_fixtures

        make_fixtures.create_dicom_series(img_dir, make_fixtures.img_vol)
        make_fixtures.create_dicom_series(mask_dir, make_fixtures.mask_vol, True)

    # 1. Run Pyradiomics (using SimpleITK array wrapper to avoid GDCM strictness)
    import make_fixtures

    sitk_image = sitk.GetImageFromArray(make_fixtures.img_vol)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))
    sitk_mask = sitk.GetImageFromArray(make_fixtures.mask_vol)
    sitk_mask.SetSpacing((1.0, 1.0, 1.0))

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.settings["binWidth"] = 25.0

    pyrad_result = extractor.execute(sitk_image, sitk_mask)

    pyrad_features = {}
    for k, v in pyrad_result.items():
        fast_key = to_fastrad_key(k)
        if fast_key:
            pyrad_features[fast_key] = float(v)

    # 2. Run fastrad
    image = MedicalImage.from_dicom(img_dir)
    mask = Mask.from_dicom(mask_dir)

    settings = FeatureSettings(
        feature_classes=["firstorder"], bin_width=25.0, device="cpu"
    )
    fastrad_ext = FeatureExtractor(settings)

    fastrad_features = fastrad_ext.extract(image, mask)

    # 3. Compare
    assert len(pyrad_features) > 0, "Pyradiomics returned no features"
    assert len(fastrad_features) > 0, "Fastrad returned no features"

    errors = []
    for k, ref_val in pyrad_features.items():
        if k not in fastrad_features:
            errors.append(f"Missing {k}")
            continue
        fast_val = fastrad_features[k]
        diff = abs(fast_val - ref_val)
        if str(ref_val) == "nan" and str(fast_val) == "nan":
            continue
        if diff >= 1e-4:
            errors.append(
                f"{k}: pyrad={ref_val:.4f}, fastrad={fast_val:.4f}, diff={diff:.4f}"
            )

    if errors:
        for e in errors:
            print(e)
        pytest.fail(f"{len(errors)} features mismatched")

    print("All first order features match pyradiomics!")
