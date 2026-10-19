import os
import re

import pytest
import SimpleITK as sitk
from radiomics import featureextractor

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


def to_fastrad_key(pyrad_key):
    if not pyrad_key.startswith("original_glszm_"):
        return None

    f_name = pyrad_key.replace("original_glszm_", "")
    # Convert camel case to snake case
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", f_name).lower()
    return f"glszm:{snake}"


def test_glszm_cpu():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    import sys

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    img_dir = os.path.join(base_dir, "fixtures", "image")
    mask_dir = os.path.join(base_dir, "fixtures", "mask")

    import make_fixtures

    sitk_image = sitk.GetImageFromArray(make_fixtures.img_vol)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))
    sitk_mask = sitk.GetImageFromArray(make_fixtures.mask_vol)
    sitk_mask.SetSpacing((1.0, 1.0, 1.0))

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("glszm")
    extractor.settings["binWidth"] = 25.0

    pyrad_result = extractor.execute(sitk_image, sitk_mask)

    pyrad_features = {}
    for k, v in pyrad_result.items():
        fast_key = to_fastrad_key(k)
        if fast_key:
            pyrad_features[fast_key] = float(v)

    # Run fastrad
    image = MedicalImage.from_dicom(img_dir)
    mask = Mask.from_dicom(mask_dir)

    settings = FeatureSettings(feature_classes=["glszm"], bin_width=25.0, device="cpu")
    fastrad_ext = FeatureExtractor(settings)

    fastrad_features = fastrad_ext.extract(image, mask)

    assert len(pyrad_features) > 0, "Pyradiomics returned no features"

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

    print("All GLSZM features match pyradiomics!")
