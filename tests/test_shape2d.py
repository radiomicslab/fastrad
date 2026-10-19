import os

import numpy as np
import pytest
import SimpleITK as sitk
from radiomics import featureextractor

from fastrad import FeatureSettings


def to_fastrad_key(pyrad_key):
    if not pyrad_key.startswith("original_shape2D_"):
        return None

    f_name = pyrad_key.replace("original_shape2D_", "")
    return f"shape2D:{f_name}"


def test_shape2d_cpu():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    import sys

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    img_dir = os.path.join(base_dir, "fixtures", "image")
    mask_dir = os.path.join(base_dir, "fixtures", "mask")

    # Shape 2D requires size=1 mask dimension. Create one dynamically.
    image_vol = np.random.randint(0, 100, size=(5, 5, 5), dtype=np.uint16)
    mask_vol = np.zeros((5, 5, 5), dtype=np.uint8)
    mask_vol[2, 1:4, 1:4] = 1  # A 3x3 square on slice index = 2

    sitk_image = sitk.GetImageFromArray(image_vol)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))
    sitk_mask = sitk.GetImageFromArray(mask_vol)
    sitk_mask.SetSpacing((1.0, 1.0, 1.0))

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape2D")
    extractor.settings["force2D"] = True
    extractor.settings["force2Ddimension"] = 0

    pyrad_result = extractor.execute(sitk_image, sitk_mask)

    pyrad_features = {}
    for k, v in pyrad_result.items():
        fast_key = to_fastrad_key(k)
        if fast_key:
            pyrad_features[fast_key] = float(v)

    # Run fastrad
    # Since fastrad expects DICOMs from disk for normal execution, we can bypass MedicalImage
    # to feed it directly or temporarily save the tweaked arrays.
    import torch

    from fastrad.features import shape2d

    img_t = torch.from_numpy(image_vol).float()
    mask_t = torch.from_numpy(mask_vol).float()

    settings = FeatureSettings(
        feature_classes=["shape2D"], force2D=True, force2Ddimension=0
    )
    fastrad_features = shape2d.compute(img_t, mask_t, settings)

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

    print("All shape2D features match pyradiomics!")
