import os

import pytest
import SimpleITK as sitk
from radiomics import featureextractor

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


def to_fastrad_key(pyrad_key):
    # original_shape_Maximum3DDiameter -> shape:maximum_3d_diameter
    if not pyrad_key.startswith("original_shape_"):
        return None

    f_name = pyrad_key.replace("original_shape_", "")

    mapping = {
        "VoxelVolume": "voxel_volume",
        "MeshVolume": "mesh_volume",
        "SurfaceArea": "surface_area",
        "SurfaceVolumeRatio": "surface_volume_ratio",
        "Compactness1": "compactness_1",
        "Compactness2": "compactness_2",
        "SphericalDisproportion": "spherical_disproportion",
        "Sphericity": "sphericity",
        "Maximum3DDiameter": "maximum_3d_diameter",
        "Maximum2DDiameterSlice": "maximum_2d_diameter_slice",
        "Maximum2DDiameterColumn": "maximum_2d_diameter_column",
        "Maximum2DDiameterRow": "maximum_2d_diameter_row",
        "MajorAxisLength": "major_axis_length",
        "MinorAxisLength": "minor_axis_length",
        "LeastAxisLength": "least_axis_length",
        "Elongation": "elongation",
        "Flatness": "flatness",
    }

    if f_name in mapping:
        snake_name = mapping[f_name]
        return f"shape:{snake_name}"

    return None


def test_shape_3d():
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

    # 1. Run Pyradiomics
    import make_fixtures

    sitk_image = sitk.GetImageFromArray(make_fixtures.img_vol)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))
    sitk_mask = sitk.GetImageFromArray(make_fixtures.mask_vol)
    sitk_mask.SetSpacing((1.0, 1.0, 1.0))

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")

    pyrad_result = extractor.execute(sitk_image, sitk_mask)

    pyrad_features = {}
    for k, v in pyrad_result.items():
        fast_key = to_fastrad_key(k)
        if fast_key:
            pyrad_features[fast_key] = float(v)

    # 2. Run fastrad
    image = MedicalImage.from_dicom(img_dir)
    mask = Mask.from_dicom(mask_dir)

    settings = FeatureSettings(feature_classes=["shape"], device="cpu")
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

    print("All Shape 3D features match pyradiomics!")


if __name__ == "__main__":
    test_shape_3d()
