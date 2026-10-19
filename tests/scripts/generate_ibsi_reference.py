import json
from pathlib import Path

from radiomics import featureextractor

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "ibsi"


def generate_reference():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    img_path = FIXTURE_DIR / "phantom.nii.gz"
    mask_path = FIXTURE_DIR / "mask.nii.gz"

    if not img_path.exists() or not mask_path.exists():
        print(f"Please ensure {img_path} and {mask_path} exist before running.")
        print("You can download them by running tests/test_ibsi.py first.")
        return

    # PyRadiomics extraction using the same settings as fastrad tests
    settings = {
        "binWidth": 25.0,
        "label": 1,
        "interpolator": "sitkNearestNeighbor",
        "resampledPixelSpacing": None,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Enable all feature classes to mirror fastrad settings
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")

    result = extractor.execute(str(img_path), str(mask_path))

    # Filter output and format keys to match fastrad
    reference_features = {}
    for key, value in result.items():
        if key.startswith("original_"):
            # original_glcm_Autocorrelation -> glcm:autocorrelation
            parts = key.split("_")
            if len(parts) == 3:
                cls_name = parts[1].lower()
                feat_name = parts[2]

                # Convert from camelCase/PascalCase to snake_case equivalent used in pyradiomics/fastrad
                import re

                s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", feat_name)
                snake_feat_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

                # E.g. Compactness1 -> compactness_1, Maximum2DDiameterRow -> maximum_2d_diameter_row
                snake_feat_name = snake_feat_name.replace(
                    "compactness1", "compactness_1"
                )
                snake_feat_name = snake_feat_name.replace(
                    "compactness2", "compactness_2"
                )
                snake_feat_name = snake_feat_name.replace("maximum2_d", "maximum_2d")
                snake_feat_name = snake_feat_name.replace("maximum3_d", "maximum_3d")
                snake_feat_name = snake_feat_name.replace(
                    "10_percentile", "10th_percentile"
                )
                snake_feat_name = snake_feat_name.replace(
                    "90_percentile", "90th_percentile"
                )

                fastrad_key = f"{cls_name}:{snake_feat_name}"

                # Convert scalars from numpy to float for JSON serialization
                try:
                    reference_features[fastrad_key] = float(value)
                except (ValueError, TypeError):
                    pass

    output_path = FIXTURE_DIR / "reference.json"
    with open(output_path, "w") as f:
        json.dump(reference_features, f, indent=2)

    print(f"Saved {len(reference_features)} reference features to {output_path}")


if __name__ == "__main__":
    generate_reference()
