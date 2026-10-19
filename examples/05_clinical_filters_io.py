"""
Example 05: Clinical Filters and I/O Alignment
----------------------------------------------
This example demonstrates exactly how to interface fastrad with raw clinical data
mimicking the exact workflow of PyRadiomics (including dynamic SITK resizing and
complex Mathematical Filter combinations).
"""

import tempfile

import numpy as np
import SimpleITK as sitk

import fastrad
from fastrad import FeatureExtractor, FeatureSettings


def create_dummy_data():
    """Generates synthetic bounding boxes simulating a tumor scan."""
    img_sitk = sitk.GetImageFromArray(
        np.random.randint(0, 100, (20, 20, 20)).astype(np.float32)
    )
    img_sitk.SetSpacing((1.0, 1.0, 1.0))

    mask_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    mask_arr[5:15, 5:15, 5:15] = 1  # Central mass
    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk.SetSpacing((1.0, 1.0, 1.0))
    return img_sitk, mask_sitk


def main():
    print("--- 1. Generating Dummy Clinical Data ---")
    img_sitk, mask_sitk = create_dummy_data()

    # Save temporarily to mimic physical path loads
    with tempfile.TemporaryDirectory() as tmp:
        img_path = f"{tmp}/image.nii.gz"
        mask_path = f"{tmp}/mask.nii.gz"

        sitk.WriteImage(img_sitk, img_path)
        sitk.WriteImage(mask_sitk, mask_path)

        print("--- 2. Native File Loading & Isotropic Resampling ---")
        # Load directly, forcing crop to bbox to accelerate GPU logic
        img, mask = fastrad.load_and_align(
            image_path=img_path,
            mask_path=mask_path,
            resample_spacing=(
                2.0,
                2.0,
                2.0,
            ),  # Dynamically warp physical bounding box constraints
            crop=True,
        )
        print(f"Resampled Image Bounds: {img.tensor.shape}")

        print("--- 3. Complex Pipeline Filtration ---")
        # Generate 1 Original, 2 Laplacian of Gaussians, and 1 Square filters simultaneously
        filter_params = {"Original": {}, "LoG": {"sigma": [1.0, 3.0]}, "Square": {}}

        filtered_stack = fastrad.apply_builtin_filters(img, filter_params)
        print(
            f"Generated {len(filtered_stack)} complex image filters natively over GPU bindings."
        )

        print("--- 4. Feature Extraction Mapping ---")
        settings = FeatureSettings(
            feature_classes=["firstorder", "glcm"],
            bin_width=25.0,
            device="auto",  # Dynamically maps MPS/CUDA/CPU appropriately
        )
        extractor = FeatureExtractor(settings)

        complete_features = {}
        for filter_name, filtered_img in filtered_stack.items():
            print(f"Extracting {filter_name}...")
            # Compute mapping for this structural bounds independently
            fts = extractor.extract(filtered_img, mask)
            for f_name, value in fts.items():
                complete_features[f"{filter_name}_{f_name}"] = value

        print(
            f"\nFinal Computational Vector perfectly evaluated across {len(complete_features)} features."
        )


if __name__ == "__main__":
    main()
