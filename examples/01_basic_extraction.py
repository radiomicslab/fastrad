"""
Example 01: Basic Feature Extraction
====================================

This script demonstrates the most fundamental usage of the FastRadiomics (fastrad) library.
It covers loading a medical image and its corresponding segmentation mask, configuring
the feature extractor, and retrieving the computed features.

Requirements:
- pip install fastrad SimpleITK torch
"""

import warnings

import SimpleITK as sitk
import torch

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


def main():
    # 1. Load your medical image and mask utilizing SimpleITK
    # For this example, we assume you have a NIfTI or NRRD volume and mask.
    # Replace these paths with actual file paths to absolute clinical datasets.
    image_path = "path/to/lung_ct_scan.nii.gz"
    mask_path = "path/to/lung_tumor_segmentation.nii.gz"

    print(f"Loading Image: {image_path}")
    print(f"Loading Mask: {mask_path}")

    try:
        sitk_image = sitk.ReadImage(image_path)
        sitk_mask = sitk.ReadImage(mask_path)

        # Convert SimpleITK images to PyTorch tensors
        # Remember: SimpleITK returns arrays in (z, y, x) order.
        image_tensor = torch.tensor(
            sitk.GetArrayFromImage(sitk_image), dtype=torch.float32
        )
        mask_tensor = torch.tensor(sitk.GetArrayFromImage(sitk_mask), dtype=torch.int32)

        # Get physical spacing (z, y, x)
        spacing = sitk_image.GetSpacing()
        spacing_zyx = (spacing[2], spacing[1], spacing[0])

        # 2. Instantiate fastrad core objects
        fastrad_image = MedicalImage(tensor=image_tensor, spacing=spacing_zyx)
        fastrad_mask = Mask(tensor=mask_tensor, spacing=spacing_zyx)

    except Exception as e:
        warnings.warn(
            f"Could not load physical files (placeholder paths used). Error: {e}"
        )
        print(
            "Falling back to synthetic tensor generation for demonstration purposes..."
        )

        # Fallback for demonstration: create synthetic 64x64x64 volume
        fastrad_image = MedicalImage(
            tensor=torch.rand((64, 64, 64)) * 1000, spacing=(1.0, 1.0, 1.0)
        )

        # Create a spherical mask in the center
        mask_tensor = torch.zeros((64, 64, 64))
        # Center: 32, 32, 32; Radius: 15
        z, y, x = torch.meshgrid(
            torch.arange(64), torch.arange(64), torch.arange(64), indexing="ij"
        )
        distance = torch.sqrt((z - 32) ** 2 + (y - 32) ** 2 + (x - 32) ** 2)
        mask_tensor[distance <= 15] = 1

        fastrad_mask = Mask(tensor=mask_tensor, spacing=(1.0, 1.0, 1.0))

    # 3. Configure the Extractor Settings
    # We specify which feature classes we want to compute.
    # By default, "device='auto'" will utilize the CPU if CUDA is unavailable.
    settings = FeatureSettings(
        feature_classes=["firstorder", "shape", "glcm"], bin_width=25.0, device="cpu"
    )

    # 4. Initialize the Extractor and Execute
    extractor = FeatureExtractor(settings)

    print("\nStarting basic feature extraction...")
    features = extractor.extract(fastrad_image, fastrad_mask)

    # 5. Review the Results
    print(f"\nSuccessfully extracted {len(features)} features!")
    print("\n--- Sample Features ---")

    for feature_name, value in list(features.items())[:10]:
        print(f"{feature_name:<30}: {value:.4f}")
    print("...")


if __name__ == "__main__":
    main()
