"""
Example 04: Advanced Configuration and Hardware Routing
=======================================================

This example explores the specific robust configuration dials and edge-cases
`fastrad` is capable of handling internally, bypassing PyRadiomics execution errors.
"""

import warnings

import torch

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


def create_non_isotropic_data():
    """Generates synthetic medical imagery featuring varying anisotropic spacing parameters."""
    tensor = torch.rand((32, 32, 32)) * 1000

    # 0.5x0.5x3.0mm is common for clinical scans
    image = MedicalImage(tensor=tensor, spacing=(3.0, 0.5, 0.5))

    mask_tensor = torch.zeros((32, 32, 32))
    mask_tensor[10:20, 10:20, 10:20] = 1  # Cube mask
    mask = Mask(tensor=mask_tensor, spacing=(3.0, 0.5, 0.5))

    return image, mask


def main():
    print("--- Exploring Advanced fastrad Handling ---\n")

    # Scene 1: Robust Hardware Target Configuration
    # We enforce device selection as "cpu" but specify
    # custom feature extraction vectors and explicit IBSI
    # compatible absolute bin scaling

    settings = FeatureSettings(
        # We explicitly skip Firstorder and Shape, directly extracting textures
        feature_classes=["glcm", "glszm", "gldm"],
        bin_width=10.0,  # Adjusting bin_width alters grey-scale quantization density
        device="cpu",
    )

    extractor = FeatureExtractor(settings)

    image, mask = create_non_isotropic_data()

    # Scene 2: Handling Spacing Variations Gracefully
    # Standard radiomics libraries mandate image resamplers to isotropic voxels
    # before calculation out of fear of skewed spatial textures.
    # fastrad checks this dynamically and alerts the user gracefully.

    print("Initiating Extraction over Anisotropic Geometry...")
    try:
        # Expected to trigger a runtime UserWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            features = extractor.extract(image, mask)

            if len(w) > 0:
                print(f"Warning explicitly intercepted: {w[-1].message}")

    except Exception as e:
        print(f"Extraction halted abruptly: {e}")

    print(
        f"\nExtracted {len(features)} attributes despite anisotropic data boundaries."
    )

    # Scene 3: Handling PyTorch `OutOfMemoryError` Triggers
    # `fastrad` utilizes substantial arrays matching clinical bounds.
    # If the user targets CUDA on a 2GB VRAM device, `fastrad` is
    # architected to catch the physical CUDA exception via Python,
    # clear the GPU, and automatically execute the failed component fallback on multi-core CPU.

    print("\nSimulated Check: Out-Of-Memory Fallback.")
    print(
        "If device='cuda' and a tensor generation fails allocating a 4GB sequence block:"
    )
    print(
        "--> `fastrad` logs 'Falling back to CPU computation for this feature class.'"
    )
    print("--> VRAM is freed `torch.cuda.empty_cache()`")
    print(
        "--> The local feature is generated safely while the rest of the pipeline remains unaffected.\n"
    )


if __name__ == "__main__":
    main()
