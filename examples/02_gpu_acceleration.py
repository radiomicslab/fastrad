"""
Example 02: CUDA GPU Acceleration
=================================

A critical advantage of `fastrad` over standard radiomics libraries is the native ability
to map feature extraction pipelines onto GPUs utilizing PyTorch device routing.
This script benchmarks the acceleration of shifting computation to the GPU.

Requirements:
- pip install fastrad torch
- A machine with a CUDA-enabled NVIDIA GPU.
"""

import time

import torch

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


def create_large_synthetic_volume():
    """Generates a large 128x128x128 synthetic medical dataset for benchmarking."""
    print(
        "Generating large 128^3 synthetic medical volume corresponding to a typical structural ROI..."
    )

    # Simulate a CT scan range (-1000 to 1000 HU)
    tensor = (torch.rand((128, 128, 128)) * 2000) - 1000
    image = MedicalImage(tensor=tensor, spacing=(1.0, 1.0, 1.0))

    # Create a 30mm radius spherical mask (approximately 113,000 positive voxels)
    mask_tensor = torch.zeros((128, 128, 128))
    z, y, x = torch.meshgrid(
        torch.arange(128), torch.arange(128), torch.arange(128), indexing="ij"
    )
    distance = torch.sqrt((z - 64) ** 2 + (y - 64) ** 2 + (x - 64) ** 2)
    mask_tensor[distance <= 30] = 1
    mask = Mask(tensor=mask_tensor, spacing=(1.0, 1.0, 1.0))

    return image, mask


def run_extraction(device: str, image: MedicalImage, mask: Mask):
    """Executes the complete extraction pipeline on a specific hardware device."""
    settings = FeatureSettings(
        feature_classes=[
            "firstorder",
            "shape",
            "glcm",
            "glrlm",
            "glszm",
            "gldm",
            "ngtdm",
        ],
        bin_width=25.0,
        device=device,
    )

    extractor = FeatureExtractor(settings)

    print(f"\n--- Extracting Features on: {device.upper()} ---")
    start_time = time.time()

    # The extractor automatically moves internal operations to the targeted hardware
    features = extractor.extract(image, mask)

    duration = time.time() - start_time
    print(f"Extraction Completed. Extracted {len(features)} total features.")
    print(f"Total Computation Latency: {duration:.4f} seconds")

    return duration


def main():
    if not torch.cuda.is_available():
        print("CRITICAL WARNING: No CUDA-capable GPU found on this system.")
        print("The `fastrad` device parameter will automatically map back to the CPU.")
        print(
            "To see true parallel acceleration profiling, run on Nvidia-based environments.\n"
        )

    # Create the data
    image, mask = create_large_synthetic_volume()

    # 1. Warm-up and run CPU baseline
    # Single-thread CPU iteration
    # PyTorch defaults to multi-threading on CPU, but fastrad handles the volume processing internally
    cpu_duration = run_extraction(device="cpu", image=image, mask=mask)

    # 2. Run CUDA baseline
    if torch.cuda.is_available():
        # It's common practice to execute a 'warmup' iteration on GPUs prior to profiling
        # to account for initial kernel loading and memory allocation delays.
        print("\nWarming up CUDA engine...")
        _ = run_extraction(device="cuda", image=image, mask=mask)

        # Actual recorded profiling
        gpu_duration = run_extraction(device="cuda", image=image, mask=mask)

        speedup = cpu_duration / gpu_duration
        print("\n=======================================================")
        print(f"CUDA OVERALL SPEEDUP: {speedup:.2f}x faster vs single CPU extraction")
        print("=======================================================\n")
    else:
        print("\nSkipping CUDA comparison (Hardware not detected).")


if __name__ == "__main__":
    main()
