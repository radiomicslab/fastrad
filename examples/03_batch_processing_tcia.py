"""
Example 03: TCIA Clinical Cohort Batch Processing
=================================================

This example demonstrates a simulated clinical pipeline integrating `fastrad`
over a large cohort of patients. Instead of processing images sequentially on the CPU
(bottlenecking processing for days), `fastrad` utilizes standard high-throughput models.

This simulates evaluating 10 distinct patient IDs from the RIDER Lung CT dataset.
"""

import time

import torch

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


def load_simulated_tcia_patient(patient_id: str):
    """
    In a real scenario, you'd use pydicom and SimpleITK to query your local PACS
    or TCIA directory structure to pull the DICOM slices and specific RTStruct ROI.
    """
    # Using 64x64x64 generic patches matching roughly 10cm physical bounds.
    tensor = torch.rand((64, 64, 64)) * 1000
    image = MedicalImage(tensor=tensor, spacing=(1.0, 1.0, 1.0))

    # Spherical simulated lesion, unique random radius per patient between 10mm to 20mm
    tumor_radius = torch.randint(low=10, high=20, size=(1,)).item()
    mask_tensor = torch.zeros((64, 64, 64))
    z, y, x = torch.meshgrid(
        torch.arange(64), torch.arange(64), torch.arange(64), indexing="ij"
    )
    # Vary the absolute location to ensure shift invariance and robustness bounds
    c_z, c_y, c_x = torch.randint(low=20, high=40, size=(3,)).numpy()

    distance = torch.sqrt((z - c_z) ** 2 + (y - c_y) ** 2 + (x - c_x) ** 2)
    mask_tensor[distance <= tumor_radius] = 1

    mask = Mask(tensor=mask_tensor, spacing=(1.0, 1.0, 1.0))
    return image, mask


def main():
    print("Initializing Multi-Patient TCIA Batch Processor...")

    # 1. Define typical patient cohort mapping list
    tcia_patient_ids = [f"RIDER-{str(i).zfill(3)}" for i in range(1, 11)]

    # 2. Extract standard settings (GLCM, GLRLM heavily requested in clinical models)
    # Mapping to AUTO defaults to CUDA if present, falling automatically back to CPU
    settings = FeatureSettings(
        feature_classes=["firstorder", "shape", "glcm", "glrlm"],
        bin_width=25.0,
        device="auto",
    )

    # We initialize the extractor ONLY ONCE. The PyTorch compute graphs
    # will cleanly stream inputs effectively.
    extractor = FeatureExtractor(settings)

    # 3. Process Patients
    patient_feature_vectors = {}
    total_start = time.time()

    for pid in tcia_patient_ids:
        print(f"Processing Patient Node: {pid}...")

        # Load Volume directly into memory
        image, mask = load_simulated_tcia_patient(pid)

        # Execute extraction and save mapping
        features = extractor.extract(image, mask)
        patient_feature_vectors[pid] = features
        print(f"  -> Extracted {len(features)} attributes.")

    total_time = time.time() - total_start
    print(
        f"\nCohort Processing Complete! 10 Patients Iterated in {total_time:.4f} seconds."
    )
    print(f"Average Extract-per-patient Time: {total_time / 10:.4f} seconds.")


if __name__ == "__main__":
    main()
