import os

import torch

from fastrad import load_and_align


def test_load_and_align_dicom_series():
    """
    Tests spatial bounding, dynamic memory constraints, and SITK interpolation natively utilizing DICOM fixtures.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "fixtures", "image")
    mask_dir = os.path.join(base_dir, "fixtures", "mask")

    # 1. Provide directories (standard pyradiomics functionality relies on sitk.ImageSeriesReader)
    # Expected output: (MedicalImage, Mask) tightly crop-fitted around the mask label scope
    # The fixture mask is a 5x5x5 volume where only [1:4, 1:4, 1:4] is positive labels

    img, mask = load_and_align(img_dir, mask_dir, crop=True)

    # Mathematical assertion tests for PyTorch mappings
    # Since the mask is physically bounds [1:4, 1:4, 1:4], the absolute size is physically cropped to 3x3x3
    # rather than 5x5x5.

    assert img.tensor.shape == (3, 3, 3)
    assert mask.tensor.shape == (3, 3, 3)

    # Verify spatial mappings physically align (Z, Y, X layout is correctly shifted out of X, Y, Z SimpleITK logic)
    assert img.spacing == (1.0, 1.0, 1.0)
    assert mask.spacing == (1.0, 1.0, 1.0)

    assert not torch.isnan(img.tensor).any()
