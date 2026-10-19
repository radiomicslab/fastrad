import pytest
import torch

from fastrad import DenseFeatureExtractor, FeatureSettings, Mask, MedicalImage


def test_dense_extraction_shapes():
    # Construct a dummy 5x5x5 physical volume
    img_vol = torch.ones((5, 5, 5), dtype=torch.float32)
    # Mask out a specific area to ensure logic handles zeroes
    mask_vol = torch.ones((5, 5, 5), dtype=torch.float32)
    mask_vol[0, 0, 0] = 0

    med_img = MedicalImage(img_vol)
    med_mask = Mask(mask_vol)

    settings = FeatureSettings(feature_classes=["firstorder"], device="cpu")
    extractor = DenseFeatureExtractor(settings)

    # Kernel 3x3x3, Stride 1 -> Math: (5 - 3) // 1 + 1 = 3
    features = extractor.extract_dense(med_img, med_mask, kernel_size=3, stride=1)

    assert "firstorder:mean" in features
    mean_map = features["firstorder:mean"]
    assert mean_map.shape == (3, 3, 3)


def test_dense_stride_scaling():
    # Construct a larger dummy 10x10x10 volume
    img_vol = torch.randn((10, 10, 10), dtype=torch.float32)
    mask_vol = torch.ones((10, 10, 10), dtype=torch.float32)

    med_img = MedicalImage(img_vol)
    med_mask = Mask(mask_vol)

    settings = FeatureSettings(feature_classes=["firstorder"], device="cpu")
    extractor = DenseFeatureExtractor(settings)

    # Kernel 4, Stride 2 -> Math: (10 - 4) // 2 + 1 = 6 // 2 + 1 = 4
    features = extractor.extract_dense(med_img, med_mask, kernel_size=4, stride=2)

    assert "firstorder:energy" in features
    energy_map = features["firstorder:energy"]
    assert energy_map.shape == (4, 4, 4)


def test_dense_gpu_fallback():
    # This specifically checks tensor output is safely captured when testing dense logic
    img_vol = torch.ones((6, 6, 6), dtype=torch.float32)
    mask_vol = torch.ones((6, 6, 6), dtype=torch.float32)

    med_img = MedicalImage(img_vol)
    med_mask = Mask(mask_vol)

    settings = FeatureSettings(feature_classes=["firstorder"], device="cpu")
    extractor = DenseFeatureExtractor(settings)

    # Kernel 6x6x6, Stride 1 on 6x6x6 volume gives 1x1x1 output
    features = extractor.extract_dense(med_img, med_mask, kernel_size=6, stride=1)

    assert "firstorder:mean" in features
    assert features["firstorder:mean"].shape == (1, 1, 1)


def test_invalid_kernel():
    img_vol = torch.ones((5, 5, 5), dtype=torch.float32)
    mask_vol = torch.ones((5, 5, 5), dtype=torch.float32)

    settings = FeatureSettings(feature_classes=["firstorder"])
    extractor = DenseFeatureExtractor(settings)

    with pytest.raises(
        ValueError,
        match="Kernel size larger than tensor dimensions or invalid stride/kernel combination",
    ):
        extractor.extract_dense(
            MedicalImage(img_vol), Mask(mask_vol), kernel_size=6, stride=1
        )
