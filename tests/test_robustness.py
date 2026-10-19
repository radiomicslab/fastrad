import pytest
import torch

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage


@pytest.fixture
def base_settings():
    return FeatureSettings(
        feature_classes=[
            "firstorder",
            "shape",
            "glcm",
            "glrlm",
            "glszm",
            "gldm",
            "ngtdm",
        ],
        bin_width=25,
        device="cpu",
    )


def test_empty_mask(base_settings):
    """Test that an empty mask raises a ValueError (PyRadiomics parity)."""
    image_tensor = torch.rand(10, 10, 10)
    mask_tensor = torch.zeros(10, 10, 10)

    image = MedicalImage(tensor=image_tensor)
    mask = Mask(tensor=mask_tensor)
    extractor = FeatureExtractor(base_settings)

    with pytest.raises(ValueError, match="Mask contains no positive voxels"):
        extractor.extract(image, mask)


class LogCapture:
    def __init__(self):
        self.messages = []

    def write(self, message):
        self.messages.append(message)


def test_single_voxel_mask(base_settings):
    """Test that a single voxel mask evaluates without crashing, though features may be sparse/nan."""
    image_tensor = torch.rand(10, 10, 10)
    mask_tensor = torch.zeros(10, 10, 10)
    mask_tensor[5, 5, 5] = 1.0  # Single voxel

    image = MedicalImage(tensor=image_tensor)
    mask = Mask(tensor=mask_tensor)
    extractor = FeatureExtractor(base_settings)

    from fastrad.logger import logger

    capture = LogCapture()
    handler_id = logger.add(capture.write)

    features = extractor.extract(image, mask)

    logger.remove(handler_id)

    assert "exactly one positive voxel" in "".join(capture.messages)
    # We do not assert exact features here as they degrade gracefully (some may be nan or empty based on the module)
    assert isinstance(features, dict)


def test_non_isotropic_spacing(base_settings):
    """Test that non-isotropic spacing triggers a warning."""
    image_tensor = torch.rand(10, 10, 10)
    mask_tensor = torch.ones(10, 10, 10)

    # Highly non-isotropic spacing
    image = MedicalImage(tensor=image_tensor, spacing=(1.0, 1.0, 5.0))
    mask = Mask(tensor=mask_tensor, spacing=(1.0, 1.0, 5.0))

    # We only need one feature class to trigger the loop and general validation
    settings = FeatureSettings(
        feature_classes=["firstorder"], bin_width=25, device="cpu"
    )
    extractor = FeatureExtractor(settings)

    with pytest.warns(UserWarning, match="is not isotropic"):
        extractor.extract(image, mask)
