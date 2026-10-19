import os

from fastrad import MedicalImage, apply_builtin_filters


def test_builtin_filters():
    """
    Validates logical application of structural transformations exactly matching PyRadiomics parameterization keys.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "fixtures", "image")

    img = MedicalImage.from_dicom(img_dir)
    original_shape = img.tensor.shape

    params = {
        "Original": {},
        "LoG": {"sigma": [1.0, 3.0]},
        "Square": {},
        "Logarithm": {},
        "Exponential": {},
    }

    # 1. Pipeline Execution
    filtered_images = apply_builtin_filters(img, params)

    # Expected specific keys exactly matching PyRadiomics architecture hooks
    expected_keys = [
        "original",
        "log-sigma-1-0-mm-3D",
        "log-sigma-3-0-mm-3D",
        "square",
        "logarithm",
        "exponential",
    ]

    for key in expected_keys:
        assert key in filtered_images
        # Verify boundary coordinates mapped identically across filters bypassing GPU padding shifting
        assert filtered_images[key].tensor.shape == original_shape
        assert filtered_images[key].spacing == img.spacing
