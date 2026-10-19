import math

import torch
import torch.nn.functional as F

from .image import MedicalImage
from .logger import logger


def _get_gaussian_kernel_3d(sigma: float, size: int = 0) -> torch.Tensor:
    """
    Constructs a discrete 3D Gaussian discrete matrix for smoothing.
    """
    if size == 0:
        # Standard PyRadiomics kernel bounding rule (radius = ceil(3 * sigma))
        radius = math.ceil(3.0 * sigma)
        size = 2 * radius + 1

    grid = torch.arange(size, dtype=torch.float32)
    grid = grid - (size - 1) / 2.0

    variance = sigma**2
    # 1D kernel
    gaussian_1d = torch.exp(-(grid**2) / (2 * variance))
    # Outer product for 3D
    gaussian_2d = torch.einsum("i,j->ij", gaussian_1d, gaussian_1d)
    gaussian_3d = torch.einsum("ij,k->ijk", gaussian_2d, gaussian_1d)

    gaussian_3d = gaussian_3d / torch.sum(gaussian_3d)
    return gaussian_3d


def _get_LoG_kernel_3d(sigma: float, size: int = 0) -> torch.Tensor:
    """
    Generates an analytical 3D Laplacian of Gaussian (LoG) spatial kernel matrices.
    """
    if size == 0:
        radius = math.ceil(3.0 * sigma)
        size = 2 * radius + 1

    grid = torch.arange(size, dtype=torch.float32)
    grid = grid - (size - 1) / 2.0

    # Generate 3D grid
    z, y, x = torch.meshgrid(grid, grid, grid, indexing="ij")
    squared_dist = x**2 + y**2 + z**2
    variance = sigma**2

    # Analytical LoG expression
    scaling = -1.0 / (math.pi * variance**2)
    norm_term = 1.0 - (squared_dist / (2 * variance))
    exponential = torch.exp(-squared_dist / (2 * variance))

    log_3d = scaling * norm_term * exponential

    # Ensure zero sum
    log_3d = log_3d - torch.mean(log_3d)
    return log_3d


def get_LoG_image(image: MedicalImage, sigmas: list[float]) -> dict[str, MedicalImage]:
    """
    Applies Laplacian of Gaussian (LoG) spatial filtration.
    Mimics `pyradiomics.imageoperations.getLoGImage`.
    Yields dictionary targeting specific string namespaces e.g., 'log-sigma-1-0-mm-3D'.
    """
    logger.info(
        f"Applying analytical Laplacian of Gaussian kernels for sigmas: {sigmas}"
    )

    # Requires batched layout: (1, 1, Z, Y, X)
    tensor = image.tensor.unsqueeze(0).unsqueeze(0)
    filtered_images = {}

    for sigma in sigmas:
        kernel = _get_LoG_kernel_3d(sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(tensor.device)

        padding = kernel.shape[-1] // 2
        filtered_tensor = F.conv3d(tensor, kernel, padding=padding)
        filtered_tensor = filtered_tensor.squeeze(0).squeeze(0)

        name = f"log-sigma-{str(sigma).replace('.', '-炎')}-mm-3D"
        # PyRadiomics has a specific naming convention: 'log-sigma-1-0-mm-3D'
        name = name.replace("-炎", "-")
        filtered_images[name] = MedicalImage(
            tensor=filtered_tensor, spacing=image.spacing
        )

    return filtered_images


def get_Square_image(image: MedicalImage) -> dict[str, MedicalImage]:
    tensor = torch.square(image.tensor)
    return {"square": MedicalImage(tensor=tensor, spacing=image.spacing)}


def get_SquareRoot_image(image: MedicalImage) -> dict[str, MedicalImage]:
    tensor = torch.sqrt(torch.abs(image.tensor))
    return {"squareroot": MedicalImage(tensor=tensor, spacing=image.spacing)}


def get_Logarithm_image(image: MedicalImage) -> dict[str, MedicalImage]:
    tensor = torch.log(torch.abs(image.tensor) + 1e-6)
    return {"logarithm": MedicalImage(tensor=tensor, spacing=image.spacing)}


def get_Exponential_image(image: MedicalImage) -> dict[str, MedicalImage]:
    tensor = torch.exp(image.tensor)
    return {"exponential": MedicalImage(tensor=tensor, spacing=image.spacing)}


def apply_builtin_filters(
    image: MedicalImage, filter_types: dict[str, dict]
) -> dict[str, MedicalImage]:
    """
    Master router handling multiple simultaneous math mappings to mirror legacy automated scaling.
    Argument format mimics standard PyRadiomics filter definition dictionaries:
    e.g. {"Original": {}, "LoG": {"sigma": [1.0, 2.0]}, "Square": {}}
    """
    mapped_images = {}

    for key, params in filter_types.items():
        if key.lower() == "original":
            mapped_images["original"] = image
        elif key.lower() == "log":
            sigmas = params.get("sigma", [1.0])
            if not isinstance(sigmas, list):
                sigmas = [sigmas]
            out = get_LoG_image(image, sigmas)
            mapped_images.update(out)
        elif key.lower() == "square":
            mapped_images.update(get_Square_image(image))
        elif key.lower() == "squareroot":
            mapped_images.update(get_SquareRoot_image(image))
        elif key.lower() == "logarithm":
            mapped_images.update(get_Logarithm_image(image))
        elif key.lower() == "exponential":
            mapped_images.update(get_Exponential_image(image))
        else:
            logger.warning(
                f"Filter type {key} is currently unsupported natively via analytical scaling."
            )

    return mapped_images
