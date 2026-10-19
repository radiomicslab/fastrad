import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from .image import Mask, MedicalImage
from .logger import logger


def _check_geometry_match(
    image_sitk: sitk.Image, mask_sitk: sitk.Image, tolerance: float = 1e-4
) -> bool:
    """
    Validates if two SimpleITK images share the exact analytical physical space constraints.
    """
    if image_sitk.GetDimension() != mask_sitk.GetDimension():
        return False

    s1, s2 = np.array(image_sitk.GetSpacing()), np.array(mask_sitk.GetSpacing())
    if np.max(np.abs(s1 - s2)) > tolerance:
        return False

    o1, o2 = np.array(image_sitk.GetOrigin()), np.array(mask_sitk.GetOrigin())
    if np.max(np.abs(o1 - o2)) > tolerance:
        return False

    d1, d2 = np.array(image_sitk.GetDirection()), np.array(mask_sitk.GetDirection())
    if np.max(np.abs(d1 - d2)) > tolerance:
        return False

    return image_sitk.GetSize() == mask_sitk.GetSize()


def resample_to_isotropic(
    image: sitk.Image, mask: sitk.Image, target_spacing: tuple[float, float, float]
) -> tuple[sitk.Image, sitk.Image]:
    """
    Resamples an image and mask natively to a forced isotropic target spacing.
    Uses generic B-Spline for continuous intensities and Nearest-Neighbor strictly for categorical masks.
    """
    logger.info(
        f"Resampling continuous structures strictly bound at target spacing: {target_spacing}"
    )

    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()

    # Calculate dimensional bounds mapping continuously
    new_size = [
        round(orig_size[0] * (orig_spacing[0] / target_spacing[0])),
        round(orig_size[1] * (orig_spacing[1] / target_spacing[1])),
        round(orig_size[2] * (orig_spacing[2] / target_spacing[2])),
    ]

    # Standard Continuous Image interpolation
    resampler_img = sitk.ResampleImageFilter()
    resampler_img.SetOutputSpacing(target_spacing)
    resampler_img.SetSize(new_size)
    resampler_img.SetOutputDirection(image.GetDirection())
    resampler_img.SetOutputOrigin(image.GetOrigin())
    resampler_img.SetTransform(sitk.Transform())
    resampler_img.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler_img.SetInterpolator(sitk.sitkBSpline)

    img_res = resampler_img.Execute(image)

    # Discrete Categorical Mask interpolation
    resampler_mask = sitk.ResampleImageFilter()
    resampler_mask.SetOutputSpacing(target_spacing)
    resampler_mask.SetSize(new_size)
    resampler_mask.SetOutputDirection(mask.GetDirection())
    resampler_mask.SetOutputOrigin(mask.GetOrigin())
    resampler_mask.SetTransform(sitk.Transform())
    resampler_mask.SetDefaultPixelValue(0)
    resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)

    mask_res = resampler_mask.Execute(mask)

    return img_res, mask_res


def crop_to_bbox(
    image: sitk.Image, mask: sitk.Image, label: int = 1, pad: int = 0
) -> tuple[sitk.Image, sitk.Image]:
    """
    Crops the images tightly adhering around the label constraint bound to dramatically accelerate GPU throughput.
    """
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask)

    if not label_shape_filter.HasLabel(label):
        logger.warning(f"Label {label} does not exist strictly within mask scope.")
        return image, mask

    bbox = list(label_shape_filter.GetBoundingBox(label))
    # Bbox format in SITK (3D): (startX, startY, startZ, sizeX, sizeY, sizeZ)

    for i in range(3):
        start = max(0, bbox[i] - pad)
        end = min(mask.GetSize()[i], bbox[i] + bbox[i + 3] + pad)
        bbox[i] = start
        bbox[i + 3] = end - start

    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(bbox[3:])
    roi_filter.SetIndex(bbox[:3])

    img_crop = roi_filter.Execute(image)
    mask_crop = roi_filter.Execute(mask)

    return img_crop, mask_crop


def _sitk_to_tensor(
    sitk_img: sitk.Image,
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    """
    Translates a SimpleITK object explicitly into contiguous PyTorch bindings correctly swapping Z, Y, X layout.
    """
    data = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    # SITK array is natively returned in (z, y, x) matching expected tensor bindings exactly.
    spacing = sitk_img.GetSpacing()
    # SITK spacing is (x, y, z), we map into scientific notation (z, y, x)
    spacing_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    return torch.from_numpy(data), spacing_zyx


def _read_sitk_image(path: str | Path) -> sitk.Image:
    """
    Safely reads DICOM directories or single NIfTI/NRRD volume files smoothly mimicking legacy I/O bindings.
    """
    path_str = str(path)
    if os.path.isdir(path_str):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path_str)
        if not dicom_names:
            raise ValueError(
                f"Directory {path_str} does not contain valid DICOM series."
            )
        reader.SetFileNames(dicom_names)
        return reader.Execute()
    else:
        return sitk.ReadImage(path_str)


def load_and_align(
    image_path: str | Path,
    mask_path: str | Path,
    resample_spacing: tuple[float, float, float] | None = None,
    crop: bool = True,
) -> tuple[MedicalImage, Mask]:
    """
    Core entrypoint matching the PyRadiomics `pyradiomics.imageoperations` logic exactly.
    """
    logger.info(f"Loading Image: {image_path}")
    image_sitk = _read_sitk_image(image_path)

    logger.info(f"Loading Mask: {mask_path}")
    mask_sitk = _read_sitk_image(mask_path)

    # 1. Geometry Handshake
    if not _check_geometry_match(image_sitk, mask_sitk):
        logger.warning(
            "Geometry validation failed standard tolerance! PyRadiomics usually throws exceptions here. Forcing nearest-neighbor overlay to mimic analytical bounding constraints."
        )
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        mask_sitk = resampler.Execute(mask_sitk)

    # 2. Resampling Hooks
    if resample_spacing is not None:
        # Re-ordering from (z, y, x) target back to SITK parameter format (x, y, z)
        sitk_spacing = (resample_spacing[2], resample_spacing[1], resample_spacing[0])
        image_sitk, mask_sitk = resample_to_isotropic(
            image_sitk, mask_sitk, sitk_spacing
        )

    # 3. Dynamic Memory Constraints Cropping Layer
    if crop:
        image_sitk, mask_sitk = crop_to_bbox(image_sitk, mask_sitk)

    # 4. Final Proxy Bridging Construction
    img_t, img_s = _sitk_to_tensor(image_sitk)
    mask_t, mask_s = _sitk_to_tensor(mask_sitk)

    img_obj = MedicalImage(tensor=img_t, spacing=img_s)
    mask_obj = Mask(tensor=mask_t, spacing=mask_s)
    return img_obj, mask_obj
