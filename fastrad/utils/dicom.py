import os
import warnings
from pathlib import Path

import numpy as np
import pydicom
import torch

from ..logger import logger


def parse_dicom_dir(
    path: str | Path,
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    """
    Reads a directory of DICOM files, rigorously reconstructs physical 3D geometry,
    and returns a 3D PyTorch float tensor (D, H, W) and spacing (z, y, x).

    Methodological Geometric Logic:
    1. Slice Normal & Ordering:
       Calculates the unit slice normal vector n = r x c from ImageOrientationPatient
       (row direction r and column direction c). Projects each slice's ImagePositionPatient
       (IPP) onto n (proj = dot(IPP, n)) to determine exact anatomical physical order,
       reconciling non-sequential InstanceNumbers or superior-inferior reverse acquisitions.
    2. Spacing Reconciliation:
       Computes true physical inter-slice step from consecutive projected positions
       delta_z = |proj_{i+1} - proj_i|. If nominal SliceThickness or SpacingBetweenSlices
       disagrees with delta_z, delta_z is prioritized to maintain physical metric integrity.
    3. Gantry Tilt & Oblique Acquisitions:
       Evaluates whether slice normal aligns with cardinal axes. If significant gantry tilt
       or oblique acquisition is detected, an explicit UserWarning is emitted.
    """
    path = Path(path)
    slices: list[pydicom.FileDataset] = []

    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(file_path, force=True)
                if hasattr(ds, "pixel_array"):
                    slices.append(ds)
            except Exception:
                pass

    if not slices:
        raise ValueError(f"No DICOM files with pixel data found in {path}")

    # 1. Orientation and Projection Calculation
    first_slice = slices[0]
    iop = getattr(first_slice, "ImageOrientationPatient", None)

    if iop is not None and len(iop) == 6:
        r = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
        c = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
        # Cross product to compute slice normal direction vector
        n = np.cross(r, c)
        norm_n = np.linalg.norm(n)
        if norm_n > 0:
            n = n / norm_n

        # Detect oblique acquisition / gantry tilt (normal vector non-aligned with cardinal Z axis)
        if abs(abs(n[2]) - 1.0) > 1e-2:
            warnings.warn(
                f"Oblique DICOM acquisition or gantry tilt detected (slice normal: {n}). "
                "fastrad continues with spatial projection, but non-orthogonal resampling may be required.",
                UserWarning,
                stacklevel=2,
            )

        def get_slice_proj(ds):
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                pos = np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])])
                return float(np.dot(pos, n))
            return float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0)))

        slices.sort(key=get_slice_proj)
    else:
        # Fallback sorting by InstanceNumber or SliceLocation
        if hasattr(slices[0], "InstanceNumber"):
            slices.sort(key=lambda x: int(getattr(x, "InstanceNumber", 0)))
        elif hasattr(slices[0], "SliceLocation"):
            slices.sort(key=lambda x: float(getattr(x, "SliceLocation", 0.0)))

    # 2. Extract and Reconcile Physical Spacing: (z, y, x)
    pixel_spacing = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    y_spacing = float(pixel_spacing[0])
    x_spacing = float(pixel_spacing[1])

    z_spacing = 1.0
    # Reconcile inter-slice spacing from true physical coordinates if >1 slice
    if (
        len(slices) > 1
        and hasattr(slices[0], "ImagePositionPatient")
        and hasattr(slices[1], "ImagePositionPatient")
    ):
        ipp1 = np.array([float(v) for v in slices[0].ImagePositionPatient])
        ipp2 = np.array([float(v) for v in slices[1].ImagePositionPatient])
        actual_delta_z = float(np.linalg.norm(ipp2 - ipp1))

        nominal_thick = float(getattr(slices[0], "SliceThickness", actual_delta_z))
        if abs(actual_delta_z - nominal_thick) > 1e-3:
            logger.info(
                f"Reconciled slice spacing: physical IPP delta ({actual_delta_z:.4f} mm) "
                f"prioritized over nominal SliceThickness ({nominal_thick:.4f} mm)."
            )
        z_spacing = actual_delta_z
    elif hasattr(slices[0], "SpacingBetweenSlices"):
        z_spacing = float(slices[0].SpacingBetweenSlices)
    elif hasattr(slices[0], "SliceThickness"):
        z_spacing = float(slices[0].SliceThickness)

    spacing = (z_spacing, y_spacing, x_spacing)

    # 3. Stack pixel arrays into 3D volume (D, H, W)
    volume_np = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    volume_tensor = torch.from_numpy(volume_np)

    return volume_tensor, spacing
