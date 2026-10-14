import contextvars
from contextlib import contextmanager
from pathlib import Path

import torch

# Per-extraction memo for get_binned_image: five texture classes bin the same
# cropped image with the same settings. Active only inside
# FeatureExtractor.extract (see binning_cache), so results can never leak
# between extractions.
_BIN_CACHE: contextvars.ContextVar = contextvars.ContextVar("fastrad_bin_cache", default=None)


@contextmanager
def binning_cache():
    token = _BIN_CACHE.set({})
    try:
        yield
    finally:
        _BIN_CACHE.reset(token)

from .utils.dicom import parse_dicom_dir


class MedicalImage:
    """
    Representation of a continuous 3D medical volume (e.g., CT or MRI).
    
    Attributes:
        tensor (torch.Tensor): The 3D image array as a PyTorch FloatTensor.
        spacing (Tuple[float, float, float]): The physical voxel dimensions (z, y, x).
    """
    def __init__(self, tensor: torch.Tensor, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.tensor = tensor
        self.spacing = spacing

    def to(self, device: str | torch.device) -> "MedicalImage":
        """Moves the underlying tensor to the specified device."""
        return MedicalImage(self.tensor.to(device), self.spacing)

    @classmethod
    def from_dicom(cls, path: str | Path) -> "MedicalImage":
        """
        Creates a MedicalImage from a directory containing DICOM slices.
        
        Args:
            path: Path to the directory containing the DICOM files.
            
        """
        tensor, spacing = parse_dicom_dir(path)
        return cls(tensor=tensor, spacing=spacing)

    @classmethod
    def from_nifti(cls, path: str | Path) -> "MedicalImage":
        """
        Creates a MedicalImage from a NIfTI format image file (.nii or .nii.gz).
        
        Args:
            path: Path to the NIfTI file.
            
        Returns:
            A new instantiated MedicalImage object with extracted spacing.
        """
        import nibabel as nib
        import numpy as np
        nii = nib.load(str(path))
        data = nii.get_fdata().astype(np.float32)  # type: ignore
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        tensor = torch.from_numpy(data)
        
        header = nii.header
        zooms = header.get_zooms()  # type: ignore
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0])) if len(zooms) >= 3 else (1.0, 1.0, 1.0)
        
        return cls(tensor=tensor, spacing=spacing)

class Mask:
    """
    Representation of a binary 3D Region of Interest (ROI) mask.
    
    Attributes:
        tensor (torch.Tensor): The 3D binary mask as a uint8 tensor with values {0, 1}.
        spacing (Tuple[float, float, float]): The physical voxel dimensions (z, y, x).
    """
    def __init__(self, tensor: torch.Tensor, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        # Binary mask stored as uint8 {0, 1}: a quarter of the memory of float32,
        # so locating the ROI in a full CT volume reads 4x fewer bytes.
        self.tensor = (tensor > 0).to(torch.uint8)
        self.spacing = spacing

    def to(self, device: str | torch.device) -> "Mask":
        """Moves the underlying tensor to the specified device."""
        return Mask(self.tensor.to(device), self.spacing)

    @classmethod
    def from_dicom(cls, path: str | Path) -> "Mask":
        """
        Creates a binary Mask from a directory containing DICOM slices.
        Voxels strictly greater than 0 are set to 1.
        
        Args:
            path: Path to the directory containing the DICOM ROIs.
            
        """
        tensor, spacing = parse_dicom_dir(path)
        return cls(tensor=tensor, spacing=spacing)

    @classmethod
    def from_nifti(cls, path: str | Path) -> "Mask":
        """
        Creates a binary Mask from a NIfTI format image file (.nii or .nii.gz).
        Voxels strictly greater than 0 are set to 1.
        
        Args:
            path: Path to the NIfTI file.
            
        Returns:
            A new instantiated binary Mask object with extracted spacing.
        """
        import nibabel as nib
        import numpy as np
        nii = nib.load(str(path))
        data = nii.get_fdata().astype(np.float32)  # type: ignore
        # Reorder to (z, y, x) since nibabel loads as (x, y, z) typically, but 
        # typically radiomics handles it. PyRadiomics expects specific orientation.
        # usually z is the last axis in biological nifti, let's permute to match PyTorch (D, H, W).
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        tensor = torch.from_numpy(data)
        
        header = nii.header
        zooms = header.get_zooms()  # type: ignore
        # zooms is (x, y, z) typically, so spacing should be (z, y, x)
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0])) if len(zooms) >= 3 else (1.0, 1.0, 1.0)
        
        return cls(tensor=tensor, spacing=spacing)

def get_binned_image(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, bin_width: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes PyRadiomics-compatible binning.
    
    1. Floor divisions of pixels by bin_width anchored at minimum
    2. Maps the unique present bins within the mask to continuous integers 1..Ng
       (Empty intermediate bins are removed)
    
    Returns:
        binned_image (torch.Tensor): Remapped binned image
        Ng (torch.Tensor): The array of unique raw bin values present in the mask (ivector)
    """
    cache = _BIN_CACHE.get()
    if cache is not None:
        key = (id(image_tensor), id(mask_tensor), float(bin_width))
        hit = cache.get(key)
        if hit is not None and hit[0] is image_tensor and hit[1] is mask_tensor:
            return hit[2]
        result = _get_binned_image(image_tensor, mask_tensor, bin_width)
        cache[key] = (image_tensor, mask_tensor, result)
        return result
    return _get_binned_image(image_tensor, mask_tensor, bin_width)


def _get_binned_image(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, bin_width: float) -> tuple[torch.Tensor, torch.Tensor]:
    voxels = image_tensor[mask_tensor > 0.5]
    if voxels.numel() == 0:
        return torch.zeros_like(image_tensor), torch.empty(0, dtype=torch.float64, device=image_tensor.device)
        
    img_min = torch.min(voxels)
    minimum_binned = torch.floor(img_min / bin_width) * bin_width
    
    # Initial raw absolute binning
    raw_binned = torch.floor((image_tensor - minimum_binned) / bin_width) + 1
    raw_binned_voxels = raw_binned[mask_tensor > 0.5]
    
    # Find unique bins actually present in the mask
    unique_bins = torch.unique(raw_binned_voxels)
    Ng = unique_bins.numel()
    
    if Ng == 0:
        return torch.zeros_like(image_tensor), unique_bins.to(torch.float64)
        
    
    # We maintain raw values because NGTDM mathematical filtering requires actual relative offset intervals
    binned_image = raw_binned.to(torch.float32)
    
    return binned_image, unique_bins.to(torch.float64)

