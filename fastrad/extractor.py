import warnings

import torch

from .image import Mask, MedicalImage, binning_cache
from .logger import logger
from .settings import FeatureSettings
from .utils.device import resolve_device

# Filter the isotropic spacing warning so it only appears once per runtime.
warnings.filterwarnings("once", message=".*is not isotropic.*")

from .features import (
    compute_firstorder,
    compute_glcm,
    compute_gldm,
    compute_glrlm,
    compute_glszm,
    compute_ngtdm,
    compute_shape,
    compute_shape2d,
)

# Voxels kept around the ROI bounding box when cropping. Texture classes look at
# neighbours at distance 1, so one voxel of (masked-out) context is enough for
# every texture matrix to be identical to the uncropped computation. Shape
# features re-crop to the exact bounding box themselves (PyRadiomics' frame).
ROI_PAD = 1


def _crop_to_roi(img: torch.Tensor, mask: torch.Tensor, pad: int = ROI_PAD):
    """Crops image and mask to the mask's bounding box plus `pad` voxels.

    Done on the tensors' current device (normally the CPU), before anything is
    sent to the GPU, so only the region around the ROI is transferred and
    processed. Returns the cropped tensors and the ROI voxel count.
    """
    # Narrow one axis at a time: only the first projection reads the whole
    # volume; the next ones read the already-narrowed slab. Mask values are
    # {0, 1} (see Mask), so a max-projection > 0.5 marks ROI slices.
    lo, hi = [0] * mask.ndim, list(mask.shape)
    for dim in range(mask.ndim):
        sub = mask[tuple(slice(l, h) for l, h in zip(lo, hi))]
        proj = sub.movedim(dim, 0).reshape(sub.shape[dim], -1).amax(dim=1) > 0.5
        idx = torch.nonzero(proj).flatten()
        if idx.numel() == 0:
            return img, mask, 0
        first, last = lo[dim] + int(idx[0]), lo[dim] + int(idx[-1])
        lo[dim], hi[dim] = first, last + 1
    lo = [max(l - pad, 0) for l in lo]
    hi = [min(h + pad, n) for h, n in zip(hi, mask.shape)]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    mask_c = mask[sl]
    return img[sl], mask_c, int((mask_c > 0.5).sum())


def _to_python_floats(features: dict) -> dict:
    """Converts all feature values to Python floats with ONE device->host copy.

    Calling `.item()` on each of ~107 GPU scalars forces ~107 separate
    synchronisations; stacking them first needs one.
    """
    tensor_keys = [k for k, v in features.items() if isinstance(v, torch.Tensor)]
    out = {k: (float(v) if not isinstance(v, torch.Tensor) else None) for k, v in features.items()}
    by_device: dict = {}
    for k in tensor_keys:
        by_device.setdefault(features[k].device, []).append(k)
    for keys in by_device.values():
        vals = torch.stack([features[k].detach().reshape(()).to(torch.float64) for k in keys]).cpu().tolist()
        out.update(zip(keys, vals))
    return out


_FEATURE_MAP = {
    "firstorder": compute_firstorder,
    "shape": compute_shape,
    "shape2d": compute_shape2d,
    "glcm": compute_glcm,
    "glrlm": compute_glrlm,
    "glszm": compute_glszm,
    "gldm": compute_gldm,
    "ngtdm": compute_ngtdm
}

class FeatureExtractor:
    """
    Main orchestration engine for radiomics feature extraction.
    
    The FeatureExtractor consumes a FeatureSettings configuration and executes
    the specified feature class modules against a provided image and mask. 
    It automatically routes tensors to the requested device (CPU, CUDA, MPS)
    and handles OutOfMemory fallbacks gracefully.
    """
    def __init__(self, settings: FeatureSettings):
        """
        Initializes the FeatureExtractor with the given settings.
        
        Args:
            settings (FeatureSettings): Configuration defining which features to compute.
        """
        self.settings = settings
        self.device = resolve_device(settings.device)

    def extract(self, image: MedicalImage, mask: Mask) -> dict[str, float]:
        """
        Executes feature extraction on the given Image and Mask.
        
        Args:
            image (MedicalImage): The baseline medical volume.
            mask (Mask): The binary Region of Interest mask.
            
        Returns:
            dict[str, float]: A dictionary mapping feature names to their computed values.
        """
        # Ensure image and mask are same shape
        if image.tensor.shape != mask.tensor.shape:
            raise ValueError(f"Image shape {image.tensor.shape} and mask shape {mask.tensor.shape} do not match.")

        # Crop to the ROI before moving anything to the device: only the
        # bounding box (plus a 1-voxel margin) is transferred and processed.
        img_crop, mask_crop, mask_sum = _crop_to_roi(image.tensor, mask.tensor)
        non_blocking = self.device == "cuda"
        img_tensor = img_crop.to(self.device, non_blocking=non_blocking)
        mask_tensor = mask_crop.to(self.device, non_blocking=non_blocking)

        # Populate spacing for volume calculations
        self.settings.spacing = image.spacing

        # Robustness Checks
        
        # 1. Empty Mask Check
        if mask_sum == 0:
            raise ValueError("Mask contains no positive voxels.")
            
        # 2. Single-Voxel ROI Check
        if mask_sum == 1:
            logger.warning(
                "Mask contains exactly one positive voxel. "
                "Many spatial and textural features cannot be computed validly."
            )
            
        # 3. Non-Isotropic Spacing Check
        sp = image.spacing
        if max(sp) - min(sp) > 1e-3:
            warnings.warn(
                f"Image spacing {sp} is not isotropic. "
                "PyRadiomics guidelines recommend resampling to isotropic spacing "
                "for robust textural feature calculation.",
                UserWarning,
                stacklevel=2
            )

        # Texture classes share one grey-level binning per extraction.
        with binning_cache():
            features = self._compute_classes(img_tensor, mask_tensor)

        # Enforce PyRadiomics float backward-compatibility natively,
        # with a single device->host transfer for all features.
        if not self.settings.differentiable:
            features = _to_python_floats(features)

        return features

    def _compute_classes(self, img_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> dict:
        features = {}
        for feature_class in self.settings.feature_classes:
            if feature_class not in _FEATURE_MAP:
                raise ValueError(f"Unknown feature class: {feature_class}")
            
            compute_fn = _FEATURE_MAP[feature_class]
            
            device_type = "cuda" if self.device == "cuda" else ("cpu" if self.device == "cpu" else None)
            
            try:
                if self.settings.amp and device_type:
                    with torch.autocast(device_type=device_type):
                        class_features = compute_fn(
                            image_tensor=img_tensor,
                            mask_tensor=mask_tensor,
                            settings=self.settings
                        )
                else:
                    class_features = compute_fn(
                        image_tensor=img_tensor,
                        mask_tensor=mask_tensor,
                        settings=self.settings
                    )
            except torch.cuda.OutOfMemoryError:
                if self.device == "cuda":
                    logger.warning(
                        f"CUDA OutOfMemoryError caught while extracting {feature_class} features. "
                        f"Falling back to CPU computation for this feature class."
                    )
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    # Move tensors to CPU explicitly for this computation
                    cpu_img_tensor = img_tensor.cpu()
                    cpu_mask_tensor = mask_tensor.cpu()
                    
                    # Compute on CPU
                    if self.settings.amp:
                        with torch.autocast(device_type="cpu"):
                            class_features = compute_fn(
                                image_tensor=cpu_img_tensor,
                                mask_tensor=cpu_mask_tensor,
                                settings=self.settings
                            )
                    else:
                        class_features = compute_fn(
                            image_tensor=cpu_img_tensor,
                            mask_tensor=cpu_mask_tensor,
                            settings=self.settings
                        )
                    
                    # Free up CPU memory
                    del cpu_img_tensor
                    del cpu_mask_tensor
                else:
                    raise
            
            features.update(class_features)
        return features
