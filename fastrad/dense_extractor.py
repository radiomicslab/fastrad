import torch

from .extractor import _FEATURE_MAP, FeatureExtractor
from .image import Mask, MedicalImage
from .logger import logger


class DenseFeatureExtractor(FeatureExtractor):
    """
    Subclass of FeatureExtractor that natively outputs dense, voxel-wise feature maps
    using sliding 3D window memory-strided patch views.
    """

    def extract_dense(
        self,
        image: MedicalImage,
        mask: Mask,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
    ) -> dict[str, torch.Tensor]:
        """
        Executes dense feature extraction on the given Image and Mask.

        Args:
            image: Baseline medical volume.
            mask: Binary ROIs. Only windows containing positive mask voxels are computed.
            kernel_size: 3D window dimensions (z, y, x).
            stride: Step size for window extraction.

        Returns:
            Dict mapping feature names to dense torch.Tensor feature maps.
            Output shape is (Dz_out, Dy_out, Dx_out), matching the sliding window grid.
            Voxels with no valid input mask elements are set to NaN.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        # Move tensors to the requested device
        img_tensor = image.tensor.to(self.device)
        mask_tensor = mask.tensor.to(self.device)

        if img_tensor.shape != mask_tensor.shape:
            raise ValueError(
                f"Image and mask shape mismatch: {img_tensor.shape} != {mask_tensor.shape}"
            )

        # Update spacing implicitly for downstream routines
        self.settings.spacing = image.spacing

        kz, ky, kx = kernel_size
        sz, sy, sx = stride
        Dz, Dy, Dx = img_tensor.shape

        out_z = (Dz - kz) // sz + 1
        out_y = (Dy - ky) // sy + 1
        out_x = (Dx - kx) // sx + 1

        if out_z <= 0 or out_y <= 0 or out_x <= 0:
            raise ValueError(
                "Kernel size larger than tensor dimensions or invalid stride/kernel combination."
            )

        # PyTorch unfold extracts memory-strided patch views without copying memory
        img_patches = img_tensor.unfold(0, kz, sz).unfold(1, ky, sy).unfold(2, kx, sx)
        mask_patches = mask_tensor.unfold(0, kz, sz).unfold(1, ky, sy).unfold(2, kx, sx)

        dense_features = {}

        # Fallback to CPU for specific routines if OOM happens
        cpu_fallback_needed = set()

        # Compute features sequentially over patches.
        # Future architecture changes can vectorize logic completely.
        for zi in range(out_z):
            for yi in range(out_y):
                for xi in range(out_x):
                    p_img = img_patches[zi, yi, xi]
                    p_mask = mask_patches[zi, yi, xi]

                    # Only compute where the mask subset has valid voxels.
                    if p_mask.sum().item() == 0:
                        continue

                    for feature_class in self.settings.feature_classes:
                        if feature_class not in _FEATURE_MAP:
                            raise ValueError(f"Unknown feature class: {feature_class}")

                        compute_fn = _FEATURE_MAP[feature_class]

                        try:
                            # Standard pathway inside try
                            if feature_class in cpu_fallback_needed:
                                f_vals = compute_fn(
                                    p_img.cpu(), p_mask.cpu(), self.settings
                                )
                            else:
                                f_vals = compute_fn(p_img, p_mask, self.settings)

                        except torch.cuda.OutOfMemoryError:
                            if self.device == "cuda":
                                logger.warning(
                                    f"CUDA OutOfMemoryError caught for {feature_class} in patch {zi},{yi},{xi}. "
                                    f"Falling back to CPU computation."
                                )
                                torch.cuda.empty_cache()
                                cpu_fallback_needed.add(feature_class)
                                f_vals = compute_fn(
                                    p_img.cpu(), p_mask.cpu(), self.settings
                                )
                            else:
                                raise
                        except Exception as e:
                            logger.error(
                                f"Failed patch extraction at {zi},{yi},{xi} for {feature_class}: {e}"
                            )
                            continue

                        if f_vals:
                            for k, v in f_vals.items():
                                if k not in dense_features:
                                    dense_features[k] = torch.full(
                                        (out_z, out_y, out_x),
                                        float("nan"),
                                        device=self.device,
                                    )

                                dense_features[k][zi, yi, xi] = v

        return dense_features
