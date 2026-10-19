import torch

from fastrad.extractor import FeatureExtractor
from fastrad.image import Mask, MedicalImage
from fastrad.settings import FeatureSettings

from .logger import logger


class VoxelFeatureExtractor:
    """
    Experimental sliding-window feature extractor for voxel-wise radiomic maps.

    WARNING: Native 3D patch extraction memory consumption grows factorially.
    This calculates standard macro radiomics iteratively over spatial bounding boxes.
    It is recommended solely for small Region of Interests or with powerful GPU limits.
    """

    def __init__(self, settings: FeatureSettings, kernel_size: int = 3):
        self.settings = settings
        self.extractor = FeatureExtractor(settings)
        # Ensure kernel size is odd for symmetric padding
        if kernel_size % 2 == 0:
            raise ValueError("Voxel kernel size must be an odd integer (e.g., 3, 5).")
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def extract(self, image: MedicalImage, mask: Mask) -> dict[str, torch.Tensor]:
        device = self.extractor.device
        img_tensor = image.tensor.to(device)
        mask_tensor = mask.tensor.to(device)

        # We process ONLY the bounding box region to profoundly limit iterations
        coords = torch.nonzero(mask_tensor > 0.5, as_tuple=False)
        if coords.numel() == 0:
            return {}

        mins = coords.min(dim=0).values
        maxs = coords.max(dim=0).values

        z_min, y_min, x_min = mins
        z_max, y_max, x_max = maxs

        # Dimensions of the bounding box
        D, H, W = z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1

        # Create output maps dictionary
        output_maps = {}
        first_pass = True

        import torch.nn.functional as F

        padded_img = F.pad(
            img_tensor,
            (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad),
            mode="replicate",
        )
        padded_mask = F.pad(
            mask_tensor,
            (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad),
            mode="constant",
            value=0.0,
        )

        logger.info(
            f"Initiating VoxelFeatureExtractor for bounding volume: {D}x{H}x{W}"
        )

        for z in range(z_min, z_max + 1):
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    # We only calculate features at centers where the mask is natively present
                    if mask_tensor[z, y, x] <= 0.5:
                        continue

                    # Extract local patch context
                    z_p = z + self.pad
                    y_p = y + self.pad
                    x_p = x + self.pad

                    z_slice = slice(z_p - self.pad, z_p + self.pad + 1)
                    y_slice = slice(y_p - self.pad, y_p + self.pad + 1)
                    x_slice = slice(x_p - self.pad, x_p + self.pad + 1)

                    img_patch = padded_img[z_slice, y_slice, x_slice]
                    # We utilize the full patch as valid space to simulate standard macro conditions
                    mask_patch = torch.ones_like(img_patch)

                    patch_image = MedicalImage(img_patch, image.spacing)
                    patch_mask = Mask(mask_patch, mask.spacing)

                    try:
                        features = self.extractor.extract(patch_image, patch_mask)
                    except Exception:
                        # Log but do not interrupt map spatial generation
                        # Empty patches trigger native ValueErrors in PyRadiomics architecture bounds
                        continue

                    if first_pass:
                        for key in features:
                            output_maps[key] = torch.zeros(
                                img_tensor.shape, dtype=torch.float32, device=device
                            )
                        first_pass = False

                    for key, val in features.items():
                        output_maps[key][z, y, x] = val

        return output_maps
