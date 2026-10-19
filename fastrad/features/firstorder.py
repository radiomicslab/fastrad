import torch

from fastrad.settings import FeatureSettings

EPSILON = 1e-16


def compute(
    image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings
) -> dict[str, float]:
    # Extract voxels inside the mask
    # Mask is expected to be binary 0 or 1
    voxels = image_tensor[mask_tensor > 0.5].to(torch.float64)

    if voxels.numel() == 0:
        return {}

    # Basic stats
    mean = torch.mean(voxels)
    median = torch.median(voxels)
    maximum = torch.max(voxels)
    minimum = torch.min(voxels)

    # Range
    rng = maximum - minimum

    # Variance and Standard Deviation (population variance matches pyradiomics, degrees of freedom = 0)
    variance = torch.var(voxels, unbiased=False)
    std = torch.sqrt(variance)

    # Energy and Total Energy
    energy = torch.sum(voxels**2)
    # Voxel volume: z * y * x
    voxel_volume = settings.spacing[0] * settings.spacing[1] * settings.spacing[2]
    total_energy = energy * voxel_volume

    # Root Mean Squared
    rms = torch.sqrt(torch.mean(voxels**2))

    # Mean Absolute Deviation
    mad = torch.mean(torch.abs(voxels - mean))

    # Percentiles
    # PyTorch quantile needs float inputs
    voxels_float = voxels.to(torch.float32)
    p10 = torch.quantile(voxels_float, 0.10)
    p90 = torch.quantile(voxels_float, 0.90)
    iqr = torch.quantile(voxels_float, 0.75) - torch.quantile(voxels_float, 0.25)

    # Robust Mean Absolute Deviation (MAD within 10th and 90th percentiles)
    robust_mask = (voxels >= p10) & (voxels <= p90)
    robust_voxels = voxels[robust_mask]
    if robust_voxels.numel() > 0:
        robust_mean = torch.mean(robust_voxels)
        robust_mad = torch.mean(torch.abs(robust_voxels - robust_mean))
    else:
        robust_mad = torch.tensor(0.0, device=voxels.device)

    # Skewness and Kurtosis
    # Skewness = mean((x - mean)^3) / std^3
    # Kurtosis = mean((x - mean)^4) / std^4
    centered = voxels - mean
    m3 = torch.mean(centered**3)
    m4 = torch.mean(centered**4)

    skewness = m3 / (std**3 + EPSILON)
    kurtosis = m4 / (std**4 + EPSILON)

    # Entropy (using binned image)
    # Pyradiomics uses binned image to compute probability
    bin_width = settings.bin_width

    # Pyradiomics >= 3.0 absolute binning anchored at 0 multiples for binned metrics
    img_min = torch.min(image_tensor)
    minimum_binned = torch.floor(img_min / bin_width) * bin_width
    binned = torch.floor((voxels - minimum_binned) / bin_width) + 1

    # Calculate proportions of each bin
    binned_ints = binned.to(torch.int64)
    if binned_ints.numel() > 0:
        counts = torch.bincount(binned_ints)
        p = counts[counts > 0].to(torch.float64) / voxels.numel()
        entropy = -torch.sum(p * torch.log2(p + EPSILON))
        uniformity = torch.sum(p**2)
    else:
        entropy = torch.tensor(0.0)
        uniformity = torch.tensor(0.0)

    features = {
        "firstorder:energy": energy,
        "firstorder:total_energy": total_energy,
        "firstorder:entropy": entropy,
        "firstorder:minimum": minimum,
        "firstorder:10th_percentile": p10,
        "firstorder:90th_percentile": p90,
        "firstorder:maximum": maximum,
        "firstorder:mean": mean,
        "firstorder:median": median,
        "firstorder:interquartile_range": iqr,
        "firstorder:range": rng,
        "firstorder:mean_absolute_deviation": mad,
        "firstorder:robust_mean_absolute_deviation": robust_mad,
        "firstorder:root_mean_squared": rms,
        "firstorder:standard_deviation": std,
        "firstorder:skewness": skewness,
        "firstorder:kurtosis": kurtosis,
        "firstorder:variance": variance,
        "firstorder:uniformity": uniformity,
    }

    return features  # type: ignore
