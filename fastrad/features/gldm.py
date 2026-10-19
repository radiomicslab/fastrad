import torch

from fastrad.image import get_binned_image
from fastrad.settings import FeatureSettings

EPSILON = 1e-16


def _compute_core(
    image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings
) -> dict[str, float]:
    device = image_tensor.device

    binned_image, ivector = get_binned_image(
        image_tensor, mask_tensor, settings.bin_width
    )
    Ng = ivector.numel()
    if Ng == 0:
        return {}

    img_int = binned_image.to(torch.int64) * (mask_tensor > 0.5)

    M = mask_tensor > 0.5
    import torch.nn.functional as F

    valid_coords = torch.nonzero(M, as_tuple=True)
    valid_gray = img_int[valid_coords]

    if valid_gray.numel() == 0:
        return {}

    img_padded = F.pad(img_int, (1, 1, 1, 1, 1, 1), mode="constant", value=0)

    z = valid_coords[0] + 1
    y = valid_coords[1] + 1
    x = valid_coords[2] + 1

    shifts = [
        (0, 0, 1),
        (0, 0, -1),
        (0, 1, 0),
        (0, -1, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 1),
        (0, 1, -1),
        (0, -1, 1),
        (0, -1, -1),
        (1, 0, 1),
        (1, 0, -1),
        (-1, 0, 1),
        (-1, 0, -1),
        (1, 1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (-1, -1, 0),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
        (-1, 1, 1),
        (-1, 1, -1),
        (-1, -1, 1),
        (-1, -1, -1),
    ]

    dependence = torch.ones(valid_gray.shape[0], dtype=torch.int64, device=device)

    for dz, dy, dx in shifts:
        neighbor_vals = img_padded[z + dz, y + dy, x + dx]
        mask_match = valid_gray == neighbor_vals
        dependence += mask_match.to(torch.int64)

    valid_dep = dependence

    Nd = int(torch.max(valid_dep))

    if Nd == 0:
        return {}

    g = valid_gray - 1
    d = valid_dep - 1

    max_gl = int(torch.max(ivector))
    linear_indices = g * Nd + d
    matrix_counts = torch.bincount(linear_indices, minlength=max_gl * Nd).to(
        torch.float64
    )
    P_raw = matrix_counts[: max_gl * Nd].view(max_gl, Nd)

    valid_idx = (ivector - 1).to(torch.int64)
    P = P_raw[valid_idx, :]

    Ns = P.sum()
    if Ns == 0:
        return {}

    i_grid = ivector.clone().to(device).view(-1, 1)
    j_grid = torch.arange(1, Nd + 1, dtype=torch.float64, device=device).view(1, -1)

    pg = torch.sum(P, dim=1).view(-1, 1)
    pd = torch.sum(P, dim=0).view(1, -1)

    sde = torch.sum(P / (j_grid**2)) / Ns
    lde = torch.sum(P * (j_grid**2)) / Ns

    glnu = torch.sum(pg**2) / Ns
    glnun = torch.sum(pg**2) / (Ns**2)

    dnu = torch.sum(pd**2) / Ns
    dnun = torch.sum(pd**2) / (Ns**2)

    p_norm = P / Ns
    mu_i = torch.sum(i_grid * p_norm)
    glv = torch.sum(((i_grid - mu_i) ** 2) * p_norm)

    mu_j = torch.sum(j_grid * p_norm)
    dv = torch.sum(((j_grid - mu_j) ** 2) * p_norm)

    p_log = torch.where(p_norm > EPSILON, torch.log2(p_norm), torch.zeros_like(p_norm))
    de = -torch.sum(p_norm * p_log)

    lgle = torch.sum(pg / (i_grid**2)) / Ns
    hgle = torch.sum(pg * (i_grid**2)) / Ns

    sdlgle = torch.sum(P / ((i_grid**2) * (j_grid**2))) / Ns
    sdhgle = torch.sum(P * (i_grid**2) / (j_grid**2)) / Ns
    ldlgle = torch.sum(P * (j_grid**2) / (i_grid**2)) / Ns
    ldhgle = torch.sum(P * (i_grid**2) * (j_grid**2)) / Ns

    features = {
        "gldm:small_dependence_emphasis": sde,
        "gldm:large_dependence_emphasis": lde,
        "gldm:gray_level_non_uniformity": glnu,
        "gldm:gray_level_non_uniformity_normalized": glnun,
        "gldm:dependence_non_uniformity": dnu,
        "gldm:dependence_non_uniformity_normalized": dnun,
        "gldm:dependence_variance": dv,
        "gldm:gray_level_variance": glv,
        "gldm:dependence_entropy": de,
        "gldm:low_gray_level_emphasis": lgle,
        "gldm:high_gray_level_emphasis": hgle,
        "gldm:small_dependence_low_gray_level_emphasis": sdlgle,
        "gldm:small_dependence_high_gray_level_emphasis": sdhgle,
        "gldm:large_dependence_low_gray_level_emphasis": ldlgle,
        "gldm:large_dependence_high_gray_level_emphasis": ldhgle,
    }

    return features  # type: ignore


_compiled_compute = None


def compute(
    image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings
) -> dict[str, float]:
    global _compiled_compute
    if settings.compile:
        if _compiled_compute is None:
            _compiled_compute = torch.compile(_compute_core, mode=settings.compile_mode)
        return _compiled_compute(image_tensor, mask_tensor, settings)
    return _compute_core(image_tensor, mask_tensor, settings)
