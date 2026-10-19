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
    img_int = binned_image.to(torch.int64) * (mask_tensor > 0.5)

    M = mask_tensor > 0.5

    import torch.nn.functional as F

    valid_coords = torch.nonzero(M, as_tuple=True)
    valid_gray = img_int[valid_coords]

    if valid_gray.numel() == 0:
        return {}

    img_float = img_int.to(torch.float64)
    mask_float = M.to(torch.float64)

    img_padded = F.pad(img_float, (1, 1, 1, 1, 1, 1), mode="constant", value=0.0)
    mask_padded = F.pad(mask_float, (1, 1, 1, 1, 1, 1), mode="constant", value=0.0)

    z = valid_coords[0] + 1
    y = valid_coords[1] + 1
    x = valid_coords[2] + 1

    N_v = valid_gray.shape[0]
    neighbor_sums = torch.zeros(N_v, dtype=torch.float64, device=device)
    neighbor_counts = torch.zeros(N_v, dtype=torch.int64, device=device)

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

    for dz, dy, dx in shifts:
        neighbor_sums += img_padded[z + dz, y + dy, x + dx]
        neighbor_counts += mask_padded[z + dz, y + dy, x + dx].to(torch.int64)

    valid_neighbors = neighbor_counts > 0
    s_i_valid = torch.zeros(N_v, dtype=torch.float64, device=device)

    s_i_valid[valid_neighbors] = torch.abs(
        valid_gray[valid_neighbors].to(torch.float64)
        - (
            neighbor_sums[valid_neighbors]
            / neighbor_counts[valid_neighbors].to(torch.float64)
        )
    )

    final_valid_gray = valid_gray[valid_neighbors]
    final_s_i = s_i_valid[valid_neighbors]

    if final_valid_gray.numel() == 0:
        return {}

    max_val = int(torch.max(final_valid_gray))

    if max_val == 0:
        return {}

    n_i = torch.bincount(final_valid_gray, minlength=max_val + 1).to(torch.float64)
    s_i = torch.bincount(final_valid_gray, weights=final_s_i, minlength=max_val + 1).to(
        torch.float64
    )

    valid_mask = n_i > 0
    ivector_actual = torch.arange(0, max_val + 1, dtype=torch.float64, device=device)

    n_i = n_i[valid_mask]
    s_i = s_i[valid_mask]
    # In order to fix NGTDM values, we map from the valid continuous indices back to to true `ivector`
    # Here `ivector_actual[valid_mask]` will be `1, 2, ... Num(unique)`.
    # Notice that mapping[raw_binned] in `get_binned_image` gave it values `1...Ng`.
    # But PyRadiomics uses the original true levels! Wait, we actually need to project back to the raw bins.
    # We can just use the provided `ivector` array directly for `ivector_mapped` values!
    # Because valid_mask matches the elements present in `ivector` EXACTLY!
    ivector_mapped = ivector.clone().to(device)

    # Wait, does pyradiomics compute Nvp as sum of voxels with AT LEAST ONE neighbor?
    # Actually, as per PyRadiomics comment: "Nvp is the sum of n_i (i.e. the number of voxels with a valid region; at least 1 neighbor)."
    # Let's count Nvp as sum of voxels with at least 1 neighbor.
    # The Matrix 1 output we saw `[[ 1 0 1 ]]` means `n_i` is NOT voxels with >0 neighbors,
    # but `p_i` uses `Nvp`! Oh, for Matrix 1, Nvp was 1... but W=0.
    # If Nvp = sum(n_i with >0 neighbors), for Matrix 1 it would be 0, dividing by 0.
    # Actually, PyRadiomics calculates W for each voxel. If W=0, the voxel might just be excluded from N_i if strictly reading the comment.
    # But Matrix 1 had `1` in col 0. Let's just use Nvp = sum(n_i), which is M.sum().

    # Wait, the comment says `Nvp is the sum of n_i (i.e. the number of voxels with a valid region; at least 1 neighbor)`.
    # And then `p_i = n_i / Nvp`.
    # Let's see if we should use `M.sum()` or `n_neighbors.sum()`.
    # Let's use `M.sum()`.
    Nvp = torch.sum(n_i)
    if Nvp == 0:
        return {}

    p_i = n_i / Nvp
    Ngp = p_i.numel()

    # Coarseness
    sum_coarse = torch.sum(p_i * s_i)
    coarseness = (
        1.0 / sum_coarse
        if sum_coarse > 0
        else torch.tensor(1e6, dtype=torch.float64, device=device)
    )

    # Contrast
    i_mat = ivector_mapped.view(-1, 1) - ivector_mapped.view(1, -1)
    pi_pj = p_i.view(-1, 1) * p_i.view(1, -1)

    if Ngp > 1:
        contrast = (
            torch.sum(pi_pj * (i_mat**2)) * torch.sum(s_i) / (Nvp * Ngp * (Ngp - 1))
        )
    else:
        contrast = torch.tensor(0.0, dtype=torch.float64, device=device)

    # Busyness
    # PyRadiomics definition: sum(p_i * s_i) / sum_i(sum_j(|i*p_i - j*p_j|)) where p_i != 0 and p_j != 0
    pi_pj_mask = (p_i.view(-1, 1) > 0) & (p_i.view(1, -1) > 0)

    i_pi = ivector_mapped * p_i
    ipi_jpj = torch.abs(i_pi.view(-1, 1) - i_pi.view(1, -1))

    # Apply mask where p_i != 0 and p_j != 0
    ipi_jpj = torch.where(pi_pj_mask, ipi_jpj, torch.zeros_like(ipi_jpj))

    sum_absdiff = torch.sum(ipi_jpj)

    busyness = (
        torch.sum(p_i * s_i) / sum_absdiff
        if sum_absdiff > 0
        else torch.tensor(0.0, dtype=torch.float64, device=device)
    )

    # Complexity
    i_minus_j = torch.abs(i_mat)
    num = p_i.view(-1, 1) * s_i.view(-1, 1) + p_i.view(1, -1) * s_i.view(1, -1)
    den = p_i.view(-1, 1) + p_i.view(1, -1)

    # avoiding divide by zero
    mask_den = den > 0
    complexity = (
        torch.sum(torch.where(mask_den, i_minus_j * num / den, torch.zeros_like(den)))
        / Nvp
    )

    # Strength
    num_str = (p_i.view(-1, 1) + p_i.view(1, -1)) * (i_mat**2)
    sum_str = torch.sum(num_str)
    sum_s_i = torch.sum(s_i)
    strength = (
        sum_str / sum_s_i
        if sum_s_i > 0
        else torch.tensor(0.0, dtype=torch.float64, device=device)
    )

    features = {
        "ngtdm:coarseness": coarseness,
        "ngtdm:contrast": contrast,
        "ngtdm:busyness": busyness,
        "ngtdm:complexity": complexity,
        "ngtdm:strength": strength,
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
