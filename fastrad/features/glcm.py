import math

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

    max_gl = int(torch.max(ivector))

    angles = [
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, -1),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, -1, -1),
    ]

    # Co-occurrence counts for all 13 directions with one bincount: ROI
    # coordinates are computed once, every (voxel, neighbour) pair inside the
    # ROI is encoded as angle*Ng^2 + i*Ng + j, and the symmetric matrix is
    # counts + counts^T, exactly as the per-direction loop did.
    M = mask_tensor > 0.5
    img_int = binned_image.to(torch.int64) * M
    D, H, W = image_tensor.shape
    coords = torch.nonzero(M)  # (N, 3)
    d_t = torch.tensor(angles, dtype=torch.int64, device=device)  # (13, 3)
    N_angles = d_t.shape[0]
    nb = coords.unsqueeze(0) + d_t.unsqueeze(1)  # (13, N, 3)
    inb = (nb >= 0).all(-1) & (nb[..., 0] < D) & (nb[..., 1] < H) & (nb[..., 2] < W)
    nb = torch.where(inb.unsqueeze(-1), nb, torch.zeros_like(nb))
    nb_in_roi = inb & M[nb[..., 0], nb[..., 1], nb[..., 2]]
    v1 = (
        img_int[coords[:, 0], coords[:, 1], coords[:, 2]]
        .unsqueeze(0)
        .expand(N_angles, -1)
        - 1
    )
    v2 = img_int[nb[..., 0], nb[..., 1], nb[..., 2]] - 1
    ang = torch.arange(N_angles, device=device).view(-1, 1).expand(-1, coords.shape[0])
    lin = ((ang * max_gl + v1) * max_gl + v2)[nb_in_roi]
    counts = torch.bincount(lin, minlength=N_angles * max_gl * max_gl)[
        : N_angles * max_gl * max_gl
    ]
    counts = counts.view(N_angles, max_gl, max_gl).to(torch.float64)
    P_glcm_raw = (counts + counts.transpose(1, 2)).permute(1, 2, 0)  # (Ng, Ng, 13)

    valid_idx = (ivector - 1).to(torch.int64)
    P_glcm = P_glcm_raw[valid_idx][:, valid_idx]

    sums = torch.sum(P_glcm, dim=(0, 1))
    valid_angles = sums > 0
    if not torch.any(valid_angles):
        return {}

    P_glcm = P_glcm[:, :, valid_angles]
    sums = sums[valid_angles]

    p = P_glcm / sums.view(1, 1, -1)

    N_a = p.shape[2]

    px = torch.sum(p, dim=1)
    py = torch.sum(p, dim=0)

    i_grid = ivector.view(-1, 1, 1)
    j_grid = ivector.view(1, -1, 1)

    i_val = ivector.view(-1, 1)

    ux = torch.sum(i_grid * p, dim=(0, 1))
    uy = torch.sum(j_grid * p, dim=(0, 1))

    i_plus_j = i_grid + j_grid
    i_minus_j = torch.abs(i_grid - j_grid)

    diff_ij = i_grid - j_grid

    contrast = torch.mean(torch.sum((diff_ij**2) * p, dim=(0, 1)))

    diff_avg = torch.mean(torch.sum(i_minus_j * p, dim=(0, 1)))

    ux_m = torch.sum(i_val * px, dim=0)
    uy_m = torch.sum(i_val * py, dim=0)

    var_x = torch.sum(((i_val - ux_m.view(1, -1)) ** 2) * px, dim=0)
    var_y = torch.sum(((i_val - uy_m.view(1, -1)) ** 2) * py, dim=0)

    sigmax = torch.sqrt(var_x)
    sigmay = torch.sqrt(var_y)

    corm = torch.sum(
        (i_grid - ux_m.view(1, 1, -1)) * (j_grid - uy_m.view(1, 1, -1)) * p, dim=(0, 1)
    )
    corr = corm / (sigmax * sigmay + EPSILON)
    corr[(sigmax * sigmay) == 0] = 1.0
    correlation = torch.mean(corr)

    joint_average = torch.mean(ux)

    joint_energy = torch.mean(torch.sum(p**2, dim=(0, 1)))

    p_log = torch.where(p > EPSILON, torch.log2(p), torch.zeros_like(p))
    joint_entropy = torch.mean(-torch.sum(p * p_log, dim=(0, 1)))

    # Needs to match PyRadiomics exactly. i_grid is (Ng, 1, 1). j_grid is (1, Ng, 1). p is (Ng, Ng, N_a).
    ac_per_angle = torch.sum(i_grid * j_grid * p, dim=(0, 1))
    autocorrelation = torch.mean(ac_per_angle)

    cluster_prominence = torch.mean(
        torch.sum(
            ((i_grid + j_grid - ux.view(1, 1, -1) - uy.view(1, 1, -1)) ** 4) * p,
            dim=(0, 1),
        )
    )

    cluster_shade = torch.mean(
        torch.sum(
            ((i_grid + j_grid - ux.view(1, 1, -1) - uy.view(1, 1, -1)) ** 3) * p,
            dim=(0, 1),
        )
    )

    cluster_tendency = torch.mean(
        torch.sum(
            ((i_grid + j_grid - ux.view(1, 1, -1) - uy.view(1, 1, -1)) ** 2) * p,
            dim=(0, 1),
        )
    )

    max_prob = torch.mean(torch.amax(p, dim=(0, 1)))

    sum_squares = torch.mean(
        torch.sum(((i_grid - ux.view(1, 1, -1)) ** 2) * p, dim=(0, 1))
    )

    id_ = torch.mean(torch.sum(p / (1 + i_minus_j), dim=(0, 1)))

    idm = torch.mean(torch.sum(p / (1 + i_minus_j**2), dim=(0, 1)))

    idn = torch.mean(torch.sum(p / (1 + i_minus_j / max_gl), dim=(0, 1)))

    idmn = torch.mean(torch.sum(p / (1 + (i_minus_j**2) / (max_gl**2)), dim=(0, 1)))

    mask_ij = i_minus_j > 0
    inv_var = torch.mean(
        torch.sum(
            torch.where(mask_ij, p / (i_minus_j**2), torch.zeros_like(p)), dim=(0, 1)
        )
    )

    sum_average = torch.mean(torch.sum(i_plus_j * p, dim=(0, 1)))

    p_x_plus_y = torch.zeros((2 * max_gl + 1, N_a), dtype=torch.float64, device=device)
    p_x_minus_y = torch.zeros((max_gl + 1, N_a), dtype=torch.float64, device=device)

    flat_p = p.reshape(-1, N_a)
    flat_plus = i_plus_j.expand(-1, -1, N_a).to(torch.int64).reshape(-1, N_a)
    flat_minus = i_minus_j.expand(-1, -1, N_a).to(torch.int64).reshape(-1, N_a)

    p_x_plus_y.scatter_add_(0, flat_plus, flat_p)
    p_x_minus_y.scatter_add_(0, flat_minus, flat_p)

    diff_avg_a = torch.sum(i_minus_j * p, dim=(0, 1))

    k_minus = torch.arange(0, max_gl + 1, dtype=torch.float64, device=device).view(
        -1, 1
    )
    diff_var = torch.mean(
        torch.sum(((k_minus - diff_avg_a.view(1, -1)) ** 2) * p_x_minus_y, dim=0)
    )

    p_x_minus_y_log = torch.where(
        p_x_minus_y > EPSILON, torch.log2(p_x_minus_y), torch.zeros_like(p_x_minus_y)
    )
    diff_entropy = torch.mean(-torch.sum(p_x_minus_y * p_x_minus_y_log, dim=0))

    p_x_plus_y_log = torch.where(
        p_x_plus_y > EPSILON, torch.log2(p_x_plus_y), torch.zeros_like(p_x_plus_y)
    )
    sum_entropy = torch.mean(-torch.sum(p_x_plus_y * p_x_plus_y_log, dim=0))

    px_py = px.view(Ng, 1, -1) * py.view(1, Ng, -1)
    px_py_log = torch.where(px_py > EPSILON, torch.log2(px_py), torch.zeros_like(px_py))

    HXY1 = -torch.sum(p * px_py_log, dim=(0, 1))
    HXY2 = -torch.sum(px_py * px_py_log, dim=(0, 1))

    px_log = torch.where(px > EPSILON, torch.log2(px), torch.zeros_like(px))
    py_log = torch.where(py > EPSILON, torch.log2(py), torch.zeros_like(py))
    HX = -torch.sum(px * px_log, dim=0)
    HY = -torch.sum(py * py_log, dim=0)

    HXY = -torch.sum(p * p_log, dim=(0, 1))

    div = torch.max(HX, HY)
    imc1_vals = torch.where(div > EPSILON, (HXY - HXY1) / div, torch.zeros_like(HXY))
    imc1 = torch.mean(imc1_vals)
    imc2 = torch.mean(
        torch.sqrt(torch.clamp(1 - torch.exp(-2.0 * (HXY2 - HXY)), min=0.0))
    )

    # MCC: one small eigenproblem per angle; solved on the CPU, where
    # tiny dense eigensolvers are far cheaper than GPU solver launches.
    p_cpu, px_cpu = p.detach().cpu(), px.detach().cpu()
    # Angles with the same set of valid grey levels are solved in one batched
    # eigvals call (same LAPACK routine per matrix, so identical results).
    groups: dict = {}
    for ang in range(N_a):
        valid = (px_cpu[:, ang] > EPSILON).nonzero().squeeze(-1)
        groups.setdefault(tuple(valid.tolist()), []).append(ang)
    mcc_by_angle = [1.0] * N_a
    for valid_t, angs in groups.items():
        if len(valid_t) < 2:
            continue  # MCC = 1.0
        idx = torch.tensor(valid_t, dtype=torch.int64)
        A = p_cpu[idx][:, idx][:, :, angs].permute(2, 0, 1) / px_cpu[idx][
            :, angs
        ].T.unsqueeze(-1)
        Q = torch.matmul(A, A)
        try:
            evs = torch.sort(
                torch.linalg.eigvals(Q).real, dim=-1, descending=True
            ).values
            for k, ang in enumerate(angs):
                mcc_by_angle[ang] = math.sqrt(evs[k, 1])
        except Exception:
            pass  # keep 1.0, as before
    mcc_list = mcc_by_angle

    mcc = sum(mcc_list) / len(mcc_list) if mcc_list else 1.0

    features = {
        "glcm:contrast": contrast,
        "glcm:difference_average": diff_avg,
        "glcm:correlation": correlation,
        "glcm:joint_average": joint_average,
        "glcm:joint_energy": joint_energy,
        "glcm:joint_entropy": joint_entropy,
        "glcm:autocorrelation": autocorrelation,
        "glcm:cluster_prominence": cluster_prominence,
        "glcm:cluster_shade": cluster_shade,
        "glcm:cluster_tendency": cluster_tendency,
        "glcm:maximum_probability": max_prob,
        "glcm:sum_squares": sum_squares,
        "glcm:id": id_,
        "glcm:idm": idm,
        "glcm:idn": idn,
        "glcm:idmn": idmn,
        "glcm:inverse_variance": inv_var,
        "glcm:sum_average": sum_average,
        "glcm:difference_variance": diff_var,
        "glcm:difference_entropy": diff_entropy,
        "glcm:sum_entropy": sum_entropy,
        "glcm:imc1": imc1,
        "glcm:imc2": imc2,
        "glcm:mcc": mcc,
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
