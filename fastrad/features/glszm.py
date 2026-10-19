import torch

from fastrad.image import get_binned_image
from fastrad.logger import logger
from fastrad.settings import FeatureSettings

EPSILON = 1e-16

# Upper bound on voxels labelled per call (levels x crop size); ~64 M int32
# voxels keeps the labelling buffer near 256 MB.
_LABEL_BATCH_VOXELS = 64_000_000

import numpy as np

# Which labelling backend ran last ("cupy" or "scipy"). Benchmarks read this to
# confirm GPU runs really used on-device labelling (the fallback is silent).
LAST_LABEL_BACKEND: str | None = None


def _label_connected_components(mask_tensor: torch.Tensor) -> torch.Tensor:
    global LAST_LABEL_BACKEND
    structure = (
        np.ones((3, 3, 3), dtype=int)
        if mask_tensor.ndim == 3
        else np.ones((3, 3), dtype=int)
    )

    if mask_tensor.device.type == "cuda":
        try:
            import cupy as cp  # type: ignore
            from cupyx.scipy.ndimage import label as cupy_label  # type: ignore

            # Transfer tensor to CuPy DLPack interface for zero-copy
            if hasattr(torch, "from_dlpack"):
                mask_cp = cp.from_dlpack(mask_tensor)
            else:
                import torch.utils.dlpack as torch_dlpack

                mask_cp = cp.from_dlpack(torch_dlpack.to_dlpack(mask_tensor))

            cp_structure = cp.asarray(structure, dtype=cp.int32)
            labeled, _ = cupy_label(mask_cp, structure=cp_structure)
            LAST_LABEL_BACKEND = "cupy"

            if hasattr(torch, "from_dlpack"):
                return torch.from_dlpack(labeled).to(mask_tensor.device)
            else:
                import torch.utils.dlpack as torch_dlpack

                return torch_dlpack.from_dlpack(labeled.toDlpack()).to(
                    mask_tensor.device
                )
        except Exception as e:
            logger.debug(f"CuPy GPU dispatch failed: {e}")
            # fall through to scipy

    import scipy.ndimage

    mask_np = mask_tensor.cpu().numpy()
    labeled, _ = scipy.ndimage.label(mask_np, structure=structure)
    LAST_LABEL_BACKEND = "scipy"
    return torch.from_numpy(labeled.astype(np.int32)).to(mask_tensor.device)


_warned_fallback = False


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401  # type: ignore
        from cupyx.scipy.ndimage import label  # noqa: F401  # type: ignore

        return True
    except Exception:
        return False


def _label_zones(
    mapped_int: torch.Tensor, n_levels: int, use_slabs: bool | None = None
) -> torch.Tensor:
    """Zone labels (unique ids > 0, 0 outside the ROI) for an image of grey levels 1..n_levels.

    GPU with CuPy: many levels per labelling call, stacked as slabs separated
    by an empty plane (few large kernel launches). CPU (SciPy): one call per
    level on that level's own bounding box, which is much cheaper on a CPU.
    """
    global _warned_fallback, LAST_LABEL_BACKEND
    device = mapped_int.device
    labels = torch.zeros(mapped_int.shape, dtype=torch.int64, device=device)
    if use_slabs is None:
        use_slabs = device.type == "cuda" and _cupy_available()
    if use_slabs:
        per_level = max(1, mapped_int.numel())
        batch = max(1, min(n_levels, _LABEL_BATCH_VOXELS // per_level))
        pad_spec = (0, 0) * (mapped_int.ndim - 1) + (0, 1)
        current = 0
        for start in range(1, n_levels + 1, batch):
            lv = torch.arange(start, min(start + batch, n_levels + 1), device=device)
            slabs = mapped_int.unsqueeze(0) == lv.view((-1,) + (1,) * mapped_int.ndim)
            stacked = torch.nn.functional.pad(slabs.to(torch.int32), pad_spec)
            k, d1 = stacked.shape[0], stacked.shape[1]
            lab = _label_connected_components(
                stacked.reshape((k * d1,) + tuple(stacked.shape[2:]))
            )
            lab = lab.to(torch.int64).reshape(stacked.shape)[:, :-1]
            lab = torch.where(lab > 0, lab + current, lab)
            labels += (lab * slabs).sum(dim=0)
            current = int(lab.max())
        return labels

    if device.type == "cuda" and not _warned_fallback:
        logger.warning(
            "CuPy is not available: GLSZM zone labelling runs on the CPU (SciPy) "
            "during GPU extraction, which is much slower. Install fastrad[cuda] "
            "with the CuPy build matching your CUDA version."
        )
        _warned_fallback = True

    import scipy.ndimage

    img_np = mapped_int.cpu().numpy()
    structure = np.ones((3,) * img_np.ndim, dtype=int)
    out = np.zeros(img_np.shape, dtype=np.int64)
    current = 0
    for g_idx, sl in enumerate(scipy.ndimage.find_objects(img_np)):
        if sl is None:
            continue
        lab, n = scipy.ndimage.label(img_np[sl] == g_idx + 1, structure=structure)
        if n:
            region = out[sl]
            region[lab > 0] = lab[lab > 0] + current
            current += n
    LAST_LABEL_BACKEND = "scipy"
    return torch.from_numpy(out).to(device)


def _compute_core(
    image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings
) -> dict[str, float]:
    device = image_tensor.device

    # 1. First, strictly narrow the evaluation to the bounding box of the ROI mask
    # to avoid scanning millions of empty structural voxels in subsequent tensor steps.
    coords = torch.nonzero(mask_tensor > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return {}
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values

    if mask_tensor.ndim == 3:
        image_tensor = image_tensor[
            mins[0] : maxs[0] + 1, mins[1] : maxs[1] + 1, mins[2] : maxs[2] + 1
        ]
        mask_tensor = mask_tensor[
            mins[0] : maxs[0] + 1, mins[1] : maxs[1] + 1, mins[2] : maxs[2] + 1
        ]
    else:
        image_tensor = image_tensor[mins[0] : maxs[0] + 1, mins[1] : maxs[1] + 1]
        mask_tensor = mask_tensor[mins[0] : maxs[0] + 1, mins[1] : maxs[1] + 1]

    binned_image, ivector = get_binned_image(
        image_tensor, mask_tensor, settings.bin_width
    )
    Ng = ivector.numel()
    if Ng == 0:
        return {}

    voxels = image_tensor[mask_tensor > 0.5]

    M = mask_tensor > 0.5
    img_int = binned_image.to(torch.int64) * M

    if int(torch.max(img_int)) == 0:
        return {}

    zero_tensor = torch.tensor([0], dtype=img_int.dtype, device=device)
    _, inverse_indices = torch.unique(
        torch.cat((zero_tensor, img_int.view(-1))), return_inverse=True
    )
    mapped_int = inverse_indices[1:].view(img_int.shape)  # 0 outside ROI, 1..Ng inside
    n_levels = int(mapped_int.max())

    labels_tensor = _label_zones(mapped_int, n_levels)

    labels = labels_tensor
    valid_labels = labels[M]
    valid_gray = img_int[M]

    unique_labels, inverse_indices, counts = torch.unique(
        valid_labels, return_inverse=True, return_counts=True
    )

    N_c = unique_labels.numel()
    if N_c == 0:
        return {}

    comp_gray = torch.zeros(N_c, dtype=img_int.dtype, device=device)
    comp_gray.scatter_(0, inverse_indices, valid_gray)

    Nz = int(torch.max(counts))

    if Nz == 0:
        return {}

    g = comp_gray - 1
    s = counts - 1

    max_gl = int(torch.max(ivector))
    linear_indices = g * Nz + s
    matrix_counts = torch.bincount(linear_indices, minlength=max_gl * Nz).to(
        torch.float64
    )
    P_raw = matrix_counts[: max_gl * Nz].view(max_gl, Nz)

    valid_idx = (ivector - 1).to(torch.int64)
    P = P_raw[valid_idx, :]

    Ns = P.sum()
    if Ns == 0:
        return {}

    Np = voxels.numel()

    i_grid = ivector.clone().to(device).view(-1, 1)
    j_grid = torch.arange(1, Nz + 1, dtype=torch.float64, device=device).view(1, -1)

    pg = torch.sum(P, dim=1).view(-1, 1)
    pz = torch.sum(P, dim=0).view(1, -1)

    sae = torch.sum(pz / (j_grid**2)) / Ns
    lae = torch.sum(pz * (j_grid**2)) / Ns

    glnu = torch.sum(pg**2) / Ns
    glnun = torch.sum(pg**2) / (Ns**2)

    szn = torch.sum(pz**2) / Ns
    sznn = torch.sum(pz**2) / (Ns**2)

    zp = Ns / Np

    p_norm = P / Ns
    mu_i = torch.sum(i_grid * p_norm)
    glv = torch.sum(((i_grid - mu_i) ** 2) * p_norm)

    mu_j = torch.sum(j_grid * p_norm)
    zv = torch.sum(((j_grid - mu_j) ** 2) * p_norm)

    p_log = torch.where(p_norm > EPSILON, torch.log2(p_norm), torch.zeros_like(p_norm))
    ze = -torch.sum(p_norm * p_log)

    lglze = torch.sum(pg / (i_grid**2)) / Ns
    hglze = torch.sum(pg * (i_grid**2)) / Ns

    salgle = torch.sum(P / ((i_grid**2) * (j_grid**2))) / Ns
    sahgle = torch.sum(P * (i_grid**2) / (j_grid**2)) / Ns
    lalgle = torch.sum(P * (j_grid**2) / (i_grid**2)) / Ns
    lahgle = torch.sum(P * (i_grid**2) * (j_grid**2)) / Ns

    features = {
        "glszm:small_area_emphasis": sae,
        "glszm:large_area_emphasis": lae,
        "glszm:gray_level_non_uniformity": glnu,
        "glszm:gray_level_non_uniformity_normalized": glnun,
        "glszm:size_zone_non_uniformity": szn,
        "glszm:size_zone_non_uniformity_normalized": sznn,
        "glszm:zone_percentage": zp,
        "glszm:gray_level_variance": glv,
        "glszm:zone_variance": zv,
        "glszm:zone_entropy": ze,
        "glszm:low_gray_level_zone_emphasis": lglze,
        "glszm:high_gray_level_zone_emphasis": hglze,
        "glszm:small_area_low_gray_level_emphasis": salgle,
        "glszm:small_area_high_gray_level_emphasis": sahgle,
        "glszm:large_area_low_gray_level_emphasis": lalgle,
        "glszm:large_area_high_gray_level_emphasis": lahgle,
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
