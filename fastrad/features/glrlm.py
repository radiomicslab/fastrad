import torch

from fastrad.image import get_binned_image
from fastrad.settings import FeatureSettings

EPSILON = 1e-16

def _compute_core(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings) -> dict[str, float]:
    device = image_tensor.device
    
    binned_image, ivector = get_binned_image(image_tensor, mask_tensor, settings.bin_width)
    Ng = ivector.numel()
    if Ng == 0:
        return {}
        
    voxels = image_tensor[mask_tensor > 0.5]
        
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
        (1, -1, -1)
    ]
    
    img_int = binned_image.to(torch.int64)
    M = mask_tensor > 0.5
    D, H, W = img_int.shape

    # Run lengths for all 13 directions in one vectorised pass (no per-step loop).
    # Every ROI voxel p lies on the line {base + t*d} of direction d, where t is
    # its coordinate along d's first non-zero axis (that component is +1 for all
    # 13 directions) and base = p - t*d. Sorting voxels by (direction, line, t)
    # puts each line's ROI voxels in order; a run starts wherever the line
    # changes, t is not consecutive (the ROI is interrupted) or the grey level
    # changes. This is exactly the run definition of the original loop.
    coords = torch.nonzero(M)                                   # (N, 3)
    if coords.shape[0] == 0:
        return {}
    gl = img_int[M]                                             # (N,) same order
    d_t = torch.tensor(angles, dtype=torch.int64, device=device)  # (13, 3)
    N_a = d_t.shape[0]
    axis = torch.tensor([next(i for i in range(3) if a[i] != 0) for a in angles], device=device)
    t = coords[:, axis].T                                       # (13, N)
    base = coords.unsqueeze(0) - t.unsqueeze(-1) * d_t.unsqueeze(1)  # (13, N, 3)
    S = D + H + W                                               # makes base components >= 0
    R = 2 * S + 1
    T = max(D, H, W) + 2
    line = ((base[..., 0] + S) * R + (base[..., 1] + S)) * R + (base[..., 2] + S)  # (13, N)
    a_ids = torch.arange(N_a, device=device).view(-1, 1)
    key = (a_ids * (R ** 3) + line) * T + t                     # unique per (direction, voxel)
    key = key.reshape(-1)
    order = torch.argsort(key)
    key_s = key[order]
    g_s = gl.repeat(N_a)[order]
    a_s = a_ids.expand(-1, coords.shape[0]).reshape(-1)[order]
    same_line_next = (key_s[1:] - key_s[:-1]) == 1              # same direction+line, t consecutive
    new_run = torch.ones_like(key_s, dtype=torch.bool)
    new_run[1:] = ~(same_line_next & (g_s[1:] == g_s[:-1]))
    run_id = torch.cumsum(new_run.to(torch.int64), 0) - 1
    lengths = torch.bincount(run_id)                            # (n_runs,)
    run_gray = g_s[new_run]
    run_angle = a_s[new_run]

    Nr = int(lengths.max())
    if Nr == 0:
        return {}
    max_gl = int(torch.max(ivector))
    lin = (run_angle * max_gl + (run_gray - 1)) * Nr + (lengths - 1)
    counts = torch.bincount(lin, minlength=N_a * max_gl * Nr)[: N_a * max_gl * Nr]
    P_glrlm_raw = counts.view(N_a, max_gl, Nr).permute(1, 2, 0).to(torch.float64)

    valid_idx = (ivector - 1).to(torch.int64)
    P_glrlm = P_glrlm_raw[valid_idx, :, :]
        
    sums = torch.sum(P_glrlm, dim=(0, 1))
    valid_angles = sums > 0
    if not torch.any(valid_angles):
        return {}
        
    P_glrlm = P_glrlm[:, :, valid_angles]
    sums = sums[valid_angles]
    
    N_a = P_glrlm.shape[2]
    
    Np = voxels.numel()
    
    P = P_glrlm
    Ns = sums
    
    i_grid = ivector.clone().to(device).view(-1, 1, 1)
    j_grid = torch.arange(1, Nr + 1, dtype=torch.float64, device=device).view(1, -1, 1)
    
    pg = torch.sum(P, dim=1)
    pr = torch.sum(P, dim=0)
    
    sre = torch.mean(torch.sum(pr / (j_grid.view(-1, 1) ** 2), dim=0) / Ns)
    
    lre = torch.mean(torch.sum(pr * (j_grid.view(-1, 1) ** 2), dim=0) / Ns)
    
    glnu = torch.mean(torch.sum(pg ** 2, dim=0) / Ns)
    
    glnun = torch.mean(torch.sum(pg ** 2, dim=0) / (Ns ** 2))
    
    rlnu = torch.mean(torch.sum(pr ** 2, dim=0) / Ns)
    
    rlnun = torch.mean(torch.sum(pr ** 2, dim=0) / (Ns ** 2))
    
    rp = torch.mean(Ns / Np)
    
    p_norm = P / Ns.view(1, 1, -1)
    mu_i = torch.sum(i_grid * p_norm, dim=(0, 1))
    glv = torch.mean(torch.sum(((i_grid - mu_i.view(1, 1, -1)) ** 2) * p_norm, dim=(0, 1)))
    
    mu_j = torch.sum(j_grid * p_norm, dim=(0, 1))
    rv = torch.mean(torch.sum(((j_grid - mu_j.view(1, 1, -1)) ** 2) * p_norm, dim=(0, 1)))
    
    p_log = torch.where(p_norm > EPSILON, torch.log2(p_norm), torch.zeros_like(p_norm))
    re_feat = torch.mean(-torch.sum(p_norm * p_log, dim=(0, 1)))
    
    lglre = torch.mean(torch.sum(pg / (i_grid.squeeze(-1) ** 2), dim=0) / Ns)
    
    hglre = torch.mean(torch.sum(pg * (i_grid.squeeze(-1) ** 2), dim=0) / Ns)
    
    srlgle = torch.mean(torch.sum(P / ((i_grid ** 2) * (j_grid ** 2)), dim=(0, 1)) / Ns)
    
    srhgle = torch.mean(torch.sum(P * (i_grid ** 2) / (j_grid ** 2), dim=(0, 1)) / Ns)
    
    lrlgle = torch.mean(torch.sum(P * (j_grid ** 2) / (i_grid ** 2), dim=(0, 1)) / Ns)
    
    lrhgle = torch.mean(torch.sum(P * (i_grid ** 2) * (j_grid ** 2), dim=(0, 1)) / Ns)
    
    features = {
        "glrlm:short_run_emphasis": sre,
        "glrlm:long_run_emphasis": lre,
        "glrlm:gray_level_non_uniformity": glnu,
        "glrlm:gray_level_non_uniformity_normalized": glnun,
        "glrlm:run_length_non_uniformity": rlnu,
        "glrlm:run_length_non_uniformity_normalized": rlnun,
        "glrlm:run_percentage": rp,
        "glrlm:gray_level_variance": glv,
        "glrlm:run_variance": rv,
        "glrlm:run_entropy": re_feat,
        "glrlm:low_gray_level_run_emphasis": lglre,
        "glrlm:high_gray_level_run_emphasis": hglre,
        "glrlm:short_run_low_gray_level_emphasis": srlgle,
        "glrlm:short_run_high_gray_level_emphasis": srhgle,
        "glrlm:long_run_low_gray_level_emphasis": lrlgle,
        "glrlm:long_run_high_gray_level_emphasis": lrhgle
    }
    
    return features  # type: ignore

_compiled_compute = None

def compute(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings) -> dict[str, float]:
    global _compiled_compute
    if settings.compile:
        if _compiled_compute is None:
            _compiled_compute = torch.compile(_compute_core, mode=settings.compile_mode)
        return _compiled_compute(image_tensor, mask_tensor, settings)
    return _compute_core(image_tensor, mask_tensor, settings)
