import math

import torch

from fastrad.settings import FeatureSettings

# Marching Squares Lookup Tables
gridAngles2D = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=torch.int64)

lineTable2D = torch.tensor(
    [
        [-1, -1, -1, -1, -1],
        [3, 0, -1, -1, -1],
        [0, 1, -1, -1, -1],
        [3, 1, -1, -1, -1],
        [1, 2, -1, -1, -1],
        [1, 2, 3, 0, -1],
        [0, 2, -1, -1, -1],
        [3, 2, -1, -1, -1],
        [2, 3, -1, -1, -1],
        [2, 0, -1, -1, -1],
        [0, 1, 2, 3, -1],
        [2, 1, -1, -1, -1],
        [1, 3, -1, -1, -1],
        [1, 0, -1, -1, -1],
        [0, 3, -1, -1, -1],
        [-1, -1, -1, -1, -1],
    ],
    dtype=torch.int64,
)

vertList2D = torch.tensor(
    [[0.0, 0.5], [0.5, 1.0], [1.0, 0.5], [0.5, 0.0]], dtype=torch.float64
)

# To check if points are added:
# points_edges[0] = {0, 3}; points_edges[1] = {2, 2}
# from C: {{0, 2}, {3, 2}} -> t=0: pt 0 => vert 2; t=1: pt 3 => vert 2


def compute(
    image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings
) -> dict[str, float]:
    if not settings.force2D:
        raise ValueError(
            "Shape 2D features are only calculated for ROI that are fundamentally 2D. Please enable force2D in settings."
        )

    device = image_tensor.device
    mask = mask_tensor > 0.5

    if not torch.any(mask):
        return {}

    # Get bounding box
    nonzero = torch.nonzero(mask)
    min_dims = torch.min(nonzero, dim=0).values
    max_dims = torch.max(nonzero, dim=0).values

    dim_sizes = max_dims - min_dims + 1

    force_dim = settings.force2Ddimension
    if dim_sizes[force_dim] > 1:
        raise ValueError(
            f"Size of the mask in dimension {force_dim} is more than 1, cannot compute 2D shape"
        )

    slice_idx = min_dims[force_dim]

    if force_dim == 0:
        M2D = mask[slice_idx, :, :]
        spacing_2d = torch.tensor(
            [settings.spacing[1], settings.spacing[2]],
            dtype=torch.float64,
            device=device,
        )
    elif force_dim == 1:
        M2D = mask[:, slice_idx, :]
        spacing_2d = torch.tensor(
            [settings.spacing[0], settings.spacing[2]],
            dtype=torch.float64,
            device=device,
        )
    else:
        M2D = mask[:, :, slice_idx]
        spacing_2d = torch.tensor(
            [settings.spacing[0], settings.spacing[1]],
            dtype=torch.float64,
            device=device,
        )

    H, W = M2D.shape
    padded = torch.zeros((H + 1, W + 1), dtype=torch.uint8, device=device)
    padded[:-1, :-1] = M2D.to(torch.uint8)

    grid_angles = gridAngles2D.to(device)
    square_idx = torch.zeros((H, W), dtype=torch.uint8, device=device)
    for a_idx in range(4):
        cy = grid_angles[a_idx, 0]
        cx = grid_angles[a_idx, 1]
        square_idx |= padded[cy : cy + H, cx : cx + W] << a_idx

    valid_squares = (square_idx > 0) & (square_idx < 15)
    active_indices = torch.nonzero(valid_squares, as_tuple=True)
    active_idxs = square_idx[active_indices].long()

    line_table = lineTable2D.to(device)[active_idxs]
    vert_list = vertList2D.to(device)

    offsets = torch.stack(active_indices, dim=-1).to(torch.float64)

    total_surface = 0.0
    total_perimeter = 0.0

    all_vertices = []

    for t in range(2):
        v1_idx = line_table[:, t * 2]
        valid_lines = v1_idx >= 0
        if not valid_lines.any():
            continue

        v1_i = v1_idx[valid_lines]
        v2_i = line_table[valid_lines, t * 2 + 1]

        v1 = vert_list[v1_i]
        v2 = vert_list[v2_i]

        off = offsets[valid_lines]

        a = (v1 + off) * spacing_2d
        b = (v2 + off) * spacing_2d

        total_surface += torch.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])

        dists = torch.norm(a - b, dim=1)
        total_perimeter += torch.sum(dists)

    total_surface = abs(total_surface) / 2.0

    # Vertices for maximum diameter
    # If square_idx > 7: square_idx = square_idx ^ 15
    check_idxs = torch.where(active_idxs > 7, active_idxs ^ 15, active_idxs)

    for t_edge, (pt_idx, vert_idx) in enumerate([(0, 2), (3, 2)]):
        has_vert = (check_idxs & (1 << pt_idx)) > 0
        if has_vert.any():
            v_i = torch.full(
                (has_vert.sum(),), vert_idx, dtype=torch.long, device=device
            )  # type: ignore
            v = vert_list[v_i]
            off = offsets[has_vert]
            pt = (v + off) * spacing_2d
            all_vertices.append(pt)

    if len(all_vertices) > 0:
        all_pts = torch.cat(all_vertices, dim=0)
        unique_pts = torch.unique(all_pts, dim=0)
        max_diameter = torch.cdist(unique_pts, unique_pts).max()
    else:
        max_diameter = 0.0

    pixel_surface = M2D.sum() * spacing_2d[0] * spacing_2d[1]

    # PCA
    coords = torch.nonzero(M2D, as_tuple=False).to(torch.float64) * spacing_2d
    N = coords.shape[0]
    if N == 0:
        return {}

    if N == 1:
        major = 0.0
        minor = 0.0
        elongation = 0.0
    else:
        mean_coords = torch.mean(coords, dim=0)
        coords_centered = coords - mean_coords
        cov = torch.matmul(coords_centered.T, coords_centered) / N
        try:
            eigvals = torch.linalg.eigvalsh(cov)
            eigvals = torch.clamp(eigvals, min=0.0)
            eigvals = torch.sort(eigvals, descending=True).values
            lambda_1 = eigvals[0]
            lambda_2 = eigvals[1] if eigvals.shape[0] > 1 else 0.0

            major = 4.0 * math.sqrt(lambda_1)
            minor = 4.0 * math.sqrt(lambda_2)
            elongation = math.sqrt(lambda_2 / lambda_1) if lambda_1 > 1e-10 else 0.0
        except Exception:
            major = 0.0
            minor = 0.0
            elongation = 0.0

    psi = (total_perimeter / total_surface) if total_surface > 0 else 0.0
    sphericity = (
        (2.0 * math.sqrt(math.pi * total_surface) / total_perimeter)
        if total_perimeter > 0
        else 0.0
    )

    features = {
        "shape2D:MeshSurface": total_surface,
        "shape2D:PixelSurface": pixel_surface,
        "shape2D:Perimeter": total_perimeter,
        "shape2D:PerimeterSurfaceRatio": psi,
        "shape2D:Sphericity": sphericity,
        "shape2D:MaximumDiameter": max_diameter,
        "shape2D:MajorAxisLength": major,
        "shape2D:MinorAxisLength": minor,
        "shape2D:Elongation": elongation,
    }

    return features  # type: ignore
