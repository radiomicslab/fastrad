import numpy as np
import torch
from scipy.spatial import ConvexHull

from fastrad.logger import logger
from .shape_tables import gridAngles, triTable, vertList


def _line_extremes(P: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Rows of P that are the min or max of coordinate c among points with equal (a, b).

    Exact pre-filter for convex hulls and farthest pairs: a hull vertex can
    never lie strictly between two other points of its own grid line.
    """
    order = np.lexsort((P[:, c], P[:, b], P[:, a]))
    Q = P[order]
    new = np.empty(len(Q), dtype=bool)
    new[0] = True
    new[1:] = (Q[1:, a] != Q[:-1, a]) | (Q[1:, b] != Q[:-1, b])
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:] - 1, len(Q) - 1]
    return Q[np.unique(np.r_[starts, ends])]


def _farthest_pair_distance(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    if len(pts) > 64:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:  # collinear / degenerate: use all points
            pass
    d = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt((d * d).sum(-1).max()))


def _max_2d_diameters(verts: np.ndarray) -> list[float]:
    """Largest in-plane vertex distance for planes of constant z, y and x.

    Matches PyRadiomics: diameter[0] uses pairs with equal z (slice),
    [1] equal y (column), [2] equal x (row). Within each plane only the end
    points of each grid row can be farthest-pair candidates (exact), and the
    farthest pair of those is found on their convex hull.
    """
    out = [0.0, 0.0, 0.0]
    for i in range(3):
        u, v = [j for j in range(3) if j != i]
        cand = _line_extremes(verts, i, u, v)        # sorted by plane i
        bounds = np.flatnonzero(np.diff(cand[:, i])) + 1
        for grp in np.split(cand[:, [u, v]], bounds):
            out[i] = max(out[i], _farthest_pair_distance(grp))
    return out


def _line_extreme_mask(H: torch.Tensor, a: int, b: int, c: int) -> torch.Tensor:
    """Mask of points that are the min or max of integer coordinate c on their (a, b) line.

    Exact pre-filter for convex hulls and farthest pairs (a hull vertex cannot
    lie strictly between two other points of its own grid line). Uses one
    scatter-min and one scatter-max instead of a sort. Points must be unique.
    """
    width = int(H[:, b].max()) + 1
    _, line = torch.unique(H[:, a] * width + H[:, b], return_inverse=True)
    n = int(line.max()) + 1
    big = torch.iinfo(torch.int64).max
    lo = torch.full((n,), big, dtype=torch.int64, device=H.device).scatter_reduce(0, line, H[:, c], "amin")
    hi = torch.full((n,), -big, dtype=torch.int64, device=H.device).scatter_reduce(0, line, H[:, c], "amax")
    return (H[:, c] == lo[line]) | (H[:, c] == hi[line])


def _max_2d_diameters_t(H: torch.Tensor, P: torch.Tensor, max_pairs: int = 4_000_000) -> list[float]:
    """Batched in-plane farthest-pair distances (PyRadiomics' 2D diameters).

    H: unique vertices on the half-voxel grid (int64), P: their physical
    coordinates (float64). For each axis i, vertices are grouped by plane
    (equal H[:, i]); after the exact line-extreme filter each plane keeps only
    its row end points, and all planes' pairwise distances are evaluated in
    padded batches. Brute force over candidates, so the maximum is exact.
    """
    out = [0.0, 0.0, 0.0]
    for i in range(3):
        u, v = [j for j in range(3) if j != i]
        keep = _line_extreme_mask(H, i, u, v)
        Hc, Pc = H[keep], P[keep][:, [u, v]]
        _, plane = torch.unique(Hc[:, i], return_inverse=True)
        order = torch.argsort(plane, stable=True)
        plane, Pc = plane[order], Pc[order]
        counts = torch.bincount(plane)
        n_planes, width = counts.numel(), int(counts.max())
        start = torch.cumsum(counts, 0) - counts
        slot = torch.arange(plane.numel(), device=H.device) - start[plane]
        # Pad each plane with copies of its first point: zero extra distance.
        padded = Pc[start].unsqueeze(1).expand(n_planes, width, 2).clone()
        padded[plane, slot] = Pc
        step = max(1, max_pairs // (width * width))
        best = torch.zeros((), dtype=Pc.dtype, device=H.device)
        for g0 in range(0, n_planes, step):
            blk = padded[g0:g0 + step]
            d2 = ((blk.unsqueeze(2) - blk.unsqueeze(1)) ** 2).sum(-1)
            best = torch.maximum(best, d2.max())
        out[i] = best
    vals = torch.sqrt(torch.stack(out)).cpu().tolist()
    return vals


def calculate_mesh_features(mask_tensor, spacing):
    """Marching-cubes surface area, volume and diameters (PyRadiomics algorithm).

    All triangles are processed in one batch; the only device->host copies are
    one for the three scalar totals and one for the vertex set used by the
    diameter computation.
    """
    device = mask_tensor.device

    triTable_t = torch.tensor(triTable, device=device, dtype=torch.int64)
    vertList_t = torch.tensor(vertList, device=device, dtype=torch.float64)
    gridAngles_t = gridAngles  # python list, used for slicing only

    D, H, W = mask_tensor.shape
    # Pad by 1 on all sides so boundary faces are fully captured
    padded_mask = torch.zeros((D + 2, H + 2, W + 2), dtype=torch.uint8, device=device)
    padded_mask[1:-1, 1:-1, 1:-1] = (mask_tensor > 0.5)

    cube_idx = torch.zeros((D + 1, H + 1, W + 1), dtype=torch.uint8, device=device)
    for a_idx in range(8):
        c_z, c_y, c_x = gridAngles_t[a_idx]
        cube_idx |= padded_mask[c_z:c_z + D + 1, c_y:c_y + H + 1, c_x:c_x + W + 1] << a_idx

    valid_cubes = (cube_idx > 0) & (cube_idx < 255)
    active = torch.nonzero(valid_cubes)                       # (C, 3)
    active_idxs = cube_idx[valid_cubes].to(torch.int64)       # same order as nonzero

    # Symmetry: configurations >= 128 are looked up flipped, with the volume sign flipped
    flip = active_idxs >= 128
    signs = torch.where(flip, -1.0, 1.0).to(torch.float64)
    lookup = torch.where(flip, active_idxs ^ 255, active_idxs)

    # Coordinates relative to the input mask (padded index - 1)
    offsets = active.to(torch.float64) - 1.0                  # (C, 3)
    tri = triTable_t[lookup][:, :15].view(-1, 5, 3)           # (C, 5, 3) edge ids, -1 = none
    valid_tri = tri[:, :, 0] >= 0                             # (C, 5)
    cube_of_tri = torch.arange(tri.shape[0], device=device).view(-1, 1).expand(-1, 5)[valid_tri]
    edges = tri[valid_tri]                                    # (T, 3)

    spacing_t = torch.tensor(spacing, device=device, dtype=torch.float64)
    off = offsets[cube_of_tri]                                # (T, 3)
    a = (vertList_t[edges[:, 0]] + off) * spacing_t
    b = (vertList_t[edges[:, 1]] + off) * spacing_t
    c = (vertList_t[edges[:, 2]] + off) * spacing_t
    sgn = signs[cube_of_tri]

    area = 0.5 * torch.linalg.vector_norm(torch.cross(a - c, b - c, dim=1), dim=1)
    vol = (torch.cross(a, b, dim=1) * c).sum(dim=1) * sgn
    normals = 0.5 * torch.cross(b - a, c - a, dim=1) * sgn.unsqueeze(1)

    totals = torch.stack([area.sum(), vol.sum(), torch.linalg.vector_norm(normals.sum(dim=0))]).cpu().tolist()
    total_area, total_volume, mesh_closure_error = totals

    # Mesh-closure diagnostic. The sum of signed triangle-area vectors is zero
    # for a closed, consistently oriented mesh. PyRadiomics' marching-cubes
    # tables (used here unchanged) leave small openings on some irregular
    # boundaries, in PyRadiomics as well. For an open mesh the signed-
    # tetrahedron volume depends on the coordinate origin, which is why the
    # caller (shape.compute) builds the mesh in PyRadiomics' frame (mask
    # cropped to its bounding box, padded by one voxel): in that frame
    # MeshVolume matches PyRadiomics exactly. The value is reported for
    # diagnostics only.
    if total_area > 0 and (mesh_closure_error / total_area) > 1e-4:
        logger.debug(
            f"Marching-cubes mesh is not fully closed (closure error "
            f"{mesh_closure_error:.4f}, surface area {total_area:.1f}); "
            f"MeshVolume is reported in PyRadiomics' coordinate frame."
        )

    if edges.shape[0] > 0:
        # Deduplicate vertices on the device with integer keys. Every vertex
        # lies on the half-voxel grid (cube offset + 0, 0.5 or 1), so twice its
        # grid coordinate is an exact integer. Physical coordinates are
        # rebuilt as (half_index / 2) * spacing, which is bit-identical to the
        # (vertex + offset) * spacing values used above.
        corners = vertList_t[edges.reshape(-1)] + off.repeat_interleave(3, dim=0)  # (3T, 3)
        half = torch.round(corners * 2).to(torch.int64) + 2               # >= 0
        dims = half.max(dim=0).values + 1
        key = (half[:, 0] * dims[1] + half[:, 1]) * dims[2] + half[:, 2]
        ukey = torch.unique(key)
        hz = ukey // (dims[1] * dims[2])
        hy = (ukey // dims[2]) % dims[1]
        hx = ukey % dims[2]
        uhalf = torch.stack([hz, hy, hx], dim=1) - 2
        P = (uhalf.to(torch.float64) / 2.0) * spacing_t

        # 3D: exact line-extreme filter on the device, convex hull of the
        # (much smaller) candidate set on the host.
        cand = P[_line_extreme_mask(uhalf, 0, 1, 2)].cpu().numpy()
        try:
            hull_pts = cand[ConvexHull(cand).vertices]
        except Exception:  # flat / degenerate ROI: use all candidates
            hull_pts = cand
        d = hull_pts[:, None, :] - hull_pts[None, :, :]
        max_3d_diameter = float(np.sqrt((d * d).sum(-1).max()))
        if P.device.type == "cuda":
            # Batched brute force: a few large kernels, cheap on a GPU.
            max_2d_slice, max_2d_col, max_2d_row = _max_2d_diameters_t(uhalf, P)
        else:
            # On the CPU, per-plane convex hulls are cheaper than brute force.
            max_2d_slice, max_2d_col, max_2d_row = _max_2d_diameters(P.numpy())
    else:
        max_3d_diameter = max_2d_slice = max_2d_col = max_2d_row = 0.0

    return {
        "SurfaceArea": total_area,
        "MeshVolume": total_volume / 6.0,
        "Maximum3DDiameter": max_3d_diameter,
        "Maximum2DDiameterSlice": max_2d_slice,
        "Maximum2DDiameterColumn": max_2d_col,
        "Maximum2DDiameterRow": max_2d_row,
        "MeshClosureError": mesh_closure_error,
    }
