import math

import torch

from fastrad.features.shape_utils import calculate_mesh_features
from fastrad.settings import FeatureSettings


def compute(
    image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings
) -> dict[str, float]:
    device = mask_tensor.device  # already on the extractor's resolved device
    spacing = settings.spacing

    # Move mask to device
    mask_tensor = mask_tensor.to(device)

    # 1. Mesh Features, in PyRadiomics' coordinate frame: the mask cropped to
    # its exact bounding box and padded by one voxel. The marching-cubes mesh
    # is not perfectly closed (in PyRadiomics either: same tables), and the
    # signed-tetrahedron volume of an open mesh depends on the coordinate
    # origin, so MeshVolume (and everything derived from it) only matches
    # PyRadiomics when computed in the same frame.
    nz = torch.nonzero(mask_tensor > 0.5)
    if nz.numel() == 0:
        return {}
    lo, hi = nz.min(dim=0).values.tolist(), (nz.max(dim=0).values + 1).tolist()
    mesh_mask = torch.nn.functional.pad(
        (mask_tensor[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] > 0.5).to(
            torch.float32
        ),
        (1, 1, 1, 1, 1, 1),
    )
    # Note: spacing is provided in native order, assuming shape_utils expects (Z, Y, X)
    mesh_res = calculate_mesh_features(mesh_mask, spacing)

    surface_area = mesh_res["SurfaceArea"]
    mesh_volume = mesh_res["MeshVolume"]

    # 2. Voxel Features
    # Find all coordinates of voxels in the mask
    coords = torch.nonzero(mask_tensor, as_tuple=False).to(
        torch.float64
    )  # shape (N, 3)
    Np = coords.shape[0]

    if Np == 0:
        return {}

    spacing_t = torch.tensor(spacing, device=device, dtype=torch.float64)
    voxel_volume_scalar = float(torch.prod(spacing_t))
    voxel_volume = Np * voxel_volume_scalar

    # PCA
    physical_coords = coords * spacing_t
    physical_coords -= torch.mean(physical_coords, dim=0)  # Centered at 0
    physical_coords /= math.sqrt(Np)

    covariance = torch.matmul(physical_coords.T, physical_coords)
    eigenvalues = torch.linalg.eigvalsh(covariance)

    # Address machine errors
    eigenvalues = torch.where(
        (eigenvalues < 0) & (eigenvalues > -1e-10),
        torch.tensor(0.0, dtype=torch.float64, device=device),
        eigenvalues,
    )
    eigenvalues, _ = torch.sort(eigenvalues)  # small to large
    eigen_l, eigen_m, eigen_h = eigenvalues[0], eigenvalues[1], eigenvalues[2]

    # Base features
    surface_volume_ratio = surface_area / mesh_volume if mesh_volume > 0 else 0.0

    sphericity = (
        (36.0 * math.pi * (mesh_volume**2.0)) ** (1.0 / 3.0) / surface_area
        if surface_area > 0
        else 0.0
    )
    spherical_disproportion = (
        surface_area / ((36.0 * math.pi * (mesh_volume**2.0)) ** (1.0 / 3.0))
        if mesh_volume > 0
        else 0.0
    )

    compactness1 = (
        mesh_volume / (math.sqrt(math.pi) * (surface_area ** (2.0 / 3.0)))
        if surface_area > 0
        else 0.0
    )
    compactness2 = (
        36.0 * math.pi * (mesh_volume**2.0) / (surface_area**3.0)
        if surface_area > 0
        else 0.0
    )

    major_axis = math.sqrt(eigen_h) * 4.0 if eigen_h >= 0 else math.nan
    minor_axis = math.sqrt(eigen_m) * 4.0 if eigen_m >= 0 else math.nan
    least_axis = math.sqrt(eigen_l) * 4.0 if eigen_l >= 0 else math.nan

    elongation = (
        math.sqrt(eigen_m / eigen_h) if (eigen_h > 0 and eigen_m >= 0) else math.nan
    )
    flatness = (
        math.sqrt(eigen_l / eigen_h) if (eigen_h > 0 and eigen_l >= 0) else math.nan
    )

    return {
        "shape:voxel_volume": float(voxel_volume),
        "shape:mesh_volume": float(mesh_volume),
        "shape:surface_area": float(surface_area),
        "shape:surface_volume_ratio": surface_volume_ratio,
        "shape:compactness_1": compactness1,
        "shape:compactness_2": compactness2,
        "shape:spherical_disproportion": spherical_disproportion,
        "shape:sphericity": sphericity,
        "shape:maximum_3d_diameter": mesh_res["Maximum3DDiameter"],
        "shape:maximum_2d_diameter_slice": mesh_res["Maximum2DDiameterSlice"],
        "shape:maximum_2d_diameter_column": mesh_res["Maximum2DDiameterColumn"],
        "shape:maximum_2d_diameter_row": mesh_res["Maximum2DDiameterRow"],
        "shape:major_axis_length": major_axis,
        "shape:minor_axis_length": minor_axis,
        "shape:least_axis_length": least_axis,
        "shape:elongation": elongation,
        "shape:flatness": flatness,
    }
