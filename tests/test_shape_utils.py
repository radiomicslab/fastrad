import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor

from fastrad.features.shape_utils import calculate_mesh_features


def test_shape_utils():
    # create a simple 2x2x2 shape
    mask_vol = np.zeros((4, 4, 4), dtype=int)
    mask_vol[1:3, 1:3, 1:3] = 1

    img_vol = np.ones_like(mask_vol)

    # volume calculation
    sitk_image = sitk.GetImageFromArray(img_vol)
    sitk_mask = sitk.GetImageFromArray(mask_vol)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))
    sitk_mask.SetSpacing((1.0, 1.0, 1.0))

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.settings["padDistance"] = 5
    res = extractor.execute(sitk_image, sitk_mask)

    mask_tensor = torch.tensor(mask_vol, dtype=torch.uint8)
    fast_res = calculate_mesh_features(mask_tensor, (1.0, 1.0, 1.0))  # spacing ZYX

    print("\n--- Shape Utils Vectorized Marching Cubes ---")
    print(
        f"Pyradiomics Area: {res.get('original_shape_SurfaceArea'):.6f}, FastRad Area: {fast_res['SurfaceArea']:.6f}"
    )
    print(
        f"Pyradiomics Volume: {res.get('original_shape_MeshVolume'):.6f}, FastRad Volume: {fast_res['MeshVolume']:.6f}"
    )
    print(
        f"Pyradiomics 3D Diam: {res.get('original_shape_Maximum3DDiameter'):.6f}, FastRad 3D Diam: {fast_res['Maximum3DDiameter']:.6f}"
    )
    print(
        f"Pyradiomics 2D Slice: {res.get('original_shape_Maximum2DDiameterSlice'):.6f}, FastRad 2D Slice: {fast_res['Maximum2DDiameterSlice']:.6f}"
    )
    print(
        f"Pyradiomics 2D Col: {res.get('original_shape_Maximum2DDiameterColumn'):.6f}, FastRad 2D Col: {fast_res['Maximum2DDiameterColumn']:.6f}"
    )
    print(
        f"Pyradiomics 2D Row: {res.get('original_shape_Maximum2DDiameterRow'):.6f}, FastRad 2D Row: {fast_res['Maximum2DDiameterRow']:.6f}"
    )

    if abs(res.get("original_shape_MeshVolume") - fast_res["MeshVolume"]) >= 1e-4:
        print("VOLUME MISMATCH!")
    if abs(res.get("original_shape_SurfaceArea") - fast_res["SurfaceArea"]) >= 1e-4:
        print("AREA MISMATCH!")
    print("FINISHED!")


if __name__ == "__main__":
    test_shape_utils()
