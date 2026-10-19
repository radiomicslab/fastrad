import make_fixtures
import numpy as np
import SimpleITK as sitk
from radiomics import glcm

sitk_image = sitk.GetImageFromArray(make_fixtures.img_vol)
sitk_image.SetSpacing((1.0, 1.0, 1.0))
sitk_mask = sitk.GetImageFromArray(make_fixtures.mask_vol)
sitk_mask.SetSpacing((1.0, 1.0, 1.0))

from radiomics import featureextractor

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings["binWidth"] = 25.0
extractor.settings["distances"] = [1]
extractor.settings["force2D"] = False

# We need to get the exact binned image to match.
from radiomics import imageoperations

bb, correctedMask = imageoperations.checkMask(sitk_image, sitk_mask)
croppedImage, croppedMask = imageoperations.cropToTumorMask(
    sitk_image, correctedMask, bb
)
binnedImage, _ = imageoperations.binImage(
    extractor.settings["binWidth"], extractor.settings, croppedImage, croppedMask
)

binned_arr = sitk.GetArrayFromImage(binnedImage)
mask_arr = sitk.GetArrayFromImage(croppedMask)
print(
    f"Binned Max: {np.max(binned_arr[mask_arr == 1])}, Min: {np.min(binned_arr[mask_arr == 1])}"
)

feature_class = glcm.RadiomicsGLCM(sitk_image, sitk_mask, **extractor.settings)
feature_class.execute()

print(f"P_glcm shape: {feature_class.P_glcm.shape}")
print(f"Calculated contrast: {feature_class.getContrastFeatureValue()}")
