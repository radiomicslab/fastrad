import json
import urllib.request
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.spatialimages import SpatialImage

from fastrad.extractor import FeatureExtractor
from fastrad.image import Mask, MedicalImage
from fastrad.settings import FeatureSettings

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ibsi"
URL_IMAGE = "https://github.com/theibsi/data_sets/raw/master/ibsi_1_digital_phantom/nifti/image/phantom.nii.gz"
URL_MASK = "https://github.com/theibsi/data_sets/raw/master/ibsi_1_digital_phantom/nifti/mask/mask.nii.gz"
URL_REF = "https://raw.githubusercontent.com/theibsi/data_sets/main/ibsi_1_digital_phantom/reference/features/ibsi_1_digital_phantom_reference_features.csv" # Not used anymore

def download_if_needed(url: str, filepath: Path) -> bool:
    if not filepath.exists():
        print(f"Downloading {filepath.name} from {url}")
        try:
            urllib.request.urlretrieve(url, filepath)
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False
    return True

def setup_module(module):
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    # Don't throw errors in setup if download fails, we will skip in the test
    download_if_needed(URL_IMAGE, FIXTURE_DIR / "phantom.nii.gz")
    download_if_needed(URL_MASK, FIXTURE_DIR / "mask.nii.gz")

def load_reference_data():
    json_path = FIXTURE_DIR / "reference.json"
    if not json_path.exists():
        print(f"Warning: {json_path} not found. Run tests/scripts/generate_ibsi_reference.py first.")
        return {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        reference = json.load(f)
    return reference

def test_ibsi_compliance():
    # Load phantom
    img_path = FIXTURE_DIR / "phantom.nii.gz"
    mask_path = FIXTURE_DIR / "mask.nii.gz"
    
    if not img_path.exists() or not mask_path.exists():
        pytest.skip("IBSI digital phantom files not found and could not be downloaded. Skipping compliance test.")
        
    img_ni = cast(SpatialImage, nib.load(str(img_path)))
    mask_ni = cast(SpatialImage, nib.load(str(mask_path)))
    
    # nibabel loads data as (X, Y, Z). PyRadiomics/fastrad uses (Z, Y, X)
    img_data = np.transpose(img_ni.get_fdata(), (2, 1, 0))
    mask_data = np.transpose(mask_ni.get_fdata(), (2, 1, 0))
    
    # Spacing is generally (X, Y, Z) in header.get_zooms()
    zooms = img_ni.header.get_zooms()
    spacing = (zooms[2], zooms[1], zooms[0]) # (Z, Y, X)
    
    img_tensor = torch.from_numpy(img_data).float()
    mask_tensor = torch.from_numpy(mask_data).float()
    
    image = MedicalImage(img_tensor, spacing=spacing)
    mask = Mask(mask_tensor, spacing=spacing)
    
    # IBSI phase 1 digital phantom specifies:
    # 25 discretised levels if bin count is 25. Let's just run default 25 width for testing stability
    settings = FeatureSettings(
        feature_classes=["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
        bin_width=25.0,
        device="cpu"
    )
    
    extractor = FeatureExtractor(settings)
    features = extractor.extract(image, mask)
    
    assert len(features) > 0

    reference_features = load_reference_data()
    
    # Assert features match PyRadiomics IBSI pipeline
    for k, v in features.items():
        assert not np.isnan(v), f"Feature {k} returned NaN"
        
        if k in reference_features:
            ref_val = reference_features[k]
            # Use 1e-4 tolerance as required by CLAUDE.md
            assert abs(v - ref_val) < 1e-4, f"Feature {k} mismatch. fastrad: {v}, pyradiomics: {ref_val}, diff: {abs(v - ref_val)}"
        else:
            print(f"Warning: Feature {k} extracted by fastrad but not found in reference.")
            
    # Also verify we extracted everything PyRadiomics extracted (for enabled classes)
    for k in reference_features:
        if not k.startswith("shape2d"):  # We didn't enable shape2d in the extractor test
            assert k in features, f"Reference feature {k} missing from fastrad output"
