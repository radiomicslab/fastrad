import os

import pytest

from tests.make_fixtures import create_dicom_series, img_vol, mask_vol


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures():
    base = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base, "fixtures", "image")
    mask_dir = os.path.join(base, "fixtures", "mask")

    # Generate if fixtures directory is missing or empty
    if not os.path.exists(img_dir) or not os.listdir(img_dir):
        create_dicom_series(img_dir, img_vol)
        create_dicom_series(mask_dir, mask_vol, is_mask=True)
