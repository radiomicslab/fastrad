import os

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, ImplicitVRLittleEndian


def create_dicom_series(out_dir, volume, is_mask=False):
    os.makedirs(out_dir, exist_ok=True)
    depth, height, width = volume.shape
    for z in range(depth):
        filename = os.path.join(out_dir, f"{z:04d}.dcm")
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = UID(
            "1.2.840.10008.5.1.4.1.1.2"
        )  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = UID(f"1.2.3.{z}")
        file_meta.ImplementationClassUID = UID("1.2.3.4")
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientName = "Test^Fixture"
        ds.PatientID = "123456"
        ds.StudyInstanceUID = "1.2.3.4.5"
        ds.SeriesInstanceUID = "1.2.3.4.5.6" if not is_mask else "1.2.3.4.5.7"
        ds.SOPInstanceUID = f"1.2.3.4.5.6.{z}"
        ds.Modality = "CT" if not is_mask else "SEG"

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.Columns = width
        ds.Rows = height
        ds.InstanceNumber = z
        ds.SliceLocation = float(z)

        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0

        # pydicom requires integer types for DICOM pixel data usually
        arr = volume[z].astype(np.uint16)
        ds.PixelData = arr.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.save_as(filename)


np.random.seed(42)
# 5x5x5 volume
img_vol = np.random.randint(0, 100, size=(5, 5, 5), dtype=np.uint16)
mask_vol = np.zeros((5, 5, 5), dtype=np.uint16)
# sphere in middle
mask_vol[1:4, 1:4, 1:4] = 1

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base, "fixtures", "image")
    create_dicom_series(img_dir, img_vol)
    create_dicom_series(os.path.join(base, "fixtures", "mask"), mask_vol, is_mask=True)

    # Verify reading
    ds = pydicom.dcmread(os.path.join(img_dir, "0000.dcm"))
    assert hasattr(ds, "pixel_array")

    print("Fixtures generated and verified.")
