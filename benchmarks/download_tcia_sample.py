"""
Download the real clinical benchmark case: TCIA NSCLC-Radiomics,
subject LUNG1-001, its CT series, and its GTV-1 segmentation from the
accompanying RTSTRUCT, converted to a binary mask matching the CT grid.

*** THIS SCRIPT WAS WRITTEN WITHOUT NETWORK ACCESS TO TCIA AND HAS NEVER
    BEEN RUN. It follows the same TCIA NBIA REST API conventions already
    used successfully elsewhere in this repository
    (download_rider_pairs.py), but the RTSTRUCT-to-mask conversion path
    is unverified. Treat this as a strong starting point, not a verified
    tool -- run it, inspect the resulting mask, and confirm it matches
    the GTV-1 structure (5,920 voxels per this manuscript's benchmark
    description) before trusting it. ***

WHY THIS REPLACES THE PREVIOUS VERSION OF THIS FILE:
The previous download_tcia_sample.py fetched a series from the "RIDER Lung
CT" collection -- the wrong collection entirely -- and explicitly noted it
would generate a "dummy"/synthetic spherical mask rather than a real
segmentation, and did not even produce NRRD output. But
benchmarks/run_numerical_parity.py and benchmarks/run_runtime_performance.py
require real tests/fixtures/tcia/lung1_image.nrrd and lung1_label.nrrd
files with a genuine GTV-1 segmentation from the NSCLC-Radiomics
collection. Anyone following the old download script alone could not have
reproduced this study's headline clinical benchmark.

REQUIRES: pip install rt-utils pydicom SimpleITK requests
(rt-utils is a well-established library specifically for reading/writing
DICOM RTSTRUCT files aligned to a reference CT series --
https://github.com/qurit/rt-utils)
"""

import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
import requests
import SimpleITK as sitk

TCIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "NSCLC-Radiomics"
PATIENT_ID = "LUNG1-001"
ROI_NAME = "GTV-1"  # the structure name within the RTSTRUCT to extract

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "tcia"


def get_patient_series(collection: str, patient_id: str) -> list[dict]:
    """List all series (CT + RTSTRUCT, etc.) for one patient in one collection."""
    url = f"{TCIA_BASE_URL}/getSeries?Collection={collection}&PatientID={patient_id}"
    res = requests.get(url)
    res.raise_for_status()
    return res.json()


def download_series_dicom(series_uid: str, out_dir: Path) -> Path:
    """Download and extract one DICOM series by SeriesInstanceUID."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.glob("*.dcm")):
        print(f"  -> Already downloaded {series_uid} to {out_dir}")
        return out_dir

    url = f"{TCIA_BASE_URL}/getImage?SeriesInstanceUID={series_uid}"
    print(f"  -> Downloading series {series_uid}...")
    res = requests.get(url, stream=True)
    res.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(res.content)) as z:
        z.extractall(out_dir)
    return out_dir


def rtstruct_to_mask(ct_series_dir: Path, rtstruct_dir: Path, roi_name: str):
    """
    Convert an RTSTRUCT DICOM to a binary mask array aligned to the CT
    series grid, using rt-utils. Returns (mask_array_zyx, reference_ct_sitk_image).
    """
    from rt_utils import RTStructBuilder  # local import: optional dependency

    rtstruct_files = list(rtstruct_dir.glob("*.dcm"))
    if not rtstruct_files:
        raise FileNotFoundError(f"No RTSTRUCT DICOM file found in {rtstruct_dir}")

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(ct_series_dir),
        rt_struct_path=str(rtstruct_files[0]),
    )

    available_rois = rtstruct.get_roi_names()
    if roi_name not in available_rois:
        raise ValueError(
            f"ROI '{roi_name}' not found in RTSTRUCT. Available ROIs: {available_rois}. "
            f"Update ROI_NAME at the top of this script to match one of these exactly."
        )

    # rt-utils returns a mask with axes (row, col, slice) i.e. (y, x, z);
    # transpose to (z, y, x) to match this project's tensor convention
    # (see fastrad/io.py::_sitk_to_tensor).
    mask_yxz = rtstruct.get_roi_mask_by_name(roi_name)
    mask_zyx = np.transpose(mask_yxz, (2, 0, 1)).astype(np.uint8)

    # Build a reference SimpleITK image from the CT series to carry over
    # correct spacing/origin/direction for the mask.
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(ct_series_dir))
    reader.SetFileNames(dicom_names)
    ct_sitk = reader.Execute()

    return mask_zyx, ct_sitk


def download_one_case(patient_id: str, is_primary_case: bool = False) -> bool:
    """
    Download and convert one patient's CT + RTSTRUCT-derived mask.

    is_primary_case=True writes to the original lung1_image.nrrd /
    lung1_label.nrrd paths (for LUNG1-001, matching what
    run_numerical_parity.py / run_runtime_performance.py expect).
    Otherwise writes to tests/fixtures/tcia/multicase/<patient_id>/
    (matching what run_multicase_validation.py expects).

    Returns True on success, False if this case had to be skipped.
    """
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = FIXTURE_DIR / f"_download_work_{patient_id}"
    work_dir.mkdir(exist_ok=True)

    print(f"\n=== {patient_id} ===")
    print(f"Querying series for {patient_id} in collection {COLLECTION}...")
    series_list = get_patient_series(COLLECTION, patient_id)
    if not series_list:
        print(f"  SKIP: no series found for {patient_id} in {COLLECTION}.")
        return False

    ct_series = [s for s in series_list if s.get("Modality") == "CT"]
    rt_series = [s for s in series_list if s.get("Modality") == "RTSTRUCT"]

    if not ct_series:
        print(f"  SKIP: no CT series found for {patient_id}.")
        return False
    if not rt_series:
        print(f"  SKIP: no RTSTRUCT series found for {patient_id}.")
        return False

    ct_uid = ct_series[0]["SeriesInstanceUID"]
    rt_uid = rt_series[0]["SeriesInstanceUID"]
    print(f"  CT series: {ct_uid}")
    print(f"  RTSTRUCT series: {rt_uid}")

    ct_dir = download_series_dicom(ct_uid, work_dir / "ct")
    rt_dir = download_series_dicom(rt_uid, work_dir / "rtstruct")

    print(f"  Converting RTSTRUCT ROI '{ROI_NAME}' to a binary mask...")
    try:
        mask_zyx, ct_sitk = rtstruct_to_mask(ct_dir, rt_dir, ROI_NAME)
    except ImportError:
        print("  ERROR: rt-utils is not installed. Run: pip install rt-utils")
        return False
    except (FileNotFoundError, ValueError) as e:
        print(f"  SKIP: {e}")
        return False

    n_voxels = int(mask_zyx.sum())
    print(f"  Mask '{ROI_NAME}' extracted: {n_voxels} voxels")

    if is_primary_case:
        img_out = FIXTURE_DIR / "lung1_image.nrrd"
        mask_out = FIXTURE_DIR / "lung1_label.nrrd"
    else:
        case_dir = FIXTURE_DIR / "multicase" / patient_id
        case_dir.mkdir(parents=True, exist_ok=True)
        img_out = case_dir / "image.nrrd"
        mask_out = case_dir / "label.nrrd"

    sitk.WriteImage(ct_sitk, str(img_out))

    mask_sitk = sitk.GetImageFromArray(mask_zyx)
    mask_sitk.CopyInformation(ct_sitk)
    sitk.WriteImage(mask_sitk, str(mask_out))

    # Immediate verification, per the mask-label risk flagged during review:
    check = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_out)))
    unique_vals = np.unique(check)
    label_ok = list(unique_vals) == [0, 1]
    print(f"  Wrote {img_out}")
    print(f"  Wrote {mask_out}")
    print(
        f"  Label check: unique values = {unique_vals} "
        f"{'OK' if label_ok else '*** WARNING: NOT [0, 1] -- multi-structure mask, do not use as-is ***'}"
    )

    shutil.rmtree(work_dir, ignore_errors=True)
    return label_ok


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-primary",
        action="store_true",
        help="Re-download and OVERWRITE the primary case "
        "(lung1_image.nrrd/lung1_label.nrrd) even if it "
        "already exists. Off by default, because this "
        "file is the ground truth used by every other "
        "clinical-case benchmark script -- overwriting "
        "it silently once already caused every "
        "downstream benchmark to start measuring a "
        "different, un-flagged patient.",
    )
    parser.add_argument(
        "--skip-additional",
        action="store_true",
        help="Only handle the primary case; skip ADDITIONAL_CASES.",
    )
    args = parser.parse_args()

    # Additional cases beyond the primary LUNG1-001 case, for multi-case
    # validation (Section 2 of the review checklist's Part 5 guidance).
    # Populated from benchmarks/list_tcia_patients.py output.
    ADDITIONAL_CASES = [
        "LUNG1-002",
        "LUNG1-003",
        "LUNG1-004",
        "LUNG1-005",
        "LUNG1-006",
    ]

    primary_img = FIXTURE_DIR / "lung1_image.nrrd"
    primary_mask = FIXTURE_DIR / "lung1_label.nrrd"
    if primary_img.exists() and primary_mask.exists() and not args.force_primary:
        print(
            f"Primary case files already exist at {primary_img} / {primary_mask}. "
            f"Skipping re-download (every other clinical-case benchmark script "
            f"depends on this file staying stable -- pass --force-primary to "
            f"deliberately overwrite it, and re-run every downstream benchmark "
            f"afterward if you do)."
        )
    else:
        print("Downloading primary case (LUNG1-001)...")
        download_one_case(PATIENT_ID, is_primary_case=True)

    if args.skip_additional:
        return

    print(
        f"\nDownloading {len(ADDITIONAL_CASES)} additional cases for multi-case validation..."
    )
    results = {}
    for pid in ADDITIONAL_CASES:
        results[pid] = download_one_case(pid, is_primary_case=False)

    print("\n=== Summary ===")
    for pid, ok in results.items():
        print(f"  {pid}: {'OK' if ok else 'FAILED/SKIPPED'}")
    succeeded = [pid for pid, ok in results.items() if ok]
    print(
        f"\n{len(succeeded)}/{len(ADDITIONAL_CASES)} additional cases succeeded: {succeeded}"
    )
    print("Paste the succeeded list above into CASES in run_multicase_validation.py.")


if __name__ == "__main__":
    main()
