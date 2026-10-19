"""
List candidate patients from a TCIA collection, confirming each one
actually has both a CT series and an RTSTRUCT series available (the two
things download_tcia_sample.py needs). Prints a ready-to-paste Python list
you can drop into run_multicase_validation.py's CASES list.

*** UNVERIFIED: written without network access to TCIA. Uses the same
    NBIA REST API base URL already proven to work in download_rider_pairs.py
    and download_tcia_sample.py in this repo, but the specific
    getPatient endpoint used here has not been tested. ***

USAGE:
    python benchmarks/list_tcia_patients.py
    python benchmarks/list_tcia_patients.py --collection NSCLC-Radiomics --limit 10
"""

import argparse

import requests

TCIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def get_all_patients(collection: str) -> list[str]:
    """Returns every PatientId in a collection."""
    url = f"{TCIA_BASE_URL}/getPatient?Collection={collection}"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()
    return sorted(p["PatientId"] for p in data)


def check_patient_modalities(collection: str, patient_id: str) -> dict:
    """Returns which modalities (CT, RTSTRUCT, etc.) exist for one patient."""
    url = f"{TCIA_BASE_URL}/getSeries?Collection={collection}&PatientID={patient_id}"
    res = requests.get(url)
    res.raise_for_status()
    series_list = res.json()
    modalities = {s.get("Modality") for s in series_list}
    return {
        "patient_id": patient_id,
        "has_ct": "CT" in modalities,
        "has_rtstruct": "RTSTRUCT" in modalities,
        "n_series": len(series_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="NSCLC-Radiomics")
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="How many candidate patients to check (checking is slow, "
        "one API call per patient)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["LUNG1-001"],
        help="Patient IDs to skip (e.g. cases you already used)",
    )
    args = parser.parse_args()

    print(f"Fetching full patient list for collection '{args.collection}'...")
    all_patients = get_all_patients(args.collection)
    print(f"Found {len(all_patients)} total patients in this collection.\n")

    candidates = [p for p in all_patients if p not in args.exclude][: args.limit]
    print(
        f"Checking modality availability for the first {len(candidates)} "
        f"candidates (excluding {args.exclude})...\n"
    )

    usable = []
    for pid in candidates:
        info = check_patient_modalities(args.collection, pid)
        status = "OK" if (info["has_ct"] and info["has_rtstruct"]) else "SKIP"
        print(
            f"  [{status}] {pid}: CT={info['has_ct']}, "
            f"RTSTRUCT={info['has_rtstruct']}, {info['n_series']} series total"
        )
        if status == "OK":
            usable.append(pid)

    print(
        f"\n{len(usable)} of {len(candidates)} checked patients have both "
        f"CT and RTSTRUCT and are usable with download_tcia_sample.py's "
        f"pipeline (after changing its PATIENT_ID for each).\n"
    )

    print("Ready-to-paste list for reference (edit down to however many you want):")
    print(usable)


if __name__ == "__main__":
    main()
