import io
import zipfile
from pathlib import Path

import requests


def get_rider_pairs(limit=10):
    url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?Collection=RIDER%20Lung%20CT&Modality=CT"
    res = requests.get(url)
    res.raise_for_status()
    series = res.json()

    patients = {}
    for s in series:
        pid = s["PatientID"]
        if pid not in patients:
            patients[pid] = []
        patients[pid].append(s)

    pairs = []
    for pid, s_list in patients.items():
        if len(s_list) >= 2:
            pairs.append(
                (pid, s_list[0]["SeriesInstanceUID"], s_list[1]["SeriesInstanceUID"])
            )

    return sorted(pairs)[:limit]


def download_series(uid, extract_dir):
    out_dir = Path(extract_dir)
    if out_dir.exists() and len(list(out_dir.glob("*.dcm"))) > 50:
        print(f"  -> Already downloaded {uid}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={uid}"

    print(f"  -> Downloading {uid}...")
    res = requests.get(url, stream=True)
    res.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(res.content)) as z:
        z.extractall(out_dir)


def main():
    dest = Path(__file__).parent.parent / "tests" / "fixtures" / "tcia" / "rider"
    dest.mkdir(parents=True, exist_ok=True)

    print("Querying RIDER Lung CT test-retest pairs...")
    pairs = get_rider_pairs(10)

    for idx, (pid, uid1, uid2) in enumerate(pairs):
        print(f"[{idx + 1}/10] Patient: {pid}")
        download_series(uid1, dest / pid / "scan1")
        download_series(uid2, dest / pid / "scan2")

    print("Download complete.")


if __name__ == "__main__":
    main()
