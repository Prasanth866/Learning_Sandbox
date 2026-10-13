import io
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_and_extract(dest_dir: Path = RAW_DIR) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    mat_file = dest_dir / "student-mat.csv"
    por_file = dest_dir / "student-por.csv"

    if mat_file.exists() and por_file.exists():
        print(f"Data files already exist in {dest_dir}.")
        return

    print("Downloading UCI Student Performance dataset...")
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as outer_zip:
        inner_bytes = outer_zip.read("student.zip")
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
            for name in ["student-mat.csv", "student-por.csv"]:
                inner_zip.extract(name, dest_dir)

    print("Extracted successfully:")
    for f in [mat_file, por_file]:
        print(f"  {f.name}: {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    download_and_extract()
