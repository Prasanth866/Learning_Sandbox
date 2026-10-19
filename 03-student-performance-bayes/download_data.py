"""Data downloader for the UCI Student Performance dataset.

Downloads the student performance dataset from the UCI Machine Learning Repository,
extracts the required CSV files (`student-mat.csv` and `student-por.csv`), and verifies
their presence under `data/raw/`.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

UCI_DATASET_URL: str = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DEFAULT_RAW_DIR: Path = Path(__file__).resolve().parent / "data" / "raw"


def download_and_extract(
    dest_dir: Path = DEFAULT_RAW_DIR,
    url: str = UCI_DATASET_URL,
    timeout: int = 30,
) -> None:
    """Download and extract the UCI Student Performance dataset.

    Parameters
    ----------
    dest_dir : Path, optional
        Target directory where CSV files will be stored. Defaults to `data/raw/`.
    url : str, optional
        URL to the dataset zip archive.
    timeout : int, optional
        HTTP request timeout in seconds. Defaults to 30.

    Raises
    ------
    requests.HTTPError
        If the HTTP download request fails.
    zipfile.BadZipFile
        If the downloaded content is not a valid zip archive.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    mat_file = dest_dir / "student-mat.csv"
    por_file = dest_dir / "student-por.csv"

    if mat_file.exists() and por_file.exists():
        logger.info("Data files already exist in %s. Skipping download.", dest_dir)
        return

    logger.info("Downloading UCI Student Performance dataset from %s ...", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    logger.info("Extracting CSV files to %s ...", dest_dir)
    with zipfile.ZipFile(io.BytesIO(response.content)) as outer_zip:
        inner_bytes = outer_zip.read("student.zip")
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
            for name in ["student-mat.csv", "student-por.csv"]:
                inner_zip.extract(name, dest_dir)

    logger.info("Successfully extracted dataset files:")
    for filepath in [mat_file, por_file]:
        size_bytes = filepath.stat().st_size
        logger.info("  - %s (%d bytes)", filepath.name, size_bytes)


if __name__ == "__main__":
    download_and_extract()
