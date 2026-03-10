"""
Skrypt do pobierania zbioru danych PTB-XL z PhysioNet.
Użycie: python data/download_ptbxl.py
"""
import os
import subprocess
import sys
import zipfile
import urllib.request
import shutil

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
DATASET_DIR = os.path.join(DATA_DIR, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
ZIP_PATH = os.path.join(DATA_DIR, "ptb-xl.zip")


def download_with_wget():
    """Try downloading with wget (faster, shows progress)."""
    try:
        subprocess.run(
            ["wget", "-O", ZIP_PATH, DATASET_URL],
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_with_urllib():
    """Fallback: download with urllib."""
    print("Pobieranie PTB-XL (~1.7 GB)... To może zająć kilka minut.")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
        mb = downloaded / (1024 * 1024)
        print(f"\r  {pct}% ({mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH, reporthook=_progress)
    print()


def extract_zip():
    """Extract the downloaded zip."""
    print("Rozpakowywanie...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(ZIP_PATH)
    print("Usunięto archiwum ZIP.")


def verify():
    """Verify the dataset was downloaded correctly."""
    csv_path = os.path.join(DATASET_DIR, "ptbxl_database.csv")
    scp_path = os.path.join(DATASET_DIR, "scp_statements.csv")
    rec_dir = os.path.join(DATASET_DIR, "records500")

    ok = True
    for p, name in [(csv_path, "ptbxl_database.csv"), (scp_path, "scp_statements.csv"), (rec_dir, "records500/")]:
        if os.path.exists(p):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ Brak: {name}")
            ok = False
    return ok


def main():
    if os.path.isdir(DATASET_DIR):
        print(f"Dataset już istnieje: {DATASET_DIR}")
        if verify():
            print("Wszystko OK.")
            return
        print("Dataset niekompletny, pobieram ponownie...")
        shutil.rmtree(DATASET_DIR)

    print(f"Pobieranie PTB-XL do: {DATA_DIR}")
    if not download_with_wget():
        download_with_urllib()

    extract_zip()

    if verify():
        print("\nDataset PTB-XL pobrany i zweryfikowany pomyślnie!")
    else:
        print("\nBŁĄD: Dataset niekompletny. Spróbuj pobrać ponownie.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
