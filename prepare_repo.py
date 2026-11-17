"""Copy the dataset pack + the files downloaded from Colab into this repo, then check
that everything the app needs is present. Uses only the Python standard library.

    python prepare_repo.py --pack "C:/Downloads/Project_01" ^
                           --models models_and_reports.zip --reports reports.zip

--pack     the extracted Project_01 folder from the dataset pack (has data/, nltk_data/ ...)
--models   models_and_reports.zip downloaded from the Colab notebook (contains models/)
--reports  reports.zip downloaded from the Colab notebook (contains reports/)   [optional]
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
SKIP = {"images.npz"}                     # full training arrays stay OUT of GitHub

REQUIRED = [
    "data/main.csv", "data/feedback.csv", "data/meta.json", "data/topic_keywords.json",
    "data/example_inputs.csv", "data/prompt_test_cases.json",
    "models/supervised.joblib", "models/clusters.joblib",
    "models/cluster_descriptions.json", "models/images.keras",
]
RECOMMENDED = ["sample_images", "nltk_data", "source_notices", "reports/tabular_test_metrics.csv"]


def copy_tree(src: Path, dst: Path):
    for f in src.rglob("*"):
        if f.is_file() and f.name not in SKIP and "__MACOSX" not in f.parts:
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)


def unzip_folder(zip_path: Path, folder: str):
    """Extract <folder>/... from a Colab zip into the repo (handles nested paths)."""
    with zipfile.ZipFile(zip_path) as z:
        n = 0
        for name in z.namelist():
            parts = Path(name).parts
            if folder in parts and not name.endswith("/"):
                rel = Path(*parts[parts.index(folder):])
                out = REPO / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(name))
                n += 1
    print(f"  {zip_path.name}: {n} files -> {folder}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, help="extracted Project_01 folder")
    ap.add_argument("--models", type=Path, help="models_and_reports.zip from Colab")
    ap.add_argument("--reports", type=Path, help="reports.zip from Colab (optional)")
    a = ap.parse_args()

    if a.pack:
        pack = a.pack
        if not (pack / "data").exists():                       # user pointed at the parent
            found = [p for p in pack.rglob("Project_01") if (p / "data").exists()]
            if not found:
                sys.exit(f"No data/ folder found in {pack}")
            pack = found[0]
        print("Pack:", pack)
        for folder in ("data", "sample_images", "nltk_data", "source_notices"):
            if (pack / folder).exists():
                copy_tree(pack / folder, REPO / folder)
                print(f"  copied {folder}/")
        if (pack / "data" / "sample_images").exists():          # older pack layout
            copy_tree(pack / "data" / "sample_images", REPO / "sample_images")
    if a.models:
        unzip_folder(a.models, "models")
    if a.reports:
        unzip_folder(a.reports, "reports")

    print("\nChecklist")
    ok = True
    for p in REQUIRED:
        exists = (REPO / p).exists()
        ok &= exists
        print(f"  [{'OK' if exists else 'MISSING'}] {p}")
    for p in RECOMMENDED:
        print(f"  [{'OK' if (REPO / p).exists() else 'optional - missing'}] {p}")
    if (REPO / "data/images.npz").exists():
        print("  NOTE: data/images.npz is present locally; .gitignore keeps it out of GitHub.")
    if (REPO / ".streamlit/secrets.toml").exists():
        print("  NOTE: .streamlit/secrets.toml found; .gitignore keeps it out of GitHub.")
    print("\nReady for GitHub!" if ok else "\nSome required files are missing (see above).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
