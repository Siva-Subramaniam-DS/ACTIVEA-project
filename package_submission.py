"""Packaging script for ACTIVEA coursework submission to Blackboard.
Creates a clean, production-ready .zip archive conforming to all Aston University guidelines.
"""

import os
import zipfile
from pathlib import Path

ARCHIVE_NAME = "ACTIVEA_Codebase_Siva_Subramaniam_Raju_260322784.zip"
SHORT_ARCHIVE_NAME = "ACTIVEA_Codebase.zip"

INCLUDED_DIRECTORIES = [
    "activea",
    "Model",
    "tests",
    "web",
    "results",
]

INCLUDED_FILES = [
    "main.py",
    "demo.py",
    "remaining_algorithms.py",
    "run_ml_comparison.py",
    "evaluate_with_model.py",
    "serve_planner.py",
    "run_all_experiments.py",
    "run_comparison.py",
    "convert_endomondo_to_csv.py",
    "create_notebook.py",
    "ACTIVEA_EDA_and_Model_Optimization.ipynb",
    "requirements.txt",
    "README.md",
]

EXCLUDED_EXTENSIONS = {".pyc", ".pyo", ".log", ".tmp", ".bak", ".swp"}
EXCLUDED_PATTERNS = {"__pycache__", ".git", ".cursor", ".system_generated", "scratch", ".pytest_cache"}
EXCLUDED_LARGE_FILES = {"endomondoHR.json", "endomondoHR.csv"}

def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDED_PATTERNS:
            return True
    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True
    if path.name in EXCLUDED_LARGE_FILES:
        return True
    return False

def create_submission_zip(output_path: str):
    root_dir = Path(".")
    total_files = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add individual top-level files
        for fname in INCLUDED_FILES:
            fpath = root_dir / fname
            if fpath.exists() and not should_exclude(fpath):
                zipf.write(fpath, arcname=fname)
                total_files += 1

        # Add directories recursively
        for dname in INCLUDED_DIRECTORIES:
            dpath = root_dir / dname
            if dpath.exists():
                for fpath in dpath.rglob("*"):
                    if fpath.is_file() and not should_exclude(fpath):
                        arcname = fpath.relative_to(root_dir)
                        zipf.write(fpath, arcname=arcname)
                        total_files += 1

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Archive Created] {output_path}")
    print(f"  * Total Files Archived: {total_files}")
    print(f"  * Archive Size: {size_mb:.2f} MB")
    return size_mb

if __name__ == "__main__":
    s1 = create_submission_zip(ARCHIVE_NAME)
    s2 = create_submission_zip(SHORT_ARCHIVE_NAME)
    print(f"\nSuccessfully generated submission archives.")
