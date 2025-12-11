import sys
import subprocess
from pathlib import Path
from tqdm import tqdm
import time
import concurrent.futures
import argparse
import json, csv

"""
flags:
--workers X
"""

PYTHON = sys.executable

PDF_DIR = Path("batch_pdfs").resolve()
BATCH_JSON_DIR = Path("batch_output").resolve()
BATCH_FIG_DIR = Path("batch_output/figures").resolve()
BATCH_EXTRACTIONS_DIR = Path("batch_extractions").resolve()

# make sure output directories exist
for d in (BATCH_JSON_DIR, BATCH_FIG_DIR, BATCH_EXTRACTIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Collect PDFs
pdf_files = sorted(PDF_DIR.glob("*.pdf"))
if not pdf_files:
    print(f"No PDF files found in {PDF_DIR}. Place your PDFs there and re-run.")
    sys.exit(0)

start_time = time.time()
worker_times = []

parser = argparse.ArgumentParser(description="Batch process PDFs with parallel workers.")
parser.add_argument('--workers', type=int, default=None, help="Number of parallel worker processes (default: number of CPU cores)")
args = parser.parse_args()

PROCESS_PDF_SCRIPT = "processPDF.py"
USES_OCR = True

# Process each PDF in parallel
def handle_pdf(pdf_file):
    start = time.time()
    name = pdf_file.stem

    json_out = BATCH_JSON_DIR / f"{name}.json"
    fig_out = BATCH_FIG_DIR / name / f"{name}_ocr.json"
    lipids_out = BATCH_EXTRACTIONS_DIR / f"{name}_lipids.json"

    if not json_out.exists() or not fig_out.exists():
        (BATCH_FIG_DIR / name).mkdir(parents=True, exist_ok=True)
        cmd = [PYTHON, PROCESS_PDF_SCRIPT,
               "--pdf", str(pdf_file)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"processPDF.py failed for {pdf_file.name}: returncode={e.returncode}")
            elapsed = time.time() - start
            return

    if not lipids_out.exists():
        cmd = [PYTHON, "APIcall.py",
               "--pdf_json", str(json_out),
               "--fig_json", str(fig_out),
               "--out", str(lipids_out),
               "--max_tokens", "64000"]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"APIcall.py failed for {name}: returncode={e.returncode}")

    elapsed = time.time() - start
    return (name, elapsed)

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(handle_pdf, pdf_files), total=len(pdf_files), desc="Processing PDFs", unit="pdf"))
        worker_times.extend([r for r in results if r])

    elapsed_time = time.time() - start_time

    for name, seconds in sorted(worker_times, key=lambda x: x[1], reverse=True):
        pass

    # CSV conversion of merged JSON
    merged_json_path = BATCH_EXTRACTIONS_DIR / "merged_lipids.json"
    merged_csv_path = merged_json_path.with_suffix(".csv")

    # Merge all lipid JSONs into merged_lipids.json
    try:
        per_file_jsons = [p for p in sorted(BATCH_EXTRACTIONS_DIR.glob("*.json"))
                          if p.name != merged_json_path.name]
        merged_entries = []
        for jf in per_file_jsons:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                merged_entries.extend(data)
            elif isinstance(data, dict) and "lipids" in data and isinstance(data["lipids"], list):
                merged_entries.extend(data["lipids"])

        if per_file_jsons:
            with open(merged_json_path, "w", encoding="utf-8") as f:
                json.dump(merged_entries, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to merge JSON files: {e}")

    if merged_json_path.exists():
        try:
            with open(merged_json_path, "r", encoding="utf-8") as f:
                merged_data = json.load(f)
            if isinstance(merged_data, dict) and "lipids" in merged_data:
                merged_data = merged_data["lipids"]
            if isinstance(merged_data, list) and merged_data:
                fields = [
                    "Study_ID", "Year", "Species", "Sex", "Genetic background", "Tissue origin",
                    "Polarity", "Ionization technology", "Matrix", "Mass analyzer technology",
                    "Spatial resolution", "m/z", "ID_original", "Lipid name", "Sourcing"
                ]

                def _fill_cell(v):
                    if v is None:
                        return "/"
                    s = str(v).strip()
                    return s if s else "/"
                with open(merged_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    for row in merged_data:
                        row_out = {k: _fill_cell(row.get(k, None)) for k in fields}
                        writer.writerow(row_out)
        except Exception as e:
            pass