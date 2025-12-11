## Pipeline Overview

### 1. Preprocess PDF (`processPDF.py`)
- Parse PDF and in-text tables
- Figure & caption extraction and OCR
- Call Claude Haiku 4.5 to interpret figure, caption & OCR
- Writes two JSONs: `PDFstem.json` (combined extracted text) and `PDFstem_ocr.json` (OCR + Claude interpretation + caption)

### 2. LLM Data Extraction (`APIcall.py`)
- Combine context
- Send context + prompt to Claude Sonnet 4.5
- Normalize and write lipid objects to `PDFstem_lipids.json`

### 3. Batch Run
- Runs `processPDF.py` and `APIcall.py` in parallel
- Merges all lipid outputs to `merged_lipids.json` and converts to CSV

## Usage 
- Place all PDFs into `batch_pdfs/`
- Run `python batchRun.py --workers X`
- workers X for allocating multiple cores
