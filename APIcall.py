#!/usr/bin/env python3
"""
To use alone (i.e. not through batchRun.py):
  export ANTHROPIC_API_KEY='...'
  python APIcall.py \
      --pdf_json "json_output/Li2021OCR copy.json" \
      --fig_json "figure_ocr_output/Li2021OCR copy_ocr.json" \
      --out "extractions/Li2021_lipids.json" \
      --max_tokens XXXX
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from anthropic import Anthropic, APIStatusError, BadRequestError
from typing import List, Dict, Any

def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    # remove leading ```json / ``` and trailing ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


LIPID_FIELDS = [
    "Study_ID", "Year", "Species", "Sex", "Genetic background",
    "Tissue origin", "Polarity", "Ionization technology", "Matrix",
    "Mass analyzer technology", "Spatial resolution", "m/z",
    "ID_original", "Lipid name", "Sourcing"
]

def normalize_str(s: str | None) -> str | None:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", s.strip())
    return s if s else None

def normalize_year(s: str | None) -> int | None:
    if not s:
        return None
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else None

def normalize_by_dict(s: str | None, mapping: dict) -> str | None:
    if not s:
        return None
    t = s.lower()
    for k, v in mapping.items():
        if k in t:
            return v
    return normalize_str(s)

def normalize_polarity(s: str | None) -> str | None:
    mapping = {"neg": "negative", "pos": "positive"}
    return normalize_by_dict(s, mapping)

def normalize_ionization(s: str | None) -> str | None:
    mapping = {
        "maldi": "MALDI",
        "desi": "DESI",
        "tof-sims": "ToF-SIMS",
        "tof sims": "ToF-SIMS",
        "sims": "ToF-SIMS",
        "esi": "ESI"
    }
    return normalize_by_dict(s, mapping)

def normalize_analyzer(s: str | None) -> str | None:
    mapping = {
        "orbitrap": "Orbitrap",
        "ft-icr": "FT-ICR",
        "ft icr": "FT-ICR",
        "q-tof": "Q-TOF",
        "qtof": "Q-TOF",
        "q tof": "Q-TOF",
        "tof": "TOF",
        "triple quad": "Triple Quadrupole",
        "triple quadrupole": "Triple Quadrupole"
    }
    if not s:
        return None
    t = s.lower()
    if "tof" in t and "q" in t:
        for k in ["q-tof", "qtof", "q tof"]:
            if k in t:
                return "Q-TOF"
    if "tof" in t and "q" not in t:
        return "TOF"
    for k, v in mapping.items():
        if k in t:
            return v
    return normalize_str(s)

def normalize_spatial_res(s: str | None) -> str | None:
    if not s:
        return None
    s = normalize_str(s)
    s = s.replace("um", "µm").replace("μm", "µm")
    return s

def normalize_mz(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        m = re.search(r"(\d+\.\d+|\d+)", str(v))
        return float(m.group(1)) if m else None

# save JSON with utf-8 
def save_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        obj = dict(it)
        obj["Year"] = normalize_year(str(obj.get("Year"))) if obj.get("Year") is not None else None
        obj["Species"] = normalize_str(obj.get("Species")) or "/"
        obj["Sex"] = normalize_str(obj.get("Sex")) or "/"
        obj["Genetic background"] = normalize_str(obj.get("Genetic background")) or "/"
        obj["Tissue origin"] = normalize_str(obj.get("Tissue origin")) or "/"
        obj["Polarity"] = normalize_polarity(obj.get("Polarity")) or "/"
        obj["Ionization technology"] = normalize_ionization(obj.get("Ionization technology")) or "/"
        obj["Matrix"] = normalize_str(obj.get("Matrix")) or "/"
        obj["Mass analyzer technology"] = normalize_analyzer(obj.get("Mass analyzer technology")) or "/"
        obj["Spatial resolution"] = normalize_spatial_res(obj.get("Spatial resolution")) or "/"
        obj["m/z"] = normalize_mz(obj.get("m/z"))
        obj["ID_original"] = normalize_str(obj.get("ID_original")) or "/"
        obj["Lipid name"] = normalize_str(obj.get("Lipid name")) or "/"
        out.append(obj)
    return out

# prompt
PROMPT = """You are analyzing extracted text from a scientific PDF describing lipid identification. Extract and return a json file where each lipid is a separate object with fields like:
Study_ID – Year – Species – Sex – Genetic background – Tissue origin – Polarity – Ionization technology – Matrix – Mass analyzer technology – Spatial resolution – m/z – ID_original – Lipid name – Sourcing
Only return valid entries you can confidently extract and pay special attention to tables or tabular data and extract their lipids in context of the text.

Do not skip data in tables, structured formats or lists, even if they differ slightly from descriptions. Return empty strings for missing fields, but do not skip the entry.

Return your answer strictly as compact JSON with shared fields:
```json
{
  "fields": ["Study_ID", "Year", "Species", "Sex", "Genetic background", "Tissue origin", "Polarity", "Ionization technology", "Matrix", "Mass analyzer technology", "Spatial resolution", "m/z", "ID_original", "Lipid name", "Sourcing"],
  "lipids": [
    ["Tanaka_2017", 2017, "Mus musculus", "Male", "C57BL/6J", "Liver", "Positive", "MALDI", "DHB", "Q-TOF", "50 µm", 734.55, "PC 34:1", "Phosphatidylcholine (PC 34:1)", "Table 1"],
    ...
  ]
}
```
Each lipid should be a list of values in the same order as `fields`, using `null` for missing fields.
"""

def build_context(pdf_chunks: List[str], fig_items: List[Dict[str, Any]], max_chars: int = 12000) -> List[str]:
    """
    Build multiple context chunks (<= max_chars each) from PDF text chunks and figure OCR JSON.
    The order of context blocks is determined by the context_order list.
    """
    blocks = []

    #1) figure sections only
    for i, fig in enumerate(fig_items, start=1):
        important_bits = []
        if fig.get("refined_json_image"):
            important_bits.append(f"Description: {fig['refined_json_image']}")
        elif fig.get("caption"):
            important_bits.append(f"Caption: {fig['caption']}")
        elif fig.get("ocr_text"):
            important_bits.append(f"OCR: {fig['ocr_text']}")
        if important_bits:
            fig_text = f"[FIGURE {i}]\n" + "\n".join(important_bits)
            for seg in _split_hard(fig_text, max_chars):
                blocks.append(seg)

    #2) External tables, if any
    for chunk in pdf_chunks:
        if isinstance(chunk, dict) and "external_table_data" in chunk:
            table = chunk["external_table_data"]
            filename = table.get("filename", "unknown_table")
            records = table.get("records", [])
            if records:
                headers = list(records[0].keys())
                lines = [f"[TABLE from {filename}]"]
                lines.append("\t".join(headers))
                for row in records:
                    lines.append("\t".join(str(row.get(h, "")) for h in headers))
                for seg in _split_hard("\n".join(lines), max_chars):
                    blocks.append(seg)



    #3) parsed text
    prose_lines = []
    for ch in pdf_chunks:
        if isinstance(ch, str):
            prose_lines.extend([l.strip() for l in ch.splitlines() if l.strip()])
    prose_block = "[TEXT]\n" + "\n".join(prose_lines)
    for seg in _split_hard(prose_block, max_chars):
        blocks.append(seg)

    chunks = []
    cur = ""
    for b in blocks:
        if len(cur) + len(b) + 2 > max_chars:
            if cur:
                chunks.append(cur)
            cur = b
        else:
            cur += ("\n\n" + b) if cur else b
    if cur:
        chunks.append(cur)

    return chunks


def _split_hard(text: str, max_len: int) -> List[str]:
    if len(text) <= max_len:
        return [text]
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

# calls claude with prompt in streaming mode
def call_claude(client: Anthropic, model: str, context_text: str, max_tokens: int = 2000, retries: int = 2, sleep_s: float = 2.0) -> dict | str:
  
    prompt = PROMPT + "\n\n" + context_text

    def _content_blocks(txt: str):
        return [{"type": "text", "text": txt}]

    last_err = None
    for attempt in range(retries + 1):
        try:
            stream = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=PROMPT,
                messages=[{"role": "user", "content": _content_blocks(prompt)}],
                temperature=0,
                stream=True
            )
            # Accumulate streamed text
            text_parts = []
            try:
                for event in stream:
                    if hasattr(event, "delta") and hasattr(event.delta, "text"):
                        text_parts.append(event.delta.text)
            except Exception as e:
                print(f"Error while reading stream: {e}")

            text = _strip_code_fences("".join(text_parts).strip())
            if text:
                return text
        except (APIStatusError, BadRequestError) as e:
            last_err = e
            code = getattr(e, "status_code", None)
            msg = getattr(e, "message", str(e))
            print(f"Anthropic API error (attempt {attempt+1}/{retries+1}): {code} - {msg}")
            if code in (529, 500, 503):
                time.sleep(sleep_s * (attempt + 1))
                continue
        except Exception as e:
            last_err = e
            print(f"Unexpected error (attempt {attempt+1}/{retries+1}): {e}")
            time.sleep(sleep_s * (attempt + 1))
            continue

    print(f"Giving up after retries. Last error: {last_err}")
    return "[]"

def extract_json_block(s_or_dict) -> List[Dict[str, Any]]:
    s = _strip_code_fences(str(s_or_dict).strip())

    # Parse JSON directly
    data = json.loads(s)

    fields = data["fields"]
    rows = data["lipids"]

    # convert each row (list) into a dict
    lipids = []
    for row in rows:
        row = (row + [None] * len(fields))[:len(fields)]
        lipids.append(dict(zip(fields, row)))

    return lipids


def parse_cli():
    ap = argparse.ArgumentParser(description="Extract lipid entries with Claude from JSON + figure OCR.")
    ap.add_argument("--pdf_json", required=True, help="Path to json_output/<study>.json")
    ap.add_argument("--fig_json", required=True, help="Path to figure_ocr_output/<study>_ocr.json")
    ap.add_argument("--out", required=True, help="Output JSON path, e.g., extractions/<study>_lipids.json")
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Claude model")
    ap.add_argument("--chunk_chars", type=int, default=12000, help="Max characters per context chunk")
    ap.add_argument("--max_tokens", type=int, default=3000, help="Max output tokens per call")
    args = ap.parse_args()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Export it before running.")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError:
        raise SystemExit("ANTHROPIC_API_KEY contains non-ASCII characters. Re-export with the real key (no … or smart quotes).")
    return args, api_key

def main():
    args, api_key = parse_cli()
    client = Anthropic(api_key=api_key)
    pdf_path = Path(args.pdf_json)
    fig_path = Path(args.fig_json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pdf_chunks = json.loads(pdf_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read {pdf_path}: {e}")
    try:
        fig_items = json.loads(fig_path.read_text(encoding="utf-8"))
        if not isinstance(fig_items, list):
            fig_items = []
    except Exception:
        fig_items = []
    contexts = build_context(pdf_chunks, fig_items, max_chars=args.chunk_chars)
    combined_text = "\n\n".join(contexts)
    context_dump_path = out_path.parent / (out_path.stem + "_context.txt")
    context_dump_path.write_text(combined_text, encoding="utf-8")
    effective_model = args.model
    # Combined context logic
    if len(combined_text) < 500000:
        t_start = time.time()
        resp_payload = call_claude(client, model=effective_model, context_text=combined_text, max_tokens=args.max_tokens)
        elapsed = time.time() - t_start
        # Save raw Claude output for debugging
        raw_dir = out_path.parent / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_out_path = raw_dir / (out_path.stem + "_raw.json")
        save_json(raw_out_path, resp_payload)
        items = extract_json_block(resp_payload)
        if items and isinstance(items[0], list):
            items = [
                dict(zip(LIPID_FIELDS, row + [None] * (len(LIPID_FIELDS) - len(row))))
                for row in items if isinstance(row, list)
            ]
        if not items:
            print("No items parsed from Claude output.")
        all_items = normalize_items(items)
        save_json(out_path, all_items)
        print(f"\nSaved {len(all_items)} lipid entries to {out_path}")

if __name__ == "__main__":
    main()