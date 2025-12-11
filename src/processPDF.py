import os
import json
import re
import fitz
from pathlib import Path
from PIL import Image
import easyocr
import numpy as np
from base64 import b64encode
import io
import time
import concurrent.futures

def attach_tabular_supplement(pdf_path: Path, all_text: list):
    """Attach any tabular supplements (CSV/XLSX) that match the PDF stem.
    The function appends one dict per attached table to `all_text` with shape:
      {"external_table": {"filename": <name>, "data": [ ...records... ]}}
    """
    base_stem = pdf_path.stem.lower()
    dir_path = pdf_path.parent

    exts = {".csv", ".xlsx", ".xls"}

    attached = 0
    # pick those that contain the PDF base stem
    for p in sorted(dir_path.iterdir()):
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        if base_stem not in name_lower:
            continue
        suffix = p.suffix.lower()
        if suffix not in exts:
            continue

        try:
            # attach the file path
            all_text.append({
                "external_table_reference": {
                    "filename": p.name,
                    "path": str(p.resolve())
                }
            })
            print(f"Attached tabular reference: {p.name}")
            attached += 1
        except Exception as e:
            print(f"Failed to attach table reference {p.name}: {e}")

    if attached == 0:
        # No files matched; do nothing (silent)
        return

import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


reader = easyocr.Reader(['en'], gpu=False)

def prepare_image_for_claude(image_path, target_long_edge=1200):
    """
     make an image <5 MB for Claude.
     returns: (media_type, base64_string)
    """
    def count_colors_for_hint(img):
        # color count on a small thumbnail to decide PNG vs JPEG
        thumb = img.convert("RGB")
        thumb.thumbnail((256, 256), Image.LANCZOS)
        colors = thumb.getcolors(maxcolors=256*256)
        return len(colors) if colors else 256*256

    def resize_to_long_edge(im, long_edge):
        w, h = im.size
        scale = long_edge / float(max(w, h))
        if scale >= 1.0:
            return im.copy()
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return im.resize((new_w, new_h), Image.LANCZOS)

    with Image.open(image_path) as img:
        # Remove alpha for JPEG path but keep original for possible PNG
        img_no_meta = Image.new("RGBA", img.size)
        img_no_meta.paste(img)
        has_alpha = img_no_meta.mode in ("RGBA", "LA") or ("transparency" in img_no_meta.info)

        color_count = count_colors_for_hint(img_no_meta)
        prefer_png = (color_count < 512) and not has_alpha 
        
        long_edges = [target_long_edge, 1000, 800, 640, 512, 480]
        jpeg_qualities = [85, 75, 65, 55, 45, 35, 25]
        found_suitable = False
        # try resize steps first, then fallback to JPEG if still too big
        for le in long_edges:
            candidate = resize_to_long_edge(img_no_meta, le)

            if prefer_png and not found_suitable:
                buf = io.BytesIO()
                # Convert to palette PNG if low color count to shrink further
                pal = candidate.convert("P", palette=Image.Palette.ADAPTIVE, colors=min(256, max(16, color_count)))
                pal.save(buf, format="PNG", optimize=True)
                png_bytes = buf.getvalue()
                size_mb = len(png_bytes) / (1024 * 1024)
                print(f"  -> PNG @ long_edge={le}px, colors≈{color_count}, size={size_mb:.2f} MB")
                if size_mb <= 5:
                    found_suitable = True
                    return "image/png", b64encode(png_bytes).decode("utf-8")

            # Try JPEG in descending qualities
            for q in jpeg_qualities:
                if found_suitable:
                    break
                rgb = candidate.convert("RGB")
                buf = io.BytesIO()
                try:
                    rgb.save(buf, format="JPEG", quality=q, optimize=True, progressive=True, subsampling=2)
                except TypeError:
                    rgb.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
                jpg_bytes = buf.getvalue()
                size_mb = len(jpg_bytes) / (1024 * 1024)
                print(f"  -> JPEG @ long_edge={le}px, q={q}, size={size_mb:.2f} MB")
                if size_mb <= 5:
                    found_suitable = True
                    return "image/jpeg", b64encode(jpg_bytes).decode("utf-8")

        # Final fallback
        tiny = resize_to_long_edge(img_no_meta, 360).convert("RGB")
        buf = io.BytesIO()
        try:
            tiny.save(buf, format="JPEG", quality=30, optimize=True, progressive=True, subsampling=2)
        except TypeError:
            tiny.save(buf, format="JPEG", quality=30, optimize=True, progressive=True)
        jpg_bytes = buf.getvalue()
        size_mb = len(jpg_bytes) / (1024 * 1024)
        print(f"  -> fallback  JPEG 360px, size={size_mb:.2f} MB")
        return "image/jpeg", b64encode(jpg_bytes).decode("utf-8")

def refine_with_claude_image(caption, image_path):
    fallback_prefix = "Caption: "
    cap_stripped = caption.strip()
    if not cap_stripped or len(cap_stripped) < 10:
        caption = fallback_prefix + (cap_stripped if cap_stripped else "No caption provided.")
    elif not cap_stripped.lower().startswith("caption:"):
        caption = fallback_prefix + cap_stripped

    # prepare the image under 5MB
    original_bytes = Path(image_path).read_bytes()
    print(f"Original image size: {len(original_bytes)/(1024*1024):.2f} MB")
    media_type, img_b64 = prepare_image_for_claude(image_path)
    print(f"Prepared image media_type={media_type}")

    def call_claude():
        prompt = (
            "You are assisting in scientific data extraction from research figures. "
            "Summarize only the essential information related to lipid identification, such as species, tissue, polarity, ionization technology, matrix, analyzer, and spatial resolution. "
            "Include only concise context if necessary to clarify the measurement or condition. Do not describe visuals, shapes, or colors."
        )
        return client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + "\n\nFigure context:\n" + caption},
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}}
                    ]
                }
            ]
        )

    response = call_claude()
    if (hasattr(response, 'error') and "Overloaded" in str(response.error)) or not response.content:
        time.sleep(2)
        response = call_claude()

    plain_text = "\n".join(block.text for block in response.content if block.type == "text") if response.content else ""
    return plain_text

def extract_and_ocr_image(doc, page, page_index, image_index, pdf_name, figure_output_dir, page_text, page_rendered):
    try:
        # Extract image and handle fallback logic with page_rendered (read full page fallback)
        xref = page.get_images(full=True)[image_index][0]
        base_image = doc.extract_image(xref)
        image_ext = base_image["ext"]
        image_bytes = base_image["image"]

        pil_img = Image.open(io.BytesIO(image_bytes))
        width, height = pil_img.size
        if width < 100 or height < 100:
            if page_rendered:
                return None, None, page_rendered
            print(f"Image too small ({width}x{height}); rendering full page instead...")
            fallback_path = figure_output_dir / f"{pdf_name}_page_{page_index+1}_rendered_fullpage.png"
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            pix.save(fallback_path)
            image_path = fallback_path
            # Reuse the pil_img variable: open the fallback image, but since we just rendered it, we can keep the same variable name
            pil_img = pil_img  
            pil_img = Image.open(fallback_path)
            page_rendered = True
        else:
            image_path = figure_output_dir / f"{pdf_name}_page_{page_index+1}_image_{image_index+1}_original.{image_ext}"
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

        # Detect text boxes with detail=1 to get bounding boxes
        detected = reader.readtext(str(image_path), detail=1)

        ocr_text_pieces = []
        for bbox, text, conf in detected:
            # bbox is a list of 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)

            crop_box = (int(left), int(top), int(right), int(bottom))
            cropped_img = pil_img.crop(crop_box)

            # if height > width, rotate 90 degrees to upright
            if cropped_img.height > cropped_img.width:
                cropped_img = cropped_img.rotate(-90, expand=True)

            cropped_np = np.array(cropped_img)
            ocr_result = reader.readtext(cropped_np, detail=0)
            ocr_text_pieces.append(" ".join(ocr_result))

        ocr_text = "\n".join(ocr_text_pieces)

        figure_caption = ""
        lines = page_text.splitlines()
        lower_lines = [line.lower() for line in lines]

        candidate_captions = []
        # Pattern to match "Figure X" 
        figure_pattern = re.compile(r"^figure\s*\d+[a-zA-Z]?\.*", re.IGNORECASE)
        # Pattern to detect start of new figure or section header
        section_start_pattern = re.compile(r"^(figure\s*\d+[a-zA-Z]?|table\s*\d+|references|bibliography|appendix|chapter|section)\b", re.IGNORECASE)

        i = 0
        while i < len(lines):
            if figure_pattern.match(lower_lines[i]):
                caption_lines = [lines[i].strip()]
                i += 1
                extra_lines_collected = 0
                while i < len(lines) and extra_lines_collected < 3:
                    line_strip = lines[i].strip()
                    if line_strip == "":
                        break
                    if section_start_pattern.match(line_strip.lower()):
                        break
                    caption_lines.append(line_strip)
                    i += 1
                    extra_lines_collected += 1
                # Normalize spacing by joining lines with a space
                caption_text = " ".join(caption_lines).replace("\n", " ").strip()
                candidate_captions.append(caption_text)
            else:
                i += 1

        # If no candidate captions found, try to find partial captions of up to 3 consecutive non-empty lines anywhere
        if not candidate_captions:
            i = 0
            while i < len(lines):
                line_strip = lines[i].strip()
                if line_strip:
                    caption_lines = [line_strip]
                    j = i + 1
                    while j < len(lines) and len(caption_lines) < 3:
                        next_line = lines[j].strip()
                        if next_line == "":
                            break
                        caption_lines.append(next_line)
                        j += 1
                    caption_text = " ".join(caption_lines).replace("\n", " ").strip()
                    candidate_captions.append(caption_text)
                    i = j
                else:
                    i += 1

        if candidate_captions:
            figure_caption = max(candidate_captions, key=len).strip()

        fallback_prefix = "Caption (from PDF text): "
        if not figure_caption or len(figure_caption) < 10:
            figure_caption = fallback_prefix + (figure_caption if figure_caption else "No caption provided.")
        elif not figure_caption.startswith(fallback_prefix):
            figure_caption = fallback_prefix + figure_caption

        refined_json_image = refine_with_claude_image(figure_caption, image_path)

        return {
            "figure_file": image_path.name,
            "caption": figure_caption,
            "ocr_text": ocr_text.strip(),
            "refined_json_image": refined_json_image
        }, f"[Figure extracted: {image_path.name}]", page_rendered
    except Exception as e:
        print(f"Failed to extract image on page {page_index+1}: {e}")
        return None, None, page_rendered

def process_one_pdf(pdf_path, flag="all", fig_dir="figure_ocr_output", output_dir="json_output", ocr_json_path=None):
    if flag not in {'all', 'figure'}:
        return []


    all_text = []
    doc = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).stem
    figure_output_dir = Path(fig_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    ocr_figure_data = []

    try:
        for page_index, page in enumerate(doc):
            page_text = page.get_text()
            # Remove everything after "references" only if page is among last 5 pages
            if page_index >= len(doc) - 5:
                ref_idx = page_text.lower().find("references")
                if ref_idx != -1:
                    page_text = page_text[:ref_idx]
            all_text.append(page_text)

            if flag in {'all', 'figure'}:
                images = page.get_images(full=True)
                page_rendered = False
                for image_index in range(len(images)):
                    ocr_data, figure_text, page_rendered = extract_and_ocr_image(
                        doc, page, page_index, image_index, pdf_name,
                        figure_output_dir, page_text, page_rendered
                    )
                    if ocr_data:
                        ocr_figure_data.append(ocr_data)
                        all_text.append(figure_text)
    finally:
        # Write OCR JSON either to explicit path (ocr_json_path) or default output_dir/<pdf_stem>_ocr.json
        if ocr_figure_data:
            if ocr_json_path:
                ocr_output_path = Path(ocr_json_path)
                ocr_output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                ocr_output_path = Path(output_dir) / f"{pdf_name}_ocr.json"
            with open(ocr_output_path, "w", encoding="utf-8") as f:
                json.dump(ocr_figure_data, f, indent=2, ensure_ascii=False)

    if ocr_figure_data:
        all_text.append(json.dumps({"ocr_figures": ocr_figure_data}))

    attach_tabular_supplement(Path(pdf_path), all_text)

    # write combined JSON output to the specified output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / f"{pdf_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_text, f, indent=2, ensure_ascii=False)

    return all_text

def process_pdf_batch(pdf_dir, output_dir="json_output", fig_dir="figure_ocr_output", flag="all"):
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_dir.glob("*.pdf"))

    def handle_pdf(pdf_file):
        print(f"Processing {pdf_file}...")
        try:
            chunks = process_one_pdf(pdf_file, flag=flag, fig_dir=fig_dir, output_dir=output_dir)
            final_json_path = output_dir / f"{pdf_file.stem}.json"
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)
        except Exception as e:
            print(f"Failed processing {pdf_file}: {e}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list([_ for _ in executor.map(handle_pdf, pdf_files)])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to a PDF file to process")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: specified pdf does not exist: {pdf_path}")
        raise SystemExit(2)

    # Process the single PDF
    process_one_pdf(str(pdf_path), flag="all")
