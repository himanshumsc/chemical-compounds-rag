#!/usr/bin/env python3
"""
Extract all images from the source PDF and save them under extracted_images/.
Also records page index and a best-effort classification (first-page skeletal vs following-page ball-and-stick)
based on simple heuristics.
"""

import fitz  # PyMuPDF
from pathlib import Path
import hashlib
import io
from PIL import Image, ImageChops
import json

PDF_PATH = Path("/home/himanshu/dev/test/data/raw/chemical-compounds.pdf")
OUT_DIR = Path("/home/himanshu/dev/test/extracted_images")


def md5_bytes(data: bytes) -> str:
    h = hashlib.md5()
    h.update(data)
    return h.hexdigest()


def is_blank_white(img: Image.Image, threshold: float = 0.95) -> bool:
    # Convert to L (grayscale) and check mean brightness
    gray = img.convert("L")
    pixels = gray.getdata()
    # Count near-white pixels
    white_count = sum(1 for p in pixels if p > 240)
    ratio = white_count / len(pixels)
    return ratio >= threshold


def classify_image(page_index: int, img_idx_on_page: int) -> str:
    # Heuristic: first image on the first content page may be skeletal; next page's first might be ball-and-stick.
    # Without semantic detection, return labels by position to help manual inspection later.
    if img_idx_on_page == 0:
        return "page_first_image"
    return "page_additional_image"


def autocrop_nonwhite(pil_img: Image.Image, bg=255) -> Image.Image:
    # Convert to L and create a mask of non-white pixels
    gray = pil_img.convert("L")
    bg_img = Image.new("L", gray.size, color=bg)
    diff = ImageChops.difference(gray, bg_img)
    bbox = diff.getbbox()
    if bbox:
        return pil_img.crop(bbox)
    return pil_img


def extract_images():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = OUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    index = []
    with fitz.open(PDF_PATH) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            if not images:
                continue

            img_counter = 0
            for img in images:
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n >= 5:  # CMYK or similar
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_bytes = pix.tobytes("png")
                    digest = md5_bytes(img_bytes)

                    pil_img = Image.open(io.BytesIO(img_bytes))
                    if is_blank_white(pil_img):
                        # Skip blank/white assets
                        continue

                    label = classify_image(page_num, img_counter)
                    filename = f"page_{page_num:04d}_img_{img_counter:02d}_{label}_{digest[:8]}.png"
                    out_path = images_dir / filename
                    with open(out_path, "wb") as f:
                        f.write(img_bytes)

                    index.append({
                        "page_index": page_num,
                        "image_index_on_page": img_counter,
                        "label": label,
                        "file": str(out_path),
                        "md5": digest,
                        "width": pil_img.width,
                        "height": pil_img.height,
                    })
                    img_counter += 1
                except Exception as e:
                    # Continue extracting other images even if one fails
                    continue

    # Write an index JSON for reference
    with open(OUT_DIR / "images_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Extracted {len(index)} images to {images_dir}")
    print(f"Index written to {OUT_DIR / 'images_index.json'}")


def render_pages_highdpi():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    renders_dir = OUT_DIR / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # Heuristic: chapter starts are the Arabic start pages for each compound from our processed JSONs
    comp_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    if not comp_dir.exists():
        print("Compound directory not found; cannot map chapter starts.")
        return

    with fitz.open(PDF_PATH) as doc:
        render_index = []
        for json_file in sorted(comp_dir.glob("*.json")):
            try:
                data = json.loads(Path(json_file).read_text(encoding="utf-8"))
                name = data.get("name", "Unknown").replace(".", "").strip()
                pdf_start = data.get("pdf_start_page")
                if pdf_start is None:
                    continue
                # pdf_start_page is zero- or one-based? From earlier scripts it's zero-based for fitz usage;
                # but we stored actual pdf page index; use as-is, with bounds check.
                for offset, label in [(0, "chapter_start"), (1, "following_page")]:
                    page_idx = pdf_start + offset
                    if page_idx < 0 or page_idx >= len(doc):
                        continue
                    page = doc[page_idx]
                    # Use ~288 dpi equivalent for batch (zoom=4.0)
                    zoom = 4.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_bytes = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    cropped = autocrop_nonwhite(pil_img)
                    # Save
                    safe_name = name.replace(" ", "_")
                    out_name = f"page_{page_idx:04d}_{safe_name}_{label}.png"
                    out_path = renders_dir / out_name
                    cropped.save(out_path, format="PNG", optimize=True)
                    render_index.append({
                        "compound_name": name,
                        "page_index": page_idx,
                        "label": label,
                        "file": str(out_path),
                        "width": cropped.width,
                        "height": cropped.height,
                        "source": "pdf_render_zoom2"
                    })
            except Exception:
                continue

    with open(OUT_DIR / "renders_index.json", "w", encoding="utf-8") as f:
        json.dump(render_index, f, indent=2)

    print(f"Rendered {len(render_index)} pages to {renders_dir}")
    print(f"Index written to {OUT_DIR / 'renders_index.json'}")


def render_benzene_ultra():
    # Find Benzene file and render its start and next page at ultra-high DPI
    comp_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    out_dir = OUT_DIR / "renders_ultra"
    out_dir.mkdir(parents=True, exist_ok=True)

    benzene_file = None
    for jf in comp_dir.glob("*.json"):
        try:
            data = json.loads(Path(jf).read_text(encoding="utf-8"))
            name = data.get("name", "").lower()
            if "benzene" in name:
                benzene_file = jf
                break
        except Exception:
            continue

    if not benzene_file:
        print("Benzene JSON not found; cannot render ultra DPI.")
        return

    data = json.loads(Path(benzene_file).read_text(encoding="utf-8"))
    pdf_start = data.get("pdf_start_page")
    name = data.get("name", "Benzene").replace(".", "").strip()
    safe_name = name.replace(" ", "_")

    with fitz.open(PDF_PATH) as doc:
        for offset, label in [(0, "chapter_start_ultra"), (1, "following_page_ultra")]:
            page_idx = pdf_start + offset
            if page_idx < 0 or page_idx >= len(doc):
                continue
            page = doc[page_idx]
            # Ultra-high DPI
            zoom = 6.0  # ~432 dpi
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            cropped = autocrop_nonwhite(pil_img)
            out_path = out_dir / f"page_{page_idx:04d}_{safe_name}_{label}.png"
            cropped.save(out_path, format="PNG", optimize=True)
            print(f"Saved ultra render: {out_path}")


if __name__ == "__main__":
    extract_images()
    render_pages_highdpi()
    render_benzene_ultra()


