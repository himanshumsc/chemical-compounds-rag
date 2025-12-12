#!/usr/bin/env python3
import json
import re
from pathlib import Path

BASE = Path("/home/himanshu/dev/test")
QA_DIR = BASE / "data/processed/qa_pairs_individual_components"
RENDERS_DIR = BASE / "extracted_images/renders"


def normalize_name(name: str) -> str:
    if not name:
        return ""
    # Convert underscores to spaces
    s = name.replace("_", " ")
    # Remove punctuation and brackets commonly present in render names
    s = re.sub(r"[\(\)\[\]\{\},'\"]", "", s)
    # Replace hyphens with spaces to avoid token mismatch
    s = s.replace("-", " ")
    # Collapse multiple spaces and lowercase
    s = " ".join(s.split()).lower()
    return s


def build_render_index():
    index = {}
    for img_path in RENDERS_DIR.glob("*_chapter_start.png"):
        # Extract segment after first two underscores (page_XXXX_) up to suffix
        stem = img_path.stem  # e.g., page_0150_Benzene_chapter_start
        try:
            # Drop leading 'page_####_' and trailing '_chapter_start'
            compound_part = stem.split("_", 2)[2].rsplit("_chapter_start", 1)[0]
        except Exception:
            continue
        norm = normalize_name(compound_part)
        index[norm] = img_path
    return index


def update_image_paths(render_index):
    updated = 0
    for qa_file in sorted(QA_DIR.glob("qa_*.json")):
        data = json.loads(qa_file.read_text(encoding="utf-8"))
        raw_img = data.get("image_path", "")
        raw_name = data.get("compound_name", raw_img)
        norm_from_json = normalize_name(raw_img if raw_img else raw_name)

        # Direct match
        target = render_index.get(norm_from_json)
        if not target:
            # Try with compound_name if image_path failed
            target = render_index.get(normalize_name(raw_name))

        if target:
            # Use absolute path for robustness
            data["image_path"] = str(target)
            qa_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            updated += 1
    return updated


def rename_files_incrementally():
    files = sorted(QA_DIR.glob("qa_*.json"), key=lambda p: p.name.lower())
    for i, path in enumerate(files, start=1):
        rest = path.name[len("qa_"):]
        new_name = f"{i}_{rest}"
        path.rename(path.with_name(new_name))
    return len(files)


def main():
    render_index = build_render_index()
    count = update_image_paths(render_index)
    print(f"Updated image_path for {count} QA files")
    total = rename_files_incrementally()
    print(f"Renamed {total} QA files with incremental prefixes")


if __name__ == "__main__":
    main()


