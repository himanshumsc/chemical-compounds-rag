#!/usr/bin/env python3
"""
One-time fixer: clean dot leaders from compound_name and image_path in
qa_pairs_individual_components/*.json and rename files to clean names.
"""
import json
import re
from pathlib import Path

SRC = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components")

def clean_name(name: str) -> str:
    name = re.sub(r"\.", "", name)
    return " ".join(name.split()).strip()

def sanitize_filename(name: str) -> str:
    return name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

def main():
    for path in sorted(SRC.glob("qa_*.json")):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            old_compound_name = data.get("compound_name", "")
            cleaned = clean_name(old_compound_name)
            if not cleaned:
                continue
            changed = False
            if cleaned != old_compound_name:
                data["compound_name"] = cleaned
                changed = True
            # image_path is just the name
            old_image_path = data.get("image_path", "")
            if old_image_path:
                cleaned_image = clean_name(old_image_path)
                if cleaned_image != old_image_path:
                    data["image_path"] = cleaned_image
                    changed = True
            if changed:
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
            # Rename file if necessary
            target = path.parent / f"qa_{sanitize_filename(cleaned)}.json"
            if target.name != path.name:
                path.rename(target)
        except Exception:
            continue

if __name__ == "__main__":
    main()



