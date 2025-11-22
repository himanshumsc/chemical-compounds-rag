#!/usr/bin/env python3
from pathlib import Path
import json

QA_DIR = Path('/home/himanshu/dev/test/data/processed/qa_pairs_individual_components')

def main():
    missing = []
    total = 0
    for f in sorted(QA_DIR.glob('*_*.json')):
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            missing.append((f.name, f"<invalid json: {e}>"))
            continue
        p = d.get('image_path', '')
        total += 1
        if not p:
            missing.append((f.name, '<empty>'))
            continue
        if not Path(p).exists():
            missing.append((f.name, p))

    print(f'TOTAL_JSON={total}')
    print(f'MISSING_OR_NONEXISTENT={len(missing)}')
    for name, path in missing[:50]:
        print(f'MISS {name} -> {path}')

if __name__ == '__main__':
    main()


