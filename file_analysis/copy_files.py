#!/usr/bin/env python3
"""Copy data files to this directory"""
import shutil
from pathlib import Path

source_dir = Path("/home/himanshu")
dest_dir = Path(__file__).parent

files_to_copy = [
    "file_modified_data.json",
    "file_modified_report.md"
]

for filename in files_to_copy:
    source = source_dir / filename
    dest = dest_dir / filename
    
    if source.exists():
        print(f"Copying {filename}...")
        shutil.copy2(source, dest)
        print(f"✅ Copied {filename}")
    else:
        print(f"⚠️  {filename} not found at {source}")

print("\n✅ All files copied!")

