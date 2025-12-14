#!/usr/bin/env python3
"""
Fix ChromaDB SQLite compatibility issue
"""
import sys
import os

# Add the patch to sys.modules before importing chromadb
sys.path.insert(0, '/home/himanshu/dev/code/.venv_phi4_req/lib64/python3.11/site-packages')
import pysqlite3 as sqlite3
sys.modules['sqlite3'] = sqlite3

# Now import chromadb
import chromadb
print("ChromaDB SQLite patch applied successfully!")
