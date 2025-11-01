# Chemical Compounds PDF to ChromaDB Processing Pipeline

This document outlines the complete processing pipeline from `chemical-compounds.pdf` to ChromaDB storage, including all steps and libraries used.

## 📋 Pipeline Overview

```
chemical-compounds.pdf 
    ↓ [Step 1: PDF Preprocessing]
Page Images + Initial JSONL Chunks
    ↓ [Step 2: OCR Enrichment]
Enriched JSONL with OCR Text
    ↓ [Step 3: Embedding Generation & ChromaDB Storage]
ChromaDB Database with Multimodal Embeddings
```

---

## 🔄 Step-by-Step Processing Pipeline

### **Step 1: PDF Preprocessing**
**Script**: `pdf_preprocess.py`  
**Input**: `/home/himanshu/dev/data/raw/chemical-compounds.pdf`  
**Output**: 
- `/home/himanshu/dev/data/processed/pdf_extracted_images/` (page images)
- `/home/himanshu/dev/data/processed/chemical-compounds_chunks.jsonl` (page-level chunks)

**Process**:
1. Opens PDF document using PyMuPDF (fitz)
2. Iterates through each page
3. Extracts native text from each page using PyMuPDF's `get_text()` method
4. Extracts image metadata (xref, bounding boxes) from each page
5. Converts each page to PNG image at 200 DPI using pdf2image
6. Saves page images to `pdf_extracted_images/` directory
7. Creates JSONL records with:
   - `page_range`: Page number range
   - `text`: Extracted text content
   - `images`: List of image metadata
   - `tables`: Table information (basic detection)
   - `page_size`: Page dimensions
   - `image_path`: Path to the rendered PNG image
8. Writes all records to `chemical-compounds_chunks.jsonl`

**Libraries Used**:
- `fitz` (PyMuPDF) - PDF text and metadata extraction
- `pdf2image` - PDF page to image conversion (uses poppler)
- `PIL` (Pillow) - Image handling (via pdf2image)
- `pathlib.Path` - File path handling
- `json` - JSON serialization
- `argparse` - Command-line argument parsing

**Key Functions**:
- `extract_page_content()` - Extracts text and metadata from PDF page
- `process_pdf()` - Main processing function

---

### **Step 2: OCR Enrichment**
**Script**: `ocr_enrich_phi4_multithreaded.py`  
**Input**: 
- `/home/himanshu/dev/data/processed/chemical-compounds_chunks.jsonl`
- `/home/himanshu/dev/data/processed/pdf_extracted_images/`

**Output**: `/home/himanshu/dev/data/processed/chemical-compounds_chunks_enriched.jsonl`

**Process**:
1. Reads JSONL chunks and identifies sparse chunks (text < min_length threshold)
2. Loads Phi-4 multimodal model using transformers library
3. Processes chunks in batches using multithreaded workers
4. For each sparse chunk:
   - Loads corresponding page image
   - Resizes image if needed (max 512px) for batch processing
   - Runs Phi-4 OCR to extract text from image
   - Combines original text (if any) with OCR-extracted text
   - Generates structured `extracted_info` metadata including:
     - Summary of content
     - Detected content types (chemical compounds, formulas, tables, diagrams)
     - Text length
     - Processing method (phi4_multimodal_ocr)
   - Transforms table format if detected
   - Marks chunk as `enriched_via_ocr: true`
5. Writes enriched chunks back to JSONL file

**Libraries Used**:
- `torch` (PyTorch) - Deep learning framework
- `transformers` - Hugging Face transformers library
  - `AutoConfig` - Model configuration
  - `AutoModelForCausalLM` - Phi-4 model loading
  - `AutoProcessor` - Multimodal processor
  - `AutoTokenizer` - Text tokenizer
- `PIL.Image` - Image processing
- `threading` - Multithreading support
- `concurrent.futures.ThreadPoolExecutor` - Parallel batch processing
- `queue` - Thread-safe queue for batch processing
- `dataclasses.dataclass` - Configuration data class
- `json` - JSON serialization
- `pathlib.Path` - File path handling
- `argparse` - Command-line argument parsing
- `time` - Timing measurements

**Key Functions**:
- `load_phi4_model_and_processor()` - Loads Phi-4 model with attention implementation selection
- `run_phi4_ocr_batch()` - Processes multiple images in batch using Phi-4
- `generate_extracted_info()` - Creates structured metadata for enriched chunks
- `transform_tables_format()` - Transforms table data format
- `enrich_chunks()` - Main enrichment orchestration function

**Model**: Microsoft Phi-4 Multimodal (stored in `/home/himanshu/dev/models/PHI4_ONNX/` or similar)

---

### **Step 3: Embedding Generation & ChromaDB Storage**
**Script**: `setup_chromadb_embeddings.py`  
**Input**: `/home/himanshu/dev/data/processed/chemical-compounds_chunks_enriched.jsonl`  
**Output**: ChromaDB collection at `/home/himanshu/dev/data/chromadb/`

**Process**:
1. **ChromaDB Setup**:
   - Creates ChromaDB persistent client
   - Creates or gets collection named `chemical_compounds_multimodal`
   - Configures SQLite compatibility patch (pysqlite3)

2. **CLIP Model Loading**:
   - Loads CLIP model (ViT-B/32) for multimodal embeddings
   - Loads CLIP preprocessing pipeline

3. **Chunk Processing**:
   - Reads enriched JSONL file line by line
   - For each chunk:
     - Extracts text content
     - Generates text embedding using CLIP's text encoder
       - Truncates text to CLIP's context length (77 tokens)
       - Tokenizes and encodes text
       - Normalizes embedding vector
     - Loads page image if available
     - Generates image embedding using CLIP's image encoder
       - Opens and preprocesses image
       - Encodes image to embedding
       - Normalizes embedding vector
     - Uses text embedding as primary embedding (ChromaDB requirement)
     - Stores image embedding in metadata if available
     - Creates metadata object with:
       - `chunk_id`: Unique identifier
       - `page_range`: Page numbers
       - `text_content`: Full text content
       - `image_path`: Path to page image
       - `has_image`: Boolean flag
       - `text_length`: Text character count
       - `image_embedding`: Image embedding vector (in metadata)

4. **Batch Storage**:
   - Adds embeddings to ChromaDB in batches of 100
   - Each batch includes:
     - `ids`: Unique chunk identifiers
     - `documents`: Text content
     - `metadatas`: Metadata dictionaries
     - `embeddings`: Text embedding vectors

5. **Collection Statistics**:
   - Prints final collection count
   - Collection ready for search operations

**Libraries Used**:
- `chromadb` - Vector database for storing embeddings
  - `PersistentClient` - Persistent ChromaDB client
  - `Settings` - ChromaDB configuration
- `clip` - OpenAI CLIP model for multimodal embeddings
- `torch` (PyTorch) - Deep learning framework for CLIP
- `numpy` - Numerical operations for embedding arrays
- `PIL.Image` - Image loading and preprocessing
- `pysqlite3` - SQLite compatibility (patched into sys.modules)
- `json` - JSON parsing
- `pathlib.Path` - File path handling
- `argparse` - Command-line argument parsing
- `logging` - Logging framework
- `tqdm` - Progress bars
- `typing` - Type hints (List, Dict, Optional, Any)

**Key Functions**:
- `_setup_chromadb()` - Initializes ChromaDB client and collection
- `_load_clip_model()` - Loads CLIP model and preprocessor
- `_get_text_embedding()` - Generates CLIP text embedding
- `_get_image_embedding()` - Generates CLIP image embedding
- `process_chunks()` - Main processing function

**ChromaDB Collection Details**:
- **Collection Name**: `chemical_compounds_multimodal`
- **Embedding Dimension**: 512 (CLIP ViT-B/32)
- **Storage Type**: Persistent (SQLite backend)
- **Batch Size**: 100 chunks per batch

---

## 📦 Complete Library List

### Core Libraries
1. **PyMuPDF (fitz)** - PDF text and metadata extraction
2. **pdf2image** - PDF to image conversion (requires poppler)
3. **Pillow (PIL)** - Image processing and manipulation
4. **PyTorch (torch)** - Deep learning framework
5. **transformers** - Hugging Face library for Phi-4 model
6. **clip** - OpenAI CLIP for multimodal embeddings
7. **chromadb** - Vector database for embeddings storage
8. **numpy** - Numerical operations
9. **pysqlite3-binary** - SQLite compatibility for ChromaDB

### Standard Library Modules
- `argparse` - Command-line argument parsing
- `json` - JSON serialization/deserialization
- `pathlib` - Path handling
- `typing` - Type hints
- `logging` - Logging framework
- `threading` - Multithreading
- `concurrent.futures` - Parallel execution
- `queue` - Thread-safe queues
- `dataclasses` - Data classes
- `time` - Timing functions
- `sys` - System-specific parameters
- `os` - OS interface
- `tempfile` - Temporary file handling

### Utility Libraries
- `tqdm` - Progress bars

---

## 🔗 Data Flow Summary

```
┌─────────────────────────────────────────┐
│  chemical-compounds.pdf (Raw Data)      │
│  ~50MB PDF file                         │
└──────────────┬──────────────────────────┘
               │
               │ [pdf_preprocess.py]
               │ Libraries: PyMuPDF, pdf2image
               ▼
┌─────────────────────────────────────────┐
│  chemical-compounds_chunks.jsonl         │
│  + pdf_extracted_images/ (page PNGs)    │
│  981 page-level chunks                  │
└──────────────┬──────────────────────────┘
               │
               │ [ocr_enrich_phi4_multithreaded.py]
               │ Libraries: transformers, torch, PIL
               │ Model: Phi-4 Multimodal
               ▼
┌─────────────────────────────────────────┐
│  chemical-compounds_chunks_enriched.jsonl│
│  981 enriched chunks with OCR text       │
│  + extracted_info metadata              │
└──────────────┬──────────────────────────┘
               │
               │ [setup_chromadb_embeddings.py]
               │ Libraries: chromadb, clip, torch
               │ Model: CLIP ViT-B/32
               ▼
┌─────────────────────────────────────────┐
│  ChromaDB Collection                    │
│  Collection: chemical_compounds_        │
│              multimodal                 │
│  ~981 documents with 512-dim embeddings │
│  Persistent storage (SQLite)            │
└─────────────────────────────────────────┘
```

---

## 📊 Intermediate File Formats

### `chemical-compounds_chunks.jsonl` Structure:
```json
{
  "page_range": "1-1",
  "text": "extracted text content",
  "images": [{"index": 0, "xref": 123, ...}],
  "tables": [],
  "page_size": {"width": 612, "height": 792},
  "image_path": "/path/to/page_1.png"
}
```

### `chemical-compounds_chunks_enriched.jsonl` Structure:
```json
{
  "page_range": "1-1",
  "text": "original text\n\nOCR Extracted: ocr text",
  "images": [...],
  "tables": [...],
  "extracted_info": "{\"summary\": \"...\", \"content_types\": [...], ...}",
  "enriched_via_ocr": true
}
```

### ChromaDB Document Structure:
- **ID**: `chunk_0`, `chunk_1`, ...
- **Document**: Full text content
- **Metadata**: `{chunk_id, page_range, text_content, image_path, has_image, text_length, image_embedding}`
- **Embedding**: 512-dimensional CLIP text embedding vector

---

## 🚀 Usage Commands

### Step 1: PDF Preprocessing
```bash
python pdf_preprocess.py \
  --input /home/himanshu/dev/data/raw/chemical-compounds.pdf \
  --outdir /home/himanshu/dev/data/processed
```

### Step 2: OCR Enrichment
```bash
python ocr_enrich_phi4_multithreaded.py \
  --chunks-in /home/himanshu/dev/data/processed/chemical-compounds_chunks.jsonl \
  --images-dir /home/himanshu/dev/data/processed/pdf_extracted_images \
  --output-path /home/himanshu/dev/data/processed/chemical-compounds_chunks_enriched.jsonl \
  --limit 108 \
  --batch-size 4 \
  --num-workers 6
```

### Step 3: ChromaDB Setup
```bash
python setup_chromadb_embeddings.py \
  --jsonl /home/himanshu/dev/data/processed/chemical-compounds_chunks_enriched.jsonl \
  --chromadb-path /home/himanshu/dev/data/chromadb \
  --device cuda
```

---

## 📝 Notes

1. **Chunking Strategy**: Pages are treated as individual chunks (one chunk per page). No additional text splitting or semantic chunking is performed.

2. **Embedding Strategy**: 
   - Primary embedding: CLIP text embedding (used for similarity search)
   - Secondary embedding: CLIP image embedding (stored in metadata for hybrid search)

3. **Model Requirements**:
   - Phi-4 Multimodal model must be downloaded separately
   - CLIP model (ViT-B/32) is downloaded automatically via `clip.load()`

4. **Threading**: OCR enrichment uses multithreaded batch processing for efficiency.

5. **SQLite Compatibility**: ChromaDB requires pysqlite3 for SQLite compatibility, which is patched into sys.modules before importing chromadb.

---

**Last Updated**: 2025-01-07  
**Total Processing Steps**: 3 major steps  
**Final Output**: ChromaDB collection with multimodal embeddings

