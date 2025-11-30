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
3. **Attempts** to extract native text from each page using PyMuPDF's `get_text()` method
   - **Note**: Many pages (108 out of 981) have empty text because the PDF contains scanned images or image-based content that PyMuPDF cannot extract text from
   - Pages with extractable text will have text content; others will have empty strings
4. Extracts image metadata (xref, bounding boxes) from each page
5. Converts each page to PNG image at 200 DPI using pdf2image
   - This image conversion is critical for Step 2, where Phi-4 OCR will extract text from these images
6. Saves page images to `pdf_extracted_images/` directory
7. Creates JSONL records with:
   - `page_range`: Page number range
   - `text`: Extracted text content (may be empty for image-based pages)
   - `images`: List of image metadata
   - `tables`: Table information (basic detection)
   - `page_size`: Page dimensions
   - `image_path`: Path to the rendered PNG image
8. Writes all records to `chemical-compounds_chunks.jsonl`
   - **Important**: At this stage, many chunks will have empty text fields, which will be filled in Step 2 using OCR

**Libraries Used**:
- `fitz` (PyMuPDF) - PDF text and metadata extraction
- `pdf2image` - PDF page to image conversion (uses poppler)
- `PIL` (Pillow) - Image handling (via pdf2image)
- `pathlib.Path` - File path handling
- `json` - JSON serialization
- `argparse` - Command-line argument parsing

**Key Functions**:
- `extract_page_content()` - Attempts to extract text and metadata from PDF page using PyMuPDF
- `process_pdf()` - Main processing function

**Important Note**: 
- This step produces the initial text extraction, but many pages (image-based/scanned) will have empty text
- The actual text extraction for those pages happens in **Step 2 using Phi-4 OCR**
- The page images generated here are essential for OCR processing in the next step

---

### **Step 2: OCR Enrichment**
**Script**: `ocr_enrich_phi4_multithreaded.py`  
**Input**: 
- `/home/himanshu/dev/data/processed/chemical-compounds_chunks.jsonl`
- `/home/himanshu/dev/data/processed/pdf_extracted_images/`

**Output**: `/home/himanshu/dev/data/processed/chemical-compounds_chunks_enriched.jsonl`

**Process**:
1. Reads JSONL chunks and identifies **sparse chunks** (pages with empty text or text < min_length threshold)
   - **Critical**: This step identifies the 108+ pages that had empty text in Step 1
   - These pages require OCR extraction from their rendered images
2. Loads Phi-4 multimodal model using transformers library
3. Processes chunks in batches using multithreaded workers
4. For each sparse chunk (empty or short text):
   - Loads corresponding page image from `pdf_extracted_images/`
   - Resizes image if needed (max 512px) for batch processing
   - **Runs Phi-4 multimodal OCR** to extract text from the image
     - This is where the actual text extraction happens for image-based PDF pages
     - Phi-4 reads the rendered page image and extracts all visible text
   - Combines original text (if any) with OCR-extracted text
     - Format: `"{original_text}\n\nOCR Extracted: {ocr_text}"` (if original exists)
     - Format: `"{ocr_text}"` (if no original text)
   - Generates structured `extracted_info` metadata including:
     - Summary of content
     - Detected content types (chemical compounds, formulas, tables, diagrams)
     - Text length
     - Processing method (`phi4_multimodal_ocr` for OCR-extracted, `pymupdf_native` for native)
   - Transforms table format if detected
   - Marks chunk as `enriched_via_ocr: true` (for OCR-extracted pages)
5. Writes enriched chunks back to JSONL file
   - **Result**: All chunks now have text content (either from Step 1 native extraction or Step 2 OCR extraction)

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
- `run_phi4_ocr_batch()` - Processes multiple images in batch using Phi-4 OCR to extract text
- `generate_extracted_info()` - Creates structured metadata for enriched chunks
- `transform_tables_format()` - Transforms table data format
- `enrich_chunks()` - Main enrichment orchestration function
- `should_enrich_text()` - Determines if a chunk needs OCR enrichment (empty or sparse text)

**Model**: Microsoft Phi-4 Multimodal (stored in `/home/himanshu/dev/models/PHI4_ONNX/` or similar)

**Critical Note**: 
- This step is where **actual text extraction occurs** for pages that had empty text in Step 1
- Phi-4 multimodal OCR reads the page images and extracts all visible text
- Approximately 108+ pages are enriched via OCR in this step
- Pages that already had text from Step 1 are preserved but may also be enriched if text was sparse

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

3. **Chunk Processing** (This is where OCR-extracted text is used!):
   - Reads enriched JSONL file line by line
   - For each chunk:
     - **Extracts text content** from `chunk.get('text', '')`
       - **Critical**: This `text` field contains:
         - OCR-extracted text for pages that were enriched via Phi-4 (108+ pages)
         - Native PyMuPDF text for pages that had extractable text
         - Combined text (`original\n\nOCR Extracted: ocr_text`) if both existed
       - **Yes, OCR text is used here!** The OCR-extracted text from Step 2 is now the source for embeddings
     - **Generates text embedding** using CLIP's text encoder
       - Takes the text (including OCR-extracted text) and processes it
       - Truncates text to CLIP's context length (77 tokens) if needed
       - Tokenizes text using CLIP tokenizer
       - Encodes text through CLIP's text encoder neural network
       - Output: 512-dimensional vector representing the semantic meaning
       - Normalizes embedding vector (L2 normalization)
     - **Loads page image** if available (from `image_path` field)
     - **Generates image embedding** using CLIP's image encoder
       - Opens and preprocesses image (resize, normalize, tensor conversion)
       - Encodes image through CLIP's vision transformer (ViT)
       - Output: 512-dimensional vector representing visual content
       - Normalizes embedding vector (L2 normalization)
     - **Uses text embedding as primary embedding** (ChromaDB requirement)
       - ChromaDB stores one primary embedding per document
       - Text embedding is used for similarity search
       - Image embedding is stored in metadata for hybrid search capabilities
     - **Stores image embedding in metadata** if available
       - Allows hybrid search (combining text + image similarity)
     - Creates metadata object with:
       - `chunk_id`: Unique identifier (`chunk_0`, `chunk_1`, ...)
       - `page_range`: Page numbers
       - `text_content`: Full text content (the OCR-extracted text!)
       - `image_path`: Path to page image
       - `has_image`: Boolean flag
       - `text_length`: Text character count
       - `image_embedding`: Image embedding vector (stored in metadata as a list)

4. **Batch Storage to ChromaDB**:
   - Adds embeddings to ChromaDB in batches of 100 chunks
   - Each batch includes:
     - `ids`: Unique chunk identifiers (`["chunk_0", "chunk_1", ...]`)
     - `documents`: Full text content (including OCR-extracted text)
       - **This is what users will search through!**
       - Contains the OCR-extracted text from Step 2
     - `metadatas`: Metadata dictionaries (includes image embedding if available)
     - `embeddings`: Text embedding vectors (512-dim CLIP embeddings)
       - **These embeddings encode the semantic meaning of the OCR text**
   - ChromaDB stores this data persistently using SQLite backend
   - Vector similarity search is enabled via the embeddings

5. **Collection Statistics**:
   - Prints final collection count
   - Collection ready for search operations
   - **All OCR-extracted text is now searchable via semantic similarity!**

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

**How OCR Text Flows to ChromaDB**:
1. **Step 2 Output**: Enriched JSONL contains `text` field with OCR-extracted text
2. **Step 3 Input**: Reads `text` field from enriched JSONL (`chunk.get('text', '')`)
3. **Embedding Generation**: CLIP text encoder processes this text (including OCR text)
4. **Storage**: Text embedding (representing OCR text semantics) is stored in ChromaDB
5. **Search**: Users can search using natural language, and ChromaDB finds similar embeddings
   - The OCR-extracted text enables semantic search even for originally image-based pages!

---

## 🔄 **Complete Text Flow: OCR → Embeddings → ChromaDB**

Understanding how OCR-extracted text becomes searchable:

### **Step-by-Step Flow**:

1. **Step 1 (PDF Preprocessing)**:
   - Many pages have empty text: `"text": ""`
   - Page images are saved for OCR processing

2. **Step 2 (OCR Enrichment)**:
   - Phi-4 OCR extracts text from images
   - Enriched JSONL now has: `"text": "chemical compounds\nH3C\nOCC..."`
   - This text field contains the OCR-extracted content

3. **Step 3 (Embedding Generation)**:
   - **Reads**: `text_content = chunk.get('text', '')`
     - This reads the OCR-extracted text from Step 2!
   - **Processes**: `text_embedding = self._get_text_embedding(text_content)`
     - CLIP tokenizes the OCR text
     - CLIP encodes it through its neural network
     - Produces 512-dim vector representing semantic meaning
   - **Stores**: 
     - `documents`: The OCR text itself (for retrieval)
     - `embeddings`: The semantic vector (for similarity search)
     - `metadatas`: Additional info including image embedding

4. **ChromaDB Storage**:
   - Each document has:
     - **Document**: OCR-extracted text (what you read)
     - **Embedding**: Semantic vector of that text (what you search)
     - **Metadata**: Page info, image embedding, etc.
   
5. **Search Time**:
   - User query → CLIP embedding → Similarity search in ChromaDB
   - Returns documents with similar embeddings (semantically similar content)
   - The OCR-extracted text is what gets returned to the user!

**Key Insight**: The OCR-extracted text from Step 2 is not just stored—it's converted into semantic embeddings that enable intelligent search. Without OCR extraction, those 108+ pages would have no searchable content!

### **Visual Flow Diagram**:

```
┌─────────────────────────────────────────────────────────┐
│ Step 2: OCR Enrichment                                  │
│ Phi-4 OCR extracts: "chemical compounds\nH3C\nOCC..."│
└──────────────────┬──────────────────────────────────────┘
                    │
                    │ text field populated in JSONL
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Embedding Generation                            │
│                                                          │
│ chunk.get('text') → "chemical compounds\nH3C\nOCC..."   │
│                    │                                     │
│                    ▼                                     │
│         CLIP Tokenizer                                   │
│                    │                                     │
│                    ▼                                     │
│      CLIP Text Encoder (Neural Network)                  │
│                    │                                     │
│                    ▼                                     │
│     512-dim Vector [0.123, -0.456, 0.789, ...]          │
│                    │                                     │
│                    ▼                                     │
│         Normalized Embedding                             │
└──────────────────┬──────────────────────────────────────┘
                    │
                    │ stored in ChromaDB
                    ▼
┌─────────────────────────────────────────────────────────┐
│ ChromaDB Document Structure:                            │
│                                                          │
│ {                                                       │
│   "id": "chunk_0",                                      │
│   "document": "chemical compounds\nH3C\nOCC...",       │
│   "embedding": [0.123, -0.456, 0.789, ...],           │
│   "metadata": {                                         │
│     "text_content": "chemical compounds\nH3C\nOCC...",  │
│     "image_embedding": [...],                            │
│     "page_range": "1-1"                                  │
│   }                                                      │
│ }                                                       │
└─────────────────────────────────────────────────────────┘
```

### **What Happens During Search**:

1. **User Query**: "What is benzene?"
2. **Query Embedding**: CLIP encodes query → 512-dim vector
3. **Similarity Search**: ChromaDB finds documents with similar embeddings
4. **Results**: Returns documents containing OCR-extracted text about benzene
5. **Response**: User sees the OCR-extracted text content from those pages

**The Magic**: Even though 108+ pages originally had no text, Phi-4 OCR extracted it, and CLIP converted it into searchable semantic vectors!

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
│  Note: 108+ pages have empty text       │
│  (image-based/scanned pages)            │
└──────────────┬──────────────────────────┘
               │
               │ [ocr_enrich_phi4_multithreaded.py]
               │ Libraries: transformers, torch, PIL
               │ Model: Phi-4 Multimodal OCR
               │ Extracts text from images for
               │ pages with empty/sparse text
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

1. **Two-Stage Text Extraction**:
   - **Step 1**: Attempts native text extraction via PyMuPDF (works for text-based PDFs, fails for image-based/scanned pages)
   - **Step 2**: Uses Phi-4 multimodal OCR to extract text from rendered images (covers the 108+ pages with empty text from Step 1)
   - Final result: All chunks have text content, either from native extraction or OCR extraction

2. **Chunking Strategy**: Pages are treated as individual chunks (one chunk per page). No additional text splitting or semantic chunking is performed.

3. **Embedding Strategy**: 
   - Primary embedding: CLIP text embedding (used for similarity search)
   - Secondary embedding: CLIP image embedding (stored in metadata for hybrid search)

4. **Model Requirements**:
   - Phi-4 Multimodal model must be downloaded separately
   - CLIP model (ViT-B/32) is downloaded automatically via `clip.load()`

5. **Threading**: OCR enrichment uses multithreaded batch processing for efficiency.

6. **SQLite Compatibility**: ChromaDB requires pysqlite3 for SQLite compatibility, which is patched into sys.modules before importing chromadb.

7. **Text Extraction Accuracy**: 
   - Native PyMuPDF extraction: Works well for text-based PDFs, preserves formatting
   - Phi-4 OCR extraction: Extracts text from images/scanned pages, handles chemical formulas and structured content

---

**Last Updated**: 2025-01-07  
**Total Processing Steps**: 3 major steps  
**Final Output**: ChromaDB collection with multimodal embeddings

