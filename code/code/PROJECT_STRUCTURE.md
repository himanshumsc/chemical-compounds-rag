# Chemical Compounds RAG System - Clean Project Structure

## 📁 **Project Overview**
A complete multimodal RAG (Retrieval-Augmented Generation) system for chemical compounds database with persistent storage and Phi-4 generation.

## 🗂️ **Directory Structure**

```
/home/himanshu/dev/
├── code/                           # Source code (ALL BUILDING BLOCKS)
│   ├── download_phi4_model.py     # Download Phi-4 model (reproducibility)
│   ├── pdf_preprocess.py          # PDF → Image chunks conversion (building block)
│   ├── ocr_enrich_phi4_multithreaded.py  # OCR enrichment (building block)
│   ├── setup_multimodal_embeddings.py     # Create embeddings (building block)
│   ├── persistent_multimodal_rag.py # MAIN RAG system (production ready)
│   ├── hybrid_search.py           # Hybrid search engine (building block)
│   ├── PROJECT_STRUCTURE.md       # This file
│   └── requirements.txt          # Python dependencies
│
├── data/                          # Data storage
│   ├── raw/                       # Original data
│   │   └── chemical-compounds.pdf # Source PDF
│   ├── processed/                 # Processed data
│   │   ├── pdf_extracted_images/  # Image chunks from PDF
│   │   └── chemical-compounds_chunks_enriched.jsonl  # Final enriched data
│   └── embeddings/                # Embeddings storage
│       └── multimodal_embeddings.pkl  # Text + image embeddings
│
└── models/                        # Model storage
    └── PHI4/                      # Phi-4 multimodal model
```

## 🔄 **Pipeline Flow**

### **Step 0: Model Setup (One-time)**
```bash
python download_phi4_model.py
```
- Downloads Phi-4 multimodal model
- Output: `/models/PHI4/`

### **Step 1: PDF Preprocessing**
```bash
python pdf_preprocess.py --input /home/himanshu/dev/data/raw/chemical-compounds.pdf --outdir /home/himanshu/dev/data/processed
```
- Converts PDF to image chunks
- Output: `/data/processed/pdf_extracted_images/` + `chemical-compounds_chunks.jsonl`

### **Step 2: OCR Enrichment**
```bash
python ocr_enrich_phi4_multithreaded.py --limit 108 --batch-size 4 --num-workers 6
```
- Extracts text from image chunks using Phi-4
- Output: `/data/processed/chemical-compounds_chunks_enriched.jsonl`

### **Step 3: Embeddings Creation**
```bash
python setup_multimodal_embeddings.py
```
- Creates text and image embeddings using CLIP
- Output: `/data/embeddings/multimodal_embeddings.pkl`

### **Step 4: RAG System Usage**
```bash
python persistent_multimodal_rag.py --interactive
```
- Interactive query interface with persistent storage
- Uses embeddings for intelligent responses with Phi-4 generation

## 📊 **Data Files**

### **Required Files:**
- `chemical-compounds.pdf` - Source document
- `chemical-compounds_chunks_enriched.jsonl` - Final enriched data (981 chunks)
- `multimodal_embeddings.pkl` - Embeddings for RAG system
- `pdf_extracted_images/` - Image chunks directory

### **File Sizes:**
- PDF: ~50MB
- Enriched JSONL: ~15MB
- Embeddings: ~2MB
- Images: ~200MB

## 🚀 **Quick Start**

### **Setup Environment:**
```bash
cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate
pip install -r requirements.txt
```

### **Run RAG System:**
```bash
python persistent_multimodal_rag.py --interactive
```

### **Example Queries:**
- `ask what is benzene?`
- `ask methane properties`
- `ask chemical compounds structure`

## 🔧 **System Requirements**

- **Python**: 3.11+ (in .venv_phi4_req)
- **GPU**: NVIDIA L4 (23GB VRAM)
- **RAM**: 32GB+
- **Storage**: 50GB+ free space
- **CUDA**: 12.6+ (PyTorch compatible)

## 📈 **Performance**

- **Model Loading**: ~4-5 seconds
- **Query Processing**: ~1-2 seconds
- **Search Accuracy**: High (improved algorithm)
- **Response Quality**: Context-aware, source-attributed

## 🎯 **Key Features**

- **Complete Pipeline**: All building blocks preserved for reproducibility
- **Multimodal Search**: Text + image embeddings with CLIP
- **OCR Enhancement**: 981 chunks enriched with Phi-4
- **Hybrid Retrieval**: Multiple search strategies with intelligent key term extraction
- **Phi-4 Generation**: Context-aware response generation
- **Persistent Storage**: SQLite database for query history and caching
- **Interactive Interface**: User-friendly query system
- **Reproducible**: All components needed to recreate the system

---

**Last Updated**: 2025-09-13  
**Status**: Production Ready - Reproducible System  
**Database**: 981 chemical compound chunks, fully OCR-enhanced  
**Main System**: persistent_multimodal_rag.py  
**Reproducibility**: All building blocks preserved
