## LLM Models Used in This Project

This document summarizes the **local LLM models** referenced in this repository, including:

- **Canonical model ID / version**
- **Quantization / precision and format**
- **Local storage path**
- **Approximate or expected disk footprint**
- **Where and how they are used**

All size figures marked as “approx.” come from comments in the scripts or typical sizes documented by model authors; run the noted helper scripts to see the exact size currently on disk.

---

### Installed model folders under `/home/himanshu/dev/models`

As of the latest scan of your filesystem, **three model directories are actually present**:

- `/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED`
- `/home/himanshu/dev/models/PHI4_ONNX`
- `/home/himanshu/dev/models/QWEN_AWQ`

The sections below focus specifically on how these three installed models are used in the codebase.

---

### Gemma-3 12B (QAT, GGUF 4-bit)

- **Model ID**: `google/gemma-3-12b-it-qat-q4_0-gguf`
- **Format**: `GGUF`
- **Quantization / precision**: Q4\_0 (4‑bit, QAT; quantized weights)
- **Local path (target dir)**: `/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_GGUF`
- **Download script**: `download_gemma3_gguf.py`
  - Uses `huggingface_hub.snapshot_download` with `DEFAULT_REPO = "google/gemma-3-12b-it-qat-q4_0-gguf"`.
  - Script prints **per‑file sizes** and a summary after download.
- **Space consumption**:
  - Script warns that **Gemma-3 12B GGUF can exceed ~8 GB** depending on extra assets.
  - Exact size is determined at runtime via `target_dir.rglob("*")` and printed.
- **Usage**:
  - Intended for **GGUF-based inference** (e.g., llama.cpp-compatible stacks), referenced alongside other Gemma-3 tooling in:
    - `test_gemma3_multimodal.py`
    - `multimodal_qa_runner_gemma3.py`
    - `test_gemma3_vllm.py`

---

### Gemma-3 12B (QAT, Unquantized Checkpoint)

- **Model ID**: `google/gemma-3-12b-it-qat-q4_0-unquantized`
- **Format**: Standard Hugging Face transformer checkpoint (non-GGUF)
- **Quantization / precision**: Stored unquantized; intended for **runtime quantization** (e.g., bitsandbytes) by vLLM / Transformers.
- **Local path (target dir)**: `/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED`
- **Download script**: `download_gemma3_unquantized.py`
  - Uses `huggingface_hub.snapshot_download` after clearing the target directory.
  - Prints per‑file sizes and a summary after download.
- **Space consumption**:
  - Script warns: **“The unquantized Gemma snapshot exceeds 15 GB.”**
  - Expect **15–20 GB** depending on optimizer states and auxiliary artifacts.
- **Usage**:
  - **Actually installed** at `/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED` on your machine.
  - Intended for **vLLM / Transformers** flows where quantization is applied at load time (e.g., via bitsandbytes or vLLM’s internal quantization).
  - Used in Gemma‑3 runners:
    - `multimodal_qa_runner_gemma3.py` (`GEMMA_MODEL_PATH = Path("/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED")`)
    - `run_gemma3_qa_background.sh` (`--model-path /home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED`)
  - **Pipeline role**:
    - Serves as the **Gemma‑3 backend** for your Gemma‑specific multimodal QA + RAG runs, typically launched via `run_gemma3_qa_background.sh`.

---

### Qwen2.5-VL-7B-Instruct (Full-precision)

- **Model ID**: `Qwen/Qwen2.5-VL-7B-Instruct`
- **Format**: Hugging Face Transformers
- **Quantization / precision**: Full precision (FP16 / BF16) depending on runtime config.
- **Local path (target dir)**: `/home/himanshu/dev/models/QWEN`
- **Download script**: `download_qwen2_5_vl.py`
  - Uses `snapshot_download` with `repo_id="Qwen/Qwen2.5-VL-7B-Instruct"`.
  - Prints per‑file sizes and overall disk usage in the models folder.
- **Space consumption**:
  - Script comments: **“Qwen2.5-VL-7B-Instruct is typically ~14–16 GB.”**
  - Actual usage is printed after download via `shutil.disk_usage`.
- **Usage**:
  - Multimodal Qwen runner scripts and tests (full‑precision path), e.g.:
    - `test_qwen_only.py`
    - `test_qwen_with_gemma_context.py`
    - `test_multimodal_models.py`

---

### Qwen2.5-VL-7B-Instruct-AWQ (4-bit Quantized)

- **Model ID**: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`
- **Format**: Hugging Face Transformers (AWQ-quantized weights)
- **Quantization / precision**: **AWQ 4‑bit quantization**
- **Local path (target dir)**: `/home/himanshu/dev/models/QWEN_AWQ`
- **Download script**: `download_qwen2_5_vl_awq.py`
  - Uses `snapshot_download` with `repo_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"`.
  - Prints per‑file sizes and total disk usage in `models` directory.
- **Space consumption**:
  - Script notes: **“AWQ quantized model is typically ~2.6 GB (much smaller than full precision).”**
  - Warns if free disk space is below **5 GB** before download.
- **Usage**:
  - **Actually installed** at `/home/himanshu/dev/models/QWEN_AWQ` on your machine.
  - **Primary RAG + QA path with vLLM**:
    - `multimodal_qa_runner_vllm.py`: `DEFAULT_MODEL_PATH = Path("/home/himanshu/dev/models/QWEN_AWQ")`  
      → vLLM + ChromaDB RAG wrapper (`VLLMRagWrapper`) uses this path as the **default Qwen AWQ model**.
    - `multimodal_qa_runner.py`: Qwen runner class defaults to `model_path: str = "/home/himanshu/dev/models/QWEN_AWQ"`.
    - `modular_multimodal_rag.py`: CLI default `--qwen-path "/home/himanshu/dev/models/QWEN_AWQ"` for the modular RAG entrypoint.
    - `persistent_multimodal_rag.py`: uses `qwen_path = Path("/home/himanshu/dev/models/QWEN_AWQ")` to power the long‑running RAG loop.
  - **Transformers-based multimodal usage**:
    - `model_manager.py` (`load_qwen_model`):
      - Loads `Qwen2_5_VLForConditionalGeneration.from_pretrained(str(self.qwen_path), dtype=torch.float16, device_map="auto", trust_remote_code=True)` from this directory.
      - Loads an `AutoProcessor` from the same path to handle text + images.
    - Test / debug scripts that explicitly use this installed model:
      - `test_qwen_awq.py`, `test_qwen_awq_simple.py`, `test_qwen_only.py`
      - `test_multimodal_models.py`, `test_model_loading.py`
      - `simple_model_test.py`, `quick_multimodal_test.py`
      - `corrected_multimodal_test.py`, `working_qwen_test.py`, `debug_individual_models.py`, `test_phi4_vs_qwen.py`, `test_vllm_init.py`
  - **Pipeline role**:
    - This is the **main production multimodal LLM** in your project: it powers both **batch RAG regeneration flows** and **online multimodal QA** via vLLM and Transformers.

---

### Phi-4 (Transformers, Multimodal)

- **Model ID**: `microsoft/phi-4`
- **Format**: Hugging Face Transformers
- **Quantization / precision**: Loaded with `torch_dtype=torch.float16` (FP16) and `device_map="auto"`.
- **Local path (target dir)**: `/home/himanshu/dev/models/PHI4`
- **Download script**: `download_phi4_model.py`
  - Uses `AutoModel.from_pretrained("microsoft/phi-4", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)`.
  - Downloads and saves both **model** and **AutoProcessor** to the local directory.
- **Space consumption**:
  - Exact size is not hard-coded, but the full Phi-4 checkpoint is expected to be **tens of GB** once weights and auxiliary files are stored.
  - The script can be extended to compute and print `total_size_gb` as in the ONNX downloader (see below).
- **Usage**:
  - Used as a **multimodal baseline** in:
    - `model_manager.py` (non‑ONNX path originally).
    - `test_multimodal_models.py` and `phi4-mm.py` for multimodal QA and comparison against Qwen.

---

### Phi-4 Multimodal Instruct (ONNX, Quantized)

- **Model ID**: `microsoft/Phi-4-multimodal-instruct-onnx`
- **Format**: ONNX (for ONNX Runtime / onnxruntime-genai)
- **Quantization / precision**: Quantized ONNX (exact scheme defined by the Hugging Face repo; script comments describe it as a **“Quantized version for research comparison with Qwen AWQ”**).
- **Local path (target dir)**: `/home/himanshu/dev/models/PHI4_ONNX`
- **Download script**: `download_phi4_onnx.py`
  - Uses `snapshot_download` with `repo_id="microsoft/Phi-4-multimodal-instruct-onnx"`.
  - After download, iterates over all files, logging each file’s size and computing:
    - `Total model size: {total_size_gb:.2f} GB`.
- **Space consumption**:
  - **Computed precisely at runtime** in `download_phi4_onnx.py`:
    - `total_size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())`
    - Logged as `Total model size: X.XX GB`.
- **Usage**:
  - **Actually installed** at `/home/himanshu/dev/models/PHI4_ONNX` on your machine.
  - In `model_manager.py`, `load_phi4_model` treats this directory as the canonical Phi‑4 deployment:
    - Verifies the presence of `phi4_path / "gpu" / "gpu-int4-rtn-block-32"` (which exists under `PHI4_ONNX/gpu/gpu-int4-rtn-block-32/`).
    - Marks Phi‑4 as available for **onnxruntime-genai**; actual sessions are created later by the generator code.
  - Runners and RAG entrypoints that use this installed ONNX model:
    - `multimodal_qa_runner.py`: default `model_path: str = "/home/himanshu/dev/models/PHI4_ONNX"` for the Phi‑4 multimodal runner.
    - `modular_multimodal_rag.py`: CLI default `--phi4-path "/home/himanshu/dev/models/PHI4_ONNX"` for the modular RAG pipeline.
    - `persistent_multimodal_rag.py`: can be configured to use this ONNX directory for long‑running Phi‑4 inference.
  - Test / debug scripts using this installed model:
    - `simple_model_test.py`, `quick_multimodal_test.py`, `corrected_multimodal_test.py`
    - `test_multimodal_models.py`, `test_model_loading.py`, `debug_individual_models.py`
  - **Pipeline role**:
    - Serves as the **quantized Phi‑4 backend** for efficient multimodal inference and as a comparison point against Qwen AWQ in the modular and persistent RAG workflows.

---

## Summary Table

| Model ID                                      | Local Path                                   | Format            | Quantization / Precision              | Approx. Disk Use (from code/comments)        |
|----------------------------------------------|----------------------------------------------|-------------------|----------------------------------------|----------------------------------------------|
| `google/gemma-3-12b-it-qat-q4_0-gguf`        | `/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_GGUF` | GGUF             | Q4\_0 (4‑bit QAT)                     | \> ~8 GB (varies with extra assets)          |
| `google/gemma-3-12b-it-qat-q4_0-unquantized` | `/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED` | HF checkpoint | Unquantized (for runtime quantization) | \> ~15 GB (script warning)                   |
| `Qwen/Qwen2.5-VL-7B-Instruct`                | `/home/himanshu/dev/models/QWEN`             | Transformers      | Full precision (FP16/BF16)            | ~14–16 GB (script comment)                   |
| `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`            | `/home/himanshu/dev/models/QWEN_AWQ`         | Transformers      | AWQ 4‑bit quantized                   | ~2.6 GB (script comment)                     |
| `microsoft/phi-4`                            | `/home/himanshu/dev/models/PHI4`             | Transformers      | FP16 (`torch_dtype=torch.float16`)    | Tens of GB (not hard-coded; depends on repo) |
| `microsoft/Phi-4-multimodal-instruct-onnx`   | `/home/himanshu/dev/models/PHI4_ONNX`        | ONNX              | Quantized ONNX                        | Computed at runtime and logged in script     |

---

### How to Recompute Exact Disk Usage

For **exact current disk usage** on your machine:

- **Gemma-3 GGUF / Unquantized**:
  - Re-run `download_gemma3_gguf.py` or `download_gemma3_unquantized.py` with `--skip-disk-check` and inspect the printed per‑file sizes.
- **Qwen2.5-VL models**:
  - Re-run `download_qwen2_5_vl_awq.py` or `download_qwen2_5_vl.py`; both print per‑file sizes and `shutil.disk_usage` for the models folder.
- **Phi-4 ONNX**:
  - Run `download_phi4_onnx.py`; it logs `Total model size: X.XX GB` after summing all files.
- **Generic check for any model directory**:
  - From a shell:
    - `du -sh /home/himanshu/dev/models/*`
    - `du -sh /home/himanshu/dev/models/QWEN_AWQ`


