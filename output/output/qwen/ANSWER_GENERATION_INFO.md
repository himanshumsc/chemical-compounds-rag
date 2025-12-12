# Answer Generation Information for dev/output/qwen

## Program That Generated the Files

**Script**: `/home/himanshu/dev/code/multimodal_qa_runner.py`

**Execution Mode**: Batched execution (using `--batched` flag)

**Command Used** (inferred from logs):
```bash
python multimodal_qa_runner.py \
  --model qwen \
  --batched \
  --total-sets 178 \
  --batch-size 2 \
  --max-new-tokens 128 \
  --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components \
  --output-dir /home/himanshu/dev/output
```

---

## Prompt Structure

### How Prompts Are Constructed

1. **Source**: Questions are read directly from QA pair JSON files located at:
   - `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components/*.json`

2. **No System Prompt**: The script does NOT add any system prompt or instruction. The questions from the JSON files are used directly as prompts.

3. **Qwen Chat Template**: For Qwen model, the questions are formatted using Qwen's chat template via `apply_chat_template()`:
   ```python
   messages = [{
       "role": "user",
       "content": [
           {"type": "image", "image": image},  # For Q1 only
           {"type": "text", "text": prompt},   # The question
       ],
   }]
   templated = processor.apply_chat_template(messages, add_generation_prompt=True)
   ```

4. **Question Types**:
   - **Q1**: Question + Image (multimodal)
   - **Q2-Q4**: Question only (text-only)

---

## Example Prompts Used

Based on the QA pair files, here are the actual prompts used:

### Example 1: 1,3-Butadiene

**Q1 (with image)**:
```
"Look at the molecular structure diagram in the image. What chemical compound is shown, and what are its key properties?"
```

**Q2 (text-only)**:
```
"What is the chemical formula of 1,3-butadiene, and what type of compound is it?"
```

**Q3 (text-only)**:
```
"Explain how 1,3-butadiene is produced in the United States."
```

**Q4 (text-only)**:
```
"Discuss the industrial uses of 1,3-butadiene and the potential hazards associated with it."
```

---

## Generation Parameters

### Model Configuration
- **Model**: Qwen2.5-VL-AWQ (loaded from `/home/himanshu/dev/models/QWEN_AWQ`)
- **Model Type**: Vision-Language Model (multimodal)

### Generation Settings
- **max_new_tokens**: 128 (default, can be overridden via `--max-new-tokens`)
- **do_sample**: True
- **temperature**: 0.7
- **pad_token_id**: EOS token ID

### Code Reference (lines 88-130 in multimodal_qa_runner.py):
```python
def generate(self, prompt: str, image: Optional[Image.Image] = None, max_new_tokens: int = 128) -> GenerationResult:
    if image is not None:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[templated], images=[image], return_tensors="pt")
    else:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]
        templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[templated], return_tensors="pt")
    
    # ... generation code ...
    outputs = self.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=self.processor.tokenizer.eos_token_id,
    )
```

---

## Processing Flow

1. **Read QA Files**: Loads JSON files from `qa_pairs_individual_components/`
2. **Extract Questions**: Gets questions from `data["qa_pairs"][i]["question"]`
3. **Load Images**: For Q1, loads image from `data["image_path"]`
4. **Format Prompt**: Uses Qwen's chat template to format question as user message
5. **Generate**: Calls Qwen model's `generate()` method
6. **Post-process**: Extracts assistant response, removes role markers
7. **Save**: Writes to `dev/output/qwen/<compound>__answers.json`

---

## Key Code Locations

- **Main Script**: `/home/himanshu/dev/code/multimodal_qa_runner.py`
- **Qwen Wrapper Class**: Lines 77-189
- **Generation Method**: Lines 88-130
- **Batched Processing**: Lines 311-444
- **Post-processing**: Lines 192-204 (removes assistant role markers)

---

## Important Notes

1. **No System Prompt**: The script does not add any system-level instructions. The model uses only the questions as prompts.

2. **Chat Template**: Qwen's `apply_chat_template()` automatically formats the messages according to Qwen's expected chat format, which may include special tokens and formatting.

3. **Image Handling**: 
   - Images are sanitized (metadata stripped) before use
   - Only Q1 uses images; Q2-Q4 are text-only

4. **Answer Extraction**: The generated response is post-processed to remove any assistant role markers or system/user headers that might appear.

5. **Batch Processing**: The script processes multiple QA sets in batches (batch_size=2) for efficiency, mixing image and text-only questions in the same batch.

---

**Generated**: 2025-01-07  
**Total Files**: 178 answer JSON files  
**Model**: Qwen2.5-VL-AWQ  
**Average Answer Length**: 548.95 characters (first answer)

