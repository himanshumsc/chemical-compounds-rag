# Prompts and Generation Limits Used for Qwen vs Gemma Evaluation

**Date:** November 30, 2025  
**Models Evaluated:** Qwen RAG Concise, Gemma RAG Concise  
**Dataset:** 178 chemical compounds × 4 questions = 712 question-answer pairs

---

## 1. Character Limits Per Question Type

The following character limits were enforced for concise answer generation:

| Question Type | Character Limit | Max Tokens | Description |
|--------------|----------------|------------|-------------|
| **Q1** (Image-based) | 600 chars | 200 tokens | Image-based identification |
| **Q2** (Formula/Type) | 1,000 chars | 333 tokens | Chemical formula and compound type |
| **Q3** (Production) | 1,800 chars | 600 tokens | Production/manufacturing process |
| **Q4** (Uses/Hazards) | 2,000 chars | 666 tokens | Industrial uses and hazards |

**Note:** Max tokens were calculated as `character_limit / 3.0` to allow slightly longer answers while staying within character limits.

---

## 2. Full Prompt Structure

### 2.1 Base Prompt Template

The prompt structure used for all questions (Q1-Q4) follows this format:

```
Based on the following chemical compounds database information, please answer the user's question.

CONTEXT:
{retrieved_context_from_rag}

USER QUESTION: {original_question}

IMPORTANT INSTRUCTIONS:
- Your answer MUST be brief, concise, and to the point
- Maximum length: {character_limit} characters (strict limit)
- Focus ONLY on the most essential and relevant information
- Avoid unnecessary elaboration or repetition
- Be direct and factual
- If you exceed {character_limit} characters, your answer will be truncated

Generate a concise answer that fits within {character_limit} characters:
```

### 2.2 Prompt Components

1. **RAG Context Section:**
   - Retrieved from ChromaDB using similarity search
   - **For Q1:** Image-based similarity search (5 chunks) - finds similar molecular structures
   - **For Q2-Q4:** Text-based similarity search (5 chunks) - finds relevant text passages
   - Context is formatted as numbered sources: `Source 1: ...`, `Source 2: ...`, etc.
   - **Total context size:** ~500-1000 characters (much smaller than OpenAI's comprehensive_text)

2. **User Question:**
   - Original question from the QA dataset
   - Preserved exactly as provided

3. **Concise Instructions:**
   - Emphasizes brevity and conciseness
   - Includes specific character limit
   - Warns about truncation if limit is exceeded

### 2.3 Example Full Prompt (Q2 - Formula/Type)

**Question:** "What is the chemical formula of 1,3-butadiene, and what type of compound is it?"

**Retrieved Context:**
```
Source 1: 1,3-Butadiene (C4H6) is a conjugated diene...
Source 2: The compound is classified as an alkadiene...
Source 3: Chemical formula: C4H6, molecular weight: 54.09...
Source 4: Used in polymer manufacturing...
Source 5: Produced via steam cracking of hydrocarbons...
```

**Full Prompt:**
```
Based on the following chemical compounds database information, please answer the user's question.

CONTEXT:
Source 1: 1,3-Butadiene (C4H6) is a conjugated diene...
Source 2: The compound is classified as an alkadiene...
Source 3: Chemical formula: C4H6, molecular weight: 54.09...
Source 4: Used in polymer manufacturing...
Source 5: Produced via steam cracking of hydrocarbons...

USER QUESTION: What is the chemical formula of 1,3-butadiene, and what type of compound is it?

IMPORTANT INSTRUCTIONS:
- Your answer MUST be brief, concise, and to the point
- Maximum length: 1000 characters (strict limit)
- Focus ONLY on the most essential and relevant information
- Avoid unnecessary elaboration or repetition
- Be direct and factual
- If you exceed 1000 characters, your answer will be truncated

Generate a concise answer that fits within 1000 characters:
```

### 2.4 Example Full Prompt (Q1 - Image-based)

**Question:** "Look at the molecular structure diagram in the image. What chemical compound is shown, and what are its key properties?"

**RAG Context Retrieval for Q1:**
- Uses **image similarity search** (not text search)
- The image is saved temporarily and used to find similar molecular structure images in ChromaDB
- Retrieves 5 chunks based on image similarity
- These chunks contain text descriptions of compounds with similar structures

**Retrieved Context (from image similarity search):**
```
Source 1: 1,3-Butadiene structure shows four carbon atoms...
Source 2: The compound has double bonds between carbons 1-2 and 3-4...
Source 3: Key properties include: boiling point -4.4°C, density 0.6149 g/cm³...
Source 4: Used as a monomer in synthetic rubber production...
Source 5: Colorless gas at room temperature...
```

**Full Augmented Prompt:**
```
Based on the following chemical compounds database information, please answer the user's question.

CONTEXT:
Source 1: 1,3-Butadiene structure shows four carbon atoms...
Source 2: The compound has double bonds between carbons 1-2 and 3-4...
Source 3: Key properties include: boiling point -4.4°C, density 0.6149 g/cm³...
Source 4: Used as a monomer in synthetic rubber production...
Source 5: Colorless gas at room temperature...

USER QUESTION: Look at the molecular structure diagram in the image. What chemical compound is shown, and what are its key properties?

IMPORTANT INSTRUCTIONS:
- Your answer MUST be brief, concise, and to the point
- Maximum length: 600 characters (strict limit)
- Focus ONLY on the most essential and relevant information
- Avoid unnecessary elaboration or repetition
- Be direct and factual
- If you exceed 600 characters, your answer will be truncated

Generate a concise answer that fits within 600 characters:
```

**Final Input to Model:**
- **Image:** Passed separately via vLLM's multimodal API (`multi_modal_data: {"image": [image]}`)
- **Text Prompt:** The augmented prompt above (with RAG context)
- **Both image and text context are used together** for Q1 generation

**Key Point:** Q1 uses **BOTH**:
1. The actual image (for visual identification)
2. RAG-retrieved text chunks (found via image similarity search, providing additional context about similar compounds)

---

## 3. Chat Template Application

After constructing the prompt, it is passed through the model's chat template:

### 3.1 For Q1 (Vision Questions)

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil_object},
            {"type": "text", "text": augmented_prompt}
        ]
    }
]
templated = processor.apply_chat_template(messages, add_generation_prompt=True)
```

### 3.2 For Q2-Q4 (Text-Only Questions)

```python
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": augmented_prompt}]
    }
]
templated = processor.apply_chat_template(messages, add_generation_prompt=True)
```

### 3.3 Chat Template Output Format

The chat template adds model-specific formatting. For Qwen models, the output format is:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{augmented_prompt}<|im_end|>
<|im_start|>assistant

```

**System Message:** "You are a helpful assistant." (added automatically by chat template)

**Special Tokens:**
- `<|im_start|>` - Start of message block
- `<|im_end|>` - End of message block
- `system`, `user`, `assistant` - Role identifiers

The final prompt sent to the model includes the system message and proper formatting tokens.

---

## 4. Generation Parameters

### 4.1 vLLM Sampling Parameters

**For Q1 (Vision):**
```python
SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200,  # ~200 tokens for 600 chars
    stop=None
)
```

**For Q2 (Formula/Type):**
```python
SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=333,  # ~333 tokens for 1000 chars
    stop=None
)
```

**For Q3 (Production):**
```python
SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=600,  # ~600 tokens for 1800 chars
    stop=None
)
```

**For Q4 (Uses/Hazards):**
```python
SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=666,  # ~666 tokens for 2000 chars
    stop=None
)
```

### 4.2 Common Parameters

- **Temperature:** 0.7 (balanced creativity/consistency)
- **Top-p:** 0.9 (nucleus sampling)
- **Stop tokens:** None (no early stopping)
- **Max tokens:** Question-specific (see above)

---

## 5. RAG Configuration

### 5.1 ChromaDB Settings

- **Database Path:** `/home/himanshu/dev/data/chromadb`
- **Number of Chunks:** 5 chunks per question
- **Search Type:**
  - Q1: Image similarity search
  - Q2-Q4: Text similarity search

### 5.2 Context Formatting

Retrieved chunks are formatted as:
```
Source 1: {chunk_1_text}
Source 2: {chunk_2_text}
Source 3: {chunk_3_text}
Source 4: {chunk_4_text}
Source 5: {chunk_5_text}
```

---

## 6. Answer Post-Processing

### 6.1 Truncation Logic

If the generated answer exceeds the character limit, it is truncated:

```python
def _truncate_answer(text: str, max_chars: int) -> Tuple[str, bool]:
    """
    Truncate answer at sentence or word boundary.
    Adds '...' if truncated.
    """
    if len(text) <= max_chars:
        return text, False
    
    # Try to truncate at sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_exclamation = truncated.rfind('!')
    last_question = truncated.rfind('?')
    
    last_sentence_end = max(last_period, last_exclamation, last_question)
    
    if last_sentence_end > max_chars * 0.7:  # At least 70% of limit
        truncated = truncated[:last_sentence_end + 1]
        return truncated + " ...", True
    
    # Fallback: truncate at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space]
        return truncated + " ...", True
    
    # Last resort: hard truncate
    return truncated + " ...", True
```

### 6.2 Answer Cleaning

- Removes leading/trailing whitespace
- Removes model-specific formatting tokens if present
- Preserves sentence structure

---

## 7. Model-Specific Details

### 7.1 Qwen RAG Concise

- **Model:** Qwen2.5-VL-7B-Instruct-AWQ
- **Model Path:** `/home/himanshu/dev/models/QWEN_AWQ`
- **Inference Engine:** vLLM
- **Quantization:** AWQ (Activation-aware Weight Quantization)
- **Max Model Length:** 8192 tokens
- **GPU Memory Utilization:** 0.80
- **Enforce Eager:** True (disables CUDA graph compilation)

### 7.2 Gemma RAG Concise

- **Model:** Gemma-3 (specific variant)
- **Inference Engine:** vLLM
- **Same prompt structure and limits as Qwen**
- **Same RAG configuration**

---

## 8. Prompt Variations

### 8.1 Without RAG Context

If RAG is disabled or no context is found, the prompt is simplified:

```
{original_question}

IMPORTANT: Your answer should be brief, concise, and to the point. Focus on the most relevant information only.
```

### 8.2 Without Character Limit

If no character limit is specified, the concise instruction is:

```
IMPORTANT: Your answer should be brief, concise, and to the point. Focus on the most relevant information only.
```

---

## 9. Complete Example: Q2 Question

### 9.1 Input Question
```
"What is the chemical formula of 1,3-butadiene, and what type of compound is it?"
```

### 9.2 Retrieved RAG Context
```
Source 1: 1,3-Butadiene, also known as buta-1,3-diene, is an organic compound with the chemical formula C4H6. It is a colorless gas that is important as a monomer in the production of synthetic rubber.

Source 2: The compound is classified as a conjugated diene, meaning it has two double bonds separated by a single bond. This structure gives it unique chemical properties.

Source 3: Chemical formula: C4H6. Molecular weight: 54.09 g/mol. Boiling point: -4.4°C. The compound is an alkadiene, specifically a 1,3-diene.

Source 4: 1,3-Butadiene is used primarily in the manufacture of polymers, including polybutadiene rubber and styrene-butadiene rubber (SBR).

Source 5: The compound is produced via steam cracking of hydrocarbons, particularly from n-butane or n-butene. It is a key intermediate in the petrochemical industry.
```

### 9.3 Full Augmented Prompt (Before Chat Template)
```
Based on the following chemical compounds database information, please answer the user's question.

CONTEXT:
Source 1: 1,3-Butadiene, also known as buta-1,3-diene, is an organic compound with the chemical formula C4H6. It is a colorless gas that is important as a monomer in the production of synthetic rubber.

Source 2: The compound is classified as a conjugated diene, meaning it has two double bonds separated by a single bond. This structure gives it unique chemical properties.

Source 3: Chemical formula: C4H6. Molecular weight: 54.09 g/mol. Boiling point: -4.4°C. The compound is an alkadiene, specifically a 1,3-diene.

Source 4: 1,3-Butadiene is used primarily in the manufacture of polymers, including polybutadiene rubber and styrene-butadiene rubber (SBR).

Source 5: The compound is produced via steam cracking of hydrocarbons, particularly from n-butane or n-butene. It is a key intermediate in the petrochemical industry.

USER QUESTION: What is the chemical formula of 1,3-butadiene, and what type of compound is it?

IMPORTANT INSTRUCTIONS:
- Your answer MUST be brief, concise, and to the point
- Maximum length: 1000 characters (strict limit)
- Focus ONLY on the most essential and relevant information
- Avoid unnecessary elaboration or repetition
- Be direct and factual
- If you exceed 1000 characters, your answer will be truncated

Generate a concise answer that fits within 1000 characters:
```

### 9.4 Final Prompt (After Chat Template)

After applying the chat template, the final prompt sent to the model is:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Based on the following chemical compounds database information, please answer the user's question.

CONTEXT:
Source 1: 1,3-Butadiene, also known as buta-1,3-diene, is an organic compound with the chemical formula C4H6. It is a colorless gas that is important as a monomer in the production of synthetic rubber.

Source 2: The compound is classified as a conjugated diene, meaning it has two double bonds separated by a single bond. This structure gives it unique chemical properties.

Source 3: Chemical formula: C4H6. Molecular weight: 54.09 g/mol. Boiling point: -4.4°C. The compound is an alkadiene, specifically a 1,3-diene.

Source 4: 1,3-Butadiene is used primarily in the manufacture of polymers, including polybutadiene rubber and styrene-butadiene rubber (SBR).

Source 5: The compound is produced via steam cracking of hydrocarbons, particularly from n-butane or n-butene. It is a key intermediate in the petrochemical industry.

USER QUESTION: What is the chemical formula of 1,3-butadiene, and what type of compound is it?

IMPORTANT INSTRUCTIONS:
- Your answer MUST be brief, concise, and to the point
- Maximum length: 1000 characters (strict limit)
- Focus ONLY on the most essential and relevant information
- Avoid unnecessary elaboration or repetition
- Be direct and factual
- If you exceed 1000 characters, your answer will be truncated

Generate a concise answer that fits within 1000 characters:<|im_end|>
<|im_start|>assistant

```

### 9.5 Expected Output
```
1,3-Butadiene's chemical formula is C4H6. It is a conjugated diene, specifically classified as an alkadiene or 1,3-diene compound.
```

**Character count:** ~120 characters (well within 1000 limit)

---

## 10. Summary

### 10.1 Key Prompt Features

1. **RAG Context Integration:** 5 chunks retrieved and formatted as numbered sources
2. **Concise Instructions:** Strong emphasis on brevity and character limits
3. **Question Preservation:** Original question preserved exactly
4. **Character Limits:** Strict limits enforced per question type
5. **Post-Processing:** Truncation at sentence/word boundaries if needed

### 10.2 Generation Limits

| Question | Char Limit | Token Limit | Temperature | Top-p |
|----------|------------|-------------|-------------|-------|
| Q1 | 600 | 200 | 0.7 | 0.9 |
| Q2 | 1,000 | 333 | 0.7 | 0.9 |
| Q3 | 1,800 | 600 | 0.7 | 0.9 |
| Q4 | 2,000 | 666 | 0.7 | 0.9 |

### 10.3 Prompt Instruction

**Core instruction:** "Your answer MUST be brief, concise, and to the point"

This instruction is included in every prompt to ensure concise outputs that fit within the specified character limits.

---

---

## 11. OpenAI Baseline Prompts

The OpenAI baseline answers (used for comparison) were generated using GPT-4o with the following prompts:

### 11.1 System Prompt

```
You are an expert chemistry educator providing comprehensive, accurate answers to questions about chemical compounds.

REQUIREMENTS:
1. Answer the question based ONLY on the provided Comprehensive Compound Information
2. Use all relevant information from the comprehensive text, including timeline references and cross-references
3. Provide detailed, educational answers that demonstrate deep understanding
4. Include relevant chemical formulas, properties, historical context, and applications when available
5. If information is not in the provided text, do not make assumptions

BOUNDARY CONDITION:
All answers must be based ONLY on the provided Comprehensive Compound Information text. Do not use outside knowledge or assumptions beyond that text.

OUTPUT FORMAT:
Return ONLY the answer text, no additional explanations or formatting.
```

### 11.2 User Prompt Template (Q2-Q4)

**For Q2-Q4, OpenAI uses the FULL comprehensive_text:**

```
Question: {question}

Comprehensive Compound Information for {compound_name}:
{comprehensive_text}

Answer the question based on the comprehensive information provided above.
```

**Key Points:**
- **comprehensive_text** is the full text (typically 3000-5000 characters) from the compound's comprehensive entry
- This includes all available information about the compound: properties, production, uses, hazards, history, etc.
- Much more context than RAG chunks (which are ~500-1000 chars total from 5 chunks)

### 11.3 OpenAI Generation Parameters

- **Model:** GPT-4o
- **Temperature:** 0.7
- **Max Tokens:** 500 (for Q2-Q4)
- **Max Tokens (Q1):** Not regenerated (preserved from original generation)

### 11.4 Example OpenAI Prompt (Q2)

**Question:** "What is the chemical formula of 1,3-butadiene, and what type of compound is it?"

**Comprehensive Text:** (Full comprehensive_text from compound file, typically 3000-5000 characters)

**Full Messages:**
```python
[
    {
        "role": "system",
        "content": "You are an expert chemistry educator providing comprehensive, accurate answers to questions about chemical compounds.\n\nREQUIREMENTS:\n1. Answer the question based ONLY on the provided Comprehensive Compound Information\n2. Use all relevant information from the comprehensive text, including timeline references and cross-references\n3. Provide detailed, educational answers that demonstrate deep understanding\n4. Include relevant chemical formulas, properties, historical context, and applications when available\n5. If information is not in the provided text, do not make assumptions\n\nBOUNDARY CONDITION:\nAll answers must be based ONLY on the provided Comprehensive Compound Information text. Do not use outside knowledge or assumptions beyond that text.\n\nOUTPUT FORMAT:\nReturn ONLY the answer text, no additional explanations or formatting."
    },
    {
        "role": "user",
        "content": "Question: What is the chemical formula of 1,3-butadiene, and what type of compound is it?\n\nComprehensive Compound Information for 1,3-Butadiene:\n{full_comprehensive_text_here}\n\nAnswer the question based on the comprehensive information provided above."
    }
]
```

### 11.5 Q1 (Image-based) Handling

**Note:** Q1 answers were **NOT regenerated** using the comprehensive text generator. They were preserved from the original generation process.

**Original Q1 Generation (OpenAI):**
- **Model:** GPT-4o (with vision capabilities)
- **Input:** Image + question text only
- **Context:** **NO comprehensive text or RAG chunks** - just the image and question
- **Prompt Structure:** Likely a simple prompt asking to identify the compound from the image
- **This is different from Qwen/Gemma Q1**, which uses both image AND RAG-retrieved chunks

**Key Difference:**
- **OpenAI Q1:** Image + Question only (no additional text context)
- **Qwen/Gemma Q1:** Image + Question + RAG chunks (retrieved via image similarity search)

This explains why OpenAI Q1 answers might be more focused on visual identification, while Qwen/Gemma Q1 answers have additional context from similar compounds found via image similarity search.

### 11.6 Key Differences: OpenAI vs Qwen/Gemma

| Aspect | OpenAI Baseline | Qwen/Gemma RAG Concise |
|--------|----------------|------------------------|
| **Q1 Context** | Image + Question only (no text context) | Image + Question + RAG chunks (image similarity search, ~500-1000 chars) |
| **Q2-Q4 Context** | Full comprehensive_text (3000-5000 chars) | RAG-retrieved chunks (text similarity search, 5 chunks, ~500-1000 chars total) |
| **Context Size** | Much larger (full text) | Much smaller (selective chunks) |
| **Prompt Style** | Educational, comprehensive | Concise, brief, to the point |
| **Character Limits** | None (max_tokens=500) | Strict limits (Q1:600, Q2:1000, Q3:1800, Q4:2000) |
| **Instruction** | "Provide detailed, educational answers" | "MUST be brief, concise, and to the point" |
| **Model** | GPT-4o | Qwen2.5-VL-7B-Instruct-AWQ / Gemma-3 |
| **Inference** | OpenAI API | vLLM (local) |

**Important Notes:**

**Q1 Context Difference:**
- **OpenAI Q1:** Uses only the image and question - no additional text context
- **Qwen/Gemma Q1:** Uses image + question + RAG chunks (retrieved via image similarity search)
- This means Qwen/Gemma Q1 has **more context** than OpenAI Q1, which may affect comparison fairness

**Q2-Q4 Context Difference:**
- **OpenAI Q2-Q4:** Uses **full comprehensive_text** (3000-5000 characters) - all available information
- **Qwen/Gemma Q2-Q4:** Uses **RAG-retrieved chunks** (5 chunks, ~500-1000 characters total) - selective information
- OpenAI has **3-5x more context** than Qwen/Gemma for Q2-Q4
- This is a significant difference that affects answer quality and comprehensiveness

### 11.7 Why These Differences Matter

1. **Context Volume:** OpenAI uses full comprehensive text (much more context), while Qwen/Gemma use RAG-retrieved chunks (selective context)
2. **Answer Style:** OpenAI emphasizes comprehensiveness, Qwen/Gemma emphasize conciseness
3. **Length Control:** OpenAI has no character limits, Qwen/Gemma have strict limits
4. **This explains why:**
   - OpenAI answers are longer (mean: 522.2 chars vs Qwen: 218.7, Gemma: 231.4)
   - Qwen/Gemma answers are more concise by design
   - Evaluation metrics compare concise answers to comprehensive baseline

---

**Document Generated:** November 30, 2025  
**Source Code:** 
- Qwen/Gemma: `/home/himanshu/dev/code/multimodal_qa_runner_vllm.py`
- OpenAI: `/home/himanshu/dev/qa_generation/generate_qa_pairs_comprehensive_update.py`
**Output Examples:** 
- Qwen: `/home/himanshu/dev/output/qwen_rag_concise/`
- Gemma: `/home/himanshu/dev/output/gemma3_rag_concise/`
- OpenAI: `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive/`

