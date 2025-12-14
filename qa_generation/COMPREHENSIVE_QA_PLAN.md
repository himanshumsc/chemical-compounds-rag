# Plan: Create Comprehensive QA Pairs Version

## Objective
Create a new version of QA pairs using `comprehensive_text` instead of `main_entry_content` for questions 2, 3, and 4, while preserving question 1 (image-based) that was already updated today.

## Steps

### 1. Copy Existing Files
- **Source**: `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components/`
- **Destination**: `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive/`
- **Action**: Copy all 178 JSON files to the new directory

### 2. Create New Script
- **Script Name**: `generate_qa_pairs_comprehensive_update.py`
- **Location**: `/home/himanshu/dev/qa_generation/`
- **Functionality**:
  - Read existing QA files from `qa_pairs_individual_components_comprehensive/`
  - Read corresponding compound files from `individual_compounds/`
  - Extract `comprehensive_text` from compound files
  - Keep question 1 (index 0) unchanged
  - Regenerate answers for questions 2, 3, 4 (indices 1, 2, 3) using:
    - Original questions from existing QA files
    - `comprehensive_text` as context instead of `main_entry_content`
  - Update metadata to reflect comprehensive_text usage
  - Save updated files back to `qa_pairs_individual_components_comprehensive/`

### 3. Script Details

#### Input Sources:
- **QA Files**: `qa_pairs_individual_components_comprehensive/*.json` (copied files)
- **Compound Files**: `individual_compounds/compound_*.json`
- **Matching**: By `compound_id` or `source_file` name

#### Processing Logic:
1. For each QA file:
   - Load existing QA pairs
   - Find corresponding compound file
   - Extract `comprehensive_text` and `comprehensive_text_length`
   - Keep question 1 (image identification) as-is
   - For questions 2, 3, 4:
     - Use existing question text
     - Send to OpenAI API with `comprehensive_text` as context
     - Generate new answer based on comprehensive text
   - Update metadata:
     - Add `comprehensive_text_length`
     - Update `generated_at` timestamp
     - Add note about comprehensive_text usage

#### API Settings:
- **Model**: `gpt-4o`
- **Temperature**: `0.7`
- **Max Tokens**: `2000` (or higher if needed for comprehensive answers)
- **System Prompt**: Modified to use comprehensive_text context

#### Output:
- Updated JSON files in `qa_pairs_individual_components_comprehensive/`
- Same structure as original, but with:
  - Question 1 unchanged
  - Questions 2, 3, 4 with new answers based on comprehensive_text
  - Updated metadata

### 4. Key Differences from Original Script

| Aspect | Original Script | Comprehensive Script |
|--------|----------------|---------------------|
| Input Text | `main_entry_content` | `comprehensive_text` |
| Questions Generated | All 4 (including image Q) | Only 2, 3, 4 (keep Q1) |
| Source | Compound files only | Compound files + existing QA files |
| Output Directory | `qa_pairs_individual_components` | `qa_pairs_individual_components_comprehensive` |

### 5. Expected Results
- 178 updated JSON files
- Question 1 preserved (image-based, already updated today)
- Questions 2, 3, 4 with more comprehensive answers based on full text including timeline references
- Better answers due to richer context from `comprehensive_text`

## Implementation Order
1. Copy all files to new directory
2. Create and test the new script
3. Run script to update all 178 files
4. Verify results

