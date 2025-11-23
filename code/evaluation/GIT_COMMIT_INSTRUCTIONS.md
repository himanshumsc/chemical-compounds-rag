# Git Commit Instructions for Evaluation Files

Follow these steps to commit the evaluation files to GitHub:

## Step 1: Navigate to the project directory
```bash
cd /home/himanshu/dev
```

## Step 2: Initialize Git (if not already done)
```bash
# Check if git is initialized
if [ ! -d ".git" ]; then
    git init
    echo "# Evaluation Metrics" > README.md
    git add README.md
    git commit -m "Initial commit"
fi
```

## Step 3: Add GitHub Remote (if not already added)
```bash
# Replace with your actual GitHub repository URL
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Or if remote already exists, check it:
git remote -v
```

## Step 4: Add Evaluation Files
```bash
# Add all evaluation files
git add code/evaluation/

# Add related scripts
git add code/compare_regenerated_output.py
git add code/multimodal_qa_runner_vllm.py

# Check what will be committed
git status
```

## Step 5: Commit the Files
```bash
git commit -m "Add evaluation metrics (BLEU and BERTScore) for regenerated QWEN and OpenAI answers

- Added BLEU score calculator for regenerated data
- Added BERTScore calculator for regenerated data  
- Added comprehensive analysis reports
- Results for 712 question-answer pairs (178 files × 4 questions)
- Comparison between regenerated QWEN (vLLM) and OpenAI (comprehensive_text) answers"
```

## Step 6: Push to GitHub
```bash
# Push to main branch (or master if that's your default)
git push -u origin main

# Or if your default branch is master:
git push -u origin master
```

## Alternative: One-liner Commands

If you prefer to run everything at once:

```bash
cd /home/himanshu/dev && \
git add code/evaluation/ code/compare_regenerated_output.py code/multimodal_qa_runner_vllm.py && \
git commit -m "Add evaluation metrics (BLEU and BERTScore) for regenerated QWEN and OpenAI answers" && \
git push -u origin main
```

## Files Being Committed

### Scripts:
- `code/evaluation/scripts/bleu_score_calculator_regenerated.py`
- `code/evaluation/scripts/bertscore_calculator_regenerated.py`
- `code/evaluation/scripts/bleu_score_calculator.py`
- `code/evaluation/scripts/bertscore_calculator.py`

### Results:
- `code/evaluation/results/per_question_bleu_regenerated.csv`
- `code/evaluation/results/per_question_bertscore_regenerated.csv`
- `code/evaluation/results/summary_metrics_regenerated.json`
- `code/evaluation/results/summary_bertscore_regenerated.json`
- (and original results files)

### Documentation:
- `code/evaluation/BLEU_REGENERATED_RESULTS.md`
- `code/evaluation/BERTSCORE_REGENERATED_RESULTS.md`
- `code/evaluation/README.md`
- (and other documentation files)

### Related Files:
- `code/compare_regenerated_output.py`
- `code/multimodal_qa_runner_vllm.py`

## Note on Large Files

If you encounter issues with large result CSV/JSON files, you may want to:
1. Add them to `.gitignore` if they're too large
2. Or use Git LFS for large files:
```bash
git lfs install
git lfs track "*.csv"
git lfs track "*.json"
git add .gitattributes
```

## Troubleshooting

### If remote already exists:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

### If you need to check current branch:
```bash
git branch
```

### If you need to create and switch to main branch:
```bash
git checkout -b main
```

### If you get authentication errors:
- Use SSH instead: `git remote set-url origin git@github.com:USERNAME/REPO.git`
- Or configure GitHub CLI: `gh auth login`

