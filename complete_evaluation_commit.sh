#!/bin/bash
# Complete script to handle rename and commit all evaluation files

cd /home/himanshu/dev

echo "=========================================="
echo "Step 1: Handling directory rename"
echo "=========================================="

# Remove old bleu_evaluation from git if it exists
if git ls-files | grep -q "code/bleu_evaluation"; then
    echo "Removing old bleu_evaluation from git tracking..."
    git rm -r --cached code/bleu_evaluation/ 2>/dev/null
fi

echo ""
echo "=========================================="
echo "Step 2: Adding all evaluation files"
echo "=========================================="

# Add the new evaluation directory
git add code/evaluation/

# Add related files
git add code/compare_regenerated_output.py
git add code/multimodal_qa_runner_vllm.py

# Stage all changes (this will help git detect the rename)
git add -A

echo ""
echo "=========================================="
echo "Step 3: Checking what will be committed"
echo "=========================================="
git status --short

echo ""
echo "=========================================="
echo "Step 4: Committing changes"
echo "=========================================="

git commit -m "Rename bleu_evaluation to evaluation and add evaluation metrics

- Renamed bleu_evaluation directory to evaluation
- Added BLEU score calculator for regenerated data
- Added BERTScore calculator for regenerated data  
- Added comprehensive analysis reports (BLEU and BERTScore)
- Results for 712 question-answer pairs (178 files × 4 questions)
- Comparison between regenerated QWEN (vLLM) and OpenAI (comprehensive_text) answers
- Added comparison script for regenerated output analysis
- Added vLLM-based multimodal QA runner script"

echo ""
echo "=========================================="
echo "✅ Commit complete!"
echo "=========================================="
echo ""
echo "To push to GitHub, run:"
echo "  git push -u origin main"
echo ""

