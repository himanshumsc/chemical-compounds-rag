#!/bin/bash
# Script to commit evaluation files

cd /home/himanshu/dev

echo "=========================================="
echo "Adding evaluation files to git..."
echo "=========================================="

# Add evaluation directory and related files
git add code/evaluation/
git add code/compare_regenerated_output.py
git add code/multimodal_qa_runner_vllm.py

echo ""
echo "Files added. Checking status..."
git status --short

echo ""
echo "Committing files..."
git commit -m "Add evaluation metrics (BLEU and BERTScore) for regenerated QWEN and OpenAI answers

- Renamed bleu_evaluation to evaluation directory
- Added BLEU score calculator for regenerated data
- Added BERTScore calculator for regenerated data
- Added comprehensive analysis reports (BLEU and BERTScore)
- Results for 712 question-answer pairs (178 files × 4 questions)
- Comparison between regenerated QWEN (vLLM) and OpenAI (comprehensive_text) answers
- Added comparison script for regenerated output analysis
- Added vLLM-based multimodal QA runner script"

echo ""
echo "=========================================="
echo "Commit complete!"
echo "=========================================="
echo ""
echo "To push to GitHub, run:"
echo "  git push -u origin main"
echo ""

