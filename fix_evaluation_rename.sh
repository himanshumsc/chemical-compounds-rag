#!/bin/bash
# Script to properly handle the rename from bleu_evaluation to evaluation in git

cd /home/himanshu/dev

echo "=========================================="
echo "Fixing git rename: bleu_evaluation -> evaluation"
echo "=========================================="

# Check if bleu_evaluation still exists in git tracking
if git ls-files | grep -q "bleu_evaluation"; then
    echo ""
    echo "Old bleu_evaluation files still tracked in git:"
    git ls-files | grep "bleu_evaluation" | head -5
    
    echo ""
    echo "Removing old bleu_evaluation from git tracking..."
    git rm -r --cached code/bleu_evaluation/ 2>/dev/null || echo "  (bleu_evaluation not in git index)"
    
    echo ""
    echo "Adding new evaluation directory..."
    git add code/evaluation/
    
    echo ""
    echo "Checking status..."
    git status --short | head -20
    
    echo ""
    echo "=========================================="
    echo "Staging the rename..."
    echo "=========================================="
    
    # Stage all changes
    git add -A
    
    echo ""
    echo "Files staged. Ready to commit."
    echo ""
    echo "To commit the rename, run:"
    echo "  git commit -m 'Rename bleu_evaluation to evaluation directory'"
    echo ""
    
else
    echo ""
    echo "✅ No old bleu_evaluation found in git tracking"
    echo "Adding evaluation directory..."
    git add code/evaluation/
    git status --short | head -20
fi

