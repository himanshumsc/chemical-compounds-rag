#!/bin/bash
# Script to remove old bleu_evaluation directory from git tracking

cd /home/himanshu/dev

echo "=========================================="
echo "Removing old bleu_evaluation from git"
echo "=========================================="

# Check if bleu_evaluation exists in git
echo ""
echo "1. Checking what git sees..."
git ls-files | grep "bleu_evaluation" | head -10

echo ""
echo "2. Removing bleu_evaluation from git tracking..."
# Remove from git index but keep local files (if they exist)
git rm -r --cached code/bleu_evaluation/ 2>/dev/null

# If the directory doesn't exist locally, just remove from git
if [ ! -d "code/bleu_evaluation" ]; then
    echo "   Directory doesn't exist locally, removing from git index only"
    git rm -r --cached code/bleu_evaluation/ 2>/dev/null || echo "   Already removed from index"
else
    echo "   Directory exists locally, removing from git tracking only"
    git rm -r --cached code/bleu_evaluation/ 2>/dev/null || echo "   Already removed from index"
fi

echo ""
echo "3. Checking status..."
git status --short | grep -E "(bleu_evaluation|evaluation)" | head -20

echo ""
echo "4. Staging the removal..."
git add -A

echo ""
echo "=========================================="
echo "Ready to commit the removal"
echo "=========================================="
echo ""
echo "To commit this change, run:"
echo "  git commit -m 'Remove old bleu_evaluation directory from git tracking'"
echo ""
echo "Then push:"
echo "  git push"
echo ""

