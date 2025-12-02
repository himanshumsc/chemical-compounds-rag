#!/bin/bash
# Check what's in git for evaluation

cd /home/himanshu/dev

echo "=========================================="
echo "Checking git status for evaluation files"
echo "=========================================="

echo ""
echo "1. Recent commits:"
git log --oneline -5

echo ""
echo "2. Files in evaluation directory (tracked by git):"
git ls-files | grep "code/evaluation/" | head -20

echo ""
echo "3. Current git status:"
git status --short | head -20

echo ""
echo "4. Remote status:"
git remote -v

echo ""
echo "=========================================="

