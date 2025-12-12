#!/bin/bash
# Script to commit evaluation files to GitHub

cd /home/himanshu/dev

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "Git repository initialized."
fi

# Check if remote exists
if ! git remote | grep -q origin; then
    echo "Please add your GitHub remote:"
    echo "  git remote add origin <your-github-repo-url>"
    echo ""
    echo "Or if you want to set it up now, run:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
    exit 1
fi

# Add evaluation directory
echo "Adding evaluation files..."
git add code/evaluation/

# Add related files
git add code/compare_regenerated_output.py
git add code/multimodal_qa_runner_vllm.py

# Check status
echo ""
echo "Files to be committed:"
git status --short

# Commit
echo ""
read -p "Enter commit message (or press Enter for default): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Add evaluation metrics (BLEU and BERTScore) for regenerated QWEN and OpenAI answers"
fi

git commit -m "$commit_msg"

# Push
echo ""
read -p "Push to GitHub? (y/n): " push_confirm
if [ "$push_confirm" = "y" ] || [ "$push_confirm" = "Y" ]; then
    echo "Pushing to GitHub..."
    git push -u origin main 2>/dev/null || git push -u origin master 2>/dev/null || echo "Please specify the branch name"
else
    echo "Commit created. Push manually with: git push"
fi

echo ""
echo "Done!"

