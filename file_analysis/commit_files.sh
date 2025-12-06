#!/bin/bash
# Add and commit all .py, .json, .sh, and .md files that are not already committed

cd /home/himanshu/dev

echo "Finding uncommitted .py, .json, .sh, and .md files..."

# Find all files with these extensions that are not ignored by git
git ls-files --others --exclude-standard | grep -E '\.(py|json|sh|md)$' > /tmp/new_files.txt
git diff --name-only | grep -E '\.(py|json|sh|md)$' > /tmp/modified_files.txt

# Combine and add all files
if [ -s /tmp/new_files.txt ] || [ -s /tmp/modified_files.txt ]; then
    echo "Adding new files..."
    cat /tmp/new_files.txt | xargs -r git add
    
    echo "Adding modified files..."
    cat /tmp/modified_files.txt | xargs -r git add
    
    echo ""
    echo "Files to be committed:"
    git status --short | grep -E '\.(py|json|sh|md)$'
    
    echo ""
    read -p "Commit these files? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git commit -m "Add/update .py, .json, .sh, and .md files"
        echo "✅ Files committed!"
        echo ""
        read -p "Push to remote? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push
            echo "✅ Pushed to remote!"
        fi
    else
        echo "Commit cancelled."
    fi
else
    echo "No uncommitted .py, .json, .sh, or .md files found."
fi

rm -f /tmp/new_files.txt /tmp/modified_files.txt

