#!/usr/bin/env python3
"""
Add and commit all .py, .json, .sh, and .md files that are not already committed
"""
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return None

def main():
    dev_dir = Path("/home/himanshu/dev")
    
    print("Finding uncommitted .py, .json, .sh, and .md files...")
    print("=" * 60)
    
    # Get untracked files (not in git)
    untracked = run_cmd(
        "git ls-files --others --exclude-standard",
        cwd=dev_dir
    )
    
    # Get modified files (tracked but changed)
    modified = run_cmd(
        "git diff --name-only",
        cwd=dev_dir
    )
    
    # Get staged files
    staged = run_cmd(
        "git diff --cached --name-only",
        cwd=dev_dir
    )
    
    # Filter for our extensions
    extensions = ['.py', '.json', '.sh', '.md']
    files_to_add = []
    
    for output in [untracked, modified]:
        if output:
            for line in output.split('\n'):
                if line.strip():
                    file_path = Path(line.strip())
                    if file_path.suffix.lower() in extensions:
                        files_to_add.append(line.strip())
    
    # Also check staged files
    staged_files = []
    if staged:
        for line in staged.split('\n'):
            if line.strip():
                file_path = Path(line.strip())
                if file_path.suffix.lower() in extensions:
                    staged_files.append(line.strip())
    
    if not files_to_add and not staged_files:
        print("✅ No uncommitted .py, .json, .sh, or .md files found.")
        return
    
    print(f"\nFound {len(files_to_add)} files to add:")
    for f in files_to_add:
        print(f"  + {f}")
    
    if staged_files:
        print(f"\nFound {len(staged_files)} already staged files:")
        for f in staged_files:
            print(f"  * {f}")
    
    # Add files
    if files_to_add:
        print("\nAdding files to git...")
        for file_path in files_to_add:
            result = run_cmd(f"git add '{file_path}'", cwd=dev_dir)
            if result is None:
                print(f"  ⚠️  Failed to add: {file_path}")
            else:
                print(f"  ✅ Added: {file_path}")
    
    # Check status
    status = run_cmd("git status --short", cwd=dev_dir)
    if status:
        print("\n" + "=" * 60)
        print("Files ready to commit:")
        print(status)
        print("=" * 60)
        
        response = input("\nCommit these files? (y/n): ").strip().lower()
        if response == 'y':
            commit_msg = "Add/update .py, .json, .sh, and .md files"
            result = run_cmd(f"git commit -m '{commit_msg}'", cwd=dev_dir)
            if result is None:
                print("❌ Commit failed!")
                return
            print("✅ Files committed!")
            
            response = input("\nPush to remote? (y/n): ").strip().lower()
            if response == 'y':
                result = run_cmd("git push", cwd=dev_dir)
                if result is None:
                    print("❌ Push failed!")
                else:
                    print("✅ Pushed to remote!")
        else:
            print("Commit cancelled.")
    else:
        print("No changes to commit.")

if __name__ == "__main__":
    main()

