#!/usr/bin/env python3
"""
Automatically add and commit all .py, .json, .sh, and .md files
"""
import subprocess
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

def main():
    dev_dir = Path("/home/himanshu/dev")
    
    print("Finding uncommitted .py, .json, .sh, and .md files...")
    
    # Get untracked files
    untracked = run_cmd("git ls-files --others --exclude-standard", cwd=dev_dir)
    
    # Get modified files
    modified = run_cmd("git diff --name-only", cwd=dev_dir)
    
    extensions = ['.py', '.json', '.sh', '.md']
    files_to_add = []
    
    for output in [untracked, modified]:
        if output:
            for line in output.split('\n'):
                if line.strip():
                    file_path = Path(line.strip())
                    if file_path.suffix.lower() in extensions:
                        files_to_add.append(line.strip())
    
    if not files_to_add:
        print("✅ No uncommitted files found.")
        return
    
    print(f"\nFound {len(files_to_add)} files to add:")
    for f in files_to_add[:10]:  # Show first 10
        print(f"  + {f}")
    if len(files_to_add) > 10:
        print(f"  ... and {len(files_to_add) - 10} more")
    
    # Add all files
    print("\nAdding files to git...")
    for file_path in files_to_add:
        run_cmd(f"git add '{file_path}'", cwd=dev_dir)
    
    # Commit
    print("\nCommitting...")
    commit_msg = "Add/update .py, .json, .sh, and .md files"
    result = run_cmd(f"git commit -m '{commit_msg}'", cwd=dev_dir)
    
    if result:
        print("✅ Files committed!")
        print("\nTo push, run: cd /home/himanshu/dev && git push")
    else:
        print("⚠️  No changes to commit or commit failed")

if __name__ == "__main__":
    main()

