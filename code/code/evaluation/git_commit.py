#!/usr/bin/env python3
"""
Script to add and commit evaluation files to git
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    # Change to dev directory
    dev_dir = Path("/home/himanshu/dev")
    
    print("="*70)
    print("Git Commit: Evaluation Files")
    print("="*70)
    
    # Check if git is initialized
    print("\n1. Checking git status...")
    success, stdout, stderr = run_command("git status", cwd=dev_dir)
    if not success:
        print("   ⚠️  Git not initialized or not a git repository")
        print("   Initializing git repository...")
        run_command("git init", cwd=dev_dir)
    
    # Add evaluation directory
    print("\n2. Adding evaluation files...")
    files_to_add = [
        "code/evaluation/",
        "code/compare_regenerated_output.py",
        "code/multimodal_qa_runner_vllm.py"
    ]
    
    for file_path in files_to_add:
        full_path = dev_dir / file_path
        if full_path.exists():
            success, stdout, stderr = run_command(f"git add {file_path}", cwd=dev_dir)
            if success:
                print(f"   ✅ Added: {file_path}")
            else:
                print(f"   ⚠️  Could not add: {file_path}")
                if stderr:
                    print(f"      Error: {stderr}")
        else:
            print(f"   ⚠️  File not found: {file_path}")
    
    # Check what will be committed
    print("\n3. Checking files to be committed...")
    success, stdout, stderr = run_command("git status --short", cwd=dev_dir)
    if stdout.strip():
        print("   Files to be committed:")
        for line in stdout.strip().split('\n'):
            if line.strip():
                print(f"      {line}")
    else:
        print("   No changes to commit")
        return
    
    # Commit
    print("\n4. Committing files...")
    commit_message = """Add evaluation metrics (BLEU and BERTScore) for regenerated QWEN and OpenAI answers

- Renamed bleu_evaluation to evaluation directory
- Added BLEU score calculator for regenerated data
- Added BERTScore calculator for regenerated data
- Added comprehensive analysis reports (BLEU and BERTScore)
- Results for 712 question-answer pairs (178 files × 4 questions)
- Comparison between regenerated QWEN (vLLM) and OpenAI (comprehensive_text) answers
- Added comparison script for regenerated output analysis
- Added vLLM-based multimodal QA runner script"""
    
    success, stdout, stderr = run_command(
        f'git commit -m "{commit_message}"',
        cwd=dev_dir
    )
    
    if success:
        print("   ✅ Commit successful!")
        print(f"   {stdout.strip()}")
    else:
        print("   ❌ Commit failed")
        if stderr:
            print(f"   Error: {stderr}")
        return
    
    # Show commit info
    print("\n5. Commit information:")
    success, stdout, stderr = run_command("git log -1 --stat", cwd=dev_dir)
    if success:
        print(stdout)
    
    print("\n" + "="*70)
    print("✅ Files committed successfully!")
    print("\nTo push to GitHub, run:")
    print("   cd /home/himanshu/dev")
    print("   git push -u origin main  # or 'master' if that's your branch")
    print("="*70)

if __name__ == "__main__":
    main()

