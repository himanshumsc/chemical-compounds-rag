#!/usr/bin/env python3
"""
Main Evaluation Script: Qwen RAG Concise vs Gemma RAG Concise
Orchestrates all evaluation metrics (BLEU, ROUGE, BERTScore) for comparing
Qwen and Gemma RAG Concise outputs against OpenAI baseline.
"""

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Scripts directory
SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent / "results"
CODE_DIR = Path(__file__).parent.parent.parent

# Virtual environment - try multiple possible locations
VENV_PYTHON = CODE_DIR / ".venv_phi4_req" / "bin" / "python3"
if not VENV_PYTHON.exists():
    # Try absolute path
    VENV_PYTHON = Path("/home/himanshu/dev/code/.venv_phi4_req/bin/python3")
    if not VENV_PYTHON.exists():
        # Fallback to system python
        logger.warning(f"Virtual environment not found, using system Python: {sys.executable}")
        VENV_PYTHON = sys.executable
    else:
        logger.info(f"Using virtual environment Python: {VENV_PYTHON}")
else:
    logger.info(f"Using virtual environment Python: {VENV_PYTHON}")

# Evaluation scripts
BLEU_SCRIPT = SCRIPTS_DIR / "bleu_score_calculator_qwen_gemma.py"
ROUGE_SCRIPT = SCRIPTS_DIR / "rouge_score_calculator_qwen_gemma.py"
BERTSCORE_SCRIPT = SCRIPTS_DIR / "bertscore_calculator_qwen_gemma.py"


def run_script(script_path: Path, script_name: str) -> bool:
    """
    Run an evaluation script and return success status.
    
    Args:
        script_path: Path to the script to run
        script_name: Name of the script for logging
    
    Returns:
        True if successful, False otherwise
    """
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    logger.info("="*70)
    logger.info(f"Running {script_name}...")
    logger.info("="*70)
    
    try:
        # Run the script using the virtual environment Python
        result = subprocess.run(
            [str(VENV_PYTHON), str(script_path)],
            check=False  # Don't raise exception on non-zero exit
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {script_name} completed successfully")
            return True
        else:
            logger.error(f"❌ {script_name} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running {script_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def check_results() -> dict:
    """
    Check which result files were generated.
    
    Returns:
        Dictionary with status of each metric's result files
    """
    results_status = {
        'bleu': {
            'csv': (RESULTS_DIR / "per_question_bleu_qwen_gemma.csv").exists(),
            'summary': (RESULTS_DIR / "summary_metrics_qwen_gemma.json").exists(),
            'comparison': (RESULTS_DIR / "comparison_bleu_qwen_gemma.json").exists(),
        },
        'rouge': {
            'csv': (RESULTS_DIR / "per_question_rouge_qwen_gemma.csv").exists(),
            'summary': (RESULTS_DIR / "summary_rouge_qwen_gemma.json").exists(),
            'comparison': (RESULTS_DIR / "comparison_rouge_qwen_gemma.json").exists(),
        },
        'bertscore': {
            'csv': (RESULTS_DIR / "per_question_bertscore_qwen_gemma.csv").exists(),
            'summary': (RESULTS_DIR / "summary_bertscore_qwen_gemma.json").exists(),
            'comparison': (RESULTS_DIR / "comparison_bertscore_qwen_gemma.json").exists(),
        }
    }
    
    return results_status


def print_summary(results_status: dict, execution_times: dict):
    """
    Print execution summary.
    
    Args:
        results_status: Status of result files
        execution_times: Execution times for each script
    """
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print("\nExecution Times:")
    for metric, time_taken in execution_times.items():
        if time_taken:
            print(f"  {metric.upper()}: {time_taken:.2f} seconds")
        else:
            print(f"  {metric.upper()}: Not executed")
    
    print("\nResult Files Generated:")
    for metric, files in results_status.items():
        print(f"\n  {metric.upper()}:")
        print(f"    CSV:        {'✅' if files['csv'] else '❌'}")
        print(f"    Summary:    {'✅' if files['summary'] else '❌'}")
        print(f"    Comparison: {'✅' if files['comparison'] else '❌'}")
    
    print("\n" + "="*70)
    print("All results saved to:", RESULTS_DIR)
    print("="*70)


def load_and_display_quick_summary():
    """
    Load summary files and display quick comparison metrics.
    """
    print("\n" + "="*70)
    print("QUICK METRICS SUMMARY")
    print("="*70)
    
    # BLEU Summary
    bleu_summary = RESULTS_DIR / "summary_metrics_qwen_gemma.json"
    if bleu_summary.exists():
        try:
            with open(bleu_summary, 'r') as f:
                bleu_data = json.load(f)
            
            qwen_bleu4 = bleu_data['qwen']['overall']['bleu_4']['mean']
            gemma_bleu4 = bleu_data['gemma']['overall']['bleu_4']['mean']
            
            print("\nBLEU-4 Scores:")
            print(f"  Qwen:  {qwen_bleu4:.4f}")
            print(f"  Gemma: {gemma_bleu4:.4f}")
            print(f"  Winner: {'Qwen' if qwen_bleu4 > gemma_bleu4 else 'Gemma'}")
        except Exception as e:
            logger.warning(f"Could not load BLEU summary: {e}")
    
    # ROUGE Summary
    rouge_summary = RESULTS_DIR / "summary_rouge_qwen_gemma.json"
    if rouge_summary.exists():
        try:
            with open(rouge_summary, 'r') as f:
                rouge_data = json.load(f)
            
            qwen_rouge1 = rouge_data['qwen']['overall']['rouge1']['fmeasure']['mean']
            gemma_rouge1 = rouge_data['gemma']['overall']['rouge1']['fmeasure']['mean']
            
            print("\nROUGE-1 F-measure:")
            print(f"  Qwen:  {qwen_rouge1:.4f}")
            print(f"  Gemma: {gemma_rouge1:.4f}")
            print(f"  Winner: {'Qwen' if qwen_rouge1 > gemma_rouge1 else 'Gemma'}")
        except Exception as e:
            logger.warning(f"Could not load ROUGE summary: {e}")
    
    # BERTScore Summary
    bertscore_summary = RESULTS_DIR / "summary_bertscore_qwen_gemma.json"
    if bertscore_summary.exists():
        try:
            with open(bertscore_summary, 'r') as f:
                bertscore_data = json.load(f)
            
            qwen_f1 = bertscore_data['qwen']['overall']['f1']['mean']
            gemma_f1 = bertscore_data['gemma']['overall']['f1']['mean']
            
            print("\nBERTScore F1:")
            print(f"  Qwen:  {qwen_f1:.4f}")
            print(f"  Gemma: {gemma_f1:.4f}")
            print(f"  Winner: {'Qwen' if qwen_f1 > gemma_f1 else 'Gemma'}")
        except Exception as e:
            logger.warning(f"Could not load BERTScore summary: {e}")
    
    print("\n" + "="*70)


def main():
    """
    Main execution function - runs all evaluation scripts.
    """
    import time
    
    logger.info("="*70)
    logger.info("Qwen vs Gemma RAG Concise - Complete Evaluation")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Using Python: {VENV_PYTHON}")
    logger.info("="*70)
    
    execution_times = {
        'bleu': None,
        'rouge': None,
        'bertscore': None
    }
    
    # Run BLEU evaluation
    start_time = time.time()
    bleu_success = run_script(BLEU_SCRIPT, "BLEU Score Calculator")
    execution_times['bleu'] = time.time() - start_time if bleu_success else None
    
    if not bleu_success:
        logger.warning("BLEU evaluation failed, continuing with other metrics...")
    
    # Run ROUGE evaluation
    start_time = time.time()
    rouge_success = run_script(ROUGE_SCRIPT, "ROUGE Score Calculator")
    execution_times['rouge'] = time.time() - start_time if rouge_success else None
    
    if not rouge_success:
        logger.warning("ROUGE evaluation failed, continuing with other metrics...")
    
    # Run BERTScore evaluation
    start_time = time.time()
    bertscore_success = run_script(BERTSCORE_SCRIPT, "BERTScore Calculator")
    execution_times['bertscore'] = time.time() - start_time if bertscore_success else None
    
    if not bertscore_success:
        logger.warning("BERTScore evaluation failed")
    
    # Check results
    results_status = check_results()
    
    # Print summary
    print_summary(results_status, execution_times)
    
    # Load and display quick metrics
    load_and_display_quick_summary()
    
    # Final status
    logger.info("="*70)
    logger.info(f"Evaluation complete at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_success = bleu_success and rouge_success and bertscore_success
    if all_success:
        logger.info("✅ All evaluations completed successfully!")
    else:
        logger.warning("⚠️  Some evaluations failed. Check logs above for details.")
    
    logger.info("="*70)
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())

