#!/usr/bin/env python3
"""
Master script to run all EDA analyses for Individual Compounds Dataset.
Executes all analysis scripts in sequence and generates a comprehensive summary.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Configuration
SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = Path("/home/himanshu/MSC_FINAL/dev/EDA/output")
REPORTS_DIR = Path("/home/himanshu/MSC_FINAL/dev/EDA/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Analysis scripts in order
ANALYSIS_SCRIPTS = [
    "01_basic_statistics.py",
    "02_content_analysis.py",
    "03_reference_analysis.py",
    "04_extract_structured_data.py",  # Structured data extraction with RDKit
    "05_cleaning_and_eda.py",  # Cleaning, univariate/bivariate analysis, clustering, ML
    "06_rag_pipeline_eda.py",  # RAG pipeline EDA - text lengths with Q1-Q4 limits
    "07_text_length_eda.py",  # Text length EDA - general analysis without limits
    # "08_advanced_rdkit_analysis.py",  # Future: Advanced RDKit analysis
    # "06_chemical_property_analysis.py",  # Future: Property relationships
    # "07_text_semantic_analysis.py",  # Future: Text analysis
    # "08_compound_classification.py",  # Future: Classification
    # "09_cross_reference_network.py",  # Future: Network analysis
    # "10_data_quality_assessment.py",  # Future: Quality assessment
    # "11_visualization_dashboard.py",  # Future: Visualizations
    # "12_comprehensive_report_generator.py",  # Future: Final report
]


def run_script(script_name: str) -> tuple[bool, str]:
    """Run a single analysis script and return success status and output."""
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        return False, f"Script not found: {script_path}"
    
    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=SCRIPTS_DIR
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)


def generate_summary_report(results: dict, output_path: Path):
    """Generate a comprehensive summary report of all analyses."""
    
    report = []
    report.append("# Comprehensive EDA Summary Report\n\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n\n")
    
    # Execution Summary
    report.append("## Execution Summary\n\n")
    total_scripts = len(results)
    successful = sum(1 for r in results.values() if r["success"])
    failed = total_scripts - successful
    
    report.append(f"- **Total Scripts:** {total_scripts}\n")
    report.append(f"- **Successful:** {successful} ✅\n")
    report.append(f"- **Failed:** {failed} ❌\n")
    report.append("\n")
    
    # Script Results
    report.append("## Script Execution Results\n\n")
    report.append("| Script | Status | Notes |\n")
    report.append("|--------|--------|-------|\n")
    
    for script_name, result in results.items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        notes = result.get("notes", "")
        if len(notes) > 50:
            notes = notes[:47] + "..."
        report.append(f"| {script_name} | {status} | {notes} |\n")
    
    report.append("\n")
    
    # Output Files
    report.append("## Generated Output Files\n\n")
    report.append("### JSON Data Files\n")
    json_files = sorted(OUTPUT_DIR.glob("*.json"))
    for json_file in json_files:
        report.append(f"- `{json_file.name}`\n")
    
    report.append("\n### Markdown Reports\n")
    md_files = sorted(OUTPUT_DIR.glob("*.md"))
    for md_file in md_files:
        report.append(f"- `{md_file.name}`\n")
    
    report.append("\n")
    
    # Key Findings (if available)
    report.append("## Key Findings\n\n")
    
    # Try to load and summarize key statistics
    try:
        basic_stats_path = OUTPUT_DIR / "01_basic_statistics.json"
        if basic_stats_path.exists():
            with open(basic_stats_path, 'r') as f:
                basic_stats = json.load(f)
                report.append("### Dataset Overview\n\n")
                report.append(f"- **Total Compounds:** {basic_stats.get('total_compounds', 'N/A')}\n")
                if "compound_id" in basic_stats:
                    cid = basic_stats["compound_id"]
                    report.append(f"- **Compound ID Range:** {cid.get('min')} - {cid.get('max')}\n")
                report.append("\n")
    except Exception as e:
        report.append(f"*Could not load basic statistics: {e}*\n\n")
    
    try:
        content_path = OUTPUT_DIR / "02_content_analysis.json"
        if content_path.exists():
            with open(content_path, 'r') as f:
                content = json.load(f)
                report.append("### Content Analysis\n\n")
                if "chemical_information" in content:
                    chem = content["chemical_information"]
                    report.append(f"- **Unique Formulas:** {chem.get('unique_formulas', 'N/A')}\n")
                    report.append(f"- **Unique Elements:** {len(chem.get('element_frequency', {}))}\n")
                report.append("\n")
    except Exception as e:
        pass
    
    try:
        ref_path = OUTPUT_DIR / "03_reference_analysis.json"
        if ref_path.exists():
            with open(ref_path, 'r') as f:
                refs = json.load(f)
                report.append("### Reference Analysis\n\n")
                if "reference_counts" in refs:
                    rc = refs["reference_counts"]
                    report.append(f"- **Compounds with References:** {rc.get('compounds_with_refs', 'N/A')}\n")
                    if "statistics" in rc:
                        report.append(f"- **Average References per Compound:** {rc['statistics'].get('mean', 0):.2f}\n")
                report.append("\n")
    except Exception as e:
        pass
    
    # Next Steps
    report.append("## Next Steps\n\n")
    report.append("1. Review individual analysis reports in the `output/` directory\n")
    report.append("2. Examine generated visualizations (when implemented)\n")
    report.append("3. Review data quality assessment (when implemented)\n")
    report.append("4. Use findings to inform downstream processing\n")
    report.append("\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✅ Summary report saved to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Comprehensive EDA - Individual Compounds Dataset")
    print("=" * 60)
    print(f"\nStarting analysis at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # Run each script
    for script_name in ANALYSIS_SCRIPTS:
        success, output = run_script(script_name)
        
        results[script_name] = {
            "success": success,
            "output": output,
            "notes": "Completed successfully" if success else f"Error: {output[:100]}"
        }
        
        if success:
            print(f"✅ {script_name} completed successfully")
        else:
            print(f"❌ {script_name} failed: {output[:200]}")
    
    # Generate summary report
    print(f"\n{'=' * 60}")
    print("Generating Summary Report")
    print(f"{'=' * 60}\n")
    
    summary_path = REPORTS_DIR / "eda_comprehensive_summary.md"
    generate_summary_report(results, summary_path)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    successful = sum(1 for r in results.values() if r["success"])
    print(f"Successfully executed: {successful}/{len(results)} scripts")
    print(f"\nSummary report: {summary_path}")
    print(f"All outputs: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

