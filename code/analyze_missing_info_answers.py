#!/usr/bin/env python3
"""
Analyze missing information answers to determine if:
1. Answer EXISTS in chunks → Model failed to use context (MODEL_FAILURE)
2. Answer NOT in chunks → Wrong chunks retrieved (RETRIEVAL_FAILURE)
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemical formula pattern: Elements followed by numbers (e.g., C13H18O2, CH3COOH)
FORMULA_PATTERN = re.compile(r'\b([A-Z][a-z]?\d*)+([A-Z][a-z]?\d*)*\b')
# Molecular weight pattern: numbers followed by "g/mol" or just numbers
MOLECULAR_WEIGHT_PATTERN = re.compile(r'(\d+\.?\d*)\s*g/mol|MOLECULAR\s+WEIGHT[:\s]+(\d+\.?\d*)', re.IGNORECASE)

# Question type patterns
Q2_PATTERNS = [
    r'chemical formula',
    r'molecular weight',
    r'elements',
    r'formula.*weight',
    r'weight.*formula'
]
Q3_PATTERNS = [
    r'why.*developed',
    r'developed.*why',
    r'developed as',
    r'history',
    r'created',
    r'discovered'
]
Q4_PATTERNS = [
    r'properties',
    r'characteristics',
    r'melting point',
    r'boiling point',
    r'solubility',
    r'state',
    r'physical',
    r'chemical'
]

def classify_question_type(question: str) -> str:
    """Classify question as Q2, Q3, or Q4 based on keywords."""
    question_lower = question.lower()
    
    if any(re.search(pattern, question_lower) for pattern in Q2_PATTERNS):
        return "Q2"  # Formula and elements
    elif any(re.search(pattern, question_lower) for pattern in Q3_PATTERNS):
        return "Q3"  # Development/history
    elif any(re.search(pattern, question_lower) for pattern in Q4_PATTERNS):
        return "Q4"  # Properties
    else:
        # Default based on question index if available
        return "UNKNOWN"

def extract_compound_name(question: str) -> List[str]:
    """
    Extract compound name(s) from question.
    Returns list of possible compound name variations.
    """
    # Common patterns:
    # "What is the chemical formula of X?"
    # "Why was X developed..."
    # "What are the properties of X?"
    
    # Try to extract compound name after common question starters
    patterns = [
        r'(?:of|about|for)\s+([A-Z][^?.,!]+?)(?:\?|,|\.|$|also known)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:also known|developed|properties)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:acid|oxide|sulfate|chloride|hydroxide)',
        r'(\d+[-\(\)\w\s]+acid)',
        r'(\d+[-\(\)\w\s]+oxide)',
    ]
    
    compound_names = []
    for pattern in patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        for match in matches:
            name = match.strip()
            if len(name) > 3:  # Filter out very short matches
                compound_names.append(name)
    
    # Also try to extract from parentheses (common names)
    paren_matches = re.findall(r'\(([^)]+)\)', question)
    for match in paren_matches:
        if len(match) > 3:
            compound_names.append(match.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in compound_names:
        name_lower = name.lower()
        if name_lower not in seen:
            seen.add(name_lower)
            unique_names.append(name)
    
    return unique_names if unique_names else ["unknown"]

def search_compound_in_context(compound_names: List[str], context: str) -> Tuple[bool, int, List[str]]:
    """
    Search for compound name(s) in context.
    Returns: (found, mention_count, matched_names)
    """
    context_lower = context.lower()
    found = False
    mention_count = 0
    matched_names = []
    
    for name in compound_names:
        name_lower = name.lower()
        # Count mentions
        count = context_lower.count(name_lower)
        if count > 0:
            found = True
            mention_count += count
            matched_names.append(name)
    
    return found, mention_count, matched_names

def search_formula_in_context(context: str) -> Tuple[bool, List[str]]:
    """
    Search for chemical formulas in context.
    Returns: (found, formulas_found)
    """
    # Look for "FORMULA:" label
    formula_label_pattern = r'FORMULA[:\s]+([A-Z][a-z]?\d*[A-Za-z0-9]*)'
    formulas = re.findall(formula_label_pattern, context, re.IGNORECASE)
    
    # Also search for formula patterns in text
    formula_matches = FORMULA_PATTERN.findall(context)
    for match in formula_matches:
        formula = ''.join(match)
        if len(formula) > 2 and formula not in formulas:  # Filter out very short matches
            formulas.append(formula)
    
    return len(formulas) > 0, formulas

def search_molecular_weight_in_context(context: str) -> Tuple[bool, List[str]]:
    """
    Search for molecular weight in context.
    Returns: (found, weights_found)
    """
    weights = []
    
    # Search for "MOLECULAR WEIGHT:" pattern
    matches = MOLECULAR_WEIGHT_PATTERN.findall(context)
    for match in matches:
        weight = match[0] if match[0] else match[1]
        if weight:
            weights.append(weight)
    
    return len(weights) > 0, weights

def search_elements_in_context(context: str) -> Tuple[bool, List[str]]:
    """
    Search for element lists in context.
    Returns: (found, elements_found)
    """
    # Look for "ELEMENTS:" label
    elements_label_pattern = r'ELEMENTS[:\s]+([^\n]+)'
    matches = re.findall(elements_label_pattern, context, re.IGNORECASE)
    
    elements = []
    for match in matches:
        # Split by comma and clean
        element_list = [e.strip() for e in match.split(',')]
        elements.extend(element_list)
    
    return len(elements) > 0, elements

def search_development_info_in_context(compound_names: List[str], context: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Search for development/history information.
    Returns: (found, evidence_dict)
    """
    context_lower = context.lower()
    evidence = {
        "has_developed": False,
        "has_company": False,
        "has_year": False,
        "has_purpose": False,
        "keywords_found": []
    }
    
    # Search for development keywords
    dev_keywords = ['developed', 'discovered', 'created', 'synthesized', 'invented']
    for keyword in dev_keywords:
        if keyword in context_lower:
            evidence["has_developed"] = True
            evidence["keywords_found"].append(keyword)
    
    # Search for company/researcher names (common patterns)
    company_pattern = r'([A-Z][a-z]+\s+(?:Company|Corporation|Corp|Inc|Ltd|Researchers))'
    if re.search(company_pattern, context):
        evidence["has_company"] = True
    
    # Search for years (1900-2100)
    year_pattern = r'\b(19|20)\d{2}\b'
    if re.search(year_pattern, context):
        evidence["has_year"] = True
    
    # Search for purpose keywords
    purpose_keywords = ['because', 'to', 'for', 'purpose', 'reason', 'goal']
    for keyword in purpose_keywords:
        if keyword in context_lower:
            evidence["has_purpose"] = True
            evidence["keywords_found"].append(keyword)
            break
    
    found = evidence["has_developed"] or (evidence["has_company"] and evidence["has_year"])
    return found, evidence

def search_properties_in_context(context: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Search for property information.
    Returns: (found, evidence_dict)
    """
    context_lower = context.lower()
    evidence = {
        "has_melting_point": False,
        "has_boiling_point": False,
        "has_solubility": False,
        "has_molecular_weight": False,
        "has_state": False,
        "properties_found": []
    }
    
    # Search for property labels
    if re.search(r'MELTING\s+POINT', context, re.IGNORECASE):
        evidence["has_melting_point"] = True
        evidence["properties_found"].append("melting_point")
    
    if re.search(r'BOILING\s+POINT', context, re.IGNORECASE):
        evidence["has_boiling_point"] = True
        evidence["properties_found"].append("boiling_point")
    
    if re.search(r'SOLUBILITY', context, re.IGNORECASE):
        evidence["has_solubility"] = True
        evidence["properties_found"].append("solubility")
    
    if re.search(r'MOLECULAR\s+WEIGHT', context, re.IGNORECASE):
        evidence["has_molecular_weight"] = True
        evidence["properties_found"].append("molecular_weight")
    
    if re.search(r'STATE[:\s]+(Solid|Liquid|Gas)', context, re.IGNORECASE):
        evidence["has_state"] = True
        evidence["properties_found"].append("state")
    
    found = len(evidence["properties_found"]) > 0
    return found, evidence

def analyze_answer(
    answer: Dict[str, Any],
    question_idx: int
) -> Dict[str, Any]:
    """
    Analyze a single answer to determine if info exists in context.
    Returns analysis result dictionary.
    """
    question = answer.get('question', '')
    context = answer.get('rag_context_formatted', '')
    chunks = answer.get('rag_chunks', [])
    
    # Skip if no context
    if not context and not chunks:
        return {
            "question_idx": question_idx,
            "question": question,
            "classification": "NO_CONTEXT",
            "confidence": "low",
            "error": "No context available"
        }
    
    # Classify question type
    question_type = classify_question_type(question)
    
    # Extract compound name
    compound_names = extract_compound_name(question)
    
    # Check if compound exists in context
    compound_found, mention_count, matched_names = search_compound_in_context(compound_names, context)
    
    analysis = {
        "question_idx": question_idx,
        "question": question,
        "question_type": question_type,
        "compound_names": compound_names,
        "compound_found_in_context": compound_found,
        "compound_mention_count": mention_count,
        "matched_compound_names": matched_names,
        "classification": None,
        "confidence": "medium",
        "evidence": {}
    }
    
    # If compound not found at all → Retrieval failure
    if not compound_found:
        analysis["classification"] = "RETRIEVAL_FAILURE"
        analysis["confidence"] = "high"
        analysis["evidence"]["reason"] = "Compound name not found in any retrieved chunks"
        return analysis
    
    # Based on question type, search for expected information
    if question_type == "Q2":
        # Q2: Formula and elements
        formula_found, formulas = search_formula_in_context(context)
        weight_found, weights = search_molecular_weight_in_context(context)
        elements_found, elements = search_elements_in_context(context)
        
        analysis["evidence"]["formula_found"] = formula_found
        analysis["evidence"]["formulas"] = formulas
        analysis["evidence"]["molecular_weight_found"] = weight_found
        analysis["evidence"]["weights"] = weights
        analysis["evidence"]["elements_found"] = elements_found
        analysis["evidence"]["elements"] = elements
        
        # If formula or weight found → Model failure (info exists)
        if formula_found or weight_found:
            analysis["classification"] = "MODEL_FAILURE"
            analysis["confidence"] = "high"
            analysis["evidence"]["reason"] = "Formula or molecular weight found in context"
        else:
            analysis["classification"] = "RETRIEVAL_FAILURE"
            analysis["confidence"] = "medium"
            analysis["evidence"]["reason"] = "Compound found but formula/weight not in context"
    
    elif question_type == "Q3":
        # Q3: Development/history
        dev_found, dev_evidence = search_development_info_in_context(compound_names, context)
        
        analysis["evidence"]["development_info_found"] = dev_found
        analysis["evidence"]["development_evidence"] = dev_evidence
        
        if dev_found:
            analysis["classification"] = "MODEL_FAILURE"
            analysis["confidence"] = "high"
            analysis["evidence"]["reason"] = "Development/history information found in context"
        else:
            analysis["classification"] = "RETRIEVAL_FAILURE"
            analysis["confidence"] = "medium"
            analysis["evidence"]["reason"] = "Compound found but development info not in context"
    
    elif question_type == "Q4":
        # Q4: Properties
        props_found, props_evidence = search_properties_in_context(context)
        
        analysis["evidence"]["properties_found"] = props_found
        analysis["evidence"]["properties_evidence"] = props_evidence
        
        if props_found:
            analysis["classification"] = "MODEL_FAILURE"
            analysis["confidence"] = "high"
            analysis["evidence"]["reason"] = "Property information found in context"
        else:
            analysis["classification"] = "RETRIEVAL_FAILURE"
            analysis["confidence"] = "medium"
            analysis["evidence"]["reason"] = "Compound found but properties not in context"
    
    else:
        # Unknown question type - use general check
        analysis["classification"] = "UNKNOWN_TYPE"
        analysis["confidence"] = "low"
        analysis["evidence"]["reason"] = "Question type could not be classified"
    
    return analysis

def process_json_file(json_path: Path) -> List[Dict[str, Any]]:
    """Process a single JSON file and return list of analyses."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        answers = data.get('answers', [])
        analyses = []
        
        for idx, answer in enumerate(answers):
            question_idx = idx + 1
            
            # Only analyze filtered answers
            if answer.get('filtered_as_missing_info', False):
                analysis = analyze_answer(answer, question_idx)
                analysis["file"] = json_path.name
                analyses.append(analysis)
        
        return analyses
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def generate_summary(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from analyses."""
    total = len(analyses)
    if total == 0:
        return {}
    
    classifications = defaultdict(int)
    confidences = defaultdict(int)
    question_types = defaultdict(int)
    by_question_type = defaultdict(lambda: {"MODEL_FAILURE": 0, "RETRIEVAL_FAILURE": 0, "NO_CONTEXT": 0, "UNKNOWN_TYPE": 0})
    
    for analysis in analyses:
        classification = analysis.get("classification", "UNKNOWN")
        confidence = analysis.get("confidence", "unknown")
        q_type = analysis.get("question_type", "UNKNOWN")
        
        classifications[classification] += 1
        confidences[confidence] += 1
        question_types[q_type] += 1
        by_question_type[q_type][classification] += 1
    
    return {
        "total_analyzed": total,
        "classifications": dict(classifications),
        "confidence_distribution": dict(confidences),
        "question_type_distribution": dict(question_types),
        "by_question_type": {k: dict(v) for k, v in by_question_type.items()},
        "model_failure_rate": classifications.get("MODEL_FAILURE", 0) / total * 100,
        "retrieval_failure_rate": classifications.get("RETRIEVAL_FAILURE", 0) / total * 100
    }

def generate_markdown_report(summary: Dict[str, Any], analyses: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate a markdown report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Missing Information Answers Analysis Report\n\n")
        f.write(f"**Generated:** {Path(__file__).stat().st_mtime}\n\n")
        
        # Summary
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Answers Analyzed:** {summary.get('total_analyzed', 0)}\n")
        f.write(f"- **Model Failures (Info Exists):** {summary.get('classifications', {}).get('MODEL_FAILURE', 0)} ({summary.get('model_failure_rate', 0):.1f}%)\n")
        f.write(f"- **Retrieval Failures (Info Missing):** {summary.get('classifications', {}).get('RETRIEVAL_FAILURE', 0)} ({summary.get('retrieval_failure_rate', 0):.1f}%)\n\n")
        
        # Breakdown by question type
        f.write("## Breakdown by Question Type\n\n")
        for q_type, counts in summary.get('by_question_type', {}).items():
            f.write(f"### {q_type}\n\n")
            f.write(f"- Model Failures: {counts.get('MODEL_FAILURE', 0)}\n")
            f.write(f"- Retrieval Failures: {counts.get('RETRIEVAL_FAILURE', 0)}\n\n")
        
        # Sample analyses
        f.write("## Sample Analyses\n\n")
        f.write("### Model Failures (First 10)\n\n")
        model_failures = [a for a in analyses if a.get('classification') == 'MODEL_FAILURE'][:10]
        for analysis in model_failures:
            f.write(f"**File:** {analysis.get('file')}, **Q{analysis.get('question_idx')}**\n")
            f.write(f"- Question: {analysis.get('question', '')[:100]}...\n")
            f.write(f"- Compound: {', '.join(analysis.get('compound_names', []))}\n")
            f.write(f"- Evidence: {analysis.get('evidence', {}).get('reason', 'N/A')}\n\n")
        
        f.write("### Retrieval Failures (First 10)\n\n")
        retrieval_failures = [a for a in analyses if a.get('classification') == 'RETRIEVAL_FAILURE'][:10]
        for analysis in retrieval_failures:
            f.write(f"**File:** {analysis.get('file')}, **Q{analysis.get('question_idx')}**\n")
            f.write(f"- Question: {analysis.get('question', '')[:100]}...\n")
            f.write(f"- Compound: {', '.join(analysis.get('compound_names', []))}\n")
            f.write(f"- Evidence: {analysis.get('evidence', {}).get('reason', 'N/A')}\n\n")

def main():
    input_dir = Path("/home/himanshu/dev/output/gemma3_rag_concise_missing_ans")
    output_dir = Path("/home/himanshu/dev/output/missing_info_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    json_files = sorted(input_dir.glob("*.json"))
    
    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        sys.exit(0)
    
    logger.info(f"Processing {len(json_files)} JSON files...")
    
    all_analyses = []
    for json_path in json_files:
        logger.info(f"Processing: {json_path.name}")
        analyses = process_json_file(json_path)
        all_analyses.extend(analyses)
        logger.info(f"  Found {len(analyses)} filtered answers")
    
    logger.info(f"\nTotal analyses: {len(all_analyses)}")
    
    # Generate summary
    summary = generate_summary(all_analyses)
    logger.info(f"\nSummary:")
    logger.info(f"  Model Failures: {summary.get('classifications', {}).get('MODEL_FAILURE', 0)}")
    logger.info(f"  Retrieval Failures: {summary.get('classifications', {}).get('RETRIEVAL_FAILURE', 0)}")
    
    # Save results
    results_path = output_dir / "analysis_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "analyses": all_analyses
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to: {results_path}")
    
    # Generate markdown report
    report_path = output_dir / "analysis_report.md"
    generate_markdown_report(summary, all_analyses, report_path)
    logger.info(f"Saved report to: {report_path}")

if __name__ == "__main__":
    main()

