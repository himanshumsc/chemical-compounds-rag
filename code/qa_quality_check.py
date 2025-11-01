#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Dict, Any, List

SRC_QA_DIR = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components")
OUT_DIR = Path("/home/himanshu/dev/output/qwen")


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def load_src_map() -> Dict[str, Dict[str, Any]]:
    src_map = {}
    for p in SRC_QA_DIR.glob("*_*.json"):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
            src_map[p.name] = d
        except Exception:
            continue
    return src_map


def check_answer_against_text(answer: str, text: str) -> Dict[str, Any]:
    ans = normalize(answer)
    txt = normalize(text)
    # Coverage-like metric: proportion of unique words in answer that also appear in text
    ans_words = [w for w in ans.split() if len(w) > 2]
    txt_words = set(txt.split())
    if not ans_words:
        return {"coverage": 0.0, "oov_terms": [], "len": 0}
    in_text = [w for w in ans_words if w in txt_words]
    oov = sorted(set(w for w in ans_words if w not in txt_words))
    coverage = len(in_text) / max(1, len(set(ans_words)))
    return {"coverage": round(coverage, 3), "oov_terms": oov[:20], "len": len(ans_words)}


def check_q1_identity(answer: str, compound_name: str) -> bool:
    # Accept match if the normalized compound name appears in answer
    return normalize(compound_name) in normalize(answer)


def run_qc():
    src_map = load_src_map()
    issues: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []

    for out_file in sorted(OUT_DIR.glob("*__answers.json")):
        try:
            out = json.loads(out_file.read_text(encoding='utf-8'))
        except Exception as e:
            issues.append({"file": out_file.name, "error": f"invalid json: {e}"})
            continue

        src_name = out.get("source_file")
        src = src_map.get(src_name)
        if not src:
            issues.append({"file": out_file.name, "error": f"missing src {src_name}"})
            continue

        main_text = src.get("main_entry_content", "") or "\n".join([src.get("compound_name", ""), ""])
        compound_name = src.get("compound_name", "")
        answers = out.get("answers", [])
        if not answers:
            issues.append({"file": out_file.name, "error": "no answers"})
            continue

        # Q1 identification check
        q1_ok = check_q1_identity(answers[0].get("answer", ""), compound_name)
        cov = [check_answer_against_text(a.get("answer", ""), main_text) for a in answers]

        report.append({
            "file": out_file.name,
            "compound": compound_name,
            "q1_identifies_compound": q1_ok,
            "coverages": [c["coverage"] for c in cov],
            "avg_coverage": round(sum(c["coverage"] for c in cov) / max(1, len(cov)), 3),
            "q1_latency_s": answers[0].get("latency_s", None),
        })

        if not q1_ok:
            issues.append({"file": out_file.name, "type": "q1_identity_fail", "compound": compound_name})
        # Flag low coverage
        if any(c["coverage"] < 0.2 for c in cov):
            issues.append({"file": out_file.name, "type": "low_coverage", "coverages": [c["coverage"] for c in cov]})

    summary = {
        "checked": len(report),
        "issues": len(issues),
        "q1_identity_failures": sum(1 for i in issues if i.get("type") == "q1_identity_fail"),
        "low_coverage_cases": sum(1 for i in issues if i.get("type") == "low_coverage"),
        "details": report[:50],
        "issues_samples": issues[:50],
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_qc()


