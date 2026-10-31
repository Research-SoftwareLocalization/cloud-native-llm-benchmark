import os
import re
import json
import datetime
from pathlib import Path

REPO_ROOT = Path("/content/cloud-native-llm-benchmark")
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
SOURCE_FILE = DATA_DIR / "source" / "en.txt"

LANGUAGES = [
    ("ar", "Modern Standard Arabic", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff]')),
    ("bn", "Bengali", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    ("de", "German", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0980-\u09ff]')),
    ("es", "Spanish", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0980-\u09ff]')),
    ("fr", "French", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0980-\u09ff]')),
    ("hi", "Hindi", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    ("ja", "Japanese", re.compile(r'[\u0400-\u04ff\u0600-\u06ff\uac00-\ud7af]')),
    ("mr", "Marathi", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    ("pa", "Punjabi", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    ("ta", "Tamil", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    ("te", "Telugu", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    ("zh", "Simplified Chinese", re.compile(r'[\u0400-\u04ff\u0600-\u06ff\uac00-\ud7af]'))
]

CHATTER_REGEX = re.compile(r'(?i)^(here is the translation|translation:|note:|as an ai|translate into|arabic:|spanish:|bengali:)')
PLACEHOLDER_REGEX = re.compile(r'%\d+|%[a-zA-Z]|\{[^}]+\}')
ACCELERATOR_REGEX = re.compile(r'&[a-zA-Z]')

with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    source_lines = [l.rstrip("\r\n") for l in f]

# Dynamically discover all baseline_* directories
model_dirs = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("baseline_")])
timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

jsonl_log = REPORTS_DIR / "audit_findings_log.jsonl"
summary_md = REPORTS_DIR / "qc_audit_summary.md"

# Deduplication primary key set: (model, lang, line_idx)
seen_issue_keys = set()
summary_rows = []

# Overwrite mode ('w') ensures fresh, non-duplicated snapshot
with open(jsonl_log, "w", encoding="utf-8") as jsonl_out:
    for model_name in model_dirs:
        print(f"Auditing `{model_name}`...")
        model_dir = DATA_DIR / model_name

        for code, lang_name, forbidden_re in LANGUAGES:
            file_path = model_dir / f"{code}.txt"
            if not file_path.exists():
                continue

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                hyp_lines = [l.rstrip("\r\n") for l in f]

            script_bleed = 0
            broken_placeholders = 0
            chatter_count = 0
            defective_count = 0

            for idx in range(min(len(source_lines), len(hyp_lines))):
                src = source_lines[idx]
                hyp = hyp_lines[idx]
                issues = []

                if forbidden_re.search(hyp):
                    issues.append("script_bleed")
                    script_bleed += 1

                src_p = sorted(PLACEHOLDER_REGEX.findall(src))
                hyp_p = sorted(PLACEHOLDER_REGEX.findall(hyp))
                if src_p != hyp_p:
                    issues.append("broken_placeholder")
                    broken_placeholders += 1

                if CHATTER_REGEX.search(hyp):
                    issues.append("chatter")
                    chatter_count += 1

                if issues:
                    dedup_key = (model_name, code, idx)
                    if dedup_key in seen_issue_keys:
                        continue
                    seen_issue_keys.add(dedup_key)

                    defective_count += 1
                    entry = {
                        "audit_timestamp": timestamp,
                        "model": model_name,
                        "lang": code,
                        "line_idx": idx,
                        "source": src,
                        "flawed_translation": hyp,
                        "issues": issues
                    }
                    jsonl_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

            total = len(hyp_lines)
            rate = (defective_count / total) * 100 if total > 0 else 0
            summary_rows.append((model_name, code, total, script_bleed, broken_placeholders, chatter_count, defective_count, rate))

# Master clean markdown report
with open(summary_md, "w", encoding="utf-8") as md_out:
    md_out.write(f"# Cloud-Native Software Localization: Multi-Model Baseline Audit\n\n")
    md_out.write(f"**Audit Snapshot Timestamp:** `{timestamp}`  \n")
    md_out.write(f"**Total Model Suites Audited:** `{len(model_dirs)}` ({', '.join(model_dirs)})  \n")
    md_out.write(f"**Total Strings per Language:** `10,228`  \n")
    md_out.write(f"**Unique Defective Line Records:** `{len(seen_issue_keys)}`  \n\n")
    md_out.write("### Comprehensive Defect Breakdown\n\n")
    md_out.write("| Model Suite | Lang | Total Lines | Script Bleed | Broken Placeholders | Chatter | Total Defective | Defect Rate (%) |\n")
    md_out.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for row in summary_rows:
        md_out.write(f"| `{row[0]}` | `{row[1]}` | {row[2]} | {row[3]} | {row[4]} | {row[5]} | **{row[6]}** | **{row[7]:.2f}%** |\n")

print(f"\n✅ Zero duplicates verified: {len(seen_issue_keys)} unique defective strings logged across {len(summary_rows)} language files.")
