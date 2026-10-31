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
REPLACEMENT_CHAR_REGEX = re.compile(r'\ufffd')

with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    source_lines = [l.rstrip("\r\n") for l in f]

model_dirs = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("baseline_")])
timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

jsonl_log = REPORTS_DIR / "audit_findings_log.jsonl"
summary_md = REPORTS_DIR / "qc_audit_summary.md"

seen_issue_keys = set()
summary_rows = []

print(f"📊 Starting Full-Spectrum QC Audit across {len(model_dirs)} models...")

# Overwrite mode ('w') guarantees a fresh, clean snapshot
with open(jsonl_log, "w", encoding="utf-8") as jsonl_out:
    for model_name in model_dirs:
        print(f"  Auditing `{model_name}`...")
        model_dir = DATA_DIR / model_name

        for code, lang_name, forbidden_re in LANGUAGES:
            file_path = model_dir / f"{code}.txt"
            if not file_path.exists():
                continue

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                hyp_lines = [l.rstrip("\r\n") for l in f]

            script_contamination = 0
            broken_placeholders = 0
            broken_accelerators = 0
            encoding_corruption = 0
            chatter_count = 0
            copy_fallback = 0
            defective_count = 0

            for idx in range(min(len(source_lines), len(hyp_lines))):
                src = source_lines[idx]
                hyp = hyp_lines[idx]
                issues = []

                # 1. Script Contamination (Failure Mode 1)
                if forbidden_re.search(hyp):
                    issues.append("script_contamination")
                    script_contamination += 1

                # 2. Placeholders (%1, %d)
                src_p = sorted(PLACEHOLDER_REGEX.findall(src))
                hyp_p = sorted(PLACEHOLDER_REGEX.findall(hyp))
                if src_p != hyp_p:
                    issues.append("broken_placeholder")
                    broken_placeholders += 1

                # 3. UI Accelerators (&File)
                src_acc = len(ACCELERATOR_REGEX.findall(src))
                hyp_acc = len(ACCELERATOR_REGEX.findall(hyp))
                if src_acc > 0 and hyp_acc == 0:
                    issues.append("broken_accelerator")
                    broken_accelerators += 1

                # 4. Encoding Corruption (\ufffd - Failure Mode 4)
                if REPLACEMENT_CHAR_REGEX.search(hyp):
                    issues.append("encoding_corruption")
                    encoding_corruption += 1

                # 5. Conversational Chatter (Failure Mode 2)
                if CHATTER_REGEX.search(hyp):
                    issues.append("conversational_chatter")
                    chatter_count += 1

                # 6. Silent Copy Fallback (Failure Mode 3: words > 3 chars copied verbatim)
                src_strip = src.strip()
                hyp_strip = hyp.strip()
                if src_strip == hyp_strip and len(src_strip) > 3 and any(c.isalpha() for c in src_strip):
                    issues.append("copy_fallback")
                    copy_fallback += 1

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
            summary_rows.append((
                model_name, code, total,
                script_contamination, broken_placeholders, broken_accelerators,
                encoding_corruption, chatter_count, copy_fallback,
                defective_count, rate
            ))

# Master Markdown Report mapping to the paper's Failure Mode Taxonomy
with open(summary_md, "w", encoding="utf-8") as md_out:
    md_out.write(f"# Cloud-Native Software Localization: Comprehensive Quality Audit\n\n")
    md_out.write(f"**Audit Snapshot:** `{timestamp}`  \n")
    md_out.write(f"**Audited Model Suites ({len(model_dirs)}):** `{', '.join(model_dirs)}`  \n")
    md_out.write(f"**Corpus Size per Language:** `10,228` strings  \n")
    md_out.write(f"**Total Flagged Unique Lines:** `{len(seen_issue_keys)}`  \n\n")
    md_out.write("### Empirical Failure Mode & Invariant Breakdown\n\n")
    md_out.write("| Model Suite | Lang | Lines | Script Contam. | Broken %1 | Broken & | Corrupt () | Chatter | Copy Fallback | Total Flawed | Defect Rate (%) |\n")
    md_out.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for r in summary_rows:
        md_out.write(f"| `{r[0]}` | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | **{r[9]}** | **{r[10]:.2f}%** |\n")

print(f"\n✅ Full-spectrum audit complete. {len(seen_issue_keys)} unique defective records logged.")
