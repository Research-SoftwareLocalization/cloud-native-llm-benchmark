import os
import re
import json
import datetime
from pathlib import Path

REPO_ROOT = Path("/content/cloud-native-llm-benchmark") if Path("/content/cloud-native-llm-benchmark").exists() else Path(".")
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
XML_ENTITY_REGEX = re.compile(r'&\s*[a-zA-Z0-9._-]+;')
ACCELERATOR_REGEX = re.compile(r'&[^\s\d_;]')
REPLACEMENT_CHAR_REGEX = re.compile(r'\ufffd')
FORMULA_REGEX = re.compile(r'^[A-Z0-9_]+\([^)]*\)$')
POSIX_FMT_REGEX = re.compile(r'^%[a-zA-Z](\s*\([^)]*\))?$')
XML_ONLY_REGEX = re.compile(r'^&\s*[a-zA-Z0-9._-]+;$')
CAMEL_OR_SNAKE_REGEX = re.compile(r'([a-z][A-Z]|_[a-zA-Z])')

KNOWN_NTTS = {
    'SCSI', 'S-ATA', 'IEEE 1394', 'SD/ MMC', 'RITOn', 'kcm_partition', 'KTraderClient',
    'USB', 'PCI', 'BIOS', 'IPv4', 'IPv6', 'UUID', 'HDMI', 'TCP', 'UDP', 'DHCP', 'DNS',
    'RAM', 'CPU', 'GPU', 'Samba', 'OpenVPN', 'Cisco', 'ALSA', 'MIME', 'UTF-8', 'ASCII',
    'HTML', 'HTTP', 'HTTPS', 'LDAP', 'IMAP', 'POP3', 'SMTP', 'X11', 'OpenGL'
}

def has_accelerator(text: str) -> bool:
    cleaned = XML_ENTITY_REGEX.sub('', text)
    return bool(ACCELERATOR_REGEX.search(cleaned))

def is_legitimate_ntt(src_str: str, lang_code: str = "") -> bool:
    s = src_str.strip()
    if s in KNOWN_NTTS:
        return True
    if FORMULA_REGEX.match(s) or POSIX_FMT_REGEX.match(s) or XML_ONLY_REGEX.match(s):
        return True
    if re.match(r'^[A-Z0-9\s/\-\.]{2,12}$', s) and not s.isalpha():
        return True
    if s.isupper() and len(s) <= 6:
        return True
    if lang_code in ("de", "es", "fr"):
        if " " not in s and not CAMEL_OR_SNAKE_REGEX.search(s) and re.match(r'^[A-Za-z\-]+$', s):
            return True
    return False

with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    source_lines = [l.rstrip("\r\n") for l in f]

model_dirs = sorted([
    p.name for p in DATA_DIR.iterdir()
    if p.is_dir() and (p.name.startswith("baseline_") or p.name.startswith("repaired_"))
])
timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

jsonl_log = REPORTS_DIR / "audit_findings_log.jsonl"
summary_md = REPORTS_DIR / "qc_audit_summary.md"

seen_issue_keys = set()
summary_rows = []

print(f"📊 Running Unified QC Audit across {len(model_dirs)} suites: {model_dirs}")

with open(jsonl_log, "w", encoding="utf-8") as jsonl_out:
    for model_name in model_dirs:
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

                if forbidden_re.search(hyp):
                    issues.append("script_contamination")
                    script_contamination += 1

                src_p = sorted(PLACEHOLDER_REGEX.findall(src))
                hyp_p = sorted(PLACEHOLDER_REGEX.findall(hyp))
                if src_p != hyp_p:
                    issues.append("broken_placeholder")
                    broken_placeholders += 1

                if has_accelerator(src) and not has_accelerator(hyp):
                    issues.append("broken_accelerator")
                    broken_accelerators += 1

                if REPLACEMENT_CHAR_REGEX.search(hyp):
                    issues.append("encoding_corruption")
                    encoding_corruption += 1

                if CHATTER_REGEX.search(hyp):
                    issues.append("conversational_chatter")
                    chatter_count += 1

                src_strip, hyp_strip = src.strip(), hyp.strip()
                if src_strip == hyp_strip and len(src_strip) > 3 and any(c.isalpha() for c in src_strip):
                    if not is_legitimate_ntt(src_strip, code):
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

with open(summary_md, "w", encoding="utf-8") as md_out:
    md_out.write(f"# Cloud-Native Software Localization: Unified Quality Audit\n\n")
    md_out.write(f"**Audit Snapshot:** `{timestamp}`  \n")
    md_out.write(f"**Audited Suites ({len(model_dirs)}):** `{', '.join(model_dirs)}`  \n")
    md_out.write(f"**Corpus Size per Language:** `10,228` strings  \n")
    md_out.write(f"**Total Confirmed Defect Records:** `{len(seen_issue_keys)}`  \n\n")
    md_out.write("### Empirical Failure Mode & Invariant Breakdown\n\n")
    md_out.write("| Model Suite | Lang | Lines | Script Contam. | Broken %1 | Broken & | Corrupt () | Chatter | Copy Fallback | Total Flawed | Defect Rate (%) |\n")
    md_out.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for r in summary_rows:
        md_out.write(f"| `{r[0]}` | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | **{r[9]}** | **{r[10]:.2f}%** |\n")

print(f"✅ Unified audit complete. {len(seen_issue_keys)} confirmed defective records logged.")
