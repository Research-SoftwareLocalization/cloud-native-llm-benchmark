import os
import sys
import json
import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone
from openai import OpenAI

REPO_ROOT = Path("/content/cloud-native-llm-benchmark") if Path("/content/cloud-native-llm-benchmark").exists() else Path(".")
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
FINDINGS_LOG = REPORTS_DIR / "audit_findings_log.jsonl"
TRACE_LOG = REPORTS_DIR / "correction_trace_log.jsonl"

LANGUAGES = {
    "ar": ("Modern Standard Arabic", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff]')),
    "bn": ("Bengali", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "de": ("German", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0980-\u09ff]')),
    "es": ("Spanish", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0980-\u09ff]')),
    "fr": ("French", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0980-\u09ff]')),
    "hi": ("Hindi", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "ja": ("Japanese", re.compile(r'[\u0400-\u04ff\u0600-\u06ff\uac00-\ud7af]')),
    "mr": ("Marathi", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "pa": ("Punjabi", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "ta": ("Tamil", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "te": ("Telugu", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "zh": ("Simplified Chinese", re.compile(r'[\u0400-\u04ff\u0600-\u06ff\uac00-\ud7af]'))
}

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

def validate_single_line(src: str, hyp: str, lang_code: str):
    issues = []
    _, forbidden_re = LANGUAGES[lang_code]

    if forbidden_re.search(hyp):
        issues.append("script_contamination")

    src_p = sorted(PLACEHOLDER_REGEX.findall(src))
    hyp_p = sorted(PLACEHOLDER_REGEX.findall(hyp))
    if src_p != hyp_p:
        issues.append("broken_placeholder")

    if has_accelerator(src) and not has_accelerator(hyp):
        issues.append("broken_accelerator")

    if REPLACEMENT_CHAR_REGEX.search(hyp):
        issues.append("encoding_corruption")

    if CHATTER_REGEX.search(hyp):
        issues.append("conversational_chatter")

    src_s, hyp_s = src.strip(), hyp.strip()
    if src_s == hyp_s and len(src_s) > 3 and any(c.isalpha() for c in src_s) and lang_code not in ["en"]:
        if not is_legitimate_ntt(src_s, lang_code):
            issues.append("copy_fallback")

    return issues

def build_targeted_prompt(src: str, flawed: str, issues: list, lang_name: str):
    rules = []
    if "broken_placeholder" in issues:
        placeholders = sorted(list(set(PLACEHOLDER_REGEX.findall(src))))
        rules.append(f"- Exact Format Invariant: You MUST preserve all variable specifiers {placeholders} verbatim.")
    if "broken_accelerator" in issues:
        rules.append("- Keyboard Accelerator Invariant: Retain the keyboard shortcut marker '&' attached to an appropriate letter in the translated word.")
    if "script_contamination" in issues or "encoding_corruption" in issues:
        rules.append(f"- Script Purity Invariant: Output strictly in valid {lang_name} script. Do NOT leak foreign or corrupted characters.")
    if "copy_fallback" in issues:
        rules.append(f"- Linguistic Invariant: Do not copy the English string verbatim. Localize the UI string or developer identifier into natural {lang_name} while keeping brand names intact.")

    rules_text = "\n".join(rules)
    system_prompt = (
        f"You are a strict software localization compiler for {lang_name}.\n"
        f"Translate software UI strings while rigorously preserving programmatic invariants.\n"
        f"Rules:\n{rules_text}\n"
        f"Output ONLY the corrected single-line translation. No preambles, notes, or quotes."
    )
    user_prompt = f"English Source: {src}\nPrevious Flawed Translation: {flawed}\nCorrected Translation:"
    return system_prompt, user_prompt

def repair_model_outputs(model_name: str, target_lang: str = None, dry_run: bool = False,
                         base_url: str = "https://api.openai.com/v1",
                         api_key_env: str = "OPENAI_API_KEY",
                         model_id: str = "gpt-4o-mini"):
    if not dry_run:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_env}' is not set.")
        client = OpenAI(base_url=base_url, api_key=api_key)

    model_dir = DATA_DIR / model_name
    repaired_dir = DATA_DIR / f"repaired_{model_name.replace('baseline_', '')}"
    repaired_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not FINDINGS_LOG.exists():
        raise FileNotFoundError(f"Missing {FINDINGS_LOG}. Run audit first.")

    with open(FINDINGS_LOG, "r", encoding="utf-8") as f:
        findings = [json.loads(line) for line in f if json.loads(line).get("model") == model_name]

    if target_lang:
        findings = [f for f in findings if f.get("lang") == target_lang]

    print(f"🔍 Loaded {len(findings)} confirmed defect records for `{model_name}` ({target_lang or 'ALL'}). Mode: {'DRY RUN' if dry_run else 'LIVE REPAIR'}")

    lang_groups = {}
    for item in findings:
        lang_groups.setdefault(item["lang"], []).append(item)

    for lang_code, defect_list in lang_groups.items():
        lang_name, _ = LANGUAGES[lang_code]
        src_baseline_path = model_dir / f"{lang_code}.txt"
        out_repaired_path = repaired_dir / f"{lang_code}.txt"

        if not src_baseline_path.exists():
            continue

        # Start from existing repaired file if already created, otherwise from raw baseline
        active_input_path = out_repaired_path if out_repaired_path.exists() else src_baseline_path
        with open(active_input_path, "r", encoding="utf-8") as f:
            current_lines = [l.rstrip("\r\n") for l in f]

        repaired_count = 0
        already_clean = 0
        failed_repairs = 0
        trace_entries = []

        print(f"\n⚡ Processing {lang_code.upper()} ({len(defect_list)} candidate lines)...")

        for record in defect_list:
            idx = record["line_idx"]
            src = record["source"]
            flawed = current_lines[idx]

            confirmed_issues = validate_single_line(src, flawed, lang_code)

            trace_record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": model_name,
                "lang": lang_code,
                "line_idx": idx,
                "source": src,
                "flawed_translation": flawed,
                "validation_confirmed": bool(confirmed_issues),
                "confirmed_issues": confirmed_issues
            }

            if not confirmed_issues:
                already_clean += 1
                trace_record["status"] = "ALREADY_COMPLIANT"
                trace_entries.append(trace_record)
                continue

            if dry_run:
                repaired_count += 1
                trace_record["status"] = "DRY_RUN_CONFIRMED"
                trace_entries.append(trace_record)
                continue

            sys_prompt, usr_prompt = build_targeted_prompt(src, flawed, confirmed_issues, lang_name)
            trace_record["repair_prompt"] = sys_prompt

            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt}
                    ],
                    temperature=0.0
                )
                corrected_hyp = response.choices[0].message.content.strip().replace("\n", " ")
                post_issues = validate_single_line(src, corrected_hyp, lang_code)
                trace_record["repaired_translation"] = corrected_hyp
                trace_record["post_repair_issues"] = post_issues

                if not post_issues:
                    current_lines[idx] = corrected_hyp
                    repaired_count += 1
                    trace_record["status"] = "REPAIR_SUCCESS"
                else:
                    failed_repairs += 1
                    trace_record["status"] = "REPAIR_REJECTED_STILL_DEFECTIVE"

            except Exception as e:
                failed_repairs += 1
                trace_record["status"] = f"API_ERROR: {str(e)}"

            trace_entries.append(trace_record)

        with open(TRACE_LOG, "a", encoding="utf-8") as trace_out:
            for entry in trace_entries:
                trace_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if not dry_run:
            with open(out_repaired_path, "w", encoding="utf-8") as f:
                f.write("\n".join(current_lines) + "\n")
            print(f"✅ Saved repaired output to: {out_repaired_path.relative_to(REPO_ROOT)}")

        print(f"  📊 {lang_code}: {repaired_count} repaired, {already_clean} already compliant, {failed_repairs} failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--base_url", default="https://api.openai.com/v1")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--model_id", default="gpt-4o-mini")
    args = parser.parse_args()
    repair_model_outputs(
        args.model,
        target_lang=args.lang,
        dry_run=args.dry_run,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model_id=args.model_id
    )
