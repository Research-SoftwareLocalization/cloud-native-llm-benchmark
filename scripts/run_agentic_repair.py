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
ARCHIVE_DIR = REPO_ROOT / "archive" / "raw_baselines"
REPORTS_DIR = REPO_ROOT / "reports"
FINDINGS_LOG = REPORTS_DIR / "audit_findings_log.jsonl"
TRACE_LOG = REPORTS_DIR / "correction_trace_log.jsonl"

LANGUAGES = {
    "ar": ("Modern Standard Arabic", re.compile(r'[一-鿿가-힯Ѐ-ӿ]')),
    "bn": ("Bengali", re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]')),
    "de": ("German", re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿঀ-৿]')),
    "es": ("Spanish", re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿঀ-৿]')),
    "fr": ("French", re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿঀ-৿]')),
    "hi": ("Hindi", re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]')),
    "ja": ("Japanese", re.compile(r'[Ѐ-ӿ؀-ۿ가-힯]')),
    "mr": ("Marathi", re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]')),
    "pa": ("Punjabi", re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]')),
    "ta": ("Tamil", re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]')),
    "te": ("Telugu", re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]')),
    "zh": ("Simplified Chinese", re.compile(r'[Ѐ-ӿ؀-ۿ가-힯]'))
}

PLACEHOLDER_REGEX = re.compile(r'%\d+|%[a-zA-Z]|\{[^}]+\}')
ACCELERATOR_REGEX = re.compile(r'&[a-zA-Z]')
CHATTER_REGEX = re.compile(r'(?i)^(here is the translation|translation:|note:|as an ai|translate into)')

def validate_single_line(src: str, hyp: str, lang_code: str):
    issues = []
    _, forbidden_re = LANGUAGES[lang_code]

    if forbidden_re.search(hyp):
        issues.append("script_contamination")

    src_p = sorted(PLACEHOLDER_REGEX.findall(src))
    hyp_p = sorted(PLACEHOLDER_REGEX.findall(hyp))
    if src_p != hyp_p:
        issues.append("broken_placeholder")

    if len(ACCELERATOR_REGEX.findall(src)) > 0 and len(ACCELERATOR_REGEX.findall(hyp)) == 0:
        issues.append("broken_accelerator")

    src_s, hyp_s = src.strip(), hyp.strip()
    if src_s == hyp_s and len(src_s) > 3 and any(c.isalpha() for c in src_s) and lang_code not in ["en"]:
        issues.append("copy_fallback")

    if CHATTER_REGEX.search(hyp):
        issues.append("conversational_chatter")

    return issues

def build_targeted_prompt(src: str, flawed: str, issues: list, lang_name: str):
    rules = []
    if "broken_placeholder" in issues:
        placeholders = sorted(list(set(PLACEHOLDER_REGEX.findall(src))))
        rules.append(f"- Exact Format Invariant: You MUST preserve all variable specifiers {placeholders} verbatim.")
    if "broken_accelerator" in issues:
        accs = sorted(list(set(ACCELERATOR_REGEX.findall(src))))
        rules.append(f"- Keyboard Accelerator Invariant: Retain the keyboard shortcut marker '&' attached to an appropriate letter (source has {accs}).")
    if "script_contamination" in issues:
        rules.append(f"- Script Purity Invariant: Output strictly in {lang_name} script. Do NOT leak foreign characters.")
    if "copy_fallback" in issues:
        rules.append(f"- Linguistic Invariant: Do not copy the English string. Provide an authentic {lang_name} software translation.")

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
    archive_model_dir = ARCHIVE_DIR / model_name
    archive_model_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not FINDINGS_LOG.exists():
        raise FileNotFoundError(f"Missing {FINDINGS_LOG}. Please run audit first.")

    with open(FINDINGS_LOG, "r", encoding="utf-8") as f:
        findings = [json.loads(line) for line in f if json.loads(line).get("model") == model_name]

    if target_lang:
        findings = [f for f in findings if f.get("lang") == target_lang]

    print(f"🔍 Loaded {len(findings)} flagged records for `{model_name}` ({target_lang or 'ALL'}). Mode: {'DRY RUN' if dry_run else 'LIVE REPAIR'}")

    lang_groups = {}
    for item in findings:
        lang_groups.setdefault(item["lang"], []).append(item)

    for lang_code, defect_list in lang_groups.items():
        lang_name, _ = LANGUAGES[lang_code]
        file_path = model_dir / f"{lang_code}.txt"
        archive_path = archive_model_dir / f"{lang_code}.txt"

        if not file_path.exists():
            continue

        # Requirement 1: Archive raw baseline before touching
        if not archive_path.exists():
            shutil.copy2(file_path, archive_path)
            print(f"📦 Archived raw baseline to: {archive_path.relative_to(REPO_ROOT)}")

        with open(file_path, "r", encoding="utf-8") as f:
            current_lines = [l.rstrip("\r\n") for l in f]

        repaired_count = 0
        false_positives = 0
        failed_repairs = 0
        trace_entries = []

        print(f"\n⚡ Validating {lang_code.upper()} ({len(defect_list)} candidate defect lines)...")

        for record in defect_list:
            idx = record["line_idx"]
            src = record["source"]
            flawed = current_lines[idx]

            # Requirement 2: Re-validate defect in real time
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
                false_positives += 1
                trace_record["status"] = "SKIPPED_FALSE_POSITIVE"
                trace_entries.append(trace_record)
                continue

            if dry_run:
                repaired_count += 1
                trace_record["status"] = "DRY_RUN_CONFIRMED"
                trace_entries.append(trace_record)
                continue

            # Live API repair
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

        # Requirement 3: Append end-to-end trace logs
        with open(TRACE_LOG, "a", encoding="utf-8") as trace_out:
            for entry in trace_entries:
                trace_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Save clean in-place (Requirement 1: NO suffixes, keeps lang.txt)
        if not dry_run and repaired_count > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(current_lines) + "\n")
            print(f"✅ In-place update complete: {file_path.relative_to(REPO_ROOT)}")

        print(f"  📊 {lang_code}: {repaired_count} confirmed/repaired, {false_positives} false-positives, {failed_repairs} failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    repair_model_outputs(args.model, target_lang=args.lang, dry_run=args.dry_run)
