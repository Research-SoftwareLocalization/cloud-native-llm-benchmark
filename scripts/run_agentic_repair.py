import os
import sys
import time
import json
import re
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

REPO_ROOT = Path("/content/cloud-native-llm-benchmark") if Path("/content/cloud-native-llm-benchmark").exists() else Path(".")
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
FINDINGS_LOG = REPORTS_DIR / "audit_findings_log.jsonl"
TRACE_LOG = REPORTS_DIR / "correction_trace_log.jsonl"
TELEMETRY_LOG = REPORTS_DIR / "repair_execution_telemetry.jsonl"

LANGUAGES = {
    "ar": ("Modern Standard Arabic (العربية)", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff]')),
    "bn": ("Bengali (বাংলা script)", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "de": ("German (Deutsch)", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0980-\u09ff]')),
    "es": ("Spanish (Español)", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0980-\u09ff]')),
    "fr": ("French (Français)", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0980-\u09ff]')),
    "hi": ("Hindi (Devanagari script हिन्दी)", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "ja": ("Japanese (日本語)", re.compile(r'[\u0400-\u04ff\u0600-\u06ff\uac00-\ud7af]')),
    "mr": ("Marathi (Devanagari script मराठी)", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "pa": ("Punjabi (strictly Gurmukhi script ਪੰਜਾਬੀ, NEVER Shahmukhi/Arabic script)", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "ta": ("Tamil (தமிழ் script)", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "te": ("Telugu (తెలుగు script)", re.compile(r'[\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]')),
    "zh": ("Simplified Chinese (简体中文)", re.compile(r'[\u0400-\u04ff\u0600-\u06ff\uac00-\ud7af]'))
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
X11_COLOR_REGEX = re.compile(r'^[A-Za-z]+[1-4]$')
UA_STRING_REGEX = re.compile(r'^UADescription\s*\(.*\)$')
TZ_PATH_REGEX = re.compile(r'^[A-Za-z]+/\s*[A-Za-z_]+$')
EURO_COGNATE_PHRASE_REGEX = re.compile(r'^(\d+\s+Points|Position\s+[xyz]:?|Pixmap:?|Kig\s+figure|Dithering\s+PPI)$', re.IGNORECASE)

KNOWN_NTTS = {
    'SCSI', 'S-ATA', 'IEEE 1394', 'SD/ MMC', 'RITOn', 'kcm_partition', 'KTraderClient',
    'USB', 'PCI', 'BIOS', 'IPv4', 'IPv6', 'UUID', 'HDMI', 'TCP', 'UDP', 'DHCP', 'DNS',
    'RAM', 'CPU', 'GPU', 'Samba', 'OpenVPN', 'Cisco', 'ALSA', 'MIME', 'UTF-8', 'ASCII',
    'HTML', 'HTTP', 'HTTPS', 'LDAP', 'IMAP', 'POP3', 'SMTP', 'X11', 'OpenGL',
    'Iptc', 'NoPCL3GUI', 'SDP UUid:', 'KMobileTools EngineXP', 'Linediff: A< - > B'
}

def has_accelerator(text: str) -> bool:
    cleaned = XML_ENTITY_REGEX.sub('', text)
    return bool(ACCELERATOR_REGEX.search(cleaned))

def is_legitimate_ntt(src_str: str, lang_code: str = "") -> bool:
    s = src_str.strip()
    if s in KNOWN_NTTS:
        return True
    if (FORMULA_REGEX.match(s) or POSIX_FMT_REGEX.match(s) or XML_ONLY_REGEX.match(s)
            or X11_COLOR_REGEX.match(s) or UA_STRING_REGEX.match(s) or TZ_PATH_REGEX.match(s)):
        return True
    if re.match(r'^[A-Z0-9\s/\-\.]{2,12}$', s) and not s.isalpha():
        return True
    if s.isupper() and len(s) <= 6:
        return True
    if lang_code in ("de", "es", "fr"):
        if EURO_COGNATE_PHRASE_REGEX.match(s):
            return True
        if " " not in s and not CAMEL_OR_SNAKE_REGEX.search(s) and re.match(r'^[A-Za-z\-]+:?$', s):
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

def split_camelcase_hint(s: str) -> str:
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return spaced.replace('_', ' ')

def build_targeted_prompt(src: str, flawed: str, issues: list, lang_name: str):
    rules = []
    if "broken_placeholder" in issues:
        placeholders = sorted(list(set(PLACEHOLDER_REGEX.findall(src))))
        if placeholders:
            rules.append(f"- Exact Format Invariant: You MUST preserve all variable specifiers {placeholders} verbatim.")
        else:
            rules.append("- Pure Source Fidelity: Translate ONLY the words present in the English source string. Do NOT add any extra variables, formatting symbols, or parenthetical notes.")
    if "broken_accelerator" in issues:
        rules.append("- Keyboard Accelerator Invariant: Retain the keyboard shortcut marker '&' attached to an appropriate letter in the translated word.")
    if "script_contamination" in issues or "encoding_corruption" in issues:
        rules.append(f"- Script Purity Invariant: Output strictly in valid {lang_name}. NEVER output foreign scripts or corrupted characters.")
    if "copy_fallback" in issues:
        hint = split_camelcase_hint(src)
        rules.append(
            f"- Linguistic Invariant: Do NOT return '{src}' unchanged. "
            f"Translate the UI phrase or identifier (meaning: '{hint}') into natural {lang_name} while keeping brand prefixes intact."
        )

    rules_text = "\n".join(rules)
    system_prompt = (
        f"You are a strict software localization compiler for {lang_name}.\n"
        f"Translate the English software UI string into {lang_name} while rigorously obeying all invariants.\n"
        f"Rules:\n{rules_text}\n"
        f"Output ONLY the final single-line translation in {lang_name}. No preambles, notes, or quotes."
    )
    user_prompt = f"English UI String: {src}\n{lang_name} Translation:"
    return system_prompt, user_prompt

def query_gpu_mem_mb():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = res.stdout.strip().split(",")
            return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        pass
    return 0, 0

def repair_model_outputs(model_name: str, target_lang: str = None, dry_run: bool = False,
                         base_url: str = "https://api.openai.com/v1",
                         api_key_env: str = "OPENAI_API_KEY",
                         model_id: str = "gpt-4o-mini",
                         max_workers: int = 16):
    if not dry_run:
        api_key = os.environ.get(api_key_env, "EMPTY")
        client = OpenAI(base_url=base_url, api_key=api_key)
    else:
        client = None

    model_dir = DATA_DIR / model_name
    repaired_name = f"repaired_{model_name.replace('baseline_', '')}"
    repaired_dir = DATA_DIR / repaired_name
    repaired_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(FINDINGS_LOG, "r", encoding="utf-8") as f:
        findings = [json.loads(line) for line in f if json.loads(line).get("model") == model_name]

    if target_lang:
        findings = [f for f in findings if f.get("lang") == target_lang]

    print(f"🔍 Loaded {len(findings)} confirmed defect records for `{model_name}` -> `{repaired_name}` (Workers: {max_workers})", flush=True)

    lang_groups = {}
    for item in findings:
        lang_groups.setdefault(item["lang"], []).append(item)

    for lang_code, defect_list in sorted(lang_groups.items()):
        lang_name, _ = LANGUAGES[lang_code]
        src_baseline_path = model_dir / f"{lang_code}.txt"
        out_repaired_path = repaired_dir / f"{lang_code}.txt"

        if not src_baseline_path.exists():
            continue

        active_input_path = out_repaired_path if out_repaired_path.exists() else src_baseline_path
        with open(active_input_path, "r", encoding="utf-8") as f:
            current_lines = [l.rstrip("\r\n") for l in f]

        t_start = time.time()
        ts_start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

        def process_record(record):
            idx = record["line_idx"]
            src = record["source"]
            flawed = current_lines[idx]
            confirmed_issues = validate_single_line(src, flawed, lang_code)

            trace_record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": model_name,
                "model_id": model_id,
                "lang": lang_code,
                "line_idx": idx,
                "source": src,
                "flawed_translation": flawed,
                "validation_confirmed": bool(confirmed_issues),
                "confirmed_issues": confirmed_issues
            }

            if not confirmed_issues:
                trace_record["status"] = "ALREADY_COMPLIANT"
                return idx, None, "ALREADY_COMPLIANT", trace_record

            if dry_run:
                trace_record["status"] = "DRY_RUN_CONFIRMED"
                return idx, None, "DRY_RUN_CONFIRMED", trace_record

            sys_prompt, usr_prompt = build_targeted_prompt(src, flawed, confirmed_issues, lang_name)
            trace_record["repair_prompt"] = sys_prompt

            try:
                if "gemma" in model_id.lower():
                    messages = [{"role": "user", "content": sys_prompt + "\n\n" + usr_prompt}]
                else:
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt}
                    ]
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256
                )
                corrected_hyp = response.choices[0].message.content.strip().replace("\n", " ")
                if (corrected_hyp.startswith('"') and corrected_hyp.endswith('"')) or (corrected_hyp.startswith("'") and corrected_hyp.endswith("'")):
                    corrected_hyp = corrected_hyp[1:-1].strip()

                post_issues = validate_single_line(src, corrected_hyp, lang_code)
                trace_record["repaired_translation"] = corrected_hyp
                trace_record["post_repair_issues"] = post_issues

                if not post_issues:
                    trace_record["status"] = "REPAIR_SUCCESS"
                    return idx, corrected_hyp, "REPAIR_SUCCESS", trace_record
                else:
                    trace_record["status"] = "REPAIR_REJECTED_STILL_DEFECTIVE"
                    return idx, None, "FAILED", trace_record
            except Exception as e:
                trace_record["status"] = f"API_ERROR: {str(e)}"
                return idx, None, "FAILED", trace_record

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_record, rec) for rec in defect_list]
            for fut in as_completed(futures):
                results.append(fut.result())

        results.sort(key=lambda x: x[0])

        repaired_count = 0
        already_clean = 0
        failed_repairs = 0
        trace_entries = []

        for idx, new_hyp, status, trace_rec in results:
            trace_entries.append(trace_rec)
            if status in ("REPAIR_SUCCESS", "DRY_RUN_CONFIRMED"):
                repaired_count += 1
                if new_hyp is not None:
                    current_lines[idx] = new_hyp
            elif status == "ALREADY_COMPLIANT":
                already_clean += 1
            else:
                failed_repairs += 1

        duration_s = round(time.time() - t_start, 2)
        vram_mb, gpu_util = query_gpu_mem_mb()

        with open(TRACE_LOG, "a", encoding="utf-8") as trace_out:
            for entry in trace_entries:
                trace_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        telemetry_entry = {
            "start_utc": ts_start_utc,
            "end_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "baseline_suite": model_name,
            "repaired_suite": repaired_name,
            "model_id": model_id,
            "language": lang_code,
            "candidate_defects": len(defect_list),
            "repaired_count": repaired_count,
            "already_compliant": already_clean,
            "remaining_failed": failed_repairs,
            "wall_clock_seconds": duration_s,
            "throughput_lines_per_sec": round(len(defect_list) / duration_s, 2) if duration_s > 0 else 0,
            "gpu_vram_used_mb": vram_mb,
            "gpu_utilization_pct": gpu_util
        }
        with open(TELEMETRY_LOG, "a", encoding="utf-8") as tel_out:
            tel_out.write(json.dumps(telemetry_entry) + "\n")

        if not dry_run:
            with open(out_repaired_path, "w", encoding="utf-8") as f:
                f.write("\n".join(current_lines) + "\n")
            subprocess.run(["git", "add", str(out_repaired_path), str(TELEMETRY_LOG)], check=False)
            subprocess.run(["git", "commit", "-m", f"checkpoint(repair): {repaired_name} {lang_code} (+{repaired_count} fixed in {duration_s}s) [skip ci]"], check=False, capture_output=True)
            subprocess.run(["git", "push", "origin", "main"], check=False, capture_output=True)

        print(f"  📊 [{lang_code.upper()}] +{repaired_count} fixed | {already_clean} compliant | {failed_repairs} remaining | ⏱️ {duration_s}s ({vram_mb}MB VRAM)", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--base_url", default="https://api.openai.com/v1")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--model_id", default="gpt-4o-mini")
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()
    repair_model_outputs(
        args.model,
        target_lang=args.lang,
        dry_run=args.dry_run,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model_id=args.model_id,
        max_workers=args.max_workers
    )
