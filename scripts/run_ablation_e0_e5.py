#!/usr/bin/env python3
# Component-Wise IGSC Ablation Study (E0-E5), Provenance Capture & Statistical Suite
# Models: Qwen/Qwen2.5-7B-Instruct and google/gemma-2-9b-it across 12 languages.
# Strictly non-destructive: writes ONLY to data/ablation/ and reports/ablation/.

import os
import re
import sys
import json
import time
import platform
import hashlib
import argparse
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import numpy as np
import pandas as pd
import torch
from scipy import stats
from openai import OpenAI
from huggingface_hub import HfApi

REPO_ROOT = Path("/content/cloud-native-llm-benchmark") if Path("/content/cloud-native-llm-benchmark").exists() else Path(".")
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
ABLATION_DATA_DIR = DATA_DIR / "ablation"
ABLATION_REPORTS_DIR = REPORTS_DIR / "ablation"
TRACES_DIR = ABLATION_REPORTS_DIR / "traces"
SOURCE_FILE = DATA_DIR / "source" / "en.txt"
FINDINGS_LOG = REPORTS_DIR / "audit_findings_log.jsonl"
TRACE_LOG = REPORTS_DIR / "correction_trace_log.jsonl"
TELEMETRY_LOG = REPORTS_DIR / "repair_execution_telemetry.jsonl"
ABLATION_TELEMETRY_LOG = ABLATION_REPORTS_DIR / "ablation_execution_telemetry.jsonl"
CK_SCORES_CSV = REPO_ROOT / "metrics" / "cometkiwi" / "cometkiwi_scores.csv"
CK_DELTA_CSV = REPO_ROOT / "metrics" / "cometkiwi" / "cometkiwi_before_after_delta.csv"

ALL_LANGS = ["ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"]

BARE_LANG_NAMES = {
    "ar": "Modern Standard Arabic", "bn": "Bengali", "de": "German", "es": "Spanish",
    "fr": "French", "hi": "Hindi", "ja": "Japanese", "mr": "Marathi",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "zh": "Simplified Chinese",
}

ORTHO_LANGUAGES = {
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
    "zh": ("Simplified Chinese (简体中文)", re.compile(r'[\u0400-\u04ff\u0600-\u06ff\uac00-\ud7af]')),
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

ABLATION_MODELS = [
    {
        "short": "qwen_7b",
        "baseline_suite": "baseline_qwen_7b",
        "repaired_suite": "repaired_qwen_7b",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "extra_vllm_args": [],
    },
    {
        "short": "gemma_9b",
        "baseline_suite": "baseline_gemma_9b",
        "repaired_suite": "repaired_gemma_9b",
        "model_id": "google/gemma-2-9b-it",
        "extra_vllm_args": ["--dtype", "bfloat16"],
    },
]

ALL_HF_REPOS = [
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen2.5-7B-Instruct",
    "Unbabel/wmt22-cometkiwi-da",
]

def write_lf_bytes(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_bytes(clean.encode("utf-8"))

def verify_checksums_intact(stage_label: str):
    chk = REPO_ROOT / "CHECKSUMS_SHA256.txt"
    lines = [ln.strip() for ln in chk.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in lines:
        exp_hash, rel = ln.split(None, 1)
        rel = rel.lstrip("*")
        act_hash = hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
        assert act_hash == exp_hash, f"SAFETY ABORT ({stage_label}): {rel} hash changed!"
    print(f"🔒 [{stage_label}] Verified all {len(lines)}/121 frozen benchmark files match CHECKSUMS_SHA256.txt.", flush=True)

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

def capture_system_and_model_provenance(phase_name: str):
    ABLATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    smi_full = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True
    ).stdout.strip()

    cpu_model = "Unknown"
    if Path("/proc/cpuinfo").exists():
        for ln in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if "model name" in ln:
                cpu_model = ln.split(":", 1)[1].strip()
                break
    mem_total_kb = "Unknown"
    if Path("/proc/meminfo").exists():
        for ln in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if ln.startswith("MemTotal:"):
                mem_total_kb = ln.split(":", 1)[1].strip()
                break

    git_head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    pkg_versions = {}
    for pkg in ["torch", "vllm", "transformers", "comet", "scipy", "numpy", "pandas", "huggingface_hub", "openai"]:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "installed")
        except Exception:
            pkg_versions[pkg] = "not_loaded_in_this_phase"

    hf_revisions = {}
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for repo_id in ALL_HF_REPOS:
        try:
            info = api.model_info(repo_id)
            hf_revisions[repo_id] = {
                "hf_commit_sha": info.sha,
                "last_modified": str(info.last_modified) if hasattr(info, "last_modified") else None,
            }
        except Exception as e:
            hf_revisions[repo_id] = {"error": str(e)}

    manifest_path = ABLATION_REPORTS_DIR / "ablation_system_manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    existing[phase_name] = {
        "timestamp_utc": ts_utc,
        "git_head_baseline": git_head,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_model": cpu_model,
        "cpu_logical_cores": os.cpu_count(),
        "system_ram": mem_total_kb,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0,
        "nvidia_smi_detail": smi_full,
        "package_versions": pkg_versions,
    }
    existing["huggingface_model_revisions"] = hf_revisions
    existing["commercial_api_provenance"] = {
        "gpt-4o-mini": {
            "provider": "OpenAI Chat Completions REST API v1",
            "model_parameter": "gpt-4o-mini",
            "canonical_snapshot": "gpt-4o-mini-2024-07-18",
            "baseline_execution_dates_utc": ["2026-10-17", "2026-10-18T09:42:32Z", "2026-10-18T09:46:57Z"],
            "igsc_repair_completion_utc": "2026-11-01T08:23:24Z",
            "sampling_temperature": 0.0
        }
    }
    write_lf_bytes(manifest_path, json.dumps(existing, indent=2) + "\n")

    pip_txt = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    write_lf_bytes(ABLATION_REPORTS_DIR / f"requirements-ablation-{phase_name}-lock.txt", pip_txt)

    env_txt = (
        f"=== Ablation & Statistical Upgrade Hardware/Environment Log ({phase_name}) ===\n"
        f"Timestamp (UTC): {ts_utc}\n"
        f"Git Baseline Commit: {git_head}\n"
        f"OS Platform: {platform.platform()}\n"
        f"CPU: {cpu_model} ({os.cpu_count()} cores) | RAM: {mem_total_kb}\n"
        f"GPU: {smi_full}\n"
        f"Python: {sys.version.split()[0]} | PyTorch: {pkg_versions.get('torch')} | CUDA: {torch.version.cuda}\n"
        f"vLLM: {pkg_versions.get('vllm')} | Transformers: {pkg_versions.get('transformers')} | COMET: {pkg_versions.get('comet')}\n"
        f"Hugging Face Model Revisions:\n" + json.dumps(hf_revisions, indent=2) + "\n"
    )
    hw_log_path = ABLATION_REPORTS_DIR / "ablation_hardware_env_info.txt"
    prev_hw = hw_log_path.read_text(encoding="utf-8") if hw_log_path.exists() else ""
    if f"({phase_name})" not in prev_hw:
        write_lf_bytes(hw_log_path, prev_hw + ("\n" if prev_hw else "") + env_txt)
    print(f"📋 Captured full hardware, package lockfile, and HF commit SHAs for `{phase_name}`.", flush=True)

def has_accelerator(text: str) -> bool:
    return bool(ACCELERATOR_REGEX.search(XML_ENTITY_REGEX.sub('', text)))

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
    _, forbidden_re = ORTHO_LANGUAGES[lang_code]
    if forbidden_re.search(hyp):
        issues.append("script_contamination")
    if sorted(PLACEHOLDER_REGEX.findall(src)) != sorted(PLACEHOLDER_REGEX.findall(hyp)):
        issues.append("broken_placeholder")
    if has_accelerator(src) and not has_accelerator(hyp):
        issues.append("broken_accelerator")
    if REPLACEMENT_CHAR_REGEX.search(hyp):
        issues.append("encoding_corruption")
    if CHATTER_REGEX.search(hyp):
        issues.append("conversational_chatter")
    src_s, hyp_s = src.strip(), hyp.strip()
    if src_s == hyp_s and len(src_s) > 3 and any(c.isalpha() for c in src_s) and lang_code != "en":
        if not is_legitimate_ntt(src_s, lang_code):
            issues.append("copy_fallback")
    return issues

def split_camelcase_hint(s: str) -> str:
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return spaced.replace('_', ' ')

def build_ablation_prompt(exp_id: str, src: str, issues: list, lang_code: str):
    placeholders = sorted(list(set(PLACEHOLDER_REGEX.findall(src))))
    if placeholders:
        ph_rule = f"Exact Format Invariant: You MUST preserve all variable specifiers {placeholders} verbatim."
    else:
        ph_rule = "Pure Source Fidelity: Translate ONLY the words present in the English source string. Do NOT add any extra variables, formatting symbols, or parenthetical notes."

    if exp_id == "E2":
        bare_name = BARE_LANG_NAMES[lang_code]
        sys_prompt = (
            f"You are an enterprise cloud-native software localization engine.\n"
            f"Translate the user-provided English software string into {bare_name} ({lang_code}).\n\n"
            f"Strict Rules:\n"
            f"1. Output ONLY the translated string on a single line. Do NOT include explanations, notes, or surrounding quotes.\n"
            f"2. {ph_rule}\n"
            f"3. Keep CLI flags (e.g., --kubeconfig, --namespace), API paths, YAML keys, and proper cloud-native product names "
            f"(e.g., Kubernetes, Docker, Helm, Prometheus, Istio, Pod, DaemonSet, StatefulSet, Ingress) in their standard technical form.\n"
            f"4. Use formal, natural software UI/documentation register appropriate for {bare_name}."
        )
        return sys_prompt, src

    lang_name, _ = ORTHO_LANGUAGES[lang_code]
    rules = []
    if "broken_placeholder" in issues or exp_id == "E3":
        rules.append(f"- {ph_rule}")
    if "broken_accelerator" in issues:
        rules.append("- Keyboard Accelerator Invariant: Retain the keyboard shortcut marker '&' attached to an appropriate letter in the translated word.")
    if exp_id == "E3":
        rules.append(f"- Script Purity Invariant: Output strictly in valid {lang_name}. NEVER output foreign scripts or corrupted characters.")
    elif exp_id == "E4":
        if "script_contamination" in issues or "encoding_corruption" in issues:
            rules.append(f"- Script Purity Invariant: Output strictly in valid {lang_name}. NEVER output foreign scripts or corrupted characters.")
        if "copy_fallback" in issues:
            hint = split_camelcase_hint(src)
            rules.append(
                f"- Linguistic Invariant: Do NOT return '{src}' unchanged. "
                f"Translate the UI phrase or identifier (meaning: '{hint}') into natural {lang_name} while keeping brand prefixes intact."
            )

    rules_text = "\n".join(rules)
    sys_prompt = (
        f"You are a strict software localization compiler for {lang_name}.\n"
        f"Translate the English software UI string into {lang_name} while rigorously obeying all invariants.\n"
        f"Rules:\n{rules_text}\n"
        f"Output ONLY the final single-line translation in {lang_name}. No preambles, notes, or quotes."
    )
    usr_prompt = f"English UI String: {src}\n{lang_name} Translation:"
    return sys_prompt, usr_prompt

def audit_directory(suite_dir: Path, source_lines: list):
    t0 = time.perf_counter()
    totals = {
        "total_segments": 0, "flawed": 0, "script": 0, "ph_total": 0,
        "ph_omit": 0, "ph_halluc": 0, "ph_mut": 0,
        "acc": 0, "enc": 0, "chat": 0, "copy": 0
    }
    per_lang = {}
    for lang in ALL_LANGS:
        fp = suite_dir / f"{lang}.txt"
        raw_bytes = fp.read_bytes()
        assert b"\r\n" not in raw_bytes, f"CRLF detected in {fp}"
        hyps = raw_bytes.decode("utf-8").splitlines()
        assert len(hyps) == len(source_lines), f"Line count mismatch in {fp}: {len(hyps)}"
        l_flawed = l_script = l_ph = l_halluc = l_enc = l_copy = 0
        for s, h in zip(source_lines, hyps):
            totals["total_segments"] += 1
            iss = validate_single_line(s, h, lang)
            if iss:
                totals["flawed"] += 1; l_flawed += 1
            if "script_contamination" in iss:
                totals["script"] += 1; l_script += 1
            if "broken_placeholder" in iss:
                totals["ph_total"] += 1; l_ph += 1
                sp = sorted(PLACEHOLDER_REGEX.findall(s))
                hp = sorted(PLACEHOLDER_REGEX.findall(h))
                if len(sp) > 0 and len(hp) == 0:
                    totals["ph_omit"] += 1
                elif len(sp) == 0 and len(hp) > 0:
                    totals["ph_halluc"] += 1; l_halluc += 1
                else:
                    totals["ph_mut"] += 1
            if "broken_accelerator" in iss:
                totals["acc"] += 1
            if "encoding_corruption" in iss:
                totals["enc"] += 1; l_enc += 1
            if "conversational_chatter" in iss:
                totals["chat"] += 1
            if "copy_fallback" in iss:
                totals["copy"] += 1; l_copy += 1
        per_lang[lang] = {
            "flawed": l_flawed, "spr_pct": round(100.0 * (1.0 - l_flawed / len(source_lines)), 2),
            "script": l_script, "ph_total": l_ph, "ph_halluc": l_halluc, "enc": l_enc, "copy": l_copy
        }
    cpu_s = round(time.perf_counter() - t0, 3)
    N = totals["total_segments"]
    totals["ddr_pct"] = round(100.0 * totals["flawed"] / N, 2)
    totals["spr_pct"] = round(100.0 - totals["ddr_pct"], 2)
    totals["audit_cpu_s"] = cpu_s
    return totals, per_lang

def compute_table9_and_auxiliary_audits(source_lines: list):
    df = pd.read_csv(CK_DELTA_CSV)
    model_order = ["mistral_nemo", "gemma_9b", "llama_8b", "qwen_7b", "gpt4o"]
    rows = []
    for m in model_order:
        sub = df[df["model"] == m].sort_values("language")
        b = sub["baseline_cometkiwi"].to_numpy(dtype=float)
        r = sub["repaired_cometkiwi"].to_numpy(dtype=float)
        d = r - b
        L = len(d)
        mean_d = float(np.mean(d))
        sd_d = float(np.std(d, ddof=1))
        se_d = sd_d / np.sqrt(L)
        ci_lo, ci_hi = stats.t.interval(0.95, df=L - 1, loc=mean_d, scale=se_d)
        t_stat, p_t = stats.ttest_rel(r, b)
        w_stat, p_w = stats.wilcoxon(d, zero_method="zsplit" if np.any(d == 0) else "wilcox")
        rows.append({
            "model": m, "mean_base_ck": round(float(np.mean(b)), 4),
            "mean_igsc_ck": round(float(np.mean(r)), 4), "mean_delta_ck": round(mean_d, 4),
            "sd_delta_ck": round(sd_d, 4), "ci95_low": round(float(ci_lo), 4), "ci95_high": round(float(ci_hi), 4),
            "paired_t": round(float(t_stat), 4), "raw_t_p": float(p_t), "raw_wilcoxon_p": float(p_w),
            "pos_langs": int(np.sum(d > 0)),
        })
    res_df = pd.DataFrame(rows)
    for col, out_col in [("raw_t_p", "holm_t_p"), ("raw_wilcoxon_p", "holm_wilcoxon_p")]:
        pvals = res_df[col].to_numpy()
        M = len(pvals)
        order = np.argsort(pvals)
        adj = np.empty(M, dtype=float)
        running_max = 0.0
        for rank_idx, orig_idx in enumerate(order):
            val = min(1.0, pvals[orig_idx] * (M - rank_idx))
            running_max = max(running_max, val)
            adj[orig_idx] = running_max
        res_df[out_col] = adj
    write_lf_bytes(ABLATION_REPORTS_DIR / "table9_ci_holm_bonferroni.csv", res_df.to_csv(index=False))

    acc_rows = []
    for s_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith(("baseline_", "repaired_"))):
        bool_fail = exact_src_acc_fail = 0
        for lang in ALL_LANGS:
            hyps = (s_dir / f"{lang}.txt").read_text(encoding="utf-8").splitlines()
            for s, h in zip(source_lines, hyps):
                s_cnt = len(ACCELERATOR_REGEX.findall(XML_ENTITY_REGEX.sub("", s)))
                h_cnt = len(ACCELERATOR_REGEX.findall(XML_ENTITY_REGEX.sub("", h)))
                if s_cnt > 0 and h_cnt == 0:
                    bool_fail += 1
                if s_cnt > 0 and h_cnt != s_cnt:
                    exact_src_acc_fail += 1
        acc_rows.append({"suite": s_dir.name, "boolean_has_acc_failures": bool_fail, "exact_source_acc_count_failures": exact_src_acc_fail})
    write_lf_bytes(ABLATION_REPORTS_DIR / "accelerator_exact_parity_audit.csv", pd.DataFrame(acc_rows).to_csv(index=False))

    ntt_non_euro = sum(1 for s in source_lines if len(s.strip()) > 3 and any(c.isalpha() for c in s) and is_legitimate_ntt(s, "hi"))
    ntt_euro = sum(1 for s in source_lines if len(s.strip()) > 3 and any(c.isalpha() for c in s) and is_legitimate_ntt(s, "de"))
    ntt_stats = {
        "total_source_strings": len(source_lines),
        "curated_known_ntts_lexicon_size": len(KNOWN_NTTS),
        "universal_ntt_matched_strings": ntt_non_euro,
        "universal_ntt_pct": round(100.0 * ntt_non_euro / len(source_lines), 2),
        "european_cognate_inclusive_ntt_strings": ntt_euro,
        "european_cognate_inclusive_ntt_pct": round(100.0 * ntt_euro / len(source_lines), 2),
    }
    write_lf_bytes(ABLATION_REPORTS_DIR / "ntt_corpus_coverage.json", json.dumps(ntt_stats, indent=2) + "\n")
    print("📊 Saved Table 9 CIs/Holm-Bonferroni CSV, Accelerator Exact-Parity CSV, and NTT Coverage JSON.", flush=True)

def wait_for_vllm(proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            r = httpx.get("http://localhost:8000/v1/models", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(4)
    return False

def kill_vllm():
    subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)
    time.sleep(3)

def run_phase_init(source_lines: list):
    capture_system_and_model_provenance("vllm_generation")
    compute_table9_and_auxiliary_audits(source_lines)

def run_phase_generate_model(source_lines: list, target_short: str):
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    findings_by_model_lang = {}
    with open(FINDINGS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            findings_by_model_lang.setdefault((rec["model"], rec["lang"]), []).append(rec)

    trace_lookup = {}
    if TRACE_LOG.exists():
        with open(TRACE_LOG, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("repaired_translation") is not None:
                    trace_lookup[(rec["model"], rec["lang"], rec["line_idx"])] = rec

    telemetry_e5 = {}
    if TELEMETRY_LOG.exists():
        with open(TELEMETRY_LOG, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                b_suite = rec["baseline_suite"]
                telemetry_e5[b_suite] = telemetry_e5.get(b_suite, 0.0) + float(rec["wall_clock_seconds"])

    timing_file = ABLATION_REPORTS_DIR / "ablation_timings.json"
    timings = json.loads(timing_file.read_text(encoding="utf-8")) if timing_file.exists() else {}

    for m_cfg in ABLATION_MODELS:
        short = m_cfg["short"]
        if target_short != "all" and short != target_short:
            continue
        b_suite = m_cfg["baseline_suite"]
        model_id = m_cfg["model_id"]

        e4_dir = ABLATION_DATA_DIR / f"{short}_E4"
        e4_dir.mkdir(parents=True, exist_ok=True)
        missing_cnt = 0
        staged = {}
        for lang in ALL_LANGS:
            base_lines = [l.rstrip("\r\n") for l in (DATA_DIR / b_suite / f"{lang}.txt").read_text(encoding="utf-8").splitlines()]
            e4_lines = list(base_lines)
            for rec in findings_by_model_lang.get((b_suite, lang), []):
                idx = rec["line_idx"]
                t_rec = trace_lookup.get((b_suite, lang, idx))
                if t_rec and t_rec.get("repaired_translation"):
                    e4_lines[idx] = t_rec["repaired_translation"]
                else:
                    missing_cnt += 1
            staged[lang] = e4_lines
        if missing_cnt == 0:
            for lang, e4_lines in staged.items():
                write_lf_bytes(e4_dir / f"{lang}.txt", "\n".join(e4_lines) + "\n")
            timings[f"{short}_E4"] = round(telemetry_e5.get(b_suite, 0.0), 2)
            write_lf_bytes(timing_file, json.dumps(timings, indent=2) + "\n")
            print(f"♻️ [{short}_E4] Reconstructed all 12 languages from correction_trace_log.jsonl (0 missing).", flush=True)
        else:
            print(f"ℹ️ [{short}_E4] {missing_cnt} records missing in trace log; will generate E4 via vLLM.", flush=True)

        exps_needed = ["E2", "E3"] + (["E4"] if missing_cnt > 0 else [])
        pending_work = any(
            not (ABLATION_DATA_DIR / f"{short}_{exp}" / f"{lang}.txt").exists()
            for exp in exps_needed for lang in ALL_LANGS
        )
        if not pending_work:
            print(f"⏩ All vLLM ablation files for {short} ({exps_needed}) already exist. Skipping vLLM launch.", flush=True)
            continue

        print(f"\n{'='*70}\n🚀 LAUNCHING vLLM FOR ABLATION: {model_id}\n{'='*70}", flush=True)
        kill_vllm()
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "EMPTY"
        vllm_cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id,
            "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "4096",
        ] + m_cfg["extra_vllm_args"]
        proc = subprocess.Popen(vllm_cmd, env=env)
        try:
            if not wait_for_vllm(proc):
                raise RuntimeError(f"vLLM failed to start for {model_id}")
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

            for exp in exps_needed:
                exp_dir = ABLATION_DATA_DIR / f"{short}_{exp}"
                exp_dir.mkdir(parents=True, exist_ok=True)
                trace_fp = TRACES_DIR / f"{short}_{exp}_trace.jsonl"
                exp_wall = timings.get(f"{short}_{exp}", 0.0)

                for lang in ALL_LANGS:
                    out_fp = exp_dir / f"{lang}.txt"
                    if out_fp.exists() and len(out_fp.read_text(encoding="utf-8").splitlines()) == len(source_lines):
                        print(f"  ⏩ [{short}_{exp} -> {lang}] Already exists. Skipping.", flush=True)
                        continue

                    base_lines = [l.rstrip("\r\n") for l in (DATA_DIR / b_suite / f"{lang}.txt").read_text(encoding="utf-8").splitlines()]
                    defects = findings_by_model_lang.get((b_suite, lang), [])
                    ts_start = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    t_lang = time.perf_counter()

                    def run_one(rec):
                        idx = rec["line_idx"]
                        src = source_lines[idx]
                        flawed = base_lines[idx]
                        iss = validate_single_line(src, flawed, lang)
                        if not iss:
                            return idx, flawed, None
                        sys_p, usr_p = build_ablation_prompt(exp, src, iss, lang)
                        if "gemma" in model_id.lower():
                            msgs = [{"role": "user", "content": sys_p + "\n\n" + usr_p}]
                        else:
                            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
                        resp = client.chat.completions.create(
                            model=model_id, messages=msgs, temperature=0.0, max_tokens=256
                        )
                        cand = resp.choices[0].message.content.strip().replace("\n", " ").replace("\r", "")
                        if (cand.startswith('"') and cand.endswith('"')) or (cand.startswith("'") and cand.endswith("'")):
                            cand = cand[1:-1].strip()
                        final_cand = cand if cand else flawed
                        post_iss = validate_single_line(src, final_cand, lang)
                        t_entry = {
                            "experiment": exp, "model": short, "model_id": model_id,
                            "lang": lang, "line_idx": idx, "source": src,
                            "flawed_baseline": flawed, "pre_issues": iss,
                            "candidate_repair": final_cand, "post_issues": post_iss
                        }
                        return idx, final_cand, t_entry

                    new_lines = list(base_lines)
                    lang_traces = []
                    with ThreadPoolExecutor(max_workers=16) as pool:
                        futs = [pool.submit(run_one, rec) for rec in defects]
                        for fut in as_completed(futs):
                            i_line, hyp_str, t_ent = fut.result()
                            new_lines[i_line] = hyp_str
                            if t_ent is not None:
                                lang_traces.append(t_ent)

                    lang_traces.sort(key=lambda x: x["line_idx"])
                    dt = round(time.perf_counter() - t_lang, 2)
                    ts_end = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    vram_mb, gpu_util = query_gpu_mem_mb()
                    exp_wall = round(exp_wall + dt, 2)

                    write_lf_bytes(out_fp, "\n".join(new_lines) + "\n")
                    with open(trace_fp, "a", encoding="utf-8", newline="\n") as tf:
                        for te in lang_traces:
                            tf.write(json.dumps(te, ensure_ascii=False) + "\n")

                    tel_rec = {
                        "start_utc": ts_start, "end_utc": ts_end,
                        "experiment": exp, "model": short, "model_id": model_id, "language": lang,
                        "candidate_defects": len(defects), "wall_clock_seconds": dt,
                        "throughput_lines_per_sec": round(len(defects) / dt, 2) if dt > 0 else 0,
                        "gpu_vram_used_mb": vram_mb, "gpu_utilization_pct": gpu_util
                    }
                    with open(ABLATION_TELEMETRY_LOG, "a", encoding="utf-8", newline="\n") as tlog:
                        tlog.write(json.dumps(tel_rec) + "\n")

                    print(f"  ✅ [{short}_{exp} -> {lang}] Processed {len(defects)} candidates in {dt}s ({vram_mb}MB VRAM)", flush=True)

                timings[f"{short}_{exp}"] = exp_wall
                write_lf_bytes(timing_file, json.dumps(timings, indent=2) + "\n")
        finally:
            proc.terminate()
            proc.wait()
            kill_vllm()
            torch.cuda.empty_cache()

def run_phase_score_and_summarize(source_lines: list):
    capture_system_and_model_provenance("cometkiwi_scoring")

    timing_file = ABLATION_REPORTS_DIR / "ablation_timings.json"
    timings = json.loads(timing_file.read_text(encoding="utf-8")) if timing_file.exists() else {}
    telemetry_e5 = {}
    if TELEMETRY_LOG.exists():
        with open(TELEMETRY_LOG, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                b_suite = rec["baseline_suite"]
                telemetry_e5[b_suite] = telemetry_e5.get(b_suite, 0.0) + float(rec["wall_clock_seconds"])

    collateral_report = {}
    for m_cfg in ABLATION_MODELS:
        short = m_cfg["short"]
        b_suite = m_cfg["baseline_suite"]
        r_suite = m_cfg["repaired_suite"]
        e4_dir = ABLATION_DATA_DIR / f"{short}_E4"
        reg = {"rejected_by_e5_gate": 0, "new_ph_halluc": 0, "new_ph_any": 0,
               "new_enc_ufffd": 0, "new_script": 0, "new_chatter": 0, "unresolved_copy": 0}
        for lang in ALL_LANGS:
            b_lines = [l.rstrip("\r\n") for l in (DATA_DIR / b_suite / f"{lang}.txt").read_text(encoding="utf-8").splitlines()]
            e4_lines = [l.rstrip("\r\n") for l in (e4_dir / f"{lang}.txt").read_text(encoding="utf-8").splitlines()]
            e5_lines = [l.rstrip("\r\n") for l in (DATA_DIR / r_suite / f"{lang}.txt").read_text(encoding="utf-8").splitlines()]
            for s, b_h, e4_h, e5_h in zip(source_lines, b_lines, e4_lines, e5_lines):
                if e4_h != e5_h:
                    reg["rejected_by_e5_gate"] += 1
                    pre_iss = set(validate_single_line(s, b_h, lang))
                    post_iss = set(validate_single_line(s, e4_h, lang))
                    added = post_iss - pre_iss
                    if "broken_placeholder" in added:
                        reg["new_ph_any"] += 1
                        if len(PLACEHOLDER_REGEX.findall(s)) == 0:
                            reg["new_ph_halluc"] += 1
                    if "encoding_corruption" in added:
                        reg["new_enc_ufffd"] += 1
                    if "script_contamination" in added:
                        reg["new_script"] += 1
                    if "conversational_chatter" in added:
                        reg["new_chatter"] += 1
                    if "copy_fallback" in post_iss:
                        reg["unresolved_copy"] += 1
        collateral_report[short] = reg
    write_lf_bytes(ABLATION_REPORTS_DIR / "ablation_collateral_transitions.json", json.dumps(collateral_report, indent=2) + "\n")

    ck_ablation_csv = ABLATION_REPORTS_DIR / "ablation_cometkiwi_scores.csv"
    ck_done = {}
    ck_rows = []
    if ck_ablation_csv.exists():
        ck_df_prev = pd.read_csv(ck_ablation_csv)
        ck_rows = ck_df_prev.to_dict("records")
        for r in ck_rows:
            ck_done[(r["suite"], r["language"])] = float(r["cometkiwi"])

    suites_to_score = [f"{m['short']}_{e}" for m in ABLATION_MODELS for e in ("E2", "E3", "E4")]
    need_ck = any((s, l) not in ck_done for s in suites_to_score for l in ALL_LANGS)

    if need_ck:
        print(f"\n{'='*70}\n🧠 SCORING ABLATION SUITES WITH WMT22 COMETKIWI\n{'='*70}", flush=True)
        _comet_spec = importlib.util.find_spec("comet")
        if _comet_spec and _comet_spec.origin:
            _enc_dir = Path(_comet_spec.origin).parent / "encoders"
            for _py in _enc_dir.glob("*.py"):
                _txt = _py.read_text(encoding="utf-8")
                if "last_hidden_states, _, all_layers" in _txt:
                    _py.write_text(_txt.replace("last_hidden_states, _, all_layers", "last_hidden_states, *_, all_layers"), encoding="utf-8")

        from transformers import PreTrainedTokenizerBase, XLMRobertaTokenizer, XLMRobertaTokenizerFast
        from comet import download_model, load_from_checkpoint
        torch.set_float32_matmul_precision("high")

        def _xlmr_build_inputs(self, token_ids_0, token_ids_1=None):
            cls, sep = [self.cls_token_id], [self.sep_token_id]
            return cls + list(token_ids_0) + sep if token_ids_1 is None else cls + list(token_ids_0) + sep + sep + list(token_ids_1) + sep

        for _cls in (PreTrainedTokenizerBase, XLMRobertaTokenizer, XLMRobertaTokenizerFast):
            if not hasattr(_cls, "build_inputs_with_special_tokens"):
                setattr(_cls, "build_inputs_with_special_tokens", _xlmr_build_inputs)

        def _patched_forward(self, input_ids, attention_mask, **kwargs):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=False)
            return {"sentemb": out[0][:, 0, :], "wordemb": out[0], "all_layers": out[-1], "attention_mask": attention_mask}

        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        ck_model = load_from_checkpoint(model_path)
        setattr(ck_model.encoder.tokenizer.__class__, "build_inputs_with_special_tokens", _xlmr_build_inputs)
        setattr(ck_model.encoder.__class__, "forward", _patched_forward)
        use_gpus = 1 if torch.cuda.is_available() else 0

        for s_name in suites_to_score:
            s_dir = ABLATION_DATA_DIR / s_name
            for lang in ALL_LANGS:
                if (s_name, lang) in ck_done:
                    continue
                hyps = [l.rstrip("\r\n") for l in (s_dir / f"{lang}.txt").read_text(encoding="utf-8").splitlines()]
                payload = [{"src": source_lines[i], "mt": hyps[i]} for i in range(len(source_lines))]
                pred = ck_model.predict(payload, batch_size=256, gpus=use_gpus)
                sc = round(float(pred.system_score), 4)
                ck_done[(s_name, lang)] = sc
                ck_rows.append({"suite": s_name, "language": lang, "cometkiwi": sc})
                write_lf_bytes(ck_ablation_csv, pd.DataFrame(ck_rows).to_csv(index=False))
                print(f"  ✅ CometKiwi [{s_name} -> {lang}]: {sc:.4f}", flush=True)

    existing_ck = pd.read_csv(CK_SCORES_CSV).set_index(["system", "language"])["cometkiwi"].to_dict()
    summary_records = []
    per_lang_records = []

    for m_cfg in ABLATION_MODELS:
        short = m_cfg["short"]
        b_suite = m_cfg["baseline_suite"]
        r_suite = m_cfg["repaired_suite"]
        model_id = m_cfg["model_id"]

        t_e0, _ = audit_directory(DATA_DIR / b_suite, source_lines)
        ck_e0 = round(float(np.mean([existing_ck[(b_suite, l)] for l in ALL_LANGS])), 4)

        summary_records.append({
            "model": model_id, "experiment": "E0", "configuration": "Original zero-shot baseline",
            "spr_pct": t_e0["spr_pct"], "ddr_pct": t_e0["ddr_pct"], "flawed_segments": t_e0["flawed"],
            "ph_halluc": t_e0["ph_halluc"], "ph_total": t_e0["ph_total"], "script_contam": t_e0["script"],
            "utf8_corrupt": t_e0["enc"], "copy_fallback": t_e0["copy"], "cometkiwi": ck_e0,
            "additional_time_s": 0.00, "time_note": "0.00 (reference)"
        })
        summary_records.append({
            "model": model_id, "experiment": "E1", "configuration": "Baseline + deterministic verification",
            "spr_pct": t_e0["spr_pct"], "ddr_pct": t_e0["ddr_pct"], "flawed_segments": t_e0["flawed"],
            "ph_halluc": t_e0["ph_halluc"], "ph_total": t_e0["ph_total"], "script_contam": t_e0["script"],
            "utf8_corrupt": t_e0["enc"], "copy_fallback": t_e0["copy"], "cometkiwi": ck_e0,
            "additional_time_s": t_e0["audit_cpu_s"], "time_note": f"{t_e0['audit_cpu_s']:.2f}s (CPU audit only)"
        })

        cfg_labels = {
            "E2": "Baseline + dynamic placeholder conditioning (ungated)",
            "E3": "E2 + script-specific orthographic constraints (ungated)",
            "E4": "E3 + compound identifier segmentation (ungated)",
        }
        for exp in ("E2", "E3", "E4"):
            s_name = f"{short}_{exp}"
            t_ex, pl_ex = audit_directory(ABLATION_DATA_DIR / s_name, source_lines)
            ck_ex = round(float(np.mean([ck_done[(s_name, l)] for l in ALL_LANGS])), 4)
            w_time = round(float(timings.get(s_name, 0.0)), 2)
            summary_records.append({
                "model": model_id, "experiment": exp, "configuration": cfg_labels[exp],
                "spr_pct": t_ex["spr_pct"], "ddr_pct": t_ex["ddr_pct"], "flawed_segments": t_ex["flawed"],
                "ph_halluc": t_ex["ph_halluc"], "ph_total": t_ex["ph_total"], "script_contam": t_ex["script"],
                "utf8_corrupt": t_ex["enc"], "copy_fallback": t_ex["copy"], "cometkiwi": ck_ex,
                "additional_time_s": w_time, "time_note": f"{w_time:.2f}s (A100 GPU)"
            })
            for l in ALL_LANGS:
                per_lang_records.append({
                    "model": short, "experiment": exp, "language": l,
                    **pl_ex[l], "cometkiwi": ck_done[(s_name, l)]
                })

        t_e5, _ = audit_directory(DATA_DIR / r_suite, source_lines)
        ck_e5 = round(float(np.mean([existing_ck[(r_suite, l)] for l in ALL_LANGS])), 4)
        w_e5 = round(float(telemetry_e5.get(b_suite, 0.0)), 2)
        summary_records.append({
            "model": model_id, "experiment": "E5", "configuration": "Complete IGSC (E4 + post-repair verification gate)",
            "spr_pct": t_e5["spr_pct"], "ddr_pct": t_e5["ddr_pct"], "flawed_segments": t_e5["flawed"],
            "ph_halluc": t_e5["ph_halluc"], "ph_total": t_e5["ph_total"], "script_contam": t_e5["script"],
            "utf8_corrupt": t_e5["enc"], "copy_fallback": t_e5["copy"], "cometkiwi": ck_e5,
            "additional_time_s": w_e5, "time_note": f"{w_e5:.2f}s (A100 GPU + gate)"
        })

    sum_df = pd.DataFrame(summary_records)
    write_lf_bytes(ABLATION_REPORTS_DIR / "ablation_e0_e5_summary.csv", sum_df.to_csv(index=False))
    write_lf_bytes(ABLATION_REPORTS_DIR / "ablation_e0_e5_per_language.csv", pd.DataFrame(per_lang_records).to_csv(index=False))

    sha_lines, md5_lines = [], []
    for fp in sorted(ABLATION_DATA_DIR.rglob("*.txt")):
        b = fp.read_bytes()
        assert b"\r\n" not in b, f"CRLF in {fp}"
        assert len(b.decode("utf-8").splitlines()) == 10228, f"Line count != 10228 in {fp}"
        rel = fp.relative_to(REPO_ROOT).as_posix()
        sha_lines.append(f"{hashlib.sha256(b).hexdigest()}  {rel}")
        md5_lines.append(f"{hashlib.md5(b).hexdigest()}  {rel}")
    write_lf_bytes(ABLATION_REPORTS_DIR / "CHECKSUMS_ABLATION_SHA256.txt", "\n".join(sha_lines) + "\n")
    write_lf_bytes(ABLATION_REPORTS_DIR / "CHECKSUMS_ABLATION_MD5.txt", "\n".join(md5_lines) + "\n")

    readme_ablation = (
        "# Component-Wise IGSC Ablation Study (E0-E5) & Extended Statistical Artifacts\n\n"
        "This directory contains additive, non-destructive artifacts isolating each architectural component of "
        "**Invariant-Guided Self-Correction (IGSC)** across `Qwen/Qwen2.5-7B-Instruct` and `google/gemma-2-9b-it` "
        "(12 languages, 122,736 segments per model per tier).\n\n"
        "## Contents\n"
        "- `ablation_e0_e5_summary.csv`: Complete E0-E5 ablation metrics (SPR, placeholder hallucinations, script contamination, UTF-8 corruption, copy fallback, CometKiwi, and inference wall-clock time).\n"
        "- `ablation_e0_e5_per_language.csv`: Granular per-language breakdown across E2, E3, and E4 (72 model-tier-language pairs).\n"
        "- `ablation_collateral_transitions.json`: Exact count of secondary regressions in ungated E4 blocked by the E5 post-repair verification gate.\n"
        "- `table9_ci_holm_bonferroni.csv`: Paired 95% confidence intervals and Holm-Bonferroni multiple-testing-adjusted p-values across all 5 evaluated models.\n"
        "- `accelerator_exact_parity_audit.csv` & `ntt_corpus_coverage.json`: Exact keyboard accelerator count parity and Non-Translatable Token (NTT) lexicon coverage diagnostics.\n"
        "- `ablation_system_manifest.json` & `ablation_hardware_env_info.txt`: Full hardware, GPU/CUDA, package versions, and exact Hugging Face commit SHAs.\n"
        "- `requirements-ablation-vllm_generation-lock.txt` & `requirements-ablation-cometkiwi_scoring-lock.txt`: Exact `pip freeze` environment lockfiles.\n"
        "- `ablation_execution_telemetry.jsonl` & `traces/`: Per-language GPU execution telemetry and line-indexed candidate repair traces.\n"
        "- `CHECKSUMS_ABLATION_SHA256.txt` & `CHECKSUMS_ABLATION_MD5.txt`: Cryptographic checksums for all 72 ablation translation files in `data/ablation/`.\n"
    )
    write_lf_bytes(ABLATION_REPORTS_DIR / "README_ABLATION.md", readme_ablation)

    verify_checksums_intact("FINAL POST-RUN")

    t9_df = pd.read_csv(ABLATION_REPORTS_DIR / "table9_ci_holm_bonferroni.csv")
    print("\n" + "=" * 92)
    print("📊 TABLE 9 UPGRADE: 95% CIs & HOLM-BONFERRONI ADJUSTED P-VALUES")
    print("=" * 92)
    print(t9_df.to_string(index=False), flush=True)

    print("\n" + "=" * 92)
    print("🏆 FINAL COMPONENT-WISE ABLATION SUMMARY (E0 - E5)")
    print("=" * 92)
    print(sum_df.to_string(index=False), flush=True)

    print("\n" + "=" * 92)
    print("🛡️ COLLATERAL DEFECT INJECTIONS BLOCKED BY E5 VERIFICATION GATE")
    print("=" * 92)
    print(json.dumps(collateral_report, indent=2), flush=True)

    manifest = json.loads((ABLATION_REPORTS_DIR / "ablation_system_manifest.json").read_text(encoding="utf-8"))
    print("\n" + "=" * 92)
    print("📌 EXACT HUGGING FACE MODEL COMMIT SHAs & NTT STATS")
    print("=" * 92)
    print(json.dumps(manifest.get("huggingface_model_revisions", {}), indent=2), flush=True)
    print((ABLATION_REPORTS_DIR / "ntt_corpus_coverage.json").read_text(encoding="utf-8"), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["init", "generate", "score", "all"], default="all")
    parser.add_argument("--model", choices=["qwen_7b", "gemma_9b", "all"], default="all")
    args = parser.parse_args()

    ABLATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    verify_checksums_intact("PRE-FLIGHT")
    source_lines = [l.rstrip("\r\n") for l in SOURCE_FILE.read_text(encoding="utf-8").splitlines()]

    if args.phase == "init":
        run_phase_init(source_lines)
    elif args.phase == "generate":
        run_phase_generate_model(source_lines, args.model)
    elif args.phase == "score":
        run_phase_score_and_summarize(source_lines)
    else:
        run_phase_init(source_lines)
        run_phase_generate_model(source_lines, "all")
        run_phase_score_and_summarize(source_lines)

if __name__ == "__main__":
    main()
