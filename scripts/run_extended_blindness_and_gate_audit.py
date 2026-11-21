#!/usr/bin/env python3
# Extended Empirical Proofs & Diagnostics Suite:
#   1. Direct WMT22 CometKiwi evaluation on all 43,214 baseline-defective segments (5 models)
#   2. 5-Model Post-Repair Verification Gate Ablation (E4 ungated vs. E5 gated across all 5 models)
#   3. Bootstrap 95% CIs (B=10,000) + Paired Cohen's d_z + Holm-Bonferroni p-values (Table 9)
# Strictly non-destructive: all 121 core benchmark files remain bit-for-bit untouched.

import os
import sys
import importlib.abc
import importlib.util

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["USE_JAX"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_BLOCKED_BACKENDS = ("tensorflow", "tensorboard", "jax", "jaxlib", "flax", "keras")

class _PurePyTorchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in _BLOCKED_BACKENDS or fullname.startswith(tuple(p + "." for p in _BLOCKED_BACKENDS)):
            raise ModuleNotFoundError(f"Pure-PyTorch mode blocked unneeded backend: {fullname}")
        return None

_orig_find_spec = importlib.util.find_spec
def _safe_find_spec(name, package=None):
    if name in _BLOCKED_BACKENDS or name.startswith(tuple(p + "." for p in _BLOCKED_BACKENDS)):
        return None
    try:
        return _orig_find_spec(name, package)
    except (ModuleNotFoundError, ValueError):
        return None

sys.meta_path.insert(0, _PurePyTorchFinder())
importlib.util.find_spec = _safe_find_spec

import re
import json
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO_ROOT = Path("/content/cloud-native-llm-benchmark") if Path("/content/cloud-native-llm-benchmark").exists() else Path(".")
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
ABLATION_REPORTS_DIR = REPORTS_DIR / "ablation"
SOURCE_FILE = DATA_DIR / "source" / "en.txt"
TRACE_LOG = REPORTS_DIR / "correction_trace_log.jsonl"
CK_DELTA_CSV = REPO_ROOT / "metrics" / "cometkiwi" / "cometkiwi_before_after_delta.csv"

ALL_LANGS = ["ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"]
INDIC_LANGS = {"bn", "hi", "mr", "pa", "ta", "te"}
EURO_LANGS = {"de", "es", "fr"}

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

ALL_FIVE_MODELS = [
    ("gemma_9b", "baseline_gemma_9b", "repaired_gemma_9b", "google/gemma-2-9b-it"),
    ("mistral_nemo", "baseline_mistral_nemo", "repaired_mistral_nemo", "mistralai/Mistral-Nemo-Instruct-2407"),
    ("qwen_7b", "baseline_qwen_7b", "repaired_qwen_7b", "Qwen/Qwen2.5-7B-Instruct"),
    ("llama_8b", "baseline_llama_8b", "repaired_llama_8b", "meta-llama/Llama-3.1-8B-Instruct"),
    ("gpt4o", "baseline_gpt4o", "repaired_gpt4o", "gpt-4o-mini-2024-07-18"),
]

def read_unix_lines(filepath: Path):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\r\n") for line in f]

def write_lf_bytes(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_bytes(clean.encode("utf-8"))

def verify_checksums_intact(stage_label: str):
    for chk_path in [REPO_ROOT / "CHECKSUMS_SHA256.txt", ABLATION_REPORTS_DIR / "CHECKSUMS_ABLATION_SHA256.txt"]:
        if not chk_path.exists():
            continue
        lines = [ln.strip() for ln in chk_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines:
            exp_hash, rel = ln.split(None, 1)
            rel = rel.lstrip("*")
            act_hash = hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
            assert act_hash == exp_hash, f"SAFETY ABORT ({stage_label}): {rel} hash changed!"
        print(f"🔒 [{stage_label}] Verified {len(lines)}/{len(lines)} files in {chk_path.name}.", flush=True)

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

def load_cometkiwi_model():
    _comet_spec = _orig_find_spec("comet")
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
    return ck_model

def upgrade_table9_with_bootstrap_and_dz():
    df = pd.read_csv(CK_DELTA_CSV)
    model_order = ["mistral_nemo", "gemma_9b", "llama_8b", "qwen_7b", "gpt4o"]
    rng = np.random.default_rng(42)
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
        boots = rng.choice(d, size=(10000, L), replace=True).mean(axis=1)
        b_lo, b_hi = np.percentile(boots, [2.5, 97.5])
        cohens_dz = mean_d / sd_d if sd_d > 0 else 0.0
        t_stat, p_t = stats.ttest_rel(r, b)
        w_stat, p_w = stats.wilcoxon(d, zero_method="zsplit" if np.any(d == 0) else "wilcox")
        rows.append({
            "model": m, "mean_base_ck": round(float(np.mean(b)), 4),
            "mean_igsc_ck": round(float(np.mean(r)), 4), "mean_delta_ck": round(mean_d, 4),
            "sd_delta_ck": round(sd_d, 4), "cohens_dz": round(float(cohens_dz), 2),
            "ci95_low": round(float(ci_lo), 4), "ci95_high": round(float(ci_hi), 4),
            "boot_ci95_low": round(float(b_lo), 4), "boot_ci95_high": round(float(b_hi), 4),
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
    return res_df

def run_five_model_gate_ablation(source_lines: list):
    trace_history = {}
    with open(TRACE_LOG, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cand_raw = rec.get("repaired_translation")
            if cand_raw is not None:
                cand_clean = str(cand_raw).replace("\r", " ").replace("\n", " ").strip()
                if cand_clean:
                    trace_history.setdefault((rec["model"], rec["lang"], rec["line_idx"]), []).append(cand_clean)

    rows = []
    for short, b_suite, r_suite, model_id in ALL_FIVE_MODELS:
        e0_flawed = e4_flawed = e5_flawed = 0
        blocked_cands = new_script = new_enc = new_ph = new_chat = new_acc = total_new_inj = 0
        e4_script = e5_script = e4_enc = e5_enc = e4_ph = e5_ph = e4_copy = e5_copy = 0

        for lang in ALL_LANGS:
            b_lines = read_unix_lines(DATA_DIR / b_suite / f"{lang}.txt")
            e5_lines = read_unix_lines(DATA_DIR / r_suite / f"{lang}.txt")
            for idx, (s, b_h, e5_h) in enumerate(zip(source_lines, b_lines, e5_lines)):
                b_iss = validate_single_line(s, b_h, lang)
                e5_iss = validate_single_line(s, e5_h, lang)
                if b_iss:
                    e0_flawed += 1

                if not e5_iss:
                    e4_h = e5_h
                else:
                    cands = trace_history.get((b_suite, lang, idx), [])
                    rej = [c for c in cands if validate_single_line(s, c, lang)]
                    e4_h = rej[-1] if rej else e5_h

                e4_iss = validate_single_line(s, e4_h, lang)
                if e4_iss:
                    e4_flawed += 1
                if e5_iss:
                    e5_flawed += 1

                if "script_contamination" in e4_iss: e4_script += 1
                if "script_contamination" in e5_iss: e5_script += 1
                if "encoding_corruption" in e4_iss: e4_enc += 1
                if "encoding_corruption" in e5_iss: e5_enc += 1
                if "broken_placeholder" in e4_iss: e4_ph += 1
                if "broken_placeholder" in e5_iss: e5_ph += 1
                if "copy_fallback" in e4_iss: e4_copy += 1
                if "copy_fallback" in e5_iss: e5_copy += 1

                if e4_h != e5_h:
                    blocked_cands += 1
                    added = set(e4_iss) - set(b_iss)
                    total_new_inj += len(added)
                    if "script_contamination" in added: new_script += 1
                    if "encoding_corruption" in added: new_enc += 1
                    if "broken_placeholder" in added: new_ph += 1
                    if "conversational_chatter" in added: new_chat += 1
                    if "broken_accelerator" in added: new_acc += 1

        rows.append({
            "model": model_id,
            "e0_baseline_flawed": e0_flawed,
            "e4_ungated_flawed": e4_flawed,
            "e5_gated_flawed": e5_flawed,
            "rejected_candidates_blocked_by_gate": blocked_cands,
            "newly_injected_defect_instances_blocked": total_new_inj,
            "new_script_leaks_blocked": new_script,
            "new_utf8_ufffd_blocked": new_enc,
            "new_placeholder_bugs_blocked": new_ph,
            "new_chatter_blocked": new_chat,
            "e4_script_contam": e4_script,
            "e5_script_contam": e5_script,
            "e4_utf8_corrupt": e4_enc,
            "e5_utf8_corrupt": e5_enc,
            "e4_ph_errors": e4_ph,
            "e5_ph_errors": e5_ph,
            "e4_copy_fallback": e4_copy,
            "e5_copy_fallback": e5_copy,
        })
    gate_df = pd.DataFrame(rows)
    write_lf_bytes(ABLATION_REPORTS_DIR / "five_model_gate_ablation_e4_vs_e5.csv", gate_df.to_csv(index=False))
    return gate_df


def run_phase_subset_qe(source_lines: list):
    t9_df = upgrade_table9_with_bootstrap_and_dz()
    gate_df = run_five_model_gate_ablation(source_lines)

    defect_records = []
    for short, b_suite, r_suite, model_id in ALL_FIVE_MODELS:
        for lang in ALL_LANGS:
            b_lines = read_unix_lines(DATA_DIR / b_suite / f"{lang}.txt")
            r_lines = read_unix_lines(DATA_DIR / r_suite / f"{lang}.txt")
            for idx, (s, b_h, r_h) in enumerate(zip(source_lines, b_lines, r_lines)):
                b_iss = validate_single_line(s, b_h, lang)
                if not b_iss:
                    continue
                r_iss = validate_single_line(s, r_h, lang)
                sp = sorted(PLACEHOLDER_REGEX.findall(s))
                hp = sorted(PLACEHOLDER_REGEX.findall(b_h))
                ph_sub = "none"
                if "broken_placeholder" in b_iss:
                    if len(sp) > 0 and len(hp) == 0:
                        ph_sub = "ph_omission"
                    elif len(sp) == 0 and len(hp) > 0:
                        ph_sub = "ph_hallucination"
                    else:
                        ph_sub = "ph_mutation"

                if "broken_accelerator" in b_iss:
                    cat = "broken_accelerator"
                elif "broken_placeholder" in b_iss:
                    cat = ph_sub
                elif "encoding_corruption" in b_iss:
                    cat = "utf8_ufffd_corruption"
                elif "script_contamination" in b_iss:
                    cat = "script_digraphic_punjabi" if lang == "pa" else "script_foreign_leak"
                elif "conversational_chatter" in b_iss:
                    cat = "conversational_chatter"
                else:
                    if lang in INDIC_LANGS:
                        cat = "copy_fallback_indic"
                    elif lang in EURO_LANGS:
                        cat = "copy_fallback_european_latin"
                    else:
                        cat = "copy_fallback_ar_ja_zh"

                defect_records.append({
                    "model": short, "model_id": model_id, "lang": lang, "line_idx": idx,
                    "src": s, "base_mt": b_h, "igsc_mt": r_h,
                    "repaired_success": (len(r_iss) == 0),
                    "category": cat
                })

    print(f"\n🧠 Scoring all {len(defect_records)} baseline-defective segments (before & after IGSC) on A100 with WMT22 CometKiwi...", flush=True)
    ck_model = load_cometkiwi_model()
    use_gpus = 1 if torch.cuda.is_available() else 0

    base_payload = [{"src": r["src"], "mt": r["base_mt"]} for r in defect_records]
    igsc_payload = [{"src": r["src"], "mt": r["igsc_mt"]} for r in defect_records]

    try:
        pred_base = ck_model.predict(base_payload, batch_size=256, gpus=use_gpus, num_workers=0, progress_bar=False)
        pred_igsc = ck_model.predict(igsc_payload, batch_size=256, gpus=use_gpus, num_workers=0, progress_bar=False)
    except TypeError:
        pred_base = ck_model.predict(base_payload, batch_size=256, gpus=use_gpus)
        pred_igsc = ck_model.predict(igsc_payload, batch_size=256, gpus=use_gpus)

    for i, rec in enumerate(defect_records):
        rec["ck_base"] = float(pred_base.scores[i])
        rec["ck_igsc"] = float(pred_igsc.scores[i])
        rec["ck_delta"] = rec["ck_igsc"] - rec["ck_base"]

    seg_df = pd.DataFrame(defect_records)

    mod_rows = []
    for short, b_suite, r_suite, model_id in ALL_FIVE_MODELS:
        m_all = seg_df[seg_df["model"] == short]
        m_rep = m_all[m_all["repaired_success"]]
        rho = len(m_all) / 122736.0
        delta_qe_all = float(m_all["ck_delta"].mean())
        delta_qe_rep = float(m_rep["ck_delta"].mean()) if len(m_rep) > 0 else 0.0
        mod_rows.append({
            "model": model_id,
            "total_defective_D": len(m_all),
            "verified_repaired_count": len(m_rep),
            "defect_density_rho_pct": round(rho * 100.0, 2),
            "subset_D_base_ck": round(float(m_all["ck_base"].mean()), 4),
            "subset_D_igsc_ck": round(float(m_all["ck_igsc"].mean()), 4),
            "subset_D_delta_QE": round(delta_qe_all, 4),
            "repaired_only_base_ck": round(float(m_rep["ck_base"].mean()), 4),
            "repaired_only_igsc_ck": round(float(m_rep["ck_igsc"].mean()), 4),
            "repaired_only_delta_QE": round(delta_qe_rep, 4),
            "predicted_corpus_shift_rho_x_delta": round(rho * delta_qe_all, 4),
        })
    mod_df = pd.DataFrame(mod_rows)
    write_lf_bytes(ABLATION_REPORTS_DIR / "defect_subset_cometkiwi_by_model.csv", mod_df.to_csv(index=False))

    cat_rows = []
    for cat_name, grp in seg_df.groupby("category"):
        rep_grp = grp[grp["repaired_success"]]
        cat_rows.append({
            "defect_category": cat_name,
            "total_segments": len(grp),
            "repaired_segments": len(rep_grp),
            "flawed_mean_ck_all": round(float(grp["ck_base"].mean()), 4),
            "igsc_mean_ck_all": round(float(grp["ck_igsc"].mean()), 4),
            "mean_delta_ck_all": round(float(grp["ck_delta"].mean()), 4),
            "repaired_only_flawed_ck": round(float(rep_grp["ck_base"].mean()), 4) if len(rep_grp) > 0 else None,
            "repaired_only_igsc_ck": round(float(rep_grp["ck_igsc"].mean()), 4) if len(rep_grp) > 0 else None,
            "repaired_only_delta_ck": round(float(rep_grp["ck_delta"].mean()), 4) if len(rep_grp) > 0 else None,
            "pct_flawed_scored_above_075": round(100.0 * float((grp["ck_base"] >= 0.75).mean()), 1),
        })
    cat_df = pd.DataFrame(cat_rows).sort_values("total_segments", ascending=False)
    write_lf_bytes(ABLATION_REPORTS_DIR / "defect_subset_cometkiwi_by_category.csv", cat_df.to_csv(index=False))

    verify_checksums_intact("POST-SUBSET-QE")

    print("\n" + "=" * 96)
    print("🔬 1. EMPIRICAL PROOF OF PROPOSITION 1: SUBSET D COMETKIWI BY MODEL (NO CORPUS DILUTION)")
    print("=" * 96)
    print(mod_df.to_string(index=False), flush=True)

    print("\n" + "=" * 96)
    print("🎯 2. DIRECT SYNTACTIC BLINDNESS BY DEFECT CATEGORY (SCORED ON REPAIRED SEGMENTS)")
    print("=" * 96)
    print(cat_df.to_string(index=False), flush=True)

    print("\n" + "=" * 96)
    print("🛡️ 3. FIVE-MODEL POST-REPAIR VERIFICATION GATE ABLATION (E4 UNGATED vs E5 GATED)")
    print("=" * 96)
    print(gate_df.to_string(index=False), flush=True)

    print("\n" + "=" * 96)
    print("📊 4. UPGRADED TABLE 9 WITH BOOTSTRAP 95% CIs (B=10,000) & COHEN'S d_z")
    print("=" * 96)
    print(t9_df.to_string(index=False), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["subset_qe"], default="subset_qe")
    args = parser.parse_args()
    verify_checksums_intact("PRE-FLIGHT")
    source_lines = read_unix_lines(SOURCE_FILE)
    run_phase_subset_qe(source_lines)

if __name__ == "__main__":
    main()
