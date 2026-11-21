# End-to-End Reproducibility Guide

This repository contains the canonical source corpus, all 120 baseline and IGSC-repaired translation files (`1,227,360` localized segments), diagnostic logs (`46,818` defect trace records), CometKiwi QE tables, and execution scripts needed to reproduce every table and figure in the paper.

## 1. Byte-Level Corpus & Checksum Verification
All 121 corpus files (`data/source/en.txt` + 60 baseline files + 60 repaired files) contain exactly `10,228` non-empty lines terminated with strict Unix `LF` (`\n`). Verify cryptographic integrity on Linux/macOS:

```bash
sha256sum -c CHECKSUMS.txt
sha256sum -c CHECKSUMS_SHA256.txt
md5sum -c CHECKSUMS_MD5.txt
```

## 2. Reproducing Zero-Shot Baseline Translations
- **Open-weights models (`Llama-3.1-8B-Instruct`, `Gemma-2-9B-IT`, `Mistral-Nemo-Instruct-2407`, `Qwen2.5-7B-Instruct`):**
  ```bash
  pip install -r requirements-a100-lock.txt
  python scripts/run_all_open_weights.py
  ```
- **Commercial API (`gpt-4o-mini`, frozen snapshot in `data/baseline_gpt4o/`):**
  ```bash
  export OPENAI_API_KEY="your-key"
  python scripts/run_all_gpt4o.py
  ```
*(Note: `google/gemma-2-9b-it`, `meta-llama/Llama-3.1-8B-Instruct`, and `Unbabel/wmt22-cometkiwi-da` are gated Hugging Face repositories requiring an authenticated `HF_TOKEN`.)*

## 3. Running Deterministic Invariant Verification (6 Invariants)
Audit all suites across script purity (`I_script`), placeholder multiset parity (`I_ph`), keyboard accelerators (`I_acc`), UTF-8 decoding integrity (`I_enc`), conversational framing (`I_chat`), and NTT-whitelisted non-echoing (`I_ntt`):

```bash
python scripts/audit_and_capture_findings.py
```
Outputs: `reports/qc_audit_summary.md`, `reports/audit_findings_log.jsonl`, and `reports/placeholder_error_decomposition.csv`.

## 4. Running Invariant-Guided Self-Correction (IGSC)
Execute closed-loop, defect-conditioned self-repair ($T = 0.0$) with strict post-verification gating:

```bash
# Open-weights vLLM self-repair on NVIDIA A100
python scripts/run_open_weights_repair.py

# Commercial API self-repair
python scripts/run_agentic_repair.py
```
Outputs: `data/repaired_*/{lang}.txt`, `reports/correction_trace_log.jsonl`, `reports/repair_execution_telemetry.jsonl`, and `reports/model_level_repair_summary.csv`.

## 5. Computing Reference-Free Neural QE (CometKiwi) & Statistical Tests
Score all 120 suites (`1,227,360` segments) with `Unbabel/wmt22-cometkiwi-da` and run paired Student's $t$-tests and Wilcoxon signed-rank tests:

```bash
pip install -r requirements-metrics.txt
python scripts/compute_cometkiwi.py
python scripts/analysis.py
python scripts/generate_publication_artifacts.py
```
Outputs: `metrics/cometkiwi/cometkiwi_pivot.csv`, `metrics/cometkiwi/cometkiwi_before_after_delta.csv`, and `reports/statistical_significance_summary.csv`.

## Component-Wise IGSC Ablation (E0-E5) & Subset-Level Blindness Diagnostics

- **Component-Wise Ablation Runner (`E0`-`E5`):** `python scripts/run_ablation_e0_e5.py --phase all` (outputs in `data/ablation/` and `reports/ablation/ablation_e0_e5_summary.csv`, checksums in `reports/ablation/CHECKSUMS_ABLATION_SHA256.txt`).
- **5-Model Verification-Gate & Active-Defect Subset QE Audit:** `python scripts/run_extended_blindness_and_gate_audit.py --phase subset_qe` (outputs `five_model_gate_ablation_e4_vs_e5.csv`, `defect_subset_cometkiwi_by_model.csv`, `defect_subset_cometkiwi_by_category.csv`, and `table9_ci_holm_bonferroni.csv`).
