# Benchmarking Large Language Models for Cloud-Native Software Localization: Evaluating Generative AI as Intelligent Translation Agents

This repository contains the empirical benchmark suite, 10,228-string cloud-native software UI corpus (OPUS KDE4), deterministic compiler-invariant audit framework, and compiler-guided agentic self-repair results across **5 frontier LLMs** and **12 typologically diverse languages** (120 localization files; 1,227,360 evaluated strings across baseline and repaired tiers).

## Evaluated Model Suites (10 Suites: 5 Raw Baselines + 5 Self-Repaired)

### 1. Commercial Cloud API Tier
- **OpenAI GPT-4o-mini (`data/baseline_gpt4o/` $\to$ `data/repaired_gpt4o/`)**: Evaluated via REST API ($T=0.0$). Defect rate reduced from **0.95% (1,169)** to **0.22% (267)** (**-91.2%** across non-Latin scripts).

### 2. Open-Weights Local vLLM Tier (NVIDIA A100-SXM4-80GB)
- **Meta-Llama-3.1-8B-Instruct (`data/baseline_llama_8b/` $\to$ `data/repaired_llama_8b/`)**: Defect rate reduced from **2.45% (3,008)** to **0.12% (149)** (**-95.0%**).
- **Mistral-Nemo-Instruct-2407 (`data/baseline_mistral_nemo/` $\to$ `data/repaired_mistral_nemo/`)**: Defect rate reduced from **8.87% (10,883)** to **0.25% (305)** (**-97.2%**).
- **Google Gemma-2-9B-It (`data/baseline_gemma_9b/` $\to$ `data/repaired_gemma_9b/`)**: Defect rate reduced from **17.83% (21,888)** to **0.81% (996)** (**-95.4%**), rescuing severe low-resource Indic localization collapse.
- **Qwen-2.5-7B-Instruct (`data/baseline_qwen_7b/` $\to$ `data/repaired_qwen_7b/`)**: Defect rate reduced from **5.11% (6,266)** to **1.54% (1,887)** (**-69.9%**).

## Empirical Failure Mode Taxonomy & Headline Findings
1. **Instruction-Induced Artifact Hallucination**: Static system prompts mentioning format variables (`%1, %d`) cause 7B–12B open-weights models to hallucinate spurious `%1` placeholders into **11,932 plain-text UI strings**. Dynamic compiler-guided conditioning eliminates **98.9%** of these hallucinations (`11,932` $\to$ `130`).
2. **Programmatic Invariant Corruption**: Missing format specifiers (`%1`, `%d`, `{searchTerms}`) and keyboard accelerators (`&File`, including Unicode RTL/Indic bindings).
3. **Cross-Lingual Script Contamination & Digraphia**: Out-of-alphabet glyph leakage and Punjabi Shahmukhi/Gurmukhi script ambiguity.
4. **UTF-8 Byte-Fallback Corruption**: Multi-byte BPE token splitting (`\ufffd`) under repetition penalties.
5. **Passive Copy Fallback vs. Non-Translatable Tokens (NTTs)**: Distinguishing true untranslated leakage from mandatory hardware/code invariants (`SCSI`, `S-ATA`, `BIN2OCT(value)`, X11 color constants).

## Reproducibility & Provenance Artifacts
- `data/source/en.txt`: 10,228 English KDE UI strings (`SHA-256: 464d65825b81856c4cb600638fca87c3b3abc1692239679e1ec093677bfe281f`).
- `reports/pre_correction_checksums.sha256` & `reports/post_repair_checksums.sha256`: Cryptographic SHA-256 manifests for all 60 baseline and 60 repaired files.
- `reports/qc_audit_summary.md` & `reports/model_level_repair_summary.csv`: Complete 10-suite failure mode breakdown and before-vs-after repair tables.
- `reports/placeholder_error_decomposition.csv`: Decomposition of omitted vs. prompt-hallucinated `%1` specifiers.
- `reports/audit_findings_log.jsonl` & `reports/correction_trace_log.jsonl`: Line-by-line defect audit and repair trace logs.
- `reports/a100_system_manifest.json`, `reports/repair_execution_telemetry.jsonl`, & `hardware-env-info.txt`: Full hardware, VRAM, latency, and environment audit trails.
- `metrics/cometkiwi/`: Reference-free semantic quality estimation (`Unbabel/wmt22-cometkiwi-da`) scores and pivot matrices.


## Dual-Paradigm Evaluation: Deterministic Invariants + Reference-Free CometKiwi QE

All **10 suites (120 files, 1,227,360 strings)** evaluated with deterministic verification & WMT22 CometKiwi (`Unbabel/wmt22-cometkiwi-da`):

| model        |   n_languages |   mean_baseline_ck |   mean_repaired_ck |   mean_delta_ck |   paired_t_stat |   paired_t_p_value |   wilcoxon_p_value | significant_p05   |
|:-------------|--------------:|-------------------:|-------------------:|----------------:|----------------:|-------------------:|-------------------:|:------------------|
| gemma_9b     |            12 |             0.7395 |             0.7653 |          0.0259 |          2.1571 |        0.0539772   |        0.000488281 | False             |
| gpt4o        |            12 |             0.7877 |             0.7877 |         -0      |         -0.3192 |        0.755553    |        0.948242    | False             |
| llama_8b     |            12 |             0.7483 |             0.7531 |          0.0048 |          3.9198 |        0.00239381  |        0.000488281 | True              |
| mistral_nemo |            12 |             0.6872 |             0.7136 |          0.0264 |         10.5014 |        4.52466e-07 |        0.000488281 | True              |
| qwen_7b      |            12 |             0.7046 |             0.7069 |          0.0023 |          3.2052 |        0.0083765   |        0.000488281 | True              |

### Suite x Language CometKiwi Pivot (10 Suites x 12 Languages)

| system                |     ar |     bn |     de |     es |     fr |     hi |     ja |     mr |     pa |     ta |     te |     zh |   Mean |
|:----------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| baseline_gemma_9b     | 0.7383 | 0.7805 | 0.7757 | 0.779  | 0.7714 | 0.7874 | 0.7932 | 0.7105 | 0.5896 | 0.7551 | 0.6251 | 0.7676 | 0.7395 |
| baseline_gpt4o        | 0.7564 | 0.8107 | 0.7868 | 0.7863 | 0.7804 | 0.8064 | 0.8086 | 0.7824 | 0.8005 | 0.779  | 0.7799 | 0.7754 | 0.7877 |
| baseline_llama_8b     | 0.7023 | 0.7657 | 0.7601 | 0.766  | 0.7599 | 0.7746 | 0.7816 | 0.7303 | 0.7458 | 0.71   | 0.7309 | 0.7524 | 0.7483 |
| baseline_mistral_nemo | 0.6416 | 0.7163 | 0.7034 | 0.7164 | 0.7012 | 0.7195 | 0.7291 | 0.6632 | 0.6592 | 0.6488 | 0.6487 | 0.6988 | 0.6872 |
| baseline_qwen_7b      | 0.7104 | 0.7266 | 0.757  | 0.7721 | 0.7637 | 0.7061 | 0.7834 | 0.6844 | 0.6354 | 0.5652 | 0.5832 | 0.7676 | 0.7046 |
| repaired_gemma_9b     | 0.742  | 0.7897 | 0.778  | 0.7806 | 0.7717 | 0.7946 | 0.7996 | 0.7416 | 0.7122 | 0.7766 | 0.7268 | 0.7704 | 0.7653 |
| repaired_gpt4o        | 0.7557 | 0.811  | 0.7868 | 0.7856 | 0.7796 | 0.8066 | 0.8089 | 0.7826 | 0.8011 | 0.7791 | 0.78   | 0.7753 | 0.7877 |
| repaired_llama_8b     | 0.7058 | 0.78   | 0.7624 | 0.7683 | 0.7611 | 0.7775 | 0.7876 | 0.7331 | 0.7583 | 0.7116 | 0.7341 | 0.7575 | 0.7531 |
| repaired_mistral_nemo | 0.6783 | 0.74   | 0.7441 | 0.7511 | 0.7391 | 0.7354 | 0.7493 | 0.6868 | 0.6798 | 0.6647 | 0.6712 | 0.7237 | 0.7136 |
| repaired_qwen_7b      | 0.713  | 0.7277 | 0.7577 | 0.7725 | 0.7643 | 0.7097 | 0.7874 | 0.6863 | 0.6443 | 0.5654 | 0.5855 | 0.7684 | 0.7069 |

### Generated Figures
* `reports/pareto_reliability_vs_quality.png` — 2D Pareto frontier
* `reports/defect_rate_before_vs_after.png` — Defect rate reduction
* `reports/cometkiwi_before_vs_after.png` & `reports/cometkiwi_by_system.png` — Neural QE distributions
