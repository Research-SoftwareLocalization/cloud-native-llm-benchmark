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
