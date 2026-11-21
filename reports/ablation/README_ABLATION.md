# Component-Wise IGSC Ablation Study (E0-E5) & Extended Statistical Artifacts

This directory contains additive, non-destructive artifacts isolating each architectural component of **Invariant-Guided Self-Correction (IGSC)** across `Qwen/Qwen2.5-7B-Instruct` and `google/gemma-2-9b-it` (12 languages, 122,736 segments per model per tier).

## Contents
- `ablation_e0_e5_summary.csv`: Complete E0-E5 ablation metrics (SPR, placeholder hallucinations, script contamination, UTF-8 corruption, copy fallback, CometKiwi, and inference wall-clock time).
- `ablation_e0_e5_per_language.csv`: Granular per-language breakdown across E2, E3, and E4 (72 model-tier-language pairs).
- `ablation_collateral_transitions.json`: Exact count of secondary regressions in ungated E4 blocked by the E5 post-repair verification gate.
- `table9_ci_holm_bonferroni.csv`: Paired 95% confidence intervals and Holm-Bonferroni multiple-testing-adjusted p-values across all 5 evaluated models.
- `accelerator_exact_parity_audit.csv` & `ntt_corpus_coverage.json`: Exact keyboard accelerator count parity and Non-Translatable Token (NTT) lexicon coverage diagnostics.
- `ablation_system_manifest.json` & `ablation_hardware_env_info.txt`: Full hardware, GPU/CUDA, package versions, and exact Hugging Face commit SHAs.
- `requirements-ablation-vllm_generation-lock.txt` & `requirements-ablation-cometkiwi_scoring-lock.txt`: Exact `pip freeze` environment lockfiles.
- `ablation_execution_telemetry.jsonl` & `traces/`: Per-language GPU execution telemetry and line-indexed candidate repair traces.
- `CHECKSUMS_ABLATION_SHA256.txt` & `CHECKSUMS_ABLATION_MD5.txt`: Cryptographic checksums for all 72 ablation translation files in `data/ablation/`.
- `five_model_gate_ablation_e4_vs_e5.csv`: Post-repair verification gate ablation (`E4` ungated vs. `E5` gated) across all 5 evaluated models (613,680 segments).
- `defect_subset_cometkiwi_by_model.csv` & `defect_subset_cometkiwi_by_category.csv`: Direct segment-level WMT22 CometKiwi evaluation on all 43,214 baseline-defective segments, empirically separating statistical dilution from encoder syntactic blindness.
- `native_expert_indic_validation_sample.csv`: Deterministic seed-42 stratified sample of Indic IGSC repairs across `bn`, `hi`, `mr`, `pa`, `ta`, and `te`.
