# Cloud-Native LLM Benchmark Suite

## Overview
This repository provides a standardized benchmarking framework for evaluating open-weights and commercial Large Language Models (LLMs) on high-throughput software localization and translation tasks across 12 typologically diverse languages.

## Evaluated Model Suite
1. **Commercial APIs:**
   - OpenAI **GPT-4o-mini** (`baseline_gpt4o`) — Commercial API baseline
2. **Open-Weights Frontier (Local vLLM Inference):**
   - **Gemma-2-9B-It** (`baseline_gemma_9b`) — Google instruction-tuned baseline
   - **Qwen-2.5-7B-Instruct** (`baseline_qwen_7b`) — Multilingual and Indic performance baseline
   - **Mistral-Nemo-Instruct-2407** (`baseline_mistral_nemo`) — European and structural formatting baseline

## Evaluation Methodology
- **Corpus:** 10,228 software UI localization strings from the KDE desktop environment.
- **Target Languages:** Arabic (ar), Bengali (bn), German (de), Spanish (es), French (fr), Hindi (hi), Japanese (ja), Marathi (mr), Punjabi (pa), Tamil (ta), Telugu (te), and Mandarin Chinese (zh).
- **Metrics:** BLEU, chrF++, and structural artifact preservation parity (& accelerators, % variables, camelCase identifiers).
- **Parity Enforcement:** Strict 1:1 line sequence mapping with automated empty-line preservation and Unix LF normalization.

## Execution Environment & Hardware Audit
Experiments are executed across heterogeneous cloud computing tiers (CPU runtimes for commercial APIs and L4/A100 GPU runtimes for open-weights vLLM inference). Complete historical hardware configurations, execution dates, and dependency snapshots are meticulously maintained in `hardware-env-info.txt` and `requirements-current.txt` to ensure strict reproducibility.

## Repository Structure
- `data/`: Contains source strings and system translation outputs for all evaluated models (`baseline_gpt4o`, `baseline_gemma_9b`, `baseline_qwen_7b`, `baseline_mistral_nemo`).
- `scripts/`: Automated execution, evaluation, and logging pipelines.
- `prompts/`: Standardized zero-shot system prompts.
- `findings/`: Experimental run logs and evaluation metrics.

## License
Distributed under the MIT License. See `LICENSE` for details.
