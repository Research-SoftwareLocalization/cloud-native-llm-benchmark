# Benchmarking Large Language Models for Cloud-Native Software Localization

Empirical evaluation and agentic guardrail framework for software UI localization across 12 typologically diverse languages using an immutable 10,228-string corpus.

## Evaluated Model Suites

### 1. Commercial API Baselines
- **OpenAI GPT-4o-mini (`baseline_gpt4o`)**: Frontier commercial API baseline evaluated via REST endpoints.

### 2. Open-Weights Frontier (Local vLLM Inference)
- **Meta-Llama-3.1-8B-Instruct (`baseline_llama_8b`)**: Frontier open-weights multilingual and reasoning baseline.
- **Gemma-2-9B-It (`baseline_gemma_9b`)**: Google high-parameter instruction-tuned dense baseline.
- **Qwen-2.5-7B-Instruct (`baseline_qwen_7b`)**: Multilingual and low-resource Indic performance baseline.
- **Mistral-Nemo-Instruct-2407 (`baseline_mistral_nemo`)**: High-efficiency European and structural formatting baseline.

## Target Language Typology (12 Languages)
- **Semitic**: Arabic (`ar`)
- **Indo-Aryan (Indic)**: Bengali (`bn`), Hindi (`hi`), Marathi (`mr`), Punjabi (`pa`)
- **Dravidian (Indic)**: Tamil (`ta`), Telugu (`te`)
- **Germanic**: German (`de`)
- **Romance**: Spanish (`es`), French (`fr`)
- **Sino-Tibetan**: Simplified Chinese (`zh`)
- **Japonic**: Japanese (`ja`)

## Dataset Provenance
- **Source Corpus**: `data/source/en.txt` (10,228 real-world software UI strings from KDE/Linux desktop systems).
- **SHA-256 Checksum**: `464d65825b81856c4cb600638fca87c3b3abc1692239679e1ec093677bfe281f`
