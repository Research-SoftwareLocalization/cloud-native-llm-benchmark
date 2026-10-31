# Benchmarking Large Language Models for Cloud-Native Software Localization: Evaluating Generative AI as Intelligent Translation Agents

This repository contains the empirical benchmark suite, 10,228-string cloud-native software corpus, and agentic self-correction guardrails for software localization across 12 typologically diverse languages.

## Evaluated Model Suites (5 Models)

### 1. Commercial API Baselines
- **OpenAI GPT-4o-mini (`baseline_gpt4o`)**: Frontier commercial API baseline evaluated via REST endpoint ($T=0.0$).

### 2. Open-Weights Frontier (Local vLLM Inference)
- **Meta-Llama-3.1-8B-Instruct (`baseline_llama_8b`)**: Frontier open-weights architecture evaluated for multilingual reasoning.
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

## Empirical Failure Mode Taxonomy
1. **Cross-Lingual Script Contamination**: Polysemous, context-deprived UI tokens activating out-of-alphabet subword clusters (e.g., Hanzi/Cyrillic glyphs in Arabic/Spanish targets).
2. **Conversational Persona Leakage**: RLHF-induced conversational preambles (*"Translation:"*, *"Note:"*).
3. **Programmatic Invariant Corruption**: Missing or corrupted dynamic format specifiers (`%1`, `%d`, `{searchTerms}`) and accelerator keys (`&File`).
4. **Passive Copy Fallback**: Defaulting to verbatim copying on domain-specific terminology.

## Evaluation Pipeline Status
- **Translations (12 languages x 5 models)**: Generated.
- **Quality Audit & Invariant Analysis**: Active.
- **Automated Metric Scoring (BLEU, chrF++, COMET-22)**: Pending quality audit verification and model-specific self-correction.

## Dataset Provenance
- **Source Corpus**: `data/source/en.txt` (10,228 real-world software UI strings from KDE/Linux desktop systems).
- **SHA-256 Checksum**: `464d65825b81856c4cb600638fca87c3b3abc1692239679e1ec093677bfe281f`
