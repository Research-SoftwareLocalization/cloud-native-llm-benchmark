[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24280788.svg)](https://doi.org/10.5281/zenodo.24280788)

# Invariant-Guided Self-Correction for Cloud-Native Software Localization Pipelines: A 1.22-Million-Segment LLM Benchmark and Dual-Paradigm Evaluation Across High-Resource and Low-Resource Languages

This repository contains the canonical 10,228-string software UI localization corpus (derived from OPUS KDE4v2), six-invariant deterministic verification engine, **Invariant-Guided Self-Correction (IGSC)** pipeline, and dual-paradigm evaluation artifacts across **5 instruction-tuned LLMs** and **12 typologically diverse languages** (`10` model suites, `120` localized catalog files, `1,227,360` evaluated segments, and `46,818` defect trace records).

## Paper

**Title:** Invariant-Guided Self-Correction for Cloud-Native Software Localization Pipelines: A 1.22-Million-Segment LLM Benchmark and Dual-Paradigm Evaluation Across High-Resource and Low-Resource Languages  
**Authors:** Neeraj Kumar Sharma, Subhrakanta Panda, Lalita Bhanu Murthy Neti  
**Affiliation:** Department of Computer Science and Information Systems, Birla Institute of Technology and Science, Pilani – Hyderabad Campus, Hyderabad, 500078, India  
**Target Journal:** Language Resources and Evaluation (Springer Nature)

## Executive Summary of Benchmark Results (1,227,360 Segments)

| Model Architecture (`122,736` segs/tier) | Baseline Flawed (DDR) | IGSC Flawed (DDR) | Defect Reduction | Baseline CometKiwi | IGSC CometKiwi ($\Delta$) |
| --- | --- | --- | --- | --- | --- |
| `google/gemma-2-9b-it` | 21,888 (17.83%) | 996 (0.81%) | **-95.45%** | 0.7395 | **0.7653** (**+0.0259**) |
| `mistralai/Mistral-Nemo-Instruct-2407` | 10,883 (8.87%) | 305 (0.25%) | **-97.20%** | 0.6872 | 0.7136 (**+0.0264**) |
| `Qwen/Qwen2.5-7B-Instruct` | 6,266 (5.11%) | 1,887 (1.54%) | -69.89% | 0.7046 | 0.7069 (+0.0023) |
| `meta-llama/Llama-3.1-8B-Instruct` | 3,008 (2.45%) | **149 (0.12%)** | **-95.05%** | 0.7483 | 0.7531 (+0.0048) |
| `gpt-4o-mini` (Commercial API) | 1,169 (0.95%) | 267 (0.22%) | -77.16% | **0.7877** | **0.7877** (0.0000) |
| **Grand Total / Corpus Mean** | **43,214 (7.04%)** | **3,604 (0.59%)** | **-91.66%** | **0.7335** | **0.7453 (+0.0119)** |

### Key Scientific Findings
1. **Instruction-Induced Placeholder Hallucination (`E_halluc`):** Static zero-shot prompts warning models to preserve `%1` specifiers induce `11,932` spurious `%1` hallucinations on variable-free UI strings (`94.86%` of all baseline placeholder errors). Dynamic source-aware conditioning in IGSC eliminates **98.91%** of these hallucinations (`11,932` $\to$ `130`).
2. **Low-Resource Indic Copy-Fallback & Digraphia Rescue:** Across 6 Indo-Aryan and Dravidian locales (`bn`, `hi`, `mr`, `pa`, `ta`, `te`), IGSC rescues severe copy-fallback and Shahmukhi/Gurmukhi digraphic collisions (e.g., `Gemma-2-9B-IT` Punjabi DDR `77.87%` $\to$ `3.13%`, CometKiwi `+0.1226`; Telugu DDR `75.72%` $\to$ `3.62%`, CometKiwi `+0.1017`).
3. **Open-Weights Structural Parity:** Self-corrected `Llama-3.1-8B-Instruct` achieves a **0.12%** defect rate (`99.88%` Structural Pass Rate) in `88.99s` on a single NVIDIA A100-80GB GPU, outperforming both zero-shot (`0.95%`) and IGSC-repaired (`0.22%`) `gpt-4o-mini`.
4. **Syntactic Blindness of Neural Quality Estimators:** Reference-free neural QE (`Unbabel/wmt22-cometkiwi-da`) exhibits zero corpus-level sensitivity (`+0.0000`, $p = 0.7556$) when `gpt-4o-mini` eliminates `77.16%` of its confirmed runtime defects, proving the necessity of dual-paradigm evaluation.

## Target Languages (12)
- **Low-Resource Indic (6):** Bengali (`bn`), Hindi (`hi`), Marathi (`mr`), Punjabi (`pa`, Gurmukhi `pa-IN`), Tamil (`ta`), Telugu (`te`)
- **High/Medium-Resource Global (6):** Arabic (`ar`), Simplified Chinese (`zh`), French (`fr`), German (`de`), Japanese (`ja`), Spanish (`es`)

## Repository Structure

```text
cloud-native-llm-benchmark/
├── README.md                                  # Benchmark overview and key results
├── REPRODUCIBILITY.md                         # End-to-end execution and verification guide
├── RUN_ENVIRONMENT.md                         # Hardware topology and software runtime specs
├── LICENSE                                    # MIT License (code and verifier)
├── DATA_LICENSE.md                            # OPUS KDE4v2 & LLM output licensing terms
├── CITATION.cff                               # Citation metadata (YAML)
├── CHECKSUMS.txt                              # SHA-256 manifest for all 121 corpus files
├── CHECKSUMS_SHA256.txt                       # SHA-256 checksum manifest
├── CHECKSUMS_MD5.txt                          # MD5 checksum manifest
├── hardware-env-info.txt                      # Raw hardware & runtime audit trail
├── requirements-a100-lock.txt                 # Pinned NVIDIA A100 vLLM 0.32.0 lockfile
├── requirements-lock.txt                      # Pinned baseline generation lockfile
├── requirements-metrics.txt                   # Pinned CometKiwi & statistical analysis deps
├── data/
│   ├── source/en.txt                          # 10,228 canonical English KDE4v2 UI strings
│   ├── baseline_{gpt4o,gemma_9b,llama_8b,mistral_nemo,qwen_7b}/  # 60 zero-shot suites
│   └── repaired_{gpt4o,gemma_9b,llama_8b,mistral_nemo,qwen_7b}/  # 60 IGSC-repaired suites
├── prompts/
│   └── system_zero_shot.txt                   # Zero-shot system prompt specification
├── metrics/cometkiwi/                         # CometKiwi scores, pivots, and before/after deltas
├── findings/                                  # Experiment run metadata
├── reports/                                   # JSONL defect/repair traces, CSV summaries, plots
└── scripts/                                   # Translation, invariant audit, IGSC, & QE scripts
```

## Quick Integrity Check

```bash
git clone https://github.com/Research-SoftwareLocalization/cloud-native-llm-benchmark.git
cd cloud-native-llm-benchmark
sha256sum -c CHECKSUMS.txt
md5sum -c CHECKSUMS_MD5.txt
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [RUN_ENVIRONMENT.md](RUN_ENVIRONMENT.md) for full pipeline instructions.

## License
Original code, invariant verification scripts, and documentation are released under the [MIT License](LICENSE). Source UI strings and model outputs are governed by the terms detailed in [DATA_LICENSE.md](DATA_LICENSE.md).

## Citation

```bibtex
@article{sharma2026igsc,
  title={Invariant-Guided Self-Correction for Cloud-Native Software Localization Pipelines: A 1.22-Million-Segment LLM Benchmark and Dual-Paradigm Evaluation Across High-Resource and Low-Resource Languages},
  author={Sharma, Neeraj Kumar and Panda, Subhrakanta and Neti, Lalita Bhanu Murthy},
  journal={Language Resources and Evaluation},
  year={2026},
  publisher={Springer Nature},
  doi={10.5281/zenodo.24280788}
}
```
