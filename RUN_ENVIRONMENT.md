# Hardware and Software Execution Environment

This document summarizes the exact hardware topology and pinned software environments recorded in `hardware-env-info.txt` and `reports/a100_system_manifest.json`.

## 1. Unified IGSC Self-Correction & CometKiwi QE Environment (`requirements-a100-lock.txt`)
All open-weights Invariant-Guided Self-Correction (IGSC) passes (`42,045` candidate defects across 48 suites) and all 10-suite WMT22 CometKiwi (`Unbabel/wmt22-cometkiwi-da`, `1,227,360` segments) evaluations were executed on a unified high-memory GPU node:
- **GPU Accelerator:** NVIDIA A100-SXM4-80GB (`81,920 MiB` physical / `79.25 GB` usable VRAM, Driver `580.82.07`, CUDA `13.0`)
- **Host CPU & RAM:** 12-core `x86_64` CPU, `175.17 GB` system RAM
- **OS / Kernel:** `Linux-6.6.159+-x86_64-with-glibc2.39`
- **Core Stack:** Python `3.13.16`, PyTorch `2.13.0+cu130`, vLLM `0.32.0` (`gpu_memory_utilization=0.90`, `temperature=0.0`, 16 concurrent workers)
- **Execution Timestamp:** 1 November 2026 (`04:48:45 UTC`)

## 2. Zero-Shot Baseline Generation Runs (`requirements-lock.txt`)
- **`baseline_gpt4o` (`gpt-4o-mini`):** OpenAI Chat Completions REST API (`temperature=0.0`), Linux CPU runtime, Python `3.13.16` (17 Oct 2026).
- **`baseline_gemma_9b` (`google/gemma-2-9b-it`):** Google Colab GPU runtime (A100/L4, `bfloat16`), Python `3.13.15`, PyTorch `2.11.0+cu130` (18 Oct 2026).
- **`baseline_qwen_7b` (`Qwen/Qwen2.5-7B-Instruct`):** Google Colab NVIDIA L4 (`24 GB` VRAM), Python `3.13.15`, PyTorch `2.11.0+cu130` (19 Oct 2026).
- **`baseline_mistral_nemo` (`mistralai/Mistral-Nemo-Instruct-2407`):** Google Colab NVIDIA A100-SXM4-80GB, Python `3.13.15`, PyTorch `2.11.0+cu130` (21 Oct 2026).
- **`baseline_llama_8b` (`meta-llama/Llama-3.1-8B-Instruct`):** Google Colab NVIDIA A100-SXM4-80GB, Python `3.13.15`, PyTorch `2.11.0+cu130` (21 Oct 2026).

## 3. Pinned Dependency Lockfiles
- `requirements-a100-lock.txt`: Exact package freeze for the NVIDIA A100 vLLM `0.32.0` IGSC & QE runtime.
- `requirements-lock.txt`: Baseline open-weights generation environment.
- `requirements-metrics.txt`: Minimal neural QE and statistical analysis dependencies (`unbabel-comet`, `scipy`, `pandas`, `numpy`).
