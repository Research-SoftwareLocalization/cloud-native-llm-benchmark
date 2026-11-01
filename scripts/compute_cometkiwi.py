import os
import subprocess
import argparse
import importlib.util
import pandas as pd
import torch
from pathlib import Path

# Patch 0: Update comet/encoders/*.py on disk BEFORE importing comet
_comet_spec = importlib.util.find_spec("comet")
if _comet_spec and _comet_spec.origin:
    _enc_dir = Path(_comet_spec.origin).parent / "encoders"
    for _py in _enc_dir.glob("*.py"):
        _txt = _py.read_text(encoding="utf-8")
        if "last_hidden_states, _, all_layers" in _txt:
            _py.write_text(
                _txt.replace("last_hidden_states, _, all_layers", "last_hidden_states, *_, all_layers"),
                encoding="utf-8"
            )

from transformers import PreTrainedTokenizerBase, XLMRobertaTokenizer, XLMRobertaTokenizerFast
from comet import download_model, load_from_checkpoint

# Enable A100 Tensor Core acceleration
torch.set_float32_matmul_precision("high")

# Patch 1: Restore build_inputs_with_special_tokens removed in transformers 5.x
def _xlmr_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    cls = [self.cls_token_id]
    sep = [self.sep_token_id]
    if token_ids_1 is None:
        return cls + list(token_ids_0) + sep
    return cls + list(token_ids_0) + sep + sep + list(token_ids_1) + sep

for _cls in (PreTrainedTokenizerBase, XLMRobertaTokenizer, XLMRobertaTokenizerFast):
    if not hasattr(_cls, "build_inputs_with_special_tokens"):
        setattr(_cls, "build_inputs_with_special_tokens", _xlmr_build_inputs_with_special_tokens)

# Patch 2: Dynamic 2-tuple / 3-tuple forward wrapper for any COMET encoder class
def _patched_xlmr_forward(self, input_ids, attention_mask, **kwargs):
    out = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=False,
    )
    last_hidden_states = out[0]
    all_layers = out[-1]
    return {
        "sentemb": last_hidden_states[:, 0, :],
        "wordemb": last_hidden_states,
        "all_layers": all_layers,
        "attention_mask": attention_mask,
    }

ALL_LANGS = ["ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"]
MODELS = ["gemma_9b", "gpt4o", "llama_8b", "mistral_nemo", "qwen_7b"]

def read_lines(filepath: Path):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\r\n") for line in f]

def git_backup_metrics(out_dir: Path, system_name: str):
    try:
        subprocess.run(["git", "add", str(out_dir)], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Auto-commit: CometKiwi scores for {system_name} [skip ci]"],
            check=False, capture_output=True
        )
        subprocess.run(["git", "push", "origin", "main"], check=False, capture_output=True)
        print(f"🚀 Backed up {system_name} CometKiwi scores to GitHub!", flush=True)
    except Exception as e:
        print(f"⚠️ Git backup note: {e}", flush=True)

def build_before_after_delta(df: pd.DataFrame, out_dir: Path):
    delta_rows = []
    for m in MODELS:
        b_name, r_name = f"baseline_{m}", f"repaired_{m}"
        b_sub = df[df["system"] == b_name].set_index("language")
        r_sub = df[df["system"] == r_name].set_index("language")
        if b_sub.empty or r_sub.empty:
            continue
        for lang in ALL_LANGS:
            if lang in b_sub.index and lang in r_sub.index:
                b_val = float(b_sub.loc[lang, "cometkiwi"])
                r_val = float(r_sub.loc[lang, "cometkiwi"])
                delta_rows.append({
                    "model": m,
                    "language": lang,
                    "baseline_cometkiwi": round(b_val, 4),
                    "repaired_cometkiwi": round(r_val, 4),
                    "delta_cometkiwi": round(r_val - b_val, 4)
                })
    if delta_rows:
        delta_df = pd.DataFrame(delta_rows)
        delta_df.to_csv(out_dir / "cometkiwi_before_after_delta.csv", index=False)

        summary_rows = []
        for m in MODELS:
            m_df = delta_df[delta_df["model"] == m]
            if not m_df.empty:
                b_mean = m_df["baseline_cometkiwi"].mean()
                r_mean = m_df["repaired_cometkiwi"].mean()
                summary_rows.append({
                    "model": m,
                    "mean_baseline_cometkiwi": round(b_mean, 4),
                    "mean_repaired_cometkiwi": round(r_mean, 4),
                    "mean_delta_cometkiwi": round(r_mean - b_mean, 4)
                })
        pd.DataFrame(summary_rows).to_csv(out_dir / "cometkiwi_model_summary.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description="Compute Reference-Free COMETKiwi QE Scores (Repaired + Baselines)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="metrics/cometkiwi")
    parser.add_argument("--model_name", type=str, default="Unbabel/wmt22-cometkiwi-da")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "cometkiwi_scores.csv"
    pivot_csv = out_dir / "cometkiwi_pivot.csv"

    src_file = data_dir / "source" / "en.txt"
    src_lines = read_lines(src_file)
    print(f"📖 Loaded {len(src_lines)} source strings from {src_file}", flush=True)

    completed = {}
    results = []
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        results = existing_df.to_dict("records")
        for r in results:
            completed[(r["system"], r["language"])] = r["cometkiwi"]
        print(f"♻️ Resuming from checkpoint: {len(completed)} (system, lang) pairs already scored.", flush=True)

    # Order: Score all 5 repaired_* suites FIRST, then all 5 baseline_* suites
    repaired_suites = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("repaired_")])
    baseline_suites = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("baseline_")])
    systems = repaired_suites + baseline_suites
    print(f"🔍 Execution Queue ({len(systems)} suites — Repaired First, Then Baselines):\n   {systems}", flush=True)

    print(f"📦 Loading cached reference-free QE model: {args.model_name}...", flush=True)
    model_path = download_model(args.model_name)
    model = load_from_checkpoint(model_path)
    setattr(model.encoder.tokenizer.__class__, "build_inputs_with_special_tokens", _xlmr_build_inputs_with_special_tokens)
    setattr(model.encoder.__class__, "forward", _patched_xlmr_forward)

    use_gpus = 1 if torch.cuda.is_available() else 0
    print(f"⚡ Accelerator status: {'GPU (CUDA A100 Tensor Cores)' if use_gpus else 'CPU'}", flush=True)

    for system in systems:
        print(f"\n{'='*55}\nEvaluating Suite: {system}\n{'='*55}", flush=True)
        sys_dir = data_dir / system
        newly_scored_in_sys = 0

        for lang in ALL_LANGS:
            if (system, lang) in completed:
                print(f"⏩ [{system} -> {lang}] Already scored ({completed[(system, lang)]:.4f}). Skipping.", flush=True)
                continue

            hyp_file = sys_dir / f"{lang}.txt"
            if not hyp_file.exists():
                continue

            hyp_lines = read_lines(hyp_file)
            n_lines = min(len(src_lines), len(hyp_lines))
            data = [{"src": src_lines[i], "mt": hyp_lines[i]} for i in range(n_lines)]

            print(f"⏳ [{system} -> {lang}] Scoring {n_lines} strings...", flush=True)
            seg_scores = model.predict(data, batch_size=args.batch_size, gpus=use_gpus)
            mean_score = round(float(seg_scores.system_score), 4)

            print(f"  ✅ [{system} -> {lang}] Mean CometKiwi: {mean_score:.4f}", flush=True)
            results.append({
                "system": system,
                "language": lang,
                "total_strings": n_lines,
                "cometkiwi": mean_score
            })
            completed[(system, lang)] = mean_score
            newly_scored_in_sys += 1

            df = pd.DataFrame(results)
            df.to_csv(out_csv, index=False)
            pivot_df = df.pivot(index="system", columns="language", values="cometkiwi")
            pivot_df["Mean"] = pivot_df.mean(axis=1).round(4)
            pivot_df.to_csv(pivot_csv)
            build_before_after_delta(df, out_dir)

        if newly_scored_in_sys > 0:
            git_backup_metrics(out_dir, system)

    print(f"\n🎉 All 10 suites scored! Final results saved to:\n  - {out_csv}\n  - {pivot_csv}\n  - {out_dir / 'cometkiwi_before_after_delta.csv'}\n  - {out_dir / 'cometkiwi_model_summary.csv'}", flush=True)

if __name__ == "__main__":
    main()
