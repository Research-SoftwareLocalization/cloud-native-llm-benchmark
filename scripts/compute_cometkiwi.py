import os
import subprocess
import argparse
import pandas as pd
import torch
from pathlib import Path
from comet import download_model, load_from_checkpoint

ALL_LANGS = ["ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"]

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

def main():
    parser = argparse.ArgumentParser(description="Compute Reference-Free COMETKiwi QE Scores with Checkpointing")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="metrics/cometkiwi")
    parser.add_argument("--model_name", type=str, default="Unbabel/wmt22-cometkiwi-da")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "cometkiwi_scores.csv"
    pivot_csv = out_dir / "cometkiwi_pivot.csv"

    src_file = data_dir / "source" / "en.txt"
    if not src_file.exists():
        raise FileNotFoundError(f"Source file missing: {src_file}")
    src_lines = read_lines(src_file)
    print(f"📖 Loaded {len(src_lines)} source strings from {src_file}", flush=True)

    # Load existing checkpoint if present
    completed = {}
    results = []
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        results = existing_df.to_dict("records")
        for r in results:
            completed[(r["system"], r["language"])] = r["cometkiwi"]
        print(f"♻️ Resuming from checkpoint: {len(completed)} (system, lang) pairs already scored.", flush=True)

    # Dynamically discover all baseline_* and repaired_* directories
    systems = sorted([
        p.name for p in data_dir.iterdir()
        if p.is_dir() and (p.name.startswith("baseline_") or p.name.startswith("repaired_"))
    ])
    print(f"🔍 Discovered {len(systems)} suites to evaluate: {systems}", flush=True)

    print(f"📦 Loading reference-free QE model: {args.model_name}...", flush=True)
    model_path = download_model(args.model_name)
    model = load_from_checkpoint(model_path)
    use_gpus = 1 if torch.cuda.is_available() else 0
    print(f"⚡ Accelerator status: {'GPU (CUDA)' if use_gpus else 'CPU'}", flush=True)

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

            # Save checkpoint after every single language
            df = pd.DataFrame(results)
            df.to_csv(out_csv, index=False)
            pivot_df = df.pivot(index="system", columns="language", values="cometkiwi")
            pivot_df["Mean"] = pivot_df.mean(axis=1).round(4)
            pivot_df.to_csv(pivot_csv)

        if newly_scored_in_sys > 0:
            git_backup_metrics(out_dir, system)

    print(f"\n🎉 All suites scored! Final results saved to:\n  - {out_csv}\n  - {pivot_csv}", flush=True)

if __name__ == "__main__":
    main()
