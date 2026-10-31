import os
import argparse
import pandas as pd
from pathlib import Path
from comet import download_model, load_from_checkpoint

ALL_LANGS = ["ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"]

# Exact 5 evaluated systems in data/
LLM_SYSTEMS = {
    "baseline_gpt4o": ALL_LANGS,
    "baseline_gemma_9b": ALL_LANGS,
    "baseline_llama_8b": ALL_LANGS,
    "baseline_mistral_nemo": ALL_LANGS,
    "baseline_qwen_7b": ALL_LANGS,
}

def read_lines(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def main():
    parser = argparse.ArgumentParser(description="Compute Reference-Free COMETKiwi QE Scores")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="metrics/cometkiwi")
    parser.add_argument("--model_name", type=str, default="Unbabel/wmt22-cometkiwi-da")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_file = data_dir / "source" / "en.txt"
    src_lines = read_lines(src_file)
    print(f"📖 Loaded {len(src_lines)} source strings.")

    print(f"📦 Loading reference-free metric: {args.model_name}...")
    model_path = download_model(args.model_name)
    model = load_from_checkpoint(model_path)

    results = []

    for system, langs in LLM_SYSTEMS.items():
        print(f"\n{'='*50}\nEvaluating: {system}\n{'='*50}")
        sys_dir = data_dir / system
        if not sys_dir.exists():
            print(f"⚠️ Directory {sys_dir} not found, skipping.")
            continue

        for lang in langs:
            hyp_file = sys_dir / f"{lang}.txt"
            if not hyp_file.exists():
                print(f"⚠️ File {hyp_file} not found, skipping.")
                continue

            hyp_lines = read_lines(hyp_file)
            n_lines = min(len(src_lines), len(hyp_lines))
            data = [{"src": src_lines[i], "mt": hyp_lines[i]} for i in range(n_lines)]

            print(f"[{system} -> {lang}] Computing CometKiwi ({n_lines} strings)...")
            seg_scores = model.predict(data, batch_size=args.batch_size, gpus=1 if model.device.type == "cuda" else 0)
            mean_score = seg_scores.system_score

            print(f"  --> Mean CometKiwi: {mean_score:.4f}")
            results.append({
                "system": system,
                "language": lang,
                "total_strings": n_lines,
                "cometkiwi": round(mean_score, 4)
            })

    # Save detailed per-language scores
    df = pd.DataFrame(results)
    out_csv = out_dir / "cometkiwi_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅ CometKiwi complete! Saved to: {out_csv}")

    # Save pivot matrix (systems x languages)
    pivot_df = df.pivot(index="system", columns="language", values="cometkiwi")
    pivot_df.to_csv(out_dir / "cometkiwi_pivot.csv")
    print(f"📊 Pivot table saved to: {out_dir / 'cometkiwi_pivot.csv'}")

if __name__ == "__main__":
    main()
