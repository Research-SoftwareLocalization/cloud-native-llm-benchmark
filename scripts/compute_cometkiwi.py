import argparse
import pandas as pd
from pathlib import Path
from comet import download_model, load_from_checkpoint

ALL_LANGS = ["ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"]
INDIC_LANGS = ["bn", "hi", "mr", "pa", "ta", "te"]

LLM_SYSTEMS = {
    "baseline_gpt4o": ALL_LANGS,
    "gemini_2_5_pro": ALL_LANGS,
    "llama3_1_8b": ALL_LANGS,
    "qwen2_5_7b": ALL_LANGS,
    "krutrim2_indic": INDIC_LANGS,
}


def read_lines(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def main():
    parser = argparse.ArgumentParser(description="Compute Reference-Free COMETKiwi QE Scores")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="metrics/cometkiwi")
    parser.add_argument("--model_name", type=str, default="Unbabel/wmt22-cometkiwi-da")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_lines = read_lines(data_dir / "source" / "en.txt")
    model_path = download_model(args.model_name)
    model = load_from_checkpoint(model_path)

    rows = []
    for system, langs in LLM_SYSTEMS.items():
        for lang in langs:
            hyp_path = data_dir / system / f"{lang}.txt"
            if not hyp_path.exists():
                print(f"[SKIP] Missing {hyp_path}")
                continue

            hyps = read_lines(hyp_path)
            data = [{"src": s, "mt": h} for s, h in zip(src_lines, hyps)]
            output = model.predict(data, batch_size=32, gpus=1)
            score = round(float(output.system_score) * 100.0, 4)

            rows.append({"system": system, "language": lang, "COMETKiwi": score})
            print(f"{system} | {lang} -> COMETKiwi: {score}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cometkiwi_scores.csv", index=False)

    pivot_df = df.pivot(index="language", columns="system", values="COMETKiwi")
    pivot_df.to_csv(out_dir / "cometkiwi_pivot.csv")
    print(f"Saved {out_dir / 'cometkiwi_scores.csv'} and {out_dir / 'cometkiwi_pivot.csv'}")


if __name__ == "__main__":
    main()