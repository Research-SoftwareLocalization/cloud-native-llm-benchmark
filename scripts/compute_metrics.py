import os
import argparse
import pandas as pd
from pathlib import Path
import sacrebleu
from rouge_score import rouge_scorer
import evaluate
from bert_score import score as bert_score_fn
from comet import download_model, load_from_checkpoint

ALL_LANGS = ["ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"]
INDIC_LANGS = ["bn", "hi", "mr", "pa", "ta", "te"]

# 5-LLM System Matrix (paralleling the NMT benchmark tiers)
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
    parser = argparse.ArgumentParser(description="Compute Lexical and Neural Metrics for LLM Benchmark")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--ref_system",
        type=str,
        default="baseline_google_translate",
        help="Folder inside data/ used as reference (keep baseline_google_translate for 1:1 NMT parity, or use baseline_gpt4o)",
    )
    parser.add_argument("--out_dir", type=str, default="metrics/results")
    parser.add_argument("--skip_neural", action="store_true", help="Skip BERTScore and COMET if running on CPU only")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_lines = read_lines(data_dir / "source" / "en.txt")
    meteor_metric = evaluate.load("meteor")
    r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    comet_model = None
    if not args.skip_neural:
        print("Loading Unbabel/wmt22-comet-da...")
        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)

    lexical_rows = []
    neural_rows = []

    for system, langs in LLM_SYSTEMS.items():
        if system == args.ref_system:
            continue  # Skip self-comparison if using an LLM as reference

        for lang in langs:
            hyp_path = data_dir / system / f"{lang}.txt"
            ref_path = data_dir / args.ref_system / f"{lang}.txt"

            if not hyp_path.exists() or not ref_path.exists():
                print(f"[SKIP] Missing {hyp_path} or {ref_path}")
                continue

            hyps = read_lines(hyp_path)
            refs = read_lines(ref_path)
            assert len(hyps) == len(refs) == len(src_lines), (
                f"Line count mismatch in {system}/{lang}: hyp={len(hyps)}, ref={len(refs)}, src={len(src_lines)}"
            )

            print(f"Evaluating {system} | {lang} ({len(hyps)} segments)...")

            # 1. Lexical Metrics
            tokenize_opt = "zh" if lang == "zh" else ("ja-mecab" if lang == "ja" else "13a")
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize_opt).score
            chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
            ter = sacrebleu.corpus_ter(hyps, [refs]).score
            meteor = meteor_metric.compute(predictions=hyps, references=refs)["meteor"] * 100.0
            rouge_l = (
                sum(r_scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps))
                / len(hyps)
                * 100.0
            )

            lexical_rows.append(
                {
                    "system": system,
                    "language": lang,
                    "BLEU": round(bleu, 4),
                    "chrF": round(chrf, 4),
                    "METEOR": round(meteor, 4),
                    "ROUGE_L": round(rouge_l, 4),
                    "TER": round(ter, 4),
                }
            )

            # 2. Reference-Based Neural Metrics
            if not args.skip_neural:
                _, _, F1 = bert_score_fn(hyps, refs, lang=lang, verbose=False)
                bertscore_f1 = float(F1.mean().item()) * 100.0

                comet_data = [
                    {"src": s, "mt": h, "ref": r}
                    for s, h, r in zip(src_lines, hyps, refs)
                ]
                comet_out = comet_model.predict(comet_data, batch_size=32, gpus=1)
                comet_score = float(comet_out.system_score) * 100.0

                neural_rows.append(
                    {
                        "system": system,
                        "language": lang,
                        "BERTScore": round(bertscore_f1, 4),
                        "COMET": round(comet_score, 4),
                    }
                )

    df_lex = pd.DataFrame(lexical_rows)
    df_lex.to_csv(out_dir / "metric_scores_lexical.csv", index=False)
    print(f"Saved {out_dir / 'metric_scores_lexical.csv'}")

    if neural_rows:
        df_neu = pd.DataFrame(neural_rows)
        df_neu.to_csv(out_dir / "metric_scores_neural.csv", index=False)
        print(f"Saved {out_dir / 'metric_scores_neural.csv'}")


if __name__ == "__main__":
    main()