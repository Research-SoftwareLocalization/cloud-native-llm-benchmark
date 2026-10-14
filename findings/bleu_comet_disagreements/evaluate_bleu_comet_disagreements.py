import pandas as pd
from pathlib import Path
import sacrebleu
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
    data_dir = Path("data")
    ref_system = "baseline_google_translate"
    out_file = Path("findings/bleu_comet_disagreements/verified_metric_disagreements.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    src_lines = read_lines(data_dir / "source" / "en.txt")
    comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    disagreements = []

    for system, langs in LLM_SYSTEMS.items():
        if system == ref_system:
            continue
        for lang in langs:
            hyp_path = data_dir / system / f"{lang}.txt"
            ref_path = data_dir / ref_system / f"{lang}.txt"
            if not hyp_path.exists() or not ref_path.exists():
                continue

            hyps = read_lines(hyp_path)
            refs = read_lines(ref_path)

            comet_inputs = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src_lines, hyps, refs)]
            comet_seg_scores = comet_model.predict(comet_inputs, batch_size=32, gpus=1).scores

            for idx, (s, h, r, c_score) in enumerate(zip(src_lines, hyps, refs, comet_seg_scores)):
                sent_bleu = sacrebleu.sentence_bleu(h, [r]).score
                comet_100 = c_score * 100.0

                # High semantic accuracy (COMET >= 80) despite low lexical overlap (BLEU <= 25)
                if comet_100 >= 80.0 and sent_bleu <= 25.0:
                    disagreements.append({
                        "segment_id": idx + 1,
                        "system": system,
                        "language": lang,
                        "source_en": s,
                        "reference": r,
                        "llm_hypothesis": h,
                        "sentence_BLEU": round(sent_bleu, 2),
                        "segment_COMET": round(comet_100, 2),
                        "comet_minus_bleu": round(comet_100 - sent_bleu, 2),
                    })

    df = pd.DataFrame(disagreements)
    df.sort_values(by="comet_minus_bleu", ascending=False, inplace=True)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} verified disagreements to {out_file}")

if __name__ == "__main__":
    main()