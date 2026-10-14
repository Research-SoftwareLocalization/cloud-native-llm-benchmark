import os
import time
import argparse
from pathlib import Path
from openai import OpenAI

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "ja": "Japanese",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "zh": "Simplified Chinese",
}

SYSTEM_PROMPT_TEMPLATE = """You are an enterprise cloud-native software localization engine.
Translate the user-provided English software string into {target_language} ({lang_code}).

Strict Rules:
1. Output ONLY the translated string on a single line. Do NOT include explanations, notes, or surrounding quotes.
2. Preserve all placeholders, format specifiers, and variables (e.g., {{0}}, {{name}}, %s, %d, {{{{var}}}}, $ENV_VAR) exactly as written.
3. Keep CLI flags, API paths, YAML keys, and proper cloud-native product names (e.g., Kubernetes, Helm, Prometheus, Pod, DaemonSet, StatefulSet, Ingress) in their standard technical form.
4. Use formal, natural software UI/documentation register appropriate for {target_language}."""


def sanitize_single_line(text: str) -> str:
    """Ensure strict 1-line-per-sentence parity with data/source/en.txt."""
    cleaned = " ".join(text.strip().splitlines()).strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def translate_file(
    client: OpenAI,
    model_id: str,
    source_path: Path,
    output_path: Path,
    lang_code: str,
):
    target_language = LANGUAGE_NAMES[lang_code]
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        target_language=target_language, lang_code=lang_code
    )

    with open(source_path, "r", encoding="utf-8") as f:
        source_lines = [line.rstrip("\n") for line in f]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Support resuming if interrupted
    existing_lines = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_lines = [line.rstrip("\n") for line in f]

    start_idx = len(existing_lines)
    print(f"[{model_id} -> {lang_code}] Resuming from line {start_idx}/{len(source_lines)}")

    with open(output_path, "a", encoding="utf-8") as out_f:
        for idx in range(start_idx, len(source_lines)):
            src_text = source_lines[idx]
            if not src_text.strip():
                out_f.write("\n")
                out_f.flush()
                continue

            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": src_text},
                        ],
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=512,
                    )
                    raw_out = response.choices[0].message.content or ""
                    clean_out = sanitize_single_line(raw_out)
                    if clean_out:
                        out_f.write(clean_out + "\n")
                        out_f.flush()
                        break
                except Exception as e:
                    print(f"Line {idx} attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)
            else:
                # Fallback to source string if all retries fail so line count never drifts
                out_f.write(sanitize_single_line(src_text) + "\n")
                out_f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_folder", required=True, help="e.g., baseline_gpt4o")
    parser.add_argument("--model_id", required=True, help="e.g., gpt-4o-2024-11-20")
    parser.add_argument("--base_url", default=None, help="Optional vLLM/Gemini/Together OpenAI-compatible URL")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--indic_only", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "EMPTY")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    langs = ["bn", "hi", "mr", "pa", "ta", "te"] if args.indic_only else list(LANGUAGE_NAMES.keys())
    src_file = Path("data/source/en.txt")

    for lang in langs:
        out_file = Path("data") / args.system_folder / f"{lang}.txt"
        translate_file(client, args.model_id, src_file, out_file, lang)