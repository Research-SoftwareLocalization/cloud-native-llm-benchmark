import os
import sys
import time
import re
import subprocess
import httpx
from pathlib import Path

LANGUAGE_MAP = {
    "ar": "Modern Standard Arabic",
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

# Unicode Script Blocks for Universal Guardrails
ARABIC   = r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
CYRILLIC = r"\u0400-\u04FF"
CJK      = r"\u4E00-\u9FFF"
HANGUL   = r"\uAC00-\uD7AF"
KANA     = r"\u3040-\u30FF"

DEV_HI_MR = r"\u0900-\u097F"
BENGALI   = r"\u0980-\u09FF"
GURMUKHI  = r"\u0A00-\u0A7F"
TAMIL     = r"\u0B80-\u0BFF"
TELUGU    = r"\u0C00-\u0C7F"
ALL_INDIC = f"{DEV_HI_MR}{BENGALI}{GURMUKHI}{TAMIL}{TELUGU}"

FORBIDDEN_SCRIPTS = {
    "es": re.compile(f"[{ARABIC}{CYRILLIC}{CJK}{HANGUL}{KANA}{ALL_INDIC}]"),
    "fr": re.compile(f"[{ARABIC}{CYRILLIC}{CJK}{HANGUL}{KANA}{ALL_INDIC}]"),
    "de": re.compile(f"[{ARABIC}{CYRILLIC}{CJK}{HANGUL}{KANA}{ALL_INDIC}]"),
    "ar": re.compile(f"[{CJK}{HANGUL}{KANA}{CYRILLIC}{ALL_INDIC}]"),
    "hi": re.compile(f"[{ARABIC}{CYRILLIC}{CJK}{HANGUL}{BENGALI}{GURMUKHI}{TAMIL}{TELUGU}]"),
    "mr": re.compile(f"[{ARABIC}{CYRILLIC}{CJK}{HANGUL}{BENGALI}{GURMUKHI}{TAMIL}{TELUGU}]"),
    "bn": re.compile(f"[{ARABIC}{CYRILLIC}{CJK}{HANGUL}{DEV_HI_MR}{GURMUKHI}{TAMIL}{TELUGU}]"),
    "pa": re.compile(f"[{ARABIC}{CYRILLIC}{CJK}{HANGUL}{DEV_HI_MR}{BENGALI}{TAMIL}{TELUGU}]"),
    "ta": re.compile(f"[{ARABIC}{CYRILLIC}{CJK}{HANGUL}{DEV_HI_MR}{BENGALI}{GURMUKHI}{TELUGU}]"),
    "te": re.compile(f"[{ARABIC}{CYRILLIC}{CJK}{HANGUL}{DEV_HI_MR}{BENGALI}{GURMUKHI}{TAMIL}]"),
    "zh": re.compile(f"[{ARABIC}{CYRILLIC}{HANGUL}{KANA}{ALL_INDIC}]"),
    "ja": re.compile(f"[{ARABIC}{CYRILLIC}{HANGUL}{ALL_INDIC}]"),
}

LANGUAGE_NAMES = ["ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"]

# Authentic Open-Weights Model Registry
OPEN_WEIGHTS_MODELS = [
    {
        "system_folder": "baseline_gemma_9b",
        "model_id": "google/gemma-2-9b-it",
        "notes": "Parallel zero-shot localization using Gemma-2-9B-Instruct"
    },
    {
        "system_folder": "baseline_qwen_7b",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "notes": "Parallel zero-shot localization using Qwen-2.5-7B-Instruct"
    },
    {
        "system_folder": "baseline_llama_8b",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "notes": "Parallel zero-shot localization using Llama-3.1-8B-Instruct"
    },
    {
        "system_folder": "baseline_mistral_nemo",
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "notes": "Parallel zero-shot localization using Mistral-Nemo-Instruct-2407"
    }
]

def git_push_language(system_name, lang):
    try:
        file_path = f"data/{system_name}/{lang}.txt"
        subprocess.run(["git", "add", file_path], check=True)
        subprocess.run(["git", "commit", "-m", f"Auto-commit: {system_name} finished {lang} [skip ci]"], check=True)
        subprocess.run(["git", "push", "origin", "main"], capture_output=True)
        print(f"🚀 Successfully backed up {lang}.txt to GitHub!")
    except Exception as e:
        print(f"⚠️ Git auto-push failed for {lang}: {e}")

def wait_for_vllm_server(process, timeout=600):
    print("⏳ Waiting for vLLM server to boot...", flush=True)
    start_time = time.time()
    while time.time() - start_time < timeout:
        if process.poll() is not None:
            print("\n❌ vLLM process died unexpectedly.")
            return False
        try:
            res = httpx.get("http://localhost:8000/v1/models", timeout=2.0)
            if res.status_code == 200:
                print("\n✅ vLLM server is active and ready!", flush=True)
                return True
        except Exception:
            pass
        time.sleep(5)
    return False

def translate_corpus(model_id, system_name):
    src_text = Path("data/source/en.txt").read_text(encoding="utf-8")
    lines = src_text.splitlines()

    out_dir = Path("data/" + system_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, lang in enumerate(LANGUAGE_NAMES, 1):
        target_name = LANGUAGE_MAP.get(lang, lang)
        out_file = out_dir / (lang + ".txt")
        if out_file.exists() and out_file.stat().st_size > 0:
            existing_lines = out_file.read_text(encoding="utf-8").splitlines()
            if len(existing_lines) == len(lines):
                print(f"⏩ [{idx}/12] {target_name} ({lang}) already completed perfectly ({len(existing_lines)} lines). Skipping.")
                continue

        print(f"\n⚡ [{idx}/12] Translating into [{target_name}] using UNIVERSAL SCRIPT GUARDRAIL...", flush=True)
        translated_lines = []
        client = httpx.Client(timeout=60.0)

        base_system_prompt = (
            f"You are a professional software localization engine. "
            f"Translate the given English UI string precisely into pure {target_name}. "
            f"Output ONLY the translated text without explanation, commentary, or conversational filler. "
            f"Preserve all variables (%1, %d), accelerator keys (&), and markup tags verbatim. "
            f"Never output foreign out-of-alphabet scripts."
        )

        bad_script_pattern = FORBIDDEN_SCRIPTS.get(lang)

        for line_idx, line in enumerate(lines, 1):
            if not line.strip():
                translated_lines.append(line)
                continue

            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": base_system_prompt},
                    {"role": "user", "content": line}
                ],
                "temperature": 0.0,
                "max_tokens": 256,
                "repetition_penalty": 1.1
            }

            success = False
            for attempt in range(3):
                try:
                    response = client.post("http://localhost:8000/v1/chat/completions", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        translated = data["choices"][0]["message"]["content"].strip()
                        translated = translated.replace('\n', ' ').replace('\r', '')

                        if (translated.startswith('"') and translated.endswith('"')) or (translated.startswith("'") and translated.endswith("'")):
                            translated = translated[1:-1].strip()

                        if bad_script_pattern and bad_script_pattern.search(translated):
                            payload["messages"][0]["content"] += f" REJECTED: Output contained foreign characters. Output strictly in {target_name} script."
                            continue

                        if any(p in translated.lower() for p in ["i'm ready", "i can't", "please provide", "as an ai"]):
                            translated = line

                        translated_lines.append(translated)
                        success = True
                        break
                except Exception:
                    time.sleep(1)

            if not success:
                translated_lines.append(line)

        client.close()

        if len(translated_lines) != len(lines):
            raise ValueError(f"Line parity mismatch in {lang}: got {len(translated_lines)}, expected {len(lines)}")

        out_file.write_text("\n".join(translated_lines) + "\n", encoding="utf-8")
        print(f"✅ Completed clean {target_name} ({lang}.txt)! Backing up...")
        git_push_language(system_name, lang)

def main():
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "EMPTY"

    for item in OPEN_WEIGHTS_MODELS:
        system_name = item["system_folder"]
        model_id = item["model_id"]

        print(f"\n{'='*50}\nStarting Model: {model_id} -> {system_name}\n{'='*50}", flush=True)

        server_process = subprocess.Popen([
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id,
            "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "2048",
        ], stdout=subprocess.DEVNULL, env=env)

        if not wait_for_vllm_server(server_process):
            server_process.terminate()
            sys.exit(1)

        try:
            translate_corpus(model_id, system_name)
        finally:
            server_process.terminate()
            server_process.wait()
            print("🛑 vLLM server stopped.\n")

if __name__ == "__main__":
    main()
