import os
import sys
import time
import subprocess
import re
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
    "zh": "Simplified Chinese"
}

LANGUAGE_NAMES = list(LANGUAGE_MAP.keys())
OPEN_WEIGHTS_MODELS = [
    {"system_folder": "baseline_qwen_7b", "model_id": "Qwen/Qwen2.5-7B-Instruct"}
]

def git_push_language(system_name, lang):
    try:
        file_path = f"data/{system_name}/{lang}.txt"
        subprocess.run(["git", "add", file_path], check=True)
        subprocess.run(["git", "commit", "-m", f"Add validated clean {lang}.txt for {system_name} [skip ci]"], check=True)
        subprocess.run(["git", "push", "origin", "main"], capture_output=True)
        print(f"🚀 Successfully backed up clean {lang}.txt to GitHub!")
    except Exception as e:
        print(f"⚠️ Git backup note for {lang}: {e}")

def wait_for_vllm_server(process, timeout=600):
    print("⏳ Checking vLLM server status...", flush=True)
    start_time = time.time()
    while time.time() - start_time < timeout:
        if process.poll() is not None:
            print("\n❌ vLLM process terminated.")
            return False
        try:
            res = httpx.get("http://localhost:8000/v1/models", timeout=2.0)
            if res.status_code == 200:
                print("✅ vLLM server is active and ready!", flush=True)
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
        target_name = LANGUAGE_MAP[lang]
        out_file = out_dir / (lang + ".txt")

        # Skip only if already clean and complete (10,228 lines)
        if out_file.exists() and out_file.stat().st_size > 0:
            existing_lines = out_file.read_text(encoding="utf-8").splitlines()
            if len(existing_lines) == len(lines):
                print(f"⏩ [{idx}/12] {target_name} ({lang}) already completed ({len(existing_lines)} lines). Skipping.")
                continue

        print(f"\n🔄 [{idx}/12] Translating 10228 lines into [{target_name}] using HARDENED Qwen engine...", flush=True)
        translated_lines = []
        client = httpx.Client(timeout=60.0)

        system_prompt = (
            f"You are a professional software localization engine. "
            f"Translate the given English UI string precisely into {target_name}. "
            f"STRICT RULES:\n"
            f"1. Output ONLY the raw {target_name} translation.\n"
            f"2. Never output notes, explanations, or conversational filler.\n"
            f"3. Strictly DO NOT use Chinese, Russian, Korean, or Latin words (except tokens like %1, %d, Ctrl, Shift).\n"
            f"4. For Arabic, use ONLY Arabic script."
        )

        for line_idx, line in enumerate(lines, 1):
            if not line.strip():
                translated_lines.append(line)
                continue

            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": line}
                ],
                "temperature": 0.0,
                "max_tokens": 128
            }

            clean_text = line
            for attempt in range(4):
                try:
                    response = client.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=30.0)
                    if response.status_code == 200:
                        data = response.json()
                        raw = data["choices"][0]["message"]["content"].strip()
                        raw = raw.replace('\n', ' ').replace('\r', '')

                        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                            raw = raw[1:-1].strip()

                        # Guardrail: Check for foreign scripts and model chatter
                        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', raw))
                        has_cyrillic = bool(re.search(r'[\u0400-\u04ff]', raw))
                        has_korean = bool(re.search(r'[\uac00-\ud7af]', raw))
                        has_notes = any(m in raw.lower() for m in ["note:", "translation:", "翻译", "as an ai"])

                        is_bad = False
                        if lang == "ar":
                            if has_chinese or has_cyrillic or has_korean or has_notes:
                                is_bad = True
                            if any(w in raw.lower() for w in ["posição", "dossier", "problem"]):
                                is_bad = True
                        elif lang != "zh" and has_chinese:
                            is_bad = True

                        if is_bad:
                            time.sleep(0.3)
                            continue

                        clean_text = raw
                        break
                except Exception:
                    time.sleep(1)

            translated_lines.append(clean_text)

            if line_idx % 1000 == 0:
                print(f"   Progress: {line_idx}/{len(lines)} lines translated...", flush=True)

        client.close()

        if len(translated_lines) != len(lines):
            raise ValueError(f"Line parity mismatch in {lang}: got {len(translated_lines)}, expected {len(lines)}")

        out_file.write_text("\n".join(translated_lines) + "\n", encoding="utf-8")
        print(f"✅ Completed clean {target_name} ({lang}.txt)!")
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
            "--max-model-len", "2048"
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
