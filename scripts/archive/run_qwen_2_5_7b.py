import os
import sys
import time
import subprocess
import httpx
from pathlib import Path

LANGUAGE_NAMES = ["ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"]
OPEN_WEIGHTS_MODELS = [
    {"system_folder": "baseline_qwen_7b", "model_id": "Qwen/Qwen2.5-7B-Instruct", "notes": "Stable vLLM Qwen 2.5 7B Instruct"}
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
    print("⏳ Waiting for vLLM server to boot and load Qwen model...", flush=True)
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
        out_file = out_dir / (lang + ".txt")
        if out_file.exists() and out_file.stat().st_size > 0:
            existing_lines = out_file.read_text(encoding="utf-8").splitlines()
            if len(existing_lines) == len(lines):
                print(f"⏩ [{idx}/12] {lang} already completed perfectly ({len(existing_lines)} lines). Skipping.")
                continue
            
        print(f"\n🔄 [{idx}/12] Translating into [{lang}] using Qwen sequential engine...", flush=True)
        translated_lines = []
        client = httpx.Client(timeout=60.0)
        
        system_prompt = (
            "You are a professional software localization engine. "
            "Translate the given English UI string precisely into " + lang + ". "
            "Output ONLY the translated text without any explanation, quotes, conversational filler, or lists."
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
                            
                        # Strict anti-hallucination guardrail
                        if any(p in translated.lower() for p in ["i'm ready", "i can't", "please provide", "as an ai", "dataset"]):
                            translated = line
                            
                        translated_lines.append(translated)
                        success = True
                        break
                except Exception:
                    time.sleep(1)
            
            if not success:
                translated_lines.append(line)
                
            if line_idx % 1000 == 0:
                print(f"   Progress: {line_idx}/{len(lines)} lines translated...", flush=True)
                
        client.close()
        
        if len(translated_lines) != len(lines):
            raise ValueError(f"Line parity mismatch in {lang}: got {len(translated_lines)}, expected {len(lines)}")
            
        out_file.write_text("\n".join(translated_lines) + "\n", encoding="utf-8")
        print(f"✅ Completed {lang}. Backing up...")
        git_push_language(system_name, lang)

def main():
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "EMPTY"
    
    for item in OPEN_WEIGHTS_MODELS:
        system_name = item["system_folder"]
        model_id = item["model_id"]
        
        print(f"\n{'='*50}\nStarting Model: {model_id} -> {system_name}\n{'='*50}", flush=True)
        
        # Use sys.executable to ensure vLLM runs in the exact same Python environment
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
            print(f"🛑 vLLM server stopped.\n")

if __name__ == "__main__":
    main()
