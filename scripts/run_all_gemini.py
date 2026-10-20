import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

LANGUAGE_NAMES = [
    "ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"
]

def translate_lang(lang):
    cmd = [
        sys.executable, "scripts/translate_llm.py",
        "--system_folder", "baseline_gemini",
        "--model_id", "gemini-2.5-flash",
        "--base_url", "https://generativelanguage.googleapis.com/v1beta/openai/",
        "--api_key_env", "GEMINI_API_KEY",
        "--lang", lang
    ]
    print(f"[Gemini 2.5 Flash] Starting parallel translation worker for language: {lang}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Gemini 2.5 Flash] Error in language {lang}:\n{result.stderr}", flush=True)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    print(f"[Gemini 2.5 Flash] Successfully completed language: {lang}", flush=True)
    return lang

def git_auto_push(system_name):
    print(f"\n[Git Sync] Backing up outputs for {system_name} to GitHub...", flush=True)
    try:
        subprocess.run(["git", "add", f"data/{system_name}", "hardware-env-info.txt", "requirements-current.txt"], check=True)
        subprocess.run(["git", "commit", "-m", f"Auto-commit: Freeze translation outputs for {system_name} [skip ci]"], check=True)
        push_res = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
        if push_res.returncode == 0:
            print(f"🚀 Successfully pushed {system_name} results to GitHub!", flush=True)
        else:
            print(f"⚠️ Git push warning: {push_res.stderr}", flush=True)
    except Exception as e:
        print(f"⚠️ Git auto-push skipped or failed: {e}", flush=True)

def main():
    system_name = "baseline_gemini"
    os.makedirs(f"data/{system_name}", exist_ok=True)
    print("="*50, flush=True)
    print("Running Gemini 2.5 Flash Parallel Translation Suite", flush=True)
    print("="*50, flush=True)
    
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(translate_lang, lang): lang for lang in LANGUAGE_NAMES}
        for future in as_completed(futures):
            lang = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[Gemini 2.5 Flash] Language {lang} failed with exception: {e}", flush=True)
                sys.exit(1)
                
    # Log run and auto-push to GitHub
    subprocess.run([
        sys.executable, "scripts/log_experiment_run.py",
        "--system", system_name,
        "--model_id", "gemini-2.5-flash",
        "--notes", "Parallel zero-shot localization using Gemini 2.5 Flash via OpenAI-compatible endpoint"
    ], check=True)
    
    git_auto_push(system_name)

if __name__ == "__main__":
    main()
