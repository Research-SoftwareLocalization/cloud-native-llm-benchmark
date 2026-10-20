from pathlib import Path
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

LANGUAGE_NAMES = [
    "ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"
]

def translate_lang(lang):
    system_name = "baseline_gpt4o"
    cmd = [
        sys.executable, "scripts/translate_llm.py",
        "--system_folder", system_name,
        "--model_id", "gpt-4o-mini",
        "--base_url", "https://api.openai.com/v1",
        "--api_key_env", "OPENAI_API_KEY",
        "--lang", lang
    ]
    print(f"[GPT-4o-mini] Starting translation worker for language: {lang}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[GPT-4o-mini] Error in language {lang}:\n{result.stderr}", flush=True)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    
    print(f"[GPT-4o-mini] Successfully completed language: {lang}.", flush=True)
    return lang

def main():
    system_name = "baseline_gpt4o"
    os.makedirs(f"data/{system_name}", exist_ok=True)
    print("="*50, flush=True)
    print("Running GPT-4o-mini Translation Suite (Parallel Workers)", flush=True)
    print("="*50, flush=True)
    
    # Run 4 concurrent workers smoothly without git race conditions
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(translate_lang, lang): lang for lang in LANGUAGE_NAMES}
        for future in as_completed(futures):
            lang = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[GPT-4o-mini] Language {lang} failed with exception: {e}", flush=True)
                sys.exit(1)
                
    print("✨ All 12 languages successfully translated! Cleaning up git locks and pushing...", flush=True)
    
    # Remove any stale git index lock if present
    lock_file = Path(".git/index.lock")
    if lock_file.exists():
        lock_file.unlink()
        
    # Log experiment run
    subprocess.run([
        sys.executable, "scripts/log_experiment_run.py",
        "--system", system_name,
        "--model_id", "gpt-4o-mini",
        "--notes", "Parallel zero-shot localization using GPT-4o-mini with 4 workers"
    ], check=True)
    
    # Bulk stage, commit, and push all translation files and logs together
    subprocess.run(["git", "add", "-f", f"data/{system_name}/", "findings/"], check=True)
    
    commit_res = subprocess.run(["git", "commit", "-m", "Add all 12 translated languages for baseline_gpt4o [skip ci]"], capture_output=True, text=True)
    print(commit_res.stdout.decode() if isinstance(commit_res.stdout, bytes) else commit_res.stdout)
    
    push_res = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
    if push_res.returncode == 0:
        print("🚀 Successfully pushed all translation outputs and logs to GitHub!")
    else:
        print(f"⚠️ Git push output: {push_res.stderr.decode() if isinstance(push_res.stderr, bytes) else push_res.stderr}")

if __name__ == "__main__":
    main()
