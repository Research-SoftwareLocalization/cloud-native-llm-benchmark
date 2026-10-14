import os
import subprocess

def main():
    os.makedirs("data/baseline_gpt4o", exist_ok=True)
    cmd = [
        "python", "scripts/translate_llm.py",
        "--system_folder", "baseline_gpt4o",
        "--model_id", "gpt-4o-2024-11-20",
        "--api_key_env", "OPENAI_API_KEY"
    ]
    print("Running GPT-4o translation batch...")
    subprocess.run(cmd, check=True)
    print("GPT-4o translations complete.")

if __name__ == "__main__":
    main()