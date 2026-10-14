import os
import subprocess

def main():
    os.makedirs("data/gemini_2_5_pro", exist_ok=True)
    # Google GenAI provides an OpenAI-compatible base URL: https://generativelanguage.googleapis.com/v1beta/openai/
    base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    
    cmd = [
        "python", "scripts/translate_llm.py",
        "--system_folder", "gemini_2_5_pro",
        "--model_id", "gemini-2.5-pro",
        "--base_url", base_url,
        "--api_key_env", "GEMINI_API_KEY"
    ]
    print("Running Gemini 2.5 Pro translation batch...")
    subprocess.run(cmd, check=True)
    print("Gemini translations complete.")

if __name__ == "__main__":
    main()