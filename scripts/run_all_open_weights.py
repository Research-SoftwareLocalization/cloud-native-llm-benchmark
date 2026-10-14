import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_folder", default="llama3_1_8b")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--base_url", default="http://localhost:8000/v1", help="vLLM local server endpoint")
    parser.add_argument("--indic_only", action="store_true", help="Set for Krutrim/Indic specialized LLMs")
    args = parser.parse_args()

    os.makedirs(f"data/{args.system_folder}", exist_ok=True)
    
    cmd = [
        "python", "scripts/translate_llm.py",
        "--system_folder", args.system_folder,
        "--model_id", args.model_id,
        "--base_url", args.base_url,
        "--api_key_env", "VLLM_API_KEY" # or "EMPTY" for local vLLM
    ]
    if args.indic_only:
        cmd.append("--indic_only")

    print(f"Running translation batch for open model: {args.model_id}...")
    subprocess.run(cmd, check=True)
    print(f"Batch {args.system_folder} complete.")

if __name__ == "__main__":
    main()