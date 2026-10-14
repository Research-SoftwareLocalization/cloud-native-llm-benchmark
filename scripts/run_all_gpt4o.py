import os
import subprocess

def main():
    system_name = "baseline_gpt4o"
    model_id = "gpt-4o-mini"  # Cost-optimized commercial baseline
    os.makedirs(f"data/{system_name}", exist_ok=True)
    
    # 1. Run translation batch
    cmd = [
        "python", "scripts/translate_llm.py",
        "--system_folder", system_name,
        "--model_id", model_id,
        "--api_key_env", "OPENAI_API_KEY"
    ]
    print(f"Running {system_name} ({model_id}) translation batch...")
    subprocess.run(cmd, check=True)
    
    # 2. Log run into audit trail
    log_cmd = [
        "python", "scripts/log_experiment_run.py",
        "--system", system_name,
        "--model_id", model_id,
        "--notes", "Zero-shot localization run using gpt-4o-mini"
    ]
    subprocess.run(log_cmd, check=True)
    print(f"{system_name} execution successfully logged.")

if __name__ == "__main__":
    main()