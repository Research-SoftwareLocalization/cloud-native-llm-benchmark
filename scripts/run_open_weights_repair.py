import os
import sys
import time
import subprocess
import httpx
from pathlib import Path

REPO_ROOT = Path("/content/cloud-native-llm-benchmark") if Path("/content/cloud-native-llm-benchmark").exists() else Path(".")
os.chdir(REPO_ROOT)

OPEN_WEIGHTS_SUITE = [
    {
        "baseline_name": "baseline_llama_8b",
        "repaired_name": "repaired_llama_8b",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "extra_vllm_args": []
    },
    {
        "baseline_name": "baseline_qwen_7b",
        "repaired_name": "repaired_qwen_7b",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "extra_vllm_args": []
    },
    {
        "baseline_name": "baseline_mistral_nemo",
        "repaired_name": "repaired_mistral_nemo",
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "extra_vllm_args": []
    },
    {
        "baseline_name": "baseline_gemma_9b",
        "repaired_name": "repaired_gemma_9b",
        "model_id": "google/gemma-2-9b-it",
        "extra_vllm_args": ["--dtype", "bfloat16"]
    }
]

def wait_for_vllm(process, timeout=600):
    print("⏳ Waiting for local vLLM server to load model weights...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        if process.poll() is not None:
            print("❌ vLLM server process exited prematurely!", flush=True)
            return False
        try:
            r = httpx.get("http://localhost:8000/v1/models", timeout=2.0)
            if r.status_code == 200:
                print("✅ vLLM server is online and ready!", flush=True)
                return True
        except Exception:
            pass
        time.sleep(5)
    return False

def kill_existing_vllm():
    subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)
    time.sleep(3)

def main():
    target_filter = sys.argv[1] if len(sys.argv) > 1 else None
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "EMPTY"

    for item in OPEN_WEIGHTS_SUITE:
        b_name = item["baseline_name"]
        r_name = item["repaired_name"]
        model_id = item["model_id"]

        if target_filter and target_filter not in (b_name, r_name, model_id):
            continue

        print(f"\n{'='*65}\n🔧 STARTING SELF-REPAIR: {b_name} using {model_id}\n{'='*65}", flush=True)
        kill_existing_vllm()

        vllm_cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id,
            "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "2048",
        ] + item["extra_vllm_args"]

        server_proc = subprocess.Popen(vllm_cmd, env=env)

        if not wait_for_vllm(server_proc):
            server_proc.terminate()
            print(f"❌ Failed to start vLLM for {model_id}. Moving to next.", flush=True)
            continue

        try:
            repair_cmd = [
                sys.executable, "-u", "scripts/run_agentic_repair.py",
                "--model", b_name,
                "--base_url", "http://localhost:8000/v1",
                "--api_key_env", "OPENAI_API_KEY",
                "--model_id", model_id,
                "--max_workers", "16"
            ]
            subprocess.run(repair_cmd, env=env, check=True)

            # Re-run unified audit and push checkpoints after each model finishes
            subprocess.run([sys.executable, "scripts/audit_and_capture_findings.py"], check=True)
            subprocess.run(["git", "add", f"data/{r_name}/", "reports/"], check=True)
            subprocess.run(
                ["git", "commit", "-m", f"feat(repair): complete self-repair for {b_name} -> {r_name} [skip ci]"],
                check=False
            )
            subprocess.run(["git", "push", "origin", "main"], check=False)
            print(f"🚀 Pushed {r_name} and updated audit report to GitHub!", flush=True)

        finally:
            server_proc.terminate()
            server_proc.wait()
            kill_existing_vllm()
            print(f"🛑 Stopped vLLM server for {model_id}.", flush=True)

if __name__ == "__main__":
    main()
