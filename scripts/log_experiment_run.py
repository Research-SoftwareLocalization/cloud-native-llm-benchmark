from pathlib import Path
import os
import sys
import argparse
import json
import platform
import torch
from datetime import datetime, timezone

def log_run(system_name, model_id, notes):
    log_path = Path("findings/experiment_runs.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    runs = []
    if log_path.exists():
        try:
            runs = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            runs = []
            
    # Capture precise execution hardware footprint for reproducibility audit
    hardware_spec = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU Runtime)"
    }
    
    new_run = {
        "system": system_name,
        "model_id": model_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware_spec,
        "notes": notes
    }
    
    runs.append(new_run)
    log_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Successfully logged experiment run for {system_name} ({model_id}) at {new_run['timestamp_utc']}.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Log experiment execution run")
    parser.add_argument("--system", required=True, help="System folder name")
    parser.add_argument("--model_id", required=True, help="Model identifier")
    parser.add_argument("--notes", default="", help="Experiment notes")
    args = parser.parse_args()
    
    log_run(args.system, args.model_id, args.notes)

if __name__ == "__main__":
    main()
