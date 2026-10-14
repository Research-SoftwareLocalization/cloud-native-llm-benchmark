import os
import datetime
import json
import subprocess
import pandas as pd
from pathlib import Path

LOG_FILE = Path("metrics/results/experiments_audit_trail.csv")

def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    except Exception:
        return "unknown"

def log_run(system_name, model_id, notes=""):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    run_record = {
        "timestamp_utc": datetime.datetime.utcnow.isoformat(timespec="seconds") + "Z",
        "system": system_name,
        "model_snapshot_id": model_id,
        "git_commit": get_git_commit_hash(),
        "hostname": os.environ.get("HOSTNAME", "lightning-studio"),
        "notes": notes
    }
    
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([run_record])], ignore_index=True)
    else:
        df = pd.DataFrame([run_record])
        
    df.to_csv(LOG_FILE, index=False)
    print(f"[AUDIT TRAIL] Logged execution for {system_name} ({model_id}) at {run_record['timestamp_utc']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    log_run(args.system, args.model_id, args.notes)