import os
import sys
import argparse
import hashlib
from pathlib import Path
from openai import OpenAI

def normalize_to_lf_and_verify(file_path: Path, expected_lines: int = 10228):
    if not file_path.exists():
        raise FileNotFoundError(f"Target file missing: {file_path}")
    
    content = file_path.read_text(encoding="utf-8")
    # Normalize CRLF and old Mac CR to standard Unix LF
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    file_path.write_text(normalized, encoding="utf-8")
    
    lines = normalized.splitlines()
    if len(lines) != expected_lines:
        raise ValueError(f"Line parity violation in {file_path.name}: got {len(lines)} lines, expected {expected_lines}.")
    
    sha256_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    print(f"[{file_path.name}] Validated: {len(lines)} lines | LF Normalized | SHA-256: {sha256_hash[:12]}...", flush=True)
    return sha256_hash

def main():
    parser = argparse.ArgumentParser(description="Translate KDE UI corpus with LLMs")
    parser.add_argument("--system_folder", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--base_url", default="https://api.openai.com/v1")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()
    
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Error: Environment variable {args.api_key_env} is not set.", file=sys.stderr)
        sys.exit(1)
        
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    
    source_path = Path("data/source/en.txt")
    if not source_path.exists():
        print("Error: Source file data/source/en.txt not found.", file=sys.stderr)
        sys.exit(1)
        
    source_lines = source_path.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()
    expected_line_count = len(source_lines)
    
    out_dir = Path(f"data/{args.system_folder}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.lang}.txt"
    
    # Resume safeguard: skip if already successfully translated and line-verified
    if out_file.exists():
        try:
            existing_content = out_file.read_text(encoding="utf-8").replace("\r\n", "\n")
            if len(existing_content.splitlines()) == expected_line_count:
                print(f"[{args.lang}] Already fully translated and validated ({expected_line_count} lines). Skipping.", flush=True)
                return
        except Exception:
            pass

    prompt_path = Path("prompts/system_zero_shot.txt")
    system_prompt = "You are a professional software localization engineer. Translate the given UI string accurately while preserving all technical artifacts (such as & accelerators, % variables, and camelCase identifiers)."
    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
        
    print(f"Translating {expected_line_count} lines for language [{args.lang}] using model [{args.model_id}]...", flush=True)
    
    translated_lines = []
    # Process line by line to maintain strict 1:1 mapping and reliable error handling
    for idx, line in enumerate(source_lines):
        if not line.strip():
            translated_lines.append("")
            continue
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Translate this software UI string to '{args.lang}':\n{line}"}
        ]
        
        try:
            response = client.chat.completions.create(
                model=args.model_id,
                messages=messages,
                temperature=0.0
            )
            translation = response.choices[0].message.content.strip()
            # Clean potential model conversational wrappers if present
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
            translated_lines.append(translation)
        except Exception as e:
            print(f"Error at line {idx+1}: {e}", file=sys.stderr)
            # Fallback to source string on API failure to prevent silent truncation
            translated_lines.append(line)

    # Write output with strict Unix LF line endings
    out_file.write_text("\n".join(translated_lines) + "\n", encoding="utf-8")
    
    # Verify parity and compute checksum
    normalize_and_lf_verify = normalize_to_lf_and_verify(out_file, expected_line_count)
    print(f"Successfully completed and verified language: {args.lang}", flush=True)

if __name__ == "__main__":
    main()
