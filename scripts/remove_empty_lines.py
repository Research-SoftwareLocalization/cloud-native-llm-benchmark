"""
Remove the 3 empty source entries (lines 5389, 9699, 10186) from every data
file, producing a uniform 10,228-line corpus. Alignment is by line number.
Usage:
    python3 scripts/remove_empty_lines.py            # dry-run
    python3 scripts/remove_empty_lines.py --apply    # perform removal
"""
import argparse, glob

REMOVE_LINES = [5389, 9699, 10186]
BEFORE, AFTER = 10231, 10228
SOURCE = "data/source/en.txt"
DATA_GLOBS = [
    "data/source/en.txt",
    "data/baseline_google_translate/*.txt",
    "data/nllb200/*.txt",
    "data/indictrans2/*.txt",
    "data/microsoft_translator/*.txt",
    "data/amazon_translate/*.txt",
]

def read_lines(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src = read_lines(SOURCE)
    if len(src) == BEFORE:
        for ln in REMOVE_LINES:
            if src[ln - 1].strip() != "":
                raise SystemExit(f"ABORT: source line {ln} not empty: {src[ln-1]!r}")
        print(f"Verified: source lines {REMOVE_LINES} are empty.")

    files = []
    for pat in DATA_GLOBS:
        files.extend(sorted(glob.glob(pat)))
    print(f"Found {len(files)} data files.\n")

    to_edit, skipped, problems = [], [], []
    for f in files:
        n = len(read_lines(f))
        if n == AFTER: skipped.append(f)
        elif n == BEFORE: to_edit.append(f)
        else: problems.append((f, n))

    for f in skipped: print(f"[skip] {f} already {AFTER}")
    for f, n in problems: print(f"[WARN] {f} has {n} (expected {BEFORE}) - NOT touched")

    if not args.apply:
        print(f"\nDRY-RUN. Would edit {len(to_edit)} files. Re-run with --apply.")
        return
    if problems:
        raise SystemExit("ABORT: some files not 10,231 lines.")

    idx0 = sorted([ln - 1 for ln in REMOVE_LINES], reverse=True)
    for f in to_edit:
        lines = read_lines(f)
        for i in idx0: del lines[i]
        assert len(lines) == AFTER
        open(f, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"[done] {f} -> {AFTER}")
    print(f"\nEdited {len(to_edit)} files. All now {AFTER} lines.")

if __name__ == "__main__":
    main()
