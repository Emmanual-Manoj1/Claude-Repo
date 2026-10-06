#!/usr/bin/env python3
"""Extract OCR rows from ReceiptOCR logcat output into a JSONL file to label.

    python parse_logs.py real_logs/*.log --out to_label.jsonl

Each output line is {"id", "scan", "row", "input", "target": ""}. Fill in
"target" with the item name, or "Non-Item" for headers, totals, promotions
and so on. The labelled file can then be passed to baseline.py --test.

Scans where the OCR merged a whole column into one row (e.g. every price on
one line) are flagged "merged": true. A line-level model cannot fix those,
so they are skipped unless you pass --keep-merged.
"""

import argparse
import json
import re

SCAN_START = re.compile(r"=+ SCAN START")
ROW = re.compile(r"ROW \[(\d+)\] (.*)$")


def parse(paths):
    scans, current = [], None
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if SCAN_START.search(line):
                    current = []
                    scans.append((path, current))
                    continue
                m = ROW.search(line.rstrip("\n"))
                if m and current is not None:
                    current.append((int(m.group(1)), m.group(2).strip()))
    return scans


def is_merged(rows):
    """A scan looks column-merged if a few rows hold most of the words."""
    if len(rows) < 6:
        return True
    return max(len(text.split()) for _, text in rows) > 15


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="to_label.jsonl")
    ap.add_argument("--keep-merged", action="store_true")
    args = ap.parse_args()

    seen, written, skipped = set(), 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        for n, (path, rows) in enumerate(parse(args.logs)):
            merged = is_merged(rows)
            if merged and not args.keep_merged:
                skipped += 1
                continue
            for idx, text in rows:
                if not text or text in seen:  # repeat scans give identical rows
                    continue
                seen.add(text)
                out.write(json.dumps({"id": f"scan{n:03d}-row{idx:02d}", "scan": n,
                                      "row": idx, "input": text, "target": "",
                                      "merged": merged}, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} rows to {args.out} "
          f"({skipped} column-merged scans skipped)")


if __name__ == "__main__":
    main()
