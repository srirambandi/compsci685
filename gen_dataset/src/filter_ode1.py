#!/usr/bin/env python3
import os
import sys
import argparse
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, repo_root)
sys.path.insert(0, script_dir)

from utils import OPERATORS


def main():
    parser = argparse.ArgumentParser(
        description="Filter ODE1 data by length and allowed operators"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the raw ODE1 file (count|in_prefix<TAB>out_prefix)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Where to write filtered lines"
    )
    parser.add_argument(
        "--max_len", type=int, default=128,
        help="Maximum number of tokens per prefix (both input and output)"
    )
    args = parser.parse_args()

    allowed_ops = set(OPERATORS.keys())
    written = 0

    # open files and iterate with progress bar
    total_lines = sum(1 for _ in open(args.input, 'r', encoding='utf-8'))
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_lines, desc="Filtering", unit="line"):
            line = line.strip()
            if not line:
                continue

            # split off the count, we don't need it
            try:
                _, rest = line.split("|", 1)
                inp_prefix, out_prefix = rest.split("\t", 1)
            except ValueError:
                tqdm.write(f"Skipping malformed line: {line}")
                continue

            inp_toks = inp_prefix.split()
            out_toks = out_prefix.split()

            # length filter
            if len(inp_toks) > args.max_len or len(out_toks) > args.max_len:
                continue

            # operator filter: ensure tokens present are in allowed set
            # ops_inp = {t for t in inp_toks if t in allowed_ops}
            # ops_out = {t for t in out_toks if t in allowed_ops}
            # if not ops_inp.issubset(allowed_ops) or not ops_out.issubset(allowed_ops):
            #     continuexw

            # write passing example
            fout.write(f"{inp_prefix}\t{out_prefix}\n")
            written += 1

    print(f"Wrote {written} lines to {args.output}, from {total_lines} total lines.")

if __name__ == "__main__":
    main()
