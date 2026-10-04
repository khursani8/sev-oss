#!/usr/bin/env python3
"""Join a sevBench train pool with its labels into finetune rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pool = [json.loads(x) for x in Path(args.pool).read_text().splitlines() if x.strip()]
    labels = {json.loads(x)["id"]: json.loads(x)["choice"]
              for x in Path(args.labels).read_text().splitlines() if x.strip()}
    rows = 0
    with Path(args.out).open("w") as f:
        for i, r in enumerate(pool):
            choice = labels.get(i)
            if choice is None:
                continue
            f.write(json.dumps({"text": r["text"], "choice": choice},
                               ensure_ascii=False) + "\n")
            rows += 1
    print(f"wrote {rows} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
