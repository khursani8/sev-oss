#!/usr/bin/env python3
"""Score a vLLM-served adapter on a labeled eval split via word parse.

Generic served-tier scorer: per-task system prompt, first-word parse,
accuracy vs the split's teacher labels. Resumable.

Usage:
  .venv/bin/python scripts/served_eval.py --endpoint http://localhost:9000/v1 \
    --model sentiment-adapter --system "Classify... one word: positive, neutral, or negative." \
    --valid "positive,neutral,negative" --eval data/work/sevbench_sentiment_eval.jsonl \
    --labels data/work/sevbench_sentiment_eval_labels.jsonl --out runs/s29_sentiment_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("served_eval")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://localhost:9000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    valid = set(args.valid.split(","))
    texts = {r["id"]: r["text"] for r in
             (json.loads(ln) for ln in Path(args.eval).read_text().splitlines() if ln.strip())}
    labels = {r["id"]: r["choice"] for r in
              (json.loads(ln) for ln in Path(args.labels).read_text().splitlines() if ln.strip())}
    todo = [rid for rid in sorted(texts) if rid in labels and labels[rid]]
    logger.info("eval_rows=%d", len(todo))

    def one(rid):
        payload = json.dumps({"model": args.model, "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": texts[rid]}],
            "temperature": 0, "max_tokens": 4}).encode()
        req = urllib.request.Request(args.endpoint + "/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                word = json.loads(resp.read())["choices"][0]["message"]["content"]
            word = word.strip().strip(".,!\"'").lower()
            return rid, (word if word in valid else None)
        except Exception as err:
            logger.warning("row %s failed: %s", rid, err)
            return rid, None

    hits, n = 0, 0
    with Path(args.out).open("w") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rid, pred in pool.map(one, todo):
                n += 1
                hits += int(pred == labels[rid])
                f.write(json.dumps({"id": rid, "gold": labels[rid], "pred": pred},
                                   ensure_ascii=False) + "\n")
    logger.info("accuracy %.4f (n=%d)", hits / max(n, 1), n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
