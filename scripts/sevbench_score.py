#!/usr/bin/env python3
"""Score a Decision-2.0 model on a sevBench eval split.

Either posts to a resident nox_server endpoint (--endpoint) or loads the
model directly (--load). Accuracy against the split's teacher labels.

Usage:
  .venv-clef/bin/python scripts/sevbench_score.py --task toxic \
    --model vllm-sr/Decision-2.0-Lux-9B --eval data/work/sevbench_toxic_eval.jsonl \
    --labels data/work/sevbench_toxic_eval_labels.jsonl --out runs/sb_toxic_lux.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sevbench_score")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", default="configs/sevbench_tasks.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModel

    question = json.loads(Path(args.config).read_text())[args.task]["question"]
    texts = {r["id"]: r["text"] for r in
             (json.loads(ln) for ln in Path(args.eval).read_text().splitlines()
              if ln.strip())}
    labels = {r["id"]: r["choice"] for r in
              (json.loads(ln) for ln in Path(args.labels).read_text().splitlines()
               if ln.strip())}
    todo = [rid for rid in sorted(texts) if rid in labels and labels[rid]]
    logger.info("task=%s model=%s eval_rows=%d", args.task, args.model, len(todo))

    model = None
    if not args.endpoint:
        t0 = time.time()
        model = AutoModel.from_pretrained(
            args.model, trust_remote_code=True, dtype="bfloat16").to("cuda").eval()
        logger.info("loaded %s in %.1fs", args.model, time.time() - t0)

    hits, n = 0, 0
    with Path(args.out).open("w") as f:
        for rid in todo:
            if args.endpoint:
                payload = json.dumps({"model": "score", "state": texts[rid],
                                      "questions": {args.task: question}}).encode()
                req = urllib.request.Request(
                    args.endpoint.rstrip("/") + "/systemone", data=payload,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    ans = json.loads(resp.read())["answers"][args.task]
            else:
                with torch.inference_mode():
                    result = model.system_one(state=texts[rid],
                                              questions={args.task: question})
                ans = result["answers"][args.task]
            pred = ans.get("choice")
            ok = pred == labels[rid]
            hits += int(ok)
            n += 1
            f.write(json.dumps({"id": rid, "gold": labels[rid], "pred": pred,
                                "match": ok}, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                logger.info("scored %d/%d", n, len(todo))
            logger.debug("row done")
    logger.info("accuracy %.4f (n=%d)", hits / max(n, 1), n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
