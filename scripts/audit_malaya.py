#!/usr/bin/env python3
"""Adversarial audit of malaya's sentiment models vs our sevBench rows."""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audit_malaya")

PROBES = [
    ("Parcel saya sudah 8 hari tak sampai, saya marah!", "negative"),
    ("Servis sungguh teruk, wang saya hilang begitu sahaja.", "negative"),
    ("Terima kasih, barangan sampai dengan cepat dan selamat!", "positive"),
    ("Produk ini sangat bagus, saya berpuas hati.", "positive"),
    ("Bila parcel saya akan sampai?", "neutral"),
    ("Saya ingin menanya tentang harga bulanan.", "neutral"),
    ("Saya TIDAK marah, barang sampai dengan baik.", "positive"),
    ("Barang tak rosak, jangan risau.", "positive"),
    ("Aduan saya masih belum diselesaikan, kecewa.", "negative"),
    ("Kualiti teruk sangat, menyesal beli.", "negative"),
]


def main() -> int:
    import inspect
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    import malaya

    model = malaya.sentiment.huggingface()
    logger.info("model loaded")

    probe_texts = [p[0] for p in PROBES]
    preds = model.predict(probe_texts)
    probe_rows, probe_hits = [], 0
    for (text, expected), pred in zip(PROBES, preds):
        pred_label = pred.strip().lower() if isinstance(pred, str) else str(pred)
        ok = pred_label == expected
        probe_hits += int(ok)
        probe_rows.append({"text": text, "expected": expected,
                           "predicted": pred_label, "match": ok})
        logger.info("probe: expected=%s predicted=%s match=%s", expected, pred_label, ok)
    logger.info("sanity probes: %d/%d", probe_hits, len(PROBES))

    eval_rows = [json.loads(ln) for ln in
                 Path("data/work/sevbench_sentiment_eval_labels.jsonl").read_text().splitlines()
                 if ln.strip()]
    pool = {r["id"]: r["text"] for r in
            (json.loads(ln) for ln in
             Path("data/work/sevbench_sentiment_eval.jsonl").read_text().splitlines()
             if ln.strip())}
    rng = random.Random(5)
    sample = rng.sample([r for r in eval_rows if r["choice"]],
                        min(200, len(eval_rows)))
    batch_texts = [pool[r["id"]] for r in sample]
    golds = [r["choice"] for r in sample]

    t0 = time.time()
    preds = model.predict(batch_texts)
    dt = time.time() - t0
    hits, n, rows = 0, 0, []
    for r, gold, pred in zip(sample, golds, preds):
        pred_label = pred.strip().lower() if isinstance(pred, str) else str(pred)
        ok = pred_label == gold
        hits += int(ok)
        n += 1
        rows.append({"id": r["id"], "gold": gold, "malaya": pred_label, "match": ok})
    logger.info("eval accuracy vs our teacher: %.4f (n=%d, %.2f s/row)",
                hits / max(n, 1), n, dt / max(n, 1))
    summary = {"model": "mesolitica/sentiment-analysis-nanot5-small-malaysian-cased (malaya 5.1 default)",
               "sanity_probes": {"hits": probe_hits, "n": len(PROBES), "rows": probe_rows},
               "eval_split": {"n": n, "accuracy_vs_teacher": round(hits / max(n, 1), 4),
                              "seconds_per_row": round(dt / max(n, 1), 3)},
               "comparison": {"nox_zero_shot": 0.7656, "finetuned_ours": 0.8756,
                              "malaya_published_xlnet": 0.99353},
               "rows": rows}
    Path("reports/malaya_audit_xlnet.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote reports/malaya_audit_xlnet.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
