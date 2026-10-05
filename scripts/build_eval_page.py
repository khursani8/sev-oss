#!/usr/bin/env python3
"""Build eval_browser.html: eval data + multi-model predictions + ensemble.

Assembles the held-out eval splits with gold and per-model predictions,
the 299-row ensemble gold with four labeler votes, and the malaya audit
into one self-contained HTML page with filters and search.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OSS = Path("/mnt/data/work/sev-oss")


def jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def by_id(rows: list, id_key: str = "id") -> dict:
    return {r[id_key]: r for r in rows}


def task_rows(task: str, models: dict) -> list:
    texts = by_id(jsonl(ROOT / f"data/work/sevbench_{task}_eval.jsonl"))
    golds = by_id(jsonl(ROOT / f"data/work/sevbench_{task}_eval_labels.jsonl"))
    model_preds = {name: by_id(rows) for name, rows in models.items()}
    texts = {r["id"]: r["text"] for r in jsonl(ROOT / f"data/work/sevbench_{task}_eval.jsonl")}
    golds = {r["id"]: r["choice"] for r in jsonl(ROOT / f"data/work/sevbench_{task}_eval_labels.jsonl")}
    model_preds = {name: by_id(rows) for name, rows in models.items()}
    out = []
    for rid in sorted(texts):
        if rid not in golds or not golds[rid]:
            continue
        row = {"id": rid, "text": texts[rid][:400], "gold": golds[rid]}
        all_match = True
        any_pred = False
        for name, preds in model_preds.items():
            p = preds.get(rid, {}).get("pred")
            row[name] = p
            if p is not None:
                any_pred = True
            if p != golds[rid]:
                all_match = False
        row["models_agree"] = all_match and any_pred
        out.append(row)
    return out


def gold_rows() -> list:
    sheets = {
        "v1": (ROOT / "data/work/gold_audit_v1.jsonl", "query"),
        "v2": (ROOT / "data/work/gold_audit_v2.jsonl", "text"),
        "v3_emotion": (ROOT / "data/work/gold_audit_v3_emotion.jsonl", "text"),
        "v3_toxic": (ROOT / "data/work/gold_audit_v3_toxic.jsonl", "text"),
    }
    labelers = {
        "fable": by_id(jsonl(ROOT / "data/work/ensemble_fable_v1v2.jsonl"), "row_id"),
        "nox": by_id(jsonl(ROOT / "data/work/ensemble_nox.jsonl"), "row_id"),
        "vega": by_id(jsonl(ROOT / "data/work/ensemble_vega.jsonl"), "row_id"),
        "codex": by_id(jsonl(ROOT / "data/work/ensemble/codex_v1.jsonl"
                             + "\n".join([]) ) ) if False else
                 {f"{r['sheet']}:{r['row_id']}": r["labels"] for r in
                  (json.loads(ln) for ln in
                   Path("/mnt/data/work/sev/data/work/ensemble_codex.jsonl").read_text().splitlines()
                   if ln.strip())},
    }
    ens = json.loads((ROOT / "reports/ensemble_gold.json").read_text())["ensemble"]
    out = []
    for sheet, (path, tkey) in sheets.items():
        for i, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row.get("sheet_id", row.get("id", i + 1))
            key = f"{sheet}:{rid}"
            entry = {"sheet": sheet, "row_id": rid,
                     "text": row.get(tkey, "")[:400], "labelers": {},
                     "ensemble": ens.get(key, {})}
            for name, lrows in labelers.items():
                entry["labelers"][name] = lrows.get(key, {})
            votes = [v for v in entry["labelers"].values() for v in v.values()]
            entry["unanimous"] = bool(votes) and len(set(votes)) == 1
            out.append(entry)
    return out


def main() -> int:
    tasks = {
        "sentiment": {"nox": jsonl(ROOT / "runs/sb_sentiment_nox.jsonl"),
                      "lux": jsonl(ROOT / "runs/sb_sentiment_lux.jsonl"),
                      "finetuned": jsonl(ROOT / "runs/s29_sentiment_eval_scored.jsonl")},
        "emotion": {"nox": jsonl(ROOT / "runs/sb_emotion_nox.jsonl"),
                    "lux": jsonl(ROOT / "runs/sb_emotion_lux.jsonl")},
        "toxic": {"nox": jsonl(ROOT / "runs/sb_toxic_nox.jsonl"),
                  "lux": jsonl(ROOT / "runs/sb_toxic_lux.jsonl")},
    }
    data = {
        "tasks": {t: {"rows": task_rows(t, models), "models": list(models)}
                  for t, models in tasks.items()},
        "gold": gold_rows(),
        "audit": json.loads((ROOT / "reports/malaya_audit_xlnet.json").read_text()),
        "ensemble_meta": {"labelers": ["fable", "nox", "vega", "codex"],
                          "note": "majority vote; unanimous = all four agree"},
    }
    template = Path(ROOT / "scripts/eval_page_template.html").read_text()
    page = template.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    out = OSS / "eval_browser.html"
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
