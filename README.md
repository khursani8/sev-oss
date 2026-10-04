# sev: a System One decision stack for Malaysian language

One model, many questions. This repo packages a decision-model stack that
replaces per-task classifiers with a single System One model answering
choice, score, and yes/no questions over Malay text, with per-option
probabilities in one forward pass.

## Why

malaya (malaysia-ai) ships one specialized model per task, and our
adversarial audit (audit.md) measured its shipped sentiment model at
0.515 on independent native Malay text - a negation blind spot included -
while its published chart claims 0.994. Published in-distribution numbers
do not transfer. Everything in this repo is measured on our own held-out
native pools, reproducible from raw rows.

## Benchmark (native Malay, held-out splits, identical rows)

| Task | Our fine-tuned Qwen3-4B | Nox-4B zero-shot | malaya published |
|---|---|---|---|
| toxic | 0.870 | 0.874 | 0.818 (beaten) |
| sentiment | 0.876 | 0.766 | 0.994 (in-distribution) |
| emotion | 0.798 | 0.834 | 0.998 (in-distribution) |

## What is here

- datasets/ - eval splits + pools (first-party corpus, teacher-labeled,
  provenance documented) and the operator gold labels.
- scripts/nox_server.py - resident System One server (any Decision-2.0
  model) speaking the Typesafe envelope.
- scripts/jev_client.py - the decision client: fast tier (trained
  fields), decision tier (untrained questions), accuracy tier fallback.
- scripts/finetune_qwen_s17.py + scripts/join_s29_data.py - the
  fine-tune recipe and data join.
- scripts/served_eval.py + scripts/sevbench_score.py - eval harness.
- scripts/audit_malaya.py - the adversarial audit of malaya's shipped
  models (results in malaya_metric_analysis.md).
- tests/test_polarity.py - the polarity-flip robustness suite
  (negation cases included) against any system_one endpoint.

## Quickstart

1. Serve a decision model: `.venv-clef/bin/python scripts/nox_server.py --port 9525`
2. Run the polarity suite: `SEV_CLASSIFY_ENDPOINT=http://localhost:9525/v1 pytest tests/ -v`
3. Decide: `.venv/bin/python scripts/jev_client.py --state "..." --question sentiment:choice --decision-endpoint http://localhost:9525/v1`

MIT licensed. The malaya comparison is documented, reproducible, and
measured - see audit.md.

## Known limitations (measured, not hidden)

Nox-4B zero-shot scores 9/10 on the polarity suite: the single failure is
the negation flip ("Saya TIDAK marah, barang sampai dengan baik" reads
negative). malaya's shipped model fails the same case plus one more (8/10)
and scores 0.515 on our independent pool. Negation handling is the top
robustness gap for both stacks; fine-tuned adapters narrow it.
