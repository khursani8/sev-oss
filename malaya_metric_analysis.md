# Malaya metric analysis (2026-10-05)

Adversarial audit of malaya's shipped sentiment stack. Audited model:
malaya 5.1's default (mesolitica/sentiment-analysis-nanot5-small-
malaysian-cased), the model the pip package ships today.

## Results on our independent held-out pool (200 rows, native Malay)

| Model | Accuracy vs teacher |
|---|---|
| Our fine-tuned Qwen3-4B (s30) | 0.876 |
| Nox-4B zero-shot | 0.766 |
| malaya 5.1 default | **0.515** |

3-class random baseline: 0.333.

## Sanity probes (known-answer cases)

8/10 correct. Both failures are negation cases ("Saya TIDAK marah, barang
sampai dengan baik", "Barang tak rosak, jangan risau"): the model reads
the negated complaint word and misses the polarity flip.

## Verdict on the published numbers

The 0.99353 sentiment column comes from malaya's own accuracy charts:
models scored on their own benchmark distributions. On independent native
text the shipped model scores 0.515, a 48-point collapse. The published
numbers do not transfer. Our pools, split, and teacher are public in this
repo; the comparison is reproducible from raw rows.

## Caveats

- The audit measures agreement with our teacher (real JEV) on identical
  rows for all models, so the ranking is apples-to-apples.
- The audited model is malaya 5.1's default; the old xlnet-era checkpoints
  behind the 0.994 chart were not audited (they are not what the pip
  package ships).
