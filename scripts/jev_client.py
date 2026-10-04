#!/usr/bin/env python3
"""JEV-compatible client for the sev Malay+EN decision system (two tiers).

Mirrors the TypeSafe Jev request/response shape:

  decide(state, questions) -> {"model": ..., "answers": {qid: {...}}}

Any question with any options works. Two tiers answer them:

  fast tier    — trained needle3 enum fields (configs/demo_tools.json).
                 Real logits, per-season temperature, millisecond latency.
  accuracy tier — ANY other question: each option is scored from the served
                 4B model's own logprobs (one completions call: the prompt
                 asks for the option index; the top-20 first-token logprobs
                 carry every index token; softmax over them gives the
                 distribution). Requires escalate_endpoint for these.

Types: choice, score (probability-weighted position + legend), noul
(P(yes) for yes/no options).

CLI:
  .venv/bin/python scripts/jev_client.py \
    --state "bil eletrik tinggi sangat" \
    --question sentiment:choice --question language:choice \
    --question priority:choice:'Which priority?|1=low,2=medium,3=high' \
    [--adapter runs/adapter_s6.safetensors] \
    [--escalate-endpoint http://localhost:9000/v1]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".venv" / "lib" / "python3.12" / "site-packages"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("jev_client")

ACC_SYSTEM = ("You answer a question about the user's message by replying with "
              "exactly the option's number and nothing else.")


def confidence_of(probs: dict) -> float:
    k = len(probs)
    return max(0.0, min(1.0, (k * max(probs.values()) - 1) / (k - 1)))


def softmax(d: dict) -> dict:
    import math

    m = max(d.values())
    exps = {k: math.exp(v - m) for k, v in d.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


class JevDecisionClient:
    def __init__(self, adapter: str | None = None,
                 tools_path: str = "configs/demo_tools.json",
                 accuracy_endpoint: str | None = None,
                 accuracy_model: str = "judge",
                 decision_endpoint: str | None = None,
                 escalate_threshold: float = 0.6):
        import jax
        import jax.numpy as jnp
        from needle.model.architecture import SimpleAttentionNetwork
        from needle.model.finetune import merge_lora, render_example
        from needle.model.run import load_checkpoint
        from needle.model.tokenizer import BOS_ID, EOS_ID, PAD_ID, get_tokenizer

        self.tools = json.loads((ROOT / tools_path).read_text())
        self.fields: dict[str, tuple[str, list]] = {}
        for tool in self.tools:
            props = (tool.get("parameters") or {}).get("properties", {})
            for name, spec in props.items():
                if spec.get("enum"):
                    self.fields[name] = (tool["name"], list(spec["enum"]))
        self.jax, self.jnp = jax, jnp
        self.BOS_ID, self.EOS_ID, self.PAD_ID = BOS_ID, EOS_ID, PAD_ID
        self.render_example = render_example
        self.accuracy_endpoint = accuracy_endpoint
        self.accuracy_model = accuracy_model
        self.decision_endpoint = decision_endpoint

        self._decode_fn_cache: dict = {}
        self._temperatures: dict[str, float] = {}
        self.season_tag = (re.search(r"s(\d+)", Path(adapter).name).group(1)
                           if adapter else None)

        base = str(ROOT / "runs" / "needle3.safetensors")
        params, config = load_checkpoint(base)
        if adapter:
            adapter_path = str(ROOT / adapter) if not Path(adapter).is_absolute() else adapter
            from needle.model.checkpoints import read_adapter
            loaded = read_adapter(adapter_path)
            lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
                    for k, v in loaded["lora"].items()}
            params = merge_lora(params, lora, loaded["scale"])
            logger.info("merged adapter %s (%d groups)", adapter_path, len(lora))
        self.params = params
        self.model = SimpleAttentionNetwork(config)
        self.tokenizer = get_tokenizer(config.vocab_size)

    def _temperature(self, field: str) -> float:
        if field not in self._temperatures:
            candidates = []
            if self.season_tag:
                candidates.append(ROOT / "reports" / f"fit_{field}_s{self.season_tag}.json")
            candidates.append(ROOT / "reports" / f"fit_{field}.json")
            for candidate in candidates:
                if candidate.exists():
                    t = json.loads(candidate.read_text()).get("temperature", 1.0)
                    logger.info("temperature for %s: %.4f (%s)", field, t, candidate.name)
                    self._temperatures[field] = t
                    return t
            raise SystemExit(
                f"no calibrated temperature for field {field!r} (looked for "
                f"{candidates[0].name}); answer probabilities would be unscaled. "
                f"Run a season calibration or pass an adapter with a fit file.")
        return self._temperatures[field]

    def _score_field(self, state: str, field: str) -> dict:
        import math

        tool_name, options = self.fields[field]
        scores = {}
        for option in options:
            answer = {"name": tool_name, "arguments": {field: option}}
            prompt, target = self.render_example({"query": state, "tools": self.tools,
                                                  "answers": [answer]})
            prompt_ids = [self.BOS_ID] + self.tokenizer.encode(prompt)
            target_ids = self.tokenizer.encode(target) + [self.EOS_ID]
            total = len(prompt_ids) + len(target_ids)
            buf_len = min(self.model.config.max_seq_len,
                          max(128, (total + 127) // 128 * 128))
            if total > buf_len:
                raise ValueError(f"example too long: {total} tokens")
            buffer = self.jnp.full((1, buf_len), self.PAD_ID, dtype=self.jnp.int32)
            buffer = buffer.at[0, :len(prompt_ids)].set(
                self.jnp.array(prompt_ids, dtype=self.jnp.int32))
            buffer = buffer.at[0, len(prompt_ids):total].set(
                self.jnp.array(target_ids, dtype=self.jnp.int32))
            key = id(self.model)
            if key not in self._decode_fn_cache:
                self._decode_fn_cache[key] = self.jax.jit(
                    lambda p, t: self.model.apply({"params": p}, t))
            logprobs = self.jax.nn.log_softmax(
                self._decode_fn_cache[key](self.params, buffer)[0].astype(self.jnp.float32),
                axis=-1)
            positions = self.jnp.arange(len(prompt_ids) - 1, total - 1)
            scores[option] = float(self.jnp.sum(logprobs[positions, self.jnp.array(target_ids)]))

        max_logp = max(scores.values())
        exps = {o: math.exp(s - max_logp) for o, s in scores.items()}
        z = sum(exps.values())
        probs = {o: exps[o] / z for o in options}
        t = self._temperature(field)
        logps = {o: math.log(max(v, 1e-12)) / t for o, v in probs.items()}
        exps2 = {o: math.exp(v) for o, v in logps.items()}
        z2 = sum(exps2.values())
        return {o: v / z2 for o, v in exps2.items()}

    def _accuracy_tier(self, state: str, instructions: str,
                       options: list[str]) -> dict:
        """Score arbitrary options from the served 4B's first-token logprobs."""

        numbered = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
        prompt = (f"{instructions}\n\nMessage:\n{state}\n\nOptions:\n{numbered}\n\n"
                  "Reply with the option number only. The correct option number is")
        payload = json.dumps({"model": self.accuracy_model, "prompt": prompt,
                              "temperature": 0, "max_tokens": 8,
                              "logprobs": 20}).encode()
        req = urllib.request.Request(
            self.accuracy_endpoint.rstrip("/") + "/completions", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        lp_field = body["choices"][0].get("logprobs") or {}
        positions = lp_field.get("top_logprobs") or []
        logps: dict[int, float] = {}
        for pos in positions:
            for tok, lp in pos.items():
                clean = str(tok).strip()
                if not (clean.isascii() and clean.isdigit()):
                    continue
                if clean.isdigit() and int(clean) < len(options):
                    logps.setdefault(int(clean), float(lp))
            if len(logps) == len(options):
                break
        floor = min(logps.values()) - 5.0 if logps else -20.0
        for i in range(len(options)):
            logps.setdefault(i, floor)
        return softmax({options[i]: logps[i] for i in range(len(options))})

    def _decision_tier(self, state: str, qid: str, spec: dict, qtype: str,
                       options: list[str]) -> dict:
        """Answer an untrained question on the local Decision-2.0 tier."""
        if qtype == "noul":
            question = {"type": "noul",
                        "instructions": spec.get("instructions",
                                                 f"Answer yes or no for {qid}.")}
        elif qtype == "score":
            question = {"type": "score",
                        "instructions": spec.get("instructions",
                                                 f"Score for {qid}"),
                        "criteria": [str(o) for o in options]}
        else:
            question = {"type": "choice",
                        "instructions": spec.get("instructions",
                                                 f"Choose the best option for {qid}"),
                        "criteria": {o: o for o in options}}
        payload = json.dumps({"model": "decision-local", "state": state,
                              "questions": {qid: question}}).encode()
        req = urllib.request.Request(
            self.decision_endpoint.rstrip("/") + "/systemone", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        ans = body["answers"][qid]
        if qtype == "noul":
            return {"type": "noul", "tier": "decision",
                    "noul": float(ans.get("noul", ans.get("probability", 0.0)))}
        if qtype == "score":
            probs = {str(i): round(float(p), 4) for i, p in
                     enumerate(ans.get("probabilities", {}).values())}
            score = sum(i * p for i, p in enumerate(probs.values()))
            return {"type": "score", "tier": "decision", "score": round(score, 3),
                    "legend": {str(i): o for i, o in enumerate(options)},
                    "probabilities": probs,
                    "confidence": round(float(ans.get("confidence", 0.0)), 4)}
        probs = {k: round(float(v), 4)
                 for k, v in ans.get("probabilities", {}).items()}
        return {"type": "choice", "tier": "decision",
                "choice": ans.get("choice", max(probs, key=probs.get)),
                "probabilities": probs,
                "confidence": round(float(ans.get("confidence", 0.0)), 4)}

    def decide(self, state: str, questions: dict) -> dict:
        answers = {}
        for qid, spec in questions.items():
            qtype = spec.get("type", "choice")
            field = spec.get("field", qid)
            known = field in self.fields
            options = (self.fields[field][1] if known
                       else spec.get("options") or list((spec.get("criteria") or {}).keys()))
            if not options:
                raise SystemExit(f"question {qid!r}: no options and no trained field")

            if known:
                probs = self._score_field(state, field)
                tier = "fast"
                if spec.get("criteria") or spec.get("options"):
                    requested = spec.get("options") or list((spec.get("criteria") or {}).keys())
                    if sorted(requested) != sorted(options):
                        raise SystemExit(
                            f"question {qid!r}: requested options mismatch the "
                            f"trained schema {sorted(options)}")
            else:
                if self.decision_endpoint:
                    answers[qid] = self._decision_tier(
                        state, qid, spec, qtype, options)
                    continue
                if not self.accuracy_endpoint:
                    raise SystemExit(
                        f"question {qid!r} is not a trained field; pass "
                        "--escalate-endpoint to answer arbitrary questions on the "
                        "accuracy tier")
                instructions = spec.get("instructions", f"Choose the best option for {qid}")
                probs = self._accuracy_tier(state, instructions, options)
                tier = "accuracy"

            chosen = max(probs, key=probs.get)
            conf = confidence_of(probs)
            if qtype == "choice":
                answers[qid] = {"type": "choice", "tier": tier, "choice": chosen,
                                "probabilities": {k: round(v, 4) for k, v in probs.items()},
                                "confidence": round(conf, 4)}
            elif qtype == "score":
                score = sum(i * p for i, p in enumerate(probs.values()))
                answers[qid] = {"type": "score", "tier": tier, "score": round(score, 3),
                                "legend": {str(i): o for i, o in enumerate(options)},
                                "probabilities": {str(i): round(p, 4) for i, p in
                                                  enumerate(probs.values())},
                                "confidence": round(conf, 4)}
            elif qtype == "noul":
                lowered = {o.lower(): o for o in options}
                if set(lowered) != {"yes", "no"}:
                    raise SystemExit(f"question {qid!r}: noul needs yes/no options, "
                                     f"got {sorted(options)}")
                answers[qid] = {"type": "noul", "tier": tier,
                                "noul": probs[lowered["yes"]]}
            else:
                raise SystemExit(f"unsupported type {qtype!r}")
        return {"model": "sev-needle3-two-tier", "answers": answers}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True)
    ap.add_argument("--question", action="append", default=[],
                    help="field:type (trained) or field:type:'instructions|opt1|opt2' "
                         "(arbitrary, answered on the accuracy tier)")
    ap.add_argument("--adapter", default="runs/adapter_s6.safetensors")
    ap.add_argument("--tools", default="configs/demo_tools.json")
    ap.add_argument("--escalate-endpoint", default=None)
    ap.add_argument("--escalate-model", default="judge")
    ap.add_argument("--decision-endpoint", default=None)

    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    questions: dict = {}
    for q in args.question:
        parts = q.split(":", 2)
        name, qtype = parts[0], parts[1] if len(parts) > 1 else "choice"
        spec: dict = {"type": qtype}
        if qtype == "noul":
            spec["options"] = ["no", "yes"]
        if len(parts) > 2:
            payload = parts[2]
            instr, _, opts = payload.partition("|")
            if opts:
                spec["options"] = opts.split(",")
                spec["instructions"] = instr
            else:
                spec["instructions"] = instr
        questions[name] = spec

    client = JevDecisionClient(adapter=args.adapter, tools_path=args.tools,
                               accuracy_endpoint=args.escalate_endpoint,
                               accuracy_model=args.escalate_model,
                               decision_endpoint=args.decision_endpoint)
    response = client.decide(args.state, questions)
    rendered = json.dumps(response, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
