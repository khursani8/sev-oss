#!/usr/bin/env python3
"""Resident HTTP server exposing a Decision-2.0 model at /v1/systemone.

Mirrors the Typesafe SystemOne request/response envelope so JEV-shaped
clients can use the local model as a drop-in decision tier. Questions
pass through unmodified; the model answers every question in one forward
pass with per-option probabilities.

Usage:
  .venv-clef/bin/python scripts/nox_server.py \
    --model vllm-sr/Decision-2.0-Nox-4B --port 9020
"""
from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("nox_server")


STATE: dict = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("%s", fmt % args)

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/v1/models", "/v1"):
            self._json(200, {"model": STATE.get("model_name", ""), "ready": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/systemone"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        state = body.get("state", "")
        questions = body.get("questions") or {}
        if not questions:
            self._json(400, {"error": "no questions"})
            return
        with torch.inference_mode():
            result = STATE["model"].system_one(state=state, questions=questions)
        self._json(200, {"model": STATE.get("model_name", ""),
                         "answers": result.get("answers", {}),
                         "usage": result.get("usage", {})})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="vllm-sr/Decision-2.0-Nox-4B")
    ap.add_argument("--port", type=int, default=9020)
    args = ap.parse_args()

    from transformers import AutoModel

    STATE["model"] = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, dtype="bfloat16").to("cuda").eval()
    STATE["model_name"] = args.model
    logger.info("loaded %s, serving on :%d", args.model, args.port)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
