"""Pluggable classify fixture: point SEV_CLASSIFY_ENDPOINT at a resident
nox_server (or any server speaking the system_one envelope) to run the
polarity suite against your model. Without it the suite skips."""
import os
import json
import urllib.request

import pytest


@pytest.fixture
def classify():
    endpoint = os.environ.get("SEV_CLASSIFY_ENDPOINT")
    if not endpoint:
        pytest.skip("SEV_CLASSIFY_ENDPOINT not set")
    def classify(text: str) -> str:
        payload = json.dumps({"model": "test", "state": text, "questions": {
            "sentiment": {"type": "choice",
                          "instructions": "Overall sentiment of the message.",
                          "criteria": {"positive": "Positive sentiment",
                                       "neutral": "Neutral sentiment",
                                       "negative": "Negative sentiment"}}}}).encode()
        req = urllib.request.Request(endpoint.rstrip("/") + "/systemone", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            ans = json.loads(resp.read())["answers"]["sentiment"]
        return ans.get("choice")
    return classify
