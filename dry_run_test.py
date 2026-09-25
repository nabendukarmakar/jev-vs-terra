"""Dry-run test for benchmark.py using mocked HTTP responses."""

from __future__ import annotations

import json as json_module
import os
import re
import sys
from pathlib import Path

import httpx

import benchmark

ROOT = Path(__file__).parent


def _openai_style_content(prompt: str) -> str:
    if "Classify the user's intent" in prompt:
        intent_match = re.search(r"Allowed intents:\n(.+)", prompt, re.DOTALL)
        first_intent = "CUSTOMER_ANALYSIS"
        if intent_match:
            first_intent = intent_match.group(1).split(",")[0].strip()
        return json_module.dumps({"intent": first_intent, "confidence": 0.95})

    if "Extract entities" in prompt:
        return json_module.dumps(
            {
                "entities": {
                    "metric": "REVENUE",
                    "customer_segment": "ENTERPRISE",
                    "period": "Q2",
                },
                "confidence": 0.95,
            }
        )

    if "Decide which candidate answer is better" in prompt:
        return json_module.dumps({"winner": "B", "confidence": 0.95})

    return json_module.dumps({"answer": "UNKNOWN", "confidence": 0.5})


def _jev_systemone_response(body: dict) -> dict:
    questions = body.get("questions", {})
    state = body.get("state", {})

    if "intent" in questions:
        criteria = questions["intent"]["criteria"]
        first_choice = next(iter(criteria))
        return {
            "model": "jev-1.13.0",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": first_choice,
                    "confidence": 0.95,
                    "probabilities": {key: 0.0 for key in criteria},
                }
            },
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

    if "winner" in questions:
        return {
            "model": "jev-1.13.0",
            "answers": {
                "winner": {
                    "type": "choice",
                    "choice": "B",
                    "confidence": 0.95,
                    "probabilities": {"A": 0.1, "B": 0.9},
                }
            },
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

    answers = {}
    for field, question in questions.items():
        criteria = question["criteria"]
        first_choice = next(iter(criteria))
        answers[field] = {
            "type": "choice",
            "choice": first_choice,
            "confidence": 0.95,
            "probabilities": {key: 0.0 for key in criteria},
        }

    return {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def mock_http_post(
    url: str,
    headers: dict | None = None,
    json: dict | None = None,
    timeout: float = 60.0,
) -> httpx.Response:
    assert json is not None
    payload_text = json_module.dumps(json)
    assert "{{model}}" not in payload_text
    assert "{{prompt}}" not in payload_text
    assert "{{api_key}}" not in payload_text
    assert '"expected"' not in payload_text

    if "systemone" in url or "questions" in json:
        assert "instructions" not in json_module.dumps(json.get("questions", {}))
        return httpx.Response(200, json=_jev_systemone_response(json))

    prompt = json["messages"][-1]["content"]
    content = _openai_style_content(prompt)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
    )


def assert_jev_request_shape() -> None:
    config = benchmark.load_config()
    tests = benchmark.load_tests("tests.yaml")
    os.environ.setdefault("TYPESAFE_API_KEY", "fake-typesafe-key")

    _, _, body = benchmark.build_jev_request(config["models"]["jev"], "intent", tests["intent"][0])
    assert isinstance(body["state"], dict)
    assert body["state"]["question"]
    assert body["model"] == "jev-latest"
    assert body["questions"]["intent"]["type"] == "choice"
    assert "criteria" in body["questions"]["intent"]
    assert "instructions" not in body["questions"]["intent"]
    assert "expected" not in json_module.dumps(body)


def main() -> None:
    os.environ["TYPESAFE_API_KEY"] = "fake-typesafe-key"
    os.environ["OPENAI_API_URL"] = "https://api.openai.com/v1"
    os.environ["OPENAI_API_KEY"] = "fake-openai-key"

    benchmark.http_post = mock_http_post

    assert_jev_request_shape()
    benchmark.main()

    results_dir = ROOT / "results"
    for filename in ["intent.json", "entity.json", "judge.json"]:
        path = results_dir / filename
        assert path.exists(), f"Missing result file: {path}"
        data = json_module.loads(path.read_text(encoding="utf-8"))
        assert "results" in data
        assert "summary" in data
        print(f"OK {path} ({len(data['results'])} results)")

    results_text = " ".join(path.read_text(encoding="utf-8") for path in results_dir.glob("*.json"))
    assert "fake-typesafe-key" not in results_text
    assert "fake-openai-key" not in results_text
    print("Dry run passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"Dry run failed: {exc}", file=sys.stderr)
        sys.exit(1)
