"""
Simple LLM benchmark POC.

Compares configured models on:
  1. Intent Classification
  2. Entity / Concept Resolution
  3. LLM-as-a-Judge

Usage:
    python benchmark.py
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent
ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Allows tests to inject a mock HTTP POST function.
http_post: Callable[..., httpx.Response] = httpx.post


def load_config(path: str = "config.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return substitute_env_values(config, os.environ)


def load_tests(path: str) -> dict:
    with open(ROOT / path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def substitute_env_values(value: Any, env: dict[str, str]) -> Any:
    """Replace ${ENV_VAR} placeholders used in config.yaml."""
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: env.get(match.group(1), match.group(0)), value)
    if isinstance(value, dict):
        return {key: substitute_env_values(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute_env_values(item, env) for item in value]
    return value


def apply_request_template(value: Any, model: str, prompt: str, api_key: str) -> Any:
    """Replace {{model}}, {{prompt}}, and {{api_key}} in request templates."""
    replacements = {
        "{{model}}": model,
        "{{prompt}}": prompt,
        "{{api_key}}": api_key,
    }

    if isinstance(value, str):
        result = value
        for placeholder, replacement in replacements.items():
            result = result.replace(placeholder, replacement)
        return result
    if isinstance(value, dict):
        return {
            key: apply_request_template(item, model, prompt, api_key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            apply_request_template(item, model, prompt, api_key) for item in value
        ]
    return value


ENTITY_CRITERIA = {
    "metric": {
        "REVENUE": "Revenue or sales amount",
        "PROFIT": "Profit or margin",
        "ORDERS": "Order count",
    },
    "customer_segment": {
        "ENTERPRISE": "Enterprise customers",
        "SMB": "Small and medium business customers",
        "CONSUMER": "Consumer customers",
        "ALL": "All customer segments",
    },
    "period": {
        "Q1": "First quarter",
        "Q2": "Second quarter",
        "Q3": "Third quarter",
        "Q4": "Fourth quarter",
    },
}


def build_jev_state(test: dict, benchmark_type: str) -> dict[str, Any]:
    """Build structured state for the SystemOne API. Content lives here, not in instructions."""
    state: dict[str, Any] = {"question": test["question"]}

    if test.get("context"):
        state["context"] = test["context"]

    if benchmark_type == "intent":
        state["task"] = "Classify the user intent for a Text-to-SQL system."

    if benchmark_type == "entity":
        state["task"] = "Identify the metric, customer segment, and period referenced in the question."

    if benchmark_type == "judge":
        state["task"] = "Decide which candidate answer is better."
        state["candidate_a"] = test["candidate_a"]
        state["candidate_b"] = test["candidate_b"]

    return state


def build_jev_questions(benchmark_type: str, test: dict) -> dict:
    """Build typed Choice questions using criteria only (no instructions field)."""
    if benchmark_type == "intent":
        return {
            "intent": {
                "type": "choice",
                "criteria": {option: None for option in test["options"]},
            }
        }

    if benchmark_type == "entity":
        questions = {}
        for field in test["expected"]:
            criteria = ENTITY_CRITERIA.get(field, {})
            questions[field] = {
                "type": "choice",
                "criteria": {value: None for value in criteria},
            }
        return questions

    if benchmark_type == "judge":
        return {
            "winner": {
                "type": "choice",
                "criteria": {
                    "A": test["candidate_a"],
                    "B": test["candidate_b"],
                },
            }
        }

    raise ValueError(f"Unsupported benchmark type for Jev: {benchmark_type}")


def build_jev_request(
    model_cfg: dict,
    benchmark_type: str,
    test: dict,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    api_key = os.environ.get(model_cfg["api_key_env"], "")
    headers = apply_request_template(
        copy.deepcopy(model_cfg["headers"]),
        model_cfg["model"],
        "",
        api_key,
    )
    body = {
        "state": build_jev_state(test, benchmark_type),
        "model": model_cfg["model"],
        "questions": build_jev_questions(benchmark_type, test),
    }
    return model_cfg["endpoint"], {str(k): str(v) for k, v in headers.items()}, body


def parse_jev_response(benchmark_type: str, response_json: dict | None) -> dict | None:
    if not response_json:
        return None

    answers = response_json.get("answers")
    if not isinstance(answers, dict):
        return None

    if benchmark_type == "intent":
        answer = answers.get("intent", {})
        choice = answer.get("choice")
        return {
            "intent": str(choice).upper() if choice is not None else None,
            "confidence": answer.get("confidence"),
        }

    if benchmark_type == "entity":
        entities = {}
        confidences = []
        for field, answer in answers.items():
            if not isinstance(answer, dict):
                continue
            choice = answer.get("choice")
            entities[field] = str(choice).upper() if choice is not None else None
            if answer.get("confidence") is not None:
                confidences.append(answer["confidence"])
        return {
            "entities": entities,
            "confidence": sum(confidences) / len(confidences) if confidences else None,
        }

    if benchmark_type == "judge":
        answer = answers.get("winner", {})
        winner = answer.get("choice")
        return {
            "winner": str(winner).upper() if winner is not None else None,
            "confidence": answer.get("confidence"),
        }

    return None


def build_request(model_cfg: dict, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
    api_key = os.environ.get(model_cfg["api_key_env"], "")

    endpoint = model_cfg["endpoint"]
    headers = apply_request_template(copy.deepcopy(model_cfg["headers"]), model_cfg["model"], prompt, api_key)
    body = apply_request_template(
        copy.deepcopy(model_cfg["request_template"]),
        model_cfg["model"],
        prompt,
        api_key,
    )

    if not isinstance(headers, dict):
        raise ValueError("Resolved headers must be a mapping")
    if not isinstance(body, dict):
        raise ValueError("Resolved request body must be a mapping")

    return endpoint, {str(k): str(v) for k, v in headers.items()}, body


def get_nested_value(data: Any, path: str | None) -> Any:
    """Read a dot-separated path such as usage.prompt_tokens or choices.0.message.content."""
    if data is None or not path:
        return None

    current = data
    for part in path.split("."):
        if current is None:
            return None
        if part.isdigit():
            if not isinstance(current, list):
                return None
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def parse_json_response(text: str) -> dict | None:
    if not text:
        return None

    text = text.strip()
    block_match = JSON_BLOCK_PATTERN.search(text)
    if block_match:
        text = block_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def calculate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    pricing: dict,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None

    input_cost = (input_tokens / 1_000_000) * pricing.get("input_per_million", 0)
    output_cost = (output_tokens / 1_000_000) * pricing.get("output_per_million", 0)
    return input_cost + output_cost


def sanitize_error(message: str | None, api_key: str) -> str | None:
    if not message or not api_key:
        return message
    return message.replace(api_key, "***REDACTED***")


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def call_model(
    model_name: str,
    model_cfg: dict,
    prompt: str,
    *,
    benchmark_type: str,
    test: dict,
) -> dict:
    api_key = os.environ.get(model_cfg["api_key_env"], "")
    if not api_key:
        return {
            "parsed": None,
            "latency_ms": 0.0,
            "status_code": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost": None,
            "error": f"Missing API key env var: {model_cfg['api_key_env']}",
        }

    if model_cfg.get("api_style") == "systemone":
        endpoint, headers, body = build_jev_request(model_cfg, benchmark_type, test)
    else:
        endpoint, headers, body = build_request(model_cfg, prompt)

    start = time.perf_counter()
    try:
        response = http_post(
            endpoint,
            headers=headers,
            json=body,
            timeout=60.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
    except httpx.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "parsed": None,
            "latency_ms": latency_ms,
            "status_code": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost": None,
            "error": sanitize_error(str(exc), api_key),
        }

    response_json: dict[str, Any] | None = None
    if response.content:
        try:
            parsed_body = response.json()
            if isinstance(parsed_body, dict):
                response_json = parsed_body
        except json.JSONDecodeError:
            response_json = None

    input_tokens = coerce_int(get_nested_value(response_json, model_cfg["input_tokens_path"]))
    output_tokens = coerce_int(get_nested_value(response_json, model_cfg["output_tokens_path"]))
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    cost = calculate_cost(input_tokens, output_tokens, model_cfg["pricing"])

    if model_cfg.get("api_style") == "systemone":
        parsed = parse_jev_response(benchmark_type, response_json)
    else:
        extracted = get_nested_value(response_json, model_cfg["response_path"])
        text = str(extracted) if extracted is not None else None
        parsed = parse_json_response(text) if text else None

    error = None
    if response.status_code >= 400:
        error = sanitize_error(f"HTTP {response.status_code}", api_key)
    elif parsed is None:
        error = "Could not parse model response"

    return {
        "parsed": parsed,
        "latency_ms": latency_ms,
        "status_code": response.status_code,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "error": error,
    }


def build_intent_prompt(test: dict) -> str:
    options = ", ".join(test["options"])
    context_block = ""
    if test.get("context"):
        context_block = f"Context:\n{test['context']}\n\n"

    return (
        "Classify the user's intent for a Text-to-SQL system.\n\n"
        f"{context_block}"
        f"Question:\n{test['question']}\n\n"
        f"Allowed intents:\n{options}\n\n"
        "Return JSON only in this exact shape:\n"
        '{\n  "intent": "INTENT_LABEL",\n  "confidence": 0.95\n}'
    )


def build_entity_prompt(test: dict) -> str:
    context_block = ""
    if test.get("context"):
        context_block = f"Context:\n{test['context']}\n\n"

    return (
        "Extract entities from the question for a Text-to-SQL system.\n\n"
        f"{context_block}"
        f"Question:\n{test['question']}\n\n"
        "Return JSON only in this exact shape:\n"
        "{\n"
        '  "entities": {\n'
        '    "metric": "METRIC_NAME",\n'
        '    "customer_segment": "SEGMENT_NAME",\n'
        '    "period": "PERIOD_NAME"\n'
        "  },\n"
        '  "confidence": 0.95\n'
        "}"
    )


def build_judge_prompt(test: dict) -> str:
    context_block = ""
    if test.get("context"):
        context_block = f"Context:\n{test['context']}\n\n"

    return (
        "Decide which candidate answer is better.\n\n"
        f"{context_block}"
        f"Question:\n{test['question']}\n\n"
        f"Candidate A:\n{test['candidate_a']}\n\n"
        f"Candidate B:\n{test['candidate_b']}\n\n"
        "Return JSON only in this exact shape:\n"
        '{\n  "winner": "A",\n  "confidence": 0.95\n}'
    )


def score_intent(expected: str, parsed: dict | None) -> tuple[Any, float | None, bool]:
    if not parsed:
        return None, None, False
    actual = parsed.get("intent")
    confidence = parsed.get("confidence")
    correct = str(actual).upper() == str(expected).upper() if actual is not None else False
    return actual, confidence, correct


def score_entity(expected: dict, parsed: dict | None) -> tuple[Any, float | None, bool]:
    if not parsed:
        return None, None, False

    actual_entities = parsed.get("entities")
    confidence = parsed.get("confidence")
    if not isinstance(actual_entities, dict):
        return None, confidence, False

    correct = True
    for key, expected_value in expected.items():
        actual_value = actual_entities.get(key)
        if str(actual_value).upper() != str(expected_value).upper():
            correct = False
            break

    return actual_entities, confidence, correct


def score_judge(expected: str, parsed: dict | None) -> tuple[Any, float | None, bool]:
    if not parsed:
        return None, None, False

    actual = parsed.get("winner")
    confidence = parsed.get("confidence")
    if actual is None:
        return None, confidence, False

    actual_normalized = str(actual).strip().upper()
    correct = actual_normalized == str(expected).strip().upper()
    return actual_normalized, confidence, correct


PROMPT_BUILDERS = {
    "intent": build_intent_prompt,
    "entity": build_entity_prompt,
    "judge": build_judge_prompt,
}

SCORERS = {
    "intent": score_intent,
    "entity": score_entity,
    "judge": score_judge,
}


def make_result_record(
    benchmark_type: str,
    test: dict,
    model_name: str,
    api_result: dict,
) -> dict:
    parsed = api_result["parsed"]
    expected = test["expected"]
    actual, confidence, correct = SCORERS[benchmark_type](expected, parsed)

    if api_result["error"]:
        correct = False

    return {
        "test_id": test["id"],
        "model": model_name,
        "question": test["question"],
        "context": test.get("context") or "",
        "expected": expected,
        "actual": actual,
        "confidence": confidence,
        "correct": correct,
        "latency_ms": round(api_result["latency_ms"], 2),
        "input_tokens": api_result["input_tokens"],
        "output_tokens": api_result["output_tokens"],
        "total_tokens": api_result["total_tokens"],
        "cost": api_result["cost"],
        "error": api_result["error"],
        "http_status": api_result["status_code"],
    }


def summarize_results(results: list[dict], model_names: list[str]) -> dict:
    summary: dict[str, dict] = {}

    for model_name in model_names:
        model_results = [row for row in results if row["model"] == model_name]
        total_tests = len(model_results)
        correct_tests = sum(1 for row in model_results if row["correct"])

        latencies = [row["latency_ms"] for row in model_results]
        costs = [row["cost"] for row in model_results if row["cost"] is not None]

        summary[model_name] = {
            "accuracy": round(correct_tests / total_tests, 4) if total_tests else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "total_cost": round(sum(costs), 6) if costs else None,
        }

    return summary


def run_benchmark(config: dict, tests: dict) -> None:
    results_dir = ROOT / config.get("results_dir", "results")
    results_dir.mkdir(exist_ok=True)

    enabled_models = config["enabled_models"]

    for benchmark_type in config["benchmarks"]:
        print(f"Running {benchmark_type} benchmark...")
        if benchmark_type not in tests:
            print(f"  Skipping {benchmark_type}: no section in {config['tests_file']}")
            continue

        benchmark_tests = tests[benchmark_type]
        prompt_builder = PROMPT_BUILDERS[benchmark_type]
        results: list[dict] = []

        for test in benchmark_tests:
            prompt = prompt_builder(test)

            for model_name in enabled_models:
                model_cfg = config["models"][model_name]
                print(f"  {test['id']} -> {model_name}")
                api_result = call_model(
                    model_name,
                    model_cfg,
                    prompt,
                    benchmark_type=benchmark_type,
                    test=test,
                )
                results.append(make_result_record(benchmark_type, test, model_name, api_result))

        output = {
            "benchmark": benchmark_type,
            "results": results,
            "summary": summarize_results(results, enabled_models),
        }

        output_path = results_dir / f"{benchmark_type}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
            handle.write("\n")

        print(f"  Saved {output_path}")


def main() -> None:
    load_dotenv()
    config = load_config()
    tests = load_tests(config["tests_file"])
    run_benchmark(config, tests)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
