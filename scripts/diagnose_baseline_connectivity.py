"""Single-case provider diagnostic; no gold and no method changes."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stategraph_generalization_unseen10_v1"


class DiagnosticTimeout(TimeoutError):
    pass


def alarm_handler(_signum, _frame):
    raise DiagnosticTimeout("single provider call exceeded 25 seconds")


def redact(value):
    if isinstance(value, str):
        return value[:300] + ("..." if len(value) > 300 else "")
    if isinstance(value, list):
        return [redact(item) for item in value[:3]]
    if isinstance(value, dict):
        return {key: ("<redacted>" if "key" in key.casefold() else redact(item)) for key, item in value.items()}
    return value


def versions(method):
    packages = ["openai"] + (["mem0"] if method == "mem0" else ["agentic_memory"])
    result = {"python": sys.executable}
    for name in packages:
        try:
            module = importlib.import_module(name)
            result[name] = {"version": getattr(module, "__version__", "unknown"), "file": getattr(module, "__file__", None)}
        except Exception as exc:
            result[name] = f"{type(exc).__name__}: {exc}"
    return result


def wrap_request(client, label):
    original = client.create
    request = {}

    def create(**kwargs):
        request.update(redact(kwargs))
        return original(**kwargs)

    client.create = create
    return request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("mem0", "amem"))
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "external_baselines" / "e2e_validation"))
    from adapters import create_adapter

    manifest = json.loads((OUT / "METHOD_MANIFEST.json").read_text())
    case = next(item for item in manifest["cases"] if item["case_id"] == "SCB_012")
    print(json.dumps({
        "method": args.method,
        "case_id": case["case_id"],
        "python": sys.executable,
        "versions": versions(args.method),
        "model": "gpt-5-nano",
        "reasoning_effort": "minimal",
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "timeout_seconds": 25,
        "max_retries_expected": 0,
    }, ensure_ascii=False))

    adapter = create_adapter(args.method, OUT / "diagnostic" / args.method)
    adapter.reset()
    request = {}
    if args.method == "mem0":
        request = wrap_request(adapter.memory.llm.client.chat.completions, "mem0")
    else:
        request = wrap_request(adapter.memory.llm_controller.llm.client.chat.completions, "amem")
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, 25)
    stage = "init"
    started = time.monotonic()
    try:
        stage = "add_memory" if args.method == "mem0" else "add_note"
        adapter.add_memory(case["history"][0]["text"])
        print(json.dumps({"status": "PASS", "stage": stage, "elapsed_seconds": time.monotonic() - started, "request": request}, ensure_ascii=False))
    except DiagnosticTimeout as exc:
        print(json.dumps({"status": "FAIL", "failure_class": "READ_TIMEOUT", "stage": stage, "error": str(exc), "elapsed_seconds": time.monotonic() - started, "request": request}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "failure_class": "OTHER", "stage": stage, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started, "request": request}, ensure_ascii=False))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if hasattr(adapter, "close"):
            try:
                adapter.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
