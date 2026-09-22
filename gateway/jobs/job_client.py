"""Unified Gateway job API client helper.

Submits task jobs through /api/v1/job/* and polls until terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

try:
    from gateway.extraction.command_extraction import (
        ExtractionError,
        extract_main_invocation,
    )
except ImportError:
    try:
        from gateway.command_extraction import ExtractionError, extract_main_invocation
    except ImportError:
        from command_extraction import ExtractionError, extract_main_invocation


_MAX_RETRIES = int(os.environ.get("GW_CLIENT_MAX_RETRIES", "100"))
_REQUEST_TIMEOUT_SEC = float(os.environ.get("GW_CLIENT_REQUEST_TIMEOUT_SEC", "60.0"))
_RETRY_BASE_DELAY_SEC = float(os.environ.get("GW_CLIENT_RETRY_BASE_SEC", "1.0"))
_RETRY_MAX_DELAY_SEC = float(os.environ.get("GW_CLIENT_RETRY_MAX_SEC", "30.0"))


def _request_json(
    method: str, url: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SEC) as resp:
                body = resp.read().decode("utf-8") if resp else ""
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                body = exc.read().decode("utf-8") if exc.fp else ""
                raise RuntimeError(f"{method} {url} failed ({exc.code}): {body}") from exc
            last_exc = exc
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as exc:
            last_exc = exc
        if attempt < _MAX_RETRIES:
            delay = min(_RETRY_BASE_DELAY_SEC * (2 ** min(attempt, 5)), _RETRY_MAX_DELAY_SEC)
            time.sleep(delay)
    raise RuntimeError(
        f"{method} {url} failed after {_MAX_RETRIES + 1} attempts: {last_exc}"
    ) from last_exc


def _load_payload(path: str) -> dict[str, Any]:
    if not path:
        raise ValueError("--payload-file is required")
    if path == "-":
        payload = json.loads(sys.stdin.read())
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("payload file must contain a JSON object")
    return dict(payload)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    if not path or path == "-":
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_nextflow_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    argv_value = out.get("argv")
    if isinstance(argv_value, list) and argv_value:
        return out
    if argv_value not in (None, []):
        return out

    command_script = str(out.get("command_script") or "")
    if not command_script:
        raise ValueError("cannot prepare payload: command_script is required when argv is missing")
    main_script = str(out.get("main_script") or "")
    if not main_script:
        raise ValueError("cannot prepare payload: main_script is required when argv is missing")
    workdir = str(out.get("workdir") or "")
    if not workdir:
        raise ValueError("cannot prepare payload: workdir is required when argv is missing")

    try:
        tool_cwd, argv = extract_main_invocation(command_script, main_script, workdir)
    except ExtractionError as exc:
        raise ValueError(f"cannot prepare payload: {exc}") from exc
    out["argv"] = argv
    if tool_cwd:
        out["tool_cwd"] = tool_cwd
    return out


def _cancel(endpoint: str, job_id: str) -> None:
    with suppress(Exception):
        _request_json("POST", f"{endpoint}/api/v1/job/cancel/{job_id}")


def _error_message(status: dict[str, Any]) -> str:
    if status.get("error"):
        return str(status.get("error"))
    result = status.get("result")
    if isinstance(result, dict) and result.get("message"):
        return str(result.get("message"))
    return "job failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("GW_ENDPOINT", "http://localhost:8098"),
        help="Gateway HTTP endpoint (default: http://localhost:8098)",
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=["task"],
        help="Job kind (task only)",
    )
    parser.add_argument(
        "--payload-file",
        required=True,
        help="JSON file containing the exact submit payload object",
    )
    parser.add_argument(
        "--prepare-nextflow-payload",
        action="store_true",
        help="When argv is missing, derive argv/tool_cwd from command_script/main_script/workdir before submit",
    )

    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--job-id-file", default="")
    parser.add_argument("--task-id-file", default="")
    parser.add_argument("--result-file", default="")
    args = parser.parse_args()

    payload = _load_payload(args.payload_file)
    if args.prepare_nextflow_payload:
        prepared = _prepare_nextflow_payload(payload)
        prepared.pop("command_script", None)
        prepared.pop("main_script", None)
        if prepared != payload:
            _write_json(args.payload_file, prepared)
        payload = prepared
    endpoint = str(args.endpoint).rstrip("/")
    submit_resp = _request_json(
        "POST",
        f"{endpoint}/api/v1/job/submit",
        {"kind": args.kind, "payload": payload},
    )
    job_id = str(submit_resp.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"Gateway did not return job_id: {submit_resp}")

    if args.job_id_file:
        Path(args.job_id_file).write_text(job_id, encoding="utf-8")
    if args.task_id_file:
        Path(args.task_id_file).write_text(job_id, encoding="utf-8")

    def _handle_signal(signum, _frame) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while True:
        status = _request_json("GET", f"{endpoint}/api/v1/job/{job_id}")
        state_name = str(status.get("state") or "UNSPECIFIED").upper()
        if state_name in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            _write_json(args.result_file, status)
            ok = bool(status.get("ok", False))
            if ok:
                return 0
            msg = _error_message(status)
            if msg:
                print(
                    f"[job_client] terminal state={state_name} ok={ok} msg={msg}",
                    file=sys.stderr,
                )
                print(msg, file=sys.stderr)
            return 1
        time.sleep(max(0.05, float(args.poll_interval or 1.0)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
