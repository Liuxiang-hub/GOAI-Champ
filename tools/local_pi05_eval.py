#!/usr/bin/env python3
"""Create and control a local Pi05_PiperX evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


RUNNER_URL = "http://127.0.0.1:19200"
CALLBACK_URL = "http://127.0.0.1:19201"
ENV_FILE = Path("/home/user/eval-runner-slim-0.1.1/.env")
ROOT = Path("/home/user/goai_pi05")
STATE_FILE = ROOT / "local_pi05_eval_state.json"
CALLBACK_LOG = ROOT / "local-pi05-callback.log"
CALLBACK_EVENTS = ROOT / "local-pi05-callback-events.jsonl"
MOVEMENT_CONFIRMATION = "I_AM_AT_NEW_ROBOT_WITH_ESTOP"


def read_env_value(name: str) -> str:
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            value = value.strip().strip('"').strip("'")
            if value:
                return value
    raise RuntimeError(f"{name} is missing from {ENV_FILE}")


def request(method: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        RUNNER_URL + path,
        data=body,
        method=method,
        headers={
            "Authorization": "Bearer " + read_env_value("STATION_CONTROL_TOKEN"),
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"eval-runner HTTP {exc.code}: {detail}") from exc
    return json.loads(data) if data else {}


def callback_is_running() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 19201), timeout=0.2):
            return True
    except OSError:
        return False


def ensure_callback() -> None:
    if callback_is_running():
        return
    log = CALLBACK_LOG.open("ab")
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "callback"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log.close()
    for _ in range(30):
        if callback_is_running():
            return
        time.sleep(0.1)
    raise RuntimeError("local finish callback did not start")


def save_state(state: dict) -> None:
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(temp, 0o600)
    os.replace(temp, STATE_FILE)


def load_state() -> dict:
    if not STATE_FILE.exists():
        raise RuntimeError("no local Pi05 task has been dispatched")
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def trial_path(state: dict, suffix: str) -> str:
    return f"/v1/sessions/{state['evaluation_id']}/trials/1/{suffix}"


def dispatch(_: argparse.Namespace) -> None:
    ensure_callback()
    evaluation_id = "local-pi05-" + uuid.uuid4().hex
    trial_id = "stack_bowls-pi05-local-r01"
    payload = {
        "model_name": "Pi05_PiperX",
        "action_type": "joint",
        "policy_server_url": "ws://127.0.0.1:6007",
        "evaluation_plan": {
            "task": {
                "id": "local-stack_bowls",
                "name": "stack_bowls",
                "env_cfg_type": "piper",
            },
            "trials": [{
                "action_case_id": "stack_bowls_local_pi05_case_1",
                "trial_index": 1,
                "trial_id": trial_id,
                "finish_url": CALLBACK_URL + "/finish/" + evaluation_id + "/1",
                "instruction": "Stack the bowls on the table.",
            }],
        },
        "artifact": {"bucket": "", "prefix": ""},
        "callback": {"hmac_secret_ref": "EVAL_SERVER_WEBHOOK_SECRET"},
    }
    result = request("POST", f"/v1/sessions/{evaluation_id}/dispatch", payload)
    state = {
        "evaluation_id": evaluation_id,
        "trial_id": trial_id,
        "model_name": "Pi05_PiperX",
        "policy_server_url": "ws://127.0.0.1:6007",
        "dispatched_at": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)
    print(json.dumps({"state": state, "dispatch_result": result}, ensure_ascii=False))
    print("DISPATCHED_ONLY: robot motion has not been started")


def status(_: argparse.Namespace) -> None:
    state = load_state()
    print(json.dumps(request("GET", trial_path(state, "status")), ensure_ascii=False))


def start(args: argparse.Namespace) -> None:
    if args.confirm_movement != MOVEMENT_CONFIRMATION:
        raise RuntimeError("refusing motion without explicit local E-stop confirmation")
    state = load_state()
    print(json.dumps(request("POST", trial_path(state, "start")), ensure_ascii=False))


def stop(_: argparse.Namespace) -> None:
    state = load_state()
    print(json.dumps(request("POST", trial_path(state, "stop")), ensure_ascii=False))


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": self.path,
            "body": raw.decode("utf-8", errors="replace"),
        }
        with CALLBACK_EVENTS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, fmt: str, *args: object) -> None:
        print(fmt % args, flush=True)


def callback(_: argparse.Namespace) -> None:
    ThreadingHTTPServer(("127.0.0.1", 19201), CallbackHandler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("dispatch").set_defaults(func=dispatch)
    commands.add_parser("status").set_defaults(func=status)
    start_parser = commands.add_parser("start")
    start_parser.add_argument("--confirm-movement", default="")
    start_parser.set_defaults(func=start)
    commands.add_parser("stop").set_defaults(func=stop)
    commands.add_parser("callback").set_defaults(func=callback)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
