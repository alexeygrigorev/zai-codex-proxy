"""Opt-in integration tests for the zodex profile path.

These tests start the Rust proxy on a random localhost port, write an isolated
zodex-style Codex profile under a temporary CODEX_HOME, then run `codex exec`
through that profile.

They require real Z.AI credentials and can spend quota, so they only run when
RUN_ZODEX_E2E=1 is set.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CODEX_TIMEOUT_SECONDS = int(os.environ.get("ZODEX_E2E_CODEX_TIMEOUT", "180"))


def _require_enabled() -> None:
    if os.environ.get("RUN_ZODEX_E2E") != "1":
        pytest.skip("set RUN_ZODEX_E2E=1 to run real zodex integration tests")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _zai_key() -> str:
    value = os.environ.get("CODEX_PROXY_ZAI_API_KEY") or os.environ.get("ZAI_API_KEY")
    if value:
        return value
    local_env = _read_env_file(Path.home() / ".zodex" / "zai.env")
    value = local_env.get("CODEX_PROXY_ZAI_API_KEY") or local_env.get("ZAI_API_KEY")
    if value:
        return value
    pytest.skip("set CODEX_PROXY_ZAI_API_KEY or ZAI_API_KEY to run zodex e2e")


def _codex_env(base_env: dict[str, str], zodex_home: Path) -> dict[str, str]:
    env = base_env.copy()
    for key in [
        "CODEX_ACCESS_TOKEN",
        "CODEX_API_KEY",
        "CODEX_PROXY_API_KEY",
        "CODEX_PROXY_ZAI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ZAI_API_KEY",
    ]:
        env.pop(key, None)
    home = zodex_home.parent / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env["CODEX_HOME"] = str(zodex_home)
    env["ZODEX_DIR"] = str(zodex_home)
    env["CODEX_SQLITE_HOME"] = str(zodex_home)
    return env


def _codex_binary() -> str:
    configured = os.environ.get("ZODEX_E2E_CODEX_BIN")
    if configured:
        return configured
    binary = shutil.which("codex")
    if binary:
        return binary
    pytest.skip("codex CLI not found; install @openai/codex or set ZODEX_E2E_CODEX_BIN")


def _proxy_binary() -> Path:
    configured = os.environ.get("ZODEX_E2E_PROXY_BIN")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            PROJECT_ROOT / "target" / "debug" / "zai-codex-proxy",
            PROJECT_ROOT / "target" / "release" / "zai-codex-proxy",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip("proxy binary not found; run `cargo build` or set ZODEX_E2E_PROXY_BIN")


def _wait_for_proxy(port: int, timeout_seconds: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (ConnectionRefusedError, urllib.error.URLError, OSError):
            time.sleep(0.2)
    raise TimeoutError(f"proxy did not become ready on {url}")


@pytest.fixture(scope="module")
def e2e_context(tmp_path_factory: pytest.TempPathFactory):
    _require_enabled()
    proxy_port = _free_port()
    run_dir = tmp_path_factory.mktemp("zodex-e2e")
    zodex_home = run_dir / "zodex-home"
    workdir = run_dir / "workdir"
    proxy_config = run_dir / "proxy-config.json"
    proxy_log = run_dir / "proxy.log"
    zodex_home.mkdir()
    workdir.mkdir()

    zodex_config = (FIXTURES / "zodex_config.toml").read_text()
    (zodex_home / "config.toml").write_text(zodex_config.format(port=proxy_port))

    proxy_config_text = (FIXTURES / "proxy_config.json").read_text()
    proxy_config.write_text(proxy_config_text.replace("{port}", str(proxy_port)))

    proxy_env = os.environ.copy()
    proxy_env["CODEX_PROXY_ZAI_API_KEY"] = _zai_key()
    proxy_env["CODEX_PROXY_LOG_LEVEL"] = "info"

    with proxy_log.open("w") as log_file:
        proc = subprocess.Popen(
            [str(_proxy_binary()), "--config", str(proxy_config)],
            cwd=PROJECT_ROOT,
            env=proxy_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    try:
        _wait_for_proxy(proxy_port)
        yield {
            "codex_bin": _codex_binary(),
            "codex_env": _codex_env(os.environ, zodex_home),
            "port": proxy_port,
            "proxy_log": proxy_log,
            "workdir": workdir,
            "zodex_home": zodex_home,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_zodex_profile_runs_codex_exec_through_proxy(e2e_context):
    output_path = e2e_context["workdir"] / "last-message.txt"

    result = subprocess.run(
        [
            e2e_context["codex_bin"],
            "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--ignore-rules",
            "--ephemeral",
            "--cd",
            str(e2e_context["workdir"]),
            "--output-last-message",
            str(output_path),
            "Reply with exactly: zodex-ok",
        ],
        cwd=e2e_context["workdir"],
        env=e2e_context["codex_env"],
        capture_output=True,
        text=True,
        timeout=CODEX_TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, (
        f"codex exec failed\nstdout:\n{result.stdout[-2000:]}\nstderr:\n"
        f"{result.stderr[-2000:]}\nproxy log:\n"
        f"{e2e_context['proxy_log'].read_text()[-2000:]}"
    )
    assert output_path.exists()
    assert output_path.read_text().strip() == "zodex-ok"


def test_proxy_accepts_zai_web_search_tool(e2e_context):
    payload = {
        "model": "glm-5-turbo",
        "input": "Use web search if useful, then reply with exactly: zodex-web-ok",
        "tools": [
            {
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_engine": "search-prime",
                },
            }
        ],
        "stream": False,
        "max_output_tokens": 32,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{e2e_context['port']}/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=CODEX_TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode("utf-8"))

    assert response.status == 200
    assert body["status"] == "completed"
