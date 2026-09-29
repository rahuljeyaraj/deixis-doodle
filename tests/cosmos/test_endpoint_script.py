"""scripts/nebius/cosmos_endpoint.sh: syntax, settings and token redaction (never calls Nebius)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "nebius" / "cosmos_endpoint.sh"


def _bash() -> str | None:
    """A POSIX bash that runs locally. On Windows only Git Bash qualifies: the System32
    bash.exe launches WSL, which cannot see Windows paths the same way."""
    if sys.platform == "win32":
        for cand in (
            shutil.which("bash"),
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ):
            if cand and "system32" not in cand.lower() and Path(cand).is_file():
                return cand
        return None
    return shutil.which("bash")


BASH = _bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no local bash")


def _run(args, env_file: Path, **env) -> subprocess.CompletedProcess:
    e = {**os.environ, "ENV_FILE": str(env_file), "NEBIUS_BIN": "/nonexistent/nebius", **env}
    return subprocess.run(
        [BASH, str(SCRIPT), *args], capture_output=True, text=True, env=e, timeout=60
    )


def test_syntax():
    r = subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_line_endings_are_lf():
    assert b"\r\n" not in SCRIPT.read_bytes()


def test_settings_reads_token_without_printing_it(tmp_path):
    env = tmp_path / ".env"
    env.write_text('COSMOS_API_KEY="s3cr3t-token-123"\nHF_TOKEN=hf_abc\n')
    r = _run(["settings"], env)
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "s3cr3t-token-123" not in out and "hf_abc" not in out
    assert "token:       set (redacted)" in out
    assert "nvidia/Cosmos3-Nano" in out and "gpu-rtx6000-a / 1gpu-24vcpu-218gb" in out
    assert "vllm/vllm-openai:v0.29.0" in out and "500Gi" in out
    assert "--mm-encoder-tp-mode data --async-scheduling" in out


def test_fallback_token_name_and_profiles(tmp_path):
    env = tmp_path / ".env"
    env.write_text("COSMOS_ENDPOINT_TOKEN=abc\r\n")  # CRLF .env from Windows editors
    r = _run(["settings"], env, COSMOS_PROFILE="reason2")
    assert "token:       set (redacted)" in r.stdout
    assert "nvidia/Cosmos-Reason2-8B" in r.stdout and "gpu-l40s-a" in r.stdout
    r = _run(["settings"], env, COSMOS_PROFILE="edge", PLATFORM="gpu-h100-sxm")
    assert "nvidia/Cosmos3-Edge" in r.stdout and "gpu-h100-sxm" in r.stdout
    r = _run(["settings"], env, COSMOS_PROFILE="nope")
    assert r.returncode == 2


def test_create_refuses_without_cli_or_token(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    r = _run(["create"], env)
    assert r.returncode == 1 and "Nebius CLI not found" in r.stderr
    r = _run(["dry-run"], env, NEBIUS_BIN=BASH)  # an executable, so the token check runs next
    assert r.returncode == 1 and "No token" in r.stderr


def test_usage(tmp_path):
    r = _run([], tmp_path / "missing.env")
    assert r.returncode == 1 and "dry-run" in r.stdout and "watchdog" in r.stdout
