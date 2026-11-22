#!/usr/bin/env python3
"""One-command launcher: works on Windows 11 / macOS / Linux with Python 3.10+ (tested target: 3.13).

    python run.py              # create .venv, install dependencies, start the app, open the browser
    python run.py --test       # run the whole automated test-suite inside the same .venv
    python run.py --demo       # run the demo in the terminal (no browser)
    python run.py --no-browser # do not open a browser tab
Nothing else is required. A Gemini key is OPTIONAL: without it the app runs in DEMO MODE.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import venv

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
REQ = os.path.join(ROOT, "backend", "requirements.txt")
BIN = "Scripts" if os.name == "nt" else "bin"
VPY = os.path.join(VENV, BIN, "python.exe" if os.name == "nt" else "python")
MARK = os.path.join(VENV, ".p2p_requirements_hash")


def say(msg: str) -> None:
    print(f"[Paper2Product] {msg}", flush=True)


def ensure_venv() -> None:
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required. Install Python 3.13 from https://www.python.org/downloads/ "
                 "(tick 'Add python.exe to PATH') and run this again.")
    if not os.path.exists(VPY):
        say("Creating a private Python environment in .venv (one time)...")
        venv.create(VENV, with_pip=True, clear=False)


def ensure_requirements() -> None:
    with open(REQ, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    try:
        with open(MARK, encoding="utf-8") as f:
            if f.read().strip() == digest:
                return
    except OSError:
        pass
    say("Installing dependencies (one time, can take 1-3 minutes)...")
    cmd = [VPY, "-m", "pip", "install", "--disable-pip-version-check", "-r", REQ]
    if subprocess.call(cmd) != 0:
        sys.exit("\nInstalling dependencies failed. Check your internet connection and run this again.\n"
                 "If it keeps failing, send the red error text above to whoever is helping you.")
    with open(MARK, "w", encoding="utf-8") as f:
        f.write(digest)


def ensure_env_file() -> None:
    env, ex = os.path.join(ROOT, ".env"), os.path.join(ROOT, ".env.example")
    if not os.path.exists(env) and os.path.exists(ex):
        shutil.copyfile(ex, env)
        say("Created .env (optional settings). You may paste a GEMINI_API_KEY into it to enable LIVE mode.")


def free_port(start: int) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    sys.exit("No free network port found between %d and %d." % (start, start + 49))


def open_when_ready(url: str) -> None:
    import webbrowser
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        return
    say(f"Ready -> {url}")
    webbrowser.open(url)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="run the automated tests")
    ap.add_argument("--demo", action="store_true", help="run the CLI demo")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    ensure_venv()
    ensure_requirements()
    ensure_env_file()
    env = dict(os.environ, PYTHONUTF8="1")

    if a.test:
        say("Running the automated test-suite...")
        return subprocess.call([VPY, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=os.path.join(ROOT, "backend"), env=env)
    if a.demo:
        return subprocess.call([VPY, os.path.join(ROOT, "scripts", "run_demo.py")], cwd=ROOT, env=env)

    port = free_port(a.port)
    url = f"http://127.0.0.1:{port}"
    say("Starting the server. Leave this window open. Press Ctrl+C to stop.")
    if not a.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    try:
        return subprocess.call([VPY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                               cwd=os.path.join(ROOT, "backend"), env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
