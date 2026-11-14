"""UIL Research Console — native desktop launcher.

Wraps the FastAPI backend (webui/backend/main.py) and the animated
desktop frontend (webui/desktop/) in a native window via pywebview
(Edge WebView2 on Windows — no browser chrome, no Electron).

Run from the package PARENT directory (so `figpso_uil.*` imports resolve):

    cd C:\\Users\\arinb\\Downloads
    python figpso_uil/desktop.py

or from anywhere — the script fixes sys.path itself. Falls back to the
default browser if pywebview isn't installed (`pip install pywebview`).
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request

# Make `figpso_uil.*` importable no matter where we're launched from.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_PKG_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

APP_TITLE = "UIL Research Console"
HOST = "127.0.0.1"


def _free_port(preferred: int = 8710) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((HOST, port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port")


def _start_server(port: int) -> threading.Thread:
    import uvicorn
    from figpso_uil.webui.backend.main import app

    config = uvicorn.Config(app, host=HOST, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread


def _wait_ready(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def main() -> None:
    port = _free_port()
    url = f"http://{HOST}:{port}/app/"
    api_probe = f"http://{HOST}:{port}/api/runs"

    print(f"[desktop] starting backend on {HOST}:{port} ...")
    _start_server(port)
    if not _wait_ready(api_probe):
        print("[desktop] backend did not come up in time", file=sys.stderr)
        sys.exit(1)
    print(f"[desktop] backend ready — {url}")

    try:
        import webview  # pywebview
    except ImportError:
        import webbrowser
        print("[desktop] pywebview not installed (pip install pywebview);")
        print("[desktop] opening in the default browser instead.")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return

    window = webview.create_window(
        APP_TITLE,
        url,
        width=1440,
        height=900,
        min_size=(1080, 700),
        background_color="#0b0b10",
    )
    # webview.start() blocks until the window closes; the uvicorn thread
    # is daemonized so process exit tears it down cleanly.
    webview.start(debug=os.environ.get("UIL_DESKTOP_DEBUG") == "1")
    print("[desktop] window closed — shutting down.")


if __name__ == "__main__":
    main()
