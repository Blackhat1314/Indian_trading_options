#!/usr/bin/env python3
"""Start the whole TradeDesk stack on one machine (Windows, macOS, Linux).

    python run.py                 # simulator (no Dhan account needed) at real speed
    python run.py --speed 20      # simulator, 20x faster (a full session in ~20 min)
    python run.py --live          # Dhan live feed (needs DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in .env)
    python run.py --build-ui      # rebuild the React app first (needs Node 20+)

Starts the ingestor (the only live Dhan consumer), analytics, geo (world events), gateway and the backfill
worker behind the admin page (/admin → HISTORY), and prefixes their logs. The backfill worker is optional: if it stops it is
restarted and never takes the live services down. Redis must be reachable at REDIS_URL (default localhost:6379):
    docker run -d --name td-redis -p 6379:6379 redis:7-alpine
Open http://localhost:8000 when the gateway says it is up. Ctrl+C stops everything.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
COLORS = {"ingestor": "\033[33m", "analytics": "\033[36m", "geo": "\033[32m", "gateway": "\033[35m",
          "backfill": "\033[34m"}
RESET = "\033[0m"
OPTIONAL = {"backfill"}          # restarted when it exits; never stops the live stack
RESTART_AFTER = 15               # seconds


def pump(name: str, proc: subprocess.Popen, color: bool) -> None:
    tag = f"{COLORS[name]}{name:<9}{RESET}" if color else f"{name:<9}"
    for line in iter(proc.stdout.readline, ""):
        sys.stdout.write(f"{tag} | {line}")
        sys.stdout.flush()


def ui_stale(dist: Path) -> bool:
    """True when the frontend sources are newer than the last build (e.g. after a git pull)."""
    index = dist / "index.html"
    if not index.exists():
        return True
    src = ROOT / "frontend"
    files = [f for f in (src / "src").rglob("*") if f.is_file()] + [src / "index.html", src / "package.json"]
    newest = max((f.stat().st_mtime for f in files if f.exists()), default=0.0)
    return newest > index.stat().st_mtime


def redis_ok(url: str) -> bool:
    try:
        import redis
        redis.Redis.from_url(url, socket_connect_timeout=2).ping()
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="use the Dhan live feed (DATA_SOURCE=dhan)")
    mode.add_argument("--sim", action="store_true", help="use the built-in simulator (default unless .env says dhan)")
    ap.add_argument("--speed", type=float, default=None, help="simulator speed multiplier")
    ap.add_argument("--port", type=int, default=None, help="gateway port (default GATEWAY_PORT or 8000)")
    ap.add_argument("--build-ui", action="store_true", help="npm install + build the frontend first")
    a = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    if a.live:
        env["DATA_SOURCE"] = "dhan"
    elif a.sim:
        env["DATA_SOURCE"] = "sim"
    if a.speed:
        env["SIM_SPEED"] = str(a.speed)

    sys.path.insert(0, str(BACKEND))
    os.environ.update({k: v for k, v in env.items() if k in ("DATA_SOURCE", "SIM_SPEED")})
    from td.config import settings  # noqa: E402  (reads .env)

    if settings.data_source == "dhan" and not (settings.dhan_client_id and settings.dhan_access_token):
        print("DATA_SOURCE=dhan needs DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env (see .env.example).")
        return 1
    if not redis_ok(settings.redis_url):
        print(f"Redis is not reachable at {settings.redis_url}.\n"
              "Start one with:  docker run -d --name td-redis -p 6379:6379 redis:7-alpine\n"
              "(or install Redis / Memurai on Windows) and run this again.")
        return 1

    dist = ROOT / "frontend" / "dist"
    npm = shutil.which("npm")
    if a.build_ui or not dist.exists():
        if not npm:
            print("frontend/dist is missing and npm was not found — install Node 20+ or use docker compose.")
            return 1
        print("building the frontend…")
        subprocess.run([npm, "ci" if (ROOT / "frontend" / "package-lock.json").exists() else "install"],
                       cwd=ROOT / "frontend", check=True)
        subprocess.run([npm, "run", "build"], cwd=ROOT / "frontend", check=True)
    elif ui_stale(dist):
        if not npm:
            print("frontend sources changed since the last build but npm was not found — serving the previous build.")
        else:
            print("frontend sources changed — rebuilding the UI…")
            try:
                if not (ROOT / "frontend" / "node_modules").exists():
                    subprocess.run([npm, "ci"], cwd=ROOT / "frontend", check=True)
                subprocess.run([npm, "run", "build"], cwd=ROOT / "frontend", check=True)
            except (subprocess.CalledProcessError, OSError) as e:
                print(f"UI rebuild failed ({e}) — serving the previous build.")

    port = a.port or settings.gateway_port
    py = sys.executable
    cmds = {
        "ingestor": [py, "-m", "td.services.ingestor"],
        "analytics": [py, "-m", "td.services.analytics"],
        "geo": [py, "-m", "td.services.geo"],
        "gateway": [py, "-m", "uvicorn", "td.services.gateway:app", "--host", settings.gateway_host,
                    "--port", str(port), "--log-level", "warning"],
        "backfill": [py, "-m", "td.services.backfiller"],
    }
    color = sys.stdout.isatty() and os.name != "nt" or bool(os.environ.get("WT_SESSION"))
    procs: dict[str, subprocess.Popen] = {}

    def launch(name: str) -> None:
        procs[name] = subprocess.Popen(cmds[name], cwd=BACKEND, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=pump, args=(name, procs[name], color), daemon=True).start()

    for name in cmds:
        launch(name)
        time.sleep(1.5 if name == "ingestor" else 0.5)

    src = "SIMULATOR" if settings.data_source == "sim" else f"DHAN LIVE ({settings.dhan_plan} plan)"
    print(f"\n  TradeDesk · {src} · open http://localhost:{port}"
          + ("   (access keys required)" if settings.access_keys else "")
          + f"\n  Admin page (users, history pulls, system): http://localhost:{port}/admin\n")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    restart_at: dict[str, float] = {}
    try:
        while not stop.is_set():
            for name, p in list(procs.items()):
                if p.poll() is None:
                    continue
                if name in OPTIONAL:
                    if name not in restart_at:
                        print(f"{name} exited with code {p.returncode} — restarting it in {RESTART_AFTER} s "
                              "(the live desks are not affected)")
                        restart_at[name] = time.time() + RESTART_AFTER
                    elif time.time() >= restart_at[name]:
                        del restart_at[name]
                        launch(name)
                    continue
                print(f"{name} exited with code {p.returncode} — stopping the stack")
                stop.set()
                break
            stop.wait(1.0)
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
