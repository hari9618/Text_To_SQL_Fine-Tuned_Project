"""Verify the container images without Docker.

Phase 13.

Docker is not installed on the development machine, so `docker build` cannot
run here. That does not mean the images have to go unverified: the failure that
actually breaks most Dockerfiles is that the image is **missing something the
application needs** — a forgotten `COPY`, or a dependency that only ever got
installed by hand.

This script rules that out by doing what the build would do, in a temporary
directory:

1. replay every `COPY` directive into a staging tree,
2. create a clean, empty virtual environment,
3. install *only* `requirements.txt` into it,
4. import every application module with **only the staged tree** on the path,
5. start the entrypoint from that tree, with credentials supplied purely as
   environment variables (no `.env` — it is in `.dockerignore`),
6. hit `/health` and `/query`, and run the `HEALTHCHECK` command itself.

If all of that passes, the image contents are sufficient and the entrypoint
works from exactly those contents. What it cannot prove is Docker's own
execution of the build — the `useradd` step, layer caching, and whether
`psycopg[binary]` has a `linux/amd64` wheel as it does on Windows.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/verify_docker_image.py
    env\\Scripts\\python.exe scripts/verify_docker_image.py --keep   # keep the tree
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sql.config import PROJECT_ROOT  # noqa: E402

IMAGES = {
    "api": PROJECT_ROOT / "docker" / "Dockerfile",
    "tools": PROJECT_ROOT / "docker" / "Dockerfile.tools",
    # The Hugging Face Space image: same contents, different port handling.
    "space": PROJECT_ROOT / "deploy" / "Dockerfile",
}
API_MODULES = [
    "src.api.main", "src.api.service", "src.api.backends", "src.api.schemas",
    "src.sql.repair", "src.sql.validator", "src.sql.executor",
    "src.model.prompt", "src.model.repair_prompt", "src.evaluation.metrics",
]
TOOLS_MODULES = ["src.sql.config", "src.sql.executor", "src.sql.validator",
                 "dataset.generation.templates"]

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)
    return ok


def copy_directives(dockerfile: Path) -> list[str]:
    out = []
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY "):
            out.append(line.split()[1].rstrip("/"))
    return out


def stage_image(dockerfile: Path, dest: Path) -> tuple[int, int]:
    """Replay COPY into `dest`, returning (file count, bytes)."""
    dest.mkdir(parents=True, exist_ok=True)
    for src in copy_directives(dockerfile):
        source = PROJECT_ROOT / src
        if source.is_dir():
            shutil.copytree(source, dest / src,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, dest / source.name)
    files = [f for f in dest.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify container images without Docker")
    parser.add_argument("--keep", action="store_true",
                        help="keep the staging tree for inspection")
    args = parser.parse_args()

    print("=" * 72)
    print("DOCKER IMAGE VERIFICATION (no Docker required)")
    print("=" * 72)

    if shutil.which("docker"):
        print("note: docker IS on PATH here - prefer `docker build` over this script")

    # ---- 1. static ---------------------------------------------------------
    print("\n1. Build context")
    for name, dockerfile in IMAGES.items():
        check(f"{name}: Dockerfile exists", dockerfile.exists())
        if not dockerfile.exists():
            continue
        text = dockerfile.read_text(encoding="utf-8")
        base = re.search(r"^FROM\s+(\S+)", text, re.M)
        check(f"{name}: base image pinned", bool(base and ":" in base.group(1)),
              base.group(1) if base else "")
        for src in copy_directives(dockerfile):
            path = (PROJECT_ROOT / src).resolve()
            inside = PROJECT_ROOT.resolve() in path.parents
            check(f"{name}: COPY {src}", path.exists() and inside)

    dockerignore = PROJECT_ROOT / ".dockerignore"
    if check("`.dockerignore` exists", dockerignore.exists()):
        patterns = dockerignore.read_text(encoding="utf-8")
        for secret in (".env", "models/"):
            check(f"`.dockerignore` excludes {secret}", secret in patterns)

    work = Path(tempfile.mkdtemp(prefix="imageverify-"))
    try:
        # ---- 2. stage ------------------------------------------------------
        print("\n2. Staged image payloads")
        sizes = {}
        for name, dockerfile in IMAGES.items():
            n, size = stage_image(dockerfile, work / name)
            sizes[name] = (n, size)
            print(f"      {name:<6} {n:>4} files, {size / 1024:>9,.0f} KB")
        check("api image is small (source only, no data or weights)",
              sizes["api"][1] < 5 * 1024 * 1024,
              f"{sizes['api'][1] / 1024:,.0f} KB")
        check("no .env reached the staged tree",
              not (work / "api" / ".env").exists())
        check("no model weights reached the staged tree",
              not list((work / "api").rglob("*.safetensors")))

        # ---- 3. clean environment ------------------------------------------
        print("\n3. Clean virtual environment from requirements.txt alone")
        venv = work / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True,
                       capture_output=True)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        proc = subprocess.run(
            [str(python), "-m", "pip", "install", "-q", "-r",
             str(PROJECT_ROOT / "requirements.txt")],
            capture_output=True, text=True)
        check("every pin installs into an empty venv", proc.returncode == 0,
              proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
        if proc.returncode:
            return 1
        subprocess.run([str(python), "-m", "pip", "uninstall", "-y", "-q", "pytest"],
                       capture_output=True)

        # ---- 4. imports ----------------------------------------------------
        print("\n4. Imports resolve from the image tree alone")
        for image, modules in (("api", API_MODULES), ("tools", TOOLS_MODULES),
                               ("space", API_MODULES)):
            tree = work / image
            code = "import " + ", ".join(modules)
            proc = subprocess.run(
                [str(python), "-c", code], cwd=tree, capture_output=True,
                text=True, env={**os.environ, "PYTHONPATH": str(tree)})
            check(f"{image}: {len(modules)} modules import",
                  proc.returncode == 0,
                  proc.stderr.strip().splitlines()[-1] if proc.returncode else "")

        # ---- 5. the entrypoint actually serves -----------------------------
        print("\n5. Entrypoint serves requests from the image tree")
        env_path = PROJECT_ROOT / ".env"
        if not env_path.exists():
            check("`.env` present so credentials can be passed as the compose "
                  "file would", False)
            return 1
        creds = {}
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.partition("=")
                creds[key.strip()] = value.strip()

        tree = work / "api"
        port = free_port()
        env = {
            **os.environ,
            "PYTHONPATH": str(tree),
            "MODEL_BACKEND": "stub",
            **{k: creds.get(k, "") for k in
               ("PGHOST", "PGPORT", "PGDATABASE", "APP_DB_USER", "APP_DB_PASSWORD")},
        }
        server = subprocess.Popen(
            [str(python), "-m", "uvicorn", "src.api.main:app",
             "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
            cwd=tree, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        try:
            health = None
            for _ in range(60):
                if server.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=3) as r:
                        health = json.load(r)
                    break
                except (urllib.error.URLError, ConnectionError, OSError):
                    time.sleep(1)

            if not check("uvicorn started from the staged tree", health is not None,
                         "" if health else (server.stdout.read()[-400:]
                                            if server.stdout else "")):
                return 1
            check("/health reports status ok", health.get("status") == "ok")
            check("/health reaches the database", health.get("database") is True)
            check("schema fingerprint intact",
                  health.get("schema_fingerprint") == "d03619e711661bc5")

            body = json.dumps({"question": "verification", "max_rows": 1}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/query", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                answer = json.load(r)
            check("/query executes and returns rows",
                  answer.get("ok") is True and bool(answer.get("rows")))

            # the HEALTHCHECK command itself, run by the image's interpreter
            hc = ("import urllib.request,sys,json; "
                  f"r=json.load(urllib.request.urlopen('http://127.0.0.1:{port}"
                  "/health',timeout=8)); sys.exit(0 if r.get('status')=='ok' else 1)")
            check("HEALTHCHECK exits 0 while healthy",
                  subprocess.run([str(python), "-c", hc],
                                 capture_output=True).returncode == 0)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()

        dead = ("import urllib.request,sys,json; "
                "r=json.load(urllib.request.urlopen('http://127.0.0.1:9/health',"
                "timeout=2)); sys.exit(0 if r.get('status')=='ok' else 1)")
        check("HEALTHCHECK exits non-zero when nothing is listening",
              subprocess.run([str(python), "-c", dead],
                             capture_output=True).returncode != 0)
    finally:
        if args.keep:
            print(f"\nstaging tree kept at {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("image contents and entrypoint verified")
    print("still unproven without Docker: the build itself (useradd, layer")
    print("caching, linux/amd64 wheels). Run `docker build` where Docker exists.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
