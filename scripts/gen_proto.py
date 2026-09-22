"""Generate gRPC stubs for the modelworker proto and patch imports."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "modelworker" / "proto"
PROTO_FILE = PROTO_DIR / "modelworker.proto"
OUT_DIR = ROOT / "modelworker"


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def patch_imports(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"^import modelworker_pb2 as modelworker__pb2$",
        "from . import modelworker_pb2 as modelworker__pb2",
        text,
        flags=re.M,
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if not PROTO_FILE.exists():
        raise SystemExit(f"Proto not found: {PROTO_FILE}")

    run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={OUT_DIR}",
            f"--grpc_python_out={OUT_DIR}",
            str(PROTO_FILE),
        ]
    )

    patch_imports(OUT_DIR / "modelworker_pb2_grpc.py")
    print("Proto generation complete.", flush=True)


if __name__ == "__main__":
    main()
