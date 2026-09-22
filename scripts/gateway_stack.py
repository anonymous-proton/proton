from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_GATEWAY = REPO_ROOT / "scripts" / "run_gateway.sh"
OPS_COMPOSE = REPO_ROOT / "ops" / "docker-compose.yml"


def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def cmd_status() -> int:
    return _run(["docker", "compose", "-f", str(OPS_COMPOSE), "ps"])


def cmd_down() -> int:
    return _run(["docker", "compose", "-f", str(OPS_COMPOSE), "down", "--remove-orphans"])


def _print_main_help() -> None:
    print("usage: python -m scripts.gateway_stack [-h] {up,status,down} ...")
    print()
    print("positional arguments:")
    print("  {up,status,down}")
    print("    up              Start gateway + observability stack")
    print("    status          Show compose status")
    print("    down            Stop compose stack")


def _print_up_help() -> None:
    print("usage: python -m scripts.gateway_stack up [gateway args ...]")
    print()
    print("Pass-through wrapper for scripts/run_gateway.sh.")
    print("Example:")
    print("  python -m scripts.gateway_stack up --host 127.0.0.7 --http-port 8098 --all")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_main_help()
        return 0

    command = str(args[0]).strip().lower()
    rest = args[1:]

    if command == "up":
        if any(item in {"-h", "--help"} for item in rest):
            _print_up_help()
            return 0
        return _run([str(RUN_GATEWAY), *rest])
    if command == "status":
        return cmd_status()
    if command == "down":
        return cmd_down()

    print(f"python -m scripts.gateway_stack: error: unknown command: {command}", file=sys.stderr)
    _print_main_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
