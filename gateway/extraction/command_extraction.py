"""Command extraction helpers for Nextflow .command.sh."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]


class ExtractionError(ValueError):
    pass


def _host_to_container_path(path: str) -> str:
    """Convert host paths to container paths.
    
    The worker container mounts the repo at /workspace, so:
    /home/user/workspace/proton/... -> /workspace/...
    
    Also handles key=value patterns like inference.input_pdb=/home/.../file.pdb
    """
    repo_root_str = str(_REPO_ROOT)
    
    if "=" in path and not path.startswith("="):
        key, value = path.split("=", 1)
        if value.startswith(repo_root_str):
            return f"{key}=/workspace{value[len(repo_root_str):]}"
    
    if path.startswith(repo_root_str):
        return "/workspace" + path[len(repo_root_str):]
    
    return path


def _strip_line_continuations(script: str) -> str:
    return script.replace("\\\n", " ")


def _expand_shell_vars(cmd: str, workdir: str) -> str:
    """Expand common shell variables to their values.
    
    Nextflow scripts often use $CURRENT_DIR or $(pwd) to reference
    the work directory. Since we execute without a shell, we must
    expand these to the actual workdir value.
    """
    cmd = cmd.replace("$(pwd)", workdir)
    cmd = cmd.replace("`pwd`", workdir)
    
    cmd = cmd.replace("$CURRENT_DIR", workdir)
    cmd = cmd.replace("${CURRENT_DIR}", workdir)
    
    cmd = cmd.replace("$PWD", workdir)
    cmd = cmd.replace("${PWD}", workdir)
    
    return cmd


def _strip_wrapping_parens(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("("):
        stripped = stripped[1:].lstrip()
    if stripped.endswith(")"):
        stripped = stripped[:-1].rstrip()
    return stripped


def _reject_unsupported_ops(cmd: str) -> None:
    unsupported = ["|", ";", ">", "<"]
    if any(tok in cmd for tok in unsupported):
        raise ExtractionError("unsupported shell operator in command")
    if "&&" in cmd:
        raise ExtractionError("unsupported command chaining in invocation")


def _extract_cd_and_cmd(line: str) -> Tuple[str, str]:
    line = _strip_wrapping_parens(line)
    if line.lstrip().startswith("cd ") and "&&" in line:
        cd_part, cmd_part = line.split("&&", 1)
        tokens = shlex.split(cd_part.strip(), posix=True)
        if len(tokens) < 2 or tokens[0] != "cd":
            raise ExtractionError("invalid cd prefix in invocation")
        tool_cwd = tokens[1]
        cmd = cmd_part.strip()
        return tool_cwd, cmd
    return "", line.strip()


def _find_signature_lines(script: str, signature: str) -> List[str]:
    lines = []
    for raw in script.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if signature in line:
            lines.append(raw)
    return lines


def extract_main_invocation(
    command_script: str, main_script: str, workdir: str
) -> Tuple[str, List[str]]:
    if not command_script:
        raise ExtractionError("command_script is empty")
    if not main_script:
        raise ExtractionError("main_script is required")

    normalized = _strip_line_continuations(command_script)
    matches = _find_signature_lines(normalized, main_script)
    if not matches:
        raise ExtractionError("signature not found")
    if len(matches) > 1:
        raise ExtractionError("ambiguous signature")

    line = matches[0]
    tool_cwd, cmd = _extract_cd_and_cmd(line)
    if not cmd:
        raise ExtractionError("empty invocation")

    _reject_unsupported_ops(cmd)

    cmd = _expand_shell_vars(cmd, workdir)

    argv = shlex.split(cmd, posix=True)
    if not argv:
        raise ExtractionError("empty argv")
    if main_script not in argv:
        raise ExtractionError("signature not present in argv")

    container_workdir = _host_to_container_path(workdir)
    
    resolved_argv = []
    for arg in argv:
        container_arg = _host_to_container_path(arg)
        if container_arg != arg:
            resolved_argv.append(container_arg)
            continue
            
        if not os.path.isabs(arg):
            candidate = os.path.join(workdir, arg)
            if os.path.exists(candidate) and os.path.islink(candidate):
                try:
                    target = os.readlink(candidate)
                    if not os.path.isabs(target):
                        target = os.path.normpath(os.path.join(workdir, target))
                        
                    target_container = _host_to_container_path(target)
                    if target_container != target:
                        resolved_argv.append(target_container)
                        continue
                except OSError:
                    pass
        
        resolved_argv.append(arg)
        
    argv = resolved_argv

    if tool_cwd:
        return tool_cwd, argv
    return container_workdir, argv
