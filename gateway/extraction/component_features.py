"""Gateway-side workload feature extraction for GP VRAM prediction keying.

Two complementary mechanisms, both optional per component:

1. **Worker config** (``workers.yaml`` ``workload`` section) — simple argv extraction:

   .. code-block:: yaml

       workload:
         length_args:
           sequence_length: "--sequence"        # feature_name: argv_flag (or list)
           num_residues: ["--pdb_path"]
         config_args:
           checkpoint: "--ckpt_override_path"   # feature_name: argv_flag (or list)

   ``length_args`` values are converted to int directly, or via ``len()`` when the
   raw value is a string (e.g. an inline sequence).  ``config_args`` values are kept
   as strings.

2. **``workload_features.py``** per adapter directory — complex extraction (file reads,
   regex parsing, etc.):

   .. code-block:: python

       COMPONENT = "mymodel"
       def extract_workload_features(argv, env) -> Dict:
           return {
               "scaffold_length": 187,    # int/float → length feature
               "checkpoint": "v2.ckpt",   # str       → config feature
           }

   Loaded by the gateway via ``importlib.util.spec_from_file_location`` — stdlib
   only, no torch/numpy on the gateway host.

Feature classification
----------------------
* **Length feature** — numeric value (int / float) → contributes to ``length_key``
  (GP VRAM prediction input_size key).
* **Config feature** — string value → contributes to ``config_fingerprint``
  (distinguishes model checkpoint / variant; sorted ``key=val|…`` string).

Priority (highest wins)
-----------------------
Nextflow ``ext.wf_*`` submission > worker-config extraction > workload_features.py

Adding a new adapter
--------------------
For simple cases: add ``workload.length_args`` / ``workload.config_args`` to the
component entry in ``workers.yaml``.

For complex cases: create ``model_adapters/<X>_worker/workload_features.py`` with
``COMPONENT`` and ``extract_workload_features``.  No gateway code changes required.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


_COMPONENT_LENGTH_ARGS: Dict[str, Dict[str, Any]] = {}
_COMPONENT_CONFIG_ARGS: Dict[str, Dict[str, Any]] = {}
_COMPONENT_FAN_OUT_ARGS: Dict[str, Dict[str, Any]] = {}
_COMPONENT_ENV: Dict[str, Dict[str, str]] = {}
_COMPONENT_INPUT_SIZE_KEY: Dict[
    str, str
] = {}
_COMPONENT_DYNAMIC_BATCHING: Dict[str, Dict[str, Any]] = {}

_ADAPTER_EXTRACTORS: Dict[
    str, Callable[[List[str], Dict[str, str]], Dict[str, Any]]
] = {}
_ADAPTER_RETRY_ADJUSTERS: Dict[str, Callable[[List[str], int], List[str]]] = {}
_ADAPTER_RETRYABLE_CHECKERS: Dict[str, Callable[[str], bool]] = {}
_ADAPTER_UNLIMITED_RETRIES: Set[str] = set()

_MODEL_ADAPTERS_ROOT: Path = Path(__file__).resolve().parents[2] / "model_adapters"
_REPO_ROOT: str = str(Path(__file__).resolve().parents[2])


def _container_to_host_path(arg: str) -> str:
    """Reverse /workspace/... back to the host repo root path.

    Handles both plain paths and key=value patterns:
      /workspace/nextflow/work/... → /home/.../proton/nextflow/work/...
      inference.input_pdb=/workspace/... → inference.input_pdb=/home/.../...
    """
    if "=" in arg and not arg.startswith("="):
        key, value = arg.split("=", 1)
        if value.startswith("/workspace/"):
            return f"{key}={_REPO_ROOT}{value[len('/workspace') :]}"
        return arg
    if arg.startswith("/workspace/"):
        return f"{_REPO_ROOT}{arg[len('/workspace') :]}"
    return arg




def _parse_argv_value(argv: List[str], flags: Any) -> Optional[str]:
    """Return the raw string value for the first matching flag in argv.

    ``flags`` may be a single string (``"--sequence"``) or a list of strings.
    Supports both ``--flag value`` and ``--flag=value`` forms.  The flag string
    is matched with or without leading dashes (``--``, ``-``, bare name).
    """
    if isinstance(flags, str):
        flags = [flags]
    for i, a in enumerate(argv):
        for f in flags:
            if str(a).startswith(f + "="):
                return str(a).split("=", 1)[1]
            bare = f.lstrip("-")
            if str(a) in (f"--{bare}", f"-{bare}", bare, f) and i + 1 < len(argv):
                return str(argv[i + 1])
    return None


def _to_length_value(raw: str) -> Optional[int]:
    """Convert a raw argv string to an integer length.

    First tries direct int conversion (for ``--num_residues 320``).
    Falls back to ``len()`` for inline string values (e.g. sequences).
    Returns ``None`` on failure or zero.
    """
    try:
        n = int(raw)
        return n if n > 0 else None
    except (ValueError, TypeError):
        pass
    n = len(str(raw).strip())
    return n if n > 0 else None


def _extract_from_length_args(
    argv: List[str], args_map: Dict[str, Any]
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for feature_name, flags in args_map.items():
        raw = _parse_argv_value(argv, flags)
        if raw is not None:
            val = _to_length_value(raw)
            if val is not None:
                result[str(feature_name)] = val
    return result


def _extract_from_config_args(
    argv: List[str],
    args_map: Dict[str, Any],
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Extract config features from argv flags or env vars.

    Flag values starting with ``$`` are treated as env var references
    (e.g., ``model_name: "$ESM_MODEL_NAME"``).
    """
    result: Dict[str, str] = {}
    for feature_name, flags in args_map.items():
        flag_str = str(flags) if isinstance(flags, str) else ""
        if flag_str.startswith("$") and env:
            val = env.get(flag_str[1:], "")
            if val.strip():
                result[str(feature_name)] = val.strip()
            continue
        raw = _parse_argv_value(argv, flags)
        if raw is not None and str(raw).strip():
            result[str(feature_name)] = str(raw).strip()
    return result


def _classify(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Split a flat feature dict into (length_features, config_features) by value type."""
    length: Dict[str, Any] = {}
    config: Dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            if v != 0:
                length[str(k)] = v
        elif isinstance(v, str):
            stripped = v.strip()
            if stripped:
                config[str(k)] = stripped
    return length, config




def _register_spec_args(
    component: str,
    length_args: Dict[str, Any],
    config_args: Dict[str, Any],
    fan_out_args: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
    input_size_key: str = "",
    dynamic_batching: Optional[Dict[str, Any]] = None,
) -> None:
    comp = str(component or "").strip().lower()
    if not comp:
        return
    if length_args:
        _COMPONENT_LENGTH_ARGS[comp] = dict(length_args)
    if config_args:
        _COMPONENT_CONFIG_ARGS[comp] = dict(config_args)
    if fan_out_args:
        _COMPONENT_FAN_OUT_ARGS[comp] = dict(fan_out_args)
    if env:
        _COMPONENT_ENV[comp] = {str(k): str(v) for k, v in env.items()}
    if input_size_key:
        _COMPONENT_INPUT_SIZE_KEY[comp] = str(input_size_key)
    _COMPONENT_DYNAMIC_BATCHING[comp] = dict(dynamic_batching or {"enabled": False})


def _load_features_file(features_file: Path) -> None:
    """Load a single workload_features.py and register its extractor."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"_wf_{features_file.parent.name}",
            features_file,
        )
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        component = str(getattr(mod, "COMPONENT", "")).strip().lower()
        fn = getattr(mod, "extract_workload_features", None)
        if component and callable(fn):
            _ADAPTER_EXTRACTORS[component] = fn
        input_size_key = getattr(mod, "INPUT_SIZE_KEY", None)
        if component and input_size_key and component not in _COMPONENT_INPUT_SIZE_KEY:
            _COMPONENT_INPUT_SIZE_KEY[component] = str(input_size_key)
        retry_fn = getattr(mod, "adjust_argv_for_retry", None)
        if component and callable(retry_fn):
            _ADAPTER_RETRY_ADJUSTERS[component] = retry_fn
        retryable_fn = getattr(mod, "is_retryable_error", None)
        if component and callable(retryable_fn):
            _ADAPTER_RETRYABLE_CHECKERS[component] = retryable_fn
        if component and getattr(mod, "UNLIMITED_RETRIES", False):
            _ADAPTER_UNLIMITED_RETRIES.add(component)
    except Exception:
        pass


def auto_register_all(
    specs: Optional[List[Any]] = None,
    adapters_root: Optional[Path] = None,
) -> None:
    """Register all feature extractors at gateway startup.

    1. Reads ``length_args`` / ``config_args`` from each ``WorkerSpec`` in *specs*.
    2. Scans ``model_adapters/*/workload_features.py`` and loads extractor functions.

    Safe to call multiple times — re-registration overwrites existing entries.
    """
    for spec in specs or []:
        _register_spec_args(
            getattr(spec, "component", ""),
            getattr(spec, "length_args", {}) or {},
            getattr(spec, "config_args", {}) or {},
            getattr(spec, "fan_out_args", {}) or {},
            getattr(spec, "env", {}) or {},
            getattr(spec, "input_size_key", "") or "",
            getattr(spec, "dynamic_batching", None),
        )

    root = Path(adapters_root) if adapters_root else _MODEL_ADAPTERS_ROOT
    for features_file in sorted(root.glob("*/workload_features.py")):
        _load_features_file(features_file)




def extract_fan_out(
    component: str,
    payload: Dict[str, Any],
) -> Optional[int]:
    """Extract fan-out (output_sample_count) from argv using ``fan_out_args``.

    Returns the integer fan-out value if extractable, or ``None``.
    This is a fallback — ``ext.output_sample_count`` from Nextflow takes precedence.
    """
    comp = str(component or "").strip().lower()
    fan_args = _COMPONENT_FAN_OUT_ARGS.get(comp) or {}
    if not fan_args:
        return None
    argv: List[str] = list(payload.get("argv") or [])
    for _feature_name, flags in fan_args.items():
        raw = _parse_argv_value(argv, flags)
        if raw is not None:
            val = _to_length_value(raw)
            if val is not None and val > 0:
                return val
    return None


def extract_features(
    component: str,
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Extract workload features from a task payload at dispatch time.

    Returns a tuple of:
      - ``length_features``: ``{name: numeric}`` — feeds GP VRAM prediction ``input_size``
      - ``config_features``: ``{name: str}``    — feeds ``config_fingerprint`` when not submitted

    Sources are merged in priority order (later wins):
      1. ``workload_features.py`` extractor (complex logic)
      2. worker config ``workload.length_args`` / ``workload.config_args`` (simple argv)

    Workers config takes precedence over workload_features.py so that explicit
    config-level declarations can always override computed values.
    """
    comp = str(component or "").strip().lower()
    argv: List[str] = list(payload.get("argv") or [])
    spec_env = _COMPONENT_ENV.get(comp) or {}
    task_env = dict(payload.get("env") or {})
    env: Dict[str, str] = {**spec_env, **task_env}
    workdir = str(payload.get("workdir") or "")
    if workdir:
        env["_GW_WORKDIR"] = _container_to_host_path(workdir)

    if not argv:
        command_script = str(payload.get("command_script") or "")
        main_script = str(payload.get("main_script") or "")
        if command_script and main_script:
            try:
                from .command_extraction import extract_main_invocation

                _, argv = extract_main_invocation(
                    command_script,
                    main_script,
                    str(payload.get("workdir") or "/tmp"),
                )
            except Exception:
                pass

    host_argv = [_container_to_host_path(arg) for arg in argv]

    length_features: Dict[str, Any] = {}
    config_features: Dict[str, str] = {}

    fn = _ADAPTER_EXTRACTORS.get(comp)
    if fn is not None:
        try:
            raw = fn(host_argv, env) or {}
        except Exception:
            raw = {}
        lf, cf = _classify(raw)
        length_features.update(lf)
        config_features.update(cf)

    len_args = _COMPONENT_LENGTH_ARGS.get(comp) or {}
    if len_args:
        length_features.update(_extract_from_length_args(host_argv, len_args))

    cfg_args = _COMPONENT_CONFIG_ARGS.get(comp) or {}
    if cfg_args:
        config_features.update(_extract_from_config_args(host_argv, cfg_args, env=env))

    return length_features, config_features


def get_input_size_key(component: str) -> Optional[str]:
    """Return the INPUT_SIZE_KEY declared by this component's workload_features.py, or None."""
    return _COMPONENT_INPUT_SIZE_KEY.get(str(component or "").strip().lower())


def get_dynamic_batching(component: str) -> dict[str, Any]:
    comp = str(component or "").strip().lower()
    return dict(_COMPONENT_DYNAMIC_BATCHING.get(comp) or {"enabled": False})


def filter_dynamic_batch_config(
    component: str,
    config_features: dict[str, str],
    *,
    active: bool,
) -> dict[str, str]:
    """Drop only the active execution-batch argument from static identity."""
    features = dict(config_features)
    cfg = get_dynamic_batching(component)
    if not active or not cfg.get("enabled"):
        return features
    batch_arg = str(cfg.get("batch_size_arg") or "")
    args_map = _COMPONENT_CONFIG_ARGS.get(str(component or "").strip().lower()) or {}
    for name, descriptor in args_map.items():
        descriptors = descriptor if isinstance(descriptor, list) else [descriptor]
        if batch_arg in {str(item) for item in descriptors}:
            features.pop(str(name), None)
    return features


def logical_batch_size(
    component: str,
    payload: dict[str, Any],
    workload_features: dict[str, Any],
) -> int | None:
    """Return exact logical N, cross-checking MMseqs2 FASTA when available."""
    raw_declared = workload_features.get("input_batch_size")
    try:
        declared = int(raw_declared) if raw_declared is not None else None
    except (TypeError, ValueError):
        declared = None
    if declared is not None and declared <= 0:
        declared = None

    if str(component or "").strip().lower() != "mmseqs2":
        return declared

    fasta_path = ""
    for raw_arg in list(payload.get("argv") or []):
        arg = _container_to_host_path(str(raw_arg))
        value = arg.split("=", 1)[1] if "=" in arg else arg
        if value.rstrip("/").endswith((".fasta", ".fa")):
            fasta_path = value
            break
    actual: int | None = None
    if fasta_path:
        try:
            actual = (
                sum(
                    line.startswith(b">")
                    for line in Path(fasta_path).read_bytes().splitlines()
                )
                or None
            )
        except OSError:
            actual = None
    if declared is not None and actual is not None and declared != actual:
        return None
    return declared or actual


def get_retry_adjuster(component: str) -> Callable[[list[str], int], list[str]] | None:
    """Return the adjust_argv_for_retry function for this component, or None."""
    return _ADAPTER_RETRY_ADJUSTERS.get(str(component or "").strip().lower())


def get_retryable_checker(component: str) -> Callable[[str], bool] | None:
    """Return the is_retryable_error function for this component, or None."""
    return _ADAPTER_RETRYABLE_CHECKERS.get(str(component or "").strip().lower())


def has_unlimited_retries(component: str) -> bool:
    """Check if this component declared UNLIMITED_RETRIES in its adapter."""
    return str(component or "").strip().lower() in _ADAPTER_UNLIMITED_RETRIES




def extract_workload_features(
    component: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Deprecated: use extract_features() which also returns config features."""
    length, _ = extract_features(component, payload)
    return length
