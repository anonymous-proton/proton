"""Typed contract models for gateway-integrated profiling run contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


@dataclass(frozen=True)
class OverrideSchema:
    """Allowed override keys for a component contract."""

    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)

    def allowed_keys(self) -> set[str]:
        return {str(k).strip() for k in [*self.required, *self.optional] if str(k).strip()}


@dataclass(frozen=True)
class CommonAxisMap:
    """Mapping from shared DSL axes into component-specific fields."""

    input_batch_size: Optional[str] = None
    output_sample_count: Optional[str] = None


@dataclass(frozen=True)
class ComponentProfileContract:
    """Single component execution contract consumed by task/diagnostic orchestration."""

    component: str
    dispatch_contract_ref: str
    main_script_signature: str
    command_argv: List[str]
    common_axis_map: CommonAxisMap = field(default_factory=CommonAxisMap)
    override_schema: OverrideSchema = field(default_factory=OverrideSchema)
    cell_schema_version: int = 1
    cell_axis_subset: List[str] = field(default_factory=list)
    modeling_feature_schema: List[str] = field(default_factory=list)
    modeling_workload_feature_keys: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)

    def validate_override_keys(self, keys: Iterable[str]) -> None:
        allowed = self.override_schema.allowed_keys()
        unknown = sorted(
            key for key in {str(k).strip() for k in keys if str(k).strip()}
            if key not in allowed
        )
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(
                f"Unknown override keys for component '{self.component}': {joined}"
            )

    def render_argv(self, values: Mapping[str, Any]) -> List[str]:
        out: List[str] = []
        for token in self.command_argv:
            if "{" in token and "}" in token:
                try:
                    out.append(str(token).format(**values))
                except KeyError as exc:
                    missing = exc.args[0] if exc.args else "unknown"
                    raise ValueError(
                        f"Missing render value '{missing}' for component '{self.component}'"
                    ) from exc
            else:
                out.append(str(token))
        return out


def _load_single_contract(path: Path) -> ComponentProfileContract:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Contract file must be a mapping: {path}")

    component = str(payload.get("component") or "").strip().lower()
    if not component:
        raise ValueError(f"Contract file missing component: {path}")

    dispatch_ref = str(payload.get("dispatch_contract_ref") or "").strip()
    if not dispatch_ref:
        raise ValueError(f"Contract '{component}' missing dispatch_contract_ref")

    main_sig = str(payload.get("main_script_signature") or "").strip()
    if not main_sig:
        raise ValueError(f"Contract '{component}' missing main_script_signature")

    command = payload.get("command") or {}
    if not isinstance(command, Mapping):
        raise ValueError(f"Contract '{component}' command must be a mapping")
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv:
        raise ValueError(f"Contract '{component}' command.argv must be a non-empty list")
    rendered_argv = [str(item) for item in argv]

    axis_raw = payload.get("common_axis_map") or {}
    if axis_raw is None:
        axis_raw = {}
    if not isinstance(axis_raw, Mapping):
        raise ValueError(f"Contract '{component}' common_axis_map must be a mapping")
    axis_map = CommonAxisMap(
        input_batch_size=(
            str(axis_raw.get("input_batch_size")).strip()
            if axis_raw.get("input_batch_size") is not None
            else None
        ),
        output_sample_count=(
            str(axis_raw.get("output_sample_count")).strip()
            if axis_raw.get("output_sample_count") is not None
            else None
        ),
    )

    override_raw = payload.get("override_schema") or {}
    if override_raw is None:
        override_raw = {}
    if not isinstance(override_raw, Mapping):
        raise ValueError(f"Contract '{component}' override_schema must be a mapping")
    override_schema = OverrideSchema(
        required=[str(k) for k in list(override_raw.get("required") or [])],
        optional=[str(k) for k in list(override_raw.get("optional") or [])],
    )

    cell_subset_raw = payload.get("cell_axis_subset") or []
    if not isinstance(cell_subset_raw, list):
        raise ValueError(f"Contract '{component}' cell_axis_subset must be a list")
    cell_axis_subset = [str(item).strip() for item in cell_subset_raw if str(item).strip()]
    cell_schema_version_raw = payload.get("cell_schema_version", 1)
    try:
        cell_schema_version = int(cell_schema_version_raw)
    except Exception as exc:
        raise ValueError(f"Contract '{component}' cell_schema_version must be an integer >= 1") from exc
    if cell_schema_version < 1:
        raise ValueError(f"Contract '{component}' cell_schema_version must be >= 1")

    modeling_raw = payload.get("modeling") or {}
    if modeling_raw is None:
        modeling_raw = {}
    if not isinstance(modeling_raw, Mapping):
        raise ValueError(f"Contract '{component}' modeling must be a mapping")
    feature_schema_raw = modeling_raw.get("feature_schema") or []
    if not isinstance(feature_schema_raw, list):
        raise ValueError(f"Contract '{component}' modeling.feature_schema must be a list")
    workload_keys_raw = modeling_raw.get("workload_feature_keys") or []
    if not isinstance(workload_keys_raw, list):
        raise ValueError(f"Contract '{component}' modeling.workload_feature_keys must be a list")
    modeling_feature_schema = [
        str(item).strip()
        for item in feature_schema_raw
        if str(item).strip()
    ]
    modeling_workload_feature_keys = [
        str(item).strip()
        for item in workload_keys_raw
        if str(item).strip()
    ]

    defaults_raw = payload.get("defaults") or {}
    if defaults_raw is None:
        defaults_raw = {}
    if not isinstance(defaults_raw, Mapping):
        raise ValueError(f"Contract '{component}' defaults must be a mapping")
    defaults = {str(k): v for k, v in defaults_raw.items()}

    env_raw = payload.get("env") or {}
    if env_raw is None:
        env_raw = {}
    if not isinstance(env_raw, Mapping):
        raise ValueError(f"Contract '{component}' env must be a mapping")
    env = {str(k): str(v) for k, v in env_raw.items()}

    return ComponentProfileContract(
        component=component,
        dispatch_contract_ref=dispatch_ref,
        main_script_signature=main_sig,
        command_argv=rendered_argv,
        common_axis_map=axis_map,
        override_schema=override_schema,
        cell_schema_version=cell_schema_version,
        cell_axis_subset=cell_axis_subset,
        modeling_feature_schema=modeling_feature_schema,
        modeling_workload_feature_keys=modeling_workload_feature_keys,
        defaults=defaults,
        env=env,
    )


def load_contracts(contracts_dir: Path) -> Dict[str, ComponentProfileContract]:
    contracts: Dict[str, ComponentProfileContract] = {}
    if not contracts_dir.exists():
        raise FileNotFoundError(f"Contract directory missing: {contracts_dir}")

    for path in sorted(contracts_dir.glob("*.yaml")):
        if not path.is_file():
            continue
        contract = _load_single_contract(path)
        if contract.component in contracts:
            raise ValueError(f"Duplicate contract for component '{contract.component}'")
        contracts[contract.component] = contract
    return contracts


__all__ = [
    "CommonAxisMap",
    "ComponentProfileContract",
    "OverrideSchema",
    "load_contracts",
]
