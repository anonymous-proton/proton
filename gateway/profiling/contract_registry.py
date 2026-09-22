"""Shared component contract registry for task telemetry flows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .contract_models import ComponentProfileContract, load_contracts


class ContractRegistry:
    def __init__(self, contracts: Mapping[str, ComponentProfileContract]) -> None:
        self._contracts: Dict[str, ComponentProfileContract] = {
            str(key).strip().lower(): value
            for key, value in dict(contracts or {}).items()
            if str(key).strip()
        }

    @classmethod
    def from_contracts_dir(cls, contracts_dir: Optional[Path] = None) -> "ContractRegistry":
        path = Path(contracts_dir or (Path(__file__).resolve().parent / "contracts")).resolve()
        return cls(load_contracts(path))

    def supported_components(self) -> List[str]:
        return sorted(self._contracts.keys())

    def contract_for(self, component: str) -> ComponentProfileContract:
        key = str(component or "").strip().lower()
        if key not in self._contracts:
            raise ValueError(f"missing component profiling contract: {key}")
        return self._contracts[key]


__all__ = ["ContractRegistry"]
