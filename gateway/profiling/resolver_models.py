"""Typed output model for gateway profiling input resolvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ResolverResult:
    """Resolved run inputs for a single component profiling run."""

    render_values: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    profile_options: Dict[str, Any] = field(default_factory=dict)
    tool_cwd: Optional[str] = None


__all__ = ["ResolverResult"]
