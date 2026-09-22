"""Runtime guard for Protenix CCD cache objects.

Protenix caches CCD ``AtomArray`` objects with ``functools.lru_cache``.  The
standalone code later mutates per-chain fields such as ``res_id`` while building
an input-specific polymer chain, which is safe enough for one-shot execution but
unsafe for resident concurrent worker requests.  Keep the upstream cache in
place, but return a request-local copy from the adapter process.
"""

from __future__ import annotations

import copy
import functools
import logging
from typing import Any

_LOG = logging.getLogger(__name__)

_PATCH_SENTINEL = "_proton_returns_ccd_atom_array_copy"
_ORIGINAL_GETTER_ATTR = "_proton_original_get_component_atom_array"


def _copy_dynamic_attrs(source: Any, target: Any) -> Any:
    """Preserve Protenix attrs that biotite AtomArray.copy() drops."""

    try:
        attrs = vars(source)
    except TypeError:
        return target

    for name, value in attrs.items():
        try:
            copied_value = copy.deepcopy(value)
        except Exception:
            copied_value = value
        setattr(target, name, copied_value)
    return target


def install_ccd_atom_array_copy_guard(ccd_module: Any | None = None) -> bool:
    """Patch Protenix's CCD getter to return copies of cached AtomArrays.

    Args:
        ccd_module: Optional module-like object used by tests.  When omitted,
            ``protenix.data.ccd`` is imported and patched.

    Returns:
        ``True`` when this call installed the guard, ``False`` when the guard
        had already been installed.
    """

    if ccd_module is None:
        from protenix.data import ccd as ccd_module

    getter = getattr(ccd_module, "get_component_atom_array", None)
    if getter is None:
        raise AttributeError("ccd module has no get_component_atom_array()")

    if getattr(getter, _PATCH_SENTINEL, False):
        return False

    @functools.wraps(getter)
    def _copying_get_component_atom_array(*args: Any, **kwargs: Any) -> Any:
        atom_array = getter(*args, **kwargs)
        if atom_array is None:
            return None
        copy_fn = getattr(atom_array, "copy", None)
        if copy_fn is None:
            raise TypeError(
                "Protenix get_component_atom_array() returned an object without copy()"
            )
        return _copy_dynamic_attrs(atom_array, copy_fn())

    for attr in ("cache_info", "cache_clear", "cache_parameters"):
        if hasattr(getter, attr):
            setattr(_copying_get_component_atom_array, attr, getattr(getter, attr))

    setattr(_copying_get_component_atom_array, _PATCH_SENTINEL, True)
    setattr(_copying_get_component_atom_array, _ORIGINAL_GETTER_ATTR, getter)
    setattr(ccd_module, "get_component_atom_array", _copying_get_component_atom_array)
    _LOG.info("Installed Protenix CCD AtomArray copy guard for resident worker")
    return True
