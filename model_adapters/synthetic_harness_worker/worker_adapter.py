"""Bridge package for the synthetic scheduler stress harness.

The worker server resolves adapters through the ``model_adapters`` naming
convention: ``foo.bar:Baz`` becomes ``model_adapters/foo_worker/bar.py``.
The harness implementation lives at repo root in ``synthetic_harness`` for
shared runtime/config generation logic, so this thin wrapper makes the worker
loader happy without duplicating the real adapter body.
"""

from synthetic_harness.worker_adapter import SyntheticAdapter

__all__ = ["SyntheticAdapter"]
