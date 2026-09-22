"""Model adapter interface for the worker skeleton."""

from __future__ import annotations

import threading
from typing import Any


class ModelAdapter:
    """Adapter interface implemented by model-specific workers.

    The worker engine treats this as a three-stage pipeline:
      prepare -> execute -> finalize.

    - prepare: normalize/validate requests, build light-weight inputs.
    - execute: run the model (usually the heavy, stateful step).
    - finalize: post-process outputs into response payloads.

    The engine will:
      - call init_* once at startup (per stage),
      - use prepare_batch/finalize_batch if provided,
      - otherwise fall back to per-item prepare_one/finalize_one.

    ## Concurrency Contract

    All execute work uses persistent ``spawn`` actors. ``execute_processes=0``
    or the legacy ``-1`` spelling retains one ready actor and grows lazily to
    the effective execute cap; positive values additionally cap actor count.
    Parent-owned CUDA tensor storage remains alive for IPC-backed actors.

    ### Rule 1 — Classify shared state
    * *Immutable after init*: model weights (parameters), config paths,
      binary paths — safe to share, no protection needed.
    * *Mutable per-request*: model buffers, random state — must be
      isolated per request (see Rules 2–3).

    ### Rule 2 — Use ``safe_model_copy()`` (PyTorch adapters only)
    Never use ``copy.copy(model)`` — it shares mutable buffers.
    Use ``from modelworker.concurrency_utils import safe_model_copy``
    which shallow-copies (shares weights) but clones all buffers.
    Non-PyTorch adapters must implement their own state isolation.

    ### Rule 3 — Serialization via ``max_inflight_batches()``
    SERIALIZED adapters override ``max_inflight_batches() → 1``.
    The engine spawns only **one** execute loop, naturally serializing
    ``execute_batch()`` calls.  **No locks are needed.**

    ### Rule 4 — Automatic engine-level helpers
    The engine handles these automatically:
    * Execute-loop capping (via ``max_inflight_batches``)
    * GC debounce (``maybe_gc()`` after each batch)
    * Targeted actor lifecycle and hard cancellation
    Adapters should focus on business logic only.

    ### Rule 5 — Declare concurrency limits
    If your adapter cannot support concurrent ``execute_batch()``
    (e.g. process-global singletons like Hydra), override
    ``max_inflight_batches() → 1``.

    ### Rule 6 — No direct GC calls
    Do not call ``gc.collect()`` or ``torch.cuda.empty_cache()``
    directly.  The engine calls ``maybe_gc()`` after each batch.

    ### Rule 7 — File output isolation
    Use unique subdirectories or temp directories per request.
    If writing to a shared path, use atomic rename
    (write to temp file → ``os.replace()`` to final path).

    ### Rule 8 — Prepare should be CPU-only
    Prepare stage may run with higher concurrency than execute.
    Avoid creating GPU tensors in ``prepare_one`` / ``prepare_batch``
    to prevent VRAM accumulation in the prepared queue.

    ### Concurrency Safety Levels
    The label is a *declaration* of how the adapter handles concurrency;
    the engine does not interpret it as a serialization request.  Actual
    concurrency is bounded by VRAM admission at the gateway and by
    ``max_inflight_batches()`` (which defaults to 0 = no adapter cap).

    * **SERIALIZED** (default):  GPU adapters that achieve per-request
      isolation themselves through the patterns in
      ``<docs>`` (memo deepcopy, shared
      eval/no_grad reference, or per-framework buffer cloning).  The
      engine admits these concurrently; the label is for telemetry.
    * **FULL**: No shared mutable state at all (subprocess-based or
      otherwise stateless adapters like Vina-GPU, MMSeqs2).  Same
      engine behavior as SERIALIZED; the distinct label is telemetry.
    * **SINGLE**: Hard adapter-side constraint (e.g. a process-global
      singleton such as Hydra's ``GlobalHydra`` in BoltzGen) that
      requires ``max_inflight_batches() == 1``; the engine actually
      caps the in-flight count to one.

    Override ``concurrency_safety_level()`` to declare your level.
    """

    def model_name(self) -> str:
        """Human-readable model identifier surfaced via GetCapabilities."""
        return "unknown"

    def model_version(self) -> str:
        """Model version or tag surfaced via GetCapabilities."""
        return "unknown"

    def supported_buckets(self) -> list[str]:
        """Routing hints (e.g., input size buckets) for schedulers."""
        return []

    def max_batch_size(self) -> int:
        """Upper bound used by schedulers (not enforced by the engine)."""
        return 1

    def max_inflight_batches(self) -> int:
        """Hard adapter-side cap on concurrent execute_batch() calls.

        Return 0 (default): no adapter-side cap — concurrency is controlled
        entirely by ``execute_concurrency`` from workers.yaml / runtime.yaml,
        which itself is bounded by VRAM admission at the gateway level.

        Override to return 1 only if the adapter has a hard constraint that
        prevents any concurrent execution (e.g., Hydra global singleton).
        Operational concurrency tuning should be done in config, not here.
        """
        return 0

    def concurrency_safety_level(self) -> str:
        """Declare concurrency stance for engine telemetry / behavior.

        The label is a *declaration* about how the adapter handles
        concurrent ``execute_batch()`` calls; it does NOT request that
        the engine serialize them.  Actual concurrency is bounded by
        VRAM admission at the gateway and by ``max_inflight_batches()``
        (which defaults to 0 = no adapter cap).

        * ``'serialized'`` (default): GPU adapters that achieve
          per-request isolation themselves through the patterns in
          ``<docs>`` (memo deepcopy, shared
          eval/no_grad reference, or per-framework buffer cloning).
          The engine admits these concurrently; the label is for
          telemetry.
        * ``'full'``: no shared mutable state at all (subprocess-based
          or otherwise stateless adapters).  Same engine behavior as
          ``'serialized'``; the distinct label is for telemetry.
        * ``'single'``: hard adapter-side constraint (e.g. a
          process-global singleton such as Hydra's ``GlobalHydra``)
          that requires ``max_inflight_batches() == 1``; the engine
          actually caps the in-flight count to one.

        Default is ``'serialized'`` — the common case for GPU model
        adapters using the isolation patterns above.
        """
        return "serialized"

    def init_prepare(self) -> Any:
        """One-time setup for prepare stage (e.g., tokenizer)."""
        return None

    def init_execute(self) -> Any:
        """One-time setup for execute stage (e.g., load model weights)."""
        return None

    def export_execute_ctx_for_processes(self, execute_ctx: Any) -> Any:
        """Return process-transferable state for persistent actor execution.

        The default passes the execute context through torch.multiprocessing's
        serializer. CUDA tensors use IPC rather than copied storage, and the
        parent must retain the exported objects for every actor lifetime.
        Adapters with non-picklable state may export a smaller handle bundle.
        """
        return execute_ctx

    def import_execute_ctx_in_child(self, exported_execute_ctx: Any) -> Any:
        """Rebuild child-process execute state exported by the parent."""
        return exported_execute_ctx

    def init_finalize(self) -> Any:
        """One-time setup for finalize stage (e.g., post-processing caches)."""
        return None

    def prepare_one(self, request: dict[str, Any], prepare_ctx: Any) -> Any:
        """Prepare a single request. Raise to mark per-item error."""
        raise NotImplementedError

    def prepare_batch(
        self,
        requests: list[dict[str, Any]],
        prepare_ctx: Any,
    ) -> list[Any] | None:
        """Optional batch prepare. Return None to fall back to prepare_one."""
        return None

    def execute_batch(
        self,
        prepared: list[Any],
        bucket_id: str | None,
        params: dict[str, str],
        execute_ctx: Any,
        *,
        cancelled: threading.Event | None = None,
    ) -> list[Any]:
        """Run model inference over a prepared batch.

        Args:
            cancelled: Explicit ``-1`` thread-backend token set by CancelBatch.
                       Adapters that declare the kwarg should poll it at safe
                       checkpoints. Persistent actors are hard-killed by the
                       engine and never receive this token.
        """
        raise NotImplementedError

    def finalize_one(self, output: Any, finalize_ctx: Any) -> Any:
        """Finalize a single output into a response payload."""
        raise NotImplementedError

    def finalize_batch(
        self,
        outputs: list[Any],
        finalize_ctx: Any,
    ) -> list[Any] | None:
        """Optional batch finalize. Return None to fall back to finalize_one."""
        return None
