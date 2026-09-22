# PROTON GPU Pipelines Workspace

This repository is evolving from "one container per stage" execution toward a clear control-plane / data-plane architecture that keeps models resident in memory and minimizes stage-boundary serialization.

This README is a tutorial for the new design: what we are optimizing for, how the pieces fit together, and how to run it end to end.

## Design goals (why this exists)

Academic goals:

- Keep heavyweight models loaded once and reuse them across many requests to minimize memory duplication.
- Avoid stage-boundary synchronization and repeated container cold starts.
- Standardize model serving behind a small, fixed skeleton so onboarding new components is cheap.

More like implementation goals:

- Preserve our "mount-first" development model: code and data live on the host and are mounted into containers; images are dependency bases, not code snapshots.

## Architecture overview (what runs where)

```
# Control plane
gateway/ # Control plane
├─ __main__.py: the gateway supervisor entrypoint.
├─ supervisor.py: the gateway supervisor implementation.
└─ __init__.py: package marker.

# Data plane
modelworker/
├─ worker_server.py: gRPC server entrypoint inside the worker container.
│     - Loads adapter from CLI args passed by the gateway.
│     - Starts WorkerEngine (prepare/execute/finalize pipeline).
│     - Serves Health/InferBatch.
├─ worker_engine.py: async pipeline + queues, batch timing, per-item fallback.
├─ model_adapter.py: adapter interface (prepare/execute/finalize hooks).
├─ proto/modelworker.proto: gRPC service definition.
├─ modelworker_pb2.py: generated protobuf messages.
├─ nextflow_contract.md: Nextflow-Gateway contract.
└─ modelworker_pb2_grpc.py: generated gRPC stubs.

# Adapters
third_parties/RFdiffusion/rfdiffusion/serving/
└─ rfdiffusion_adapter.py: RFdiffusion adapter implementation (resident sampler).

```

## Configuration System (Split Architecture)

To simplify management across different environments (local, k8s, slurm), the configuration is split into two parts:

1. **Worker Config (`configs/workers.yaml`)**:
    - Defines **what** models to run (image, adapter, VRAM requirements, env vars).
    - This is the single source of truth for model specifications.
2. **Runtime Config (`configs/runtime.proton.yaml`)**:
    - Defines **how** to run them (GPU pool, scheduling policy, profiling options).
    - Single canonical file (`runtime.proton.yaml`).  Earlier multi-environment variants (`runtime.default.yaml`, `runtime.gpu*.yaml`, `runtime.k8s*.yaml`, `runtime.slurm*.yaml`) were removed in cleanup — the proton config is the only production-tested target.  Copy and edit it for new environments.

...