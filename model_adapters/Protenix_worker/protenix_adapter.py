"""Protenix adapter for the worker skeleton."""

from __future__ import annotations

import copy
import logging
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from configs.configs_base import (
    configs as configs_base,
)
from configs.configs_data import data_configs
from configs.configs_inference import (
    inference_configs,
)
from configs.configs_model_type import model_configs
from ml_collections.config_dict import ConfigDict
from protenix.config import parse_configs
from protenix.data.infer_data_pipeline import (
    get_inference_dataloader,
)
from protenix.utils.seed import seed_everything
from runner.inference import (
    DataDumper,
    InferenceRunner,
    download_infercence_cache,
    update_gpu_compatible_configs,
    update_inference_configs,
)

from modelworker.concurrency_utils import classify_sticky_cuda_error, maybe_gc
from modelworker.model_adapter import ModelAdapter

try:
    from .ccd_cache_guard import install_ccd_atom_array_copy_guard
except ImportError:
    from ccd_cache_guard import install_ccd_atom_array_copy_guard

_LOG = logging.getLogger(__name__)


class ProtenixAdapter(ModelAdapter):
    """Adapter that wraps Protenix InferenceRunner."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        checkpoint_dir: str | None = None,
        dump_dir: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if not model_name:
            model_name = os.environ.get(
                "PROTENIX_MODEL_NAME", "protenix_base_default_v0.5.0"
            )
        if not checkpoint_dir:
            checkpoint_dir = os.environ.get("PROTENIX_MODEL_DIR", "")
        if not dump_dir:
            dump_dir = os.environ.get("PROTENIX_DUMP_DIR", "/tmp/protenix_outputs")

        if not cache_dir:
            cache_dir = os.environ.get("PROTENIX_CACHE_DIR", checkpoint_dir)

        self._model_name = model_name
        self._checkpoint_dir = checkpoint_dir or None
        self._cache_dir = cache_dir or None
        self._dump_dir = dump_dir
        self._config = None

        self._runner = None
        self._log = logging.getLogger(__name__)

    def model_name(self) -> str:
        return "protenix"

    def model_version(self) -> str:
        return self._model_name

    def max_batch_size(self) -> int:
        raw = os.environ.get("MAX_BATCH_SIZE", "1")
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid MAX_BATCH_SIZE: {raw!r}") from exc

    def init_execute(self) -> Any:
        self._log.info("Initializing Protenix Adapter...")
        install_ccd_atom_array_copy_guard()

        raw_configs = {
            **configs_base,
            "data": data_configs,
            **inference_configs,
        }

        configs = parse_configs(
            configs=raw_configs,
            arg_str="",
            fill_required_with_null=True,
        )

        configs.model_name = self._model_name
        configs.dump_dir = self._dump_dir
        self._apply_resident_worker_overrides(configs)

        if self._checkpoint_dir:
            configs.load_checkpoint_dir = self._checkpoint_dir

        if self._cache_dir:
            self._log.info(f"Overriding data cache paths to: {self._cache_dir}")

            cache_keys = [
                "ccd_components_file",
                "ccd_components_rdkit_mol_file",
                "pdb_cluster_file",
            ]

            for key in cache_keys:
                if key in configs.data:
                    original_filename = os.path.basename(configs.data[key])
                    new_path = os.path.join(self._cache_dir, original_filename)
                    configs.data[key] = new_path

                    self._log.info(f"Configured {key} -> {new_path}")

                    if not os.path.exists(new_path):
                        self._log.warning(
                            f"Cache file not found at overridden path: {new_path}. Download might trigger."
                        )

        if configs.model_name not in model_configs:
            raise ValueError(f"Unknown model_name: {configs.model_name}")

        model_specifics = ConfigDict(model_configs[configs.model_name])
        configs.update(model_specifics)
        configs = update_gpu_compatible_configs(configs)

        download_infercence_cache(configs)

        self._runner = InferenceRunner(configs)
        self._log.info(f"Protenix initialized. Model: {configs.model_name}")

        self._config = configs

        return self._runner

    def import_execute_ctx_in_child(self, exported_execute_ctx: Any) -> Any:
        """Reinstall process-local CCD cache protection in each spawn actor."""
        install_ccd_atom_array_copy_guard()
        return exported_execute_ctx

    @staticmethod
    def _resident_num_workers() -> int:
        raw = os.environ.get("PROTENIX_INFERENCE_NUM_WORKERS", "0")
        try:
            parsed = int(raw)
        except Exception:
            _LOG.warning(
                "Invalid PROTENIX_INFERENCE_NUM_WORKERS=%r; using 0 for resident worker",
                raw,
            )
            return 0
        return max(0, parsed)

    def _apply_resident_worker_overrides(self, configs: Any) -> None:
        """Apply resident-worker safety overrides before creating DataLoaders.

        Protenix upstream defaults to ``num_workers=16`` for standalone
        inference.  Inside the resident gRPC worker, that forks DataLoader
        workers while the process already has active gRPC / executor threads.
        In practice this can leave an execute thread blocked in
        ``DataLoader.__next__`` after inference has already produced outputs,
        preventing the gateway from receiving the final BatchResponse.  Keep
        resident inference single-process by default; users can explicitly
        opt back in via ``PROTENIX_INFERENCE_NUM_WORKERS`` for experiments.
        """
        configs.num_workers = self._resident_num_workers()

    def prepare_one(self, request: dict[str, Any], prepare_ctx: Any) -> Any:
        if request.get("argv"):
            request = self._from_nextflow_task(request)

        prepared = self._normalize_request(request)
        return prepared

    def _parse_cli_args(self, argv: list[str]) -> dict[str, Any]:
        """Helper to parse argparse-style arguments."""
        parsed = {}
        i = 0
        while i < len(argv):
            arg = argv[i]
            if not arg.startswith("-"):
                i += 1
                continue

            key = arg.lstrip("-")
            val = True

            if "=" in key:
                key, val = key.split("=", 1)
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                val = argv[i + 1]
                i += 2
            else:
                i += 1

            parsed[key] = val
        return parsed

    def _from_nextflow_task(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Parse CLI arguments from Nextflow/Shell script into request dictionary.
        Mappings based on inference_demo.sh
        """
        argv = request.get("argv") or []
        if not isinstance(argv, list):
            return request

        cli_args = self._parse_cli_args(argv)
        overrides = request.copy()


        try:
            if "N_sample" in cli_args:
                overrides["sample_diffusion.N_sample"] = int(cli_args["N_sample"])

            if "N_step" in cli_args:
                overrides["sample_diffusion.N_step"] = int(cli_args["N_step"])

            if "N_cycle" in cli_args:
                overrides["model.N_cycle"] = int(cli_args["N_cycle"])

            if "seed" in cli_args:
                overrides["seeds"] = [int(cli_args["seed"])]
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid numeric Protenix CLI argument") from exc

        if "input_json_path" in cli_args:
            overrides["input_json_path"] = cli_args["input_json_path"]

        if "dump_dir" in cli_args:
            overrides["dump_dir"] = cli_args["dump_dir"]

        if "triangle_attention" in cli_args:
            overrides["triangle_attention"] = cli_args["triangle_attention"]

        if "triangle_multiplicative" in cli_args:
            overrides["triangle_multiplicative"] = cli_args["triangle_multiplicative"]

        return overrides

    def _normalize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate and populate defaults based on inference_demo.sh"""
        req = dict(request)

        if "sample_name" not in req:
            req["sample_name"] = f"job_{time.time_ns() // 1_000_000}"

        req.setdefault("seeds", [101])
        req.setdefault("triangle_attention", "triattention")
        req.setdefault("triangle_multiplicative", "cuequivariance")

        req.setdefault("model.N_cycle", 10)
        req.setdefault("sample_diffusion.N_sample", 5)
        req.setdefault("sample_diffusion.N_step", 200)

        if (
            "sequence" not in req
            and "msa_path" not in req
            and "input_json_path" not in req
        ):
            self._log.warning(
                f"Request {req['sample_name']} missing 'sequence', 'msa_path' or 'input_json_path'"
            )

        return req

    def execute_batch(
        self,
        prepared: list[dict[str, Any]],
        bucket_id: str | None,
        params: dict[str, str],
        execute_ctx: Any,
        *,
        cancelled=None,
    ) -> list[Any]:
        runner = execute_ctx
        outputs = []

        for idx, req in enumerate(prepared):
            if cancelled is not None and cancelled.is_set():
                self._log.warning(
                    f"[execute] Cancelled before request {idx + 1}/{len(prepared)}"
                )
                break

            self._log.info(
                f"[execute] Processing request {idx + 1}/{len(prepared)}: {req.get('sample_name')}"
            )

            if "dump_dir" in req:
                req_dump_dir = os.path.abspath(req["dump_dir"])
            else:
                raise ValueError("dump_dir not present in argv")
            try:
                os.makedirs(req_dump_dir, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"failed to create Protenix dump directory: {req_dump_dir}"
                ) from exc

            if self._config is None:
                raise RuntimeError("Protenix execute config is not initialized")
            local_dumper = DataDumper(
                base_dir=req_dump_dir,
                need_atom_confidence=self._config.need_atom_confidence,
                sorted_by_ranking_score=self._config.sorted_by_ranking_score,
            )

            with tempfile.TemporaryDirectory() as _tmp_dir:
                if "input_json_path" in req:
                    if os.path.exists(req["input_json_path"]):
                        input_json_path = os.path.abspath(req["input_json_path"])
                    else:
                        raise ValueError(
                            f"input_json file not present: {req['input_json_path']}"
                        )
                else:
                    raise ValueError("input_json_path not present in request")

                current_configs = (
                    runner.configs.copy_and_resolve_references()
                    if hasattr(runner.configs, "copy_and_resolve_references")
                    else runner.configs.copy()
                )

                current_configs.input_json_path = input_json_path
                current_configs.dump_dir = req_dump_dir
                self._apply_resident_worker_overrides(current_configs)

                self._apply_config_overrides(current_configs, req)

                result_meta = {
                    "sample_name": req["sample_name"],
                    "output_dir": req_dump_dir,
                    "pdb_files": [],
                    "cif_files": [],
                    "confidence_files": [],
                    "error": None,
                }

                try:
                    dataloader = get_inference_dataloader(configs=current_configs)

                    memo = {id(runner.model): runner.model}
                    local_runner = copy.deepcopy(runner, memo)
                    local_runner.update_model_configs(current_configs)

                    self._run_inference_pipeline_parallel_safe(
                        local_runner,
                        current_configs,
                        req,
                        local_dumper,
                        dataloader,
                        cancelled=cancelled,
                    )

                    self._collect_outputs(req_dump_dir, result_meta)

                except Exception as e:
                    sticky_cuda_class = classify_sticky_cuda_error(e)
                    if sticky_cuda_class is not None:
                        self._log.error(
                            "[execute] fatal_cuda class=%s sample=%s",
                            sticky_cuda_class,
                            req["sample_name"],
                        )
                        raise
                    self._log.error(f"Inference failed for {req['sample_name']}: {e}")
                    self._log.error(traceback.format_exc())
                    result_meta["error"] = str(e)

                outputs.append(result_meta)

        return outputs

    def _apply_config_overrides(self, config: Any, overrides: dict[str, Any]) -> None:
        """Helper to apply dot-notation keys (e.g. model.N_cycle) to ConfigDict."""
        for k, v in overrides.items():
            if k in ["sample_name", "argv", "input_json_path"]:
                continue

            if k == "seeds":
                config.seeds = v
                continue

            if "." in k:
                parts = k.split(".")
                target = config
                try:
                    for part in parts[:-1]:
                        target = target.get(part)
                        if target is None:
                            break
                    if target is not None:
                        target[parts[-1]] = v
                except Exception as e:
                    self._log.warning(f"Failed to set config key {k}: {e}")
            else:
                if k in config:
                    config[k] = v

    def finalize_one(self, output: Any, finalize_ctx: Any) -> Any:
        if output.get("error"):
            raise RuntimeError(f"Protenix execution failed: {output['error']}")

        return {
            "sample_name": output["sample_name"],
            "pdb_files": output["pdb_files"],
            "cif_files": output["cif_files"],
            "confidence_files": output["confidence_files"],
            "output_dir": output["output_dir"],
        }

    def _run_inference_pipeline_parallel_safe(
        self, local_runner, configs, req, dumper, dataloader, *, cancelled=None
    ):
        """Thread-safe inference pipeline that uses a localized dumper."""
        for seed in configs.seeds:
            if cancelled is not None and cancelled.is_set():
                self._log.warning(
                    f"Cancelled before seed {seed} for {req['sample_name']}"
                )
                return

            seed_everything(seed=seed, deterministic=configs.deterministic)

            for batch in dataloader:
                if cancelled is not None and cancelled.is_set():
                    self._log.warning(
                        f"Cancelled before batch for {req['sample_name']}"
                    )
                    return
                data, atom_array, data_error_message = batch[0]

                if len(data_error_message) > 0:
                    raise RuntimeError(f"Data pipeline error: {data_error_message}")

                n_token = data["N_token"].item()
                new_configs = update_inference_configs(configs, n_token)

                local_runner.update_model_configs(new_configs)

                t_start = time.time()
                prediction = local_runner.predict(data)
                t_end = time.time()

                sample_name = data.get("sample_name", req["sample_name"])
                dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type=data["entity_poly_type"],
                )

                self._log.info(
                    f"Inference done for {req['sample_name']} seed {seed} in {t_end - t_start:.2f}s"
                )

                cleanup_interval = os.environ.get(
                    "PROTENIX_CUDA_CLEANUP_INTERVAL_S", "30.0"
                )
                try:
                    maybe_gc(interval_s=float(cleanup_interval))
                except ValueError as exc:
                    raise ValueError(
                        "invalid PROTENIX_CUDA_CLEANUP_INTERVAL_S: "
                        f"{cleanup_interval!r}"
                    ) from exc

    def _collect_outputs(self, output_dir: str, meta: dict[str, Any]) -> None:
        path = Path(output_dir)
        if not path.exists():
            return

        try:
            st = os.stat(output_dir)
            target_uid = st.st_uid
            target_gid = st.st_gid
        except Exception:
            target_uid = -1
            target_gid = -1

        for item in path.rglob("*"):
            try:
                if target_uid != -1:
                    os.chown(item, target_uid, target_gid)
                if item.is_file():
                    os.chmod(item, 0o664)
                elif item.is_dir():
                    os.chmod(item, 0o775)
            except Exception as e:
                self._log.warning(f"Failed to fix permissions for {item}: {e}")

            if item.is_file():
                if item.suffix == ".pdb":
                    meta["pdb_files"].append(str(item.resolve()))
                elif item.suffix == ".cif":
                    meta["cif_files"].append(str(item.resolve()))
                elif item.suffix == ".json":
                    meta["confidence_files"].append(str(item.resolve()))
