"""AlphaFold3 adapter for the resident worker runtime."""

from __future__ import annotations

import datetime
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from modelworker.model_adapter import ModelAdapter

_LOG = logging.getLogger(__name__)
_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}
_DEFAULT_BUCKETS = (
    256,
    512,
    768,
    1024,
    1280,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
)


class AlphaFold3Adapter(ModelAdapter):
    """Adapter for the target ``run_alphafold.py --norun_data_pipeline`` path."""

    def __init__(
        self, *, model_dir: Optional[str] = None, repo_root: Optional[str] = None
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self._repo_root = Path(
            repo_root
            or os.environ.get("ALPHAFOLD3_REPO_ROOT")
            or root / "third_parties" / "alphafold3"
        )
        self._model_dir = str(
            model_dir
            or os.environ.get("ALPHAFOLD3_MODEL_DIR")
            or "/mnt/ssd/proton/alphafold3/models"
        )
        self._version = os.environ.get("ALPHAFOLD3_MODEL_VERSION", "alphafold3")
        self._resident_config = {
            "model_dir": self._model_dir,
            "num_diffusion_samples": self._env_int(
                "ALPHAFOLD3_NUM_DIFFUSION_SAMPLES", 1
            ),
            "num_diffusion_steps": self._env_int("ALPHAFOLD3_NUM_DIFFUSION_STEPS", 200),
            "num_recycles": self._env_int("ALPHAFOLD3_NUM_RECYCLES", 10),
            "flash_attention_implementation": os.environ.get(
                "ALPHAFOLD3_FLASH_ATTENTION_IMPLEMENTATION", "triton"
            ),
        }

    def model_name(self) -> str:
        return "alphafold3"

    def model_version(self) -> str:
        return self._version

    def max_batch_size(self) -> int:
        return 1

    def concurrency_safety_level(self) -> str:
        return "serialized"

    def max_inflight_batches(self) -> int:
        return 1

    def init_execute(self) -> Any:
        """Export only lightweight config; the spawned actor owns JAX state."""
        return {"resident_config": dict(self._resident_config)}

    def import_execute_ctx_in_child(self, exported_execute_ctx: Any) -> Any:
        if not isinstance(exported_execute_ctx, dict) or not isinstance(
            exported_execute_ctx.get("resident_config"), dict
        ):
            raise TypeError("AlphaFold3 actor context requires resident_config")
        self._resident_config = dict(exported_execute_ctx["resident_config"])
        af3 = self._import_upstream()
        cache_dir = os.environ.get("ALPHAFOLD3_JAX_COMPILATION_CACHE_DIR")
        if cache_dir:
            af3.jax.config.update("jax_compilation_cache_dir", cache_dir)

        gpu_device = self._env_int("ALPHAFOLD3_GPU_DEVICE", 0)
        devices = af3.jax.local_devices(backend="gpu")
        if not devices:
            raise RuntimeError("AlphaFold3 requires a JAX GPU device")
        if gpu_device >= len(devices):
            raise RuntimeError(
                f"ALPHAFOLD3_GPU_DEVICE={gpu_device} but only "
                f"{len(devices)} GPU(s) are visible"
            )

        runner = af3.ModelRunner(
            config=af3.make_model_config(
                flash_attention_implementation=self._resident_config[
                    "flash_attention_implementation"
                ],
                num_diffusion_samples=self._resident_config["num_diffusion_samples"],
                num_diffusion_steps=self._resident_config["num_diffusion_steps"],
                num_recycles=self._resident_config["num_recycles"],
                return_embeddings=False,
                return_distogram=False,
            ),
            device=devices[gpu_device],
            model_dir=Path(self._resident_config["model_dir"]),
        )
        _ = runner.model_params
        return {
            "af3": af3,
            "runner": runner,
            "resident_config": dict(self._resident_config),
        }

    def prepare_one(self, request: Dict[str, Any], prepare_ctx: Any) -> Dict[str, Any]:
        req = (
            self._from_nextflow_task(request) if request.get("argv") else dict(request)
        )
        normalized = self._normalize_request(req)
        normalized["fold_inputs"] = list(self._load_fold_inputs(normalized))
        return normalized

    def execute_batch(
        self,
        prepared: List[Dict[str, Any]],
        bucket_id: Optional[str],
        params: Dict[str, str],
        execute_ctx: Any,
        *,
        cancelled=None,
    ) -> List[Any]:
        outputs = []
        for req in prepared:
            if cancelled is not None and cancelled.is_set():
                break
            try:
                outputs.append(self._execute_one(req, execute_ctx, cancelled=cancelled))
            except Exception as exc:
                _LOG.exception("AlphaFold3 request failed: %s", req.get("request_id"))
                outputs.append(
                    {
                        "request_id": req.get("request_id"),
                        "output_dir": req.get("request_output_dir"),
                        "error": str(exc),
                    }
                )
        return outputs

    def finalize_one(self, output: Any, finalize_ctx: Any) -> Dict[str, Any]:
        if output.get("error"):
            raise RuntimeError(f"AlphaFold3 execution failed: {output['error']}")
        if not output.get("cif_files"):
            raise RuntimeError("AlphaFold3 execution produced no CIF outputs")
        return output

    def _execute_one(
        self, req: Dict[str, Any], execute_ctx: Dict[str, Any], *, cancelled=None
    ) -> Dict[str, Any]:
        if req["run_data_pipeline"]:
            raise ValueError(
                "AlphaFold3 adapter targets the no-data-pipeline invocation"
            )
        self._assert_resident_config(req, execute_ctx["resident_config"])
        af3 = execute_ctx["af3"]
        output_dir = Path(req["request_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        for fold_input in req["fold_inputs"]:
            if cancelled is not None and cancelled.is_set():
                break
            af3.process_fold_input(
                fold_input=fold_input,
                data_pipeline_config=None,
                model_runner=execute_ctx["runner"],
                output_dir=output_dir / fold_input.sanitised_name(),
                buckets=req["buckets"],
                ref_max_modified_date=datetime.date.fromisoformat(
                    req["max_template_date"]
                ),
                conformer_max_iterations=req["conformer_max_iterations"],
                resolve_msa_overlaps=req["resolve_msa_overlaps"],
                force_output_dir=True,
            )
        return self._collect_result(req)

    def _normalize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req = dict(request)
        json_path = req.get("json_path")
        if not json_path:
            raise ValueError("AlphaFold3 request must include json_path")
        request_id = str(
            req.get("request_id") or req.get("task_id") or uuid.uuid4().hex[:12]
        )
        output_root = Path(str(req.get("output_dir") or "results"))
        normalized = {
            "request_id": request_id,
            "json_path": str(json_path),
            "request_output_dir": str(output_root),
            "model_dir": str(req.get("model_dir") or self._model_dir),
            "run_data_pipeline": self._coerce_bool(req.get("run_data_pipeline"), False),
            "buckets": self._parse_buckets(req.get("buckets")),
            "max_template_date": str(req.get("max_template_date") or ""),
            "conformer_max_iterations": self._optional_int(
                req.get("conformer_max_iterations")
            ),
            "resolve_msa_overlaps": self._coerce_bool(
                req.get("resolve_msa_overlaps"), True
            ),
            "num_diffusion_samples": self._int_value(
                req.get("num_diffusion_samples"),
                self._resident_config["num_diffusion_samples"],
            ),
            "num_diffusion_steps": self._int_value(
                req.get("num_diffusion_steps"),
                self._resident_config["num_diffusion_steps"],
            ),
            "num_recycles": self._int_value(
                req.get("num_recycles"), self._resident_config["num_recycles"]
            ),
        }
        normalized["request_output_dir"] = str(
            Path(normalized["request_output_dir"])
            / f"proton_af3_{self._safe_name(request_id)}"
        )
        return normalized

    def _from_nextflow_task(self, request: Dict[str, Any]) -> Dict[str, Any]:
        parsed = self._parse_cli_args([str(arg) for arg in request.get("argv") or []])
        workdir = Path(
            str(request.get("workdir") or (request.get("env") or {}).get("PWD") or ".")
        )
        merged = dict(request)
        merged.update(parsed)
        for key in ("json_path", "output_dir"):
            if key in merged and not Path(str(merged[key])).is_absolute():
                merged[key] = str(workdir / str(merged[key]))
        return merged

    @staticmethod
    def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        i = 0
        while i < len(argv):
            arg = argv[i]
            if not arg.startswith("-"):
                i += 1
                continue
            raw_key = arg.lstrip("-")
            if "=" in raw_key:
                name, value = raw_key.split("=", 1)
                parsed[name.replace("-", "_")] = value
                i += 1
                continue
            key = raw_key.replace("-", "_")
            if key.startswith("no"):
                parsed[key[2:]] = False
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                parsed[key] = argv[i + 1]
                i += 2
            else:
                parsed[key] = True
                i += 1
        return parsed

    def _import_upstream(self) -> Any:
        repo_root = str(self._repo_root)
        if self._repo_root.exists() and repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import run_alphafold as af3_run

        return af3_run

    def _load_fold_inputs(self, req: dict[str, Any]) -> Iterable[Any]:
        return self._import_upstream().folding_input.load_fold_inputs_from_path(
            Path(req["json_path"])
        )

    def _collect_result(self, req: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(req["request_output_dir"])
        files = sorted(path for path in output_dir.rglob("*") if path.is_file())
        return {
            "request_id": req["request_id"],
            "output_dir": str(output_dir),
            "cif_files": [str(path) for path in files if path.suffix == ".cif"],
            "confidence_files": [
                str(path) for path in files if path.name.endswith("_confidences.json")
            ],
            "ranking_score_files": [
                str(path) for path in files if path.name.endswith("_ranking_scores.csv")
            ],
            "error": None,
        }

    def _assert_resident_config(
        self, req: dict[str, Any], resident: dict[str, Any]
    ) -> None:
        for key in (
            "model_dir",
            "num_diffusion_samples",
            "num_diffusion_steps",
            "num_recycles",
        ):
            if str(req[key]) != str(resident[key]):
                raise ValueError(
                    f"AlphaFold3 resident worker initialized with {key}={resident[key]!r}, got {req[key]!r}"
                )

    @staticmethod
    def _parse_buckets(raw: Any) -> tuple[int, ...]:
        if raw in (None, ""):
            return _DEFAULT_BUCKETS
        parts = raw.replace(",", " ").split() if isinstance(raw, str) else list(raw)
        try:
            return tuple(int(part) for part in parts)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid AlphaFold3 buckets: {raw!r}") from exc

    @staticmethod
    def _safe_name(value: str) -> str:
        return (
            "".join(
                ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value).strip()
            )
            or "alphafold3_job"
        )

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        raw = str(value).strip().lower()
        if raw in _TRUE:
            return True
        if raw in _FALSE:
            return False
        return default

    @staticmethod
    def _int_value(value: Any, default: int) -> int:
        try:
            return int(default if value in (None, "") else value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid AlphaFold3 integer: {value!r}") from exc

    @classmethod
    def _optional_int(cls, value: Any) -> int | None:
        try:
            return None if value in (None, "") else int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid AlphaFold3 integer: {value!r}") from exc

    @classmethod
    def _env_int(cls, name: str, default: int) -> int:
        return cls._int_value(os.environ.get(name), default)
