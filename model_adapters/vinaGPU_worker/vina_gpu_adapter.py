"""Vina-GPU adapter mimicking the reference code EXACTLY."""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid as _uuid
from pathlib import Path
from typing import Any

from modelworker.model_adapter import ModelAdapter

_LOG = logging.getLogger(__name__)


class VinaGPUAdapter(ModelAdapter):
    """Spawn-safe Vina-GPU CLI adapter."""

    def __init__(
        self,
        *,
        model_name: str = "vinagpu_v2.1",
        binary_path: str | None = None,
        dump_dir: str | None = None,
    ) -> None:
        self._model_name = os.environ.get("VINA_GPU_MODEL_NAME", model_name)


        candidates = [
            "/vina/AutoDock-Vina-GPU-2.1/QuickVina2-GPU-2-1",
            "/vina/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1",
            "QuickVina2-GPU-2-1",
            "AutoDock-Vina-GPU-2-1",
        ]

        if binary_path:
            candidates.insert(0, binary_path)

        self._binary_path: str = next(
            (c for c in candidates if os.path.exists(c) or shutil.which(c)),
            str(os.environ.get("VINA_GPU_BINARY") or candidates[1]),
        )

        self._dump_dir = os.environ.get(
            "VINA_GPU_DUMP_DIR", dump_dir or "/tmp/vinagpu_outputs"
        )
        self._log = logging.getLogger(__name__)

    def model_name(self) -> str:
        return "vinagpu"

    def model_version(self) -> str:
        return self._model_name

    def concurrency_safety_level(self) -> str:
        return "full"

    def init_execute(self) -> Any:
        self._log.info(
            f"Initializing Vina-GPU Adapter with binary: {self._binary_path}"
        )
        if not os.path.exists(self._binary_path):
            self._log.warning(f"Binary not found at {self._binary_path}")
        return self._binary_path

    def prepare_one(self, request: dict[str, Any], prepare_ctx: Any) -> Any:
        if request.get("argv"):
            request = self._from_nextflow_task(request)
        return self._normalize_request(request)

    def _from_nextflow_task(self, request: dict[str, Any]) -> dict[str, Any]:
        """Parse CLI arguments."""
        argv = request.get("argv") or []
        if not isinstance(argv, list):
            return request
        overrides = request.copy()

        i = 0
        while i < len(argv):
            arg = str(argv[i])
            if arg.startswith("-"):
                key = arg.lstrip("-")
                is_next_value = False
                if i + 1 < len(argv):
                    next_arg = str(argv[i + 1])
                    if not next_arg.startswith("-"):
                        is_next_value = True
                    else:
                        try:
                            float(next_arg)
                            is_next_value = True
                        except ValueError:
                            is_next_value = False

                if is_next_value:
                    overrides[key] = argv[i + 1]
                    i += 2
                else:
                    overrides[key] = True
                    i += 1
            else:
                i += 1
        return overrides

    def _normalize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        req = dict(request)

        req.setdefault("thread", 2048)
        req.setdefault("search_depth", 3)
        req.setdefault("batch_size", 1)
        req.setdefault("size_x", 20)
        req.setdefault("size_y", 20)
        req.setdefault("size_z", 20)
        req.setdefault("center_x", 0)
        req.setdefault("center_y", 0)
        req.setdefault("center_z", 0)

        if "sample_name" not in req:
            if "ligand" in req and isinstance(req["ligand"], str):
                req["sample_name"] = Path(req["ligand"]).stem
            else:
                try:
                    req["sample_name"] = f"job_{int(time.time() * 1000)}"
                except (OSError, OverflowError, ValueError):
                    req["sample_name"] = "job_unknown"

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

        outputs = []

        for req in prepared:
            self._log.info(f"Processing Vina-GPU request for {req.get('sample_name')}")

            result_meta = {
                "sample_name": req.get("sample_name"),
                "docked_pdbqt": None,
                "scores": [],
                "log_file": None,
                "error": None,
            }

            tmp_dir_obj = tempfile.TemporaryDirectory(
                prefix=f"vinagpu_{req['sample_name']}_"
            )
            tmp_root = Path(tmp_dir_obj.name)

            try:
                binary_parent = Path(self._binary_path).resolve().parent
                prebaked_bins = sorted(binary_parent.glob("Kernel*.bin"))
                if prebaked_bins:
                    opencl_dir = binary_parent
                else:
                    opencl_dir = tmp_root
                    opencl_src_dir = binary_parent / "OpenCL"
                    if opencl_src_dir.is_dir():
                        opencl_link = tmp_root / "OpenCL"
                        if not opencl_link.exists():
                            os.symlink(opencl_src_dir, opencl_link)


                base_dir = tmp_root / "docking"
                targets_dir = base_dir / "targets"
                ligands_dir = base_dir / "ligands"
                docked_dir = base_dir / "docked"

                targets_dir.mkdir(parents=True, exist_ok=True)
                ligands_dir.mkdir(parents=True, exist_ok=True)
                docked_dir.mkdir(parents=True, exist_ok=True)

                run_cwd = Path(req.get("workdir", self._dump_dir))

                receptor_raw = req.get("receptor")
                if receptor_raw is None:
                    raise FileNotFoundError("Receptor not found: missing 'receptor'")
                receptor_path = Path(receptor_raw)
                if not receptor_path.is_absolute():
                    receptor_path = run_cwd / receptor_path

                if not receptor_path.exists():
                    raise FileNotFoundError(f"Receptor not found: {receptor_path}")

                receptor_filename = receptor_path.name
                shutil.copy(receptor_path, targets_dir / receptor_filename)

                ligand_raw = req.get("ligand")
                if not ligand_raw:
                    ligand_dir_raw = req.get("ligand_directory")
                    if not ligand_dir_raw:
                        raise FileNotFoundError(
                            "Vina-GPU: neither 'ligand' nor 'ligand_directory' "
                            "provided in request — cannot stage docking input."
                        )
                    ligand_dir = Path(ligand_dir_raw)
                    if not ligand_dir.is_absolute():
                        ligand_dir = run_cwd / ligand_dir
                    if not ligand_dir.exists():
                        raise FileNotFoundError(
                            f"Vina-GPU: ligand_directory not found: {ligand_dir}"
                        )
                    pdbqt_files = sorted(ligand_dir.glob("*.pdbqt"))
                    if not pdbqt_files:
                        raise FileNotFoundError(
                            f"Vina-GPU: no *.pdbqt files in {ligand_dir}"
                        )
                    if len(pdbqt_files) > 1:
                        self._log.warning(
                            "Vina-GPU: ligand_directory %s contains %d files; "
                            "using first %s (NF wrapper convention is single ligand "
                            "per task workdir).",
                            ligand_dir,
                            len(pdbqt_files),
                            pdbqt_files[0].name,
                        )
                    ligand_src = pdbqt_files[0]
                else:
                    ligand_src = Path(ligand_raw)
                    if not ligand_src.is_absolute():
                        ligand_src = run_cwd / ligand_src
                    if not ligand_src.exists():
                        raise FileNotFoundError(f"Ligand not found: {ligand_src}")

                ligand_filename = ligand_src.name
                shutil.copy(ligand_src, ligands_dir / ligand_filename)


                rel_receptor = f"docking/targets/{receptor_filename}"
                rel_ligand_dir = "docking/ligands/"

                if "workdir" in req:
                    output_dir = Path(req["workdir"]) / "output_ligands"
                else:
                    output_dir = docked_dir

                output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    str(self._binary_path),
                    "--receptor",
                    rel_receptor,
                    "--ligand_directory",
                    rel_ligand_dir,
                    "--output_directory",
                    str(output_dir) + os.sep,
                    "--center_x",
                    str(req["center_x"]),
                    "--center_y",
                    str(req["center_y"]),
                    "--center_z",
                    str(req["center_z"]),
                    "--size_x",
                    str(req["size_x"]),
                    "--size_y",
                    str(req["size_y"]),
                    "--size_z",
                    str(req["size_z"]),
                    "--thread",
                    str(req["thread"]),
                    "--search_depth",
                    str(req["search_depth"]),
                    "--opencl_binary_path",
                    str(opencl_dir),
                ]

                batch_size = max(1, int(req.get("batch_size", 1)))
                self._log.info(
                    f"Running command (cwd={tmp_root}, batch_size={batch_size}): {' '.join(cmd)}"
                )

                proc_result = subprocess.run(
                    cmd,
                    cwd=str(tmp_root),
                    capture_output=True,
                    text=True,
                )

                log_content = proc_result.stdout + "\n" + proc_result.stderr
                log_path = tmp_root / "vina.log"
                log_path.write_text(log_content)
                result_meta["log_file"] = str(log_path)

                if proc_result.returncode != 0:
                    self._log.error(
                        f"Vina-GPU exited with code {proc_result.returncode}"
                    )
                    self._log.error(f"Stdout: {proc_result.stdout}")
                    self._log.error(f"Stderr: {proc_result.stderr}")
                    raise RuntimeError(f"Binary returned {proc_result.returncode}")

                expected_out_name = Path(ligand_filename).stem + "_out.pdbqt"
                docked_file = output_dir / expected_out_name

                if not docked_file.exists():
                    found = list(output_dir.glob("*.pdbqt"))
                    if found:
                        docked_file = found[0]
                    else:
                        self._log.error(f"Vina Log: {log_content}")
                        raise FileNotFoundError(f"No output file found in {output_dir}")

                unique_id = Path(tmp_root).name.split("_")[-1]
                final_out_dir = Path(self._dump_dir) / req["sample_name"] / unique_id
                final_out_dir.mkdir(parents=True, exist_ok=True)

                final_pdbqt = final_out_dir / docked_file.name
                shutil.copy(docked_file, final_pdbqt)
                shutil.copy(log_path, final_out_dir / "vina.log")
                legacy_out_dir = Path(self._dump_dir) / req["sample_name"]
                legacy_out_dir.mkdir(parents=True, exist_ok=True)
                _tmp = legacy_out_dir / f".tmp_{_uuid.uuid4().hex}"
                shutil.copy(docked_file, _tmp)
                os.replace(str(_tmp), str(legacy_out_dir / docked_file.name))
                _tmp = legacy_out_dir / f".tmp_{_uuid.uuid4().hex}"
                shutil.copy(log_path, _tmp)
                os.replace(str(_tmp), str(legacy_out_dir / "vina.log"))

                self._log.info(f"Output generated at: {docked_file}")
                scores = []
                for line in log_content.splitlines():
                    match = re.search(r"^\s+\d+\s+(-?\d+\.\d+)\s+\d+\.\d+", line)
                    if match:
                        scores.append(float(match.group(1)))

                result_meta["scores"] = scores
                self._log.info(
                    f"Success {req['sample_name']}, Best: {scores[0] if scores else 'N/A'}"
                )

            except Exception as e:
                self._log.error(f"Error processing {req.get('sample_name')}: {e}")
                result_meta["error"] = str(e)
            finally:
                pass

            outputs.append(result_meta)

        return outputs

    def finalize_one(self, output: Any, finalize_ctx: Any) -> Any:
        if output.get("error"):
            raise RuntimeError(f"Vina-GPU execution failed: {output['error']}")
        return output
