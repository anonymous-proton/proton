"""RFdiffusion adapter for the worker skeleton."""

from __future__ import annotations

import ast
import copy
import glob
import logging
import os
import pickle
import re
import random
import time
from pathlib import Path
from numbers import Integral
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf

from rfdiffusion.inference import utils as iu
from rfdiffusion.util import writepdb, writepdb_multi
from modelworker.model_adapter import ModelAdapter


class RFDiffusionAdapter(ModelAdapter):
    """Adapter that reuses a single sampler instance across requests."""

    def __init__(
        self,
        *,
        base_config_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        schedule_dir: Optional[str] = None,
    ) -> None:
        if not base_config_path:
            base_config_path = os.environ.get("RFDIFFUSION_BASE_CONFIG", "")
        if not model_dir:
            model_dir = os.environ.get("RFDIFFUSION_MODEL_DIR", "")
        if not schedule_dir:
            schedule_dir = os.environ.get("RFDIFFUSION_SCHEDULE_DIR", "")

        self._base_config_path = Path(base_config_path) if base_config_path else None
        self._model_dir = model_dir or None
        self._schedule_dir = schedule_dir or None
        self._sampler = None
        self._log = logging.getLogger(__name__)

    def model_name(self) -> str:
        return "rfdiffusion"

    def model_version(self) -> str:
        return "v0"

    def init_execute(self) -> Any:
        base_path = self._base_config_path or self._default_base_config()
        conf = OmegaConf.load(base_path)
        if self._model_dir:
            conf.inference.model_directory_path = self._model_dir
        if self._schedule_dir:
            conf.inference.schedule_directory_path = self._schedule_dir
        sampler = iu.sampler_selector(conf)
        self._sampler = sampler
        return sampler

    def max_batch_size(self) -> int:
        return int(os.environ.get("MAX_BATCH_SIZE", "1"))

    def prepare_one(self, request: Dict[str, Any], prepare_ctx: Any) -> Any:
        if request.get("argv"):
            request = self._from_nextflow_task(request)
        prepared = self._normalize_request(request)
        self._validate_request(prepared)
        return prepared

    def execute_batch(
        self,
        prepared: List[Any],
        bucket_id: Optional[str],
        params: Dict[str, str],
        execute_ctx: Any,
        *,
        cancelled=None,
    ) -> List[Any]:
        sampler = execute_ctx
        outputs = []
        for idx, req in enumerate(prepared, start=1):
            if cancelled is not None and cancelled.is_set():
                self._log.warning("[execute] cancelled before request %s/%s", idx, len(prepared))
                break

            self._log.info("[execute] start request %s/%s", idx, len(prepared))

            memo = {id(sampler.model): sampler.model, id(sampler.allatom): sampler.allatom}
            local_sampler = copy.deepcopy(sampler, memo)

            try:
                result = self._run_request(local_sampler, req, cancelled=cancelled)
                outputs.append(result)
            finally:
                local_sampler.model = None
                del local_sampler

            self._log.info("[execute] done request %s/%s", idx, len(prepared))
        return outputs

    def finalize_one(self, output: Any, finalize_ctx: Any) -> Any:
        """
        Return success/failure status.

        No need to return file manifest - Nextflow collects output files
        via wildcard patterns defined in gw_signature.
        """
        if isinstance(output, dict) and output.get("pdb_files"):
            return {"ok": True}
        return output

    @staticmethod
    def _default_base_config() -> Path:
        here = Path(__file__).resolve()
        return here.parents[2] / "config" / "inference" / "base.yaml"

    def _normalize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request = dict(request)
        normalized = {
            "inference": dict(request.get("inference", {})),
            "contigmap": dict(request.get("contigmap", {})),
            "potentials": dict(request.get("potentials", {})),
        }

        if "input_pdb" in request:
            normalized["inference"]["input_pdb"] = request["input_pdb"]
        if "output_prefix" in request:
            normalized["inference"]["output_prefix"] = request["output_prefix"]
        if "num_designs" in request:
            normalized["inference"]["num_designs"] = request["num_designs"]
        if "design_startnum" in request:
            normalized["inference"]["design_startnum"] = request["design_startnum"]
        if "deterministic" in request:
            normalized["inference"]["deterministic"] = request["deterministic"]
        if "seed_list" in request:
            normalized["inference"]["seed_list"] = request["seed_list"]
        if "cautious" in request:
            normalized["inference"]["cautious"] = request["cautious"]
        if "write_trajectory" in request:
            normalized["inference"]["write_trajectory"] = request["write_trajectory"]
        if "profile" in request:
            normalized["inference"]["profile"] = request["profile"]

        if "contigs" in request:
            normalized["contigmap"]["contigs"] = request["contigs"]

        if "guide_scale" in request:
            normalized["potentials"]["guide_scale"] = request["guide_scale"]
        if "guiding_potentials" in request:
            normalized["potentials"]["guiding_potentials"] = request[
                "guiding_potentials"
            ]
        if "substrate" in request:
            normalized["potentials"]["substrate"] = request["substrate"]

        if "model_dir" in request:
            normalized["model_dir"] = request["model_dir"]
        if "schedule_dir" in request:
            normalized["schedule_dir"] = request["schedule_dir"]

        return normalized

    def _from_nextflow_task(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Hydra-style argv into a nested dict.

        argv: ["python", "script.py", "inference.input_pdb=/x.pdb", ...]
        → {"inference": {"input_pdb": "/x.pdb"}, ...}
        """
        argv = request.get("argv") or []
        workdir = request.get("workdir", "") or ""
        if not isinstance(argv, list):
            return request

        overrides = [arg for arg in argv if "=" in arg]

        parsed = OmegaConf.from_dotlist(overrides)
        result = OmegaConf.to_container(parsed, resolve=True)

        if workdir:
            result["_workdir"] = workdir
        return result

    def _validate_request(self, request: Dict[str, Any]) -> None:
        inference = request.get("inference", {})
        contigmap = request.get("contigmap", {})

        missing = []
        if not inference.get("input_pdb"):
            missing.append("input_pdb")
        if not inference.get("output_prefix"):
            missing.append("output_prefix")
        if inference.get("num_designs") in (None, ""):
            missing.append("num_designs")
        if not contigmap.get("contigs"):
            missing.append("contigs")
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

    def _run_request(self, sampler: Any, request: Dict[str, Any], *, cancelled=None) -> Dict[str, Any]:
        self._apply_request(sampler, request)
        return self._run_inference(sampler, cancelled=cancelled)

    def _apply_request(self, sampler: Any, request: Dict[str, Any]) -> None:
        conf = sampler._conf

        model_dir = request.get("model_dir")
        if model_dir:
            if conf.inference.model_directory_path is None:
                conf.inference.model_directory_path = model_dir
            elif conf.inference.model_directory_path != model_dir:
                raise ValueError("model_dir differs from loaded model")

        schedule_dir = request.get("schedule_dir")
        if schedule_dir:
            conf.inference.schedule_directory_path = schedule_dir

        inference = request.get("inference", {})
        contigmap = request.get("contigmap", {})
        potentials = request.get("potentials", {})

        for key, value in inference.items():
            setattr(conf.inference, key, value)
        for key, value in contigmap.items():
            setattr(conf.contigmap, key, value)
        for key, value in potentials.items():
            if key == "guiding_potentials":
                value = self._normalize_guiding_potentials(value)
            setattr(conf.potentials, key, value)

        if conf.inference.profile:
            sampler.reset_timers()

    def _run_inference(self, sampler: Any, *, cancelled=None) -> Dict[str, Any]:
        log = self._log
        inf_conf = sampler.inf_conf

        if inf_conf.deterministic:
            self._make_deterministic()

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            log.info("Found GPU with device_name %s", device_name)
        else:
            log.info("No GPU detected; falling back to CPU")

        design_startnum = inf_conf.design_startnum
        if design_startnum == -1:
            existing = glob.glob(inf_conf.output_prefix + "*.pdb")
            indices = [-1]
            for entry in existing:
                match = re.match(r".*_(\d+)\.pdb$", entry)
                if not match:
                    continue
                indices.append(int(match.groups()[0]))
            design_startnum = max(indices) + 1

        num_designs = int(inf_conf.num_designs)
        design_indices = list(range(design_startnum, design_startnum + num_designs))
        seed_list = self._expand_seed_list(inf_conf.seed_list, num_designs, log)
        seed_map = None
        if seed_list is not None:
            seed_map = {design_indices[i]: seed_list[i] for i in range(len(design_indices))}

        results = {
            "output_prefix": inf_conf.output_prefix,
            "design_indices": [],
            "skipped_indices": [],
            "pdb_files": [],
            "trb_files": [],
            "traj_files": [],
        }

        log.info(
            "Starting inference: designs=%s output_prefix=%s",
            num_designs,
            inf_conf.output_prefix,
        )
        for i_idx, i_des in enumerate(design_indices, start=1):
            if cancelled is not None and cancelled.is_set():
                log.warning("Cancelled before design %s/%s", i_idx, num_designs)
                break

            if seed_map is not None:
                self._make_deterministic(seed_map[i_des])
            elif inf_conf.deterministic:
                self._make_deterministic(i_des)

            out_prefix = f"{inf_conf.output_prefix}_{i_des}"
            log.info("Design %s/%s -> %s", i_idx, num_designs, out_prefix)
            if inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
                results["skipped_indices"].append(i_des)
                log.info("Design %s skipped (cautious)", i_des)
                continue

            start_time = time.time()
            x_init, seq_init = sampler.sample_init()
            denoised_xyz_stack = []
            px0_xyz_stack = []
            seq_stack = []
            plddt_stack = []

            x_t = torch.clone(x_init)
            seq_t = torch.clone(seq_init)
            num_reverse_steps = int(sampler.t_step_input) - inf_conf.final_step + 1
            log.info(
                "Reverse diffusion steps per design: %s (t_step_input=%s, final_step=%s)",
                num_reverse_steps,
                sampler.t_step_input,
                inf_conf.final_step,
            )

            for t in range(int(sampler.t_step_input), inf_conf.final_step - 1, -1):
                if cancelled is not None and cancelled.is_set():
                    log.warning("Cancelled at denoising step t=%s for design %s", t, i_des)
                    break

                px0, x_t, seq_t, plddt = sampler.sample_step(
                    t=t, x_t=x_t, seq_init=seq_t, final_step=inf_conf.final_step
                )
                px0_xyz_stack.append(px0)
                denoised_xyz_stack.append(x_t)
                seq_stack.append(seq_t)
                plddt_stack.append(plddt[0])

            denoised_xyz_stack = torch.stack(denoised_xyz_stack)
            denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0])
            px0_xyz_stack = torch.stack(px0_xyz_stack)
            px0_xyz_stack = torch.flip(px0_xyz_stack, [0])
            plddt_stack = torch.stack(plddt_stack)

            os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
            final_seq = seq_stack[-1]
            final_seq = torch.where(
                torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
            )

            bfacts = torch.ones_like(final_seq.squeeze())
            bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
            out = f"{out_prefix}.pdb"

            writepdb(
                out,
                denoised_xyz_stack[0, :, :4],
                final_seq,
                sampler.binderlen,
                chain_idx=sampler.chain_idx,
                bfacts=bfacts,
                idx_pdb=sampler.idx_pdb,
            )
            results["pdb_files"].append(out)

            trb = dict(
                config=OmegaConf.to_container(sampler._conf, resolve=True),
                plddt=plddt_stack.cpu().numpy(),
                device=torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "CPU",
                time=time.time() - start_time,
            )
            if hasattr(sampler, "contig_map"):
                for key, value in sampler.contig_map.get_mappings().items():
                    trb[key] = value
            trb_path = f"{out_prefix}.trb"
            with open(trb_path, "wb") as f_out:
                pickle.dump(trb, f_out)
            results["trb_files"].append(trb_path)
            log.info("Design %s done in %.2fs", i_des, time.time() - start_time)

            if inf_conf.write_trajectory:
                traj_prefix = (
                    os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
                )
                os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

                out = f"{traj_prefix}_Xt-1_traj.pdb"
                writepdb_multi(
                    out,
                    denoised_xyz_stack,
                    bfacts,
                    final_seq.squeeze(),
                    use_hydrogens=False,
                    backbone_only=False,
                    chain_ids=sampler.chain_idx,
                )
                results["traj_files"].append(out)

                out = f"{traj_prefix}_pX0_traj.pdb"
                writepdb_multi(
                    out,
                    px0_xyz_stack,
                    bfacts,
                    final_seq.squeeze(),
                    use_hydrogens=False,
                    backbone_only=False,
                    chain_ids=sampler.chain_idx,
                )
                results["traj_files"].append(out)

            results["design_indices"].append(i_des)
            log.info("Finished design %s in %.2f minutes", out_prefix, (time.time() - start_time) / 60.0)

        if inf_conf.profile and hasattr(sampler, "timing"):
            timing = sampler.timing
            steps = max(int(timing.get("steps", 0)), 1)
            total = timing["preprocess"] + timing["model"] + timing["postprocess"]
            results["timing"] = {
                "preprocess_s": timing["preprocess"],
                "model_s": timing["model"],
                "postprocess_s": timing["postprocess"],
                "total_s": total,
                "steps": steps,
            }

        return results

    @staticmethod
    def _normalize_guiding_potentials(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
        if isinstance(value, ListConfig):
            return list(value)
        if isinstance(value, (list, tuple)):
            return list(value)
        return value

    @staticmethod
    def _make_deterministic(seed: int = 0) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _expand_seed_list(seed_list: Any, num_designs: int, log: logging.Logger) -> Optional[List[int]]:
        if seed_list is None:
            return None
        if isinstance(seed_list, str):
            seed_str = seed_list.strip()
            if not seed_str:
                return None
            try:
                seed_list = ast.literal_eval(seed_str)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    f"inference.seed_list must be an int or list-like, got {seed_list!r}"
                ) from exc
        if isinstance(seed_list, ListConfig):
            seed_list = list(seed_list)
        if isinstance(seed_list, Integral):
            base_seed = int(seed_list)
            return [base_seed + i for i in range(num_designs)]
        if isinstance(seed_list, (list, tuple)):
            seeds = [int(seed) for seed in seed_list]
            if len(seeds) < num_designs:
                raise ValueError(
                    f"inference.seed_list length {len(seeds)} < num_designs {num_designs}"
                )
            if len(seeds) > num_designs:
                log.warning(
                    "inference.seed_list has %d entries; truncating to num_designs=%d",
                    len(seeds),
                    num_designs,
                )
                seeds = seeds[:num_designs]
            return seeds
        raise ValueError(
            "inference.seed_list must be an int or list-like, got "
            f"{type(seed_list).__name__}"
        )
