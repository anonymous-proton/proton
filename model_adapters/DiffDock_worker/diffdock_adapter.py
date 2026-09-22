"""DiffDock adapter for the worker skeleton."""

from __future__ import annotations

import copy
import logging
import os
import tempfile
import uuid
from argparse import Namespace
from functools import partial
from typing import Any

import numpy as np
import torch
import yaml
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader

from modelworker.concurrency_utils import safe_model_copy
from modelworker.model_adapter import ModelAdapter


def _sinusoidal_timestep_embedding(
    timesteps: Any,
    *,
    embedding_scale: float,
    embedding_dim: int,
) -> Any:
    """Picklable equivalent of DiffDock's local sinusoidal lambda."""
    from utils.diffusion_utils import (
        sinusoidal_embedding,
    )

    return sinusoidal_embedding(embedding_scale * timesteps, embedding_dim)


def _make_timestep_embedding_picklable(model: Any, args: Any) -> None:
    if getattr(args, "embedding_type", "sinusoidal") != "sinusoidal":
        return
    model.timestep_emb_func = partial(
        _sinusoidal_timestep_embedding,
        embedding_scale=float(getattr(args, "embedding_scale", 10000)),
        embedding_dim=int(args.sigma_embed_dim),
    )


class DiffDockAdapter(ModelAdapter):
    """Adapter for DiffDock molecular docking."""

    def __init__(
        self,
        *,
        model_name: str = "diffdock_default",
        model_dir: str | None = None,
        confidence_model_dir: str | None = None,
        inference_config_path: str | None = None,
        base_work_dir: str = "/tmp/diffdock_work",
    ) -> None:
        self._model_name = os.environ.get("DIFFDOCK_MODEL_NAME", model_name)

        default_model_dir = os.environ.get(
            "DIFFDOCK_MODEL_DIR", "workdir/v1.1/score_model"
        )
        default_conf_dir = os.environ.get(
            "DIFFDOCK_CONFIDENCE_DIR", "workdir/v1.1/confidence_model"
        )
        default_inf_conf = os.environ.get(
            "DIFFDOCK_INFERENCE_CONFIG", "default_inference_args.yaml"
        )

        self._model_dir = model_dir or default_model_dir
        self._confidence_model_dir = confidence_model_dir or default_conf_dir
        self._inference_config_path = inference_config_path or default_inf_conf
        self._base_work_dir = os.environ.get("DIFFDOCK_WORK_DIR", base_work_dir)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log = logging.getLogger(__name__)
        self._inference_defaults: dict[str, Any] = {}

    def model_name(self) -> str:
        return "diffdock"

    def model_version(self) -> str:
        return self._model_name

    def max_batch_size(self) -> int:
        return int(os.environ.get("MAX_BATCH_SIZE", "1"))

    def init_execute(self) -> Any:
        """Load DiffDock models and Inference Defaults."""
        self._log.info("Initializing DiffDock...")
        self._log.info(
            "Starting Lazy Imports of DiffDock modules... (This might take a while)"
        )

        try:
            from datasets.process_mols import (
                write_mol_with_coords,
            )
            from utils.diffusion_utils import (
                get_t_schedule,
            )
            from utils.diffusion_utils import (
                t_to_sigma as t_to_sigma_compl,
            )
            from utils.inference_utils import (
                InferenceDataset,
            )
            from utils.sampling import (
                randomize_position,
                sampling,
            )
            from utils.utils import get_model

        except ImportError:
            import sys

            self._log.warning(f"Import failed. Current sys.path: {sys.path}")
            raise

        self._log.info("DiffDock modules imported successfully.")

        modules = {
            "DataLoader": DataLoader,
            "InferenceDataset": InferenceDataset,
            "write_mol_with_coords": write_mol_with_coords,
            "get_t_schedule": get_t_schedule,
            "randomize_position": randomize_position,
            "sampling": sampling,
        }

        self._log.info(f"Score Model Dir: {self._model_dir}")
        self._log.info(f"Confidence Model Dir: {self._confidence_model_dir}")

        if os.path.exists(self._inference_config_path):
            with open(self._inference_config_path) as f:
                self._inference_defaults = yaml.safe_load(f) or {}
            self._log.info(
                f"Loaded inference defaults from {self._inference_config_path}"
            )
        else:
            self._log.warning(
                f"Inference config not found at {self._inference_config_path}. Using hardcoded defaults."
            )
            self._inference_defaults = {}

        if not os.path.exists(self._model_dir):
            raise FileNotFoundError(f"Score model dir not found: {self._model_dir}")

        with open(f"{self._model_dir}/model_parameters.yml") as f:
            score_model_args = Namespace(**yaml.full_load(f))

        confidence_args = None
        if self._confidence_model_dir and os.path.exists(self._confidence_model_dir):
            with open(f"{self._confidence_model_dir}/model_parameters.yml") as f:
                confidence_args = Namespace(**yaml.full_load(f))

        t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

        score_ckpt_name = self._inference_defaults.get(
            "ckpt", "best_ema_inference_epoch_model.pt"
        )
        conf_ckpt_name = self._inference_defaults.get(
            "confidence_ckpt", "best_model_epoch75.pt"
        )
        old_score_model = self._inference_defaults.get("old_score_model", False)
        old_confidence_model = self._inference_defaults.get(
            "old_confidence_model", True
        )

        model = get_model(
            score_model_args,
            self._device,
            t_to_sigma=t_to_sigma,
            no_parallel=True,
            old=old_score_model,
        )
        _make_timestep_embedding_picklable(model, score_model_args)

        ckpt_path = f"{self._model_dir}/{score_ckpt_name}"
        if not os.path.exists(ckpt_path):
            self._log.warning(
                f"Checkpoint {ckpt_path} not found. Trying best_model.pt fallback."
            )
            ckpt_path = f"{self._model_dir}/best_model.pt"

        self._log.info(f"Loading Score Model from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self._device)
        model.eval()

        confidence_model = None
        if confidence_args:
            confidence_model = get_model(
                confidence_args,
                self._device,
                t_to_sigma=t_to_sigma,
                no_parallel=True,
                confidence_mode=True,
                old=old_confidence_model,
            )
            _make_timestep_embedding_picklable(confidence_model, confidence_args)
            conf_ckpt_path = f"{self._confidence_model_dir}/{conf_ckpt_name}"
            if not os.path.exists(conf_ckpt_path):
                self._log.warning(
                    f"Conf Checkpoint {conf_ckpt_path} not found. Trying best_model.pt fallback."
                )
                conf_ckpt_path = f"{self._confidence_model_dir}/best_model.pt"

            self._log.info(f"Loading Confidence Model from {conf_ckpt_path}")
            state_dict = torch.load(conf_ckpt_path, map_location=torch.device("cpu"))
            confidence_model.load_state_dict(state_dict, strict=True)
            confidence_model = confidence_model.to(self._device)
            confidence_model.eval()

        return {
            "model": model,
            "confidence_model": confidence_model,
            "score_args": score_model_args,
            "conf_args": confidence_args,
            "t_to_sigma": t_to_sigma,
            "modules": modules,
        }

    def prepare_one(self, request: dict[str, Any], prepare_ctx: Any) -> Any:
        if request.get("argv"):
            request = self._from_nextflow_task(request)

        prepared = self._normalize_request(request)
        return prepared

    def _parse_cli_args(self, argv: list[str]) -> dict[str, Any]:
        parsed = {}
        i = 0
        while i < len(argv):
            arg = argv[i]
            if not arg.startswith("-"):
                i += 1
                continue
            key = arg.lstrip("-")
            if "=" in key:
                key, val = key.split("=", 1)
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                val = argv[i + 1]
                i += 2
            else:
                val = True
                i += 1
            parsed[key] = val
        return parsed

    def _from_nextflow_task(self, request: dict[str, Any]) -> dict[str, Any]:
        argv = request.get("argv") or []
        if not isinstance(argv, list):
            return request

        cli_args = self._parse_cli_args(argv)
        overrides = request.copy()

        int_keys = [
            "samples_per_complex",
            "batch_size",
            "inference_steps",
            "actual_steps",
            "gnina_poses_to_optimize",
        ]
        float_keys = [
            "initial_noise_std_proportion",
            "temp_sampling_tr",
            "temp_psi_tr",
            "temp_sigma_data_tr",
            "temp_sampling_rot",
            "temp_psi_rot",
            "temp_sigma_data_rot",
            "temp_sampling_tor",
            "temp_psi_tor",
            "temp_sigma_data_tor",
            "gnina_autobox_add",
        ]
        bool_keys = [
            "save_visualisation",
            "no_final_step_noise",
            "old_score_model",
            "old_confidence_model",
            "choose_residue",
            "gnina_minimize",
            "gnina_full_dock",
        ]

        init_args = [
            "model_dir",
            "model_name",
            "confidence_model_dir",
            "inference_config_path",
            "base_work_dir",
            "device",
        ]

        for k, v in cli_args.items():
            if k in init_args:
                continue

            if k in int_keys:
                overrides[k] = int(v)
            elif k in float_keys:
                overrides[k] = float(v)
            elif k in bool_keys:
                overrides[k] = v if isinstance(v, bool) else v.lower() == "true"
            else:
                overrides[k] = v

        return overrides

    def _normalize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        merged_req = {
            "config": "default_inference_args.yaml",
            "protein_ligand_csv": None,
            "complex_name": None,
            "protein_path": None,
            "protein_sequence": None,
            "ligand_description": "CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1",
            "loglevel": "WARNING",
            "out_dir": "results/user_inference",
            "save_visualisation": False,
            "samples_per_complex": 10,
            "model_dir": None,
            "ckpt": "best_ema_inference_epoch_model.pt",
            "confidence_model_dir": None,
            "confidence_ckpt": "best_model.pt",
            "batch_size": 10,
            "no_final_step_noise": True,
            "inference_steps": 20,
            "actual_steps": None,
            "old_score_model": False,
            "old_confidence_model": True,
            "initial_noise_std_proportion": -1.0,
            "choose_residue": False,
            "temp_sampling_tr": 1.0,
            "temp_psi_tr": 0.0,
            "temp_sigma_data_tr": 0.5,
            "temp_sampling_rot": 1.0,
            "temp_psi_rot": 0.0,
            "temp_sigma_data_rot": 0.5,
            "temp_sampling_tor": 1.0,
            "temp_psi_tor": 0.0,
            "temp_sigma_data_tor": 0.5,
            "gnina_minimize": False,
            "gnina_path": "gnina",
            "gnina_log_file": "gnina_log.txt",
            "gnina_full_dock": False,
            "gnina_autobox_add": 4.0,
            "gnina_poses_to_optimize": 1,
        }

        config_path = request.get("config", merged_req["config"])
        if config_path and os.path.exists(config_path):
            try:
                import yaml

                with open(config_path) as f:
                    config_dict = yaml.load(f, Loader=yaml.FullLoader) or {}

                for key, value in config_dict.items():
                    if key not in request:
                        if isinstance(value, list) and isinstance(
                            merged_req.get(key), list
                        ):
                            merged_req[key].extend(value)
                        else:
                            merged_req[key] = value
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to load config file {config_path}: {e}"
                )

        merged_req.update(request)

        if not merged_req.get("complex_name"):
            merged_req["complex_name"] = f"complex_{uuid.uuid4().hex[:8]}"

        return merged_req

    def execute_batch(
        self,
        prepared: list[dict[str, Any]],
        bucket_id: str | None,
        params: dict[str, str],
        execute_ctx: Any,
    ) -> list[Any]:

        model = execute_ctx["model"]
        confidence_model = execute_ctx["confidence_model"]
        score_args = execute_ctx["score_args"]
        conf_args = execute_ctx["conf_args"]
        t_to_sigma = execute_ctx["t_to_sigma"]

        mods = execute_ctx["modules"]
        InferenceDataset = mods["InferenceDataset"]
        DataLoader = mods["DataLoader"]
        get_t_schedule = mods["get_t_schedule"]
        randomize_position = mods["randomize_position"]
        sampling = mods["sampling"]
        write_mol_with_coords = mods["write_mol_with_coords"]

        outputs = []

        if not os.path.exists(self._base_work_dir):
            os.makedirs(self._base_work_dir, exist_ok=True)

        for req in prepared:
            final_out_dir = os.path.join(req["out_dir"], req["complex_name"])
            os.makedirs(final_out_dir, exist_ok=True)

            with tempfile.TemporaryDirectory(dir=self._base_work_dir) as tmp_work_dir:
                complex_name = req["complex_name"]
                protein_path = req.get("protein_path")
                protein_seq = req.get("protein_sequence")
                ligand_desc = req["ligand_description"]

                if (
                    protein_path
                    and not os.path.exists(protein_path)
                    and (protein_path.startswith("ATOM") or len(protein_path) > 255)
                ):
                    pdb_file = os.path.join(tmp_work_dir, "input_protein.pdb")
                    with open(pdb_file, "w") as f:
                        f.write(protein_path)
                    protein_path = pdb_file

                result_meta = {"complex_name": complex_name, "poses": [], "error": None}

                try:
                    local_model = safe_model_copy(model)
                    local_conf_model = (
                        safe_model_copy(confidence_model) if confidence_model else None
                    )

                    test_dataset = InferenceDataset(
                        out_dir=tmp_work_dir,
                        complex_names=[complex_name],
                        protein_files=[protein_path],
                        ligand_descriptions=[ligand_desc],
                        protein_sequences=[protein_seq],
                        lm_embeddings=True,
                        receptor_radius=score_args.receptor_radius,
                        remove_hs=score_args.remove_hs,
                        c_alpha_max_neighbors=score_args.c_alpha_max_neighbors,
                        all_atoms=score_args.all_atoms,
                        atom_radius=score_args.atom_radius,
                        atom_max_neighbors=score_args.atom_max_neighbors,
                        knn_only_graph=False
                        if not hasattr(score_args, "not_knn_only_graph")
                        else not score_args.not_knn_only_graph,
                    )
                    test_loader = DataLoader(
                        dataset=test_dataset, batch_size=1, shuffle=False
                    )

                    confidence_dataset = None
                    if (
                        confidence_model is not None
                        and not conf_args.use_original_model_cache
                    ):
                        confidence_dataset = InferenceDataset(
                            out_dir=tmp_work_dir,
                            complex_names=[complex_name],
                            protein_files=[protein_path],
                            ligand_descriptions=[ligand_desc],
                            protein_sequences=[protein_seq],
                            lm_embeddings=True,
                            receptor_radius=conf_args.receptor_radius,
                            remove_hs=conf_args.remove_hs,
                            c_alpha_max_neighbors=conf_args.c_alpha_max_neighbors,
                            all_atoms=conf_args.all_atoms,
                            atom_radius=conf_args.atom_radius,
                            atom_max_neighbors=conf_args.atom_max_neighbors,
                            precomputed_lm_embeddings=test_dataset.lm_embeddings,
                            knn_only_graph=False
                            if not hasattr(score_args, "not_knn_only_graph")
                            else not score_args.not_knn_only_graph,
                        )

                    tr_schedule = get_t_schedule(
                        inference_steps=req["inference_steps"],
                        sigma_schedule=req.get("sigma_schedule", "expbeta"),
                    )

                    for idx, orig_complex_graph in enumerate(test_loader):
                        if not orig_complex_graph.success[0]:
                            raise RuntimeError("Failed to process complex graph")
                        original_center = getattr(
                            orig_complex_graph, "original_center", None
                        )
                        if original_center is None:
                            raise RuntimeError("Failed to process complex graph")

                        data_list = [
                            copy.deepcopy(orig_complex_graph)
                            for _ in range(req["samples_per_complex"])
                        ]
                        confidence_data_list = (
                            [
                                copy.deepcopy(confidence_dataset[idx])
                                for _ in range(req["samples_per_complex"])
                            ]
                            if confidence_dataset
                            else None
                        )

                        randomize_position(
                            data_list,
                            score_args.no_torsion,
                            False,
                            score_args.tr_sigma_max,
                            initial_noise_std_proportion=req[
                                "initial_noise_std_proportion"
                            ],
                            choose_residue=req["choose_residue"],
                        )

                        data_list, confidence = sampling(
                            data_list=data_list,
                            model=local_model,
                            inference_steps=req["actual_steps"]
                            or req["inference_steps"],
                            tr_schedule=tr_schedule,
                            rot_schedule=tr_schedule,
                            tor_schedule=tr_schedule,
                            device=self._device,
                            t_to_sigma=t_to_sigma,
                            model_args=score_args,
                            confidence_model=local_conf_model,
                            confidence_data_list=confidence_data_list,
                            confidence_model_args=conf_args,
                            batch_size=req["batch_size"],
                            no_final_step_noise=req["no_final_step_noise"],
                            temp_sampling=[
                                req["temp_sampling_tr"],
                                req["temp_sampling_rot"],
                                req["temp_sampling_tor"],
                            ],
                            temp_psi=[
                                req["temp_psi_tr"],
                                req["temp_psi_rot"],
                                req["temp_psi_tor"],
                            ],
                            temp_sigma_data=[
                                req["temp_sigma_data_tr"],
                                req["temp_sigma_data_rot"],
                                req["temp_sigma_data_tor"],
                            ],
                        )

                        lig = orig_complex_graph.mol[0]
                        ligand_pos = np.asarray(
                            [
                                g["ligand"].pos.cpu().numpy()
                                + original_center.cpu().numpy()
                                for g in data_list
                            ]
                        )

                        if confidence is not None:
                            if isinstance(conf_args.rmsd_classification_cutoff, list):
                                confidence = confidence[:, 0]
                            confidence = confidence.cpu().numpy()
                            re_order = np.argsort(confidence)[::-1]
                            confidence, ligand_pos = (
                                confidence[re_order],
                                ligand_pos[re_order],
                            )

                        for rank, pos in enumerate(ligand_pos):
                            mol_pred = copy.deepcopy(lig)
                            if score_args.remove_hs:
                                mol_pred = RemoveAllHs(mol_pred)

                            conf_val = (
                                confidence[rank] if confidence is not None else 0.0
                            )

                            if rank == 0:
                                rank1_path = os.path.join(final_out_dir, "rank1.sdf")
                                write_mol_with_coords(mol_pred, pos, rank1_path)

                            file_name = f"rank{rank + 1}_confidence{conf_val:.2f}.sdf"
                            sdf_path = os.path.join(final_out_dir, file_name)
                            write_mol_with_coords(mol_pred, pos, sdf_path)

                            with open(sdf_path) as f:
                                sdf_content = f.read()

                            result_meta["poses"].append(
                                {
                                    "rank": rank + 1,
                                    "sdf_content": sdf_content,
                                    "confidence": float(conf_val),
                                    "file_name": file_name,
                                }
                            )

                except Exception as e:
                    self._log.error(f"DiffDock failed for {complex_name}: {e}")
                    result_meta["error"] = str(e)

                outputs.append(result_meta)

        return outputs

    def finalize_one(self, output: Any, finalize_ctx: Any) -> Any:
        if output.get("error"):
            raise RuntimeError(f"DiffDock failed: {output['error']}")
        return output
