"""PROTON adapters for BoltzGen pipeline steps."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any

import hydra
import torch
from boltzgen.data import mol
from boltzgen.model.models.boltz import Boltz
from boltzgen.task.predict.predict import Predict
from boltzgen.utils.quiet import quiet_startup
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from modelworker.concurrency_utils import safe_model_copy
from modelworker.model_adapter import ModelAdapter

_LOG = logging.getLogger(__name__)
_moldir_cache_lock = threading.Lock()

PROTOCOL_CONFIGS = {
    "protein-anything": {},
    "peptide-anything": {
        "analysis": ["largest_hydrophobic=false", "largest_hydrophobic_refolded=false"],
        "filtering": [
            "filter_cysteine=true",
            "alpha=0.01",
            "+refolding_rmsd_threshold=2",
        ],
    },
    "protein-small_molecule": {
        "analysis": ["affinity_metrics=true"],
        "filtering": ["use_affinity=true"],
    },
    "nanobody-anything": {
        "analysis": ["largest_hydrophobic=false", "largest_hydrophobic_refolded=false"],
        "filtering": ["filter_cysteine=true"],
    },
}


class WorkerPredict(Predict):
    def __init__(self, existing_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.existing_model = existing_model

    def run(self, config: OmegaConf = None, run_prediction=True) -> None:
        quiet_startup()

        if len(self.data.predict_set) == 0:
            print("No predictions required")
            return

        with torch.no_grad():
            if self.matmul_precision is not None:
                torch.set_float32_matmul_precision(self.matmul_precision)

            if self.trainer is None:
                self.trainer = {}

            self.trainer["devices"] = 1
            self.trainer["accelerator"] = "gpu"
            self.trainer["logger"] = False

            self.model_module = safe_model_copy(self.existing_model)

            self.model_module.predict_args = dict(self.predict_args)

            _LOG.info(f"Updated model predict_args: {self.model_module.predict_args}")

            from pytorch_lightning import Trainer

            callbacks = [self.writer]

            self.lightning_trainer = Trainer(
                default_root_dir=self.output,
                strategy="auto",
                callbacks=callbacks,
                **self.trainer,
            )

            if run_prediction:
                self.lightning_trainer.predict(
                    self.model_module,
                    datamodule=self.data,
                    return_predictions=False,
                )


class BoltzGenBaseAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
        self.model = None

    def checkpoint_path(self) -> str:
        raise NotImplementedError

    def config_name(self) -> str:
        raise NotImplementedError

    def max_inflight_batches(self) -> int:
        return int(os.environ.get("MAX_INFLIGHT_BATCHES", "1"))

    def init_execute(self) -> Any:
        checkpoint = self.checkpoint_path()
        _LOG.info(f"Loading model from {checkpoint}...")

        quiet_startup()

        predict_args = {
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
        }

        self.model = Boltz.load_from_checkpoint(
            checkpoint,
            strict=True,
            map_location="cpu",
            predict_args=predict_args,
            weights_only=False,
        )

        self.model.eval()

        if torch.cuda.is_available():
            self.model.cuda()
            _LOG.info("Model moved to GPU.")

        try:
            _LOG.info("Compiling pairformer module...")
            self.model.pairformer_module = torch.compile(
                self.model.pairformer_module,
                dynamic=True,
                fullgraph=False,
            )

            _LOG.info("Compiling token transformer...")
            self.model.structure_module.score_model.token_transformer = torch.compile(
                self.model.structure_module.score_model.token_transformer,
                dynamic=True,
                fullgraph=False,
            )

        except Exception as e:
            _LOG.warning(f"Compilation failed, continuing without compile: {e}")

        _LOG.info("Model loaded and compiled successfully.")
        return self.model

    def prepare_one(self, request: dict[str, Any], prepare_ctx: Any) -> Any:
        args = request.get("argv", [])
        _LOG.info(f"Prepare request args: {args}")

        parser = argparse.ArgumentParser()
        parser.add_argument("--output", default=None)
        parser.add_argument("--num_designs", type=int, default=1)
        parser.add_argument(
            "--protocol",
            default="protein-anything",
            choices=[
                "protein-anything",
                "peptide-anything",
                "protein-small_molecule",
                "nanobody-anything",
            ],
        )
        parser.add_argument("--skip_inverse_folding", action="store_true")
        parser.add_argument("--reuse", action="store_true")
        parser.add_argument("--diffusion_batch_size", type=int, default=None)
        parser.add_argument("--inverse_fold_avoid", type=str, default=None)
        parser.add_argument("--budget", type=int, default=30)
        parser.add_argument("--sampling_steps", type=int, default=None)
        parser.add_argument("--recycling_steps", type=int, default=None)

        parsed, _ = parser.parse_known_args(args)

        input_yaml = None
        for arg in args:
            if arg.endswith((".yaml", ".yml")):
                input_yaml = arg
                break

        if input_yaml and not os.path.isabs(input_yaml):
            workdir = request.get("workdir")
            if workdir:
                input_yaml = os.path.join(workdir, input_yaml)
                _LOG.info(f"Resolved relative input_yaml to: {input_yaml}")

        parsed.input_yaml = input_yaml
        _LOG.info(f"Identified input_yaml: {input_yaml}")

        overrides = []
        if parsed.output:
            overrides.append(f"output={parsed.output}")

        if parsed.num_designs:
            overrides.append(f"+data.cfg.multiplicity={parsed.num_designs}")

        if parsed.sampling_steps is not None:
            overrides.append(f"sampling_steps={parsed.sampling_steps}")
        if parsed.recycling_steps is not None:
            overrides.append(f"recycling_steps={parsed.recycling_steps}")

        self._add_step_specific_overrides(parsed, args, overrides)

        _LOG.info(f"Generated overrides: {overrides}")
        return overrides

    def _add_step_specific_overrides(self, parsed_args, args, overrides):
        pass

    def execute_batch(
        self,
        prepared: list[Any],
        bucket_id: str | None,
        params: dict[str, str],
        execute_ctx: Any,
        *,
        cancelled=None,
    ) -> list[Any]:
        results = []
        for overrides in prepared:
            try:

                config_dir = os.environ.get("BOLTZGEN_CONFIG_DIR")
                if not config_dir:
                    raise ValueError(
                        "BOLTZGEN_CONFIG_DIR environment variable must be set."
                    )

                if not os.path.exists(config_dir):
                    raise FileNotFoundError(
                        f"Config directory not found at {config_dir}. Please set BOLTZGEN_CONFIG_DIR env var."
                    )

                moldir = os.environ.get("BOLTZGEN_MOLDIR")
                if moldir and os.path.exists(moldir):
                    overrides.append(f"data.cfg.moldir={moldir}")

                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability()
                    use_kernels = capability[0] >= 8
                else:
                    use_kernels = False
                overrides.append(f"+override.use_kernels={use_kernels}")

                with hydra.initialize_config_dir(
                    config_dir=config_dir, version_base=None
                ):
                    cfg = hydra.compose(
                        config_name=self.config_name(), overrides=overrides
                    )

                    _LOG.info(f"Instantiating data module with config: {cfg.data}")
                    data = instantiate(cfg.data)


                    _LOG.info(f"DEBUG: mol module ID: {id(mol)}")
                    mol_modules = [
                        m
                        for name, m in sys.modules.items()
                        if "boltzgen.data.mol" in name
                        and hasattr(m, "MOLDIR_ZIP_CACHE")
                    ]
                    _LOG.info(
                        f"DEBUG: Found {len(mol_modules)} mol modules in sys.modules: {[m.__name__ for m in mol_modules]}"
                    )

                    with _moldir_cache_lock:
                        for m in mol_modules:
                            _LOG.info(
                                f"DEBUG: Clearing cache for module {m.__name__} (ID: {id(m)})"
                            )
                            if hasattr(m, "MOLDIR_ZIP_CACHE"):
                                _LOG.info(
                                    f"DEBUG: Cache keys BEFORE clear: {list(m.MOLDIR_ZIP_CACHE.keys())}"
                                )
                                keys_to_remove = list(m.MOLDIR_ZIP_CACHE.keys())
                                for key in keys_to_remove:
                                    try:
                                        m.MOLDIR_ZIP_CACHE[key].close()
                                        _LOG.info(f"DEBUG: Closed ZipFile for {key}")
                                    except Exception as e:
                                        _LOG.warning(
                                            f"Failed to close ZipFile for {key}: {e}"
                                        )
                                    del m.MOLDIR_ZIP_CACHE[key]

                                _LOG.info(
                                    f"DEBUG: Cache keys AFTER clear: {list(m.MOLDIR_ZIP_CACHE.keys())}"
                                )

                    _LOG.info(f"Instantiating writer with config: {cfg.writer}")
                    writer = instantiate(cfg.writer)


                    task = WorkerPredict(
                        existing_model=self.model,
                        data=data,
                        writer=writer,
                        checkpoint=self.checkpoint_path(),
                        output=cfg.output,
                        name=cfg.name,
                        recycling_steps=cfg.get("recycling_steps", 3),
                        sampling_steps=cfg.get("sampling_steps", 200),
                        diffusion_samples=cfg.get("diffusion_samples", 1),
                        compile_pairformer=cfg.get("compile_pairformer", False),
                        compile_structure=cfg.get("compile_structure", False),
                        use_ema=cfg.get("use_ema", False),
                        override=cfg.get("override", None),
                        keys_dict_out=cfg.get("keys_dict_out", None),
                        trainer=OmegaConf.to_container(
                            cfg.get("trainer", {}), resolve=True
                        )
                        if cfg.get("trainer")
                        else {},
                    )

                    _LOG.info("Running prediction task...")
                    task.run(config=cfg)
                    failed = int(getattr(writer, "failed", 0))
                    if failed:
                        raise RuntimeError(
                            f"BoltzGen writer reported {failed} failed prediction(s)"
                        )
                    results.append({"status": "success", "output_dir": cfg.output})

            except Exception as e:
                _LOG.exception("Error during execution")
                results.append({"status": "error", "error": str(e)})
                raise e

        return results

    def finalize_one(self, output: Any, finalize_ctx: Any) -> Any:
        return output


class BoltzGenDesignAdapter(BoltzGenBaseAdapter):
    def checkpoint_path(self):
        ckpt = os.environ.get("BOLTZGEN_DESIGN_CHECKPOINT")
        if not ckpt:
            raise ValueError(
                "BOLTZGEN_DESIGN_CHECKPOINT environment variable must be set."
            )
        return ckpt

    def config_name(self):
        return "design"

    def _add_step_specific_overrides(self, parsed_args, args, overrides):
        if parsed_args.input_yaml:
            overrides.append(f"data.cfg.yaml_path=[{parsed_args.input_yaml}]")
        diffusion_batch_size = parsed_args.diffusion_batch_size
        if diffusion_batch_size is None:
            diffusion_batch_size = 1 if parsed_args.num_designs < 100 else 10
        num_batches = math.ceil(parsed_args.num_designs / diffusion_batch_size)
        overrides.append(f"diffusion_samples={diffusion_batch_size}")
        overrides[:] = [o for o in overrides if "+data.cfg.multiplicity=" not in o]
        overrides.append(f"+data.cfg.multiplicity={num_batches}")
        overrides.append(f"+data.cfg.skip_existing={parsed_args.reuse}")


class BoltzGenInverseFoldAdapter(BoltzGenBaseAdapter):
    def checkpoint_path(self):
        ckpt = os.environ.get("BOLTZGEN_IFOLD_CHECKPOINT")
        if not ckpt:
            raise ValueError(
                "BOLTZGEN_IFOLD_CHECKPOINT environment variable must be set."
            )
        return ckpt

    def config_name(self):
        return "inverse_fold"

    def _add_step_specific_overrides(self, parsed_args, args, overrides):
        if parsed_args.output:
            overrides.append(f"data.design_dir={parsed_args.output}")
        overrides.append(f"+data.skip_existing={parsed_args.reuse}")
        overrides.append("+data.skip_existing_kind=inverse_fold")
        inverse_fold_avoid = parsed_args.inverse_fold_avoid
        if inverse_fold_avoid is None:
            if parsed_args.protocol in ["peptide-anything", "nanobody-anything"]:
                inverse_fold_avoid = "C"
            else:
                inverse_fold_avoid = ""
        exclude_residues = []
        if inverse_fold_avoid:
            from boltzgen.data import const

            for letter in inverse_fold_avoid:
                exclude_residues.append(const.prot_letter_to_token[letter])
        overrides.append(
            f"override.inverse_fold_args.inverse_fold_restriction=[{', '.join(exclude_residues)}]"
        )


class BoltzGenFoldingAdapter(BoltzGenBaseAdapter):
    def checkpoint_path(self):
        ckpt = os.environ.get("BOLTZGEN_FOLDING_CHECKPOINT")
        if not ckpt:
            raise ValueError(
                "BOLTZGEN_FOLDING_CHECKPOINT environment variable must be set."
            )
        return ckpt

    def config_name(self):
        return "fold"

    def _add_step_specific_overrides(self, parsed_args, args, overrides):
        if parsed_args.output:
            overrides.append(f"data.design_dir={parsed_args.output}")
        overrides.append(f"+data.skip_existing={parsed_args.reuse}")
        overrides.append("+data.skip_existing_kind=folded")


class BoltzGenDesignFoldingAdapter(BoltzGenBaseAdapter):
    def checkpoint_path(self):
        ckpt = os.environ.get("BOLTZGEN_FOLDING_CHECKPOINT")
        if not ckpt:
            raise ValueError(
                "BOLTZGEN_FOLDING_CHECKPOINT environment variable must be set."
            )
        return ckpt

    def config_name(self):
        return "fold"

    def _add_step_specific_overrides(self, parsed_args, args, overrides):
        if parsed_args.output:
            overrides.append(f"data.design_dir={parsed_args.output}")
        overrides.append("+writer.designfolding=True")
        overrides.append("+data.cfg.return_designfolding=True")
        overrides.append(f"+data.skip_existing={parsed_args.reuse}")
        overrides.append("+data.skip_existing_kind=design_folded")


class BoltzGenAffinityAdapter(BoltzGenBaseAdapter):
    def checkpoint_path(self):
        ckpt = os.environ.get("BOLTZGEN_AFFINITY_CHECKPOINT")
        if not ckpt:
            raise ValueError(
                "BOLTZGEN_AFFINITY_CHECKPOINT environment variable must be set."
            )
        return ckpt

    def config_name(self):
        return "affinity"

    def _add_step_specific_overrides(self, parsed_args, args, overrides):
        if parsed_args.output:
            overrides.append(f"data.design_dir={parsed_args.output}")
        overrides.append(f"+data.skip_existing={parsed_args.reuse}")
        overrides.append("+data.skip_existing_kind=affinity")


class BoltzGenAnalysisAdapter(BoltzGenBaseAdapter):
    def checkpoint_path(self):
        ckpt = os.environ.get("BOLTZGEN_FOLDING_CHECKPOINT")
        if not ckpt:
            ckpt = os.environ.get("BOLTZGEN_DESIGN_CHECKPOINT")
        if not ckpt:
            raise ValueError(
                "BOLTZGEN_FOLDING_CHECKPOINT or BOLTZGEN_DESIGN_CHECKPOINT must be set for analysis."
            )
        return ckpt

    def config_name(self):
        return "analysis"

    def prepare_one(self, request, prepare_ctx):
        overrides = super().prepare_one(request, prepare_ctx)

        new_overrides = []
        design_dir_val = None

        for o in overrides:
            if o.startswith(("output=", "+output=")):
                design_dir_val = o.split("=", 1)[1]
            elif o.startswith("data.design_dir="):
                pass
            else:
                new_overrides.append(o)

        if design_dir_val:
            new_overrides.append(f"design_dir={design_dir_val}")

        return new_overrides

    def _add_step_specific_overrides(self, parsed_args, args, overrides):
        do_design_folding = parsed_args.protocol in [
            "protein-anything",
            "protein-small_molecule",
        ]
        overrides.append(f"+designfolding_metrics={do_design_folding}")
        overrides.append(f"delta_sasa_original={parsed_args.skip_inverse_folding}")
        overrides.append(f"noncovalents_original={parsed_args.skip_inverse_folding}")
        overrides.append(f"+allatom_fold_metrics={parsed_args.skip_inverse_folding}")
        overrides.append(f"+data.skip_existing={parsed_args.reuse}")
        overrides.append("+data.skip_existing_kind=analyzed")
        protocol_analysis = PROTOCOL_CONFIGS.get(parsed_args.protocol, {}).get(
            "analysis", []
        )
        overrides.extend(protocol_analysis)

    def max_inflight_batches(self) -> int:
        return int(os.environ.get("MAX_INFLIGHT_BATCHES", "1"))

    def init_execute(self):
        pass

    def execute_batch(
        self, prepared, bucket_id, params, execute_ctx, *, cancelled=None
    ):
        results = []
        for overrides in prepared:
            try:
                design_dir_val = None
                for o in overrides:
                    if o.startswith("design_dir="):
                        design_dir_val = o.split("=", 1)[1]
                        break

                if design_dir_val:
                    dd = Path(design_dir_val)
                    if dd.exists():
                        refold_design_cif = dd / "refold_design_cif"
                        fold_out_design_npz = dd / "fold_out_design_npz"
                        refold_cif = dd / "refold_cif"
                        fold_out_npz = dd / "fold_out_npz"

                        intermediate_dirs = [
                            dd / "intermediate_designs_inverse_folded",
                            dd / "intermediate_designs",
                            dd / "designs",
                        ]

                        found_dir = None


                        if (
                            refold_design_cif.exists() and fold_out_design_npz.exists()
                        ) or (refold_cif.exists() and fold_out_npz.exists()):
                            pass

                        else:
                            for int_dir in intermediate_dirs:
                                if int_dir.exists():
                                    refold_design_cif_int = (
                                        int_dir / "refold_design_cif"
                                    )
                                    fold_out_design_npz_int = (
                                        int_dir / "fold_out_design_npz"
                                    )
                                    refold_cif_int = int_dir / "refold_cif"
                                    fold_out_npz_int = int_dir / "fold_out_npz"

                                    if (
                                        refold_design_cif_int.exists()
                                        and fold_out_design_npz_int.exists()
                                    ) or (
                                        refold_cif_int.exists()
                                        and fold_out_npz_int.exists()
                                    ):
                                        found_dir = int_dir
                                        break

                        if found_dir:
                            _LOG.info(
                                f"Auto-detected analysis directory: design_dir={found_dir}"
                            )
                            overrides = [
                                o for o in overrides if not o.startswith("design_dir=")
                            ]
                            overrides.append(f"design_dir={found_dir}")

                config_dir = os.environ.get("BOLTZGEN_CONFIG_DIR")
                if not config_dir:
                    raise ValueError(
                        "BOLTZGEN_CONFIG_DIR environment variable must be set."
                    )

                moldir = os.environ.get("BOLTZGEN_MOLDIR")
                if moldir and os.path.exists(moldir):
                    overrides.append(f"data.cfg.moldir={moldir}")

                with hydra.initialize_config_dir(
                    config_dir=config_dir, version_base=None
                ):
                    cfg = hydra.compose(
                        config_name=self.config_name(), overrides=overrides
                    )

                    _LOG.info(
                        f"Instantiating Analysis task with overrides: {overrides}"
                    )
                    task = hydra.utils.instantiate(cfg)

                    _LOG.info("Running Analysis task")
                    task.run()
                    results.append("Analysis completed successfully")
            except Exception as e:
                _LOG.error(f"Error during Analysis execution: {e}")
                import traceback

                traceback.print_exc()
                results.append(f"Error: {e}")
                raise e
        return results


class BoltzGenFilteringAdapter(BoltzGenBaseAdapter):
    def checkpoint_path(self):
        ckpt = os.environ.get("BOLTZGEN_FOLDING_CHECKPOINT")
        if not ckpt:
            raise ValueError(
                "BOLTZGEN_FOLDING_CHECKPOINT environment variable must be set."
            )
        return ckpt

    def config_name(self):
        return "filtering"

    def prepare_one(self, request, prepare_ctx):
        overrides = super().prepare_one(request, prepare_ctx)

        new_overrides = []
        design_dir_val = None

        for o in overrides:
            if o.startswith(("output=", "+output=")):
                design_dir_val = o.split("=", 1)[1]
            elif o.startswith("data.design_dir="):
                pass
            else:
                new_overrides.append(o)

        if design_dir_val:
            new_overrides.append(f"+design_dir={design_dir_val}")
            new_overrides.append(f"+outdir={Path(design_dir_val).parent}")

        return new_overrides

    def _add_step_specific_overrides(self, parsed_args, args, overrides):
        overrides.append(f"from_inverse_folded={not parsed_args.skip_inverse_folding}")
        do_design_folding = parsed_args.protocol in [
            "protein-anything",
            "protein-small_molecule",
        ]
        overrides.append(f"+filter_designfolding={do_design_folding}")
        use_affinity = parsed_args.protocol in ["protein-small_molecule"]
        overrides.append(f"use_affinity={use_affinity}")
        overrides.append(f"budget={parsed_args.budget}")
        protocol_filtering = PROTOCOL_CONFIGS.get(parsed_args.protocol, {}).get(
            "filtering", []
        )
        overrides.extend(protocol_filtering)

    def max_inflight_batches(self) -> int:
        return int(os.environ.get("MAX_INFLIGHT_BATCHES", "1"))

    def init_execute(self):
        pass

    def execute_batch(
        self, prepared, bucket_id, params, execute_ctx, *, cancelled=None
    ):
        results = []
        for overrides in prepared:
            try:
                config_dir = os.environ.get("BOLTZGEN_CONFIG_DIR")
                if not config_dir:
                    raise ValueError(
                        "BOLTZGEN_CONFIG_DIR environment variable must be set."
                    )

                with hydra.initialize_config_dir(
                    config_dir=config_dir, version_base=None
                ):
                    cfg = hydra.compose(
                        config_name=self.config_name(), overrides=overrides
                    )

                    if "data" in cfg:
                        with open_dict(cfg):
                            del cfg["data"]

                    _LOG.info(
                        f"Instantiating Filtering task with overrides: {overrides}"
                    )
                    task = hydra.utils.instantiate(cfg)

                    _LOG.info("Running Filtering task")
                    task.run(cfg)
                    results.append("Filtering completed successfully")
            except Exception as e:
                _LOG.error(f"Error during Filtering execution: {e}")
                import traceback

                traceback.print_exc()
                results.append(f"Error: {e}")
                raise e
        return results
