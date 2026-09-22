"""ESM adapter for the worker skeleton."""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
)

try:
    from Bio import SeqIO
except ImportError:
    SeqIO = None

from modelworker.model_adapter import ModelAdapter

_LOG = logging.getLogger(__name__)


class ESMAdapter(ModelAdapter):
    """Adapter for ESM-based protein variant generation."""

    def __init__(
        self,
        *,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = os.environ.get("ESM_MODEL_NAME", model_name)
        self._cache_dir = os.environ.get("ESM_CACHE_DIR", cache_dir)
        self._device = None
        self._log = logging.getLogger(__name__)

    def model_name(self) -> str:
        return "esm_variant_gen"

    def model_version(self) -> str:
        return self._model_name

    def max_batch_size(self) -> int:
        return int(os.environ.get("MAX_BATCH_SIZE", "1"))

    def init_execute(self) -> Any:
        """Load ESM model with hardware optimization (device_map='auto')."""
        self._log.info(f"Loading ESM model: {self._model_name}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, cache_dir=self._cache_dir
            )
            model = AutoModelForMaskedLM.from_pretrained(
                self._model_name,
                cache_dir=self._cache_dir,
                device_map="auto",
            )

            try:
                if any(d.startswith("cuda") for d in model.hf_device_map.values()):
                    self._device = torch.device(
                        "cuda:0"
                    )
                else:
                    self._device = torch.device("cpu")
            except Exception:
                self._device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )

            model.eval()
            self._log.info(f"ESM model loaded. Main input device: {self._device}")

            return {"model": model, "tokenizer": tokenizer, "device": self._device}

        except Exception as e:
            self._log.error(f"Failed to load ESM model: {e}")
            raise

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

    def _load_sequence_from_fasta(self, fasta_file):
        """Loads the first sequence from a FASTA file."""
        if SeqIO is None:
            logging.error(
                "Cannot load from FASTA file. Biopython library not installed."
            )
            logging.error("Install using: pip install biopython")
            return None, None

        try:
            with open(fasta_file) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    seq_str = str(record.seq).upper()
                    if not seq_str:
                        logging.warning(
                            f"Skipping empty sequence for record {record.id} in {fasta_file}"
                        )
                        continue
                    if all(c in "ACDEFGHIKLMNPQRSTVWYXBZJUO" for c in seq_str):
                        logging.info(
                            f"Loaded sequence ID: {record.id} (Length: {len(seq_str)}) from {fasta_file}"
                        )
                        return record.id, seq_str
                    else:
                        logging.warning(
                            f"Skipping record {record.id} due to non-standard characters in sequence."
                        )
                        return None, None
                logging.error(f"No valid protein sequences found in {fasta_file}")
                return None, None
        except FileNotFoundError:
            logging.error(f"Error: FASTA file not found at {fasta_file}")
            return None, None
        except Exception as e:
            logging.error(f"Error reading FASTA file {fasta_file}: {e}")
            return None, None

    def _from_nextflow_task(self, request: dict[str, Any]) -> dict[str, Any]:
        argv = request.get("argv") or []
        if not isinstance(argv, list):
            return request

        cli_args = self._parse_cli_args(argv)
        overrides = request.copy()

        if "ref_sequence" in cli_args:
            overrides["sequence"] = cli_args["ref_sequence"]
            overrides["reference_stem"] = "cli_sequence"
        elif "ref_fasta_file" in cli_args:
            fasta_path = Path(cli_args["ref_fasta_file"])
            ref_id, ref_seq = self._load_sequence_from_fasta(str(fasta_path))
            if ref_seq:
                overrides["sequence"] = ref_seq
                overrides["reference_stem"] = fasta_path.stem
                overrides["reference_name"] = ref_id or fasta_path.stem
            else:
                self._log.error(f"Could not load a valid sequence from {fasta_path}")
        if "output_dir" in cli_args:
            overrides["output_dir"] = cli_args["output_dir"]
        if "skip_existing" in cli_args:
            val = cli_args["skip_existing"]
            overrides["skip_existing"] = (
                val if isinstance(val, bool) else val.lower() == "true"
            )
        if "num_variants" in cli_args:
            overrides["num_variants"] = int(cli_args["num_variants"])
        if "max_mutations" in cli_args:
            overrides["max_mutations"] = int(cli_args["max_mutations"])
        if "top_k" in cli_args:
            overrides["top_k"] = int(cli_args["top_k"])
        if "batch_size" in cli_args:
            overrides["batch_size"] = int(cli_args["batch_size"])

        return overrides

    def _normalize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        req = dict(request)
        if "sequence" not in req:
            raise ValueError(
                "Request must contain 'sequence' (or --ref_sequence in argv)."
            )

        seq = req["sequence"].upper()
        if not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
            raise ValueError(
                "Invalid characters in sequence. Only standard amino acids allowed."
            )
        req["sequence"] = seq

        req.setdefault("num_variants", 10)
        req.setdefault("max_mutations", 1)
        req.setdefault("top_k", 5)
        req.setdefault("batch_size", 1)
        for key in ("num_variants", "max_mutations", "top_k", "batch_size"):
            value = int(req[key])
            if value <= 0:
                raise ValueError(f"{key} must be positive")
            req[key] = value

        return req

    def execute_batch(
        self,
        prepared: list[dict[str, Any]],
        bucket_id: str | None,
        params: dict[str, str],
        execute_ctx: Any,
    ) -> list[Any]:

        model = execute_ctx["model"]
        tokenizer = execute_ctx["tokenizer"]
        device = execute_ctx["device"]

        outputs = []
        for req in prepared:
            self._log.info(
                f"Processing ESM request for sequence len={len(req['sequence'])}"
            )

            out_dir_str = req.get("output_dir", "")
            out_dir = Path(out_dir_str).absolute()
            out_dir.mkdir(parents=True, exist_ok=True)

            ref_stem = req.get("reference_stem", "reference")

            try:
                variants = self._generate_mutations(
                    reference_sequence=req["sequence"],
                    model=model,
                    tokenizer=tokenizer,
                    num_total_variants=req["num_variants"],
                    max_mutations=req["max_mutations"],
                    top_k=req["top_k"],
                    input_device=device,
                    batch_size=req["batch_size"],
                )

                output_data = {
                    "reference": {
                        "name": req.get("reference_name", "ref"),
                        "sequence": req["sequence"],
                    },
                    "variants": variants,
                }
                self._publish_outputs(out_dir, ref_stem, output_data, variants)

                outputs.append({"variants": variants, "output_dir": str(out_dir)})

            except RuntimeError:
                raise
            except Exception as e:
                self._log.error(f"ESM generation failed: {e}")
                outputs.append({"error": str(e)})

        return outputs

    @staticmethod
    def _publish_outputs(
        out_dir: Path,
        ref_stem: str,
        output_data: dict[str, Any],
        variants: list[dict[str, Any]],
    ) -> None:
        """Replace one attempt's complete output set in a reused workdir."""
        with tempfile.TemporaryDirectory(prefix=".esm-publish-", dir=out_dir) as tmp:
            staged = Path(tmp)
            with open(staged / "variants.json", "w") as f:
                json.dump(output_data, f, indent=4)
            for idx, var in enumerate(variants):
                fasta_header = f"{ref_stem}_esm_{idx}"
                with open(staged / f"{fasta_header}.fasta", "w") as f:
                    f.write(f">{fasta_header}\n{var['sequence']}\n")

            for stale in out_dir.glob(f"{ref_stem}_esm_*.fasta"):
                stale.unlink()
            os.replace(staged / "variants.json", out_dir / "variants.json")
            for fasta in staged.glob(f"{ref_stem}_esm_*.fasta"):
                os.replace(fasta, out_dir / fasta.name)

    def finalize_one(self, output: Any, finalize_ctx: Any) -> Any:
        if "error" in output:
            raise RuntimeError(f"ESM execution failed: {output['error']}")
        return output

    def _perform_one_mutation(
        self,
        current_sequence,
        model,
        tokenizer,
        top_k,
        input_device,
        mutated_positions,
        batch_size=1,
    ):
        """Helper to perform a single mutation step (Logic extracted from script)."""
        seq_len = len(current_sequence)
        possible_indices = list(set(range(seq_len)) - mutated_positions)
        max_attempts_per_step = min(len(possible_indices) * 3, 30)
        attempts = 0
        batch_size = max(1, int(batch_size)) if batch_size else 1

        while attempts < max_attempts_per_step and possible_indices:
            attempts += 1
            batch_indices = random.sample(
                possible_indices, k=min(batch_size, len(possible_indices))
            )

            masked_sequences = []
            original_residues = []
            for mut_idx in batch_indices:
                original_residue = current_sequence[mut_idx]
                masked_sequence = list(current_sequence)
                masked_sequence[mut_idx] = tokenizer.mask_token
                masked_sequences.append("".join(masked_sequence))
                original_residues.append(original_residue)

            try:
                inputs = tokenizer(
                    masked_sequences, return_tensors="pt", padding=True
                ).to(input_device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                mask_positions = (
                    inputs["input_ids"] == tokenizer.mask_token_id
                ).nonzero(as_tuple=False)
                mask_index_map = {}
                for row_idx, col_idx in mask_positions.tolist():
                    if row_idx not in mask_index_map:
                        mask_index_map[row_idx] = col_idx

                for idx_in_batch, mut_idx in enumerate(batch_indices):
                    mask_token_index = mask_index_map.get(idx_in_batch)
                    if mask_token_index is None:
                        possible_indices.remove(mut_idx)
                        continue

                    masked_token_logits = logits[idx_in_batch, mask_token_index, :]
                    probabilities = torch.softmax(masked_token_logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(
                        probabilities, top_k, dim=-1
                    )

                    top_k_tokens = [
                        tokenizer.decode(idx.item()) for idx in top_k_indices.cpu()
                    ]
                    valid_indices = [
                        i
                        for i, token in enumerate(top_k_tokens)
                        if len(token.strip()) == 1
                        and token.strip().isupper()
                        and token.strip() in "ACDEFGHIKLMNPQRSTVWY"
                    ]

                    if not valid_indices:
                        possible_indices.remove(mut_idx)
                        continue

                    filtered_probs = top_k_probs.cpu()[valid_indices]
                    filtered_tokens = [top_k_tokens[i] for i in valid_indices]

                    prob_sum = torch.sum(filtered_probs)
                    if prob_sum <= 0:
                        possible_indices.remove(mut_idx)
                        continue
                    normalized_probs = filtered_probs / prob_sum

                    sample_attempts = 0
                    max_sample_attempts = min(
                        len(set(filtered_tokens) - {original_residues[idx_in_batch]}), 5
                    )
                    while sample_attempts < max_sample_attempts:
                        sampled_token_index = torch.multinomial(
                            normalized_probs, 1
                        ).item()
                        new_residue = filtered_tokens[sampled_token_index]
                        if new_residue != original_residues[idx_in_batch]:
                            variant_sequence_list = list(current_sequence)
                            variant_sequence_list[mut_idx] = new_residue
                            new_sequence = "".join(variant_sequence_list)
                            return (
                                new_sequence,
                                mut_idx,
                                original_residues[idx_in_batch],
                                new_residue,
                            )
                        sample_attempts += 1

                    possible_indices.remove(mut_idx)

            except RuntimeError:
                raise
            except Exception as e:
                self._log.warning(f"Error during mutation prediction: {e}")
                for mut_idx in batch_indices:
                    if mut_idx in possible_indices:
                        possible_indices.remove(mut_idx)
                continue

        return None

    def _generate_mutations(
        self,
        reference_sequence,
        model,
        tokenizer,
        num_total_variants,
        max_mutations,
        top_k,
        input_device,
        batch_size,
    ) -> list[dict[str, Any]]:
        """Core generation logic adapted from script."""
        all_variants = []
        variants_generated_count = 0
        variants_per_level = defaultdict(int)

        base_count = num_total_variants // max_mutations
        remainder = num_total_variants % max_mutations
        for i in range(1, max_mutations + 1):
            variants_per_level[i] = base_count + (1 if i <= remainder else 0)

        total_attempts = 0
        max_total_attempts = num_total_variants * 10

        for num_mut in range(1, max_mutations + 1):
            target_count = variants_per_level[num_mut]
            if target_count == 0:
                continue

            generated_for_level = 0
            attempts_for_level = 0
            max_attempts_for_level = target_count * (5 + num_mut * 2)
            sequences_at_this_level = set()

            while (
                generated_for_level < target_count
                and attempts_for_level < max_attempts_for_level
                and total_attempts < max_total_attempts
            ):
                attempts_for_level += 1
                total_attempts += 1
                current_sequence = reference_sequence
                mutation_history = []
                mutated_positions = set()
                successful_variant = True

                for step in range(num_mut):
                    mutation_result = self._perform_one_mutation(
                        current_sequence,
                        model,
                        tokenizer,
                        top_k,
                        input_device,
                        mutated_positions,
                        batch_size,
                    )

                    if mutation_result:
                        new_sequence, pos, orig_res, new_res = mutation_result
                        if orig_res == new_res:
                            successful_variant = True
                            current_sequence = new_sequence
                            break

                        current_sequence = new_sequence
                        mutated_positions.add(pos)
                        mutation_history.append(
                            {
                                "step": step + 1,
                                "position": pos + 1,
                                "original_residue": orig_res,
                                "new_residue": new_res,
                            }
                        )
                    else:
                        successful_variant = False
                        break

                if (
                    successful_variant
                    and current_sequence not in sequences_at_this_level
                    and len(mutation_history) > 0
                ):
                    sequences_at_this_level.add(current_sequence)

                    pos_str = "_".join(
                        map(str, sorted([h["position"] for h in mutation_history]))
                    )
                    variant_name = f"variant_{variants_generated_count + 1}_muts{num_mut}_pos{pos_str}"

                    all_variants.append(
                        {
                            "name": variant_name,
                            "sequence": current_sequence,
                            "num_mutations": len(mutation_history),
                            "mutation_history": mutation_history,
                        }
                    )
                    generated_for_level += 1
                    variants_generated_count += 1

        return all_variants
