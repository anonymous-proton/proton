"""ProteinMPNN adapter for the worker skeleton."""

import copy
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from protein_mpnn_utils import (
        ProteinMPNN,
        StructureDatasetPDB,
        _S_to_seq,
        _scores,
        parse_PDB,
        tied_featurize,
    )
except ImportError:
    import sys
    from protein_mpnn_utils import (
        ProteinMPNN,
        StructureDatasetPDB,
        _S_to_seq,
        _scores,
        parse_PDB,
        tied_featurize,
    )

from modelworker.model_adapter import ModelAdapter
from modelworker.concurrency_utils import safe_model_copy

_LOG = logging.getLogger(__name__)


class ProteinMPNNAdapter(ModelAdapter):
    """Adapter for ProteinMPNN sequence design."""

    def __init__(
        self,
        *,
        model_name: str = "v_48_030",
        ca_only: bool = False,
        use_soluble_model: bool = False,
        path_to_model_weights: Optional[str] = None,
    ) -> None:
        self._model_name = os.environ.get("PROTEINMPNN_MODEL_NAME", model_name)
        self._ca_only = os.environ.get("PROTEINMPNN_CA_ONLY", str(ca_only)).lower() in ("true", "1", "yes")
        self._use_soluble = os.environ.get("PROTEINMPNN_USE_SOLUBLE", str(use_soluble_model)).lower() in ("true", "1", "yes")
        self._path_to_model_weights = os.environ.get("PROTEINMPNN_MODEL_DIR", path_to_model_weights)
        
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def model_name(self) -> str:
        return "proteinmpnn"

    def model_version(self) -> str:
        return self._model_name

    def max_batch_size(self) -> int:
        return int(os.environ.get("MAX_BATCH_SIZE", "1"))

    def init_execute(self) -> Any:
        """Load ProteinMPNN model weights."""
        _LOG.info(f"Initializing ProteinMPNN (Model: {self._model_name}, CA-Only: {self._ca_only})...")

        if self._path_to_model_weights:
            model_folder_path = self._path_to_model_weights
            if not model_folder_path.endswith("/"):
                model_folder_path = Path(model_folder_path + "/")
            if self._ca_only:
                model_folder_path = str(model_folder_path / "ca_model_weights") + "/"
            elif self._use_soluble:
                model_folder_path = str(model_folder_path / "soluble_model_weights") + "/"
            else:
                model_folder_path = str(model_folder_path / "vanilla_model_weights") + "/"
            
            if not os.path.exists(model_folder_path):
                 _LOG.warning(f"Weights path {model_folder_path} not found. Trying to find them via env vars or manual config.")
        else:
            repo_root = Path(__file__).resolve().parent
            
            if self._ca_only:
                model_folder_path = str(repo_root / "ca_model_weights") + "/"
            elif self._use_soluble:
                model_folder_path = str(repo_root / "soluble_model_weights") + "/"
            else:
                model_folder_path = str(repo_root / "vanilla_model_weights") + "/"
            
            if not os.path.exists(model_folder_path):
                 _LOG.warning(f"Weights path {model_folder_path} not found. Trying to find them via env vars or manual config.")

        checkpoint_path = os.path.join(model_folder_path, f"{self._model_name}.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model weights not found at: {checkpoint_path}")

        _LOG.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        
        hidden_dim = 128
        num_layers = 3
        
        model = ProteinMPNN(
            ca_only=self._ca_only,
            num_letters=21,
            node_features=hidden_dim,
            edge_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            augment_eps=0.0,
            k_neighbors=checkpoint['num_edges']
        )
        
        model.to(self._device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        _LOG.info("ProteinMPNN model loaded successfully.")
        
        return {
            "model": model,
            "device": self._device,
            "alphabet": 'ACDEFGHIKLMNPQRSTVWYX',
            "alphabet_dict": dict(zip('ACDEFGHIKLMNPQRSTVWYX', range(21)))
        }

    def prepare_one(self, request: Dict[str, Any], prepare_ctx: Any) -> Any:
        if request.get("argv"):
            request = self._from_nextflow_task(request)

        prepared = self._normalize_request(request)
        return prepared

    def _parse_cli_args(self, argv: List[str]) -> Dict[str, Any]:
        """Simple argparse parser."""
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
            elif i + 1 < len(argv) and not argv[i+1].startswith("-"):
                val = argv[i+1]
                i += 2
            else:
                val = True
                i += 1
            parsed[key] = val
        return parsed

    def _from_nextflow_task(self, request: Dict[str, Any]) -> Dict[str, Any]:
        argv = request.get("argv") or []
        if not isinstance(argv, list):
            return request

        cli_args = self._parse_cli_args(argv)
        overrides = request.copy()

        if "pdb_path" in cli_args: overrides["pdb_path"] = cli_args["pdb_path"]
        if "out_folder" in cli_args: overrides["out_folder"] = cli_args["out_folder"]
        if "jsonl_path" in cli_args: overrides["jsonl_path"] = cli_args["jsonl_path"]
        if "chain_id_jsonl" in cli_args: overrides["chain_id_jsonl"] = cli_args["chain_id_jsonl"]
        if "fixed_positions_jsonl" in cli_args: overrides["fixed_positions_jsonl"] = cli_args["fixed_positions_jsonl"]
        if "omit_AA_jsonl" in cli_args: overrides["omit_AA_jsonl"] = cli_args["omit_AA_jsonl"]
        if "bias_AA_jsonl" in cli_args: overrides["bias_AA_jsonl"] = cli_args["bias_AA_jsonl"]
        if "tied_positions_jsonl" in cli_args: overrides["tied_positions_jsonl"] = cli_args["tied_positions_jsonl"]
        if "pssm_jsonl" in cli_args: overrides["pssm_jsonl"] = cli_args["pssm_jsonl"]
        if "bias_by_res_jsonl" in cli_args: overrides["bias_by_res_jsonl"] = cli_args["bias_by_res_jsonl"]
        if "path_to_fasta" in cli_args: overrides["path_to_fasta"] = cli_args["path_to_fasta"]

        if "pdb_path_chains" in cli_args: overrides["chains_to_design"] = cli_args["pdb_path_chains"]
        if "omit_AAs" in cli_args: overrides["omit_AAs"] = cli_args["omit_AAs"]
        
        if "num_seq_per_target" in cli_args: overrides["num_seq_per_target"] = int(cli_args["num_seq_per_target"])
        if "batch_size" in cli_args: overrides["batch_size"] = int(cli_args["batch_size"])
        if "max_length" in cli_args: overrides["max_length"] = int(cli_args["max_length"])
        if "sampling_temp" in cli_args: overrides["sampling_temp"] = cli_args["sampling_temp"]
        if "seed" in cli_args: overrides["seed"] = int(cli_args["seed"])
        if "backbone_noise" in cli_args: overrides["backbone_noise"] = float(cli_args["backbone_noise"])
        if "pssm_multi" in cli_args: overrides["pssm_multi"] = float(cli_args["pssm_multi"])
        if "pssm_threshold" in cli_args: overrides["pssm_threshold"] = float(cli_args["pssm_threshold"])

        for flag in ["save_score", "save_probs", "score_only", "conditional_probs_only", "conditional_probs_only_backbone",
                     "unconditional_probs_only", "pssm_log_odds_flag", "pssm_bias_flag"]:
            if flag in cli_args:
                val = cli_args[flag]
                overrides[flag] = bool(int(val)) if isinstance(val, (int, str)) and str(val).isdigit() else bool(val)

        return overrides

    def _normalize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req = dict(request)
        
        if "pdb_content" not in req and "pdb_path" not in req:
            raise ValueError("Request must contain 'pdb_content' string or 'pdb_path'.")
            
        req.setdefault("num_seq_per_target", 1)
        req.setdefault("sampling_temp", "0.1")
        req.setdefault("batch_size", 1)
        req.setdefault("seed", 0)
        req.setdefault("backbone_noise", 0.0)
        req.setdefault("max_length", 200000)
        req.setdefault("out_folder", None)
        
        req.setdefault("save_score", 0)
        req.setdefault("save_probs", 0)
        req.setdefault("score_only", 0)
        req.setdefault("conditional_probs_only", 0)
        req.setdefault("unconditional_probs_only", 0)
        
        req.setdefault("pssm_multi", 0.0)
        req.setdefault("pssm_threshold", 0.0)
        req.setdefault("pssm_log_odds_flag", 0)
        req.setdefault("pssm_bias_flag", 0)
        
        req.setdefault("omit_AAs", "X")
        
        return req

    def _load_jsonl(self, path: Optional[str]) -> Optional[Dict]:
        if not path or not os.path.isfile(path):
            return None
        result = {}
        with open(path, 'r') as f:
            for line in f:
                result.update(json.loads(line))
        return result

    def execute_batch(
        self,
        prepared: List[Dict[str, Any]],
        bucket_id: Optional[str],
        params: Dict[str, str],
        execute_ctx: Any,
    ) -> List[Any]:
        
        model = execute_ctx["model"]
        device = execute_ctx["device"]
        alphabet = execute_ctx["alphabet"]
        outputs = []

        for req in prepared:
            import copy
            memo = {id(model): model}
            local_model = copy.deepcopy(model, memo)
            local_model.augment_eps = req.get("backbone_noise", 0.0)
            
            seed = int(req["seed"])
            if seed == 0: seed = int(np.random.randint(0, 1000))
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            out_folder = req.get("out_folder")
            if out_folder:
                os.makedirs(os.path.join(out_folder, 'seqs'), exist_ok=True)
                if req["save_score"]: os.makedirs(os.path.join(out_folder, 'scores'), exist_ok=True)
                if req["save_probs"]: os.makedirs(os.path.join(out_folder, 'probs'), exist_ok=True)

            with tempfile.TemporaryDirectory() as tmp_dir:
                pdb_path = req.get("pdb_path")
                if not pdb_path:
                    if "pdb_content" in req:
                        pdb_path = os.path.join(tmp_dir, "input.pdb")
                        with open(pdb_path, "w") as f: f.write(req["pdb_content"])
                    else:
                        raise ValueError("Request contains neither 'pdb_path' nor 'pdb_content'")
                
                parsed_dicts = parse_PDB(pdb_path, ca_only=self._ca_only)
                pdb_dict = parsed_dicts[0]
                pdb_dict["name"] = req.get("name", pdb_dict.get("name", "query"))
                name_ = pdb_dict["name"]

                all_chain_list = [item[-1:] for item in list(pdb_dict) if item[:9]=='seq_chain']

                def _get_cfg(k):
                    v = req.get(k)
                    return v if isinstance(v, dict) else self._load_jsonl(v)

                chain_id_jsonl_data = _get_cfg("chain_id_jsonl")
                if chain_id_jsonl_data:
                    chain_id_dict = chain_id_jsonl_data
                else:
                    designed_chain_list = req.get("chains_to_design", all_chain_list)
                    if isinstance(designed_chain_list, str): designed_chain_list = designed_chain_list.split()
                    fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
                    chain_id_dict = {name_: (designed_chain_list, fixed_chain_list)}

                fixed_positions_dict = _get_cfg("fixed_positions_jsonl")
                omit_AA_dict = _get_cfg("omit_AA_jsonl")
                bias_AA_dict = _get_cfg("bias_AA_jsonl")
                pssm_dict = _get_cfg("pssm_jsonl")
                bias_by_res_dict = _get_cfg("bias_by_res_jsonl")
                tied_positions_dict = _get_cfg("tied_positions_jsonl")

                bias_AAs_np = np.zeros(len(alphabet))
                if bias_AA_dict:
                    for n, AA in enumerate(alphabet):
                        if AA in bias_AA_dict: bias_AAs_np[n] = bias_AA_dict[AA]

                omit_AAs_list = req.get("omit_AAs", "X")
                omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

                temp_in = req["sampling_temp"]
                temperatures = [float(x) for x in temp_in.split()] if isinstance(temp_in, str) else ([temp_in] if not isinstance(temp_in, list) else temp_in)

                num_seq_per_target = req["num_seq_per_target"]
                batch_size = req["batch_size"]
                total_sequences = max(1, num_seq_per_target) * batch_size
                batch_schedule = []
                remaining = total_sequences
                while remaining > 0:
                    batch_schedule.append(min(batch_size, remaining))
                    remaining -= batch_size

                BATCH_COPIES = max(batch_schedule) if batch_schedule else batch_size

                generated_results = []
                score_acc, global_score_acc, probs_acc, log_probs_acc, S_acc, mask_acc = [], [], [], [], [], []
                fasta_chunks: List[str] = []

                with torch.no_grad():
                    batch_clones = [copy.deepcopy(pdb_dict) for _ in range(BATCH_COPIES)]
                    (X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, 
                     visible_list_list, masked_list_list, masked_chain_length_list_list, 
                     chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, 
                     tied_pos_list_of_lists_list, pssm_coef, pssm_bias, 
                     pssm_log_odds_all, bias_by_res_all, tied_beta) = tied_featurize(
                        batch_clones, device, chain_id_dict, fixed_positions_dict, 
                        omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, 
                        ca_only=self._ca_only
                    )
                    pssm_log_odds_mask = (pssm_log_odds_all > req.get("pssm_threshold", 0.0)).float()

                    for temp in temperatures:
                        for j, current_batch_size in enumerate(batch_schedule):
                            randn_2 = torch.randn(chain_M.shape, device=device)
                            
                            if tied_positions_dict is None:
                                sample_dict = local_model.sample(
                                    X, randn_2, S, chain_M, chain_encoding_all, residue_idx, 
                                    mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, 
                                    bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, 
                                    omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, 
                                    pssm_bias=pssm_bias, pssm_multi=req.get("pssm_multi", 0.0), 
                                    pssm_log_odds_flag=bool(req.get("pssm_log_odds_flag", 0)), 
                                    pssm_log_odds_mask=pssm_log_odds_mask, 
                                    pssm_bias_flag=bool(req.get("pssm_bias_flag", 0)), 
                                    bias_by_res=bias_by_res_all
                                )
                            else:
                                sample_dict = local_model.tied_sample(
                                    X, randn_2, S, chain_M, chain_encoding_all, residue_idx, 
                                    mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, 
                                    bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, 
                                    omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, 
                                    pssm_bias=pssm_bias, pssm_multi=req.get("pssm_multi", 0.0), 
                                    pssm_log_odds_flag=bool(req.get("pssm_log_odds_flag", 0)), 
                                    pssm_log_odds_mask=pssm_log_odds_mask, 
                                    pssm_bias_flag=bool(req.get("pssm_bias_flag", 0)), 
                                    tied_pos=tied_pos_list_of_lists_list[0], 
                                    tied_beta=tied_beta, bias_by_res=bias_by_res_all
                                )
                            
                            S_sample = sample_dict["S"]
                            
                            log_probs = local_model(
                                X, S_sample, mask, chain_M*chain_M_pos, residue_idx, 
                                chain_encoding_all, randn_2, use_input_decoding_order=True, 
                                decoding_order=sample_dict["decoding_order"]
                            )
                            
                            mask_for_loss = mask*chain_M*chain_M_pos
                            scores = _scores(S_sample, log_probs, mask_for_loss)
                            scores_batch = scores.cpu().data.numpy()[:current_batch_size]
                            
                            global_scores = _scores(S_sample, log_probs, mask)
                            global_scores_batch = global_scores.cpu().data.numpy()[:current_batch_size]

                            if out_folder:
                                score_acc.append(scores_batch)
                                global_score_acc.append(global_scores_batch)
                                probs_acc.append(sample_dict["probs"][:current_batch_size].cpu().data.numpy())
                                log_probs_acc.append(log_probs[:current_batch_size].cpu().data.numpy())
                                S_acc.append(S_sample[:current_batch_size].cpu().data.numpy())
                                mask_acc.append(mask_for_loss[:current_batch_size].cpu().data.numpy())

                            for b_ix in range(current_batch_size):
                                seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                                recovery = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21), -1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                                
                                res_item = {
                                    "sequence": seq, "score": float(scores_batch[b_ix]), "global_score": float(global_scores_batch[b_ix]),
                                    "recovery": float(recovery.cpu().numpy()), "temperature": temp, "seed": seed
                                }
                                generated_results.append(res_item)
                                
                                if out_folder:
                                    masked_chain_length_list = masked_chain_length_list_list[b_ix]
                                    masked_list = masked_list_list[b_ix]
                                    
                                    def _format_seq(s_str):
                                        start, end, parts = 0, 0, []
                                        for m_l in masked_chain_length_list:
                                            end += m_l
                                            parts.append(s_str[start:end])
                                            start = end
                                        res_s = "".join(list(np.array(parts)[np.argsort(masked_list)]))
                                        l_idx = 0
                                        for mc_l in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                            l_idx += mc_l
                                            res_s = res_s[:l_idx] + '/' + res_s[l_idx:]
                                            l_idx += 1
                                        return res_s

                                    ali_file = os.path.join(out_folder, 'seqs', f"{name_}.fa")
                                    _header = ""
                                    if not fasta_chunks:
                                        native_seq = _format_seq(_S_to_seq(S[b_ix], chain_M[b_ix]))
                                        _header = f">{name_}, score={scores_batch[b_ix]:.4f}, fixed_chains={visible_list_list[b_ix]}, designed_chains={masked_list_list[b_ix]}, seed={seed}\n{native_seq}\n"
                                    _body = f">T={temp}, sample={len(generated_results)}, score={scores_batch[b_ix]:.4f}, global_score={global_scores_batch[b_ix]:.4f}, seq_recovery={res_item['recovery']:.4f}\n{_format_seq(seq)}\n"
                                    fasta_chunks.append(_header + _body)

                if out_folder:
                    if fasta_chunks:
                        import uuid as _uuid
                        ali_file = os.path.join(out_folder, 'seqs', f"{name_}.fa")
                        _tmp = ali_file + f".tmp_{_uuid.uuid4().hex[:8]}"
                        with open(_tmp, 'w') as f:
                            f.write("".join(fasta_chunks))
                        os.replace(_tmp, ali_file)
                    if req["save_score"] and score_acc:
                        np.savez(os.path.join(out_folder, 'scores', f'{name_}.npz'), score=np.concatenate(score_acc, 0), global_score=np.concatenate(global_score_acc, 0))
                    if req["save_probs"] and probs_acc:
                        np.savez(os.path.join(out_folder, 'probs', f'{name_}.npz'), probs=np.concatenate(probs_acc, 0), log_probs=np.concatenate(log_probs_acc, 0), S=np.concatenate(S_acc, 0), mask=np.concatenate(mask_acc, 0))

            outputs.append(generated_results)
        return outputs

    def finalize_one(self, output: Any, finalize_ctx: Any) -> Any:
        return {"sequences": output}
