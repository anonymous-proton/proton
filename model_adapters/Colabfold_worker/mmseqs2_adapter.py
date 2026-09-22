"""MMSeqs2 adapter for the worker skeleton."""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from modelworker.model_adapter import ModelAdapter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "third_parties" / "Colabfold") not in sys.path:
    sys.path.append(str(_REPO_ROOT / "third_parties" / "Colabfold"))

try:
    from colabfold.mmseqs.search import (
        mmseqs_search_monomer,
        mmseqs_search_pair,
    )
    from colabfold.input import (
        get_queries,
        msa_to_str,
        safe_filename,
    )
except ImportError:
    logging.getLogger(__name__).warning(
        "Could not import colabfold modules. Make sure third_parties/Colabfold is in PYTHONPATH."
    )
    mmseqs_search_monomer = None
    mmseqs_search_pair = None
    get_queries = None
    msa_to_str = None
    safe_filename = None

_LOG = logging.getLogger(__name__)


class MMSeqs2Adapter(ModelAdapter):
    """Spawn-safe ColabFold MMseqs2 CLI adapter."""

    def __init__(
        self,
        *,
        mmseqs_bin: Optional[str] = None,
        db_base: Optional[str] = None,
    ) -> None:
        if not mmseqs_bin:
            mmseqs_bin = os.environ.get("MMSEQS_BIN", "mmseqs")
        if not db_base:
            db_base = os.environ.get("MMSEQS_DB_DIR", "")

        self._mmseqs_bin = Path(mmseqs_bin)
        self._db_base = Path(db_base) if db_base else None

        self._log = logging.getLogger(__name__)

    def model_name(self) -> str:
        return "mmseqs2"

    def model_version(self) -> str:
        return "colabfold_search_v3_fix_bool"

    def concurrency_safety_level(self) -> str:
        return "full"

    def init_execute(self) -> Any:
        self._log.info("Initializing MMSeqs2 Adapter...")

        if not self._db_base or not self._db_base.exists():
            self._log.warning(
                f"MMSeqs2 DB directory not found or not set: {self._db_base}"
            )

        try:
            subprocess.check_call(
                [str(self._mmseqs_bin), "version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self._log.error(f"Failed to run mmseqs binary at {self._mmseqs_bin}: {e}")
            raise

        return "READY"

    def prepare_one(self, request: Dict[str, Any], prepare_ctx: Any) -> Any:
        if request.get("argv"):
            request = self._from_nextflow_task(request)

        return self._normalize_request(request)

    def _from_nextflow_task(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses CLI arguments robustly.
        """
        argv = request.get("argv") or []
        if not isinstance(argv, list) or not argv:
            return request

        overrides = request.copy()

        start_idx = 0
        if len(argv) > 0 and (
            "colabfold_search" in str(argv[0]) or "python" in str(argv[0])
        ):
            start_idx = 1

        args = argv[start_idx:]

        i = 0
        positional_keys = ["fasta_path", "db_base", "output_dir"]
        pos_idx = 0

        while i < len(args):
            arg = str(args[i])

            if arg.startswith("-"):
                key = arg.lstrip("-").replace("-", "_")

                if key in ["keep_duplicates", "help", "h"]:
                    overrides[key] = True
                    i += 1
                    continue

                if i + 1 < len(args):
                    val = args[i + 1]
                    if str(val).lower() in ["true", "yes"]:
                        val = True
                    elif str(val).lower() in ["false", "no"]:
                        val = False
                    else:
                        try:
                            if "." in str(val):
                                val = float(val)
                            else:
                                val = int(val)
                        except ValueError:
                            val = str(val)

                    overrides[key] = val
                    i += 2
                else:
                    overrides[key] = True
                    i += 1
            else:
                if pos_idx < len(positional_keys):
                    overrides[positional_keys[pos_idx]] = arg
                    pos_idx += 1
                i += 1

        return overrides

    def _normalize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req = dict(request)

        if "fasta_path" not in req and "sequence" not in req:
            self._log.warning("Request missing 'fasta_path' or 'sequence'")

        raw_filter = req.get("filter", True)
        is_filter = bool(raw_filter) and (str(raw_filter) != "0")
        req["filter"] = is_filter

        if is_filter:
            req.setdefault("qsc", 0.8)
            req.setdefault("max_accept", 100000)
        else:
            req.setdefault("qsc", -20.0)
            req.setdefault("max_accept", 1000000)

        req.setdefault("use_env", True)
        req.setdefault("use_templates", False)
        req.setdefault("use_env_pairing", False)
        req.setdefault("db_load_mode", 2)
        req.setdefault("threads", 64)
        req.setdefault("prefilter_mode", 0)
        req.setdefault("expand_eval", math.inf)
        req.setdefault("align_eval", 10)
        req.setdefault("search_eval", 0.1)
        req.setdefault("diff", 3000)
        req.setdefault("num_iterations", 3)
        req.setdefault("pairing_strategy", 0)
        req.setdefault("unpack", True)

        if "gpu" not in req:
            req["gpu"] = 1

        return req

    def execute_batch(
        self,
        prepared: List[Dict[str, Any]],
        bucket_id: Optional[str],
        params: Dict[str, str],
        execute_ctx: Any,
        *,
        cancelled=None,
    ) -> List[Any]:
        if any(
            dependency is None
            for dependency in (
                mmseqs_search_monomer,
                mmseqs_search_pair,
                get_queries,
                msa_to_str,
                safe_filename,
            )
        ):
            raise RuntimeError("ColabFold MMseqs2 dependencies are unavailable")
        assert get_queries is not None
        assert msa_to_str is not None
        assert safe_filename is not None
        assert mmseqs_search_monomer is not None
        assert mmseqs_search_pair is not None
        outputs = []

        for idx, req in enumerate(prepared):
            self._log.info(f"Processing request {idx + 1}/{len(prepared)}")

            fasta_path = req.get("fasta_path")
            sequence = req.get("sequence")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                output_dir = Path(req.get("output_dir", tmp_path / "out"))
                output_dir.mkdir(parents=True, exist_ok=True)

                db_base = Path(req.get("db_base", self._db_base))

                query_input = None
                if fasta_path:
                    p = Path(fasta_path)
                    if p.exists():
                        query_input = p
                elif sequence:
                    query_file = tmp_path / "input.fasta"
                    with open(query_file, "w") as f:
                        f.write(f">query\n{sequence}\n")
                    query_input = query_file

                if not query_input:
                    outputs.append({"error": "No input sequence provided"})
                    continue

                try:
                    queries, is_complex = get_queries(query_input, None)

                    keep_duplicates = req.get("keep_duplicates", False)
                    queries_unique = []

                    for job_number, (
                        raw_jobname,
                        query_sequences,
                        a3m_lines,
                    ) in enumerate(queries):
                        query_sequences = (
                            [query_sequences]
                            if isinstance(query_sequences, str)
                            else query_sequences
                        )
                        if keep_duplicates:
                            query_seqs_unique = list(query_sequences)
                            query_seqs_cardinality = [1] * len(query_sequences)
                        else:
                            query_seqs_unique = []
                            for x in query_sequences:
                                if x not in query_seqs_unique:
                                    query_seqs_unique.append(x)
                            query_seqs_cardinality = [0] * len(query_seqs_unique)
                            for seq in query_sequences:
                                seq_idx = query_seqs_unique.index(seq)
                                query_seqs_cardinality[seq_idx] += 1
                        queries_unique.append(
                            [raw_jobname, query_seqs_unique, query_seqs_cardinality]
                        )

                    formal_query_file = output_dir / "query.fas"
                    with formal_query_file.open("w") as f:
                        for _, (raw_jobname, query_sequences, _) in enumerate(
                            queries_unique
                        ):
                            for j, seq in enumerate(query_sequences):
                                f.write(f">{101 + j}\n{seq}\n")

                    subprocess.check_call(
                        [
                            str(self._mmseqs_bin),
                            "createdb",
                            str(formal_query_file),
                            str(output_dir / "qdb"),
                            "--shuffle",
                            "0",
                        ]
                    )

                    with (output_dir / "qdb.lookup").open("w") as f:
                        id_counter = 0
                        file_number = 0
                        for _, (raw_jobname, query_sequences, _) in enumerate(
                            queries_unique
                        ):
                            for seq in query_sequences:
                                raw_jobname_first = raw_jobname.split()[0]
                                f.write(
                                    f"{id_counter}\t{raw_jobname_first}\t{file_number}\n"
                                )
                                id_counter += 1
                            file_number += 1

                    db1 = Path(req.get("db1", "uniref30_2302_db"))
                    db2 = Path(req.get("db2", ""))
                    db3 = Path(req.get("db3", "colabfold_envdb_202108_db"))
                    db4 = Path(req.get("db4", "spire_ctg10_2401_db"))

                    arg_use_env = int(bool(req.get("use_env")))
                    arg_use_templates = int(bool(req.get("use_templates")))
                    arg_filter = int(bool(req.get("filter")))
                    arg_unpack = int(bool(req.get("unpack")))
                    arg_gpu = int(req.get("gpu", 1))

                    common_args = {
                        "dbbase": db_base,
                        "base": output_dir,
                        "uniref_db": db1,
                        "mmseqs": self._mmseqs_bin,
                        "prefilter_mode": int(req.get("prefilter_mode", 0)),
                        "s": float(req.get("s", 8)),
                        "db_load_mode": int(req.get("db_load_mode", 2)),
                        "threads": int(req.get("threads", 8)),
                        "gpu": arg_gpu,
                        "gpu_server": int(req.get("gpu_server", 0)),
                        "split": int(req.get("split", 0)),
                        "unpack": arg_unpack,
                        "search_eval": float(req.get("search_eval", 0.1)),
                        "num_iterations": int(req.get("num_iterations", 3)),
                        "search_type": req.get("search_type", None),
                        "min_aln_len": req.get("min_aln_len", None),
                    }

                    mmseqs_search_monomer(
                        template_db=db2,
                        metagenomic_db=db3,
                        use_env=arg_use_env,
                        use_templates=arg_use_templates,
                        filter=arg_filter,
                        expand_eval=float(req.get("expand_eval", math.inf)),
                        align_eval=int(req.get("align_eval", 10)),
                        diff=int(req.get("diff", 3000)),
                        qsc=float(req.get("qsc", 0.8)),
                        max_accept=int(req.get("max_accept", 100000)),
                        **common_args,
                    )

                    do_pairing = is_complex
                    if "is_complex" in req:
                        do_pairing = req["is_complex"]

                    if do_pairing:
                        mmseqs_search_pair(
                            pairing_strategy=int(req.get("pairing_strategy", 0)),
                            pair_env=False,
                            **common_args,
                        )
                        if req.get("use_env_pairing", False):
                            mmseqs_search_pair(
                                spire_db=db4,
                                pair_env=True,
                                pairing_strategy=int(req.get("pairing_strategy", 0)),
                                **common_args,
                            )

                    if arg_unpack:
                        search_id = 0
                        for job_number, (
                            raw_jobname,
                            query_sequences,
                            query_seqs_cardinality,
                        ) in enumerate(queries_unique):
                            unpaired_msa = []
                            paired_msa = None
                            if len(query_seqs_cardinality) > 1:
                                paired_msa = []

                            for seq in query_sequences:
                                a3m_p = output_dir / f"{search_id}.a3m"
                                if a3m_p.exists():
                                    with a3m_p.open("r") as f:
                                        unpaired_msa.append(f.read())
                                    a3m_p.unlink()

                                if req.get("use_env_pairing", False):
                                    env_pair_p = (
                                        output_dir / f"{search_id}.env.paired.a3m"
                                    )
                                    pair_p = output_dir / f"{search_id}.paired.a3m"
                                    if env_pair_p.exists():
                                        with open(pair_p, "a") as file_pair:
                                            with open(env_pair_p, "r") as file_pair_env:
                                                while chunk := file_pair_env.read(
                                                    10 * 1024 * 1024
                                                ):
                                                    file_pair.write(chunk)
                                        env_pair_p.unlink()

                                if len(query_seqs_cardinality) > 1:
                                    pair_p = output_dir / f"{search_id}.paired.a3m"
                                    if pair_p.exists():
                                        with pair_p.open("r") as f:
                                            paired_msa.append(f.read())
                                        pair_p.unlink()

                                pair_p_cleanup = output_dir / f"{search_id}.paired.a3m"
                                if pair_p_cleanup.exists():
                                    pair_p_cleanup.unlink()

                                search_id += 1

                            msa = msa_to_str(
                                unpaired_msa,
                                paired_msa,
                                query_sequences,
                                query_seqs_cardinality,
                            )
                            (output_dir / f"{job_number}.a3m").write_text(msa)

                        for job_number, (raw_jobname, _, _) in enumerate(
                            queries_unique
                        ):
                            safe_name = safe_filename(raw_jobname)
                            src = output_dir / f"{job_number}.a3m"
                            dst = output_dir / f"{safe_name}.a3m"
                            if src.exists():
                                if dst.exists():
                                    dst.unlink()
                                os.rename(src, dst)

                        for db_to_rm in ["qdb", "qdb_h"]:
                            p = output_dir / db_to_rm
                            if p.exists():
                                subprocess.call(
                                    [str(self._mmseqs_bin), "rmdb", str(p)],
                                    stdout=subprocess.DEVNULL,
                                )

                    result_meta = self._collect_outputs(output_dir, req)
                    outputs.append(result_meta)

                except Exception as e:
                    self._log.error(f"MMSeqs2 execution failed: {e}")
                    import traceback

                    self._log.error(traceback.format_exc())
                    outputs.append({"error": str(e)})

        return outputs

    def _collect_outputs(self, output_dir: Path, req: Dict[str, Any]) -> Dict[str, Any]:
        a3m_files = []
        for item in output_dir.glob("*.a3m"):
            a3m_files.append(str(item.resolve()))

        dump_dir = req.get("dump_dir")
        if dump_dir:
            dump_path = Path(dump_dir)
            dump_path.mkdir(parents=True, exist_ok=True)
            new_files = []
            for f in a3m_files:
                dest = dump_path / Path(f).name
                if dest.exists():
                    dest.unlink()
                shutil.move(f, dest)
                new_files.append(str(dest))
            a3m_files = new_files
            output_dir = dump_path

        return {"a3m_files": a3m_files, "output_dir": str(output_dir)}

    def finalize_one(self, output: Any, finalize_ctx: Any) -> Any:
        if output.get("error"):
            raise RuntimeError(f"MMSeqs2 execution failed: {output['error']}")
        return output
