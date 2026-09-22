"""Gateway-managed input resolvers for component profiling runs."""

from __future__ import annotations

import csv
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .denovo_inputs import resolve_denovo_artifacts
from .runtime_paths import DEFAULT_DATASET_ROOT, PATHS
from .resolver_models import ResolverResult
from .sample_catalog import lookup_sample

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = DEFAULT_DATASET_ROOT


def _coerce_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return parsed if parsed >= 1 else int(default)


def _coerce_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return "true" if bool(value) else "false"
    if isinstance(value, str):
        return (
            "true" if value.strip().lower() in {"1", "true", "yes", "on"} else "false"
        )
    return "false"


def _sample_id_from_combo(combo: Mapping[str, Any]) -> str:
    raw_sample_id = str(combo.get("sample_id") or "").strip()
    if raw_sample_id:
        return raw_sample_id
    scaffold = str(combo.get("scaffold") or "").strip()
    ligand = str(combo.get("ligand") or "").strip()
    if scaffold and ligand:
        return f"{scaffold}_{ligand}"
    raise ValueError("sample_id is required (or scaffold+ligand must both be provided)")


def resolve_component_combo(combo: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize common sample fields from combo + samples manifest metadata."""
    merged: Dict[str, Any] = dict(combo)
    sample_id = _sample_id_from_combo(merged)
    sample_meta = lookup_sample(sample_id)

    resolved: Dict[str, Any] = {}
    resolved.update({k: v for k, v in sample_meta.items() if v is not None})
    resolved.update({k: v for k, v in merged.items() if v is not None})
    resolved["sample_id"] = sample_id
    return resolved


@lru_cache(maxsize=1)
def _load_samplesheet_contig_map() -> Dict[str, str]:
    samplesheet_root = _REPO_ROOT / "nextflow" / "samplesheet"
    source_dirs = [samplesheet_root / "handpicked", samplesheet_root / "casp"]

    mapping: Dict[str, str] = {}
    for source_dir in source_dirs:
        if not source_dir.exists() or not source_dir.is_dir():
            continue
        for csv_path in sorted(source_dir.glob("*.csv")):
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        sample_id = str(
                            row.get("sequence") or row.get("sample_id") or ""
                        ).strip()
                        contig = str(
                            row.get("contigs") or row.get("contig") or ""
                        ).strip()
                        if not sample_id or not contig:
                            continue
                        for key in (
                            sample_id,
                            sample_id.lower(),
                            csv_path.stem,
                            csv_path.stem.lower(),
                        ):
                            mapping[key] = contig
            except OSError:
                continue
    return mapping


def _lookup_contig(sample_id: str) -> Optional[str]:
    sid = str(sample_id).strip()
    if not sid:
        return None
    table = _load_samplesheet_contig_map()
    return table.get(sid) or table.get(sid.lower())


def _resolve_rfdiffusion_input_pdb(sample_id: str, dataset: str) -> str:
    default_pdb = Path(DATASET_ROOT) / "docked" / "processed" / f"{sample_id}.pdb"
    if dataset:
        dataset_pdb = (
            Path(DATASET_ROOT) / dataset.lower() / "docked" / f"{sample_id}.pdb"
        )
        if dataset_pdb.exists():
            return str(dataset_pdb.resolve())
    return str(default_pdb.resolve())


def _load_generate_protenix_json() -> Callable[..., Any]:
    script_path = (_REPO_ROOT / "scripts" / "generate_protenix_input.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "_gateway_generate_protenix_input", str(script_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load protenix input generator: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "generate_protenix_json", None)
    if not callable(fn):
        raise RuntimeError(
            "scripts/generate_protenix_input.py must expose generate_protenix_json(...)"
        )
    return fn


_GENERATE_PROTENIX_JSON = _load_generate_protenix_json()


def _first_csv_path(raw: str) -> str:
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    return tokens[0] if tokens else str(raw)


def _resolve_diffdock_ligand_path(
    ligand: Optional[str], dataset: Optional[str]
) -> Optional[str]:
    if not ligand:
        return None
    dataset_key = (dataset or "").lower()
    if dataset_key == "casp":
        base = getattr(PATHS.inputs, "casp_ligands_mol2", None)
        return str(Path(base) / f"{ligand}.mol2") if base else None
    if dataset_key == "handpicked":
        base = getattr(PATHS.inputs, "handpicked_ligands_mol2", None)
        return str(Path(base) / f"{ligand}.mol2") if base else None
    base = getattr(PATHS.inputs, "casf_coreset", None)
    if base:
        return str(Path(base) / ligand / f"{ligand}_ligand.mol2")
    return None


def _read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: str | None = None
    seq_lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:] or "seq"
                seq_lines = []
                continue
            seq_lines.append(line)
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def _materialize_batch(source_fasta: Path, dest_fasta: Path, batch_size: int) -> None:
    records = _read_fasta(source_fasta)
    if not records:
        raise ValueError(
            f"mmseqs2 source FASTA has no sequence records: {source_fasta}"
        )

    count = max(1, int(batch_size))
    selected: List[Tuple[str, str]] = []
    for idx in range(count):
        header, seq = records[idx % len(records)]
        selected.append((f"{header}_b{idx + 1}", seq))

    dest_fasta.parent.mkdir(parents=True, exist_ok=True)
    with dest_fasta.open("w", encoding="utf-8") as handle:
        for header, seq in selected:
            handle.write(f">{header}\n{seq}\n")


def _load_center_xyz(path: Path) -> Tuple[str, str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"center_x", "center_y", "center_z"}:
                values[key] = value
    if not {"center_x", "center_y", "center_z"} <= set(values):
        raise ValueError(f"vina_gpu center file missing keys: {path}")
    return values["center_x"], values["center_y"], values["center_z"]


def _resolve_vina_ligand_pdbqt(*, ligand: str, dataset: str) -> Path:
    base = Path(DATASET_ROOT) / dataset.lower() / "ligands" / "pdbqt"
    candidate = base / f"{ligand}.pdbqt"
    if not candidate.exists():
        raise FileNotFoundError(f"vina_gpu ligand pdbqt not found: {candidate}")
    return candidate.resolve()


def _resolve_rfdiffusion(
    *, spec: Any, combo: Mapping[str, Any], workdir: Path, **_: Any
) -> ResolverResult:
    resolved = resolve_component_combo(combo)

    sample_id = str(resolved.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("rfdiffusion requires sample_id")

    dataset = str(resolved.get("dataset") or "").strip()
    contig = str(
        resolved.get("actual_contig") or resolved.get("contig") or ""
    ).strip() or _lookup_contig(sample_id)
    if not contig:
        raise ValueError(f"rfdiffusion contig missing for sample_id={sample_id}")

    resolved_workdir = Path(workdir).resolve()
    out_dir = resolved_workdir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_designs = _coerce_int(
        resolved.get(
            "num_designs",
            resolved.get(
                "output_sample_count", getattr(spec, "output_sample_count", 1)
            ),
        ),
        getattr(spec, "output_sample_count", 1),
    )
    step = _coerce_int(resolved.get("step"), 50)
    with_overhead = _coerce_bool_text(resolved.get("with_overhead", False))
    input_pdb = _resolve_rfdiffusion_input_pdb(sample_id=sample_id, dataset=dataset)
    if not Path(input_pdb).exists():
        raise FileNotFoundError(f"rfdiffusion input_pdb not found: {input_pdb}")

    return ResolverResult(
        render_values={
            **resolved,
            "run_id": str(spec.run_id),
            "workdir": str(resolved_workdir),
            "output_prefix": str((out_dir / "rfdiffusion_out").resolve()),
            "model_dir": str(PATHS.inputs.rfdiffusion_models),
            "input_pdb": input_pdb,
            "num_designs": num_designs,
            "actual_contig": str(contig),
            "step": step,
            "with_overhead": with_overhead,
            "input_batch_size": int(spec.input_batch_size),
            "output_sample_count": int(spec.output_sample_count),
        }
    )


def _resolve_protenix(
    *, spec: Any, combo: Mapping[str, Any], workdir: Path, **_: Any
) -> ResolverResult:
    resolved = resolve_component_combo(combo)
    sample_id = str(resolved.get("sample_id") or "").strip()
    dataset = str(resolved.get("dataset") or "").strip() or None
    scaffold = str(resolved.get("scaffold") or "").strip() or None
    ligand = str(resolved.get("ligand") or "").strip() or None

    if not resolved.get("msa_path") or not resolved.get("fasta_path"):
        artifacts = resolve_denovo_artifacts(
            sample_id, scaffold=scaffold, ligand=ligand, dataset=dataset
        )
        resolved.setdefault("msa_path", str(artifacts.msa_a3m or ""))
        resolved.setdefault("fasta_path", str(artifacts.run_fasta_esm or ""))

    msa_path = str(resolved.get("msa_path") or "").strip()
    fasta_path = str(resolved.get("fasta_path") or "").strip()
    if not msa_path or not fasta_path:
        raise ValueError(
            f"protenix requires msa_path and fasta_path for sample_id={sample_id}"
        )
    if not Path(msa_path).exists():
        raise FileNotFoundError(f"protenix MSA file not found: {msa_path}")
    if not Path(fasta_path).exists():
        raise FileNotFoundError(f"protenix FASTA file not found: {fasta_path}")

    resolved_workdir = Path(workdir).resolve()
    input_dir = resolved_workdir / "inputs"
    results_dir = resolved_workdir / "results"
    input_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_size = _coerce_int(
        resolved.get("batch_size", getattr(spec, "input_batch_size", 1)),
        getattr(spec, "input_batch_size", 1),
    )
    _GENERATE_PROTENIX_JSON(msa_path, fasta_path, str(input_dir), count=batch_size)

    step = _coerce_int(resolved.get("step"), 200)
    cycle = _coerce_int(resolved.get("cycle"), 4)
    num_samples = _coerce_int(
        resolved.get(
            "num_samples",
            resolved.get(
                "output_sample_count", getattr(spec, "output_sample_count", 1)
            ),
        ),
        getattr(spec, "output_sample_count", 1),
    )

    return ResolverResult(
        render_values={
            **resolved,
            "run_id": str(spec.run_id),
            "workdir": str(resolved_workdir),
            "input_json_path": str((input_dir / "protenix_input.json").resolve()),
            "dump_dir": str(results_dir.resolve()),
            "batch_size": batch_size,
            "step": step,
            "cycle": cycle,
            "num_samples": num_samples,
            "input_batch_size": int(spec.input_batch_size),
            "output_sample_count": int(spec.output_sample_count),
        }
    )


def _resolve_proteinmpnn(
    *, spec: Any, combo: Mapping[str, Any], workdir: Path, **_: Any
) -> ResolverResult:
    resolved = resolve_component_combo(combo)
    sample_id = str(resolved.get("sample_id") or "").strip()
    dataset = str(resolved.get("dataset") or "").strip() or None
    scaffold = str(resolved.get("scaffold") or "").strip() or None
    ligand = str(resolved.get("ligand") or "").strip() or None

    if not resolved.get("protein_path"):
        artifacts = resolve_denovo_artifacts(
            sample_id, scaffold=scaffold, ligand=ligand, dataset=dataset
        )
        if artifacts.run_pdb:
            resolved["protein_path"] = str(artifacts.run_pdb)

    protein_path = str(resolved.get("protein_path") or "").strip()
    if not protein_path:
        raise ValueError(f"proteinmpnn requires protein_path for sample_id={sample_id}")
    protein_path = _first_csv_path(protein_path)
    if not Path(protein_path).exists():
        raise FileNotFoundError(f"proteinmpnn protein file not found: {protein_path}")

    resolved_workdir = Path(workdir).resolve()
    out_folder = resolved_workdir / "output" / "proteinmpnn_out"
    out_folder.mkdir(parents=True, exist_ok=True)

    batch_size = _coerce_int(
        resolved.get("batch_size", getattr(spec, "input_batch_size", 1)),
        getattr(spec, "input_batch_size", 1),
    )
    num_seq_per_target = _coerce_int(
        resolved.get(
            "num_seq_per_target",
            resolved.get(
                "output_sample_count", getattr(spec, "output_sample_count", 1)
            ),
        ),
        getattr(spec, "output_sample_count", 1),
    )
    sampling_temp = str(resolved.get("sampling_temp") or "0.15")

    return ResolverResult(
        render_values={
            **resolved,
            "run_id": str(spec.run_id),
            "workdir": str(resolved_workdir),
            "protein_path": protein_path,
            "out_folder": str(out_folder.resolve()),
            "batch_size": batch_size,
            "num_seq_per_target": num_seq_per_target,
            "sampling_temp": sampling_temp,
            "input_batch_size": int(spec.input_batch_size),
            "output_sample_count": int(spec.output_sample_count),
        }
    )


def _resolve_diffdock(
    *, spec: Any, combo: Mapping[str, Any], workdir: Path, **_: Any
) -> ResolverResult:
    resolved = resolve_component_combo(combo)
    sample_id = str(resolved.get("sample_id") or "").strip()
    dataset = str(resolved.get("dataset") or "").strip() or None
    scaffold = str(resolved.get("scaffold") or "").strip() or None
    ligand = str(resolved.get("ligand") or "").strip() or None

    artifacts = resolve_denovo_artifacts(
        sample_id, scaffold=scaffold, ligand=ligand, dataset=dataset
    )

    protein_path = str(resolved.get("protein_path") or "").strip()
    if not protein_path:
        protein_path = str(artifacts.convert_pdb or artifacts.run_pdb or "").strip()
    if not protein_path:
        raise ValueError(f"diffdock requires protein_path for sample_id={sample_id}")
    if not Path(protein_path).exists():
        raise FileNotFoundError(f"diffdock protein file not found: {protein_path}")

    ligand_path = str(resolved.get("ligand_path") or "").strip()
    if not ligand_path:
        ligand_path = str(
            _resolve_diffdock_ligand_path(ligand=ligand, dataset=dataset) or ""
        ).strip()
    if not ligand_path:
        raise ValueError(f"diffdock requires ligand_path for sample_id={sample_id}")
    if not Path(ligand_path).exists():
        raise FileNotFoundError(f"diffdock ligand file not found: {ligand_path}")

    resolved_workdir = Path(workdir).resolve()
    out_dir = resolved_workdir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = _coerce_int(
        resolved.get("batch_size", getattr(spec, "input_batch_size", 1)),
        getattr(spec, "input_batch_size", 1),
    )
    samples_per_complex = _coerce_int(
        resolved.get(
            "samples_per_complex",
            resolved.get(
                "output_sample_count", getattr(spec, "output_sample_count", 1)
            ),
        ),
        getattr(spec, "output_sample_count", 1),
    )
    steps = _coerce_int(resolved.get("steps"), 20)

    return ResolverResult(
        render_values={
            **resolved,
            "run_id": str(spec.run_id),
            "workdir": str(resolved_workdir),
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "out_dir": str(out_dir.resolve()),
            "batch_size": batch_size,
            "samples_per_complex": samples_per_complex,
            "steps": steps,
            "input_batch_size": int(spec.input_batch_size),
            "output_sample_count": int(spec.output_sample_count),
        }
    )


def _resolve_esm(
    *, spec: Any, combo: Mapping[str, Any], workdir: Path, **_: Any
) -> ResolverResult:
    resolved = resolve_component_combo(combo)
    sample_id = str(resolved.get("sample_id") or "").strip()
    dataset = str(resolved.get("dataset") or "").strip() or None
    scaffold = str(resolved.get("scaffold") or "").strip() or None
    ligand = str(resolved.get("ligand") or "").strip() or None

    if not resolved.get("ref_fasta"):
        artifacts = resolve_denovo_artifacts(
            sample_id, scaffold=scaffold, ligand=ligand, dataset=dataset
        )
        resolved.setdefault("ref_fasta", str(artifacts.run_fasta_esm or ""))

    ref_fasta = str(resolved.get("ref_fasta") or "").strip()
    if not ref_fasta:
        raise ValueError(f"esm requires ref_fasta for sample_id={sample_id}")
    if not Path(ref_fasta).exists():
        raise FileNotFoundError(f"esm FASTA file not found: {ref_fasta}")

    resolved_workdir = Path(workdir).resolve()
    out_dir = resolved_workdir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = _coerce_int(
        resolved.get("batch_size", getattr(spec, "input_batch_size", 1)),
        getattr(spec, "input_batch_size", 1),
    )
    num_variants = _coerce_int(
        resolved.get(
            "num_variants",
            resolved.get(
                "output_sample_count", getattr(spec, "output_sample_count", 1)
            ),
        ),
        getattr(spec, "output_sample_count", 1),
    )
    max_mutations = _coerce_int(resolved.get("max_mutations"), 1)
    top_k = _coerce_int(resolved.get("top_k"), 5)
    model = str(resolved.get("model") or "facebook/esm2_t33_650M_UR50D")

    return ResolverResult(
        render_values={
            **resolved,
            "run_id": str(spec.run_id),
            "workdir": str(resolved_workdir),
            "ref_fasta": ref_fasta,
            "out_dir": str(out_dir.resolve()),
            "batch_size": batch_size,
            "num_variants": num_variants,
            "max_mutations": max_mutations,
            "top_k": top_k,
            "model": model,
            "input_batch_size": int(spec.input_batch_size),
            "output_sample_count": int(spec.output_sample_count),
        }
    )


def _resolve_mmseqs2(
    *, spec: Any, combo: Mapping[str, Any], workdir: Path, **_: Any
) -> ResolverResult:
    resolved = resolve_component_combo(combo)
    sample_id = str(resolved.get("sample_id") or "").strip()
    dataset = str(resolved.get("dataset") or "").strip() or None
    scaffold = str(resolved.get("scaffold") or "").strip() or None
    ligand = str(resolved.get("ligand") or "").strip() or None

    artifacts = resolve_denovo_artifacts(
        sample_id, scaffold=scaffold, ligand=ligand, dataset=dataset
    )
    source_fasta = str(resolved.get("fasta_path") or "").strip()
    if not source_fasta:
        source_fasta = str(artifacts.run_fasta or artifacts.run_fasta_esm or "").strip()
    if not source_fasta:
        raise ValueError(f"mmseqs2 requires fasta_path for sample_id={sample_id}")

    source_fasta_path = Path(source_fasta).expanduser().resolve()
    if not source_fasta_path.exists():
        raise FileNotFoundError(f"mmseqs2 source FASTA not found: {source_fasta_path}")

    batch_size = _coerce_int(
        resolved.get("batch_size", getattr(spec, "input_batch_size", 1)),
        getattr(spec, "input_batch_size", 1),
    )
    num_iterations = _coerce_int(resolved.get("num_iterations"), 3)

    resolved_workdir = Path(workdir).resolve()
    inputs_dir = resolved_workdir / "inputs"
    out_dir = resolved_workdir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    batched_fasta = inputs_dir / f"{sample_id}_mmseqs_bs{batch_size}.fasta"
    _materialize_batch(source_fasta_path, batched_fasta, batch_size)

    db_base = str(
        resolved.get("db_base") or "/mnt/nfs/new/proton/mmseqs2/db"
    ).strip()
    db1 = str(resolved.get("db1") or "uniref30_2302_db").strip()

    return ResolverResult(
        render_values={
            **resolved,
            "run_id": str(spec.run_id),
            "workdir": str(resolved_workdir),
            "fasta_path": str(batched_fasta.resolve()),
            "output_dir": str(out_dir.resolve()),
            "db_base": db_base,
            "db1": db1,
            "num_iterations": num_iterations,
            "batch_size": int(batch_size),
            "input_batch_size": int(spec.input_batch_size),
            "output_sample_count": int(spec.output_sample_count),
        }
    )


def _resolve_vina_gpu(
    *, spec: Any, combo: Mapping[str, Any], workdir: Path, **_: Any
) -> ResolverResult:
    resolved = resolve_component_combo(combo)
    sample_id = str(resolved.get("sample_id") or "").strip()
    dataset = str(resolved.get("dataset") or "").strip()
    scaffold = str(resolved.get("scaffold") or "").strip()
    ligand = str(resolved.get("ligand") or "").strip()
    if not dataset or not scaffold or not ligand:
        raise ValueError(
            f"vina_gpu requires dataset/scaffold/ligand for sample_id={sample_id}"
        )

    artifacts = resolve_denovo_artifacts(
        sample_id, scaffold=scaffold, ligand=ligand, dataset=dataset
    )
    receptor_path = artifacts.receptor_pdbqt
    center_path = artifacts.receptor_center
    if not receptor_path or not center_path:
        raise ValueError(
            f"vina_gpu requires receptor_pdbqt/receptor_center for sample_id={sample_id}"
        )
    if not Path(receptor_path).exists():
        raise FileNotFoundError(f"vina_gpu receptor file not found: {receptor_path}")
    if not Path(center_path).exists():
        raise FileNotFoundError(
            f"vina_gpu receptor center file not found: {center_path}"
        )

    center_x, center_y, center_z = _load_center_xyz(Path(center_path))
    ligand_pdbqt = _resolve_vina_ligand_pdbqt(ligand=ligand, dataset=dataset)

    batch_size = _coerce_int(
        resolved.get("batch_size", getattr(spec, "input_batch_size", 1)),
        getattr(spec, "input_batch_size", 1),
    )
    thread = _coerce_int(resolved.get("thread"), 8000)
    search_depth = _coerce_int(resolved.get("search_depth"), 3)

    resolved_workdir = Path(workdir).resolve()
    return ResolverResult(
        render_values={
            **resolved,
            "run_id": str(spec.run_id),
            "workdir": str(resolved_workdir),
            "sample_name": sample_id,
            "receptor_path": str(Path(receptor_path).resolve()),
            "ligand_pdbqt_path": str(ligand_pdbqt),
            "center_x": center_x,
            "center_y": center_y,
            "center_z": center_z,
            "batch_size": int(batch_size),
            "thread": int(thread),
            "search_depth": int(search_depth),
            "input_batch_size": int(spec.input_batch_size),
            "output_sample_count": int(spec.output_sample_count),
        }
    )



_BOLTZGEN_SAMPLESHEET_DIRS = [
    Path("/mnt/nfs/new/proton/boltzgen/samplesheet/handpicked"),
    Path("/mnt/nfs/new/proton/boltzgen/samplesheet/casp"),
]


def _find_boltzgen_input_yaml(sample_id: str) -> str:
    """Locate the BoltzGen input YAML generated by ``csv_to_boltzgen_yaml.py``."""
    candidates = [
        sample_id,
        sample_id.lower(),
        sample_id.upper(),
    ]
    for search_dir in _BOLTZGEN_SAMPLESHEET_DIRS:
        if not search_dir.is_dir():
            continue
        for name in candidates:
            yaml_path = search_dir / f"{name}.yaml"
            if yaml_path.exists():
                return str(yaml_path.resolve())
    raise FileNotFoundError(
        f"boltzgen input YAML not found for sample_id={sample_id}. "
        f"Generate it via: python scripts/csv_to_boltzgen_yaml.py --csv <csv>"
    )


def _boltzgen_common(
    *,
    spec: Any,
    combo: Mapping[str, Any],
    workdir: Path,
) -> Tuple[Dict[str, Any], str, Path]:
    """Shared setup for all BoltzGen resolvers."""
    resolved = resolve_component_combo(combo)
    sample_id = str(resolved.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("boltzgen requires sample_id")
    input_yaml = _find_boltzgen_input_yaml(sample_id)
    resolved_workdir = Path(workdir).resolve()
    resolved_workdir.mkdir(parents=True, exist_ok=True)
    resolved["input_yaml"] = input_yaml
    resolved["sample_id"] = sample_id
    return resolved, input_yaml, resolved_workdir


def _resolve_boltzgen_design(
    *, spec: Any, combo: Mapping[str, Any], workdir: Path, **_: Any
) -> ResolverResult:
    resolved, input_yaml, resolved_workdir = _boltzgen_common(
        spec=spec, combo=combo, workdir=workdir
    )
    out_dir = resolved_workdir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_designs = _coerce_int(
        resolved.get(
            "num_designs",
            resolved.get(
                "output_sample_count", getattr(spec, "output_sample_count", 1)
            ),
        ),
        getattr(spec, "output_sample_count", 1),
    )
    diffusion_batch_size = _coerce_int(resolved.get("diffusion_batch_size"), 1)
    protocol = str(resolved.get("protocol") or "protein-anything")

    return ResolverResult(
        render_values={
            **resolved,
            "run_id": str(spec.run_id),
            "workdir": str(resolved_workdir),
            "input_yaml": input_yaml,
            "output_dir": str(out_dir.resolve()),
            "num_designs": int(num_designs),
            "diffusion_batch_size": int(diffusion_batch_size),
            "protocol": protocol,
            "input_batch_size": int(spec.input_batch_size),
            "output_sample_count": int(spec.output_sample_count),
        }
    )


def _resolve_boltzgen_inverse_fold(
    *, spec: Any, combo: Mapping[str, Any], workdir: Path, **_: Any
) -> ResolverResult:
    resolved, input_yaml, resolved_workdir = _boltzgen_common(
        spec=spec, combo=combo, workdir=workdir
    )
    out_dir = resolved_workdir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    budget = _coerce_int(resolved.get("budget"), 30)
    num_seq_per_target = _coerce_int(
        resolved.get(
            "num_seq_per_target",
            resolved.get(
                "output_sample_count", getattr(spec, "output_sample_count", 1)
            ),
        ),
        getattr(spec, "output_sample_count", 1),
    )
    protocol = str(resolved.get("protocol") or "protein-anything")

    return ResolverResult(
        render_values={
            **resolved,
            "run_id": str(spec.run_id),
            "workdir": str(resolved_workdir),
            "input_yaml": input_yaml,
            "output_dir": str(out_dir.resolve()),
            "budget": int(budget),
            "num_seq_per_target": int(num_seq_per_target),
            "protocol": protocol,
            "input_batch_size": int(spec.input_batch_size),
            "output_sample_count": int(spec.output_sample_count),
        }
    )


def _resolve_boltzgen_step(
    *, spec: Any, combo: Mapping[str, Any], workdir: Path, **_: Any
) -> ResolverResult:
    """Generic resolver for boltzgen_folding, boltzgen_design_folding, boltzgen_affinity,
    boltzgen_analysis, and boltzgen_filtering."""
    resolved, input_yaml, resolved_workdir = _boltzgen_common(
        spec=spec, combo=combo, workdir=workdir
    )
    out_dir = resolved_workdir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_designs = _coerce_int(
        resolved.get(
            "num_designs",
            resolved.get(
                "output_sample_count", getattr(spec, "output_sample_count", 1)
            ),
        ),
        getattr(spec, "output_sample_count", 1),
    )
    protocol = str(resolved.get("protocol") or "protein-anything")
    sampling_steps = _coerce_int(resolved.get("sampling_steps"), 200)
    recycling_steps = _coerce_int(resolved.get("recycling_steps"), 3)

    return ResolverResult(
        render_values={
            **resolved,
            "run_id": str(spec.run_id),
            "workdir": str(resolved_workdir),
            "input_yaml": input_yaml,
            "output_dir": str(out_dir.resolve()),
            "num_designs": int(num_designs),
            "protocol": protocol,
            "sampling_steps": int(sampling_steps),
            "recycling_steps": int(recycling_steps),
            "input_batch_size": int(spec.input_batch_size),
            "output_sample_count": int(spec.output_sample_count),
        }
    )


_ResolverFn = Callable[..., ResolverResult]

_RESOLVERS: Dict[str, _ResolverFn] = {
    "rfdiffusion": _resolve_rfdiffusion,
    "protenix": _resolve_protenix,
    "proteinmpnn": _resolve_proteinmpnn,
    "diffdock": _resolve_diffdock,
    "esm": _resolve_esm,
    "mmseqs2": _resolve_mmseqs2,
    "vina_gpu": _resolve_vina_gpu,
    "boltzgen_design": _resolve_boltzgen_design,
    "boltzgen_inverse_fold": _resolve_boltzgen_inverse_fold,
    "boltzgen_folding": _resolve_boltzgen_step,
    "boltzgen_design_folding": _resolve_boltzgen_step,
    "boltzgen_affinity": _resolve_boltzgen_step,
    "boltzgen_analysis": _resolve_boltzgen_step,
    "boltzgen_filtering": _resolve_boltzgen_step,
}


def resolve_component_inputs(
    component: str,
    spec: Any,
    combo: Mapping[str, Any],
    workdir: Path | str,
    output_dir: Path | str,
    level: str,
) -> ResolverResult:
    """Resolve component-specific profiling run inputs through a common gateway layer."""
    resolved_component = str(component or "").strip().lower()
    resolver = _RESOLVERS.get(resolved_component)
    if resolver is None:
        raise ValueError(f"unsupported component '{resolved_component}'")
    result = resolver(
        spec=spec,
        combo=dict(combo or {}),
        workdir=Path(workdir).resolve(),
        output_dir=Path(output_dir).resolve(),
        level=level,
    )
    if not isinstance(result, ResolverResult):
        raise RuntimeError(
            f"resolver for '{resolved_component}' must return ResolverResult"
        )
    return result


__all__ = ["resolve_component_combo", "resolve_component_inputs"]
