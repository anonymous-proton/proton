"""Resolve denovo-derived inputs for gateway profiling runs."""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .runtime_paths import DEFAULT_DATASET_ROOT
from .sample_catalog import lookup_sample


@dataclass
class DenovoArtifacts:
    base_dir: Path
    convert_dir: Path
    run_dir: Path
    msa_dir: Path
    dataset: Optional[str] = None
    msa_a3m: Optional[Path] = None
    convert_pdb: Optional[Path] = None
    receptor_pdbqt: Optional[Path] = None
    receptor_center: Optional[Path] = None
    run_fasta: Optional[Path] = None
    run_fasta_esm: Optional[Path] = None
    run_pdb: Optional[Path] = None
    docked_pdbqt: Optional[Path] = None


def _find_first(patterns: Iterable[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[0])
    return None


def _valid_pdb(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return None
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if "ATOM" in line or line.startswith("ATOM"):
                    return path
                if idx > 200:
                    break
    except Exception:
        return None
    return None


def _candidate_bases(sample_id: str, scaffold: Optional[str], ligand: Optional[str], dataset: str) -> List[Path]:
    dataset_root = DEFAULT_DATASET_ROOT / str(dataset).lower() / "nextflow"
    sample_lower = sample_id.lower()
    sample_upper = sample_id.upper()
    bases: List[Path] = [
        dataset_root / "denovo" / sample_id,
        dataset_root / "denovo" / sample_lower,
        dataset_root / "denovo" / sample_upper,
        dataset_root / sample_id,
        dataset_root / sample_lower,
        dataset_root / sample_upper,
        dataset_root,
    ]
    if scaffold and ligand:
        bases.append(dataset_root / f"denovo_{scaffold}_{ligand}")
        bases.append(dataset_root / f"{scaffold}_{ligand}")
    seen: set[str] = set()
    unique: List[Path] = []
    for base in bases:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        unique.append(base)
    return unique


def resolve_denovo_artifacts(
    sample_id: str,
    *,
    scaffold: Optional[str] = None,
    ligand: Optional[str] = None,
    dataset: Optional[str] = None,
) -> DenovoArtifacts:
    meta = lookup_sample(sample_id)
    scaffold = scaffold or meta.get("scaffold")
    ligand = ligand or meta.get("ligand")
    dataset = dataset or meta.get("dataset")
    if not dataset:
        raise ValueError(
            f"Dataset is required for sample_id={sample_id}; set dataset in samples.yaml "
            "or pass it explicitly."
        )

    bases = _candidate_bases(sample_id, scaffold, ligand, dataset)
    base_dir = next((base for base in bases if base.exists()), None)
    if base_dir is None:
        tried = ", ".join(str(base) for base in bases)
        raise FileNotFoundError(f"No denovo directory found for {sample_id}; tried: {tried}")

    convert_dir = base_dir / "convert"
    run_dir = base_dir / "run"
    msa_dir = base_dir / "msa"

    prefixes: List[str] = [sample_id, sample_id.lower(), sample_id.upper()]
    if scaffold and ligand:
        prefixes.extend([f"{scaffold}_{ligand}", f"{scaffold.lower()}_{ligand.lower()}"])
    prefixes = list(dict.fromkeys(prefixes))

    handpicked = str(dataset).strip().lower() == "handpicked"

    def convert_patterns(suffix: str) -> List[str]:
        patterns: List[str] = []
        for prefix in prefixes:
            patterns.append(str(convert_dir / f"{prefix}_0_mpnn_0_esm_0_seed_101_sample_0{suffix}"))
            patterns.append(str(convert_dir / f"{prefix}_0_mpnn_0_esm_0{suffix}"))
            patterns.append(str(convert_dir / f"{prefix}_*{suffix}"))
        if handpicked and scaffold:
            patterns.append(str(convert_dir / f"{scaffold}_receptor{suffix}"))
            patterns.append(str(convert_dir / f"{scaffold.upper()}_receptor{suffix}"))
            patterns.append(str(convert_dir / f"{scaffold.lower()}_receptor{suffix}"))
        return patterns

    def run_patterns(suffix: str) -> List[str]:
        patterns: List[str] = []
        for prefix in prefixes:
            patterns.append(str(run_dir / f"{prefix}_0_mpnn_0_esm_0_seed_101_sample_0{suffix}"))
            patterns.append(str(run_dir / f"{prefix}_*{suffix}"))
        if handpicked and scaffold and ligand:
            patterns.append(str(run_dir / f"{scaffold}_receptor_{ligand}{suffix}"))
            patterns.append(str(run_dir / f"{scaffold.upper()}_receptor_{ligand.upper()}{suffix}"))
            patterns.append(str(run_dir / f"{scaffold.lower()}_receptor_{ligand.lower()}{suffix}"))
        return patterns

    artifacts = DenovoArtifacts(
        base_dir=base_dir,
        convert_dir=convert_dir,
        run_dir=run_dir,
        msa_dir=msa_dir,
        dataset=dataset,
        msa_a3m=_find_first([str(msa_dir / f"{p}_0_mpnn_0_esm_0.a3m") for p in prefixes]),
        convert_pdb=_find_first(convert_patterns(".pdb")),
        receptor_pdbqt=_find_first(convert_patterns("_receptor.pdbqt")),
        receptor_center=_find_first(convert_patterns("_receptor_center.txt")),
        run_fasta=_find_first(
            [str(run_dir / f"{p}_0_mpnn_0.fasta") for p in prefixes]
            + [str(run_dir / f"{p}_0.fasta") for p in prefixes]
        ),
        run_fasta_esm=_find_first([str(run_dir / f"{p}_0_mpnn_0_esm_0.fasta") for p in prefixes]),
        run_pdb=_find_first([str(run_dir / f"{p}_0.pdb") for p in prefixes]),
        docked_pdbqt=_find_first(run_patterns("_docked.pdbqt")),
    )
    artifacts.convert_pdb = _valid_pdb(artifacts.convert_pdb)
    artifacts.run_pdb = _valid_pdb(artifacts.run_pdb)
    if artifacts.convert_pdb is None and artifacts.run_pdb is None:
        raise FileNotFoundError(
            f"Missing usable denovo protein structure for sample_id={sample_id} (dataset={dataset}); "
            f"expected convert/run PDB under {base_dir}."
        )
    return artifacts

