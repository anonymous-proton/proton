#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import sys


def _existing_input_path(path_value: str) -> Path:
    """Resolve input path without following symlinks, so staged Nextflow symlinks remain valid."""
    candidate = Path(path_value).expanduser()
    if candidate.exists():
        return candidate

    absolute_candidate = candidate.absolute()
    if absolute_candidate.exists():
        return absolute_candidate

    raise FileNotFoundError(f"Input path not found: {path_value}")

def read_fasta_sequences(fasta_file):
    """
    Read all sequences from a FASTA file.
    Return list of tuples: [(header, sequence), ...]
    """
    records = []
    header = None
    seq_chunks = []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append((header, ''.join(seq_chunks)))
    return records

def create_msa_files(msa_file: str, msa_dir: str, query_sequence: str) -> None:
    """Create pairing.a3m and non_pairing.a3m files from the input MSA file."""
    with open(msa_file, "r") as f:
        lines = f.readlines()
    
    with open(os.path.join(msa_dir, "pairing.a3m"), 'w') as f:
        indexed_sequence = "".join(query_sequence)
        f.write(f">UniRef100_query_0/\n{indexed_sequence}\n")
    
    with open(os.path.join(msa_dir, "non_pairing.a3m"), 'w') as f:
        f.write(f">query\n{query_sequence}\n")
        
        current_header = None
        current_seq = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                if current_header and current_seq:
                    f.write(f"{current_header}\n{''.join(current_seq)}\n")
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_header and current_seq:
            f.write(f"{current_header}\n{''.join(current_seq)}\n")

def generate_protenix_json(msa_file: str, fasta_file: str, output_dir: str, count: int = 1) -> list:
    """
    Generate Protenix input JSON and required A3M files into the given output directory.
    If FASTA contains multiple sequences, print a warning and use only the first sequence.
    """
    count = int(count)
    if count < 1:
        raise ValueError(f"count must be >= 1 (got {count})")

    msa_path = _existing_input_path(msa_file)
    fasta_path = _existing_input_path(fasta_file)
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not msa_path.exists():
        raise FileNotFoundError(f"MSA file not found: {msa_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    print(f"Using MSA file: {msa_path}")
    print(f"Using FASTA file: {fasta_path}")
    print(f"Output directory: {out_dir}")

    records = read_fasta_sequences(str(fasta_path))
    if len(records) == 0:
        raise ValueError("Could not read sequence from FASTA file")
    if len(records) > 1:
        first_header = records[0][0] or "seq0"
        print(f"[WARN] Multiple sequences detected in FASTA ({len(records)}). "
              f"Using only the first one: {first_header}")

    query_sequence = records[0][1]
    if not query_sequence:
        raise ValueError("First sequence is empty in FASTA file")

    create_msa_files(str(msa_path), str(out_dir), query_sequence)

    base_name = Path(msa_path).stem
    json_data = []
    for idx in range(count):
        sample_name = base_name if idx == 0 else f"{base_name}_rep{idx}"
        json_data.append(
            {
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": query_sequence,
                            "count": 1,
                            "msa": {
                                "precomputed_msa_dir": str(out_dir),
                                "pairing_db": "uniref100"
                            }
                        }
                    }
                ],
                "name": sample_name
            }
        )

    json_path = out_dir / "protenix_input.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON file saved to: {json_path}")

    return json_data

def main():
    parser = argparse.ArgumentParser(
        description="Generate Protenix input JSON and A3M files into an output directory"
    )
    parser.add_argument("msa_file", help="Input MSA in A3M format")
    parser.add_argument('--fasta', '-f', required=True, help='Input FASTA file')
    parser.add_argument('--output', '-o', required=True, help='Output directory to write protenix_input.json and A3M files')
    parser.add_argument('--count', '-c', type=int, default=1, help='Number of inputs to emit (duplicates the FASTA/MSA content with unique names)')
    args = parser.parse_args()

    try:
        generate_protenix_json(args.msa_file, args.fasta, args.output, args.count)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
