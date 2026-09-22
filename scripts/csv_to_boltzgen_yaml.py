import pandas as pd
import os
import sys
import argparse
from rdkit import Chem

AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def get_smiles_from_sdf(sdf_path):
    """
    Reads an SDF file and returns the Canonical SMILES string.
    """
    if not os.path.exists(sdf_path):
        print(f"Warning: SDF file not found at {sdf_path}")
        return None
    
    try:
        suppl = Chem.SDMolSupplier(sdf_path)
        for mol in suppl:
            if mol is not None:
                return Chem.MolToSmiles(mol, isomericSmiles=True)
        return None
    except Exception as e:
        print(f"Error parsing SDF {sdf_path}: {e}")
        return None

def parse_pdb_residues(pdb_path):
    """
    Parses PDB to map {Chain: {ResNum: 'A'}}
    """
    residues = {}
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21]
                res_name = line[17:20].strip()
                try:
                    res_num = int(line[22:26].strip())
                except ValueError:
                    continue

                if res_name in AA_MAP:
                    if chain_id not in residues:
                        residues[chain_id] = {}
                    if res_num not in residues[chain_id]:
                        residues[chain_id][res_num] = AA_MAP[res_name]
    return residues

def convert_contigs_to_sequence(contig_str, residue_map):
    """
    Converts RFdiffusion contig '[5-10/A32-42]' to Boltzgen sequence '5..10SEQ...'
    """
    clean_str = contig_str.replace('[', '').replace(']', '').replace(' ', '')
    segments = clean_str.split('/')
    
    boltzgen_seq_parts = []
    
    for seg in segments:
        if not seg: continue
        
        if seg[0].isalpha():
            chain = seg[0]
            region = seg[1:]
            
            if '-' in region:
                start, end = map(int, region.split('-'))
            else:
                start = end = int(region)
            
            seq_chunk = ""
            if chain in residue_map:
                for r in range(start, end + 1):
                    if r in residue_map[chain]:
                        seq_chunk += residue_map[chain][r]
                    else:
                        print(f"Warning: Residue {chain}{r} missing in PDB, using 'X'")
                        seq_chunk += 'X'
            else:
                print(f"Warning: Chain {chain} not found in PDB residue map.")
            
            boltzgen_seq_parts.append(seq_chunk)
            
        elif seg[0].isdigit():
            boltzgen_seq_parts.append(seg.replace('-', '..'))
            
        elif seg == '0':
            pass
            
    return "".join(boltzgen_seq_parts)

def generate_yaml_from_csv(csv_path, output_dir):
    """
    Main function to process CSV and generate YAMLs.
    """
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, row in df.iterrows():
        seq_id = row['sequence']
        docked_pdb = row['docked_pdb']
        ligand_sdf = row['ligand_sdf']
        contigs = row['contigs']
        
        print(f"Processing {seq_id}...")

        smiles = get_smiles_from_sdf(ligand_sdf)
        if not smiles:
            print(f"  - Failed to get SMILES for {seq_id}, skipping or using placeholder.")
            smiles = "C"

        try:
            pdb_residues = parse_pdb_residues(docked_pdb)
            protein_seq = convert_contigs_to_sequence(contigs, pdb_residues)
        except Exception as e:
            print(f"  - Error parsing PDB/Contigs for {seq_id}: {e}")
            continue

        
        yaml_content = f"""entities:
  # Designed protein based on contigs
  - protein:
      id: A
      sequence: {protein_seq}

  # Ligand defined from SDF (SMILES)
  - ligand:
      id: L
      smiles: '{smiles}'
      binding_types: B
"""
        
        output_file = os.path.join(output_dir, f"{seq_id}.yaml")
        with open(output_file, 'w') as f:
            f.write(yaml_content)
            
        print(f"  -> Generated {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Boltzgen YAML from CSV")
    parser.add_argument("--csv", required=True, help="Input CSV file path")
    parser.add_argument("--outdir", default=".", help="Output directory for YAML files")
    
    args = parser.parse_args()
    
    generate_yaml_from_csv(args.csv, args.outdir)