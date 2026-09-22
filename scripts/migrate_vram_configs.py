import sys
from pathlib import Path

mappings = {
    "rfdiffusion": {"weight_vram_mb": 500, "activation_vram_mb": 7500},
    "protenix": {"weight_vram_mb": 6000, "activation_vram_mb": 6000},
    "proteinmpnn": {"weight_vram_mb": 200, "activation_vram_mb": 3800},
    "diffdock": {"weight_vram_mb": 1000, "activation_vram_mb": 3000},
    "esm": {"weight_vram_mb": 1500, "activation_vram_mb": 2500},
    "mmseqs2": {"weight_vram_mb": 2000, "activation_vram_mb": 6000},
    "vina_gpu": {"weight_vram_mb": 1000, "activation_vram_mb": 11000},
    "boltzgen_design": {"weight_vram_mb": 6000, "activation_vram_mb": 14000},
    "boltzgen_inverse_fold": {"weight_vram_mb": 6000, "activation_vram_mb": 14000},
    "boltzgen_folding": {"weight_vram_mb": 6000, "activation_vram_mb": 14000},
    "boltzgen_design_folding": {"weight_vram_mb": 6000, "activation_vram_mb": 14000},
    "boltzgen_affinity": {"weight_vram_mb": 6000, "activation_vram_mb": 14000},
    "boltzgen_analysis": {"weight_vram_mb": 6000, "activation_vram_mb": 14000},
    "boltzgen_filtering": {"weight_vram_mb": 6000, "activation_vram_mb": 14000},
}

config_files = [
    "configs/workers.gpu01.yaml",
    "configs/workers.gpu0123-cfg.yaml",
    "configs/workers.gpu.yaml",
    "configs/workers.k8s-baseline.yaml",
    "configs/workers.slurm-baseline.yaml",
    "configs/workers.yaml"
]

repo_root = Path("/home/proton/proton")

for cf in config_files:
    path = repo_root / cf
    if not path.exists():
        print(f"Skipping {cf}")
        continue
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    in_components = False
    current_comp = None
    
    for line in lines:
        stripped = line.strip()
        if stripped == "components:":
            in_components = True
        elif stripped == "workers:":
            in_components = False
        
        if in_components:
            if ":" in line and not stripped.startswith("-") and line.startswith("    ") and not line.startswith("      "):
                name = stripped.split(":")[0].strip()
                if name in mappings:
                    current_comp = name
                else:
                    current_comp = None
            
            if current_comp and "gpu_memory_required_mb:" in line:
                indent = line[:line.find("gpu_memory_required_mb:")]
                m = mappings[current_comp]
                new_lines.append(f"{indent}weight_vram_mb: {m['weight_vram_mb']}\n")
                new_lines.append(f"{indent}activation_vram_mb: {m['activation_vram_mb']}\n")
                continue
        
        new_lines.append(line)
    
    with open(path, 'w') as f:
        f.writelines(new_lines)
    print(f"Updated {cf}")
