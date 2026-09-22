import subprocess
import pandas as pd
from io import StringIO
from pathlib import Path
import argparse
import numpy as np


def load_ncu_rep(
    rep_path: Path,
    dump_csv: bool = False,
    dump_dir: Path | None = None
) -> pd.DataFrame:
    """
    Load a single ncu-rep file and return raw metric dataframe.
    Optionally dump raw CSV for debugging.
    """

    cmd = [
        "bash",
        "/usr/local/cuda/bin/ncu",
        "--import", str(rep_path),
        "--csv",
        "--page", "raw"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    df = pd.read_csv(StringIO(result.stdout))

    if dump_csv:
        if dump_dir is None:
            dump_dir = rep_path.parent

        dump_dir.mkdir(parents=True, exist_ok=True)

        dump_path = dump_dir / f"{rep_path.stem}_raw.csv"
        df.to_csv(dump_path, index=False)

        print(f"[DEBUG] raw ncu csv dumped -> {dump_path}")

    return df

def process_ncu_timeline(
    timeline_path: str | Path,
    rep_dir: str | Path,
    output_path: str | Path = "representative_metrics.csv"
):
    """
    Core logic to fuse multiple ncu-rep files using external time (%) CSV.
    Can be called directly from other scripts.
    """
    timeline_path = Path(timeline_path)
    rep_dir = Path(rep_dir)
    output_path = Path(output_path)

    print(f"[+] Loading timeline: {timeline_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline file not found: {timeline_path}")

    timeline = pd.read_csv(timeline_path)

    timeline["weight"] = timeline["Time (%)"] / timeline["Time (%)"].sum()

    aggregated = {}
    unit_row = None

    to_exclude = [
        'ID', 'Process ID', 'Process Name', 'Host Name', 'Kernel Name', 
        'Context', 'Stream', 'Block Size', 'Grid Size', 'Device', 'CC', 
        'c2clink__enabled_mask', 'c2clink__present'
    ]

    for _, row in timeline.iterrows():
        idx = int(row["Index"])
        weight = row["weight"]

        rep_path = rep_dir / f"kernel_{idx}.ncu-rep"
        print(f"[+] Processing {rep_path} (weight={weight:.4f})")
        
        if not rep_path.exists():
            print(f"[Warning] File not found: {rep_path}. Skipping...")
            continue

        df = load_ncu_rep(rep_path, dump_csv=False, dump_dir=Path("./"))

        if unit_row is None:
            unit_row = df.iloc[0]

        data_rows = df.iloc[1:]
        
        metric_columns = [
            c for c in df.columns
            if c not in to_exclude
        ]

        for _, d_row in data_rows.iterrows():
            for metric in metric_columns:
                val = d_row[metric]

                try:
                    if isinstance(val, str):
                        val = val.replace(',', '')
                    
                    f_val = float(val)
                    
                    if np.isnan(f_val):
                        continue
                    
                    aggregated.setdefault(metric, 0.0)
                    aggregated[metric] += f_val * weight

                except (ValueError, TypeError):
                    continue

    result_df = (
        pd.DataFrame.from_dict(aggregated, orient="index", columns=["value"])
          .reset_index()
          .rename(columns={"index": "metric"})
          .sort_values("metric")
    )

    if unit_row is not None:
        result_df["unit"] = result_df["metric"].map(unit_row)
    else:
        result_df["unit"] = ""
        
    result_df = result_df[["metric", "unit", "value"]]

    print(f"[+] Writing result to {output_path}")
    result_df.to_csv(output_path, index=False)

    print("Done! Time-weighted representative NCU metrics generated!")
    return result_df



def main():
    parser = argparse.ArgumentParser(
        description="Fuse multiple ncu-rep files using external time (%) CSV"
    )
    parser.add_argument(
        "--timeline",
        type=Path,
        required=True,
        help="CSV with Index, Time (%), Name"
    )
    parser.add_argument(
        "--rep-dir",
        type=Path,
        required=True,
        help="Directory containing kernel_{index}.ncu-rep"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("representative_metrics.csv"),
        help="Output CSV"
    )

    args = parser.parse_args()

    process_ncu_timeline(
        timeline_path=args.timeline,
        rep_dir=args.rep_dir,
        output_path=args.output
    )


if __name__ == "__main__":
    main()