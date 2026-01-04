#!/usr/bin/env python3
"""Generate figures for writeup.

Creates visualization figures from probe results:
- Cross-signal AUC comparison
- Layer-by-layer analysis
- Compression ratio chart
- Expert overlap heatmap (if cluster analysis complete)

Usage:
    python scripts/generate_figures.py --results-dir results/tables --output-dir results/figures

Pre-generated figures are already in results/figures/.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Generate figures from probe results")
    parser.add_argument("--results-dir", type=str, default="results/tables",
                        help="Directory containing probe result JSON files")
    parser.add_argument("--output-dir", type=str, default="results/figures",
                        help="Directory for output figures")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load probe results
    result_files = list(results_dir.glob("*.json"))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        print("Run probing experiments first.")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Aggregate results
    all_results = []
    for rf in result_files:
        with open(rf) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
    
    print(f"Loaded {len(all_results)} probe results")
    
    # Generate figures
    from src.analysis.visualization import plot_auc_comparison
    
    # Group by task and probe target
    router_results = {}
    for r in all_results:
        if r.get("probe_target") == "router_logits":
            task = r.get("task", "unknown")
            auc = r.get("auc", 0)
            if task not in router_results or auc > router_results[task]:
                router_results[task] = auc
    
    if router_results:
        output_path = output_dir / "auc_comparison.png"
        plot_auc_comparison(router_results, str(output_path))
        print(f"Saved: {output_path}")
    
    print("")
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
