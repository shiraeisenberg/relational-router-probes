#!/usr/bin/env python3
"""
Generate all figures for the Substack post.

Usage:
    python scripts/generate_figures.py

Output files (in results/figures/):
    - cross_signal_comparison_v2.png  (with Enron results)
    - layer_wise_auc_v2.png  (2x3 grid with all 5 signals)
    - compression_ratio_v2.png
    - wikipedia_vs_enron.png  (NEW: comparison figure)
    - results_table_v2.png

These figures include Enron power probe results showing:
- Power (Wikipedia): 0.608 AUC (inconclusive - noisy labels)
- Power (Enron): 0.755 AUC (confirmed - power exercised)
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.visualization import generate_all_figures


def main():
    """Generate all visualization figures."""
    output_dir = project_root / "results" / "figures"
    
    print("=" * 60)
    print("Generating figures for Substack post (with Enron results)")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    
    figures = generate_all_figures(output_dir)
    
    print()
    print("=" * 60)
    print("Generated figures:")
    print("=" * 60)
    for name, path in figures.items():
        print(f"  {name}: {path.name}")
    
    print()
    print("✓ All figures generated successfully!")
    print(f"  View in: {output_dir}")
    

if __name__ == "__main__":
    main()
