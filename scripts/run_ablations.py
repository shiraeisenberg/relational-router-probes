#!/usr/bin/env python3
"""Phase 3: Run ablation studies across signals, layers, and pooling methods.

Ablations to run:
1. Pooling sensitivity: mean vs max vs last-token
2. Layer-by-layer analysis: which layers encode which signals
3. Expert cluster analysis: do signals share experts or are they disjoint
4. Cross-dataset transfer: DailyDialog → MELD

This is a stub. Core ablations are partially implemented in src/analysis/.

Usage:
    python scripts/run_ablations.py --ablation pooling
    python scripts/run_ablations.py --ablation layers
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["all", "pooling", "layers", "experts", "transfer"],
                        help="Which ablation to run")
    args = parser.parse_args()
    
    print("Phase 3: Ablation Studies")
    print("")
    print(f"Ablation: {args.ablation}")
    print("")
    print("Core ablation implementations are in src/analysis/:")
    print("  - layer_analysis.py: Layer-by-layer probing")
    print("  - expert_clusters.py: Expert association analysis")
    print("  - transfer.py: Cross-dataset transfer")
    print("  - baselines.py: BoW and punctuation baselines")
    print("")
    print("Full systematic ablation runner not yet implemented.")
    print("See AGENTS.md Phase 3 for task list.")


if __name__ == "__main__":
    main()
