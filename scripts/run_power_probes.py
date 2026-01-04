#!/usr/bin/env python3
"""Phase 2: Run power differential probes on Wikipedia Talk Pages.

This is a stub script. The actual power probe pipeline runs via Modal:

    # Extract training data
    modal run src/routing/modal_app.py --extract --dataset wikipedia_talk --split train --max-samples 5000
    
    # Extract validation data  
    modal run src/routing/modal_app.py --extract --dataset wikipedia_talk --split validation --max-samples 1000
    
    # Run power probes
    modal run src/routing/modal_app.py --power-probe --train-cache <train.npz> --eval-cache <val.npz>

See src/routing/modal_app.py for the full implementation.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run power probes on Wikipedia Talk")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 15])
    parser.parse_args()
    
    print("Phase 2: Power Differential Probes")
    print("")
    print("This is a stub. Run power probes via Modal:")
    print("")
    print("  modal run src/routing/modal_app.py --power-probe \\")
    print("    --train-cache <train_cache.npz> \\")
    print("    --eval-cache <eval_cache.npz>")
    print("")
    print("See README.md for full instructions.")


if __name__ == "__main__":
    main()
