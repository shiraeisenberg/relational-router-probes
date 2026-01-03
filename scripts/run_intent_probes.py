#!/usr/bin/env python3
"""Phase 1: Run intent classification probes on DailyDialog.

This script:
1. Loads DailyDialog dataset
2. Extracts router logits via Modal
3. Trains intent probes (4-class dialogue act)
4. Compares router logits vs residual stream baseline
5. Saves results to results/tables/

Usage:
    python scripts/run_intent_probes.py
    python scripts/run_intent_probes.py --layers 4 8 12 15
    python scripts/run_intent_probes.py --pooling mean max last
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np

from src.data.dailydialog import load_dailydialog
from src.routing.extraction import load_olmoe_with_hooks, extract_router_logits
from src.routing.aggregation import pool_router_logits
from src.probing.intent_probe import train_intent_probe, train_intent_probe_residual
from src.probing.linear_probe import compare_probes


def main():
    parser = argparse.ArgumentParser(description="Run intent probes on DailyDialog")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 15])
    parser.add_argument("--pooling", nargs="+", default=["mean"])
    parser.add_argument("--output-dir", type=str, default="results/tables")
    parser.add_argument("--cache-dir", type=str, default="results/extractions")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Phase 1: Intent Classification Probes")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading DailyDialog...")
    try:
        train_turns = load_dailydialog(split="train")
        test_turns = load_dailydialog(split="test")
        print(f"  Loaded {len(train_turns)} train, {len(test_turns)} test turns")
    except NotImplementedError as e:
        print(f"  ERROR: {e}")
        print("  Please implement the DailyDialog loader first.")
        return
    
    # Extract texts and labels
    texts = [t.text for t in train_turns]
    labels = np.array([t.dialogue_act for t in train_turns])
    
    # Check for cached extractions
    cache_file = cache_dir / f"dailydialog_train_layers{'_'.join(map(str, args.layers))}.npz"
    
    if cache_file.exists():
        print(f"\n[2/4] Loading cached router logits from {cache_file}...")
        cached = np.load(cache_file, allow_pickle=True)
        router_logits_by_layer = cached["router_logits"].item()
    else:
        print("\n[2/4] Extracting router logits (this may take a while)...")
        print("  Consider running via Modal for faster extraction.")
        
        # TODO: Implement Modal-based extraction
        # For now, placeholder
        print("  ERROR: Router extraction not yet implemented.")
        print("  Run `modal run src/routing/modal_app.py` first.")
        return
    
    # Train probes
    print("\n[3/4] Training probes...")
    results = []
    
    for layer in args.layers:
        for pooling in args.pooling:
            print(f"  Layer {layer}, pooling={pooling}...")
            
            # Pool router logits
            X_router = np.array([
                pool_router_logits(router_logits_by_layer[text][layer], method=pooling)
                for text in texts
            ])
            
            # Train router probe
            router_result = train_intent_probe(
                X_router, labels, layer=layer, pooling=pooling
            )
            results.append(router_result)
            
            print(f"    Router AUC: {router_result.auc:.3f}")
    
    # Save results
    print("\n[4/4] Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"intent_probes_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print(f"  Saved to {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r.task} | layer={r.layer} | {r.pooling} | AUC={r.auc:.3f}")


if __name__ == "__main__":
    main()
