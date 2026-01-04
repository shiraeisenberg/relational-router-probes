#!/usr/bin/env python3
"""Phase 1: Run intent classification probes on DailyDialog.

This script:
1. Loads DailyDialog dataset
2. Checks for cached extractions (or runs Modal extraction)
3. Trains intent probes (4-class dialogue act)
4. Compares router logits vs residual stream baseline
5. Saves results to results/tables/

Usage:
    # Run with cached extractions
    python scripts/run_intent_probes.py
    
    # Specify layers and pooling methods
    python scripts/run_intent_probes.py --layers 4 8 12 15 --pooling mean max last
    
    # Force re-extraction (ignores cache)
    python scripts/run_intent_probes.py --no-cache
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data.dailydialog import load_dailydialog, DialogueTurn
from src.routing.extraction import (
    load_extraction_cache,
    find_latest_cache,
    ExtractionCache,
)
from src.probing.linear_probe import train_probe, ProbeResult


# Label encoding for dialogue acts
DIALOGUE_ACT_LABELS = ["inform", "question", "directive", "commissive"]


def prepare_extraction_input(
    turns: list[DialogueTurn],
    output_path: Path
) -> Path:
    """Prepare DailyDialog turns for Modal extraction.
    
    Creates a JSON file with texts and sample IDs for the Modal extractor.
    
    Args:
        turns: List of DialogueTurn objects
        output_path: Where to save the JSON file
        
    Returns:
        Path to the created JSON file
    """
    data = [
        {"id": turn.turn_id, "text": turn.text}
        for turn in turns
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)
    
    print(f"Prepared {len(data)} samples for extraction: {output_path}")
    return output_path


def run_intent_probes(
    cache: ExtractionCache,
    turns: list[DialogueTurn],
    layers: list[int],
    pooling_methods: list[str],
    split: str = "train",
) -> list[ProbeResult]:
    """Train intent probes on cached extractions.
    
    Args:
        cache: Loaded extraction cache
        turns: DialogueTurn objects with labels
        layers: Which layers to probe
        pooling_methods: Which pooling methods to test
        split: "train" for train/test split, "validation" for validation
        
    Returns:
        List of ProbeResult objects
    """
    # Build sample_id to index mapping
    sample_id_to_idx = {sid: i for i, sid in enumerate(cache.sample_ids)}
    
    # Build sample_id to label mapping
    sample_id_to_label = {turn.turn_id: turn.dialogue_act for turn in turns}
    
    # Filter to samples that are in both cache and turns
    valid_sample_ids = [
        sid for sid in cache.sample_ids
        if sid in sample_id_to_label
    ]
    
    if len(valid_sample_ids) < len(cache.sample_ids):
        print(f"  Warning: {len(cache.sample_ids) - len(valid_sample_ids)} samples in cache not found in turns")
    
    # Get labels
    labels = np.array([
        DIALOGUE_ACT_LABELS.index(sample_id_to_label[sid])
        for sid in valid_sample_ids
    ])
    
    # Train/test split (80/20)
    n_samples = len(valid_sample_ids)
    n_train = int(0.8 * n_samples)
    
    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    results = []
    
    for layer in layers:
        for pooling in pooling_methods:
            print(f"  Layer {layer}, pooling={pooling}...")
            
            # Get pooled router logits
            X_router = cache.get_pooled_router_logits(layer, method=pooling)
            
            # Filter to valid samples
            valid_indices = [sample_id_to_idx[sid] for sid in valid_sample_ids]
            X_router = X_router[valid_indices]
            
            X_train = X_router[train_idx]
            X_test = X_router[test_idx]
            
            # Train router probe
            router_result = train_probe(
                X_train, y_train,
                X_test, y_test,
                task="intent",
                probe_target="router_logits",
                layer=layer,
                pooling=pooling,
                class_labels=DIALOGUE_ACT_LABELS,
            )
            results.append(router_result)
            print(f"    Router AUC: {router_result.auc:.3f}, Acc: {router_result.accuracy:.3f}")
            
            # Get pooled residual stream for comparison
            X_residual = cache.get_pooled_residual_stream(layer, method=pooling)
            X_residual = X_residual[valid_indices]
            
            X_train_res = X_residual[train_idx]
            X_test_res = X_residual[test_idx]
            
            # Train residual probe
            residual_result = train_probe(
                X_train_res, y_train,
                X_test_res, y_test,
                task="intent",
                probe_target="residual_stream",
                layer=layer,
                pooling=pooling,
                class_labels=DIALOGUE_ACT_LABELS,
            )
            results.append(residual_result)
            print(f"    Residual AUC: {residual_result.auc:.3f}, Acc: {residual_result.accuracy:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run intent probes on DailyDialog")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 15],
                        help="Which layers to probe")
    parser.add_argument("--pooling", nargs="+", default=["mean"],
                        help="Pooling methods to test")
    parser.add_argument("--output-dir", type=str, default="results/tables",
                        help="Where to save probe results")
    parser.add_argument("--cache-dir", type=str, default="results/extractions",
                        help="Where to find/save extraction caches")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore existing cache, force re-extraction")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Which split to probe")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (for testing with fixtures)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Phase 1: Intent Classification Probes")
    print("=" * 60)
    print(f"  Layers: {args.layers}")
    print(f"  Pooling: {args.pooling}")
    print(f"  Split: {args.split}")
    
    # =========================================================================
    # Step 1: Load DailyDialog
    # =========================================================================
    print("\n[1/4] Loading DailyDialog...")
    
    try:
        data_dir = Path(args.data_dir) if args.data_dir else None
        turns, stats = load_dailydialog(split=args.split, data_dir=data_dir, verbose=True)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        return 1
    
    # Limit samples if requested (for testing)
    if args.max_samples and len(turns) > args.max_samples:
        print(f"  Limiting to {args.max_samples} samples (from {len(turns)})")
        turns = turns[:args.max_samples]
    
    # =========================================================================
    # Step 2: Check for cached extractions
    # =========================================================================
    print("\n[2/4] Checking for cached extractions...")
    
    cache_path = None
    if not args.no_cache:
        cache_path = find_latest_cache("dailydialog", args.split, cache_dir)
    
    if cache_path:
        print(f"  Found cache: {cache_path}")
        cache = load_extraction_cache(cache_path)
        print(f"  Loaded {cache.n_samples} samples, {cache.n_layers} layers")
    else:
        print("  No cache found. You need to run extraction first.")
        print("\n  To extract router logits:")
        print("  1. Prepare the input file:")
        
        input_file = cache_dir / f"dailydialog_{args.split}_input.json"
        prepare_extraction_input(turns, input_file)
        
        print(f"\n  2. Run Modal extraction:")
        print(f"     modal run src/routing/modal_app.py \\")
        print(f"       --input {input_file} \\")
        print(f"       --dataset dailydialog \\")
        print(f"       --split {args.split}")
        print(f"\n  3. Re-run this script:")
        print(f"     python scripts/run_intent_probes.py --split {args.split}")
        return 1
    
    # Verify layers are in cache
    missing_layers = [l for l in args.layers if l not in cache.layers]
    if missing_layers:
        print(f"  Warning: Layers {missing_layers} not in cache. Available: {cache.layers}")
        args.layers = [l for l in args.layers if l in cache.layers]
        if not args.layers:
            print("  ERROR: No valid layers to probe.")
            return 1
    
    # =========================================================================
    # Step 3: Train probes
    # =========================================================================
    print("\n[3/4] Training intent probes...")
    
    results = run_intent_probes(
        cache=cache,
        turns=turns,
        layers=args.layers,
        pooling_methods=args.pooling,
        split=args.split,
    )
    
    # =========================================================================
    # Step 4: Save results
    # =========================================================================
    print("\n[4/4] Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"intent_probes_{args.split}_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print(f"  Saved to {output_file}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Target':<20} {'Layer':<8} {'Pooling':<10} {'AUC':<8} {'Accuracy':<10}")
    print("-" * 56)
    
    for r in results:
        print(f"{r.probe_target:<20} {r.layer:<8} {r.pooling:<10} {r.auc:<8.3f} {r.accuracy:<10.3f}")
    
    # Check minimum bar
    router_aucs = [r.auc for r in results if r.probe_target == "router_logits"]
    if router_aucs:
        best_router_auc = max(router_aucs)
        print(f"\nBest router AUC: {best_router_auc:.3f}")
        if best_router_auc >= 0.75:
            print("✓ Meets minimum bar (AUC ≥ 0.75)")
        else:
            print("✗ Below minimum bar (AUC < 0.75)")
    
    return 0


if __name__ == "__main__":
    exit(main())
