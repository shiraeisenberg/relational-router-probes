#!/usr/bin/env python3
"""Generate synthetic tension/repair pairs via Claude API.

Creates ~500 two-turn dialogue exchanges labeled as:
- escalation: Speaker B increases conflict
- repair: Speaker B de-escalates
- neutral: Neither escalates nor repairs

Requires ANTHROPIC_API_KEY in .env file.

Usage:
    python scripts/generate_tension_pairs.py --n-pairs 500

Output: data/synthetic/tension_pairs.json
"""

import argparse
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Generate tension pairs via Claude")
    parser.add_argument("--n-pairs", type=int, default=500,
                        help="Total pairs to generate (~167 per class)")
    parser.add_argument("--output-dir", type=str, default="data/synthetic",
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Pairs to request per API call")
    args = parser.parse_args()
    
    from src.data.synthetic_tension import generate_tension_pairs
    
    n_per_class = args.n_pairs // 3
    
    print(f"Generating {args.n_pairs} tension pairs ({n_per_class} per class)...")
    print(f"Output: {args.output_dir}/tension_pairs.json")
    print("")
    
    pairs, stats = generate_tension_pairs(
        n_per_class=n_per_class,
        batch_size=args.batch_size,
        output_dir=Path(args.output_dir),
        verbose=True,
    )
    
    print("")
    print(f"Generated {len(pairs)} pairs successfully.")


if __name__ == "__main__":
    main()
