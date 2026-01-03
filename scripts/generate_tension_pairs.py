#!/usr/bin/env python3
"""Generate synthetic tension/repair pairs via Claude."""

import asyncio
import argparse
from pathlib import Path

async def main():
    parser = argparse.ArgumentParser(description="Generate tension pairs")
    parser.add_argument("--n-pairs", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    args = parser.parse_args()
    
    print(f"Generating {args.n_pairs} tension pairs...")
    print("Not yet implemented. See src/data/synthetic_tension.py")

if __name__ == "__main__":
    asyncio.run(main())
