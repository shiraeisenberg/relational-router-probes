#!/usr/bin/env python3
"""Generate figures for writeup."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--results-dir", type=str, default="results/tables")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    args = parser.parse_args()
    
    print("Generating figures...")
    print("Not yet implemented. Run after completing probing experiments.")

if __name__ == "__main__":
    main()
