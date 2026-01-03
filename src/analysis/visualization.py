"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_auc_comparison(
    results: dict[str, float],
    output_path: str = "results/figures/auc_comparison.png"
):
    """Bar chart comparing AUC across signals."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    signals = list(results.keys())
    aucs = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(signals, aucs)
    plt.ylabel("AUC")
    plt.title("Router Logit Probe AUC by Signal Type")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_expert_overlap_heatmap(
    overlaps: dict[tuple[str, str], float],
    output_path: str = "results/figures/expert_overlap.png"
):
    """Heatmap of expert overlap between signals."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    signals = sorted(set(s for pair in overlaps.keys() for s in pair))
    n = len(signals)
    
    matrix = np.eye(n)
    for (a, b), overlap in overlaps.items():
        i, j = signals.index(a), signals.index(b)
        matrix[i, j] = overlap
        matrix[j, i] = overlap
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=signals, yticklabels=signals, 
                annot=True, cmap="YlOrRd", vmin=0, vmax=1)
    plt.title("Expert Overlap Between Signals")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_layer_analysis(
    layer_results: dict[int, float],
    signal_type: str,
    output_path: str = "results/figures/layer_analysis.png"
):
    """Line plot of AUC across layers."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    layers = sorted(layer_results.keys())
    aucs = [layer_results[l] for l in layers]
    
    plt.figure(figsize=(8, 5))
    plt.plot(layers, aucs, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("AUC")
    plt.title(f"{signal_type.capitalize()} Probe AUC by Layer")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
