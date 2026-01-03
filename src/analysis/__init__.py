"""Analysis and visualization tools.

This module provides:
- Expert cluster analysis (which experts encode which signals)
- Comparative analysis across signals and layers
- Cross-dataset transfer tests
- Baseline comparisons (BoW, punctuation)
- Visualization utilities
"""

from .expert_clusters import (
    compute_expert_associations,
    compute_expert_overlap,
    ExpertClusterAnalysis,
)
from .comparative import compare_signals, generate_comparison_table
from .layer_analysis import analyze_layer_by_layer
from .transfer import test_cross_dataset_transfer
from .baselines import run_bow_baseline, run_punctuation_baseline
from .visualization import (
    plot_auc_comparison,
    plot_expert_overlap_heatmap,
    plot_layer_analysis,
)

__all__ = [
    "compute_expert_associations",
    "compute_expert_overlap",
    "ExpertClusterAnalysis",
    "compare_signals",
    "generate_comparison_table",
    "analyze_layer_by_layer",
    "test_cross_dataset_transfer",
    "run_bow_baseline",
    "run_punctuation_baseline",
    "plot_auc_comparison",
    "plot_expert_overlap_heatmap",
    "plot_layer_analysis",
]
