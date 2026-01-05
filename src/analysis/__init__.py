# Analysis module for MoE Router Probe results
#
# Main visualization functions for generating publication-ready figures

from .visualization import (
    create_cross_signal_comparison,
    create_layerwise_auc,
    create_compression_ratio,
    create_wikipedia_vs_enron,
    create_results_table,
    generate_all_figures,
    RESULTS,
    COLORS,
)

__all__ = [
    'create_cross_signal_comparison',
    'create_layerwise_auc',
    'create_compression_ratio',
    'create_wikipedia_vs_enron',
    'create_results_table',
    'generate_all_figures',
    'RESULTS',
    'COLORS',
]

# Optional imports for backwards compatibility (require additional dependencies)
try:
    from .expert_clusters import (
        compute_expert_associations,
        compute_expert_overlap,
        ExpertClusterAnalysis,
    )
    from .comparative import compare_signals, generate_comparison_table
    from .layer_analysis import analyze_layer_by_layer
    from .transfer import test_cross_dataset_transfer
    from .baselines import run_bow_baseline, run_punctuation_baseline
    
    __all__.extend([
        "compute_expert_associations",
        "compute_expert_overlap",
        "ExpertClusterAnalysis",
        "compare_signals",
        "generate_comparison_table",
        "analyze_layer_by_layer",
        "test_cross_dataset_transfer",
        "run_bow_baseline",
        "run_punctuation_baseline",
    ])
except ImportError:
    # These modules require pandas/other dependencies
    pass
