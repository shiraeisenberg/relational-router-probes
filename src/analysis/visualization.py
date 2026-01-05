"""
Visualization module for MoE Router Probe results.

Generates publication-ready figures for the Substack post including:
- Cross-signal comparison bar chart (with Enron power results)
- Layer-wise AUC plots (2x3 grid)
- Compression ratio visualization
- Wikipedia vs Enron comparison
- Results summary table
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional

# Set style for publication-quality figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color scheme
COLORS = {
    'router': '#2ecc71',        # Green for router logits
    'router_light': '#a9dfbf',  # Light green for inconclusive
    'residual': '#3498db',      # Blue for residual stream
    'residual_light': '#aed6f1', # Light blue for inconclusive
    'threshold': '#e74c3c',     # Red for success threshold
    'baseline': '#95a5a6',      # Gray for random baseline
    'success': '#27ae60',       # Darker green for success
    'warning': '#f39c12',       # Orange for warning
    'failure': '#c0392b',       # Dark red for failure
}

# Complete results data
RESULTS = {
    "Intent": {
        "router": 0.841, 
        "residual": 0.877, 
        "inconclusive": False,
        "layers": {4: (0.841, 0.801), 8: (0.841, 0.876), 12: (0.753, 0.877), 15: (0.731, 0.863)}
    },
    "Emotion": {
        "router": 0.879, 
        "residual": 0.938, 
        "inconclusive": False,
        "layers": {4: (0.863, 0.801), 8: (0.879, 0.874), 12: (0.776, 0.938), 15: (0.768, 0.889)}
    },
    "Power (Wikipedia)": {
        "router": 0.608, 
        "residual": 0.677, 
        "inconclusive": True,
        "layers": {4: (0.608, 0.631), 8: (0.587, 0.666), 12: (0.594, 0.677), 15: (0.588, 0.668)}
    },
    "Power (Enron)": {
        "router": 0.755, 
        "residual": 0.929, 
        "inconclusive": False,
        "layers": {4: (0.641, 0.595), 8: (0.709, 0.656), 12: (0.699, 0.854), 15: (0.755, 0.929)}
    },
    "Tension": {
        "router": 0.995, 
        "residual": 1.000, 
        "inconclusive": False,
        "layers": {4: (0.991, 0.947), 8: (0.995, 0.986), 12: (0.983, 1.000), 15: (0.978, 1.000)}
    },
}


def create_cross_signal_comparison(
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Figure 1: Cross-Signal Comparison Bar Chart
    
    Shows router vs residual AUC for all signals including:
    - Power (Wikipedia) with "noisy labels" annotation
    - Power (Enron) with "power exercised" annotation
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    signals = list(RESULTS.keys())
    x = np.arange(len(signals))
    width = 0.35
    
    router_vals = [RESULTS[s]["router"] for s in signals]
    residual_vals = [RESULTS[s]["residual"] for s in signals]
    inconclusive = [RESULTS[s].get("inconclusive", False) for s in signals]
    
    # Create bars with hatching for inconclusive results
    for i, (signal, router_val, residual_val, is_inconclusive) in enumerate(
        zip(signals, router_vals, residual_vals, inconclusive)
    ):
        router_color = COLORS['router_light'] if is_inconclusive else COLORS['router']
        residual_color = COLORS['residual_light'] if is_inconclusive else COLORS['residual']
        hatch = '//' if is_inconclusive else ''
        edge_color = '#666666' if is_inconclusive else None
        
        ax.bar(x[i] - width/2, router_val, width, 
               color=router_color, hatch=hatch, edgecolor=edge_color or router_color,
               linewidth=1.5 if is_inconclusive else 0)
        ax.bar(x[i] + width/2, residual_val, width,
               color=residual_color, hatch=hatch, edgecolor=edge_color or residual_color,
               linewidth=1.5 if is_inconclusive else 0)
    
    # Add threshold lines
    ax.axhline(y=0.65, color=COLORS['threshold'], linestyle='--', linewidth=2, 
               label='Success threshold (0.65)', zorder=1)
    ax.axhline(y=0.50, color=COLORS['baseline'], linestyle=':', linewidth=1.5,
               label='Random baseline (0.50)', zorder=1)
    
    # Add value labels on bars
    for i, (router_val, residual_val) in enumerate(zip(router_vals, residual_vals)):
        ax.text(x[i] - width/2, router_val + 0.02, f'{router_val:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(x[i] + width/2, residual_val + 0.02, f'{residual_val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add annotations for Wikipedia and Enron
    wiki_idx = signals.index("Power (Wikipedia)")
    enron_idx = signals.index("Power (Enron)")
    
    ax.annotate('noisy labels', xy=(x[wiki_idx], 0.55), fontsize=9,
                ha='center', va='top', style='italic', color='#666666')
    ax.annotate('power exercised', xy=(x[enron_idx], 0.55), fontsize=9,
                ha='center', va='top', style='italic', color='#27ae60')
    
    # Labels and formatting
    ax.set_ylabel('AUC (One-vs-Rest)')
    ax.set_title('MoE Router Logits Encode Content AND Power-in-Action', 
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(signals, fontsize=11)
    ax.set_ylim(0.4, 1.08)
    
    # Custom legend
    router_patch = mpatches.Patch(color=COLORS['router'], label='Router Logits (64-dim)')
    residual_patch = mpatches.Patch(color=COLORS['residual'], label='Residual Stream (2048-dim)')
    inconclusive_patch = mpatches.Patch(facecolor=COLORS['router_light'], 
                                         edgecolor='#666666', hatch='//',
                                         label='Inconclusive (noisy labels)')
    threshold_line = plt.Line2D([0], [0], color=COLORS['threshold'], linestyle='--', 
                                 linewidth=2, label='Success threshold (0.65)')
    baseline_line = plt.Line2D([0], [0], color=COLORS['baseline'], linestyle=':',
                                linewidth=1.5, label='Random baseline (0.50)')
    
    ax.legend(handles=[router_patch, residual_patch, inconclusive_patch, 
                       threshold_line, baseline_line],
              loc='upper right', framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_layerwise_auc(
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 9)
) -> plt.Figure:
    """
    Figure 2: Layer-wise AUC (2x3 grid)
    
    Shows how signal encoding changes across layers for each signal type.
    Includes new Power (Enron) subplot.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    signals = list(RESULTS.keys())
    layers = [4, 8, 12, 15]
    
    for idx, signal in enumerate(signals):
        ax = axes[idx]
        data = RESULTS[signal]
        is_inconclusive = data.get("inconclusive", False)
        
        router_auc = [data["layers"][l][0] for l in layers]
        residual_auc = [data["layers"][l][1] for l in layers]
        
        # Line styles
        router_style = '--' if is_inconclusive else '-'
        residual_style = '--' if is_inconclusive else '-'
        router_alpha = 0.6 if is_inconclusive else 1.0
        residual_alpha = 0.6 if is_inconclusive else 1.0
        
        ax.plot(layers, router_auc, marker='o', color=COLORS['router'], 
                linewidth=2, markersize=8, label='Router', 
                linestyle=router_style, alpha=router_alpha)
        ax.plot(layers, residual_auc, marker='s', color=COLORS['residual'],
                linewidth=2, markersize=8, label='Residual',
                linestyle=residual_style, alpha=residual_alpha)
        
        # Add threshold line
        ax.axhline(y=0.65, color=COLORS['threshold'], linestyle='--', 
                   linewidth=1, alpha=0.7)
        ax.axhline(y=0.50, color=COLORS['baseline'], linestyle=':', 
                   linewidth=1, alpha=0.5)
        
        # Title with annotation for inconclusive
        title = signal
        if is_inconclusive:
            title += " (inconclusive)"
        ax.set_title(title, fontweight='bold')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Validation AUC')
        ax.set_xticks(layers)
        ax.set_ylim(0.45, 1.05)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Highlight best layer
        best_router_layer = layers[np.argmax(router_auc)]
        best_router_val = max(router_auc)
        ax.annotate(f'L{best_router_layer}', 
                    xy=(best_router_layer, best_router_val),
                    xytext=(best_router_layer + 0.5, best_router_val + 0.03),
                    fontsize=8, color=COLORS['router'], fontweight='bold')
    
    # Hide the 6th subplot (we only have 5 signals)
    axes[5].set_visible(False)
    
    # Add overall title
    fig.suptitle('Layer-wise AUC: Signal Encoding Across Transformer Depth', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_compression_ratio(
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Figure 3: Compression Ratio Visualization
    
    Shows 32× compression with ~95% signal retention.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate retention percentages
    signals = [s for s in RESULTS.keys() if not RESULTS[s].get("inconclusive", False)]
    retentions = []
    for signal in signals:
        router = RESULTS[signal]["router"]
        residual = RESULTS[signal]["residual"]
        retention = (router / residual) * 100
        retentions.append(retention)
    
    x = np.arange(len(signals))
    bars = ax.bar(x, retentions, color=COLORS['router'], edgecolor='white', linewidth=1.5)
    
    # Add 100% line
    ax.axhline(y=100, color=COLORS['residual'], linestyle='-', linewidth=2,
               label='Residual Stream (baseline)')
    
    # Add value labels
    for i, (bar, ret) in enumerate(zip(bars, retentions)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{ret:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('Signal Retention (%)')
    ax.set_title('32× Compression with ~95% Signal Retention\n(Router: 64-dim vs Residual: 2048-dim)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(signals, fontsize=11)
    ax.set_ylim(0, 115)
    
    # Add compression annotation
    ax.annotate('32× fewer dimensions', 
                xy=(0.5, 0.15), xycoords='axes fraction',
                fontsize=16, fontweight='bold', color=COLORS['router'],
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=COLORS['router'], linewidth=2))
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_wikipedia_vs_enron(
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Figure 4: Wikipedia vs Enron Comparison
    
    Side-by-side comparison showing the importance of operationalization.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    datasets = ['Wikipedia Talk\n(Speaker Identity)', 'Enron\n(Power Exercise)']
    x = np.arange(len(datasets))
    width = 0.35
    
    wiki_data = RESULTS["Power (Wikipedia)"]
    enron_data = RESULTS["Power (Enron)"]
    
    router_vals = [wiki_data["router"], enron_data["router"]]
    residual_vals = [wiki_data["residual"], enron_data["residual"]]
    
    # Create bars
    bars1 = ax.bar(x - width/2, router_vals, width, color=COLORS['router'],
                   label='Router Logits (64-dim)', edgecolor='white')
    bars2 = ax.bar(x + width/2, residual_vals, width, color=COLORS['residual'],
                   label='Residual Stream (2048-dim)', edgecolor='white')
    
    # Make Wikipedia bars lighter/hatched
    bars1[0].set_facecolor(COLORS['router_light'])
    bars1[0].set_hatch('//')
    bars1[0].set_edgecolor('#666666')
    bars2[0].set_facecolor(COLORS['residual_light'])
    bars2[0].set_hatch('//')
    bars2[0].set_edgecolor('#666666')
    
    # Add threshold lines
    ax.axhline(y=0.65, color=COLORS['threshold'], linestyle='--', linewidth=2,
               label='Success threshold (0.65)')
    ax.axhline(y=0.50, color=COLORS['baseline'], linestyle=':', linewidth=1.5,
               label='Random baseline (0.50)')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add delta annotation
    delta = enron_data["router"] - wiki_data["router"]
    relative_delta = (delta / wiki_data["router"]) * 100
    
    # Draw arrow between router bars
    ax.annotate('', 
                xy=(x[1] - width/2, enron_data["router"]),
                xytext=(x[0] - width/2, wiki_data["router"]),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], 
                               lw=2, connectionstyle='arc3,rad=0.3'))
    
    ax.text(0.5, 0.72, f'+{delta:.3f} AUC (+{relative_delta:.0f}%)',
            fontsize=14, fontweight='bold', color=COLORS['success'],
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor=COLORS['success'], linewidth=2))
    
    # Formatting
    ax.set_ylabel('AUC (One-vs-Rest)', fontsize=12)
    ax.set_title('Same Architecture, Different Operationalization', 
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylim(0.4, 1.05)
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_results_table(
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 4)
) -> plt.Figure:
    """
    Figure 5: Results Summary Table as Image
    
    Creates a publication-ready table image with color coding.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Table data
    columns = ['Signal', 'Router AUC', 'Residual AUC', 'Retention', 'Hypothesis', 'Status']
    
    rows = [
        ['Intent (4-class)', '0.841', '0.877', '96%', 'H1: >=0.90', 'MET'],
        ['Emotion (7-class)', '0.879', '0.938', '94%', 'H2: 0.70-0.85', 'MET'],
        ['Power (Wikipedia)', '0.608', '0.677', '90%', 'H3: >=0.65', 'NOT MET'],
        ['Power (Enron)', '0.755', '0.929', '81%', 'H3b: >=0.70', 'MET'],
        ['Tension (3-class)', '0.995', '1.000', '99.5%', 'Exploratory', 'EXCEPTIONAL'],
    ]
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.22, 0.13, 0.15, 0.12, 0.18, 0.15]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Header styling
    for j in range(len(columns)):
        cell = table[(0, j)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    
    # Row colors based on status
    row_colors = [
        '#d5f5e3',  # Intent - light green (success)
        '#d5f5e3',  # Emotion - light green (success)
        '#fadbd8',  # Power (Wikipedia) - light red (failure)
        '#d5f5e3',  # Power (Enron) - light green (success)
        '#abebc6',  # Tension - darker green (exceptional)
    ]
    
    for i, color in enumerate(row_colors):
        for j in range(len(columns)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(color)
            # Bold the status column
            if j == 5:
                status = rows[i][j]
                if status in ['MET', 'EXCEPTIONAL']:
                    cell.set_text_props(color='#27ae60', fontweight='bold')
                else:
                    cell.set_text_props(color='#c0392b', fontweight='bold')
    
    # Add title
    ax.set_title('Cross-Signal Summary: Phase 1 & 2 Results', 
                 fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    return fig


def generate_all_figures(output_dir: Path) -> dict:
    """
    Generate all figures and save to the specified directory.
    
    Returns dict mapping figure names to paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Figure 1: Cross-Signal Comparison
    path = output_dir / 'cross_signal_comparison_v2.png'
    create_cross_signal_comparison(path)
    figures['cross_signal'] = path
    
    # Figure 2: Layer-wise AUC
    path = output_dir / 'layer_wise_auc_v2.png'
    create_layerwise_auc(path)
    figures['layerwise'] = path
    
    # Figure 3: Compression Ratio
    path = output_dir / 'compression_ratio_v2.png'
    create_compression_ratio(path)
    figures['compression'] = path
    
    # Figure 4: Wikipedia vs Enron
    path = output_dir / 'wikipedia_vs_enron.png'
    create_wikipedia_vs_enron(path)
    figures['wiki_vs_enron'] = path
    
    # Figure 5: Results Table
    path = output_dir / 'results_table_v2.png'
    create_results_table(path)
    figures['table'] = path
    
    print(f"\n✓ Generated {len(figures)} figures in {output_dir}")
    return figures


# Legacy function for backwards compatibility
def plot_auc_comparison(
    results: dict[str, float],
    output_path: str = "results/figures/auc_comparison.png"
):
    """Bar chart comparing AUC across signals (legacy function)."""
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
    """Heatmap of expert overlap between signals (legacy function)."""
    import seaborn as sns
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
    """Line plot of AUC across layers (legacy function)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    layers = sorted(layer_results.keys())
    aucs = [layer_results[l] for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, aucs, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Layer")
    plt.ylabel("AUC")
    plt.title(f"Layer-wise AUC for {signal_type}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent.parent / "results" / "figures"
    generate_all_figures(output_dir)
