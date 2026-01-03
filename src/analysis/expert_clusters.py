"""Expert cluster analysis.

Analyzes which experts are associated with which signals,
and computes overlap between expert sets across signals.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ExpertClusterAnalysis:
    """Analysis of which experts activate for a signal."""
    signal_type: str
    layer: int
    top_k_experts: list[int]
    activation_scores: dict[int, float] = field(default_factory=dict)
    overlap_with: dict[str, float] = field(default_factory=dict)


def compute_expert_associations(
    router_logits: np.ndarray,
    labels: np.ndarray,
    top_k: int = 5
) -> dict[str, dict]:
    """Find which experts are most associated with each label.
    
    Args:
        router_logits: Shape (n_samples, n_experts)
        labels: Shape (n_samples,)
        top_k: Number of top experts to return per label
        
    Returns:
        Dict mapping label → {top_experts, mean_activations}
    """
    associations = {}
    
    for label in np.unique(labels):
        mask = labels == label
        mean_logits = router_logits[mask].mean(axis=0)
        top_experts = np.argsort(mean_logits)[-top_k:][::-1]
        
        associations[str(label)] = {
            "top_experts": top_experts.tolist(),
            "mean_activations": {
                int(e): float(mean_logits[e]) for e in top_experts
            }
        }
    
    return associations


def compute_expert_overlap(
    assoc_a: dict,
    assoc_b: dict,
    top_k: int = 5
) -> float:
    """Compute Jaccard overlap between expert sets.
    
    Args:
        assoc_a: Expert associations for signal A
        assoc_b: Expert associations for signal B
        top_k: Use top-k experts from each label
        
    Returns:
        Jaccard similarity (0-1)
    """
    experts_a = set()
    experts_b = set()
    
    for label_data in assoc_a.values():
        experts_a.update(label_data["top_experts"][:top_k])
    
    for label_data in assoc_b.values():
        experts_b.update(label_data["top_experts"][:top_k])
    
    if not experts_a or not experts_b:
        return 0.0
    
    intersection = len(experts_a & experts_b)
    union = len(experts_a | experts_b)
    
    return intersection / union


def compute_all_overlaps(
    signal_associations: dict[str, dict],
    top_k: int = 5
) -> dict[tuple[str, str], float]:
    """Compute pairwise overlap between all signals.
    
    Args:
        signal_associations: Dict mapping signal_name → associations
        top_k: Use top-k experts
        
    Returns:
        Dict mapping (signal_a, signal_b) → overlap
    """
    signals = list(signal_associations.keys())
    overlaps = {}
    
    for i, sig_a in enumerate(signals):
        for sig_b in signals[i+1:]:
            overlap = compute_expert_overlap(
                signal_associations[sig_a],
                signal_associations[sig_b],
                top_k=top_k
            )
            overlaps[(sig_a, sig_b)] = overlap
    
    return overlaps
