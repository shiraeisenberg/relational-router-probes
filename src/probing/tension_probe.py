"""Tension dynamics probing.

Probes router logits for tension classification:
- escalation: Increases interpersonal tension
- repair: De-escalates and restores warmth
- neutral: Neither escalates nor repairs

Uses synthetic tension pairs.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from .linear_probe import train_probe, ProbeResult


TENSION_LABELS = ["escalation", "repair", "neutral"]


def train_tension_probe(
    router_logits: np.ndarray,
    tension_labels: np.ndarray,
    layer: int,
    pooling: str = "mean",
    test_size: float = 0.2,
    random_state: int = 42
) -> ProbeResult:
    """Train tension classification probe on router logits.
    
    Args:
        router_logits: Pooled router logits, shape (n_samples, 64)
        tension_labels: Tension labels, shape (n_samples,)
        layer: Layer number
        pooling: Pooling method used
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        ProbeResult with all metrics
    """
    if tension_labels.dtype == object:
        label_to_idx = {label: i for i, label in enumerate(TENSION_LABELS)}
        tension_labels = np.array([label_to_idx[l] for l in tension_labels])
    
    X_train, X_test, y_train, y_test = train_test_split(
        router_logits, tension_labels,
        test_size=test_size, random_state=random_state, stratify=tension_labels
    )
    
    return train_probe(
        X_train, y_train, X_test, y_test,
        task="tension", probe_target="router_logits",
        layer=layer, pooling=pooling, class_labels=TENSION_LABELS
    )
