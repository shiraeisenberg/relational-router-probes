"""Power differential probing.

Probes router logits for speaker status classification:
- high-status (admin)
- low-status (non-admin)

Uses Wikipedia Talk Pages dataset.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from .linear_probe import train_probe, ProbeResult


POWER_LABELS = ["low_status", "high_status"]


def train_power_probe(
    router_logits: np.ndarray,
    is_admin: np.ndarray,
    layer: int,
    pooling: str = "mean",
    test_size: float = 0.2,
    random_state: int = 42
) -> ProbeResult:
    """Train power differential probe on router logits.
    
    Args:
        router_logits: Pooled router logits, shape (n_samples, 64)
        is_admin: Boolean admin status, shape (n_samples,)
        layer: Layer number
        pooling: Pooling method used
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        ProbeResult with all metrics
    """
    power_labels = is_admin.astype(int)  # 0=low, 1=high
    
    X_train, X_test, y_train, y_test = train_test_split(
        router_logits, power_labels,
        test_size=test_size, random_state=random_state, stratify=power_labels
    )
    
    return train_probe(
        X_train, y_train, X_test, y_test,
        task="power", probe_target="router_logits",
        layer=layer, pooling=pooling, class_labels=POWER_LABELS
    )
