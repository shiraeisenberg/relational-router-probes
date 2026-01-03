"""Intent classification probing.

Probes router logits for dialogue act classification:
- inform: Providing information
- question: Asking for information
- directive: Requesting action
- commissive: Committing to action

Uses DailyDialog dataset.
"""

import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split

from .linear_probe import train_probe, ProbeResult


INTENT_LABELS = ["inform", "question", "directive", "commissive"]


def train_intent_probe(
    router_logits: np.ndarray,
    intent_labels: np.ndarray,
    layer: int,
    pooling: str = "mean",
    test_size: float = 0.2,
    random_state: int = 42
) -> ProbeResult:
    """Train intent classification probe on router logits.
    
    Args:
        router_logits: Pooled router logits, shape (n_samples, 64)
        intent_labels: Intent labels (0-3 or string), shape (n_samples,)
        layer: Layer number
        pooling: Pooling method used
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        ProbeResult with all metrics
    """
    # Convert string labels to integers if needed
    if intent_labels.dtype == object:
        label_to_idx = {label: i for i, label in enumerate(INTENT_LABELS)}
        intent_labels = np.array([label_to_idx[l] for l in intent_labels])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        router_logits,
        intent_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=intent_labels
    )
    
    # Train probe
    result = train_probe(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task="intent",
        probe_target="router_logits",
        layer=layer,
        pooling=pooling,
        class_labels=INTENT_LABELS,
    )
    
    return result


def train_intent_probe_residual(
    residual_stream: np.ndarray,
    intent_labels: np.ndarray,
    layer: int,
    pooling: str = "mean",
    test_size: float = 0.2,
    random_state: int = 42
) -> ProbeResult:
    """Train intent probe on residual stream (baseline).
    
    Args:
        residual_stream: Pooled residual stream, shape (n_samples, 2048)
        intent_labels: Intent labels, shape (n_samples,)
        layer: Layer number
        pooling: Pooling method used
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        ProbeResult with all metrics
    """
    # Convert string labels to integers if needed
    if intent_labels.dtype == object:
        label_to_idx = {label: i for i, label in enumerate(INTENT_LABELS)}
        intent_labels = np.array([label_to_idx[l] for l in intent_labels])
    
    X_train, X_test, y_train, y_test = train_test_split(
        residual_stream,
        intent_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=intent_labels
    )
    
    result = train_probe(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task="intent",
        probe_target="residual",
        layer=layer,
        pooling=pooling,
        class_labels=INTENT_LABELS,
    )
    
    return result
