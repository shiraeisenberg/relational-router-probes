"""Generic linear probe training utilities.

This module provides the core probe training and evaluation logic
used by all signal-specific probing modules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


@dataclass
class ProbeResult:
    """Result of training a linear probe."""
    task: str                           # intent, emotion, formality, power, etc.
    probe_target: str                   # router_logits, residual
    layer: int
    pooling: str
    auc: float
    accuracy: float
    f1_macro: float
    confusion_matrix: np.ndarray
    n_train: int
    n_test: int
    trained_at: datetime = field(default_factory=datetime.now)
    class_labels: Optional[list[str]] = None
    probe: Optional[Any] = None         # Trained probe (for saving)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task": self.task,
            "probe_target": self.probe_target,
            "layer": self.layer,
            "pooling": self.pooling,
            "auc": self.auc,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "trained_at": self.trained_at.isoformat(),
            "class_labels": self.class_labels,
        }


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "unknown",
    probe_target: str = "router_logits",
    layer: int = 0,
    pooling: str = "mean",
    class_labels: Optional[list[str]] = None,
    max_iter: int = 1000,
    random_state: int = 42
) -> ProbeResult:
    """Train a linear probe and evaluate.
    
    Args:
        X_train: Training features, shape (n_train, n_features)
        y_train: Training labels, shape (n_train,)
        X_test: Test features, shape (n_test, n_features)
        y_test: Test labels, shape (n_test,)
        task: Task name for logging
        probe_target: What we're probing (router_logits, residual)
        layer: Layer number
        pooling: Pooling method used
        class_labels: Optional list of class label names
        max_iter: Maximum iterations for LogisticRegression
        random_state: Random seed
        
    Returns:
        ProbeResult with all metrics
    """
    n_classes = len(np.unique(y_train))
    multiclass = n_classes > 2
    
    # Train probe
    # Note: multi_class parameter is deprecated in sklearn 1.5+, using default (auto->multinomial)
    probe = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state
    )
    probe.fit(X_train, y_train)
    
    # Predictions
    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)
    
    # Compute AUC
    if multiclass:
        # Multi-class AUC (one-vs-rest)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            # If some classes not in test set
            auc = 0.0
    else:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    
    # Other metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return ProbeResult(
        task=task,
        probe_target=probe_target,
        layer=layer,
        pooling=pooling,
        auc=auc,
        accuracy=accuracy,
        f1_macro=f1,
        confusion_matrix=conf_matrix,
        n_train=len(y_train),
        n_test=len(y_test),
        class_labels=class_labels,
        probe=probe,
    )


def compare_probes(
    router_result: ProbeResult,
    residual_result: ProbeResult
) -> dict:
    """Compare router probe to residual stream baseline.
    
    Args:
        router_result: Result from router logit probe
        residual_result: Result from residual stream probe
        
    Returns:
        Comparison dict with deltas and ratios
    """
    return {
        "task": router_result.task,
        "layer": router_result.layer,
        "pooling": router_result.pooling,
        "router_auc": router_result.auc,
        "residual_auc": residual_result.auc,
        "auc_delta": router_result.auc - residual_result.auc,
        "router_accuracy": router_result.accuracy,
        "residual_accuracy": residual_result.accuracy,
        "accuracy_delta": router_result.accuracy - residual_result.accuracy,
        "compression_ratio": 2048 / 64,  # residual_dim / router_dim
    }
