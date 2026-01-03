"""Relationship outcome probing.

Probes router logits for predicting SOTOPIA interaction outcomes
from early-turn routing patterns.

This is a stretch goal - token-level routing may be too local
for longitudinal relationship dynamics.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RelationshipProbeResult:
    """Result of training relationship prediction probe."""
    task: str
    probe_target: str
    layer: int
    pooling: str
    r2: float
    mse: float
    n_train: int
    n_test: int
    trained_at: datetime


def train_relationship_probe(
    router_logits: np.ndarray,
    relationship_scores: np.ndarray,
    layer: int,
    pooling: str = "mean",
    test_size: float = 0.2,
    random_state: int = 42
) -> RelationshipProbeResult:
    """Train relationship outcome prediction probe.
    
    This is a regression task predicting continuous relationship scores.
    
    Args:
        router_logits: Pooled router logits from early turns, shape (n_episodes, 64)
        relationship_scores: SOTOPIA relationship scores, shape (n_episodes,)
        layer: Layer number
        pooling: Pooling method used
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        RelationshipProbeResult with regression metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(
        router_logits, relationship_scores,
        test_size=test_size, random_state=random_state
    )
    
    # Use Ridge regression for continuous prediction
    probe = Ridge(alpha=1.0, random_state=random_state)
    probe.fit(X_train, y_train)
    
    y_pred = probe.predict(X_test)
    
    return RelationshipProbeResult(
        task="relationship",
        probe_target="router_logits",
        layer=layer,
        pooling=pooling,
        r2=r2_score(y_test, y_pred),
        mse=mean_squared_error(y_test, y_pred),
        n_train=len(y_train),
        n_test=len(y_test),
        trained_at=datetime.now(),
    )
