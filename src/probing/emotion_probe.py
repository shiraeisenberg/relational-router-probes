"""Emotion classification probing.

Probes router logits for 7-class emotion classification:
- neutral, anger, disgust, fear, happiness, sadness, surprise

Uses DailyDialog and MELD datasets.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from .linear_probe import train_probe, ProbeResult


EMOTION_LABELS = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def train_emotion_probe(
    router_logits: np.ndarray,
    emotion_labels: np.ndarray,
    layer: int,
    pooling: str = "mean",
    test_size: float = 0.2,
    random_state: int = 42
) -> ProbeResult:
    """Train emotion classification probe on router logits.
    
    Args:
        router_logits: Pooled router logits, shape (n_samples, 64)
        emotion_labels: Emotion labels, shape (n_samples,)
        layer: Layer number
        pooling: Pooling method used
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        ProbeResult with all metrics
    """
    if emotion_labels.dtype == object:
        label_to_idx = {label: i for i, label in enumerate(EMOTION_LABELS)}
        emotion_labels = np.array([label_to_idx[l] for l in emotion_labels])
    
    X_train, X_test, y_train, y_test = train_test_split(
        router_logits, emotion_labels,
        test_size=test_size, random_state=random_state, stratify=emotion_labels
    )
    
    return train_probe(
        X_train, y_train, X_test, y_test,
        task="emotion", probe_target="router_logits",
        layer=layer, pooling=pooling, class_labels=EMOTION_LABELS
    )
