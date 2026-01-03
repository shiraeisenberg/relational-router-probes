"""Tests for linear probing."""

import pytest
import numpy as np


def test_train_probe_binary():
    """Test binary classification probe."""
    from src.probing.linear_probe import train_probe
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 64)
    y = (X[:, 0] > 0).astype(int)  # Simple linear rule
    
    result = train_probe(
        X[:80], y[:80],  # train
        X[80:], y[80:],  # test
        task="test",
        layer=0,
    )
    
    assert result.auc > 0.5  # Better than random
    assert 0 <= result.accuracy <= 1
    assert result.n_train == 80
    assert result.n_test == 20


def test_train_probe_multiclass():
    """Test multi-class classification probe."""
    from src.probing.linear_probe import train_probe
    
    np.random.seed(42)
    X = np.random.randn(200, 64)
    y = np.random.randint(0, 4, size=200)  # 4 classes
    
    result = train_probe(
        X[:160], y[:160],
        X[160:], y[160:],
        task="test_multiclass",
        layer=0,
    )
    
    assert result.auc >= 0  # AUC defined
    assert result.confusion_matrix.shape == (4, 4)


def test_probe_result_to_dict():
    """Test result serialization."""
    from src.probing.linear_probe import train_probe
    
    np.random.seed(42)
    X = np.random.randn(100, 64)
    y = (X[:, 0] > 0).astype(int)
    
    result = train_probe(X[:80], y[:80], X[80:], y[80:])
    d = result.to_dict()
    
    assert "auc" in d
    assert "accuracy" in d
    assert "confusion_matrix" in d
    assert isinstance(d["confusion_matrix"], list)  # Serializable
