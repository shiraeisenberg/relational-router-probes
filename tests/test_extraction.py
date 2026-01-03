"""Tests for router logit extraction."""

import pytest
import torch
import numpy as np


def test_pool_router_logits_mean():
    """Test mean pooling."""
    from src.routing.aggregation import pool_router_logits
    
    logits = torch.randn(10, 64)  # 10 tokens, 64 experts
    pooled = pool_router_logits(logits, method="mean")
    
    assert pooled.shape == (64,)
    assert torch.allclose(pooled, logits.mean(dim=0))


def test_pool_router_logits_max():
    """Test max pooling."""
    from src.routing.aggregation import pool_router_logits
    
    logits = torch.randn(10, 64)
    pooled = pool_router_logits(logits, method="max")
    
    assert pooled.shape == (64,)
    assert torch.allclose(pooled, logits.max(dim=0).values)


def test_pool_router_logits_last():
    """Test last-token pooling."""
    from src.routing.aggregation import pool_router_logits
    
    logits = torch.randn(10, 64)
    pooled = pool_router_logits(logits, method="last")
    
    assert pooled.shape == (64,)
    assert torch.allclose(pooled, logits[-1])


def test_pool_router_logits_numpy():
    """Test numpy input."""
    from src.routing.aggregation import pool_router_logits
    
    logits = np.random.randn(10, 64)
    pooled = pool_router_logits(logits, method="mean")
    
    assert isinstance(pooled, np.ndarray)
    assert pooled.shape == (64,)
