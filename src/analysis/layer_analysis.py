"""Layer-by-layer analysis."""

import numpy as np
from typing import Callable


def analyze_layer_by_layer(
    extract_fn: Callable,
    train_fn: Callable,
    layers: list[int] = [0, 4, 8, 12, 15],
) -> dict[int, dict]:
    """Run probes across multiple layers."""
    results = {}
    for layer in layers:
        results[layer] = train_fn(layer=layer)
    return results
