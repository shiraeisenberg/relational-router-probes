"""Router logit extraction and aggregation.

This module provides tools for:
- Loading OLMoE with forward hooks to capture router logits
- Extracting router logits for batches of text
- Aggregating token-level logits to example-level representations

The router logits are 64-dimensional (one per expert) and provide
a compressed representation of routing decisions.
"""

from .extraction import (
    load_olmoe_with_hooks,
    extract_router_logits,
    extract_residual_stream,
    RouterLogits,
)
from .aggregation import (
    pool_router_logits,
    pool_residual_stream,
)

__all__ = [
    "load_olmoe_with_hooks",
    "extract_router_logits",
    "extract_residual_stream",
    "RouterLogits",
    "pool_router_logits",
    "pool_residual_stream",
]
