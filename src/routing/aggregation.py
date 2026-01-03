"""Token-to-example aggregation for router logits.

Router logits are token-level (seq_len, n_experts). For classification,
we need to aggregate to example-level (n_experts,) representations.

Pooling strategies:
- mean: Average over all tokens (default, recommended)
- max: Maximum over all tokens (captures strongest routing)
- last: Last token only (autoregressive prediction)
"""

import torch
import numpy as np
from typing import Union


def pool_router_logits(
    token_logits: Union[torch.Tensor, np.ndarray],
    method: str = "mean",
    exclude_padding: bool = True,
    attention_mask: Union[torch.Tensor, np.ndarray, None] = None
) -> Union[torch.Tensor, np.ndarray]:
    """Pool token-level router logits to example-level.
    
    Args:
        token_logits: Shape (seq_len, n_experts) or (batch, seq_len, n_experts)
        method: Pooling method - "mean", "max", or "last"
        exclude_padding: Whether to exclude padding tokens (requires attention_mask)
        attention_mask: Binary mask (1 for real tokens, 0 for padding)
        
    Returns:
        Pooled logits of shape (n_experts,) or (batch, n_experts)
    """
    is_numpy = isinstance(token_logits, np.ndarray)
    if is_numpy:
        token_logits = torch.from_numpy(token_logits)
    
    # Handle 2D (single example) vs 3D (batch)
    if token_logits.dim() == 2:
        token_logits = token_logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, seq_len, n_experts = token_logits.shape
    
    if exclude_padding and attention_mask is not None:
        if isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask)
        # Expand mask for broadcasting
        mask = attention_mask.unsqueeze(-1).expand_as(token_logits)
        token_logits = token_logits * mask
    
    if method == "mean":
        if exclude_padding and attention_mask is not None:
            # Compute mean over non-padding tokens
            sum_logits = token_logits.sum(dim=1)
            n_tokens = attention_mask.sum(dim=1, keepdim=True).expand_as(sum_logits)
            pooled = sum_logits / n_tokens.clamp(min=1)
        else:
            pooled = token_logits.mean(dim=1)
            
    elif method == "max":
        pooled = token_logits.max(dim=1).values
        
    elif method == "last":
        if exclude_padding and attention_mask is not None:
            # Get last non-padding token
            last_idx = attention_mask.sum(dim=1) - 1
            pooled = token_logits[torch.arange(batch_size), last_idx]
        else:
            pooled = token_logits[:, -1, :]
            
    else:
        raise ValueError(f"Unknown pooling method: {method}. Use 'mean', 'max', or 'last'.")
    
    if squeeze_output:
        pooled = pooled.squeeze(0)
    
    if is_numpy:
        pooled = pooled.numpy()
    
    return pooled


def pool_residual_stream(
    hidden_states: Union[torch.Tensor, np.ndarray],
    method: str = "mean",
    attention_mask: Union[torch.Tensor, np.ndarray, None] = None
) -> Union[torch.Tensor, np.ndarray]:
    """Pool token-level hidden states to example-level.
    
    Same interface as pool_router_logits but for residual stream.
    
    Args:
        hidden_states: Shape (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        method: Pooling method
        attention_mask: Binary mask for padding exclusion
        
    Returns:
        Pooled hidden states of shape (hidden_dim,) or (batch, hidden_dim)
    """
    return pool_router_logits(hidden_states, method=method, attention_mask=attention_mask)


def batch_pool_logits(
    all_logits: dict[str, dict[int, torch.Tensor]],
    layers: list[int],
    method: str = "mean"
) -> dict[str, dict[int, np.ndarray]]:
    """Pool logits for all samples across specified layers.
    
    Args:
        all_logits: Dict mapping sample_id → {layer: token_logits}
        layers: Which layers to pool
        method: Pooling method
        
    Returns:
        Dict mapping sample_id → {layer: pooled_logits (n_experts,)}
    """
    results = {}
    
    for sample_id, layer_logits in all_logits.items():
        results[sample_id] = {}
        for layer in layers:
            if layer in layer_logits:
                token_logits = layer_logits[layer]
                pooled = pool_router_logits(token_logits, method=method)
                results[sample_id][layer] = pooled.numpy() if isinstance(pooled, torch.Tensor) else pooled
    
    return results
