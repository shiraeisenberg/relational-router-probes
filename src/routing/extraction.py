"""Router logit extraction from OLMoE.

This module provides functions to load OLMoE with forward hooks
that capture router logits at each MoE layer.

OLMoE-1B-7B architecture:
- 16 MoE transformer layers
- 64 experts per layer
- Router location: model.model.layers[i].mlp.gate
- Router logits shape: (batch, seq_len, 64)
"""

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RouterLogits:
    """Extracted router logits for a text sample."""
    sample_id: str
    layer: int
    token_logits: torch.Tensor    # shape: (seq_len, 64)
    pooled_logits: torch.Tensor   # shape: (64,) after aggregation
    pooling_method: str           # mean, max, last
    n_tokens: int


def load_olmoe_with_hooks(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    model_name: str = "allenai/OLMoE-1B-7B-0924"
) -> tuple:
    """Load OLMoE with forward hooks to capture router logits.
    
    Args:
        device: Device to load model on
        dtype: Data type for model weights
        model_name: HuggingFace model name
        
    Returns:
        Tuple of (model, tokenizer, captured_dict)
        captured_dict will be populated with router logits after forward pass
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    captured = {}
    
    def make_hook(layer_idx: int):
        def hook(module, input, output):
            # output is router logits: (batch, seq_len, n_experts)
            captured[layer_idx] = output.detach().cpu()
        return hook
    
    # Register hooks on router gates
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            layer.mlp.gate.register_forward_hook(make_hook(idx))
    
    return model, tokenizer, captured


def extract_router_logits(
    model,
    tokenizer,
    captured: dict,
    texts: list[str],
    layers: list[int] = [4, 8, 12, 15],
    batch_size: int = 1,
    max_length: int = 512
) -> dict[str, dict[int, torch.Tensor]]:
    """Extract router logits for a batch of texts.
    
    Args:
        model: Loaded OLMoE model
        tokenizer: Tokenizer
        captured: Dict that hooks populate with logits
        texts: List of texts to process
        layers: Which layers to extract from
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        Dict mapping text → {layer: logits tensor (seq_len, 64)}
    """
    results = {}
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(model.device)
            
            # Forward pass populates captured dict
            _ = model(**inputs)
            
            # Extract logits for requested layers
            for j, text in enumerate(batch_texts):
                results[text] = {}
                for layer in layers:
                    if layer in captured:
                        # Get logits for this sample (handle batching)
                        if len(batch_texts) == 1:
                            logits = captured[layer][0]  # (seq_len, 64)
                        else:
                            logits = captured[layer][j]  # (seq_len, 64)
                        results[text][layer] = logits.clone()
            
            # Clear captured for next batch
            captured.clear()
    
    return results


def extract_residual_stream(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int] = [4, 8, 12, 15],
    batch_size: int = 1,
    max_length: int = 512
) -> dict[str, dict[int, torch.Tensor]]:
    """Extract residual stream activations for baseline comparison.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        texts: List of texts to process
        layers: Which layers to extract from
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        Dict mapping text → {layer: hidden states tensor (seq_len, 2048)}
    """
    results = {}
    captured_residual = {}
    
    def make_residual_hook(layer_idx: int):
        def hook(module, input, output):
            # output[0] is hidden states: (batch, seq_len, hidden_dim)
            captured_residual[layer_idx] = output[0].detach().cpu()
        return hook
    
    # Register hooks on layer outputs
    hooks = []
    for idx in layers:
        hook = model.model.layers[idx].register_forward_hook(make_residual_hook(idx))
        hooks.append(hook)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(model.device)
            
            _ = model(**inputs)
            
            for j, text in enumerate(batch_texts):
                results[text] = {}
                for layer in layers:
                    if layer in captured_residual:
                        if len(batch_texts) == 1:
                            hidden = captured_residual[layer][0]
                        else:
                            hidden = captured_residual[layer][j]
                        results[text][layer] = hidden.clone()
            
            captured_residual.clear()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return results
