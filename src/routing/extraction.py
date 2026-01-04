"""Router logit extraction from OLMoE.

This module provides functions to load OLMoE with forward hooks
that capture router logits at each MoE layer, plus caching utilities.

OLMoE-1B-7B architecture:
- 16 MoE transformer layers
- 64 experts per layer
- Router location: model.model.layers[i].mlp.gate
- Router logits shape: (seq_len, 64)

Cache format (.npz):
- router_logits: (n_samples, n_layers, max_seq_len, 64)
- residual_stream: (n_samples, n_layers, max_seq_len, 2048)
- token_counts: (n_samples,) - actual sequence lengths for masking
- metadata: JSON string with extraction info
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import json
import numpy as np
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


# =============================================================================
# Extraction Cache
# =============================================================================

@dataclass
class ExtractionCache:
    """Cached extraction results with metadata."""
    router_logits: np.ndarray        # (n_samples, n_layers, max_seq_len, 64)
    residual_stream: np.ndarray      # (n_samples, n_layers, max_seq_len, 2048)
    token_counts: np.ndarray         # (n_samples,) actual lengths
    sample_ids: list[str]
    layers: list[int]
    metadata: dict
    
    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)
    
    @property
    def n_layers(self) -> int:
        return len(self.layers)
    
    @property
    def max_seq_len(self) -> int:
        return self.router_logits.shape[2]
    
    def get_router_logits(
        self,
        sample_idx: int,
        layer: int,
        masked: bool = True
    ) -> np.ndarray:
        """Get router logits for a sample at a specific layer.
        
        Args:
            sample_idx: Index of sample
            layer: Layer number
            masked: If True, return only up to actual token count
            
        Returns:
            Router logits of shape (seq_len, 64) or (actual_len, 64) if masked
        """
        layer_idx = self.layers.index(layer)
        logits = self.router_logits[sample_idx, layer_idx]
        
        if masked:
            n_tokens = self.token_counts[sample_idx]
            return logits[:n_tokens]
        return logits
    
    def get_residual_stream(
        self,
        sample_idx: int,
        layer: int,
        masked: bool = True
    ) -> np.ndarray:
        """Get residual stream for a sample at a specific layer.
        
        Args:
            sample_idx: Index of sample  
            layer: Layer number
            masked: If True, return only up to actual token count
            
        Returns:
            Hidden states of shape (seq_len, 2048) or (actual_len, 2048) if masked
        """
        layer_idx = self.layers.index(layer)
        hidden = self.residual_stream[sample_idx, layer_idx]
        
        if masked:
            n_tokens = self.token_counts[sample_idx]
            return hidden[:n_tokens]
        return hidden
    
    def get_pooled_router_logits(
        self,
        layer: int,
        method: str = "mean"
    ) -> np.ndarray:
        """Get pooled router logits for all samples at a layer.
        
        Args:
            layer: Layer number
            method: Pooling method (mean, max, last)
            
        Returns:
            Pooled logits of shape (n_samples, 64)
        """
        from src.routing.aggregation import pool_router_logits
        
        layer_idx = self.layers.index(layer)
        pooled = []
        
        for i in range(self.n_samples):
            n_tokens = self.token_counts[i]
            logits = self.router_logits[i, layer_idx, :n_tokens]
            pooled_logits = pool_router_logits(logits, method=method)
            pooled.append(pooled_logits)
        
        return np.stack(pooled)
    
    def get_pooled_residual_stream(
        self,
        layer: int,
        method: str = "mean"
    ) -> np.ndarray:
        """Get pooled residual stream for all samples at a layer.
        
        Args:
            layer: Layer number
            method: Pooling method (mean, max, last)
            
        Returns:
            Pooled hidden states of shape (n_samples, 2048)
        """
        from src.routing.aggregation import pool_residual_stream
        
        layer_idx = self.layers.index(layer)
        pooled = []
        
        for i in range(self.n_samples):
            n_tokens = self.token_counts[i]
            hidden = self.residual_stream[i, layer_idx, :n_tokens]
            pooled_hidden = pool_residual_stream(hidden, method=method)
            pooled.append(pooled_hidden)
        
        return np.stack(pooled)


def generate_cache_filename(
    dataset: str,
    split: str,
    output_dir: Optional[Path] = None
) -> Path:
    """Generate cache filename with timestamp.
    
    Args:
        dataset: Dataset name (e.g., 'dailydialog')
        split: Split name (e.g., 'train')
        output_dir: Output directory (defaults to results/extractions/)
        
    Returns:
        Path like results/extractions/dailydialog_train_20260104_001500.npz
    """
    if output_dir is None:
        output_dir = Path("results/extractions")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset}_{split}_{timestamp}.npz"
    
    return output_dir / filename


def save_extraction_cache(
    router_logits,  # Dict[sample_id -> {layer: (seq_len, 64)}] or ndarray (n, n_layers, seq, 64)
    residual_stream,  # Dict[sample_id -> {layer: (seq_len, 2048)}] or ndarray
    token_counts,  # Dict[sample_id -> int] or ndarray (n,)
    sample_ids: list[str],
    layers: list[int],
    output_path: Path,
    dataset: str = "unknown",
    split: str = "unknown",
    model_name: str = "allenai/OLMoE-1B-7B-0924",
    max_length: int = 512,
) -> Path:
    """Save extraction results to .npz cache.
    
    Args:
        router_logits: Dict[sample_id -> {layer: (seq_len, 64)}] OR 
                       ndarray of shape (n_samples, n_layers, max_seq_len, 64)
        residual_stream: Dict[sample_id -> {layer: (seq_len, 2048)}] OR
                         ndarray of shape (n_samples, n_layers, max_seq_len, 2048)
        token_counts: Dict[sample_id -> n_tokens] OR ndarray of shape (n_samples,)
        sample_ids: Ordered list of sample IDs
        layers: List of layer numbers extracted
        output_path: Where to save the cache
        dataset: Dataset name for metadata
        split: Split name for metadata
        model_name: Model name for metadata
        max_length: Max sequence length used
        
    Returns:
        Path to saved cache file
    """
    n_samples = len(sample_ids)
    n_layers = len(layers)
    
    # Check if inputs are already arrays (for testing) or dicts (from Modal)
    if isinstance(router_logits, np.ndarray):
        # Already in array format - use directly
        router_array = router_logits.astype(np.float32)
        residual_array = residual_stream.astype(np.float32)
        token_count_array = np.asarray(token_counts, dtype=np.int32)
        max_seq = router_array.shape[2]
    else:
        # Dict format from Modal extraction - need to pad and stack
        # Find max sequence length in this batch
        max_seq = max(token_counts.values())
        
        # Pre-allocate padded arrays
        router_array = np.zeros((n_samples, n_layers, max_seq, 64), dtype=np.float32)
        residual_array = np.zeros((n_samples, n_layers, max_seq, 2048), dtype=np.float32)
        token_count_array = np.zeros(n_samples, dtype=np.int32)
        
        # Fill arrays
        for i, sample_id in enumerate(sample_ids):
            n_tokens = token_counts[sample_id]
            token_count_array[i] = n_tokens
            
            for j, layer in enumerate(layers):
                if sample_id in router_logits and layer in router_logits[sample_id]:
                    router = router_logits[sample_id][layer]
                    seq_len = min(router.shape[0], max_seq)
                    router_array[i, j, :seq_len] = router[:seq_len]
                
                if sample_id in residual_stream and layer in residual_stream[sample_id]:
                    residual = residual_stream[sample_id][layer]
                    seq_len = min(residual.shape[0], max_seq)
                    residual_array[i, j, :seq_len] = residual[:seq_len]
    
    # Build metadata
    metadata = {
        "model_name": model_name,
        "extraction_timestamp": datetime.now().isoformat(),
        "n_samples": n_samples,
        "layers_extracted": layers,
        "max_length": max_length,
        "max_seq_in_batch": max_seq,
        "dataset": dataset,
        "split": split,
        "sample_ids": sample_ids,
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        router_logits=router_array,
        residual_stream=residual_array,
        token_counts=token_count_array,
        metadata=json.dumps(metadata),
    )
    
    print(f"Saved cache to {output_path}")
    print(f"  Samples: {n_samples}")
    print(f"  Layers: {layers}")
    print(f"  Max seq len: {max_seq}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return output_path


def load_extraction_cache(cache_path: Path) -> ExtractionCache:
    """Load extraction cache from .npz file.
    
    Args:
        cache_path: Path to .npz cache file
        
    Returns:
        ExtractionCache object with loaded data
    """
    cache_path = Path(cache_path)
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    data = np.load(cache_path, allow_pickle=True)
    
    metadata = json.loads(str(data["metadata"]))
    
    return ExtractionCache(
        router_logits=data["router_logits"],
        residual_stream=data["residual_stream"],
        token_counts=data["token_counts"],
        sample_ids=metadata["sample_ids"],
        layers=metadata["layers_extracted"],
        metadata=metadata,
    )


def find_latest_cache(
    dataset: str,
    split: str,
    cache_dir: Optional[Path] = None
) -> Optional[Path]:
    """Find the most recent cache file for a dataset/split.
    
    Args:
        dataset: Dataset name
        split: Split name
        cache_dir: Directory to search (defaults to results/extractions/)
        
    Returns:
        Path to latest cache file, or None if not found
    """
    if cache_dir is None:
        cache_dir = Path("results/extractions")
    cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        return None
    
    pattern = f"{dataset}_{split}_*.npz"
    matches = list(cache_dir.glob(pattern))
    
    if not matches:
        return None
    
    # Sort by modification time, return most recent
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def list_caches(cache_dir: Optional[Path] = None) -> list[dict]:
    """List all available extraction caches with their metadata.
    
    Args:
        cache_dir: Directory to search
        
    Returns:
        List of dicts with cache info
    """
    if cache_dir is None:
        cache_dir = Path("results/extractions")
    cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        return []
    
    caches = []
    for path in cache_dir.glob("*.npz"):
        try:
            data = np.load(path, allow_pickle=True)
            metadata = json.loads(str(data["metadata"]))
            caches.append({
                "path": path,
                "dataset": metadata.get("dataset", "unknown"),
                "split": metadata.get("split", "unknown"),
                "n_samples": metadata.get("n_samples", 0),
                "timestamp": metadata.get("extraction_timestamp", ""),
                "size_mb": path.stat().st_size / 1024 / 1024,
            })
        except Exception as e:
            caches.append({
                "path": path,
                "error": str(e),
            })
    
    return caches
