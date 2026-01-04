"""Modal app for OLMoE router logit extraction.

This module provides GPU-accelerated extraction of router logits and residual
stream activations from OLMoE-1B-7B using Modal serverless infrastructure.

Usage:
    # Dry run to verify setup
    modal run src/routing/modal_app.py --dry-run
    
    # Extract from a list of texts
    modal run src/routing/modal_app.py --input texts.json --output results.npz
    
    # With custom batch size
    modal run src/routing/modal_app.py --input texts.json --batch-size 16
"""

import modal
from datetime import datetime
from typing import Optional
import json

# Modal app configuration
app = modal.App("olmoe-router-extraction")

# Volume for caching model weights
model_volume = modal.Volume.from_name("olmoe-weights-cache", create_if_missing=True)

# Container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.35",
        "accelerate",
        "numpy",
        "tqdm",
    )
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/model-cache": model_volume},
)
class OLMoEExtractor:
    """Modal class for extracting router logits from OLMoE."""
    
    model_name: str = "allenai/OLMoE-1B-7B-0924"
    
    @modal.enter()
    def load_model(self):
        """Load model on container startup."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os
        
        # Use cached weights if available
        cache_dir = "/model-cache/hf_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        
        print(f"Loading {self.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set up hooks for capturing router logits
        self.captured_router = {}
        self.captured_residual = {}
        
        # Get number of layers
        self.n_layers = len(self.model.model.layers)
        print(f"Model has {self.n_layers} layers")
        
        # Register hooks on router gates
        def make_router_hook(layer_idx: int):
            def hook(module, input, output):
                # OLMoE gate outputs (seq_len, n_experts) - no batch dim at this point
                # We store as-is, shape: (seq_len, 64)
                self.captured_router[layer_idx] = output.detach().cpu()
            return hook
        
        def make_residual_hook(layer_idx: int):
            def hook(module, input, output):
                # Layer output is tuple, first element is hidden states
                # Shape: (batch, seq_len, hidden_dim) = (1, seq_len, 2048)
                if isinstance(output, tuple):
                    self.captured_residual[layer_idx] = output[0].detach().cpu()
                else:
                    self.captured_residual[layer_idx] = output.detach().cpu()
            return hook
        
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                layer.mlp.gate.register_forward_hook(make_router_hook(idx))
            layer.register_forward_hook(make_residual_hook(idx))
        
        self.model.eval()
        print("Model loaded and hooks registered.")
        
        # Commit volume to save cached weights
        model_volume.commit()
    
    @modal.method()
    def extract_batch(
        self,
        texts: list[str],
        sample_ids: list[str],
        layers: Optional[list[int]] = None,
        max_length: int = 512,
    ) -> dict:
        """Extract router logits and residual stream for a batch of texts.
        
        Args:
            texts: List of text strings to process
            sample_ids: Corresponding IDs for each text
            layers: Which layers to extract (None = all)
            max_length: Maximum sequence length
            
        Returns:
            Dict with router_logits, residual_stream, and metadata
        """
        import torch
        import numpy as np
        
        if layers is None:
            layers = list(range(self.n_layers))
        
        results = {
            "sample_ids": sample_ids,
            "layers": layers,
            "router_logits": {},      # sample_id -> {layer: ndarray}
            "residual_stream": {},    # sample_id -> {layer: ndarray}
            "token_counts": {},       # sample_id -> n_tokens
            "metadata": {
                "model_name": self.model_name,
                "extraction_timestamp": datetime.now().isoformat(),
                "n_layers": self.n_layers,
                "max_length": max_length,
            }
        }
        
        with torch.no_grad():
            for text, sample_id in zip(texts, sample_ids):
                # Clear captured dicts
                self.captured_router.clear()
                self.captured_residual.clear()
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.model.device)
                
                # Forward pass
                _ = self.model(**inputs)
                
                n_tokens = inputs.attention_mask.sum().item()
                results["token_counts"][sample_id] = n_tokens
                
                # Extract router logits for requested layers
                results["router_logits"][sample_id] = {}
                results["residual_stream"][sample_id] = {}
                
                for layer in layers:
                    if layer in self.captured_router:
                        # Router gate output shape: (seq_len, 64) - no batch dim
                        # Cast from bfloat16 to float32 for numpy compatibility
                        router = self.captured_router[layer].float().numpy()
                        results["router_logits"][sample_id][layer] = router
                    
                    if layer in self.captured_residual:
                        # Layer output shape: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
                        # Cast from bfloat16 to float32 for numpy compatibility
                        residual = self.captured_residual[layer][0].float().numpy()
                        results["residual_stream"][sample_id][layer] = residual
        
        return results


@app.function(image=image, timeout=1800)  # 30 min for first-run model download
def extract_texts(
    texts: list[str],
    sample_ids: Optional[list[str]] = None,
    batch_size: int = 8,
    layers: Optional[list[int]] = None,
    max_length: int = 512,
) -> dict:
    """Extract router logits and residual stream for a list of texts.
    
    This is the main entry point for extraction. It handles batching and
    aggregates results from the GPU worker.
    
    Args:
        texts: List of texts to process
        sample_ids: Optional IDs for each text (defaults to indices)
        batch_size: Number of texts per batch
        layers: Which layers to extract (None = all)
        max_length: Maximum sequence length
        
    Returns:
        Combined results dict with all extractions
    """
    from tqdm import tqdm
    
    if sample_ids is None:
        sample_ids = [f"sample_{i:05d}" for i in range(len(texts))]
    
    if len(texts) != len(sample_ids):
        raise ValueError(f"texts ({len(texts)}) and sample_ids ({len(sample_ids)}) must have same length")
    
    # Initialize extractor
    extractor = OLMoEExtractor()
    
    all_results = {
        "sample_ids": [],
        "layers": None,
        "router_logits": {},
        "residual_stream": {},
        "token_counts": {},
        "metadata": None,
        "errors": [],
    }
    
    # Process in batches with progress bar
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Extracting"):
        batch_texts = texts[i:i + batch_size]
        batch_ids = sample_ids[i:i + batch_size]
        
        try:
            batch_results = extractor.extract_batch.remote(
                texts=batch_texts,
                sample_ids=batch_ids,
                layers=layers,
                max_length=max_length,
            )
            
            # Merge results
            all_results["sample_ids"].extend(batch_results["sample_ids"])
            all_results["router_logits"].update(batch_results["router_logits"])
            all_results["residual_stream"].update(batch_results["residual_stream"])
            all_results["token_counts"].update(batch_results["token_counts"])
            
            if all_results["layers"] is None:
                all_results["layers"] = batch_results["layers"]
            if all_results["metadata"] is None:
                all_results["metadata"] = batch_results["metadata"]
                
        except Exception as e:
            error_msg = f"Batch {i//batch_size} failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            all_results["errors"].append({
                "batch_start": i,
                "batch_ids": batch_ids,
                "error": str(e),
            })
            
            # Try with smaller batch if OOM
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                print(f"OOM detected, trying samples individually...")
                for j, (text, sid) in enumerate(zip(batch_texts, batch_ids)):
                    try:
                        single_result = extractor.extract_batch.remote(
                            texts=[text],
                            sample_ids=[sid],
                            layers=layers,
                            max_length=max_length,
                        )
                        all_results["sample_ids"].append(sid)
                        all_results["router_logits"].update(single_result["router_logits"])
                        all_results["residual_stream"].update(single_result["residual_stream"])
                        all_results["token_counts"].update(single_result["token_counts"])
                    except Exception as e2:
                        print(f"  Sample {sid} failed: {e2}")
                        all_results["errors"].append({
                            "sample_id": sid,
                            "error": str(e2),
                        })
    
    # Update metadata with processing stats
    if all_results["metadata"]:
        all_results["metadata"]["n_samples_processed"] = len(all_results["sample_ids"])
        all_results["metadata"]["n_errors"] = len(all_results["errors"])
    
    return all_results


@app.local_entrypoint()
def main(
    input: Optional[str] = None,
    output: Optional[str] = None,
    batch_size: int = 8,
    layers: Optional[str] = None,
    dry_run: bool = False,
):
    """CLI entrypoint for router logit extraction.
    
    Args:
        input: Path to JSON file with texts (list of strings or list of {id, text})
        output: Path to save results (NPZ format)
        batch_size: Batch size for processing
        layers: Comma-separated layer indices (e.g., "4,8,12,15"), default all
        dry_run: If True, just load model and process 2 test samples
    """
    import numpy as np
    
    # Parse layers
    layer_list = None
    if layers:
        layer_list = [int(x.strip()) for x in layers.split(",")]
    
    if dry_run:
        print("=" * 60)
        print("DRY RUN: Testing Modal setup and model loading")
        print("=" * 60)
        
        test_texts = [
            "Hello, how are you doing today?",
            "I'm doing great, thanks for asking! The weather is beautiful.",
        ]
        test_ids = ["test_001", "test_002"]
        
        print(f"\nProcessing {len(test_texts)} test samples...")
        print(f"  Batch size: {batch_size}")
        print(f"  Layers: {layer_list or 'all'}")
        
        results = extract_texts.remote(
            texts=test_texts,
            sample_ids=test_ids,
            batch_size=batch_size,
            layers=layer_list,
        )
        
        print("\n" + "=" * 60)
        print("DRY RUN RESULTS")
        print("=" * 60)
        
        if results.get("metadata"):
            print(f"\nMetadata:")
            for k, v in results["metadata"].items():
                print(f"  {k}: {v}")
        else:
            print("\nNo metadata (extraction may have failed)")
        
        print(f"\nProcessed {len(results['sample_ids'])} samples:")
        for sid in results["sample_ids"]:
            n_tokens = results["token_counts"][sid]
            n_layers = len(results["router_logits"][sid])
            
            # Get shapes from first layer
            first_layer = list(results["router_logits"][sid].keys())[0]
            router_shape = results["router_logits"][sid][first_layer].shape
            residual_shape = results["residual_stream"][sid][first_layer].shape
            
            print(f"  {sid}:")
            print(f"    tokens: {n_tokens}")
            print(f"    router_logits shape: {router_shape} x {n_layers} layers")
            print(f"    residual_stream shape: {residual_shape} x {n_layers} layers")
        
        if results["errors"]:
            print(f"\nErrors: {len(results['errors'])}")
            for err in results["errors"]:
                print(f"  {err}")
        else:
            print(f"\nNo errors!")
        
        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE - Modal setup verified!")
        print("=" * 60)
        return
    
    # Regular extraction
    if not input:
        print("Error: --input required (unless using --dry-run)")
        print("Usage: modal run src/routing/modal_app.py --input texts.json --output results.npz")
        return
    
    # Load input texts
    print(f"Loading texts from {input}...")
    with open(input) as f:
        data = json.load(f)
    
    # Handle both list of strings and list of {id, text} dicts
    if isinstance(data[0], str):
        texts = data
        sample_ids = [f"sample_{i:05d}" for i in range(len(texts))]
    else:
        texts = [d["text"] for d in data]
        sample_ids = [d.get("id", f"sample_{i:05d}") for i, d in enumerate(data)]
    
    print(f"Loaded {len(texts)} texts")
    
    # Extract
    results = extract_texts.remote(
        texts=texts,
        sample_ids=sample_ids,
        batch_size=batch_size,
        layers=layer_list,
    )
    
    # Save results
    if output:
        print(f"\nSaving results to {output}...")
        
        # Flatten nested dicts for npz storage
        save_dict = {
            "sample_ids": np.array(results["sample_ids"]),
            "layers": np.array(results["layers"]),
            "metadata": json.dumps(results["metadata"]),
        }
        
        # Save token counts
        save_dict["token_counts"] = np.array([
            results["token_counts"][sid] for sid in results["sample_ids"]
        ])
        
        # Save router logits and residual stream per layer
        for layer in results["layers"]:
            router_arrays = []
            residual_arrays = []
            
            for sid in results["sample_ids"]:
                if sid in results["router_logits"] and layer in results["router_logits"][sid]:
                    router_arrays.append(results["router_logits"][sid][layer])
                if sid in results["residual_stream"] and layer in results["residual_stream"][sid]:
                    residual_arrays.append(results["residual_stream"][sid][layer])
            
            # Note: These will have variable seq_len, so we store as object array
            save_dict[f"router_logits_layer{layer}"] = np.array(router_arrays, dtype=object)
            save_dict[f"residual_stream_layer{layer}"] = np.array(residual_arrays, dtype=object)
        
        np.savez_compressed(output, **save_dict)
        print(f"Saved to {output}")
    
    # Print summary
    print(f"\nExtraction complete:")
    print(f"  Samples processed: {len(results['sample_ids'])}")
    print(f"  Layers: {results['layers']}")
    print(f"  Errors: {len(results['errors'])}")

