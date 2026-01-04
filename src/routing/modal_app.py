"""Modal app for OLMoE router logit extraction and probing.

This module provides GPU-accelerated extraction of router logits and residual
stream activations from OLMoE-1B-7B using Modal serverless infrastructure.

Architecture:
- Extractions are saved directly to a Modal Volume (no large data transfer)
- Probing runs on Modal using the same Volume
- Only small JSON results are returned to local machine

Usage:
    # Dry run to verify setup
    modal run src/routing/modal_app.py --dry-run
    
    # Run full extraction (saves to Modal Volume)
    modal run src/routing/modal_app.py --extract --dataset dailydialog --split train
    
    # Run probes on cached extractions
    modal run src/routing/modal_app.py --probe --cache-name dailydialog_train_20260104_000000.npz
    
    # List cached extractions
    modal run src/routing/modal_app.py --list-caches
"""

import modal
from datetime import datetime
from typing import Optional
import json

# Modal app configuration
app = modal.App("olmoe-router-extraction")

# Volumes
model_volume = modal.Volume.from_name("olmoe-weights-cache", create_if_missing=True)
extraction_volume = modal.Volume.from_name("extraction-cache", create_if_missing=True)

# Paths
MODEL_CACHE_DIR = "/model-cache"
EXTRACTION_DIR = "/extractions"

# Container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.35",
        "accelerate",
        "datasets>=2.14",
        "numpy",
        "scikit-learn>=1.3",
        "tqdm",
    )
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours for large extractions
    volumes={
        MODEL_CACHE_DIR: model_volume,
        EXTRACTION_DIR: extraction_volume,
    },
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
        cache_dir = f"{MODEL_CACHE_DIR}/hf_cache"
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
                self.captured_router[layer_idx] = output.detach().cpu()
            return hook
        
        def make_residual_hook(layer_idx: int):
            def hook(module, input, output):
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
    def extract_and_save(
        self,
        texts: list[str],
        sample_ids: list[str],
        dataset: str,
        split: str,
        batch_size: int = 8,
        layers: Optional[list[int]] = None,
        max_length: int = 512,
    ) -> dict:
        """Extract router logits and save directly to Volume.
        
        Args:
            texts: List of text strings to process
            sample_ids: Corresponding IDs for each text
            dataset: Dataset name for cache filename
            split: Split name for cache filename
            batch_size: Batch size for processing
            layers: Which layers to extract (None = all)
            max_length: Maximum sequence length
            
        Returns:
            Dict with status and cache path (no large arrays)
        """
        import torch
        import numpy as np
        from tqdm import tqdm
        import os
        
        if layers is None:
            layers = list(range(self.n_layers))
        
        n_samples = len(texts)
        print(f"Extracting {n_samples} samples...")
        
        # Collect all results
        router_logits = {}
        residual_stream = {}
        token_counts = {}
        errors = []
        
        with torch.no_grad():
            for i, (text, sample_id) in enumerate(tqdm(
                zip(texts, sample_ids), 
                total=n_samples,
                desc="Extracting"
            )):
                try:
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
                    token_counts[sample_id] = n_tokens
                    
                    # Extract router logits for requested layers
                    router_logits[sample_id] = {}
                    residual_stream[sample_id] = {}
                    
                    for layer in layers:
                        if layer in self.captured_router:
                            router = self.captured_router[layer].float().numpy()
                            router_logits[sample_id][layer] = router
                        
                        if layer in self.captured_residual:
                            residual = self.captured_residual[layer][0].float().numpy()
                            residual_stream[sample_id][layer] = residual
                            
                except Exception as e:
                    print(f"Error on sample {sample_id}: {e}")
                    errors.append({"sample_id": sample_id, "error": str(e)})
        
        # Compute max sequence length
        max_seq = max(token_counts.values()) if token_counts else 0
        
        # Build padded arrays
        print("Building padded arrays...")
        n_layers = len(layers)
        router_array = np.zeros((n_samples, n_layers, max_seq, 64), dtype=np.float32)
        residual_array = np.zeros((n_samples, n_layers, max_seq, 2048), dtype=np.float32)
        token_count_array = np.zeros(n_samples, dtype=np.int32)
        
        for i, sample_id in enumerate(sample_ids):
            if sample_id in token_counts:
                token_count_array[i] = token_counts[sample_id]
                
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
            "model_name": self.model_name,
            "extraction_timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "layers_extracted": layers,
            "max_length": max_length,
            "max_seq_in_batch": max_seq,
            "dataset": dataset,
            "split": split,
            "sample_ids": sample_ids,
            "n_errors": len(errors),
        }
        
        # Save to Volume
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_name = f"{dataset}_{split}_{timestamp}.npz"
        cache_path = f"{EXTRACTION_DIR}/{cache_name}"
        
        os.makedirs(EXTRACTION_DIR, exist_ok=True)
        
        print(f"Saving to {cache_path}...")
        np.savez_compressed(
            cache_path,
            router_logits=router_array,
            residual_stream=residual_array,
            token_counts=token_count_array,
            metadata=json.dumps(metadata),
        )
        
        # Get file size
        file_size_mb = os.path.getsize(cache_path) / 1024 / 1024
        
        # Commit volume to persist
        extraction_volume.commit()
        
        print(f"Saved {cache_name} ({file_size_mb:.1f} MB)")
        
        return {
            "status": "success",
            "cache_name": cache_name,
            "cache_path": cache_path,
            "n_samples": n_samples,
            "n_layers": n_layers,
            "max_seq_len": max_seq,
            "file_size_mb": file_size_mb,
            "n_errors": len(errors),
            "errors": errors[:10] if errors else [],  # Only return first 10 errors
        }


@app.function(
    image=image,
    cpu=4,  # Probing doesn't need GPU
    memory=16384,  # 16GB RAM for loading large caches
    timeout=1800,
    volumes={EXTRACTION_DIR: extraction_volume},
)
def run_probes(
    cache_name: str,
    layers: list[int] = [4, 8, 12, 15],
    pooling_methods: list[str] = ["mean"],
    task: str = "intent",
) -> dict:
    """Run linear probes on cached extractions.
    
    Args:
        cache_name: Name of cache file in extraction volume
        layers: Which layers to probe
        pooling_methods: Which pooling methods to test
        task: Task name (intent, emotion)
        
    Returns:
        Dict with probe results (small JSON)
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    
    cache_path = f"{EXTRACTION_DIR}/{cache_name}"
    
    print(f"Loading cache from {cache_path}...")
    data = np.load(cache_path, allow_pickle=True)
    
    router_logits = data["router_logits"]
    residual_stream = data["residual_stream"]
    token_counts = data["token_counts"]
    metadata = json.loads(str(data["metadata"]))
    
    sample_ids = metadata["sample_ids"]
    all_layers = metadata["layers_extracted"]
    n_samples = len(sample_ids)
    
    print(f"Loaded {n_samples} samples, layers: {all_layers}")
    
    # Load labels from HuggingFace
    from datasets import load_dataset
    from collections import defaultdict
    
    print("Loading DailyDialog labels from HuggingFace...")
    hf_dataset = load_dataset("benjaminbeilharz/better_daily_dialog", split=metadata["split"])
    
    # Build turn_id to label mapping
    # Schema: dialog_id, utterance, turn_type (1-4), emotion (0-6)
    dialogues = defaultdict(list)
    for example in hf_dataset:
        dialogues[example["dialog_id"]].append({
            "act": example["turn_type"],
            "emotion": example["emotion"],
        })
    
    # Map sample_ids to labels
    # sample_id format: daily_{split}_{dialog_idx:05d}_t{turn_idx:02d}
    labels = []
    valid_indices = []
    
    for i, sample_id in enumerate(sample_ids):
        # Parse sample_id
        parts = sample_id.split("_")
        dialog_idx = int(parts[2])
        turn_idx = int(parts[3][1:])  # Remove 't' prefix
        
        dialog_ids = sorted(dialogues.keys())
        if dialog_idx < len(dialog_ids):
            dialog_id = dialog_ids[dialog_idx]
            if turn_idx < len(dialogues[dialog_id]):
                if task == "intent":
                    label = dialogues[dialog_id][turn_idx]["act"]
                    # Map 0->1 (dummy to inform)
                    if label == 0:
                        label = 1
                    labels.append(label - 1)  # Convert to 0-indexed
                else:  # emotion
                    labels.append(dialogues[dialog_id][turn_idx]["emotion"])
                valid_indices.append(i)
    
    labels = np.array(labels)
    print(f"Matched {len(labels)} samples with labels")
    
    # Filter to valid indices
    router_logits = router_logits[valid_indices]
    residual_stream = residual_stream[valid_indices]
    token_counts = token_counts[valid_indices]
    
    # Train/test split (80/20)
    n_valid = len(labels)
    n_train = int(0.8 * n_valid)
    
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_valid)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    # Pooling function
    def pool(arr, counts, method):
        """Pool (n_samples, seq_len, dim) -> (n_samples, dim)"""
        pooled = []
        for i in range(len(arr)):
            n_tokens = counts[i]
            if method == "mean":
                pooled.append(arr[i, :n_tokens].mean(axis=0))
            elif method == "max":
                pooled.append(arr[i, :n_tokens].max(axis=0))
            elif method == "last":
                pooled.append(arr[i, n_tokens-1])
        return np.stack(pooled)
    
    results = []
    
    for layer in layers:
        if layer not in all_layers:
            print(f"Layer {layer} not in cache, skipping")
            continue
            
        layer_idx = all_layers.index(layer)
        
        for pooling in pooling_methods:
            print(f"  Layer {layer}, pooling={pooling}...")
            
            # Pool router logits
            X_router = pool(router_logits[:, layer_idx], token_counts, pooling)
            X_train_r = X_router[train_idx]
            X_test_r = X_router[test_idx]
            
            # Train router probe
            probe_r = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
            probe_r.fit(X_train_r, y_train)
            y_pred_r = probe_r.predict(X_test_r)
            y_prob_r = probe_r.predict_proba(X_test_r)
            
            try:
                auc_r = roc_auc_score(y_test, y_prob_r, multi_class="ovr", average="macro")
            except:
                auc_r = 0.0
            
            results.append({
                "task": task,
                "probe_target": "router_logits",
                "layer": layer,
                "pooling": pooling,
                "auc": float(auc_r),
                "accuracy": float(accuracy_score(y_test, y_pred_r)),
                "f1_macro": float(f1_score(y_test, y_pred_r, average="macro")),
                "n_train": len(y_train),
                "n_test": len(y_test),
            })
            print(f"    Router AUC: {auc_r:.3f}, Acc: {accuracy_score(y_test, y_pred_r):.3f}")
            
            # Pool residual stream
            X_residual = pool(residual_stream[:, layer_idx], token_counts, pooling)
            X_train_s = X_residual[train_idx]
            X_test_s = X_residual[test_idx]
            
            # Train residual probe
            probe_s = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
            probe_s.fit(X_train_s, y_train)
            y_pred_s = probe_s.predict(X_test_s)
            y_prob_s = probe_s.predict_proba(X_test_s)
            
            try:
                auc_s = roc_auc_score(y_test, y_prob_s, multi_class="ovr", average="macro")
            except:
                auc_s = 0.0
            
            results.append({
                "task": task,
                "probe_target": "residual_stream",
                "layer": layer,
                "pooling": pooling,
                "auc": float(auc_s),
                "accuracy": float(accuracy_score(y_test, y_pred_s)),
                "f1_macro": float(f1_score(y_test, y_pred_s, average="macro")),
                "n_train": len(y_train),
                "n_test": len(y_test),
            })
            print(f"    Residual AUC: {auc_s:.3f}, Acc: {accuracy_score(y_test, y_pred_s):.3f}")
    
    return {
        "task": task,
        "cache_name": cache_name,
        "n_samples": n_valid,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "results": results,
    }


@app.function(
    image=image,
    volumes={EXTRACTION_DIR: extraction_volume},
    timeout=60,
)
def list_caches() -> list[dict]:
    """List all cached extractions in the Volume."""
    import os
    import numpy as np
    
    caches = []
    
    if not os.path.exists(EXTRACTION_DIR):
        return caches
    
    for filename in os.listdir(EXTRACTION_DIR):
        if filename.endswith(".npz"):
            path = f"{EXTRACTION_DIR}/{filename}"
            try:
                data = np.load(path, allow_pickle=True)
                metadata = json.loads(str(data["metadata"]))
                size_mb = os.path.getsize(path) / 1024 / 1024
                
                caches.append({
                    "name": filename,
                    "dataset": metadata.get("dataset", "unknown"),
                    "split": metadata.get("split", "unknown"),
                    "n_samples": metadata.get("n_samples", 0),
                    "timestamp": metadata.get("extraction_timestamp", ""),
                    "size_mb": round(size_mb, 1),
                })
            except Exception as e:
                caches.append({
                    "name": filename,
                    "error": str(e),
                })
    
    return caches


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    extract: bool = False,
    probe: bool = False,
    list: bool = False,
    dataset: str = "dailydialog",
    split: str = "train",
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    cache_name: Optional[str] = None,
    layers: str = "4,8,12,15",
    pooling: str = "mean",
    task: str = "intent",
):
    """CLI entrypoint for extraction and probing.
    
    Args:
        dry_run: Test with 2 samples
        extract: Run extraction and save to Volume
        probe: Run probes on cached extraction
        list: List cached extractions
        dataset: Dataset name
        split: Split name
        batch_size: Batch size for extraction
        max_samples: Limit number of samples
        cache_name: Cache file to probe (for --probe)
        layers: Comma-separated layer indices
        pooling: Comma-separated pooling methods
        task: Task name (intent, emotion)
    """
    layer_list = [int(x.strip()) for x in layers.split(",")]
    pooling_list = [x.strip() for x in pooling.split(",")]
    
    # List caches
    if list:
        print("Cached extractions:")
        caches = list_caches.remote()
        if not caches:
            print("  (none)")
        for cache in caches:
            if "error" in cache:
                print(f"  {cache['name']}: ERROR - {cache['error']}")
            else:
                print(f"  {cache['name']}: {cache['dataset']}/{cache['split']}, "
                      f"{cache['n_samples']} samples, {cache['size_mb']} MB")
        return
    
    # Dry run
    if dry_run:
        print("=" * 60)
        print("DRY RUN: Testing extraction with 2 samples")
        print("=" * 60)
        
        extractor = OLMoEExtractor()
        result = extractor.extract_and_save.remote(
            texts=[
                "Hello, how are you doing today?",
                "I'm doing great, thanks for asking!",
            ],
            sample_ids=["test_001", "test_002"],
            dataset="test",
            split="dryrun",
            batch_size=2,
            layers=layer_list,
        )
        
        print("\nResult:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        
        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE")
        print("=" * 60)
        return
    
    # Extract
    if extract:
        print("=" * 60)
        print(f"EXTRACTION: {dataset}/{split}")
        print("=" * 60)
        
        # Load data from HuggingFace
        from datasets import load_dataset
        from collections import defaultdict
        
        print(f"Loading {dataset} from HuggingFace...")
        hf_dataset = load_dataset("benjaminbeilharz/better_daily_dialog", split=split)
        
        # Group by dialog_id
        dialogues = defaultdict(list)
        for example in hf_dataset:
            dialogues[example["dialog_id"]].append(example["utterance"])
        
        # Build texts and sample_ids
        texts = []
        sample_ids = []
        
        for dialog_idx, dialog_id in enumerate(sorted(dialogues.keys())):
            for turn_idx, text in enumerate(dialogues[dialog_id]):
                sample_id = f"daily_{split}_{dialog_idx:05d}_t{turn_idx:02d}"
                texts.append(text.strip())
                sample_ids.append(sample_id)
        
        print(f"Prepared {len(texts)} samples from {len(dialogues)} dialogues")
        
        if max_samples:
            texts = texts[:max_samples]
            sample_ids = sample_ids[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        # Run extraction
        extractor = OLMoEExtractor()
        result = extractor.extract_and_save.remote(
            texts=texts,
            sample_ids=sample_ids,
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            layers=None,  # Extract all layers
        )
        
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"  Cache: {result['cache_name']}")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Layers: {result['n_layers']}")
        print(f"  Size: {result['file_size_mb']:.1f} MB")
        print(f"  Errors: {result['n_errors']}")
        
        if result["errors"]:
            print("  First errors:")
            for err in result["errors"][:5]:
                print(f"    {err}")
        
        return
    
    # Probe
    if probe:
        if not cache_name:
            print("Error: --cache-name required for --probe")
            print("Run with --list-caches to see available caches")
            return
        
        print("=" * 60)
        print(f"PROBING: {cache_name}")
        print("=" * 60)
        print(f"  Layers: {layer_list}")
        print(f"  Pooling: {pooling_list}")
        print(f"  Task: {task}")
        
        result = run_probes.remote(
            cache_name=cache_name,
            layers=layer_list,
            pooling_methods=pooling_list,
            task=task,
        )
        
        print("\n" + "=" * 60)
        print("PROBE RESULTS")
        print("=" * 60)
        print(f"\n{'Target':<20} {'Layer':<8} {'Pooling':<10} {'AUC':<8} {'Accuracy':<10}")
        print("-" * 56)
        
        for r in result["results"]:
            print(f"{r['probe_target']:<20} {r['layer']:<8} {r['pooling']:<10} "
                  f"{r['auc']:<8.3f} {r['accuracy']:<10.3f}")
        
        # Find best router AUC
        router_aucs = [r["auc"] for r in result["results"] if r["probe_target"] == "router_logits"]
        if router_aucs:
            best_auc = max(router_aucs)
            print(f"\nBest router AUC: {best_auc:.3f}")
            if best_auc >= 0.75:
                print("✓ Meets minimum bar (AUC ≥ 0.75)")
            else:
                print("✗ Below minimum bar (AUC < 0.75)")
        
        return
    
    # No action specified
    print("Usage:")
    print("  modal run src/routing/modal_app.py --dry-run")
    print("  modal run src/routing/modal_app.py --extract --dataset dailydialog --split train")
    print("  modal run src/routing/modal_app.py --list")
    print("  modal run src/routing/modal_app.py --probe --cache-name <name>")
