# Relational State Estimation from MoE Router Logits — Agent Reference

> **This document is the primary reference for AI agents working on this codebase.**  
> Read this FIRST before making any changes.

*Last updated: 2026-01-03*

---

## Executive Summary

This project extends prior MoE router characterization work from mono-axial style signals (formality) to **dyadic relational signals**—properties of speaker relationships rather than individual text properties.

**Prior finding:** Router logits (64-dim) encode formality at AUC ≈ 1.0 with 32× compression vs residual stream. Routing is epiphenomenal (M ≈ 1.0)—excellent for classification, not for steering.

**This sprint asks:** Does routing also encode relational dynamics like power differential, emotional valence, intent category, and interpersonal warmth?

**Model:** OLMoE-1B-7B (fully open, interpretability-friendly)

**Approach:** Linear probing on router logits for relational signal classification, building on existing formality probe infrastructure.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        OLMoE-1B-7B                               │
├─────────────────────────────────────────────────────────────────┤
│  Input → Embedding → [MoE Blocks × 16] → Output                  │
│                           │                                      │
│                    ┌──────┴──────┐                               │
│                    │  MoE Block   │                               │
│                    ├─────────────┤                               │
│                    │ Router MLP  │ ← We probe HERE               │
│                    │   (gate)    │   (64-dim logits)             │
│                    │     ↓       │                               │
│                    │ Top-k Gate  │                               │
│                    │     ↓       │                               │
│                    │ Expert FFNs │                               │
│                    └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘

Key locations:
- Router gate: model.model.layers[i].mlp.gate
- Router logits shape: (seq_len, 64)
- Residual stream shape: (seq_len, 2048)
```

---

## Project Phases

### Phase 1: Infrastructure & Intent/Emotion Probes

**Goal:** Port existing tooling; establish baseline probes for non-relational signals on dialogue data.

#### Tasks
- [x] Port formality probe code from prior project
- [x] Set up Modal pipeline for OLMoE router logit extraction
- [x] Load and preprocess DailyDialog dataset
- [ ] Verify formality replication on DailyDialog subset (sanity check)
- [x] Train intent category probes: router logits → {inform, question, directive, commissive}
- [x] Train emotion probes: router logits → 7-class emotion
- [x] Compare router AUC vs residual stream AUC for each signal
- [ ] Document pooling sensitivity (mean vs max vs last-token)

#### Exit Criteria
- [x] Intent probe AUC ≥ 0.75 on held-out DailyDialog *(achieved 0.905)*
- [x] Emotion probe AUC documented (even if lower) *(achieved 0.849)*
- [x] Compression ratio comparison table: router vs residual *(see results above)*
- [ ] Pooling sensitivity documented

#### Key Files
```
src/data/dailydialog.py       # DailyDialog loader with dialogue act + emotion labels
src/data/meld.py              # MELD loader (stretch)
src/probing/linear_probe.py   # Generic linear probe (port from prior project)
src/probing/intent_probe.py   # Intent-specific probing
src/probing/emotion_probe.py  # Emotion-specific probing
src/routing/extraction.py     # Router logit extraction (port from prior project)
src/routing/aggregation.py    # Token→example pooling strategies
scripts/run_intent_probes.py  # Phase 1 experiment runner
```

#### Data Structures
```python
@dataclass
class DialogueTurn:
    """Single turn in a dialogue"""
    turn_id: str
    dialogue_id: str
    speaker: int           # 0 or 1 (alternating speakers)
    text: str
    dialogue_act: str      # inform, question, directive, commissive
    emotion: str           # neutral, anger, disgust, fear, happiness, sadness, surprise
    turn_index: int

@dataclass
class LoadStats:
    """Statistics from loading DailyDialog.
    
    Returned alongside list[DialogueTurn] from load_dailydialog():
        turns, stats = load_dailydialog(split="train")
        stats.print_summary()
    """
    n_dialogues: int
    n_turns: int
    n_skipped: int
    skipped_reasons: list[tuple[int, str]]  # (line_number, reason)
    act_distribution: dict[str, int]
    emotion_distribution: dict[str, int]
    
    def print_summary(self) -> None:
        """Print formatted loading summary."""

@dataclass
class RouterLogits:
    """Extracted router logits for a text sample"""
    sample_id: str
    layer: int
    token_logits: torch.Tensor    # shape: (seq_len, 64)
    pooled_logits: torch.Tensor   # shape: (64,) after aggregation
    pooling_method: str           # mean, max, last
    n_tokens: int

@dataclass
class ProbeResult:
    """Result of training a linear probe"""
    task: str                     # intent, emotion, formality, power, etc.
    probe_target: str             # router_logits, residual
    layer: int
    pooling: str
    auc: float
    accuracy: float
    f1_macro: float               # For multi-class
    confusion_matrix: np.ndarray
    n_train: int
    n_test: int
    trained_at: datetime

@dataclass
class ExtractionCache:
    """Pre-pooled extraction cache stored on Modal Volume.
    
    Architecture note: We pool DURING extraction (not after loading) to reduce
    cache size from ~65 GB to ~235 MB for 10K samples.
    
    File format: .npz with compressed arrays
    Naming: {dataset}_{split}_{pooling}_{timestamp}.npz
    """
    # Arrays (pre-pooled)
    router_logits_pooled: np.ndarray   # (n_samples, n_layers, 64)
    residual_pooled: np.ndarray        # (n_samples, n_layers, 2048)
    token_counts: np.ndarray           # (n_samples,)
    
    # Metadata (JSON string in .npz)
    model_name: str                    # "allenai/OLMoE-1B-7B-0924"
    extraction_timestamp: str          # ISO format
    n_samples: int
    layers_extracted: list[int]        # e.g., [4, 8, 12, 15]
    pooling: str                       # "mean", "max", or "last"
    dataset: str                       # "dailydialog"
    split: str                         # "train", "validation", "test"
    sample_ids: list[str]              # For label matching
```

---

## Phase 1 Results (10K samples, DailyDialog train)

*Extracted 2026-01-04, pooling=mean, layers=[4, 8, 12, 15]*

### Intent Classification (4-class)

**Class distribution:** inform (55%), question (30%), directive (9%), commissive (6%)

| Target | Layer | AUC | Accuracy |
|--------|-------|-----|----------|
| router_logits | 4 | **0.905** | 0.792 |
| residual_stream | 4 | 0.921 | 0.785 |
| router_logits | 8 | 0.902 | 0.780 |
| residual_stream | 8 | 0.940 | 0.839 |
| router_logits | 12 | 0.872 | 0.758 |
| residual_stream | 12 | 0.946 | 0.860 |
| router_logits | 15 | 0.865 | 0.733 |
| residual_stream | 15 | 0.935 | 0.844 |

**Finding:** Router logits achieve 96% of residual stream AUC (0.905 vs 0.940) with 32× fewer dimensions (64 vs 2048). **H1 confirmed** (predicted ≥0.90, achieved 0.905).

### Emotion Classification (7-class)

**Class distribution:** neutral (75%), happiness (18%), surprise (3%), sadness (1%), anger (1%), fear (<1%), disgust (<1%)

| Target | Layer | AUC | Accuracy |
|--------|-------|-----|----------|
| router_logits | 4 | 0.840 | 0.791 |
| residual_stream | 4 | 0.798 | 0.789 |
| router_logits | 8 | **0.849** | 0.790 |
| residual_stream | 8 | 0.891 | 0.812 |
| router_logits | 12 | 0.799 | 0.785 |
| residual_stream | 12 | 0.911 | 0.810 |
| router_logits | 15 | 0.779 | 0.770 |
| residual_stream | 15 | 0.889 | 0.802 |

**Finding:** Router encodes emotion but weaker than intent (0.849 vs 0.905). 93% retention vs residual (0.849 vs 0.911). **H2 confirmed** (predicted 0.70-0.85, achieved 0.849).

### Key Patterns

1. **Router signal peaks at layers 4-8** — early routing decisions encode intent/emotion
2. **Residual signal peaks at layer 12** — continued refinement through later layers
3. **Router beats residual at layer 4 for emotion** (0.840 vs 0.798) — emotion encoded early
4. **Class imbalance affects both tasks** — neutral (75%) and inform (55%) dominate

### Implementation Notes

- **Modal Volume architecture:** Extractions stored on `extraction-cache` volume, probes run remotely, only small JSON results returned locally
- **Pre-pooled format:** ~235 MB cache vs ~65 GB for per-token storage (280× reduction)
- **Batch size:** 32 samples (up from 8), ~4.7 samples/sec on A10G
- **Layers extracted:** [4, 8, 12, 15] only (4 layers vs 16) — sufficient for probe analysis
- **Cost:** ~$1-2 for 10K sample extraction + probing

---

### Phase 2: Power & Dyadic Signal Probes

**Goal:** Test whether routing encodes relational properties of dyads, not just individual text properties.

#### Tasks
- [ ] Load Wikipedia Talk Pages dataset (admin/non-admin labels)
- [ ] Train power differential probes: router logits → {high-status, low-status}
- [ ] Analyze expert clusters for power (do power-linked experts overlap with formality experts?)
- [ ] Generate SOTOPIA interaction logs via their pipeline
- [ ] Train dyadic outcome probes: router logits at turn N → relationship score at end
- [ ] Generate synthetic tension/repair pairs (analogous to formal/informal pairs)
- [ ] Train tension probes: router logits → {escalation, repair, neutral}
- [ ] Expert cluster analysis for each signal type

#### Exit Criteria
- [ ] Power probe AUC documented
- [ ] Expert overlap analysis: power vs formality vs intent
- [ ] Tension probe AUC on synthetic data
- [ ] SOTOPIA relationship prediction AUC (stretch)

#### Key Files
```
src/data/wikipedia_talk.py    # Wikipedia Talk Pages loader
src/data/sotopia.py           # SOTOPIA log loader/generator
src/data/synthetic_tension.py # Synthetic tension/repair pair generator
src/probing/power_probe.py    # Power differential probing
src/probing/tension_probe.py  # Tension dynamics probing
src/probing/relationship_probe.py  # SOTOPIA outcome prediction
src/analysis/expert_clusters.py    # Expert activation analysis
scripts/run_power_probes.py   # Phase 2 experiment runner
scripts/generate_tension_pairs.py  # Synthetic data generation
```

#### Data Structures
```python
@dataclass
class WikipediaTurn:
    """Single turn from Wikipedia Talk Pages"""
    turn_id: str
    thread_id: str
    author_id: str
    text: str
    is_admin: bool            # Power label
    timestamp: datetime
    reply_to: Optional[str]   # Parent turn_id

@dataclass
class SOTOPIAEpisode:
    """Complete SOTOPIA interaction episode"""
    episode_id: str
    scenario: str
    agent1_profile: dict
    agent2_profile: dict
    turns: list[dict]         # [{speaker, text, turn_idx}, ...]
    outcomes: dict            # {believability, relationship, goal, ...}

@dataclass
class TensionPair:
    """Synthetic tension/repair pair"""
    pair_id: str
    context: str              # Shared conversation context
    escalation: str           # Response that escalates tension
    repair: str               # Response that de-escalates
    neutral: str              # Neutral continuation

@dataclass
class ExpertClusterAnalysis:
    """Analysis of which experts activate for which signals"""
    signal_type: str          # formality, power, intent, emotion, tension
    layer: int
    top_k_experts: list[int]  # Experts most associated with signal
    activation_scores: dict[int, float]  # Expert ID → mean activation
    overlap_with: dict[str, float]  # Other signal → Jaccard overlap
```

---

### Phase 3: Analysis & Synthesis

**Goal:** Statistical analysis, robustness checks, writeup.

#### Tasks
- [ ] Comparative AUC table across all signal types
- [ ] Layer-by-layer analysis (which layers encode which signals?)
- [ ] Expert overlap heatmap (do signals share experts or are they disjoint?)
- [ ] Pooling sensitivity ablation (mean vs max vs last-token)
- [ ] Cross-dataset transfer test (DailyDialog → MELD)
- [ ] Dumb baselines (BoW, punctuation counts) to contextualize probe lift
- [ ] Generate figures for writeup
- [ ] Draft Substack post
- [ ] Prepare GitHub repo for release

#### Exit Criteria
- [ ] Complete results table in results/
- [ ] All figures in results/figures/
- [ ] Substack draft in docs/writeup.md
- [ ] README updated with findings

#### Key Files
```
src/analysis/comparative.py   # Cross-signal AUC comparison
src/analysis/layer_analysis.py # Layer-by-layer probing
src/analysis/transfer.py      # Cross-dataset transfer tests
src/analysis/baselines.py     # BoW, punctuation baselines
src/analysis/visualization.py # Figure generation
scripts/run_ablations.py      # Systematic ablation runner
scripts/generate_figures.py   # Figure generation
docs/writeup.md               # Substack draft
```

---

## Datasets

### Tier 1 (Primary — Must Use)

#### DailyDialog
- **Size:** 13,118 dialogues, 103,607 utterances
- **Annotations:** Dialogue act (4 classes), Emotion (7 classes)
- **Source:** http://yanran.li/dailydialog.html
- **Signals:** Intent categories, emotional valence

```python
# Expected schema after loading
{
    "dialogue_id": "daily_001",
    "turns": [
        {"text": "...", "act": "inform", "emotion": "neutral", "speaker": 0},
        {"text": "...", "act": "question", "emotion": "happiness", "speaker": 1},
    ]
}
```

#### SOTOPIA Logs
- **Size:** 90 scenarios × N episodes (generate via their pipeline)
- **Annotations:** 7-dimensional social outcomes
- **Source:** https://github.com/sotopia-lab/sotopia
- **Signals:** Relationship trajectory, goal completion

#### Synthetic Tension Pairs
- **Size:** ~500 pairs (generate via Claude)
- **Annotations:** Escalation/repair/neutral by construction
- **Signals:** Tension dynamics

### Tier 2 (Secondary — If Time Permits)

#### Wikipedia Talk Pages
- **Size:** 240,000+ exchanges
- **Annotations:** Editor status (admin/non-admin)
- **Source:** Cornell Conversational Analysis Toolkit
- **Signals:** Power differential

#### MELD
- **Size:** 1,433 dialogues, 13,708 utterances (Friends TV)
- **Annotations:** Emotion (7), Sentiment (3)
- **Source:** https://github.com/declare-lab/MELD
- **Signals:** Multi-party emotion (transfer test)

---

## Infrastructure

| Component | Details |
|-----------|---------|
| **Model** | OLMoE-1B-7B (HuggingFace: allenai/OLMoE-1B-7B-0924) |
| **Compute** | Modal serverless A10G GPUs |
| **Probe Training** | scikit-learn LogisticRegression |
| **Embeddings** | Router logits (64-dim) vs residual stream (2048-dim) |
| **Budget** | <$50 total |

### Modal Setup
```bash
pip install modal
modal setup          # Browser-based auth
modal token new      # If needed
```

### Dependencies
```
torch>=2.0
transformers>=4.35
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm
modal
anthropic          # For synthetic data generation
```

---

## Code Patterns

### Router Logit Extraction
```python
# src/routing/extraction.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_olmoe_with_hooks(device="cuda"):
    """Load OLMoE with hooks to capture router logits."""
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMoE-1B-7B-0924",
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    
    captured = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            captured[layer_idx] = output.detach().cpu()
        return hook
    
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            layer.mlp.gate.register_forward_hook(make_hook(idx))
    
    return model, tokenizer, captured
```

### Token-to-Example Aggregation
```python
# src/routing/aggregation.py
import torch

def pool_router_logits(
    token_logits: torch.Tensor,  # (seq_len, n_experts)
    method: str = "mean"
) -> torch.Tensor:
    """Pool token-level router logits to example-level."""
    if method == "mean":
        return token_logits.mean(dim=0)
    elif method == "max":
        return token_logits.max(dim=0).values
    elif method == "last":
        return token_logits[-1]
    else:
        raise ValueError(f"Unknown pooling method: {method}")
```

### Linear Probe Training
```python
# src/probing/linear_probe.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

def train_probe(
    X_train, y_train, X_test, y_test, multiclass: bool = False
) -> dict:
    """Train linear probe and return metrics."""
    probe = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial" if multiclass else "auto"
    )
    probe.fit(X_train, y_train)
    
    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)
    
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr") if multiclass else roc_auc_score(y_test, y_prob[:, 1])
    
    return {
        "probe": probe,
        "auc": auc,
        "accuracy": (y_pred == y_test).mean(),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
```

### Expert Cluster Analysis
```python
# src/analysis/expert_clusters.py
import numpy as np

def compute_expert_associations(
    router_logits: np.ndarray,  # (n_samples, n_experts)
    labels: np.ndarray,
    top_k: int = 5
) -> dict:
    """Find which experts are most associated with each label."""
    associations = {}
    for label in np.unique(labels):
        mask = labels == label
        mean_logits = router_logits[mask].mean(axis=0)
        top_experts = np.argsort(mean_logits)[-top_k:][::-1]
        associations[label] = {
            "top_experts": top_experts.tolist(),
            "mean_activations": {int(e): float(mean_logits[e]) for e in top_experts}
        }
    return associations

def compute_expert_overlap(assoc_a: dict, assoc_b: dict, top_k: int = 5) -> float:
    """Compute Jaccard overlap between expert sets."""
    experts_a = set(e for d in assoc_a.values() for e in d["top_experts"][:top_k])
    experts_b = set(e for d in assoc_b.values() for e in d["top_experts"][:top_k])
    return len(experts_a & experts_b) / len(experts_a | experts_b) if experts_a | experts_b else 0
```

---

## Hypotheses

### H1 (Strong): Intent encodes at parity with formality
**Prediction:** Router logits achieve AUC ≥ 0.90 for 4-class intent classification.

### H2 (Medium): Emotion encodes but weaker than intent
**Prediction:** Emotion AUC 0.70-0.85, lower than intent.

### H3 (Exploratory): Power differential encodes
**Prediction:** Power probe AUC ≥ 0.65 on Wikipedia Talk.

### H4 (High Risk): Relationship prediction from early turns
**Prediction:** Some signal (AUC > 0.55) for SOTOPIA outcomes.

### Null Result Interpretations
- **Intent doesn't encode:** Routing more sensitive to surface than semantic
- **Emotion doesn't encode:** Emotion in attention, not FFN routing
- **Power doesn't encode:** Power requires cross-turn modeling
- **Relationship prediction fails:** Routing too local for longitudinal dynamics

---

## Success Criteria

### Minimum Viable Outcome
- Intent probe AUC ≥ 0.75 on held-out DailyDialog
- Clear compression ratio comparison (router vs residual)
- Documented characterization of which signals encode

### Target Outcome
- 2+ relational signals encode cleanly (AUC ≥ 0.75)
- Expert cluster analysis reveals functional specialization
- Clear narrative for Jordan's social world model use case

### Stretch Outcome
- Relationship prediction shows signal (AUC > 0.60)
- Cross-dataset transfer works (DailyDialog → MELD)
- Short paper draft for workshop submission

---

## Cost Management

**Budget:** <$50 total

**Cost killers to avoid:**
- Don't leave Modal functions idling
- Batch extractions aggressively (100+ samples per GPU call)
- Cache extracted logits locally
- Don't re-run extraction when you only need to re-train probes

---

## References

- Muennighoff et al. (2024). OLMoE. arXiv:2409.02060
- Lai et al. (2025). SAFEx. arXiv:2506.17368
- Danescu-Niculescu-Mizil et al. Echoes of Power. (Cornell)
- Li et al. (2017). DailyDialog. IJCNLP.
- Zhou et al. (2025). S3AP. arXiv:2509.00559
- Zhou et al. (2024). SOTOPIA. ICLR.

---

## Agent Workflow

1. **Read this document fully** before any implementation
2. **Check README.md** for current phase and progress
3. **Implement tasks** for current phase only
4. **Update README.md** checkboxes after completing tasks
5. **Document blockers** in README.md if stuck
6. **Do not create** separate UPDATE.md, CHANGELOG.md, or STATUS.md files

---

*This document is the source of truth for project architecture and requirements.*
