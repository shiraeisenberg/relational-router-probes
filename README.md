# Relational State Estimation from MoE Router Logits

**Do MoE router logits encode relational signals like power, emotion, and interpersonal dynamics?**

This project extends prior work showing router logits encode formality (AUC ≈ 1.0, 32× compression) to test whether they also encode **dyadic relational signals**—properties of speaker relationships rather than individual text.

## Project Status: Phase 1 — In Progress

**Last Updated:** 2026-01-03

---

## Research Questions

1. **Intent:** Do router logits encode dialogue acts (inform, question, directive, commissive)?
2. **Emotion:** Do router logits encode emotional valence (7-class)?
3. **Power:** Do router logits encode speaker status (admin vs non-admin)?
4. **Tension:** Do router logits encode escalation vs repair dynamics?
5. **Relationship:** Can early-turn routing predict conversation outcomes?

## Prior Findings (Formality Sprint)

| Finding | Result |
|---------|--------|
| Router logits encode formality | AUC ≈ 1.0 |
| Compression ratio | 32× (64-dim vs 2048-dim) |
| Expert specialization | Disjoint clusters (0 overlap) |
| Mediation index | M ≈ 1.0 (epiphenomenal) |
| Expert ablation | -29% formality when ablated |

**Implication:** Router logits are excellent for **classification** but not for **steering**. This sprint extends the classification capability to relational signals.

---

## Architecture

```
OLMoE-1B-7B
├── 16 MoE transformer layers
├── 64 experts per layer
├── Router: model.model.layers[i].mlp.gate
└── Router logits: (seq_len, 64)

We probe router logits → relational labels
```

---

## Current Progress

### Phase 1: Infrastructure & Intent/Emotion Probes ✅ COMPLETE
- [x] Port formality probe code from prior project
- [x] Set up Modal pipeline for router logit extraction
- [x] Load DailyDialog dataset (HuggingFace integration)
- [x] Extraction caching (pre-pooled, ~235MB vs ~65GB)
- [x] Train intent probes (4-class) — **Val AUC 0.841**
- [x] Train emotion probes (7-class) — **Val AUC 0.879**
- [x] Compare router vs residual stream AUC
- [x] Train→Validation evaluation

**Key Finding:** Router logits (64-dim) retain 93-96% of residual stream (2048-dim) signal. 32× compression with minimal information loss.

#### Phase 1 Final Results (Train→Validation)

| Task | Train Router AUC | Val Router AUC | Val Residual AUC | Retention |
|------|------------------|----------------|------------------|-----------|
| Intent (4-class) | 0.905 | **0.841** | 0.877 | 96% |
| Emotion (7-class) | 0.849 | **0.879** | 0.938 | 94% |

**Hypotheses:**
- ✓ **H1 (Intent ≥0.90):** Achieved 0.905 train, 0.841 validation
- ✓ **H2 (Emotion 0.70-0.85):** Achieved 0.849 train, 0.879 validation

### Phase 2: Power & Dyadic Signals ← CURRENT
- [ ] Load Wikipedia Talk Pages (power labels)
- [ ] Train power differential probes
- [ ] Generate SOTOPIA interaction logs
- [ ] Train relationship prediction probes
- [ ] Generate synthetic tension pairs
- [ ] Train tension probes

### Phase 3: Analysis & Synthesis
- [ ] Comparative AUC table
- [ ] Layer-by-layer analysis
- [ ] Expert overlap heatmap
- [ ] Cross-dataset transfer tests
- [ ] Substack writeup

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/shiraeisenberg/relational-router-probes.git
cd relational-router-probes

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up Modal for GPU compute
pip install modal
modal setup

# Set environment variables
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY (for synthetic data generation)

# Run experiments (once implemented)
python scripts/run_intent_probes.py
```

---

## Project Structure

```
├── AGENTS.md                 # AI agent reference (read first)
├── README.md                 # This file
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data/                 # Dataset loaders
│   │   ├── __init__.py
│   │   ├── dailydialog.py    # DailyDialog (intent, emotion)
│   │   ├── wikipedia_talk.py # Wikipedia Talk Pages (power)
│   │   ├── sotopia.py        # SOTOPIA logs (relationship)
│   │   ├── meld.py           # MELD (transfer test)
│   │   └── synthetic_tension.py  # Generated tension pairs
│   │
│   ├── routing/              # Router logit extraction
│   │   ├── __init__.py
│   │   ├── extraction.py     # Hook-based logit capture
│   │   ├── aggregation.py    # Token→example pooling
│   │   └── modal_app.py      # Modal GPU extraction
│   │
│   ├── probing/              # Linear probe training
│   │   ├── __init__.py
│   │   ├── linear_probe.py   # Generic probe utilities
│   │   ├── intent_probe.py   # Dialogue act probing
│   │   ├── emotion_probe.py  # Emotion probing
│   │   ├── power_probe.py    # Power differential probing
│   │   ├── tension_probe.py  # Tension dynamics probing
│   │   └── relationship_probe.py  # SOTOPIA outcome prediction
│   │
│   └── analysis/             # Results analysis
│       ├── __init__.py
│       ├── expert_clusters.py    # Expert activation analysis
│       ├── comparative.py        # Cross-signal comparison
│       ├── layer_analysis.py     # Layer-by-layer probing
│       ├── transfer.py           # Cross-dataset transfer
│       ├── baselines.py          # BoW, punctuation baselines
│       └── visualization.py      # Figure generation
│
├── scripts/                  # Experiment runners
│   ├── run_intent_probes.py
│   ├── run_power_probes.py
│   ├── run_ablations.py
│   ├── generate_tension_pairs.py
│   └── generate_figures.py
│
├── data/                     # Datasets (gitignored, except schema)
│   ├── README.md
│   ├── dailydialog/
│   ├── wikipedia_talk/
│   ├── sotopia/
│   ├── meld/
│   └── synthetic/
│
├── results/                  # Experiment outputs
│   ├── probes/               # Trained probe checkpoints
│   ├── extractions/          # Cached router logits
│   ├── figures/              # Generated plots
│   └── tables/               # CSV results
│
├── docs/
│   └── writeup.md            # Substack draft
│
└── tests/
    ├── __init__.py
    ├── test_extraction.py
    ├── test_probing.py
    └── test_data_loaders.py
```

---

## Datasets

| Dataset | Size | Labels | Signal | Priority |
|---------|------|--------|--------|----------|
| **DailyDialog** | 13K dialogues | Act (4), Emotion (7) | Intent, Emotion | Tier 1 |
| **SOTOPIA** | 90 scenarios | 7-dim outcomes | Relationship | Tier 1 |
| **Synthetic** | ~500 pairs | Escalation/Repair | Tension | Tier 1 |
| Wikipedia Talk | 240K exchanges | Admin/non-admin | Power | Tier 2 |
| MELD | 13K utterances | Emotion (7) | Transfer test | Tier 2 |

---

## Infrastructure

| Component | Details |
|-----------|---------|
| **Model** | OLMoE-1B-7B |
| **Compute** | Modal serverless A10G |
| **Probes** | scikit-learn LogisticRegression |
| **Budget** | <$50 total |

---

## Success Criteria

| Level | Criteria |
|-------|----------|
| **Minimum** | Intent AUC ≥ 0.75, documented comparison |
| **Target** | 2+ signals AUC ≥ 0.75, expert specialization |
| **Stretch** | Relationship prediction, cross-dataset transfer |

---

## Implications for Social World Models

If router logits encode relational signals:

1. **Real-time social state estimation** at 32× less compute
2. **Features for S3AP parsing** (CMU social world model)
3. **Tension maintenance detection** for interactive fiction
4. **User fingerprinting** from stable routing patterns

---

## References

- [OLMoE](https://arxiv.org/abs/2409.02060) (Muennighoff et al., 2024)
- [SAFEx](https://arxiv.org/abs/2506.17368) (Lai et al., 2025)
- [Echoes of Power](https://www.cs.cornell.edu/~cristian/Echoes_of_power.html) (Cornell)
- [DailyDialog](http://yanran.li/dailydialog.html) (Li et al., 2017)
- [SOTOPIA](https://arxiv.org/abs/2310.11667) (Zhou et al., 2024)
- [S3AP](https://arxiv.org/abs/2509.00559) (Zhou et al., 2025)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Read [AGENTS.md](AGENTS.md) for detailed implementation guidance.*
