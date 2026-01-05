# Relational State Estimation from MoE Router Logits

Research code for ["What Routers Know (And Don't)"](https://shiraeis.substack.com/p/what-routers-know-and-dont). We probe 64-dimensional MoE router logits from OLMoE-1B-7B to test whether routing decisions encode relational signals—properties of speaker relationships rather than individual text properties. Key finding: router logits achieve 94-99% of residual stream classification performance (32× compression) for intent, emotion, and tension, but weakly encode speaker power/status. MoE routing appears optimized for *content type*, not *social context*.

---

## Key Results

| Signal | Router AUC | Residual AUC | Retention | Notes |
|--------|-----------|--------------|-----------|-------|
| Intent (4-class) | 0.841 | 0.877 | 96% | Dialogue acts: inform, question, directive, commissive |
| Emotion (7-class) | 0.879 | 0.938 | 94% | DailyDialog emotion labels |
| Power/Wikipedia (binary) | 0.608 | 0.677 | 90% | Admin vs non-admin (inconclusive—see below) |
| Power/Enron (binary) | TBD | TBD | TBD | Downward vs upward communication |
| **Tension (3-class)** | **0.995** | **1.000** | **99.5%** | Escalation vs repair vs neutral |

**Insight:** Router logits strongly encode *what is being said* (intent, emotion, tension dynamics). The weak Wikipedia Talk result may reflect that admin status is a noisy proxy for power signals rather than an architectural limitation. Routers may encode power *when exercised* (directives, deference) but not speaker identity *per se*. Enron experiments (downward vs upward communication) will clarify whether routers distinguish "directive from senior" vs "request from junior."

---

## Repository Structure

```
├── AGENTS.md                 # AI agent reference doc (research notes)
├── LICENSE                   # MIT License
├── README.md                 # This file
├── requirements.txt          # Pinned dependencies
├── .env.example              # Environment variable template
│
├── src/
│   ├── data/                 # Dataset loaders
│   │   ├── dailydialog.py    # DailyDialog (intent, emotion)
│   │   ├── wikipedia_talk.py # Wikipedia Talk Pages (power/speaker identity)
│   │   ├── enron.py          # Enron emails (power/communication direction)
│   │   ├── synthetic_tension.py  # Claude-generated tension pairs
│   │   ├── sotopia.py        # SOTOPIA (not implemented)
│   │   └── meld.py           # MELD (not implemented)
│   │
│   ├── routing/              # Router logit extraction
│   │   ├── extraction.py     # Local extraction + caching
│   │   ├── aggregation.py    # Token→example pooling
│   │   └── modal_app.py      # Modal GPU extraction (main pipeline)
│   │
│   ├── probing/              # Linear probe training
│   │   ├── linear_probe.py   # Core probe utilities
│   │   ├── intent_probe.py   # Dialogue act probing
│   │   ├── emotion_probe.py  # Emotion probing
│   │   ├── power_probe.py    # Power differential probing
│   │   └── tension_probe.py  # Tension dynamics probing
│   │
│   └── analysis/             # Results analysis (partially implemented)
│       ├── visualization.py  # Figure generation
│       └── ...
│
├── scripts/                  # Experiment entry points
│   ├── run_intent_probes.py  # Phase 1: intent/emotion probes
│   ├── run_power_probes.py   # Phase 2: power probes (stub)
│   ├── generate_tension_pairs.py  # Generate synthetic data (stub)
│   └── generate_figures.py   # Figure generation (stub)
│
├── data/                     # Datasets (mostly gitignored)
│   ├── README.md             # Download instructions
│   └── synthetic/            # Generated tension pairs
│       └── tension_pairs.json
│
├── results/                  # Experiment outputs (gitignored)
│   ├── extractions/          # Cached router logits
│   ├── figures/              # Generated plots
│   └── tables/               # JSON probe results
│
└── tests/                    # Unit tests
```

---

## Reproduction Instructions

### Prerequisites

1. **Python 3.10+** with a virtual environment
2. **Modal account** for GPU compute (free tier works)
3. **Anthropic API key** (only for tension pair generation)

### Setup

```bash
git clone https://github.com/shiraeisenberg/relational-router-probes.git
cd relational-router-probes

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Modal setup (one-time)
modal setup

# Environment variables
cp .env.example .env
# Edit .env if you need to generate tension pairs (requires ANTHROPIC_API_KEY)
```

### Phase 1: DailyDialog Intent/Emotion Probes

**What it does:** Extracts router logits from OLMoE, trains linear probes for dialogue act and emotion classification, compares to residual stream baseline.

```bash
# 1. Run extraction on Modal (~35 min for 10K samples, ~$1-2)
modal run src/routing/modal_app.py --extract --dataset dailydialog --split train --max-samples 10000

# 2. List available caches
modal run src/routing/modal_app.py --show-caches

# 3. Run probes on cached extractions
modal run src/routing/modal_app.py --probe --cache-name <cache_name_from_step_2> --task intent
modal run src/routing/modal_app.py --probe --cache-name <cache_name_from_step_2> --task emotion
```

**Expected output:** JSON results saved to Modal Volume; AUC printed to console.

### Phase 2: Wikipedia Talk Power Probes

**What it does:** Extracts router logits for admin/non-admin utterances, trains binary classification probes.

```bash
# 1. Extract training data (~$1)
modal run src/routing/modal_app.py --extract --dataset wikipedia_talk --split train --max-samples 5000

# 2. Extract validation data (~$0.50)
modal run src/routing/modal_app.py --extract --dataset wikipedia_talk --split validation --max-samples 1000

# 3. Run power probes
modal run src/routing/modal_app.py --power-probe \
  --train-cache <train_cache_name> \
  --eval-cache <validation_cache_name>
```

**Expected output:** Train/eval AUC for each layer; expect ~0.60-0.68 AUC (weak signal).

### Synthetic Tension Pair Generation

**What it does:** Uses Claude API to generate 500 two-turn dialogue pairs labeled as escalation/repair/neutral.

```bash
# Requires ANTHROPIC_API_KEY in .env
python -c "
from src.data.synthetic_tension import generate_tension_pairs
pairs, stats = generate_tension_pairs(n_per_class=167, verbose=True)
"
```

**Output:** `data/synthetic/tension_pairs.json` (~500 pairs)

### Tension Probes

```bash
# 1. Extract tension pair embeddings
modal run src/routing/modal_app.py --extract --dataset tension --split train

# 2. Run tension probes
modal run src/routing/modal_app.py --tension-probe --cache-name <tension_cache_name>
```

**Expected output:** AUC ~0.99 (router captures tension dynamics very well).

### Enron Power Probes (Communication Direction)

**What it does:** Tests whether routers encode power *when exercised* — distinguishing emails from senior→junior (directives, delegation) vs junior→senior (requests, deference). This is a stronger test than Wikipedia Talk, which only tests speaker identity.

```bash
# 1. Extract training data (~$0.80)
modal run src/routing/modal_app.py --extract --dataset enron --split train --max-samples 5000

# 2. Extract validation data (~$0.20)
modal run src/routing/modal_app.py --extract --dataset enron --split validation --max-samples 1000

# 3. Run power probes (communication direction: downward vs upward)
modal run src/routing/modal_app.py --power-probe \
  --train-cache <enron_train_cache_name> \
  --eval-cache <enron_validation_cache_name>
```

**Expected output:** If H3b holds (AUC ≥ 0.70), routers encode power through linguistic markers. If similar to Wikipedia (~0.60), suggests routers truly don't encode relational power dynamics.

### Figure Generation

Pre-generated figures are in `results/figures/`. To regenerate:

```bash
python scripts/generate_figures.py --results-dir results/tables --output-dir results/figures
```

---

## Compute Costs & Runtimes

| Task | Samples | GPU | Time | Cost |
|------|---------|-----|------|------|
| DailyDialog extraction | 10,000 | A10G | ~35 min | ~$1.50 |
| Wikipedia Talk extraction | 5,000 | A10G | ~18 min | ~$0.80 |
| Enron extraction | 5,000 | A10G | ~18 min | ~$0.80 |
| Tension pair extraction | 500 | A10G | ~3 min | ~$0.15 |
| Probe training | any | CPU | ~1 min | ~$0.02 |
| **Total reproduction** | — | — | ~1.5 hr | **<$6** |

All extraction runs on Modal serverless. Probing is CPU-only and cheap.

---

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{eisenberg2026routerlogits,
  author = {Eisenberg, Shira},
  title = {Router Logits Encode Relational Dynamics},
  year = {2026},
  howpublished = {Hidden State Substack},
  url = {https://substack.com/placeholder}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **OLMoE** (Muennighoff et al., 2024) for the open MoE architecture
- **DailyDialog** (Li et al., 2017) for dialogue act/emotion annotations
- **ConvoKit** (Cornell) for Wikipedia Talk Pages corpus
- **Modal** for serverless GPU compute

---

## Notes

- `AGENTS.md` contains detailed research notes and is intentionally included
- The `extraction_10k_pooled.txt` file is a log from a prior extraction run
- Some modules (`src/data/meld.py`, `src/data/sotopia.py`) are stubs for future work
- Results may vary slightly with different random seeds
