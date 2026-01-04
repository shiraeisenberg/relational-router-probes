# Data Directory

This directory contains datasets for relational signal probing. Most subdirectories are gitignored (only `.gitkeep` files and `synthetic/tension_pairs.json` are committed).

## Structure

```
data/
├── dailydialog/      # DailyDialog (intent, emotion) — download required
├── wikipedia_talk/   # Wikipedia Talk Pages (power) — auto-downloaded via ConvoKit
├── sotopia/          # SOTOPIA logs (relationship) — not implemented
├── meld/             # MELD (transfer test) — not implemented
└── synthetic/        # Generated tension pairs — included
    └── tension_pairs.json
```

## What's Included

### Synthetic Tension Pairs (Committed)

`synthetic/tension_pairs.json` contains 501 Claude-generated two-turn dialogues:
- 167 escalation examples
- 167 repair examples  
- 167 neutral examples

These are used for tension probe training. To regenerate:
```bash
python scripts/generate_tension_pairs.py --n-pairs 500
```

## Download Instructions

### DailyDialog (Optional)

The loader defaults to HuggingFace Hub (`benjaminbeilharz/better_daily_dialog`), so no local download is required.

For local files (original format):
1. Download from http://yanran.li/dailydialog.html
2. Extract to this structure:
```
data/dailydialog/
├── train/
│   ├── dialogues_text.txt
│   ├── dialogues_act.txt
│   └── dialogues_emotion.txt
├── validation/
│   └── ...
└── test/
    └── ...
```

### Wikipedia Talk Pages (Auto-download)

ConvoKit downloads the corpus automatically on first use:
```python
from src.data.wikipedia_talk import load_wikipedia_talk
turns, stats = load_wikipedia_talk(n_samples=5000)
```

The corpus (~300MB) is cached by ConvoKit.

### MELD / SOTOPIA

Not implemented in this release. See stubs in `src/data/`.

## Label Mappings

### Dialogue Acts (DailyDialog)
| Code | Label |
|------|-------|
| 1 | inform |
| 2 | question |
| 3 | directive |
| 4 | commissive |

### Emotions (DailyDialog)
| Code | Label |
|------|-------|
| 0 | neutral |
| 1 | anger |
| 2 | disgust |
| 3 | fear |
| 4 | happiness |
| 5 | sadness |
| 6 | surprise |

### Tension (Synthetic)
| Label | Description |
|-------|-------------|
| escalation | Response increases conflict |
| repair | Response de-escalates |
| neutral | Neither escalates nor repairs |

### Power (Wikipedia Talk)
| Label | Description |
|-------|-------------|
| admin | Wikipedia administrator |
| non-admin | Regular editor |
