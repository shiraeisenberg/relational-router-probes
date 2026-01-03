# Datasets

This directory contains datasets for relational signal probing.

## Structure

```
data/
├── dailydialog/      # DailyDialog (intent, emotion)
├── wikipedia_talk/   # Wikipedia Talk Pages (power)
├── sotopia/          # SOTOPIA logs (relationship)
├── meld/             # MELD (transfer test)
└── synthetic/        # Generated tension pairs
```

---

## DailyDialog

**Source:** http://yanran.li/dailydialog.html  
**Paper:** Li et al. (2017) "DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset"

### Download Instructions

1. **Visit the official page:** http://yanran.li/dailydialog.html

2. **Download the dataset:** Click the download link (usually a zip file like `ijcnlp_dailydialog.zip`)

3. **Extract and organize:** After extracting, you should have files organized by split. Move them into this structure:

```
data/dailydialog/
├── train/
│   ├── dialogues_text.txt
│   ├── dialogues_act.txt
│   └── dialogues_emotion.txt
├── validation/
│   ├── dialogues_text.txt
│   ├── dialogues_act.txt
│   └── dialogues_emotion.txt
└── test/
    ├── dialogues_text.txt
    ├── dialogues_act.txt
    └── dialogues_emotion.txt
```

**Note:** The downloaded archive may have a different folder structure (e.g., `train/`, `validation/`, `test/` inside an `ijcnlp_dailydialog/` folder). Just ensure the final structure matches the above.

### File Format

- **dialogues_text.txt**: One dialogue per line. Turns are separated by ` __eou__ ` (End Of Utterance).
- **dialogues_act.txt**: One line per dialogue. Space-separated integers (1-4) for each turn's dialogue act.
- **dialogues_emotion.txt**: One line per dialogue. Space-separated integers (0-6) for each turn's emotion.

### Label Mappings

**Dialogue Acts (1-4):**
| Code | Label |
|------|-------|
| 1 | inform |
| 2 | question |
| 3 | directive |
| 4 | commissive |

**Emotions (0-6):**
| Code | Label |
|------|-------|
| 0 | neutral |
| 1 | anger |
| 2 | disgust |
| 3 | fear |
| 4 | happiness |
| 5 | sadness |
| 6 | surprise |

### Verification

After downloading, run the loader to verify:

```python
from src.data.dailydialog import load_dailydialog

turns, stats = load_dailydialog(split="train")
print(f"Loaded {stats['n_dialogues']} dialogues, {stats['n_turns']} turns")
```

---

## Wikipedia Talk Pages

**Source:** Cornell Conversational Analysis Toolkit

Download and extract editor metadata with admin status.

---

## SOTOPIA

**Source:** https://github.com/sotopia-lab/sotopia

Generate episodes using their pipeline and save to `sotopia/episodes/`.

---

## MELD

**Source:** https://github.com/declare-lab/MELD

Download CSV files to `meld/`.

---

## Synthetic Tension Pairs

Run `python scripts/generate_tension_pairs.py` to generate.
Pairs saved to `synthetic/tension_pairs.json`.
