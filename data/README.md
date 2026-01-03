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

## DailyDialog

**Source:** http://yanran.li/dailydialog.html

Download and extract to `dailydialog/`:
- `dialogues_text.txt` — Dialogue texts
- `dialogues_act.txt` — Dialogue act labels (1-4)
- `dialogues_emotion.txt` — Emotion labels (0-6)

## Wikipedia Talk Pages

**Source:** Cornell Conversational Analysis Toolkit

Download and extract editor metadata with admin status.

## SOTOPIA

**Source:** https://github.com/sotopia-lab/sotopia

Generate episodes using their pipeline and save to `sotopia/episodes/`.

## MELD

**Source:** https://github.com/declare-lab/MELD

Download CSV files to `meld/`.

## Synthetic Tension Pairs

Run `python scripts/generate_tension_pairs.py` to generate.
Pairs saved to `synthetic/tension_pairs.json`.
