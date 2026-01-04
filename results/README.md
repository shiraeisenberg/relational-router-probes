# Results Directory

This directory contains experiment outputs. Most contents are gitignored.

## Structure

```
results/
├── extractions/     # Cached router logits (.npz files)
├── figures/         # Generated plots (.png, .svg)
├── probes/          # Trained probe checkpoints (.pkl)
└── tables/          # Probe results (.json)
```

## Regenerating Results

### Extractions

Run via Modal:
```bash
modal run src/routing/modal_app.py --extract --dataset dailydialog --split train --max-samples 10000
```

Cache files are saved to Modal Volume and can be listed with:
```bash
modal run src/routing/modal_app.py --show-caches
```

### Figures

Pre-generated figures are committed. To regenerate:
```bash
python scripts/generate_figures.py --results-dir results/tables --output-dir results/figures
```

### Probe Results

Probing runs on Modal with cached extractions:
```bash
modal run src/routing/modal_app.py --probe --cache-name <cache_name> --task intent
```

Results are printed to console and can be saved locally.

## Included Files

- `figures/*.png` — Pre-generated visualizations for writeup
- `extractions/test_cache.npz` — Small test cache for unit tests

