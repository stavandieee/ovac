# OVAC (Open-Vocabulary Aerial Coordination)

This repository contains code and paper-facing result summaries for reproducing the OVAC experiments.

## Reproducibility
See `OVAC_REPRODUCIBILITY_MANIFEST.md` for:
- exact commands to reproduce Tables 4–6
- dataset setup notes (VisDrone / DOTA)
- paths to saved JSON summaries under `results/`

### Quick commands (SAR)
```bash
# Table 4 (surrogate SAR evaluation)
PYTHONPATH="$(pwd)" python3 -u run_sar_trials.py --all --n-trials 100 --seed 0 --output-dir results/coordination

# Table 5 (threshold sweep)
PYTHONPATH="$(pwd)" python3 -u run_sar_trials.py --threshold-sweep --n-trials 100 --seed 0 --output-dir results/coordination

# Table 6 (plan quality)
PYTHONPATH="$(pwd)" python3 generate_table6.py
