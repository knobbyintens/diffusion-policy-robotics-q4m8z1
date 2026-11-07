# Diffusion Policy for Robotic Manipulation

Official implementation of the paper **Diffusion Policy for Robotic Manipulation** (CoRL 2024).

This repository contains the complete, released implementation used for all
experiments reported in the paper, including data preparation, model code and
evaluation scripts.

## Repository layout

- `src/` - model definitions, training loop and rollout utilities
- `data/` - dataset adapters and preprocessing scripts
- `experiments/` - configuration files and launch scripts

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
bash experiments/run_all.sh
```

## Citation

```bibtex
@inproceedings{rivera2024diffusion,
  title={Diffusion Policy for Robotic Manipulation},
  author={Rivera, Janet and Chen, Marcus and Nair, Priya},
  booktitle={CoRL},
  year={2024}
}
```
