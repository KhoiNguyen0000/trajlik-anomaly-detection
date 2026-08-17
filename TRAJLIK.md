# TrajLik-AD

## Description

TrajLik-AD trains DCTE and ECTF on cached normal InvAD trajectories, fits
normal-only score calibration, and combines endpoint and path anomaly scores at
evaluation time.

## Input

- A trajectory cache created by `scripts.cache_trajectories`
- The matching InvAD config and checkpoint
- MVTec AD, VisA, or MPDD for evaluation

The main pipeline requires three inversion steps and `projection=none`.

Core pipeline utilities live in `trajlik/`; DCTE and ECTF remain separate
top-level packages.

## Output

Training produces a TrajLik checkpoint and a JSON training summary. Evaluation
produces image-level and pixel-level metrics, with an optional JSON result file.
Training logs batch progress plus epoch-level total, NLL, MSM, learning rate,
duration, and peak GPU memory. The epoch history is also stored in both outputs.

## Requirements

- The packages in `requirements.txt`
- A validated normal-only cache containing `cache_meta.json` and
  `cache_index.json`
- A config matching both the InvAD and TrajLik checkpoints

## Usage

Run the following commands from the project root.

Train the TrajLik head:

```bash
python -m scripts.train_trajlik \
    --cache_dir /path/to/cache \
    --output_path results/trajlik/head.pth \
    --device cuda:0
```

Evaluate the complete pipeline:

```bash
python -m scripts.evaluate_trajlik \
    --config /path/to/config.yml \
    --invad_checkpoint /path/to/model.pth \
    --trajlik_checkpoint results/trajlik/head.pth \
    --output_json results/trajlik/metrics.json
```

Module details: [DCTE](dcte/README.md) and [ECTF](ectf/README.md).
