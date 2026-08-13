# Trajectory Cache

This script creates normal-image DDIM trajectories from a trained InvAD model
for use by TrajLik-AD. It extracts EfficientNet features and caches `z_0`,
`z_seq`, `eps_seq`, and `delta_z_seq` for each training image.
Every sample is also marked with `split=train` and `is_normal=true`; the
validator uses these fields instead of dataset-specific path conventions.

## Inputs

- An InvAD YAML config containing the dataset path and model architecture.
- A matching checkpoint, selected with either `--save_dir` or
  `--checkpoint_path`.
- The normal training split of the dataset.

## Usage

Run from `baseline/InversionAD`.

Smoke test on eight images using the EMA checkpoint:

```bash
python -m src.cache_trajectories \
    --config configs/exp_dit_ad/all.yml \
    --save_dir results/exp_dit_base_ad/all \
    --cache_dir /kaggle/working/cache_smoke \
    --num_inversion_steps 3 \
    --batch_size 1 \
    --num_workers 0 \
    --max_images 8 \
    --projection none \
    --storage_dtype float32 \
    --autocast_dtype auto \
    --use_ema_model \
    --device cuda:0
```

For a full run, remove `--max_images` and change `--cache_dir`. To use a
specific checkpoint, replace `--save_dir` and `--use_ema_model` with:

```bash
--checkpoint_path /path/to/checkpoint.pth
```

The config architecture must match the checkpoint architecture.
The metadata records the preprocessing mode and a SHA-256 fingerprint of the
complete config so a cache cannot silently lose its provenance.

## Output

The cache directory contains one `.pt` file per image, `cache_meta.json`, and
`cache_index.json`. The index supports provenance checks and category-balanced
head training without loading all tensor files. With EfficientNet-B4, three
inversion steps, and no projection, each
`.pt` file contains:

```text
z_0:         (272, 16, 16)
z_seq:       (3, 272, 16, 16)
eps_seq:     (3, 272, 16, 16)
delta_z_seq: (3, 272, 16, 16)
a_end_coarse: (16, 16)
```

`a_end_coarse` is always computed from the original, unprojected final latent.
This preserves the official InvAD endpoint score when projection caching is on.

## Validation

```bash
python test/cache_trajectories_check.py \
    --cache_dir /kaggle/working/cache_smoke
```

Use `--force` only when the existing cache may be overwritten.

## Train DCTE + ECTF and fit normal calibration

```bash
python -m src.train_trajlik \
    --cache_dir /path/to/cache \
    --output_path results/trajlik/head.pth \
    --device cuda:0 \
    --epochs 50 \
    --batch_size 4
```

The command uses a category-stratified 95/5 normal split, balances training
categories, optimizes conditional NLL plus MSM, and fits both empirical CDFs
only on the held-out normal subset. The checkpoint records the exact arguments,
cache fingerprint, split indices, seed, and package versions.

The end-to-end online pipeline currently accepts only `--projection none`.
Linear projection caches remain an explicit storage experiment and are rejected
by head training until the same saved projector is wired into online inference.
