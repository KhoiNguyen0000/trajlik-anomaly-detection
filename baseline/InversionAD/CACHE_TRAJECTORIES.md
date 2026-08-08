# Trajectory Cache

This document describes how to cache normal DDIM inversion trajectories for the
TrajLik-AD training pipeline.

## Overview

Trajectory caching takes a trained InvAD checkpoint as input. The backbone and
denoiser remain frozen while every normal training image is passed through the
feature extractor and a three-step DDIM inversion.

For each image, the cache stores:

| Key | Shape | Description |
| --- | --- | --- |
| `z_0` | `(C, H, W)` | Initial backbone feature tensor |
| `z_seq` | `(S, C, H, W)` | Intermediate inversion states |
| `eps_seq` | `(S, C, H, W)` | Predicted noise at each step |
| `delta_z_seq` | `(S, C, H, W)` | Displacement between consecutive states |

The default inversion schedule uses `S = 3` steps. Only normal images from the
training split are cached. Test images and anomalous images must not be included.

## Pipeline

```text
Trained InvAD checkpoint + normal training images
    -> frozen backbone extracts z_0
    -> frozen denoiser runs three-step DDIM inversion
    -> optional projection and dtype conversion
    -> one trajectory cache file per image
```

## Configuration

For a single-class experiment, use a single-class dataset configuration:

```yaml
data:
  dataset_name: mvtec_ad
  category: bottle
  train: true
```

The checkpoint and cached images must belong to the same class.

For a unified multi-class experiment, use:

```yaml
data:
  dataset_name: mvtec_ad_all
  train: true
```

In this case, use one InvAD checkpoint trained on all classes. The same
checkpoint and the same projection weights must be used for every class.

## Recommended Baseline

First create an uncompressed FP32 cache. This is the reference used to validate
all storage optimizations:

```bash
python -m src.cache_trajectories \
    --config configs/exp_dit_ad/all.yml \
    --save_dir results/exp_dit_gigant_ad/all \
    --cache_dir cache/mvtec_ad_all/fp32 \
    --num_inversion_steps 3 \
    --projection none \
    --storage_dtype float32 \
    --use_ema_model \
    --device cuda:0
```

Use `--force` only when an existing cache directory should be replaced.

## Storage Modes

Projection and storage precision are independent options.

| Projection | Storage type | Result |
| --- | --- | --- |
| `none` | `float32` | Full information; reference cache |
| `none` | `float16` | Full channel count; approximately half the storage |
| `linear` | `float32` | Reduced channel count in FP32 |
| `linear` | `float16` | Reduced channel count and FP16 storage |

After obtaining the FP32 reference result, create an FP16 cache with:

```bash
python -m src.cache_trajectories \
    --config configs/exp_dit_ad/all.yml \
    --save_dir results/exp_dit_gigant_ad/all \
    --cache_dir cache/mvtec_ad_all/fp16 \
    --num_inversion_steps 3 \
    --projection none \
    --storage_dtype float16 \
    --use_ema_model \
    --device cuda:0
```

FP16 should become the default only after its scores and final evaluation
metrics are sufficiently close to the FP32 reference.

Linear projection reduces the feature channels, for example from 272 to 68:

```bash
--projection linear --proj_dim 68 --storage_dtype float16
```

Projection can discard information and should only be enabled after separate
validation. When projection is enabled, `projector.pt` is part of the model
artifacts and must be preserved.

## Output Layout

A multi-class cache may be organized as:

```text
cache/mvtec_ad_all/fp32/
|-- bottle_good_000.pt
|-- bottle_good_001.pt
|-- cable_good_000.pt
|-- ...
|-- projector.pt          # present only for linear projection
`-- cache_meta.json
```

Each image file contains a dictionary similar to:

```python
{
    "z_0": torch.Tensor,
    "z_seq": torch.Tensor,
    "eps_seq": torch.Tensor,
    "delta_z_seq": torch.Tensor,
    "source_path": "data/mvtec_ad/bottle/train/good/000.png",
    "category": "bottle",
}
```

With three inversion steps, no projection, and EfficientNet-B4 features, the
expected shapes are:

```text
z_0:         (272, 16, 16)
z_seq:       (3, 272, 16, 16)
eps_seq:     (3, 272, 16, 16)
delta_z_seq: (3, 272, 16, 16)
```

## Verification

Run the cache verification script from the `baseline/InversionAD` directory:

```bash
python test/cache_trajectories_check.py \
    --cache_dir cache/mvtec_ad_all/fp32
```

Categories can also be checked explicitly:

```bash
python test/cache_trajectories_check.py \
    --cache_dir cache/mvtec_ad_all/fp32 \
    --expected_categories bottle cable capsule carpet grid hazelnut leather \
        metal_nut pill screw tile toothbrush transistor wood zipper
```

The script verifies that:

- The number of cached image files matches the normal training set.
- Every sequence contains exactly three steps.
- All samples use the requested storage type and channel count.
- No anomalous or test image appears in the cache.
- A multi-class cache contains samples from every expected class.

## Multi-Class Cache

For a unified cache, the script reads normal training images from all configured
classes and processes them with one shared all-class InvAD checkpoint. Category
names are included in cache filenames and sample metadata to prevent collisions.

Do not combine caches produced by independently trained single-class InvAD
checkpoints into one unified cache. A unified cache must be generated by one
shared all-class checkpoint.
