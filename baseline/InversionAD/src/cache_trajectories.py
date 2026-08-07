import argparse
import copy
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import diffusion
from src.backbones import get_backbone, get_backbone_feature_shape
from src.datasets import build_dataset
from src.denoiser import get_denoiser
from src.trajectory_projector import TrajectoryProjector

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--num_inversion_steps", type=int, default=3)
    parser.add_argument("--proj_dim", type=int, default=68)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_ema_model", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--projection",
        choices=["none", "linear"],
        default="linear",
        help="none: keep original channel; linear: project down to proj_dim",
    )
    parser.add_argument(
        "--storage_dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="FP16 or BF16 for .pt files (saves space)",
    )
    return parser.parse_args()


def prepare_cache_dir(cache_dir: Path, force: bool):
    if cache_dir.exists():
        if not force:
            raise FileExistsError(
                f"Cache directory already exists: {cache_dir}. "
                "Use --force to overwrite."
            )
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True)


def load_checkpoint(model, save_dir: Path, use_ema_model: bool):
    checkpoint_name = (
        "model_ema_latest.pth"
        if use_ema_model
        else "model_latest.pth"
    )
    checkpoint_path = save_dir / checkpoint_name

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    return checkpoint_path


@torch.no_grad()
def cache_trajectories(config: dict, args):
    config = copy.deepcopy(config)

    device_name = args.device or config["meta"]["device"]
    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not found ")
    # prepare cache directory
    cache_dir = Path(args.cache_dir)
    prepare_cache_dir(cache_dir, args.force)

    # only get normal set from dataset
    dataset_config = copy.deepcopy(config["data"])
    dataset_config.update(
        train=True,
        normal_only=True,
        anom_only=False,
    )

    dataset = build_dataset(**dataset_config)

    batch_size = args.batch_size or dataset_config["batch_size"]
    num_workers = (
        args.num_workers if args.num_workers is not None else dataset_config["num_workers"]
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=dataset_config.get("pin_memory", True),
    )

    # inversion step for cache model
    diffusion_config = copy.deepcopy(config["diffusion"])
    diffusion_config["num_sampling_steps"] = str(
        args.num_inversion_steps
    )

    feature_shape = get_backbone_feature_shape(
        model_type=config["backbone"]["model_type"]
    )



def sanitize_filename(path: str, category: str) -> str:
    path = Path(path)

    # Ex:
    # .../bottle/train/good/000.png
    # -> bottle_good_000
    stem = f"{category}_{path.parent.name}_{path.stem}"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem)


def resolve_storage_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    return mapping[name]
