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
from debugpy.launcher import output
from jupyter_lsp import non_blocking
from sympy.integrals.meijerint_doc import category
from torch.utils.data import DataLoader
from tqdm import tqdm
from webencodings import labels

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
    in_channels = feature_shape[0]

    # denoiser
    denoiser = get_denoiser(
        **diffusion_config,
        input_shape = feature_shape,
    ).to(device).eval()

    #feature extractor
    feature_extractor = get_backbone(
        **config["backbone"],
    ).to(device).eval()

    checkpoint_path = load_checkpoint(
        denoiser,
        Path(args.save_dir),
        args.use_ema_model,
    )

    storage_dtype = resolve_storage_dtype(
        args.storage_dtype
    )

    projector = TrajectoryProjector(
        in_channels=in_channels,
        proj_dim=args.proj_dim,
        projection=args.projection,
        storage_dtype=storage_dtype
    ).to(device).eval()

    num_cached = 0

    for batch in tqdm(loader, desc="Caching trajectories"):
        images = batch["samples"].to(
            device,
            non_blocking=True,
        )

        labels = batch["clslabels"].to(
            device,
            non_blocking=True,
        )

        # Backbone's type still fp32
        z_0, _ = feature_extractor(images)

        start_t = torch.zeros(
            z_0.shape[0],
            dtype=torch.long,
            device= device,
        )

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _,z_seq,eps_seq, delta_z_seq = (
                denoiser.ddim_reverse_sample(
                    z_0,
                    start_t,
                    labels,
                    eta=0.0,
                    return_intermediates=True,
                )
            )

            projected = projector.project_and_compress(
                z_0,
                z_seq,
                eps_seq,
                delta_z_seq,
            )

        for index, source_path in enumerate(batch["filenames"]):
            category = batch["clsnames"][index]
            filename = sanitize_filename(source_path, category)

            output = {
                key: value[index].detach().cpu()
                for key, value in projected.items()
            }

            output["source_path"] = source_path
            output["category"] = category

            torch.save(output, cache_dir / f"{filename}.pt")
            num_cached += 1

    if args.projection == "linear":
        torch.save(
            projector.state_dict(),
            cache_dir / "projector.pt",
        )

    effective_channels = (
        args.proj_dim
        if args.projection == "linear"
        else in_channels
    )

    metadata = {
        "num_images": num_cached,
        "num_steps": args.num_inversion_steps,
        "in_channels": in_channels,
        "output_channels": effective_channels,
        "projection": args.projection,
        "proj_dim": (
            args.proj_dim
            if args.projection == "linear"
            else None
        ),
        "storage_dtype": args.storage_dtype,
        "backbone": config["backbone"]["model_type"],
        "dataset": dataset_config["dataset_name"],
        "category": dataset_config.get("category"),
    }

    with open(cache_dir / "cache_meta.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    logger.info("Cached %d images into %s", num_cached, cache_dir)

def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    cache_trajectories(config, args)


# HELPER CLASS
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

if __name__ == "__main__":
    main()
