import argparse
import importlib.metadata
import json
import logging
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dcte import DCTE, MSMLoss
from ectf import EndpointConditionedTrajectoryFlow
from trajlik.model import TrajLikHead
from trajlik.normal_tail import EmpiricalTailCalibrator
from trajlik.trajectory_cache_dataset import (
    TrajectoryCacheDataset,
    balanced_category_sampler,
    stratified_normal_split,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DCTE + ECTF on a normal-only trajectory cache",
    )
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--calibration_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--trajectory_dim", type=int, default=64)
    parser.add_argument("--dcte_layers", type=int, default=2)
    parser.add_argument("--dcte_heads", type=int, default=4)
    parser.add_argument("--flow_blocks", type=int, default=4)
    parser.add_argument("--flow_bins", type=int, default=8)
    parser.add_argument("--lambda_msm", type=float, default=1.0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_head(input_dim, args):
    dcte = DCTE(
        input_dim=input_dim,
        projection_dim=args.projection_dim,
        token_dim=args.token_dim,
        trajectory_dim=args.trajectory_dim,
        num_steps=3,
        num_heads=args.dcte_heads,
        num_layers=args.dcte_layers,
        feedforward_dim=4 * args.token_dim,
        dropout=0.0,
    )
    ectf = EndpointConditionedTrajectoryFlow(
        trajectory_dim=args.trajectory_dim,
        z0_dim=input_dim,
        condition_dim=64,
        global_dim=32,
        position_dim=16,
        conditioner_hidden_dim=128,
        coupling_hidden_dim=128,
        num_blocks=args.flow_blocks,
        num_bins=args.flow_bins,
    )
    msm_loss = MSMLoss(
        input_dim=input_dim,
        projection_dim=args.projection_dim,
        token_dim=args.token_dim,
        trajectory_dim=args.trajectory_dim,
    )
    return TrajLikHead(
        dcte=dcte,
        ectf=ectf,
        msm_loss=msm_loss,
        lambda_msm=args.lambda_msm,
    )


def package_versions():
    names = ("torch", "torchvision", "numpy", "scipy", "scikit-learn")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@torch.no_grad()
def fit_calibrator(head, loader, device):
    head.eval()
    endpoint_scores = []
    path_scores = []
    progress = tqdm(
        loader,
        desc="Fitting calibration",
        unit="batch",
        leave=False,
        disable=not logger.isEnabledFor(logging.INFO),
    )
    for batch in progress:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = head(batch, mask=False)
        endpoint_scores.append(output["a_end_coarse"].cpu())
        path_scores.append(output["path_nll"].cpu())
    return EmpiricalTailCalibrator().fit(
        torch.cat(endpoint_scores, dim=0),
        torch.cat(path_scores, dim=0),
    )


def load_trajlik_checkpoint(checkpoint_path, device="cpu"):
    """Rebuild a head and calibrator from a self-describing checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    args = SimpleNamespace(**checkpoint["training_args"])
    input_dim = int(checkpoint["cache_metadata"]["output_channels"])
    head = build_head(input_dim, args).to(device)
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    head.eval()
    calibrator = EmpiricalTailCalibrator.from_state_dict(
        checkpoint["calibrator_state_dict"]
    ).to(device)
    return head, calibrator, checkpoint


def train(args):
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_clip <= 0:
        raise ValueError("epochs, batch_size, and grad_clip must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = TrajectoryCacheDataset(args.cache_dir)
    if int(dataset.metadata["num_steps"]) != 3:
        raise ValueError("Main TrajLik protocol requires exactly three inversion steps")
    if dataset.metadata.get("projection") != "none":
        raise ValueError(
            "Online-equivalent head training currently requires projection=none. "
            "Projected caches need their projector reapplied during online "
            "inference and must not be used silently."
        )

    training_indices, calibration_indices = stratified_normal_split(
        dataset.categories,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    training_subset = Subset(dataset, training_indices.tolist())
    calibration_subset = Subset(dataset, calibration_indices.tolist())
    sampler = balanced_category_sampler(
        dataset.categories,
        training_indices.tolist(),
        seed=args.seed,
    )
    training_loader = DataLoader(
        training_subset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
    )
    calibration_loader = DataLoader(
        calibration_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    input_dim = int(dataset.metadata["output_channels"])
    head = build_head(input_dim, args).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in head.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainable_parameters = sum(
        parameter.numel() for parameter in head.parameters() if parameter.requires_grad
    )
    logger.info(
        "Training TrajLik | device=%s | cache=%s | train=%d | calibration=%d "
        "| categories=%d | channels=%d | trainable_params=%d",
        device,
        args.cache_dir,
        len(training_subset),
        len(calibration_subset),
        len(set(dataset.categories)),
        input_dim,
        trainable_parameters,
    )

    training_history = []
    for epoch in range(args.epochs):
        head.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        epoch_totals = {"loss": 0.0, "nll_loss": 0.0, "msm_loss": 0.0}
        num_batches = 0
        progress = tqdm(
            training_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            leave=False,
            disable=not logger.isEnabledFor(logging.INFO),
        )
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = head.training_loss(batch)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            batch_metrics = {
                key: float(output[key].detach().item()) for key in epoch_totals
            }
            for key, value in batch_metrics.items():
                epoch_totals[key] += value
            num_batches += 1
            progress.set_postfix(
                loss=f"{batch_metrics['loss']:.4f}",
                nll=f"{batch_metrics['nll_loss']:.4f}",
                msm=f"{batch_metrics['msm_loss']:.4f}",
            )

        epoch_seconds = time.perf_counter() - epoch_started
        averages = {
            key: value / max(num_batches, 1)
            for key, value in epoch_totals.items()
        }
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        epoch_record = {
            "epoch": epoch + 1,
            **averages,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "duration_seconds": epoch_seconds,
            "peak_memory_mb": peak_memory_mb,
        }
        training_history.append(epoch_record)
        logger.info(
            "Epoch %d/%d | loss=%.6f | nll=%.6f | msm=%.6f | lr=%.2e "
            "| time=%.1fs | peak_mem=%.1f MB",
            epoch + 1,
            args.epochs,
            averages["loss"],
            averages["nll_loss"],
            averages["msm_loss"],
            optimizer.param_groups[0]["lr"],
            epoch_seconds,
            peak_memory_mb,
        )

    calibrator = fit_calibrator(head, calibration_loader, device)
    logger.info(
        "Fitted normal calibration on %d held-out images",
        len(calibration_subset),
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "head_state_dict": head.state_dict(),
        "calibrator_state_dict": calibrator.state_dict(),
        "training_args": vars(args),
        "cache_metadata": dataset.metadata,
        "training_indices": training_indices,
        "calibration_indices": calibration_indices,
        "training_history": training_history,
        "package_versions": package_versions(),
    }
    torch.save(checkpoint, output_path)
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "training_args": vars(args),
                "cache_metadata": dataset.metadata,
                "package_versions": checkpoint["package_versions"],
                "num_training_images": training_indices.numel(),
                "num_calibration_images": calibration_indices.numel(),
                "training_history": training_history,
            },
            file,
            indent=2,
        )
    logger.info("Saved TrajLik head and normal calibration to %s", output_path)
    return checkpoint


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train(parse_args())


if __name__ == "__main__":
    main()
