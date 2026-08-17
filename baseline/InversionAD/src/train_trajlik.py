import argparse
import importlib.metadata
import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.project_root import add_project_root_to_path

add_project_root_to_path()

from dcte import DCTE, MSMLoss
from ectf import EndpointConditionedTrajectoryFlow
from normal_tail import EmpiricalTailCalibrator
from trajectory_cache_dataset import (
    TrajectoryCacheDataset,
    balanced_category_sampler,
    stratified_normal_split,
)
from trajlik import TrajLikHead


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
    for batch in loader:
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

    for epoch in range(args.epochs):
        head.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in training_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = head.training_loss(batch)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += output["loss"].item()
            num_batches += 1
        logger.info(
            "Epoch %d/%d normal head loss: %.6f",
            epoch + 1,
            args.epochs,
            epoch_loss / max(num_batches, 1),
        )

    calibrator = fit_calibrator(head, calibration_loader, device)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "head_state_dict": head.state_dict(),
        "calibrator_state_dict": calibrator.state_dict(),
        "training_args": vars(args),
        "cache_metadata": dataset.metadata,
        "training_indices": training_indices,
        "calibration_indices": calibration_indices,
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
            },
            file,
            indent=2,
        )
    logger.info("Saved TrajLik head and normal calibration to %s", output_path)
    return checkpoint


def main():
    logging.basicConfig(level=logging.INFO)
    train(parse_args())


if __name__ == "__main__":
    main()
