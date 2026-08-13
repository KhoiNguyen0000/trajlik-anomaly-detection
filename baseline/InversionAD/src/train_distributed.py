
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from pathlib import Path

import copy
import argparse
import yaml
import time

from src.datasets import build_dataset
from src.utils import get_optimizer, get_lr_scheduler, init_distributed
from src.denoiser import get_denoiser, Denoiser
from src.backbones import get_backbone, get_backbone_feature_shape
import src.evaluate as evaluate

from einops import rearrange

try:
    import wandb
except ImportError:
    wandb = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
import multiprocessing as mp
import logging
import torch.distributed as dist

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class DistributedEvalSampler(Sampler):
    """Shard evaluation data without DistributedSampler's padding duplicates."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = max(0, len(self.dataset) - self.rank)
        return (remaining + self.num_replicas - 1) // self.num_replicas

def parse_args():
    parser = argparse.ArgumentParser(description="InvAD Training")
    
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')
    parser.add_argument(
        "--devices", type=str, nargs="+", default=["cuda:0"],
    )
    parser.add_argument(
        "--port", type=int, default=29500,
    )
    args = parser.parse_args()
    return args

def postprocess(x):
    x = x / 2 + 0.5
    return x.clamp(0, 1)

def convert2image(x):
    if x.dim() == 3:
        return x.permute(1, 2, 0).cpu().numpy()
    elif x.dim() == 4:
        return x.permute(0, 2, 3, 1).cpu().numpy()
    else:
        return x.cpu().numpy()

def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.info(exc)
    return config
    
def main(config):

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass
    
    # Initialize distributed training
    world_size, rank = init_distributed()
    logger.info(f"Initalized distributed training with world size {world_size} and rank {rank}")
    use_wandb = False
    if rank > 0:
        logger.setLevel(logging.ERROR)
    else:
        # Load environment variables from .env file
        try:
            if load_dotenv is not None:
                load_dotenv()
            use_wandb = (
                wandb is not None
                and os.getenv("WANDB_API_KEY") is not None
            )
            if use_wandb:
                wandb.login(key=os.getenv("WANDB_API_KEY"))
        except Exception:
            use_wandb = False
        if use_wandb:
            # create wandb project
            project = os.environ.get("WANDB_PROJECT")
            if project is None:
                raise ValueError("Please set the WANDB_PROJECT environment variable.")
            entity = os.environ.get("WANDB_ENTITY")
            if entity is None:
                raise ValueError("Please set the WANDB_ENTITY environment variable.")
            wandb.init(project=project, entity=entity, config=config)
    # set seed
    seed = config['meta']['seed'] + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_config = copy.deepcopy(config['data'])
    device = torch.device('cuda:0')
    batch_size = config['data']['batch_size']
    train_config = copy.deepcopy(config['data'])
    train_config.update(train=True, normal_only=True, anom_only=False)
    train_dataset = build_dataset(**train_config)
    dataset_config['train'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank)
    anom_samplers = [DistributedEvalSampler(
        dataset=anom_ds,
        num_replicas=world_size,
        rank=rank) for anom_ds in anom_dataset.datasets]
    normal_samplers = [DistributedEvalSampler(
        dataset=normal_ds,
        num_replicas=world_size,
        rank=rank) for normal_ds in normal_dataset.datasets]
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    )
    eval_batch_size = max(1, batch_size // world_size)
    anom_loaders = [torch.utils.data.DataLoader(
        anom_ds,
        sampler=anom_sampler,
        batch_size=eval_batch_size,
        pin_memory=True,
        num_workers=2,
        persistent_workers=False,
        drop_last=False,
    ) for anom_ds, anom_sampler in zip(anom_dataset.datasets, anom_samplers)]
    normal_loaders = [torch.utils.data.DataLoader(
        normal_ds,
        sampler=normal_sampler,
        batch_size=eval_batch_size,
        pin_memory=True,
        num_workers=2,
        persistent_workers=False,
        drop_last=False,
    ) for normal_ds, normal_sampler in zip(normal_dataset.datasets, normal_samplers)]
    
    diff_in_sh = get_backbone_feature_shape(model_type=config['backbone']['model_type'])
    logger.info(f"Using input shape {diff_in_sh} for the diffusion model")
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    ema_decay = config['diffusion']['ema_decay']
    model_ema = copy.deepcopy(model)
    model.to(device)
    model_ema.to(device)
    
    model = torch.nn.parallel.DistributedDataParallel(model, static_graph=True)
    model_ema = torch.nn.parallel.DistributedDataParallel(model_ema, static_graph=True)
    for p in model_ema.parameters():
        p.requires_grad = False

    backbone_kwargs = config['backbone']
    logger.info(f"Using feature space reconstruction with {backbone_kwargs['model_type']} backbone")
    
    feature_extractor = get_backbone(**backbone_kwargs)
    feature_extractor.to(device).eval()

    optimizer = get_optimizer([model], **config['optimizer'])
    scheduler = None
    if config['optimizer']['scheduler_type'] != 'none':
        scheduler = get_lr_scheduler(optimizer, **config['optimizer'], iter_per_epoch=len(train_loader))

    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # save config
    if rank == 0:
        save_path = save_dir / "config.yaml"
        with open(save_path, 'w') as f:
            yaml.dump(config, f)
        logger.info(f"Config is saved at {save_path}")
    
    model.train()
    logger.info(f"Steps per epoch: {len(train_loader)}")
    
    es_count = 0
    import time

    for epoch in range(config['optimizer']['num_epochs']):
        train_sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader):
            # --- timing start ---
            t0 = time.time()

            # Data loading
            t1 = time.time()
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            img = img.to(device)
            labels = labels.to(device)
            t2 = time.time()

            # Feature extraction (forward pass of backbone)
            with torch.no_grad():
                x, _ = feature_extractor(img)  # (B, c, h, w)
            t3 = time.time()

            # Model forward and backward
            loss = model(x, labels)  
            optimizer.zero_grad()
            loss.backward()
            if config['optimizer']['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['grad_clip'])
            optimizer.step()
            t4 = time.time()

            # Scheduler update
            if scheduler is not None:
                scheduler.step()

            # EMA update
            for ema_param, model_param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1.0 - ema_decay)
            t5 = time.time()

            # Logging
            if i % config["logging"]["log_interval"] == 0 and rank == 0:
                learning_rate = optimizer.param_groups[0]["lr"]
                logger.info(f"Epoch {epoch}, Iter {i}, Loss {loss.item():.4f}, LR {learning_rate:.6f}")
                if use_wandb:
                    wandb.log({
                        "Loss": loss.item(),
                        "LR": learning_rate,
                        "Time/Data [ms]": (t2 - t1) * 1000,
                        "Time/Forward [ms]": (t3 - t2) * 1000,
                        "Time/Backward [ms]": (t4 - t3) * 1000,
                        "Time/Total [ms]": (t5 - t0) * 1000
                    })

        if (epoch + 1) % config["logging"]["save_interval"] == 0 and rank == 0:
            save_path = save_dir / f"model_latest_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            save_path = save_dir / f"model_ema_latest.pth"
            torch.save(model_ema.state_dict(), save_path)
            logger.info(f"Model is saved at {save_dir}")

        
        if (epoch + 1) % config["evaluation"]["eval_interval"] == 0:
            all_results = {}
            categories = [ds.category for ds in anom_dataset.datasets]
            for anom_loader, normal_loader in zip(anom_loaders, normal_loaders):
                logger.info(f"Evaluating on {anom_loader.dataset.category} dataset")
                metrics_dict = evaluate.evaluate_dist(
                    model,
                    feature_extractor,
                    anom_loader,
                    normal_loader,
                    config, 
                    diff_in_sh,
                    epoch + 1,
                    config["evaluation"]["eval_step"],
                    device,
                    world_size=world_size,
                    rank=rank,
                )
                if rank == 0:
                    all_results.update(metrics_dict)
                evaluate.distributed_barrier()
            
            if rank == 0:
                # Compute averages only on the process that owns gathered results.
                avg_results = {}
                keys = ["I-AUROC", "I-AP", "I-F1Max", "P-AUROC", "P-AP", "P-F1Max", "PRO", "mAD"]
                for key in keys:
                    avg_results[key] = np.mean(
                        [all_results[cat][key] for cat in all_results]
                    )
                logger.info(f"Average results: {avg_results}")
                current_auc = avg_results["I-AUROC"]

                if use_wandb:
                    for cat in categories:
                        wandb.log({
                            f"{cat}/I-AUROC": all_results[cat]["I-AUROC"],
                            f"{cat}/I-AP": all_results[cat]["I-AP"],
                            f"{cat}/I-F1Max": all_results[cat]["I-F1Max"],
                            f"{cat}/P-AUROC": all_results[cat]["P-AUROC"],
                            f"{cat}/P-AP": all_results[cat]["P-AP"],
                            f"{cat}/P-F1Max": all_results[cat]["P-F1Max"],
                            f"{cat}/PRO": all_results[cat]["PRO"],
                            f"{cat}/mAD": all_results[cat]["mAD"]
                        })
                    
                    wandb.log({
                        "I-AUROC": current_auc,
                        "I-AP": avg_results["I-AP"],
                        "I-F1Max": avg_results["I-F1Max"],
                        "P-AUROC": avg_results["P-AUROC"],
                        "P-AP": avg_results["P-AP"],
                        "P-F1Max": avg_results["P-F1Max"],
                        "PRO": avg_results["PRO"],
                        "mAD": avg_results["mAD"]
                    })
                logger.info(f"AUC: {current_auc} at epoch {epoch}")
            
            evaluate.distributed_barrier()
    logger.info("Training is done!")
    
    # save model
    if rank == 0:
        save_path = save_dir / "model_latest.pth"
        torch.save(model.state_dict(), save_path)
        save_path = save_dir / "model_ema_latest.pth"
        torch.save(model_ema.state_dict(), save_path)
        logger.info(f"Model is saved at {save_dir}")


    
    
    
        
      
            
    
        
            
            
            
            
            
            
            
            
    
    
