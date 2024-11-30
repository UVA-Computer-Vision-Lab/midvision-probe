"""
MIT License

Copyright (c) 2024 Mohamed El Banani
Copyright (c) 2024 Xuweiyi Chen  

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.functional import interpolate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.utils.losses import DepthLoss
from evals.utils.metrics import evaluate_depth, match_scale_and_shift
from evals.utils.optim import cosine_decay_linear_warmup
from PIL import Image
import matplotlib.pyplot as plt
import wandb
import numpy as np
import torchvision

wandb.require("core")
import csv

import matplotlib.cm as cm
import matplotlib.colors
import typing
import json


def ddp_setup(rank: int, world_size: int, port: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    n_epochs,
    detach_model,
    loss_fn,
    rank=0,
    world_size=1,
    valid_loader=None,
    scale_invariant=False,
    wandb_use=False,
    is_final=False,
    is_navi=False,
):
    for ep in range(n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0
        pbar = tqdm(train_loader) if rank == 0 else train_loader
        for i, batch in enumerate(pbar):
            images = batch["image"].to(rank)
            target = batch["depth"].to(rank)

            optimizer.zero_grad()
            if detach_model:
                with torch.no_grad():
                    feats = model(images)
                    if isinstance(feats, (tuple, list)):
                        feats = [_f.detach() for _f in feats]
                    else:
                        feats = feats.detach()
            else:
                feats = model(images)
            pred = probe(feats)
            pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

            if scale_invariant:
                pred = match_scale_and_shift(pred, target)
                pred = pred.clamp(min=0.001, max=1.0)

            if pred.shape == 16:
                batch_size = 16
                num_chunks = pred.shape[0] // batch_size  # 2

                losses = []
                for i in range(num_chunks):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size

                    pred_chunk = pred[start_idx:end_idx, :, :, :]
                    target_chunk = target[start_idx:end_idx, :, :, :]

                    loss_chunk = loss_fn(pred_chunk, target_chunk)
                    losses.append(loss_chunk)
                loss = sum(losses) / num_chunks
            else:
                loss = loss_fn(pred, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.item()
            train_loss += loss

            if rank == 0 and wandb_use:
                _loss = train_loss / (i + 1)
                pbar.set_description(
                    f"{ep} | loss: {loss:.4f} ({_loss:.4f}) probe_lr: {pr_lr:.2e}"
                )
                # Log metrics to WandB
                wandb.log({"train_loss": _loss, "probe_lr": pr_lr, "epoch": ep})

        train_loss /= len(train_loader)

        if rank == 0 and wandb_use:
            logger.info(f"train loss {ep}   | {train_loss:.4f}")
            # Log first batch images after each epoch
            log_first_batch_images(
                model, probe, valid_loader, rank, wandb_use=wandb_use, is_navi=is_navi
            )
            if valid_loader is not None and is_final:
                val_loss, val_metrics = validate(
                    model, probe, valid_loader, loss_fn, scale_invariant=scale_invariant
                )
                logger.info(f"valid loss {ep}   | {val_loss:.4f}")
                for metric in val_metrics:
                    logger.info(f"valid SA {metric:10s} | {val_metrics[metric]:.4f}")
                    # Log validation metrics to WandB
                    wandb.log({f"valid_{metric}": val_metrics[metric], "epoch": ep})


# Visualization and logging function for first batch
def log_first_batch_images(model, probe, loader, rank, wandb_use=False, is_navi=False):
    model.eval()  # Set model to evaluation mode
    probe.eval()

    pred_images, target_images = [], []
    with torch.inference_mode():
        batch = next(iter(loader))  # Get the first batch of the test set
        images = batch["image"].to(rank)
        target = batch["depth"].to(rank)

        feat = model(images)
        pred = probe(feat).detach()
        pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

        # Visualize and log first few images
        for i in range(min(8, pred.shape[0])):  # Log up to 8 images
            if is_navi:
                pred_colored, target_colored = visualize_depth_navi(pred[i], target[i])
            else:
                pred_colored, target_colored = visualize_depth(pred[i], target[i])
            pred_images.append(pred_colored)
            target_images.append(target_colored)

        if wandb_use:
            # Convert to torch tensors and log to WandB
            pred_images_tensor = torch.tensor(np.array(pred_images)).permute(0, 3, 1, 2)
            target_images_tensor = torch.tensor(np.array(target_images)).permute(
                0, 3, 1, 2
            )
            wandb.log(
                {
                    "predictions": [wandb.Image(pred) for pred in pred_images_tensor],
                    "targets": [wandb.Image(target) for target in target_images_tensor],
                }
            )


def save_images_to_png(
    pred,
    target,
    segmentation_map,
    batch_idx,
    task,
    save_dir,
    pred_type="sigmoid",
    colormap="inferno",
    is_navi=False,
    image_average=True,
    scale_invariant=False,
    nyu_crop=True,
    num_levels=5,
):
    """
    Saves predicted and target images as PNGs with RGB colormap applied and stores metrics for both global and centroid levels.

    Args:
    - pred (torch.Tensor): Predicted values (e.g., depth, normal).
    - target (torch.Tensor): Target values.
    - batch_idx (int): Batch index for saving filenames.
    - task (str): The task being performed (e.g., 'depth', 'normal').
    - save_dir (str): Directory where PNG files will be saved.
    - pred_type (str): Prediction type, affects the scaling (sigmoid, tanh).
    - colormap (str): Colormap to be applied to the images.
    - is_navi (bool): Flag indicating whether the task is for 'navi'.
    - image_average (bool): Whether to average the metrics across the image.
    - scale_invariant (bool): Whether to compute scale-invariant metrics.
    - nyu_crop (bool): Whether to apply the NYU crop.
    - num_levels (int): Number of centroid levels for reporting metrics.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving images and metrics to {save_dir}")

    for i in range(pred.shape[0]):  # Iterate through the batch
        # Compute metrics for each instance (image) BEFORE normalization
        single_pred = pred[i].unsqueeze(
            0
        )  # Reshape to match batch dimension for evaluate_depth
        single_target = target[i].unsqueeze(
            0
        )  # Reshape to match batch dimension for evaluate_depth
        if not is_navi:
            single_segmentation_map = segmentation_map[i].unsqueeze(
                0
            )  # Reshape to match batch dimension for evaluate_depth
        else:
            single_segmentation_map = None

        global_metrics, level_metrics, _ = evaluate_depth(
            single_pred,
            single_target,
            single_segmentation_map,
            image_average=image_average,
            scale_invariant=scale_invariant,
            nyu_crop=nyu_crop,
            num_levels=num_levels,
            is_navi=is_navi,
        )

        # Save instance-level global and level metrics to a text file
        metrics_path_txt = os.path.join(save_dir, f"metrics_{task}_{batch_idx}_{i}.txt")
        with open(metrics_path_txt, "w") as f:
            f.write(f"Metrics for image {i} in batch {batch_idx}:\n")
            f.write(f"Global Metrics:\n")
            for key in global_metrics:
                f.write(f"{key}: {global_metrics[key].item():.4f}\n")

            # Save metrics for each centroid level
            f.write("\nCentroid-Level Metrics:\n")
            for level in range(1, num_levels + 1):
                f.write(f"Level {level}:\n")
                for key in level_metrics[f"level_{level}"]:
                    f.write(
                        f"  {key}: {level_metrics[f'level_{level}'][key].item():.4f}\n"
                    )

        # Save instance-level metrics to a JSON file
        metrics_path_json = os.path.join(
            save_dir, f"metrics_{task}_{batch_idx}_{i}.json"
        )
        metrics_data = {
            "global_metrics": {
                key: global_metrics[key].item() for key in global_metrics
            },
            "level_metrics": {
                f"Level {level}": {
                    key: level_metrics[f"level_{level}"][key].item()
                    for key in level_metrics[f"level_{level}"]
                }
                for level in range(1, num_levels + 1)
            },
        }
        with open(metrics_path_json, "w") as json_file:
            json.dump(metrics_data, json_file, indent=4)

        # Visualization and image saving
        if is_navi:
            pred_colored, target_colored = visualize_depth_navi(
                pred[i], target[i], colormap=colormap
            )
        else:
            pred_colored, target_colored = visualize_depth(
                pred[i], target[i], colormap=colormap
            )
        pred_image = (pred_colored * 255).astype(np.uint8)
        target_image = (target_colored * 255).astype(np.uint8)

        # Save predicted and target images
        pred_image_path = os.path.join(save_dir, f"pred_{task}_{batch_idx}_{i}.png")
        target_image_path = os.path.join(save_dir, f"target_{task}_{batch_idx}_{i}.png")

        Image.fromarray(pred_image).save(pred_image_path)
        Image.fromarray(target_image).save(target_image_path)


def plot_segment_area_vs_d1(segment_metrics, output_dir="plots"):
    # Extract areas and d1 ratios from the segment_metrics list
    areas = [entry["area"] for entry in segment_metrics]
    d1_ratios = [entry["d1_ratio"] for entry in segment_metrics]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(areas, d1_ratios, alpha=0.6)
    plt.title("Segment Area vs. D1 Accuracy")
    plt.xlabel("Segment Area (pixels)")
    plt.ylabel("D1 Accuracy")
    plt.grid(True)

    # Generate a timestamp to create a unique filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot with a timestamp in the filename
    plot_filename = f"segment_area_vs_d1_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved as {plot_path}")


def validate(
    model,
    probe,
    loader,
    loss_fn,
    verbose=True,
    scale_invariant=False,
    aggregate=True,
    render_images=True,  # Flag to control image rendering
    wandb_use=False,
    is_navi=False,
    output_dir="result",
):
    total_loss = 0.0
    global_metrics = None
    level_metrics = None  # To hold level-wise metrics
    task = "depth"
    save_images_once = True  # Flag to ensure we save images only for the first batch
    all_segment_metrics = []  # To accumulate segment-wise metrics across batches

    # Create a timestamp-based directory for saving images
    if render_images:
        # TODO: update this
        model_name = (
            model.module.checkpoint_name.split("_")[0]
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model.checkpoint_name.split("_")[0]
        )
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Generate a timestamp
        save_dir = os.path.join(
            f"{output_dir}/{task}/{task}_images", f"{task}_{model_name}_{timestamp}"
        )
    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluation") if verbose else loader
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].cuda()
            target = batch["depth"].cuda()
            if not is_navi:
                segmentation_map = batch["segmentation"].cuda()
            else:
                segmentation_map = None

            feat = model(images)
            pred = probe(feat).detach()
            pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

            loss = loss_fn(pred, target)
            total_loss += loss.item()

            batch_global_metrics, batch_level_metrics, batch_segment_metrics = (
                evaluate_depth(
                    pred,
                    target,
                    segmentation_map,
                    scale_invariant=scale_invariant,
                    nyu_crop=not is_navi,
                    is_navi=is_navi,
                )
            )
            all_segment_metrics.extend(batch_segment_metrics)

            # Aggregating global metrics
            if global_metrics is None:
                global_metrics = {
                    key: [value] for key, value in batch_global_metrics.items()
                }
            else:
                for key, value in batch_global_metrics.items():
                    global_metrics[key].append(value)
                    if wandb_use:
                        wandb.log({f"valid_{key}": batch_global_metrics[key]})

            # Aggregating level metrics
            if level_metrics is None:
                level_metrics = {
                    level: {
                        key: [value]
                        for key, value in batch_level_metrics[level].items()
                    }
                    for level in batch_level_metrics
                }
            else:
                for level in batch_level_metrics:
                    for key, value in batch_level_metrics[level].items():
                        level_metrics[level][key].append(value)

            # Save predicted and target images only for the first batch if render_images is enabled
            if render_images and save_images_once:
                save_images_to_png(
                    pred,
                    target,
                    segmentation_map,
                    batch_idx,
                    task,
                    save_dir,
                    is_navi=is_navi,
                    scale_invariant=scale_invariant,
                    nyu_crop=not is_navi,
                    image_average=False,
                )

                if batch_idx == 5:
                    save_images_once = False

    # Aggregate global metrics
    for key in global_metrics:
        global_metrics[key] = (
            torch.cat(global_metrics[key], dim=0).mean()
            if aggregate
            else global_metrics[key]
        )

    # Aggregate level metrics
    for level in level_metrics:
        for key in level_metrics[level]:
            level_metrics[level][key] = (
                torch.cat(level_metrics[level][key], dim=0).mean()
                if aggregate
                else level_metrics[level][key]
            )

    total_loss = total_loss / len(loader)
    if not is_navi:
        if not scale_invariant:
            plot_segment_area_vs_d1(all_segment_metrics, output_dir=save_dir)

    return total_loss, global_metrics, level_metrics


def visualize_depth(pred, target, colormap="inferno"):
    # Convert tensors to numpy arrays
    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()

    # Apply a color map (e.g., 'viridis') to the predictions and targets
    pred_colored = plt.get_cmap(colormap)(pred / np.max(pred))[
        :, :, :3
    ]  # Only take RGB values
    target_colored = plt.get_cmap(colormap)(target / np.max(target))[:, :, :3]

    return pred_colored, target_colored


def visualize_depth_navi(pred, target, colormap="inferno"):
    # Convert tensors to numpy arrays
    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()

    mask = target != 0.0

    pred_minn = np.min(pred[mask])
    pred_maxx = np.max(pred[mask])

    pred_colored = plt.get_cmap(colormap)((pred - pred_minn) / (pred_maxx - pred_minn))[
        :, :, :3
    ]
    pred_colored[~mask] = 1

    target_minn = np.min(target[mask])
    target_maxx = np.max(target[mask])

    target_colored = plt.get_cmap(colormap)(
        (target - target_minn) / (target_maxx - target_minn)
    )[:, :, :3]
    target_colored[~mask] = 1

    return pred_colored, target_colored


def load_checkpoint(cfg, model, probe):
    print("Evaluating model")
    ckpt_path = cfg.ckpt_path.replace("\$", "$").replace(
        "dense_1-2-3-4", "dense_\[1\,2\,3\,4\]"
    )
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    # model.load_state_dict(remove_module_prefix(checkpoint["model"]))
    probe.load_state_dict(remove_module_prefix(checkpoint["probe"]))
    print("Model and Probe loaded")


def remove_module_prefix(state_dict):
    return {key.replace("module.", ""): val for key, val in state_dict.items()}


def train_model(rank, world_size, cfg):
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    # ===== SETUP WANDB =====
    if rank == 0 and cfg.wandb.use:
        sanitized_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        wandb.init(
            project="ssl-depth-probing-experiments-batchnormed-revision-v1",
            config=sanitized_cfg,
            name=f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            group="seed: " + str(cfg.system.random_seed),
        )

    # ===== GET DATA LOADERS =====
    # validate and test on single gpu
    trainval_loader = build_loader(cfg.dataset, "trainval", cfg.batch_size, world_size)
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1)
    trainval_loader.dataset.__getitem__(0)

    # ===== Get models =====
    print("Building model and probe")
    model = instantiate(cfg.backbone)
    probe = instantiate(
        cfg.probe, feat_dim=model.feat_dim, max_depth=trainval_loader.dataset.max_depth
    )

    # Load checkpoint if in evaluation mode
    if cfg.is_eval and cfg.ckpt_path != "":
        print("Loading checkpoint")
        load_checkpoint(cfg, model, probe)
    # setup experiment name
    # === job info
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")
    train_dset = trainval_loader.dataset.name
    test_dset = test_loader.dataset.name
    model_info = [
        f"{model.checkpoint_name:40s}",
        f"{model.patch_size:2d}",
        f"{str(model.layer):5s}",
        f"{model.output:10s}",
    ]
    probe_info = [f"{probe.name:25s}"]
    batch_size = cfg.batch_size * cfg.system.num_gpus
    train_info = [
        f"{cfg.system.random_seed}",
        f"{cfg.optimizer.n_epochs:3d}",
        f"{cfg.optimizer.warmup_epochs:4.2f}",
        f"{str(cfg.optimizer.probe_lr):>10s}",
        f"{str(cfg.optimizer.model_lr):>10s}",
        f"{batch_size:4d}",
        f"{train_dset:10s}",
        f"{test_dset:10s}",
    ]

    # define exp_name
    exp_name = "_".join([timestamp] + model_info + probe_info + train_info)
    exp_name = f"{exp_name}_{cfg.note}" if cfg.note != "" else exp_name
    exp_name = exp_name.replace(" ", "")  # remove spaces

    # ===== SETUP LOGGING =====
    if rank == 0:
        exp_path = Path(__file__).parent / f"depth_exps/{exp_name}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # move to cuda
    model = model.to(rank)
    probe = probe.to(rank)

    # very hacky ... SAM gets some issues with DDP finetuning
    model_name = model.checkpoint_name
    if "sam" in model_name or "vit-mae" in model_name:
        h, w = trainval_loader.dataset.__getitem__(0)["image"].shape[-2:]
        model.resize_pos_embed(image_size=(h, w))

    # move to DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        probe = DDP(probe, device_ids=[rank])

    if cfg.optimizer.model_lr == 0:
        optimizer = torch.optim.AdamW(
            [{"params": probe.parameters(), "lr": cfg.optimizer.probe_lr}]
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": probe.parameters(), "lr": cfg.optimizer.probe_lr},
                {"params": model.parameters(), "lr": cfg.optimizer.model_lr},
            ]
        )

    lambda_fn = lambda epoch: cosine_decay_linear_warmup(  # noqa: E731
        epoch,
        cfg.optimizer.n_epochs * len(trainval_loader),
        cfg.optimizer.warmup_epochs * len(trainval_loader),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    loss_fn = DepthLoss()
    if not cfg.is_eval:
        train(
            model,
            probe,
            trainval_loader,
            optimizer,
            scheduler,
            cfg.optimizer.n_epochs,
            detach_model=(cfg.optimizer.model_lr == 0),
            loss_fn=loss_fn,
            rank=rank,
            world_size=world_size,
            wandb_use=cfg.wandb.use,
            valid_loader=test_loader,
            is_navi=cfg.dataset.name == "navi_reldepth",
        )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")

        # Scale-Aware validation
        test_sa_loss, test_sa_global_metrics, test_sa_level_metrics = validate(
            model,
            probe,
            test_loader,
            loss_fn,
            wandb_use=cfg.wandb.use,
            render_images=True,
            is_navi=(cfg.dataset.name == "navi_reldepth" or cfg.dataset.name == "navi"),
            output_dir=cfg.output_dir,
        )
        logger.info(f"Scale-Aware Final test loss       | {test_sa_loss:.4f}")
        for metric in test_sa_global_metrics:
            logger.info(
                f"Final test SA {metric:10s} | {test_sa_global_metrics[metric]:.4f}"
            )
        results_sa = [
            f"{test_sa_global_metrics.get(_m, 'N/A')}" for _m in test_sa_global_metrics
        ]

        # Scale-Invariant validation
        test_si_loss, test_si_global_metrics, test_si_level_metrics = validate(
            model,
            probe,
            test_loader,
            loss_fn,
            scale_invariant=True,
            render_images=False,
            is_navi=(cfg.dataset.name == "navi_reldepth" or cfg.dataset.name == "navi"),
        )
        logger.info(f"Scale-Invariant Final test loss   | {test_si_loss:.4f}")
        for metric in test_si_global_metrics:
            logger.info(
                f"Final test SI {metric:10s} | {test_si_global_metrics[metric]:.4f}"
            )
        results_si = [
            f"{test_si_global_metrics.get(_m, 'N/A')}" for _m in test_si_global_metrics
        ]

        # Prepare metrics for logging (global + centroid)
        results_metrics = results_sa + results_si

        # Ensure matching for centroid-level metrics for scale-aware and scale-invariant validation
        for level in test_sa_level_metrics:
            for metric in test_sa_level_metrics[level]:
                results_metrics.append(f"{test_sa_level_metrics[level][metric]:.4f}")

        for level in test_si_level_metrics:
            for metric in test_si_level_metrics[level]:
                results_metrics.append(f"{test_si_level_metrics[level][metric]:.4f}")

        # Conditionally include "stuff and things" metrics for datasets that are NOT `navi_reldepth`
        if cfg.dataset.name != "navi_reldepth":
            stuff_things_columns = [
                "stuff_d1",
                "stuff_d2",
                "stuff_d3",
                "stuff_rmse",
                "stuff_pixels",
                "things_d1",
                "things_d2",
                "things_d3",
                "things_rmse",
                "things_pixels",
            ]

            # Append "stuff and things" metrics for SA
            results_metrics += [
                f"{test_sa_global_metrics.get(metric, 'N/A')}"
                for metric in stuff_things_columns
            ]

            # Append "stuff and things" metrics for SI
            results_metrics += [
                f"{test_si_global_metrics.get(metric, 'N/A')}"
                for metric in stuff_things_columns
            ]

        # Update column titles for centroid metrics
        centroid_columns_sa = [
            f"Level {level} {_m} SA"
            for level in test_sa_level_metrics
            for _m in test_sa_level_metrics[level]
        ]
        centroid_columns_si = [
            f"Level {level} {_m} SI"
            for level in test_si_level_metrics
            for _m in test_si_level_metrics[level]
        ]

        # Base column titles
        column_titles = (
            "Timestamp, Model Checkpoint, Patch Size, Layer, Model Output, "
            "Probe Name, Random Seed, Num Epochs, Warmup Epochs, Probe LR, Model LR, Batch Size, "
            "Train Dataset, Test Dataset, "
            + ", ".join([f"{_m} SA" for _m in test_sa_global_metrics])
            + ", "
            + ", ".join([f"{_m} SI" for _m in test_si_global_metrics])
            + ", "
            + ", ".join(centroid_columns_sa)
            + ", "
            + ", ".join(centroid_columns_si)
        )

        # Add "stuff and things" columns only if dataset is NOT `navi_reldepth`
        if cfg.dataset.name != "navi_reldepth":
            stuff_things_columns_full = [
                "stuff_d1 SA",
                "stuff_d2 SA",
                "stuff_d3 SA",
                "stuff_rmse SA",
                "stuff_pixels SA",
                "things_d1 SA",
                "things_d2 SA",
                "things_d3 SA",
                "things_rmse SA",
                "things_pixels SA",
                "stuff_d1 SI",
                "stuff_d2 SI",
                "stuff_d3 SI",
                "stuff_rmse SI",
                "stuff_pixels SI",
                "things_d1 SI",
                "things_d2 SI",
                "things_d3 SI",
                "things_rmse SI",
                "things_pixels SI",
            ]
            column_titles += ", " + ", ".join(stuff_things_columns_full)

        column_titles = column_titles.split(", ") + ["ckpt_path"]

        # Results summary
        model_info = [info.replace(",", "-") for info in model_info]
        probe_info = [info.replace(",", "-") for info in probe_info]
        train_info = [info.replace(",", "-") for info in train_info]

        exp_info = model_info + probe_info + train_info
        log_row = (
            [timestamp] + exp_info + results_metrics + [str(exp_path / "ckpt.pth")]
        )

        # Define the directory where the log file will be saved
        result_dir = os.path.join(f"{cfg.output_dir}/result", "depth")
        os.makedirs(result_dir, exist_ok=True)  # Ensure the directory exists

        # Define the log file path
        log_file_path = f"depth_results_{test_dset}_final.csv"

        # Adjust file name if batchnorm is used
        if cfg.backbone.add_norm:
            log_file_path = f"depth_results_{test_dset}_final_with_batchnorm.csv"

        log_file_path = os.path.join(result_dir, log_file_path)

        print(f"Saving results to {log_file_path}")

        # Write column titles if the file is empty or new
        if not os.path.exists(log_file_path) or os.stat(log_file_path).st_size == 0:
            with open(log_file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_titles)

        # Append results to the log file
        with open(log_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(log_row)

        # Save final model checkpoint if training
        if not cfg.is_eval:
            model_state_dict = model.state_dict()
            probe_state_dict = probe.state_dict()

            # save final model
            ckpt_path = exp_path / "ckpt.pth"
            checkpoint = {
                "cfg": cfg,
                "model": model_state_dict,
                "probe": probe_state_dict,
            }
            torch.save(checkpoint, ckpt_path)
            logger.info(f"Saved checkpoint at {ckpt_path}")
    if world_size > 1:
        destroy_process_group()


@hydra.main(config_name="depth_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()
