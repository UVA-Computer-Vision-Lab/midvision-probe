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
import torch.nn.functional as F
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.nn.functional import interpolate
from evals.datasets.builder import build_loader
from evals.utils.losses import angular_loss
from evals.utils.metrics import evaluate_surface_norm
from evals.utils.optim import cosine_decay_linear_warmup
import json
import wandb

wandb.require("core")
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def ddp_setup(rank: int, world_size: int, port: int = 12355):
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
    rank=0,
    world_size=1,
    valid_loader=None,
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
            mask = batch["depth"].to(rank) > 0
            target = batch["snorm"].to(rank)

            optimizer.zero_grad()
            if detach_model:
                with torch.no_grad():
                    feats = model(images)
                    if type(feats) is tuple or type(feats) is list:
                        feats = [_f.detach() for _f in feats]
                    else:
                        feats = feats.detach()

            else:
                feats = model(images)
            pred = probe(feats)
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic")

            uncertainty = pred.shape[1] > 3
            loss = angular_loss(pred, target, mask, uncertainty_aware=uncertainty)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.item()
            train_loss += loss

            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(f"{ep} | loss: {_loss:.4f} probe_lr: {pr_lr:.2e}")

                if wandb_use:
                    wandb.log({"train_loss": _loss, "lr": pr_lr, "epoch": ep})

        train_loss /= len(train_loader)

        if rank == 0 and wandb_use:
            logger.info(f"Epoch {ep} | train loss: {train_loss:.4f}")
            log_first_batch_images(
                model,
                probe,
                valid_loader,
                rank,
                wandb_use=wandb_use,
                is_navi=is_navi,
            )

            if valid_loader is not None and is_final:
                valid_loss, valid_metrics = validate(model, probe, valid_loader)
                logger.info(f"Final valid loss       | {valid_loss:.4f}")
                for metric in valid_metrics:
                    logger.info(
                        f"Final valid {metric:10s} | {valid_metrics[metric]:.4f}"
                    )
                    if wandb_use:
                        wandb.log(
                            {f"valid_{metric}": valid_metrics[metric], "epoch": ep}
                        )


# Visualization and logging function for first batch
def log_first_batch_images(model, probe, loader, rank, wandb_use=False, is_navi=False):
    model.eval()  # Set model to evaluation mode
    probe.eval()

    pred_images, target_images = [], []
    with torch.inference_mode():
        batch = next(iter(loader))  # Get the first batch of the test set
        images = batch["image"].to(rank)
        target = batch["snorm"].to(rank)
        mask = batch["depth"].to(rank) > 0

        feat = model(images)
        pred = probe(feat).detach()
        pred = interpolate(pred, size=target.shape[-2:], mode="bicubic")

        # Visualize and log first few images
        for i in range(min(8, pred.shape[0])):  # Log up to 8 images
            if is_navi:
                pred_colored, target_colored = visualize_snorm_navi(
                    pred[i], target[i], mask[i]
                )
            else:
                pred_colored, target_colored = visualize_snorm(pred[i], target[i])
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
    mask,
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
    # TODO fix this function
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
        mask_single = mask[i].unsqueeze(0)

        # TODO: Fix this function for evaluate_surface_norm
        global_metrics, level_metrics, _ = evaluate_surface_norm(
            single_pred,
            single_target,
            single_segmentation_map,
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
            pred_colored, target_colored = visualize_snorm_navi(
                pred[i][:3, :, :], target[i][:3, :, :], mask_single
            )
        else:
            pred_colored, target_colored = visualize_snorm(
                pred[i][:3, :, :], target[i][:3, :, :]
            )
        # pred_image = (pred_colored * 255).astype(np.uint8)
        # target_image = (target_colored * 255).astype(np.uint8)
        pred_image = np.squeeze(pred_colored * 255).astype(np.uint8)
        target_image = np.squeeze(target_colored * 255).astype(np.uint8)

        # Check the shape and ensure it's valid for saving
        assert (
            pred_image.shape[-1] == 3 and len(pred_image.shape) == 3
        ), f"Pred image has invalid shape: {pred_image.shape}"
        assert (
            target_image.shape[-1] == 3 and len(target_image.shape) == 3
        ), f"Target image has invalid shape: {target_image.shape}"

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


def tensor_to_numpy(tensor_in):
    """torch tensor to numpy array"""
    if tensor_in is not None:
        if tensor_in.ndim == 3:
            # (C, H, W) -> (H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(1, 2, 0).numpy()
        elif tensor_in.ndim == 4:
            # (B, C, H, W) -> (B, H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(0, 2, 3, 1).numpy()
        else:
            raise Exception("invalid tensor size")
    return tensor_in


def normal_to_rgb(normal, normal_mask=None):
    """surface normal map to RGB
    (used for visualization)

    NOTE: x, y, z are mapped to R, G, B
    NOTE: [-1, 1] are mapped to [0, 255]
    """
    if torch.is_tensor(normal):
        normal = tensor_to_numpy(normal)
        normal_mask = tensor_to_numpy(normal_mask)

    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_norm[normal_norm < 1e-12] = 1e-12
    normal = normal / normal_norm

    normal_rgb = (((normal + 1) * 0.5) * 255).astype(np.uint8)
    if normal_mask is not None:
        normal_rgb = normal_rgb * normal_mask  # (B, H, W, 3)
    return normal_rgb


def visualize_snorm(pred, target):
    """
    Visualizes surface normals by treating the X, Y, Z components of the normal vectors
    as RGB values. The prediction is converted using normal_to_rgb, while the target is
    assumed to be in the correct format for visualization.
    """
    # Convert tensors to numpy arrays if needed
    # pred = pred.squeeze().cpu().numpy()
    # target = target.squeeze().cpu().numpy()

    pred_colored = normal_to_rgb(pred)
    target_colored = normal_to_rgb(target)
    return pred_colored, target_colored


def visualize_snorm_navi(pred, target, mask):
    pred_colored = normal_to_rgb(pred, mask)
    target_colored = normal_to_rgb(target, mask)

    mask = mask.squeeze().cpu().numpy()
    mask = mask[None, :, :]
    pred_colored[~mask] = 1
    target_colored[~mask] = 1
    return pred_colored, target_colored


def validate(
    model,
    probe,
    loader,
    verbose=True,
    aggregate=True,
    wandb_use=False,
    is_navi=False,
    render_images=True,
    output_dir="result",
    backbone_name="backbone",
):
    total_loss = 0.0
    global_metrics = None
    task = "normal-nyu-navi"
    save_images_once = True  # Ensure saving images only for the first batch
    level_metrics = None  # To hold level-wise metrics
    all_segment_metrics = []  # To accumulate segment-wise metrics across batches

    # Create a timestamp-based directory for saving images
    if render_images:
        model_name = backbone_name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(
            f"{output_dir}/{task}/{task}_images", f"{task}_{model_name}_{timestamp}"
        )

    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluation") if verbose else loader
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].cuda()
            mask = batch["depth"].cuda() > 0
            target = batch["snorm"].cuda()
            if not is_navi:
                segmentation_map = batch["segmentation"].cuda()
            else:
                segmentation_map = None
            feats = model(images)
            pred = probe(feats)
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic")

            uncertainty = pred.shape[1] > 3
            loss = angular_loss(pred, target, mask, uncertainty_aware=uncertainty)
            total_loss += loss.item()

            batch_global_metrics, batch_level_metrics, batch_segment_metrics = (
                evaluate_surface_norm(
                    pred.detach(), target, segmentation_map, is_navi=is_navi
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
                    mask,
                    batch_idx,
                    task,
                    save_dir,
                    is_navi=is_navi,
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
        plot_segment_area_vs_d1(all_segment_metrics, output_dir=save_dir)

    return total_loss, global_metrics, level_metrics


def load_checkpoint(cfg, model, probe):
    print("Evaluating model")
    ckpt_path = (
        cfg.ckpt_path.strip()
        .replace("\$", "$")
        .replace("dense_1-2-3-4", "dense_\[1\,2\,3\,4\]")
    )
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(remove_module_prefix(checkpoint["model"]))
    probe.load_state_dict(remove_module_prefix(checkpoint["probe"]))
    print("Model and Probe loaded")


def remove_module_prefix(state_dict):
    return {key.replace("module.", ""): val for key, val in state_dict.items()}


def train_model(rank, world_size, cfg):
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    # ==== SETUP W&B ====
    if rank == 0 and cfg.wandb.use:
        sanitized_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=f"ssl-snorm-probing-experiments-batchnormed-v2-{cfg.dataset.name}",
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
    probe = instantiate(cfg.probe, feat_dim=model.feat_dim)

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
        f"{cfg.optimizer.probe_lr:4.2e}",
        f"{cfg.optimizer.model_lr:4.2e}",
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
        exp_path = Path(__file__).parent / f"snorm_exps/{exp_name}"
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

    lambda_fn = lambda epoch: cosine_decay_linear_warmup(
        epoch,
        cfg.optimizer.n_epochs * len(trainval_loader),
        cfg.optimizer.warmup_epochs * len(trainval_loader),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    if not cfg.is_eval:
        train(
            model,
            probe,
            trainval_loader,
            optimizer,
            scheduler,
            cfg.optimizer.n_epochs,
            detach_model=(cfg.optimizer.model_lr == 0),
            rank=rank,
            world_size=world_size,
            wandb_use=cfg.wandb.use,
            valid_loader=test_loader,
        )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")

        # Validation for surface normals
        test_loss, test_global_metrics, test_level_metrics = validate(
            model,
            probe,
            test_loader,
            wandb_use=cfg.wandb.use,
            render_images=True,
            is_navi=(cfg.dataset.name == "navi_reldepth" or cfg.dataset.name == "navi"),
            output_dir=cfg.output_dir,
            backbone_name=cfg.backbone_name,
        )

        logger.info(f"Final test loss       | {test_loss:.4f}")
        # Log global metrics
        for metric in test_global_metrics:
            logger.info(
                f"Final test {metric:10s} | {test_global_metrics[metric].item():.4f}"
            )

        results_metrics = [
            f"{test_global_metrics[_m].item():.4f}" for _m in test_global_metrics
        ]

        # Log centroid level metrics and gather them for CSV
        level_results = []
        for level in test_level_metrics:
            logger.info(f"Level {level}:")
            for metric, value in test_level_metrics[level].items():
                logger.info(f"{metric}: {value:.4f}")
                level_results.append(f"{value:.4f}")

        # Collect metrics for STUFF and THINGS directly from the test_global_metrics
        if cfg.dataset.name == "navi_reldepth":
            stuff_things_columns = []
            stuff_things_results = []
        else:
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
            stuff_things_results = [
                f"{test_global_metrics.get(_m, 0):.4f}" for _m in stuff_things_columns
            ]

        # Add column titles to the log file, including STUFF and THINGS metrics
        centroid_columns = [
            f"Level {level} {_m}"
            for level in test_level_metrics
            for _m in test_level_metrics[level]
        ]

        column_titles = (
            "Timestamp, Model Checkpoint, Patch Size, Layer, Model Output, "
            "Probe Name, Random Seed, Num Epochs, Warmup Epochs, Probe LR, Model LR, Batch Size, "
            "Train Dataset, Test Dataset, "
            + ", ".join([f"{_m}" for _m in test_global_metrics])
            + ", "
            + ", ".join(centroid_columns)
            + ", "
            + ", ".join(stuff_things_columns)
        )
        column_titles = column_titles.split(", ")

        # Result summary
        model_info = [info.replace(",", "-") for info in model_info]
        probe_info = [info.replace(",", "-") for info in probe_info]
        train_info = [info.replace(",", "-") for info in train_info]

        exp_info = model_info + probe_info + train_info

        # Combine all metrics, including STUFF and THINGS
        log_row = (
            [timestamp]
            + exp_info
            + results_metrics
            + level_results
            + stuff_things_results
        )

        # Define the directory where the log file will be saved
        result_dir = os.path.join(f"{cfg.output_dir}/result", "normal-nyu-navi")
        os.makedirs(result_dir, exist_ok=True)  # Ensure the directory exists

        # Define the log file path
        log_file_path = f"snorm_results_{test_dset}_final.csv"

        # Write column titles if the file is empty or new
        if cfg.backbone.add_norm:
            log_file_path = f"snorm_results_{test_dset}_final_with_batchnorm.csv"

        log_file_path = os.path.join(result_dir, log_file_path)
        if not os.path.exists(log_file_path) or os.stat(log_file_path).st_size == 0:
            with open(log_file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_titles)

        # Append results as a new row in the CSV file
        print("Updating CSV: ", log_file_path)
        with open(log_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(log_row)

        # Save final model checkpoint if training
        if not cfg.is_eval:
            model_state_dict = model.state_dict()
            probe_state_dict = probe.state_dict()

            # Save final model
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


@hydra.main(config_name="snorm_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()
