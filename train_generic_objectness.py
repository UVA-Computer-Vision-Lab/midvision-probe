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
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from PIL import Image
import wandb
import numpy as np
import csv
from torch.nn.functional import interpolate
from evals.datasets.voc import VOC
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from evals.utils.optim import cosine_decay_linear_warmup
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

wandb.require("core")


def compute_precision_recall(pred_mask, gt_mask):
    """
    Compute precision and recall based on predicted and ground truth binary masks.

    Args:
        pred_mask (np.ndarray): Binary predicted mask.
        gt_mask (np.ndarray): Binary ground truth mask.

    Returns:
        precision (float): Precision value.
        recall (float): Recall value.
    """
    # True Positive (TP): Predicted 1, Ground truth 1
    TP = np.logical_and(pred_mask == 1, gt_mask == 1).sum()

    # False Positive (FP): Predicted 1, Ground truth 0
    FP = np.logical_and(pred_mask == 1, gt_mask == 0).sum()

    # False Negative (FN): Predicted 0, Ground truth 1
    FN = np.logical_and(pred_mask == 0, gt_mask == 1).sum()

    # Precision = TP / (TP + FP)
    precision = TP / (TP + FP + 1e-6)  # Avoid division by zero
    # Recall = TP / (TP + FN)
    recall = TP / (TP + FN + 1e-6)

    return precision, recall


def compute_f_measure(precision, recall, beta=0.3):
    """
    Compute the F-measure given precision and recall.

    Args:
        precision (float): Precision value.
        recall (float): Recall value.
        beta (float): Weighting factor for precision and recall (default is 0.3).

    Returns:
        f_measure (float): Computed F-measure.
    """
    beta_sq = beta**2
    f_measure = (
        (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-6)
    )  # Avoid division by zero
    return f_measure


def compute_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Compute the Intersection over Union (IoU) score.

    Args:
        pred_mask (np.ndarray): Predicted mask.
        gt_mask (np.ndarray): Ground truth mask.
        threshold (float): Threshold for binarizing masks (default is 0.5).

    Returns:
        iou (float): IoU score.
    """
    # Binarize the predicted mask
    pred_mask_bin = (pred_mask >= threshold).astype(np.uint8)

    # True Positive (TP): Intersection
    intersection = np.logical_and(pred_mask_bin == 1, gt_mask == 1).sum()

    # Union
    union = np.logical_or(pred_mask_bin == 1, gt_mask == 1).sum()

    # IoU = TP / (TP + FP + FN)
    iou = intersection / (union + 1e-6)  # Avoid division by zero
    return iou


def compute_accuracy(pred_mask, gt_mask, threshold=0.5):
    """
    Compute the accuracy score.

    Args:
        pred_mask (np.ndarray): Predicted mask.
        gt_mask (np.ndarray): Ground truth mask.
        threshold (float): Threshold for binarizing masks (default is 0.5).

    Returns:
        accuracy (float): Accuracy score.
    """
    # Binarize the predicted mask
    pred_mask_bin = (pred_mask >= threshold).astype(np.uint8)

    # True Positive + True Negative (correct predictions)
    correct = np.sum(pred_mask_bin == gt_mask)

    # Total number of pixels
    total_pixels = gt_mask.size

    # Accuracy = (TP + TN) / Total pixels
    accuracy = correct / total_pixels
    return accuracy


def compute_corloc(pred_mask, gt_mask, threshold=0.5):
    """
    Compute Correct Localization (CorLoc) score.

    Args:
        pred_mask (np.ndarray): Predicted mask.
        gt_mask (np.ndarray): Ground truth mask.
        threshold (float): Threshold for binarizing masks (default is 0.5).

    Returns:
        corloc (float): CorLoc score.
    """
    # Binarize the predicted mask
    pred_mask_bin = (pred_mask >= threshold).astype(np.uint8)

    # True Positive: Intersection
    intersection = np.logical_and(pred_mask_bin == 1, gt_mask == 1).sum()

    # Union
    union = np.logical_or(pred_mask_bin == 1, gt_mask == 1).sum()

    # Compute IoU
    iou = intersection / (union + 1e-6)  # Avoid division by zero

    # CorLoc is considered correct if IoU >= threshold (commonly 0.5)
    corloc = 1 if iou >= threshold else 0

    return corloc


# def predict(processor, dataset, output_dir, wandb_use=False):
#     """
#     Process the dataset, predict masks, save results, and log results to WandB if enabled.
#     This version logs errors during prediction and includes CorLoc metric.

#     Args:
#         processor: MaskCutProcessor for mask prediction.
#         dataset: Dataset to predict on.
#         wandb_use: Whether to log results to wandb.
#     """
#     # Initialize running averages for F-measure, IoU, Accuracy, and CorLoc
#     avg_metrics = {"F-measure": 0.0, "IoU": 0.0, "Accuracy": 0.0, "CorLoc": 0.0}
#     num_samples = 0
#     errors = []  # To collect any errors that occur during prediction

#     for i, data in enumerate(tqdm(dataset)):
#         try:
#             orig_image_path = data["original_image_path"]
#             gt_binary_mask = data["gt_binary_mask"]
#             num_objects = data["num_objects"]

#             # Process the image to get bounding boxes, individual masks, and combined mask
#             bboxes, masks, combined_mask = processor.process_image(
#                 orig_image_path, num_pseudo_masks=num_objects
#             )

#             orig_image = Image.open(orig_image_path).convert("RGB")

#             # Compute metrics for the current prediction
#             precision, recall = compute_precision_recall(combined_mask, gt_binary_mask)
#             f_measure = compute_f_measure(precision, recall)
#             iou = compute_iou(combined_mask, gt_binary_mask)
#             accuracy = compute_accuracy(combined_mask, gt_binary_mask)
#             corloc = compute_corloc(combined_mask, gt_binary_mask)

#             metrics = {
#                 "F-measure": f_measure,
#                 "IoU": iou,
#                 "Accuracy": accuracy,
#                 "CorLoc": corloc,
#             }

#             # Update running average of the metrics
#             num_samples += 1
#             for key in avg_metrics:
#                 avg_metrics[key] = (
#                     avg_metrics[key] * (num_samples - 1) + metrics[key]
#                 ) / num_samples

#             # Log images and metrics to WandB, if enabled
#             if wandb_use:
#                 wandb.log(
#                     {
#                         "F-measure": metrics["F-measure"],
#                         "IoU": metrics["IoU"],
#                         "Accuracy": metrics["Accuracy"],
#                         "CorLoc": metrics["CorLoc"],
#                         "Avg F-measure": avg_metrics["F-measure"],
#                         "Avg IoU": avg_metrics["IoU"],
#                         "Avg Accuracy": avg_metrics["Accuracy"],
#                         "Avg CorLoc": avg_metrics["CorLoc"],
#                     }
#                 )
#                 log_images(
#                     orig_image,
#                     gt_binary_mask,
#                     combined_mask,
#                     wandb_use=wandb_use,
#                 )

#         except Exception as e:
#             error_message = f"Error processing sample {i}: {str(e)}"
#             errors.append(error_message)
#             print(error_message)

#             # Log the error to WandB if enabled
#             if wandb_use:
#                 wandb.log({"Error": error_message})

#             continue

#     # Log the final averages after processing all samples
#     if wandb_use:
#         wandb.log(
#             {
#                 "Final Avg F-measure": avg_metrics["F-measure"],
#                 "Final Avg IoU": avg_metrics["IoU"],
#                 "Final Avg Accuracy": avg_metrics["Accuracy"],
#                 "Final Avg CorLoc": avg_metrics["CorLoc"],
#             }
#         )

#     # If any errors occurred, log them all together
#     if errors:
#         error_log = "\n".join(errors)
#         if wandb_use:
#             wandb.log({"Error Log": error_log})
#         print("Errors encountered during prediction:\n", error_log)

#     return avg_metrics, errors


def prepare_output_dir(img_path, output_dir):
    """
    Prepare a directory for saving outputs based on the image filename (without the extension),
    inside the given base output directory.

    Args:
        img_path (str): Path to the input image.
        output_dir (str): Base directory for saving outputs.

    Returns:
        str: Path to the output subdirectory.
    """
    # Extract filename without extension
    base_name = os.path.basename(img_path)
    file_name = os.path.splitext(base_name)[0]

    # Create the output subdirectory based on the filename inside the base output_dir
    final_output_dir = os.path.join(output_dir, file_name)

    # Create the directory if it doesn't exist
    os.makedirs(final_output_dir, exist_ok=True)

    return final_output_dir


def log_images(orig_image, gt_binary_mask, combined_mask, wandb_use=False):
    """
    Log the original image, ground truth mask, combined mask, and pseudo-masks to WandB.

    Args:
        orig_image (PIL.Image): The original image.
        gt_binary_mask (np.ndarray): The ground truth binary mask.
        combined_mask (np.ndarray): The combined predicted mask.
        pseudo_masks (list of np.ndarray): List of predicted pseudo-masks.
        wandb_use (bool): Whether to log results to WandB.
    """
    # Convert original image and masks to tensors for logging
    batch_size = orig_image.shape[0]

    for i in range(batch_size):
        orig_image_tensor = torch.tensor(np.array(orig_image[i]))  # [C, H, W]

        # Ensure that masks are cast to the correct data type (uint8 for images)
        gt_binary_mask_tensor = torch.tensor(
            np.array(gt_binary_mask[i]), dtype=torch.uint8
        )
        combined_mask_tensor = torch.tensor(
            np.array(combined_mask[i]), dtype=torch.uint8
        )

        # Log the original image, ground truth mask, and combined mask
        log_data = {
            "Original Image": wandb.Image(orig_image_tensor),
            "Ground Truth Mask": wandb.Image(gt_binary_mask_tensor),
            "Combined Mask": wandb.Image(combined_mask_tensor),
        }

        # Log all data to WandB
        if wandb_use:
            wandb.log(log_data)


def train(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    n_epochs,
    detach_model=None,
    loss_fn=None,  # For binary classification, you could use BCEWithLogitsLoss
    rank=0,
    world_size=1,
    valid_loader=None,
    output_dir="result",
    wandb_use=False,
):
    for ep in range(n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0
        pbar = tqdm(train_loader) if rank == 0 else train_loader

        for i, batch in enumerate(pbar):
            orig_image = batch["original_image"].to(rank)
            gt_binary_mask = batch["gt_binary_mask"].to(rank)

            optimizer.zero_grad()

            # Detach model if specified
            if detach_model:
                with torch.no_grad():
                    feats = model(orig_image)
                    feats = (
                        feats.detach()
                        if not isinstance(feats, (tuple, list))
                        else [_f.detach() for _f in feats]
                    )
            else:
                feats = model(orig_image)

            # Forward pass through probe and resize prediction to match ground truth size
            pred = probe(feats)
            pred = F.interpolate(pred, size=gt_binary_mask.shape[-2:], mode="bilinear")

            # Compute binary classification loss
            loss = loss_fn(pred, gt_binary_mask.float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.item()
            train_loss += loss

            # Update progress bar and log to WandB if needed
            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(
                    f"{ep} | loss: {loss:.4f} ({_loss:.4f}) probe_lr: {pr_lr:.2e}"
                )
                if wandb_use:
                    wandb.log({"train_loss": _loss, "probe_lr": pr_lr, "epoch": ep})

        if valid_loader is not None:
            validation(model, probe, valid_loader, output_dir, wandb_use)


def validation(model, probe, test_dataloader, output_dir="result", wandb_use=False):
    """
    Perform validation on the test dataset using the given model and probe.

    Args:
        model: The model to use for prediction.
        probe: The probe to use for prediction.
        test_dataloader: DataLoader for the test dataset.
        output_dir: Directory for saving output images.
        wandb_use: Whether to log results to WandB.

    Returns:
        dict: Average metrics for the test dataset.
        list: Errors encountered during prediction.
    """
    avg_metrics = {"F-measure": 0, "IoU": 0, "Accuracy": 0, "CorLoc": 0}
    num_samples = 0

    for i, data in enumerate(tqdm(test_dataloader)):
        orig_image = data["original_image"].cuda()
        orig_image_rgb = data["original_image_rgb"]
        gt_binary_mask = data["gt_binary_mask"].cuda()

        with torch.no_grad():
            feats = model(orig_image)
            pred = probe(feats)
            pred = interpolate(pred, size=gt_binary_mask.shape[-2:], mode="bilinear")

        # Apply a 0.5 threshold to get binary predictions
        binary_pred = (pred > 0.5).float()
        binary_pred = binary_pred.cpu().numpy()
        gt_binary_mask = gt_binary_mask.cpu().numpy()
        # Compute metrics
        precision, recall = compute_precision_recall(binary_pred, gt_binary_mask)
        f_measure = compute_f_measure(precision, recall)
        iou = compute_iou(binary_pred, gt_binary_mask)
        accuracy = compute_accuracy(binary_pred, gt_binary_mask)
        corloc = compute_corloc(binary_pred, gt_binary_mask)

        metrics = {
            "F-measure": f_measure,
            "IoU": iou,
            "Accuracy": accuracy,
            "CorLoc": corloc,
        }

        # Update running average of the metrics
        num_samples += 1
        for key in avg_metrics:
            avg_metrics[key] = (
                avg_metrics[key] * (num_samples - 1) + metrics[key]
            ) / num_samples

        # Log images and metrics to WandB, if enabled
        if wandb_use:
            wandb.log(
                {
                    "F-measure": metrics["F-measure"],
                    "IoU": metrics["IoU"],
                    "Accuracy": metrics["Accuracy"],
                    "CorLoc": metrics["CorLoc"],
                    "Avg F-measure": avg_metrics["F-measure"],
                    "Avg IoU": avg_metrics["IoU"],
                    "Avg Accuracy": avg_metrics["Accuracy"],
                    "Avg CorLoc": avg_metrics["CorLoc"],
                }
            )
            gt_binary_mask = data["gt_binary_mask"].cpu().numpy()
            log_images(
                orig_image_rgb,
                gt_binary_mask,
                binary_pred,
                wandb_use=wandb_use,
            )

    return avg_metrics


def train_model(rank, world_size, cfg: DictConfig):
    # Load VOC Dataset
    sanitized_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="ssl-objectness-eval-train-v1",
        config=sanitized_cfg,
        name=f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    if cfg.dataset.name == "voc":
        trainval_dataset = VOC(cfg.dataset, "trainval")
        test_dataset = VOC(cfg.dataset, "test")
    else:
        trainval_dataset = VOC(cfg.dataset, "trainval")
        trainval_size = int(0.8 * len(trainval_dataset))
        test_size = len(trainval_dataset) - trainval_size
        trainval_dataset, test_dataset = random_split(
            trainval_dataset, [trainval_size, test_size]
        )

    print(f"Training/Validation set size: {len(trainval_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    # Wrap datasets with DataLoader
    trainval_loader = DataLoader(
        trainval_dataset,
        batch_size=cfg.batch_size,  # Adjust as per your batch size
        shuffle=True,  # Shuffle for training
        num_workers=cfg.num_workers,  # Adjust based on available CPU cores
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,  # Same batch size or as needed
        shuffle=False,  # Usually no shuffle for testing
        num_workers=cfg.num_workers,
    )
    # Instantiate the backbone and processor
    model = instantiate(cfg.backbone)
    model.eval()
    probe = instantiate(cfg.probe, feat_dim=model.feat_dim)

    # Prediction on training dataset
    if rank == 0:
        exp_path = Path(__file__).parent / f"objectness_exps/{cfg.experiment_name}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # move to cuda
    model = model.to(rank)
    probe = probe.to(rank)

    # very hacky ... SAM gets some issues with DDP finetuning
    model_name = model.checkpoint_name
    if "sam" in model_name or "vit-mae" in model_name:
        h, w = trainval_loader.dataset.__getitem__(0)["original_image"].shape[-2:]
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
    criterion = nn.BCELoss()

    logger.info("Starting prediction on the training dataset...")
    train(
        model,
        probe,
        trainval_loader,
        optimizer,
        scheduler,
        cfg.optimizer.n_epochs,
        detach_model=(cfg.optimizer.model_lr == 0),
        loss_fn=criterion,
        rank=rank,
        world_size=world_size,
        # valid_loader=test_loader,
        valid_loader=None,
        output_dir=cfg.output_dir,
        wandb_use=cfg.wandb.use,
    )

    # Prediction on test dataset
    if rank == 0:
        logger.info("Starting prediction on the test dataset...")
        avg_metrics = validation(
            model,
            probe,
            test_loader,
            output_dir=cfg.output_dir,
            wandb_use=cfg.wandb.use,
        )
        # Get model name
        model_name = cfg.model_name

        # Log train and test avg metrics into a final summary CSV file
        if cfg.dataset.name == "voc":
            filename = "final_results_summary_voc.csv"
        else:
            filename = "final_results_summary_voc12.csv"
        final_log_file = os.path.join(cfg.output_dir, "trained_objectness", filename)
        os.makedirs(os.path.dirname(final_log_file), exist_ok=True)
        # Prepare the column headers if the CSV doesn't exist
        if not os.path.exists(final_log_file):
            with open(final_log_file, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Model Name",
                        "Test Avg F-measure",
                        "Test Avg IoU",
                        "Test Avg Accuracy",
                        "Test Avg CorLoc",
                    ]
                )

        # Append the final averages to the CSV
        with open(final_log_file, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    model_name,
                    avg_metrics["F-measure"],
                    avg_metrics["IoU"],
                    avg_metrics["Accuracy"],
                    avg_metrics["CorLoc"],
                ]
            )


@hydra.main(config_name="objectness_train", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()
