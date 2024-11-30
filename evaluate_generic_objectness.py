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
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from PIL import Image
from evals.datasets.builder import build_loader
from evals.models.maskcut_processor import MaskCutProcessor
import matplotlib.pyplot as plt
import wandb
import numpy as np
import csv
from evals.datasets.voc import VOC

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


def predict(processor, dataset, output_dir, wandb_use=False):
    """
    Process the dataset, predict masks, save results, and log results to WandB if enabled.
    This version logs errors during prediction and includes CorLoc metric.

    Args:
        processor: MaskCutProcessor for mask prediction.
        dataset: Dataset to predict on.
        wandb_use: Whether to log results to wandb.
    """
    # Initialize running averages for F-measure, IoU, Accuracy, and CorLoc
    avg_metrics = {"F-measure": 0.0, "IoU": 0.0, "Accuracy": 0.0, "CorLoc": 0.0}
    num_samples = 0
    errors = []  # To collect any errors that occur during prediction

    for i, data in enumerate(tqdm(dataset)):
        try:
            orig_image_path = data["original_image_path"]
            gt_binary_mask = data["gt_binary_mask"]
            num_objects = data["num_objects"]

            # Process the image to get bounding boxes, individual masks, and combined mask
            bboxes, masks, combined_mask = processor.process_image(
                orig_image_path, num_pseudo_masks=num_objects
            )

            orig_image = Image.open(orig_image_path).convert("RGB")

            # Compute metrics for the current prediction
            precision, recall = compute_precision_recall(combined_mask, gt_binary_mask)
            f_measure = compute_f_measure(precision, recall)
            iou = compute_iou(combined_mask, gt_binary_mask)
            accuracy = compute_accuracy(combined_mask, gt_binary_mask)
            corloc = compute_corloc(combined_mask, gt_binary_mask)

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
                log_images(
                    orig_image,
                    gt_binary_mask,
                    combined_mask,
                    wandb_use=wandb_use,
                )

        except Exception as e:
            error_message = f"Error processing sample {i}: {str(e)}"
            errors.append(error_message)
            print(error_message)

            # Log the error to WandB if enabled
            if wandb_use:
                wandb.log({"Error": error_message})

            continue

    # Log the final averages after processing all samples
    if wandb_use:
        wandb.log(
            {
                "Final Avg F-measure": avg_metrics["F-measure"],
                "Final Avg IoU": avg_metrics["IoU"],
                "Final Avg Accuracy": avg_metrics["Accuracy"],
                "Final Avg CorLoc": avg_metrics["CorLoc"],
            }
        )

    # If any errors occurred, log them all together
    if errors:
        error_log = "\n".join(errors)
        if wandb_use:
            wandb.log({"Error Log": error_log})
        print("Errors encountered during prediction:\n", error_log)

    return avg_metrics, errors


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
    orig_image_tensor = torch.tensor(np.array(orig_image)).permute(2, 0, 1)  # [C, H, W]

    # Ensure that masks are cast to the correct data type (uint8 for images)
    gt_binary_mask_tensor = torch.tensor(np.array(gt_binary_mask), dtype=torch.uint8)
    combined_mask_tensor = torch.tensor(np.array(combined_mask), dtype=torch.uint8)

    # Log the original image, ground truth mask, and combined mask
    log_data = {
        "Original Image": wandb.Image(orig_image_tensor),
        "Ground Truth Mask": wandb.Image(gt_binary_mask_tensor),
        "Combined Mask": wandb.Image(combined_mask_tensor),
    }

    # Log all data to WandB
    if wandb_use:
        wandb.log(log_data)


@hydra.main(config_name="objectness_eval", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    # Load VOC Dataset
    sanitized_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="ssl-objectness-eval-final",
        config=sanitized_cfg,
        name=f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    trainval_dataset = VOC(cfg.dataset, "trainval")
    test_dataset = VOC(cfg.dataset, "test")

    # Instantiate the backbone and processor
    model = instantiate(cfg.backbone)
    model = model.cuda()

    # Set the model to evaluation mode
    model.eval()
    processor = MaskCutProcessor(
        backbone=model, patch_size=16, tau=0.15, fixed_size=480
    )

    # Prediction on training dataset
    logger.info("Starting prediction on the training dataset...")
    train_avg_metrics, train_errors = predict(
        processor,
        trainval_dataset,
        output_dir=cfg.output_dir,
        wandb_use=cfg.wandb.use,
    )
    logger.info(f"Training metrics: {train_avg_metrics}")

    # Prediction on test dataset
    logger.info("Starting prediction on the test dataset...")
    test_avg_metrics, test_errors = predict(
        processor,
        test_dataset,
        output_dir=cfg.output_dir,
        wandb_use=cfg.wandb.use,
    )
    logger.info(f"Test metrics: {test_avg_metrics}")

    # Log the final metrics in the logger for both train and test
    logger.info(f"Final training average metrics: {train_avg_metrics}")
    logger.info(f"Final test average metrics: {test_avg_metrics}")

    # Get model name
    model_name = model.checkpoint_name

    # Log train and test avg metrics into a final summary CSV file
    final_log_file = os.path.join(cfg.output_dir, "final_results_summary.csv")

    # Prepare the column headers if the CSV doesn't exist
    if not os.path.exists(final_log_file):
        with open(final_log_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Model Name",
                    "Train Avg F-measure",
                    "Train Avg IoU",
                    "Train Avg Accuracy",
                    "Train Avg CorLoc",
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
                train_avg_metrics["F-measure"],
                train_avg_metrics["IoU"],
                train_avg_metrics["Accuracy"],
                train_avg_metrics["CorLoc"],
                test_avg_metrics["F-measure"],
                test_avg_metrics["IoU"],
                test_avg_metrics["Accuracy"],
                test_avg_metrics["CorLoc"],
            ]
        )


if __name__ == "__main__":
    main()
