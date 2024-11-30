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

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb
import csv
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from evals.datasets.builder import build_loader

wandb.require("core")


def cosine_similarity_batch(tensor1, tensor2):
    """Compute the cosine similarity between two batches of tensors."""
    return F.cosine_similarity(tensor1, tensor2, dim=-1)


def compute_metrics(gt_labels, pred_labels):
    """Compute classification metrics based on ground truth and predicted labels."""
    accuracy = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }
    return metrics


def predict_batch(model, dataloader, output_dir, wandb_use=False):
    """
    Process the dataloader in batches, run inference, compute similarity scores,
    and compare with ground truth in a batched manner.

    Args:
        model: The ViT or CNN model to be used for feature extraction.
        dataloader: The DataLoader that provides batches of data.
        output_dir: Directory to save prediction results.
        wandb_use: Whether to log results to wandb.

    Returns:
        The list of dictionaries containing id, ground truth label (p), and predicted label.
        Also returns the computed classification metrics.
    """
    results = []
    errors = []  # To log any errors during prediction
    all_gt_labels = []
    all_pred_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    for batch in tqdm(dataloader):
        try:
            img_ref, img_left, img_right, p, ids = batch

            # Move inputs to GPU if available
            img_ref, img_left, img_right = (
                img_ref.to(device),
                img_left.to(device),
                img_right.to(device),
            )
            p = p.to(device)

            # Run inference with the model
            with torch.no_grad():
                if model.arch == "vit":  # For ViT model, use class token
                    features_ref = model(img_ref)
                    features_left = model(img_left)
                    features_right = model(img_right)
                else:  # For CNN, use global average pooling
                    features_ref = F.adaptive_avg_pool2d(model(img_ref), 1).flatten(1)
                    features_left = F.adaptive_avg_pool2d(model(img_left), 1).flatten(1)
                    features_right = F.adaptive_avg_pool2d(model(img_right), 1).flatten(
                        1
                    )
            # Calculate cosine similarities for the batch
            similarity_left = cosine_similarity_batch(features_ref, features_left)
            similarity_right = cosine_similarity_batch(features_ref, features_right)

            # Determine prediction (0 if left is more similar, else 1)
            predictions = torch.where(similarity_left > similarity_right, 0, 1)

            # Collect results for the batch
            for i in range(len(ids)):
                result = {
                    "id": ids[i].item(),
                    "gt": p[i].item(),
                    "prediction": predictions[i].item(),
                }
                results.append(result)

                # Append to full ground truth and predictions lists for metric calculation
                all_gt_labels.append(p[i].item())
                all_pred_labels.append(predictions[i].item())

                # Log to WandB if enabled
                if wandb_use:
                    wandb.log(
                        {
                            "id": ids[i].item(),
                            "gt": p[i].item(),
                            "prediction": predictions[i].item(),
                        }
                    )

        except Exception as e:
            error_message = f"Error processing batch: {str(e)}"
            errors.append(error_message)
            print(error_message)

            if wandb_use:
                wandb.log({"Error": error_message})

    # Save results to CSV
    # csv_file = os.path.join(output_dir, "predictions.csv")
    # with open(csv_file, mode="w", newline="") as file:
    #     writer = csv.DictWriter(file, fieldnames=["id", "gt", "prediction"])
    #     writer.writeheader()
    #     writer.writerows(results)

    # Compute classification metrics after all batches are processed
    metrics = compute_metrics(all_gt_labels, all_pred_labels)

    # Log final metrics to WandB if enabled
    if wandb_use:
        wandb.log(metrics)

    return results, metrics, errors


@hydra.main(config_name="model_percepture", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    # Convert the config to a dictionary for easier use in WandB
    sanitized_cfg = OmegaConf.to_container(cfg, resolve=True)

    # Initialize WandB
    wandb.init(
        project="ssl-model_percepture",
        config=sanitized_cfg,
        name=f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # Load test DataLoader
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, num_workers=8)

    # Instantiate the backbone model
    model = instantiate(cfg.backbone)
    model = model.cuda()

    # Set model to evaluation mode
    model.eval()

    # Log the start of prediction
    logger.info("Starting prediction on the test dataset...")

    # Predict on the test dataset using batched prediction
    test_results, test_metrics, test_errors = predict_batch(
        model=model,
        dataloader=test_loader,
        output_dir=cfg.output_dir,
        wandb_use=cfg.wandb.use,
    )

    # Log the test metrics
    logger.info(f"Test metrics: {test_metrics}")

    # Get the model's checkpoint name, if available
    model_name = cfg.experiment_model

    # Prepare the final results summary CSV file
    os.makedirs(cfg.output_dir, exist_ok=True)
    final_log_file = os.path.join(cfg.output_dir, "final_results_summary.csv")

    # If CSV doesn't exist, write the headers
    if not os.path.exists(final_log_file):
        with open(final_log_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Model Name",
                    "Test Accuracy",
                    "Test F1-Score",
                    "Test Precision",
                    "Test Recall",
                ]
            )

    # Append the final test average metrics to the CSV
    with open(final_log_file, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                model_name,
                test_metrics["accuracy"],
                test_metrics["f1_score"],
                test_metrics["precision"],
                test_metrics["recall"],
            ]
        )

    logger.info(f"Final results saved to {final_log_file}")


if __name__ == "__main__":
    main()
