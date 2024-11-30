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

from datetime import datetime

import hydra
import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from evals.datasets.builder import build_loader
from evals.utils.correspondence import (
    compute_binned_performance,
    estimate_correspondence_xyz,
    project_3dto2d,
)
from evals.utils.transformations import so3_rotation_angle, transform_points_Rt
import os
import csv
import matplotlib.pyplot as plt
from matplotlib import patches
import json


def visualize_and_save_correspondences(
    img0, img1, c_uv0, c_uv1, c_err2d, output_dir, threshold=5
):
    """
    Visualizes and saves correspondences between two images.

    Parameters:
    img0 (ndarray): First image (NxMx3).
    img1 (ndarray): Second image (NxMx3).
    c_uv0 (tensor): Correspondence UV coordinates in image 0.
    c_uv1 (tensor): Correspondence UV coordinates in image 1.
    c_err2d (tensor): 2D pixel errors for each correspondence.
    output_dir (str): Directory to save the images.
    threshold (int): Threshold for determining correct correspondences.
    """
    # Save original images without lines, axes, or titles
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    img0_min = img0.min()
    img0_max = img0.max()
    img0_renormalized = (img0 - img0_min) / (img0_max - img0_min)
    axs[0].imshow(img0_renormalized)
    img1_min = img1.min()
    img1_max = img1.max()
    img1_renormalized = (img1 - img1_min) / (img1_max - img1_min)
    axs[1].imshow(img1_renormalized)
    for ax in axs:
        ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    plt.savefig(
        os.path.join(output_dir, "original_views.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    # Save images with correspondences, without axes or titles
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow(img0_renormalized)
    axs[1].imshow(img1_renormalized)
    for i in range(c_uv0.shape[0]):
        color = "green" if c_err2d[i] < threshold else "red"
        axs[0].plot(c_uv0[i, 0], c_uv0[i, 1], "o", color=color, markersize=5)
        axs[1].plot(c_uv1[i, 0], c_uv1[i, 1], "o", color=color, markersize=5)

        # Draw lines between corresponding points
        con = patches.ConnectionPatch(
            xyA=(c_uv1[i, 0], c_uv1[i, 1]),
            xyB=(c_uv0[i, 0], c_uv0[i, 1]),
            coordsA="data",
            coordsB="data",
            axesA=axs[1],
            axesB=axs[0],
            color=color,
            linewidth=1,
        )
        axs[1].add_artist(con)

    for ax in axs:
        ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    plt.savefig(
        os.path.join(output_dir, "correspondences.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


# Main function modification to save errors and viewpoint change in JSON format
def save_results_to_json(c_err2d, c_err3d, rel_ang, output_dir):
    """
    Saves the 2D and 3D errors along with viewpoint change to a JSON file.

    Parameters:
    c_err2d (tensor): 2D error values.
    c_err3d (tensor): 3D error values.
    rel_ang (tensor): Relative viewpoint change angle.
    output_dir (str): Directory to save the JSON file.
    """
    # Count c_err2d values below specific thresholds
    c_err2d_counts = {
        "below_5px": (c_err2d < 5).sum().item(),
        "below_25px": (c_err2d < 25).sum().item(),
        "below_50px": (c_err2d < 50).sum().item(),
    }

    # Count c_err3d values below specific thresholds
    c_err3d_counts = {
        "below_0.01m": (c_err3d < 0.01).sum().item(),
        "below_0.02m": (c_err3d < 0.02).sum().item(),
        "below_0.05m": (c_err3d < 0.05).sum().item(),
    }

    # Prepare results dictionary
    results = {
        "viewpoint_change_deg": rel_ang.item(),
        "2d_error_counts": c_err2d_counts,
        "3d_error_counts": c_err3d_counts,
    }

    with open(os.path.join(output_dir, "correspondence_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)


# Example usage: Visualize one pair and save results
def visualize_and_save_one_pair(
    batch, feats_0, feats_1, xyz_grid_0, xyz_grid_1, Rt_gt, intrinsics, cfg, output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    i = 0  # Choose the first pair in the batch
    c_xyz0, c_xyz1, c_dist, c_uv0, c_uv1 = estimate_correspondence_xyz(
        feats_0[i], feats_1[i], xyz_grid_0[i], xyz_grid_1[i], cfg.num_corr
    )
    c_uv0 = c_uv0 / cfg.scale_factor
    c_uv1 = c_uv1 / cfg.scale_factor

    c_xyz0in1 = transform_points_Rt(c_xyz0, Rt_gt.float())
    c_err3d = (c_xyz0in1[0] - c_xyz1).norm(p=2, dim=1)  # 3D error

    c_xyz1in1_uv = project_3dto2d(c_xyz1, intrinsics[i])
    c_xyz0in1_uv = project_3dto2d(c_xyz0in1[0], intrinsics[i])
    c_err2d = (c_xyz0in1_uv - c_xyz1in1_uv).norm(p=2, dim=1)

    # Get images from the batch for visualization
    img0 = batch["image_0"][i].permute(1, 2, 0).cpu().numpy()
    img1 = batch["image_1"][i].permute(1, 2, 0).cpu().numpy()

    # Visualize and save images
    visualize_and_save_correspondences(
        img0, img1, c_uv0.cpu(), c_uv1.cpu(), c_err2d.cpu(), output_dir, threshold=50
    )

    # Calculate relative angle for viewpoint change and save metrics to JSON
    rel_ang = so3_rotation_angle(Rt_gt[:, :3, :3])[0]
    rel_ang = rel_ang * 180.0 / np.pi  # Convert to degrees
    save_results_to_json(c_err2d, c_err3d, rel_ang, output_dir)


# Modified main function to include time-stamped directory for each run
@hydra.main("./configs", "navi_correspondence", None)
def main(cfg: DictConfig):
    wandb.init(
        project="navi_correspondence_project", config=OmegaConf.to_container(cfg)
    )
    model_name = cfg.model_name
    output_dir = os.path.join(
        cfg.output_dir,
        f"navi_correspondence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model_name,
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load model and data as before
    model = instantiate(
        cfg.backbone, output="dense", return_multilayer=cfg.multilayer
    ).to("cuda")
    loader = build_loader(cfg.dataset, "test", 4, 1, pair_dataset=True)

    feats_0, feats_1, xyz_grid_0, xyz_grid_1, Rt_gt, intrinsics = [], [], [], [], [], []

    for idx, batch in enumerate(tqdm(loader)):
        feat_0 = model(batch["image_0"].cuda())
        feat_1 = model(batch["image_1"].cuda())
        if cfg.multilayer:
            feat_0, feat_1 = torch.cat(feat_0, dim=1), torch.cat(feat_1, dim=1)

        feats_0 = feat_0.detach().cpu()
        feats_1 = feat_1.detach().cpu()
        Rt_gt = batch["Rt_01"]
        intrinsics = batch["intrinsics_1"]

        xyz_grid_0 = nn_F.interpolate(
            batch["xyz_grid_0"], scale_factor=cfg.scale_factor, mode="nearest"
        )
        xyz_grid_1 = nn_F.interpolate(
            batch["xyz_grid_1"], scale_factor=cfg.scale_factor, mode="nearest"
        )

        # Save correspondence visuals and metrics for one example in the batch
        visualize_and_save_one_pair(
            batch,
            feats_0,
            feats_1,
            xyz_grid_0,
            xyz_grid_1,
            Rt_gt,
            intrinsics,
            cfg,
            output_dir + "/" + str(idx),
        )

    wandb.finish()


if __name__ == "__main__":
    main()
