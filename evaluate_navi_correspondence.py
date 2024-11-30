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


def visualize_correspondences(img0, img1, c_uv0, c_uv1, c_err2d, threshold=5):
    """
    Visualizes correspondences between two images with correct ones in green and incorrect ones in red.

    Parameters:
    img0 (ndarray): First image (NxMx3).
    img1 (ndarray): Second image (NxMx3).
    c_uv0 (tensor): Correspondence UV coordinates in image 0.
    c_uv1 (tensor): Correspondence UV coordinates in image 1.
    c_err2d (tensor): 2D pixel errors for each correspondence.
    threshold (int): Threshold for determining correct correspondences.
    """

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    axs[0].imshow(img0)
    axs[1].imshow(img1)

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

    axs[0].set_title("Image 0 (source)")
    axs[1].set_title("Image 1 (target)")

    plt.show()


# Example usage: Visualize one pair
def visualize_one_pair(
    batch, feats_0, feats_1, xyz_grid_0, xyz_grid_1, Rt_gt, intrinsics, cfg
):
    """
    Visualize correspondence for one pair of images from the batch.
    """
    # Estimate correspondences
    i = 0  # Choose the first pair in the batch (for example)
    c_xyz0, c_xyz1, c_dist, c_uv0, c_uv1 = estimate_correspondence_xyz(
        feats_0[0][i], feats_1[0][i], xyz_grid_0[0][i], xyz_grid_1[0][i], cfg.num_corr
    )

    # Project into 2D and calculate errors
    c_uv0 = c_uv0 / cfg.scale_factor
    c_uv1 = c_uv1 / cfg.scale_factor
    c_xyz0in1 = transform_points_Rt(c_xyz0, Rt_gt[i].float())
    c_err2d = (c_uv0 - c_uv1).norm(p=2, dim=1)  # 2D error

    # Get images from the batch for visualization
    img0 = batch["image_0"][i].permute(1, 2, 0).cpu().numpy()  # Convert to NxMx3 format
    img1 = batch["image_1"][i].permute(1, 2, 0).cpu().numpy()

    # Visualize correspondences
    visualize_correspondences(
        img0, img1, c_uv0.cpu(), c_uv1.cpu(), c_err2d.cpu(), threshold=50
    )


@hydra.main("./configs", "navi_correspondence", None)
def main(cfg: DictConfig):
    wandb.init(
        project="navi_correspondence_project", config=OmegaConf.to_container(cfg)
    )

    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")
    loader = build_loader(cfg.dataset, "test", 4, 1, pair_dataset=True)
    _ = loader.dataset.__getitem__(0)

    # extract features
    feats_0 = []
    feats_1 = []
    xyz_grid_0 = []
    xyz_grid_1 = []
    Rt_gt = []
    intrinsics = []

    for batch in tqdm(loader):
        feat_0 = model(batch["image_0"].cuda())
        feat_1 = model(batch["image_1"].cuda())
        if cfg.multilayer:
            feat_0 = torch.cat(feat_0, dim=1)
            feat_1 = torch.cat(feat_1, dim=1)
        feats_0.append(feat_0.detach().cpu())
        feats_1.append(feat_1.detach().cpu())
        Rt_gt.append(batch["Rt_01"])
        intrinsics.append(batch["intrinsics_1"])

        # scale down to avoid a huge matching problem
        xyz_grid_0_i = nn_F.interpolate(
            batch["xyz_grid_0"], scale_factor=cfg.scale_factor, mode="nearest"
        )
        xyz_grid_1_i = nn_F.interpolate(
            batch["xyz_grid_1"], scale_factor=cfg.scale_factor, mode="nearest"
        )
        xyz_grid_0.append(xyz_grid_0_i)
        xyz_grid_1.append(xyz_grid_1_i)

        visualize_one_pair(
            batch, feats_0, feats_1, xyz_grid_0, xyz_grid_1, Rt_gt, intrinsics, cfg
        )

    feats_0 = torch.cat(feats_0, dim=0)
    feats_1 = torch.cat(feats_1, dim=0)
    xyz_grid_0 = torch.cat(xyz_grid_0, dim=0)
    xyz_grid_1 = torch.cat(xyz_grid_1, dim=0)
    Rt_gt = torch.cat(Rt_gt, dim=0).float()[:, :3, :4]
    intrinsics = torch.cat(intrinsics, dim=0).float()

    num_instances = len(loader.dataset)
    err_3d = []
    err_2d = []
    for i in tqdm(range(num_instances)):
        c_xyz0, c_xyz1, c_dist, c_uv0, c_uv1 = estimate_correspondence_xyz(
            feats_0[i], feats_1[i], xyz_grid_0[i], xyz_grid_1[i], cfg.num_corr
        )

        c_uv0 = c_uv0 / cfg.scale_factor
        c_uv1 = c_uv1 / cfg.scale_factor

        c_xyz0in1 = transform_points_Rt(c_xyz0, Rt_gt[i].float())
        c_err3d = (c_xyz0in1 - c_xyz1).norm(p=2, dim=1)

        c_xyz1in1_uv = project_3dto2d(c_xyz1, intrinsics[i])
        c_xyz0in1_uv = project_3dto2d(c_xyz0in1, intrinsics[i])
        c_err2d = (c_xyz0in1_uv - c_xyz1in1_uv).norm(p=2, dim=1)

        err_3d.append(c_err3d.detach().cpu())
        err_2d.append(c_err2d.detach().cpu())

    err_3d = torch.stack(err_3d, dim=0).float()
    err_2d = torch.stack(err_2d, dim=0).float()
    results = []

    metric_thresh = [0.01, 0.02, 0.05]
    for _th in metric_thresh:
        recall_i = 100 * (err_3d < _th).float().mean()
        print(f"Recall at {_th:>.2f} m:  {recall_i:.2f}")
        results.append(f"{recall_i:5.02f}")
        wandb.log({f"3D Recall ({_th:.2f}m)": recall_i})  # Log to WandB

    px_thresh = [5, 25, 50]
    for _th in px_thresh:
        recall_i = 100 * (err_2d < _th).float().mean()
        print(f"Recall at {_th:>3d}px:  {recall_i:.2f}")
        results.append(f"{recall_i:5.02f}")
        wandb.log({f"2D Recall ({_th}px)": recall_i})  # Log to WandB

    # compute rel_ang
    rel_ang = so3_rotation_angle(Rt_gt[:, :3, :3])
    rel_ang = rel_ang * 180.0 / np.pi

    # compute thresholded recall -- 0.2decimeter = 2cm
    rec_2cm = (err_3d < 0.02).float().mean(dim=1)
    bin_rec = compute_binned_performance(rec_2cm, rel_ang, [0, 30, 60, 90, 120])
    for bin_acc in bin_rec:
        results.append(f"{bin_acc * 100:5.02f}")
        wandb.log({f"Bin Rec {i * 30}-{(i + 1) * 30}°": bin_acc * 100})

    # Define the header for the CSV file
    header = [
        "Time",
        "Model Checkpoint",
        "Patch Size",
        "Layer",
        "Output",
        "Num Correspondences",
        "Scale Factor",
        "Dataset",
        "3D Recall (0.01m)",
        "3D Recall (0.02m)",
        "3D Recall (0.05m)",
        "2D Recall (5px)",
        "2D Recall (25px)",
        "2D Recall (50px)",
        "Bin Rec 0-30°",
        "Bin Rec 30-60°",
        "Bin Rec 60-90°",
        "Bin Rec 90-120°",
    ]

    # result summary with actual data
    time = datetime.now().strftime("%d%m%Y-%H%M")
    exp_info = [
        model.checkpoint_name,
        model.patch_size,
        str(model.layer),
        model.output,
        cfg.num_corr,
        cfg.scale_factor,
    ]
    dset = loader.dataset.name
    results = exp_info + [dset] + results

    # CSV file path
    csv_file = f"{cfg.output_dir}/navi_correspondence_final.csv"

    # Check if the file exists
    file_exists = os.path.isfile(csv_file)

    # Open the CSV file in append mode, write header if it's a new file
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header if the file does not exist
        if not file_exists:
            writer.writerow(header)

        # Append the log data to the CSV file
        writer.writerow([time] + results)
    # Finalize WandB run
    wandb.finish()


if __name__ == "__main__":
    main()
