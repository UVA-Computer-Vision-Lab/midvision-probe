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

import os
import csv
import json
from datetime import datetime

import hydra
import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from matplotlib import patches

from evals.datasets.scannet_pairs import ScanNetPairsDataset
from evals.utils.correspondence import (
    compute_binned_performance,
    estimate_correspondence_depth,
    project_3dto2d,
)
from evals.utils.transformations import so3_rotation_angle, transform_points_Rt


def visualize_and_save_correspondences(
    img0, img1, uv0, uv1, err2d, output_dir, threshold=5
):
    os.makedirs(output_dir, exist_ok=True)

    # Save the original images
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow((img0 + 1) / 2)
    axs[1].imshow((img1 + 1) / 2)
    for ax in axs:
        ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    plt.savefig(
        os.path.join(output_dir, "original_views.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    # Save images with correspondences
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow((img0 + 1) / 2)
    axs[1].imshow((img1 + 1) / 2)
    for i in range(uv0.shape[0]):
        color = "green" if err2d[i] < threshold else "red"
        axs[0].plot(uv0[i, 0], uv0[i, 1], "o", color=color, markersize=5)
        axs[1].plot(uv1[i, 0], uv1[i, 1], "o", color=color, markersize=5)
        con = patches.ConnectionPatch(
            xyA=(uv1[i, 0], uv1[i, 1]),
            xyB=(uv0[i, 0], uv0[i, 1]),
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
    # Save images with correspondences
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow((img0 + 1) / 2)
    axs[1].imshow((img1 + 1) / 2)
    for i in range(200):
        color = "green" if err2d[i] < threshold else "red"
        axs[0].plot(uv0[i, 0], uv0[i, 1], "o", color=color, markersize=5)
        axs[1].plot(uv1[i, 0], uv1[i, 1], "o", color=color, markersize=5)
        con = patches.ConnectionPatch(
            xyA=(uv1[i, 0], uv1[i, 1]),
            xyB=(uv0[i, 0], uv0[i, 1]),
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
        os.path.join(output_dir, "correspondences_sparse200.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def save_results_to_json(err2d, c_err3d, rel_ang, output_dir):
    err2d_counts = {
        "below_5px": (err2d < 5).sum().item(),
        "below_10px": (err2d < 10).sum().item(),
        "below_20px": (err2d < 20).sum().item(),
        "below_30px": (err2d < 30).sum().item(),
        "below_40px": (err2d < 40).sum().item(),
        "below_50px": (err2d < 50).sum().item(),
    }

    err3d_counts = {
        "below_1cm": (c_err3d < 0.01).sum().item(),
        "below_2cm": (c_err3d < 0.02).sum().item(),
        "below_5cm": (c_err3d < 0.05).sum().item(),
        "below_15cm": (c_err3d < 0.15).sum().item(),
        "below_25cm": (c_err3d < 0.25).sum().item(),
        "below_35cm": (c_err3d < 0.35).sum().item(),
        "below_50cm": (c_err3d < 0.50).sum().item(),
    }
    results = {
        "rel_ang": rel_ang.item(),
        "2d_error_counts": err2d_counts,
        "3d_error_counts": err3d_counts,
    }

    with open(os.path.join(output_dir, "correspondence_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)


@hydra.main("./configs", "scannet_correspondence", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    wandb.init(
        project="scannet-correspondence",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"eval_{cfg.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # Initialize model and dataset
    model = instantiate(
        cfg.backbone, output="dense", return_multilayer=cfg.multilayer
    ).to("cuda")
    dataset = ScanNetPairsDataset()
    loader = DataLoader(
        dataset, 8, num_workers=4, drop_last=False, pin_memory=True, shuffle=False
    )

    err_2d = []
    err_3d = []
    R_gt = []
    model_name = cfg.model_name
    output_dir = os.path.join(
        cfg.output_dir,
        f"scannet_correspondence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model_name,
    )
    os.makedirs(output_dir, exist_ok=True)

    # Process each pair
    for i in tqdm(range(len(dataset))):

        instance = dataset.__getitem__(i)
        rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"]), dim=0)
        deps = torch.stack((instance["depth_0"], instance["depth_1"]), dim=0)
        K_mat = instance["K"].clone()
        Rt_gt = instance["Rt_1"].float()[:3, :4]
        R_gt.append(Rt_gt[:3, :3])

        feats = model(rgbs.cuda())
        if cfg.multilayer:
            feats = torch.cat(feats, dim=1)

        feats = feats.detach().cpu()
        deps = nn_F.interpolate(deps, scale_factor=cfg.scale_factor)
        K_mat[:2, :] *= cfg.scale_factor

        # Compute correspondences
        corr_xyz0, corr_xyz1, corr_dist = estimate_correspondence_depth(
            feats[0], feats[1], deps[0], deps[1], K_mat.clone(), cfg.num_corr
        )

        # Compute error
        corr_xyz0in1 = transform_points_Rt(corr_xyz0, Rt_gt)
        c_err3d = (corr_xyz0in1 - corr_xyz1).norm(p=2, dim=1)

        uv_0in0 = project_3dto2d(corr_xyz0, K_mat.clone())
        uv_0in1 = project_3dto2d(corr_xyz0in1, K_mat.clone())
        uv_1in1 = project_3dto2d(corr_xyz1, K_mat.clone())
        corr_err2d = (uv_0in1 - uv_1in1).norm(p=2, dim=1)
        err_2d.append(corr_err2d.detach().cpu())
        err_3d.append(c_err3d.detach().cpu())

        # Visualization and saving
        rel_ang = so3_rotation_angle(Rt_gt[:3, :3].unsqueeze(0))
        rel_ang = rel_ang * 180 / np.pi
        img0 = rgbs[0].permute(1, 2, 0).cpu().numpy()
        img1 = rgbs[1].permute(1, 2, 0).cpu().numpy()

        if i % 10 == 0:
            instance_output_dir = os.path.join(output_dir, f"instance_{i}")

            visualize_and_save_correspondences(
                img0,
                img1,
                uv_0in0.cpu() / cfg.scale_factor,
                uv_1in1.cpu() / cfg.scale_factor,
                corr_err2d.cpu(),
                instance_output_dir,
            )
            save_results_to_json(
                corr_err2d.cpu(), c_err3d.cpu(), rel_ang, instance_output_dir
            )

            break

        # Log to W&B
        if cfg.wandb_use:
            wandb.log({"Batch": i, "2D Error": corr_err2d.mean().item()})

    # Save final error stack
    err_2d = torch.stack(err_2d, dim=0).float()
    err_3d = torch.stack(err_3d, dim=0).float()
    R_gt = torch.stack(R_gt, dim=0).float()

    metric_thresh = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for _th in metric_thresh:
        recall_3d_i = 100 * (err_3d < _th).float().mean()
        print(f"Recall at {_th:>.2f} m:  {recall_3d_i:.2f}")
        wandb.log({f"3D Recall ({_th:.2f}m)": recall_3d_i})  # Log to WandB

    # 2D error metrics
    px_thresh = [1, 2, 5, 15, 25, 35, 50]
    for _th in px_thresh:
        recall_i = 100 * (err_2d < _th).float().mean()
        print(f"Recall at {_th:>2d} pixels:  {recall_i:.2f}")
        wandb.log({f"2D Recall @{_th}px": recall_i})
    # breakpoint()
    rel_ang = so3_rotation_angle(R_gt)
    rel_ang = rel_ang * 180 / np.pi

    results = []
    rec_2cm = 100 * (err_3d < 0.02).float().mean(dim=1)
    bin_rec = compute_binned_performance(rec_2cm, rel_ang, [0, 30, 60, 90, 120])
    for bin_acc in enumerate(bin_rec):
        results.append(f"{bin_acc[0] * 100:5.02f}")
        wandb.log({f"Bin Rec {i * 30}-{(i + 1) * 30}°": bin_acc * 100})
    # CSV output
    time = datetime.now().strftime("%d%m%Y-%H%M")
    csv_file = os.path.join(cfg.output_dir, "scannet_correspondence_final.csv")
    header = [
        "Time",
        "Model Checkpoint",
        "Patch Size",
        "Layer",
        "Output",
        "Dataset",
        "Num Correspondences",
        "Scale Factor",
        "2D Recall (1px)",
        "2D Recall (2px)",
        "2D Recall (5px)",
        "2D Recall (15px)",
        "2D Recall (25px)",
        "2D Recall (35px)",
        "2D Recall (50px)",
        "3D Recall (0.01m)",
        "3D Recall (0.02m)",
        "3D Recall (0.05m)",
        "3D Recall (0.1m)",
        "3D Recall (0.2m)",
        "3D Recall (0.3m)",
        "3D Recall (0.4m)",
        "3D Recall (0.5m)",
        "Bin Rec 0-30°",
        "Bin Rec 30-60°",
        "Bin Rec 60-90°",
        "Bin Rec 90-120°",
    ]
    exp_info = [
        model.checkpoint_name,
        model.patch_size,
        str(model.layer),
        model.output,
        loader.dataset.name,
        str(cfg.num_corr),
        str(cfg.scale_factor),
    ]
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not os.path.isfile(csv_file):
            writer.writerow(header)
        writer.writerow(
            [time]
            + exp_info
            + [f"{100 * (err_2d < t).float().mean():5.02f}" for t in px_thresh]
            + [f"{100 * (err_3d < t).float().mean():5.02f}" for t in metric_thresh]
            + [f"{bin_acc}" for bin_acc in bin_rec]
        )

    wandb.finish()


if __name__ == "__main__":
    main()
