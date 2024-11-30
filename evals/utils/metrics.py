"""
MIT License

Copyright (c) 2024 Mohamed El Banani

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

import torch
from loguru import logger
from evals.utils.oneformer_id2label import STUFF, THINGS


def depth_rmse(depth_pr, depth_gt, image_average=False):
    assert depth_pr.shape == depth_gt.shape, f"{depth_pr.shape} != {depth_gt.shape}"

    if len(depth_pr.shape) == 4:
        depth_pr = depth_pr.squeeze(1)
        depth_gt = depth_gt.squeeze(1)

    # compute RMSE for each image and then average
    valid = (depth_gt > 0).detach().float()

    # clamp to 1 for empty depth images
    num_valid = valid.sum(dim=(1, 2))
    if (num_valid == 0).any():
        num_valid = num_valid.clamp(min=1)
        logger.warning("GT depth is empty. Clamping to avoid error.")

    # compute pixelwise squared error
    sq_error = (depth_gt - depth_pr).pow(2)
    sum_masked_sqe = (sq_error * valid).sum(dim=(1, 2))
    rmse_image = (sum_masked_sqe / num_valid).sqrt()

    return rmse_image.mean() if image_average else rmse_image


# def evaluate_depth(
#     depth_pr, depth_gt, image_average=False, scale_invariant=False, nyu_crop=False
# ):
#     # nyu_crop must be set to False
#     # (1). Our predictions are not 480 x 640
#     # (2). We later select only the valid points for metrics computation
#     nyu_crop = False

#     assert depth_pr.shape == depth_gt.shape, f"{depth_pr.shape} != {depth_gt.shape}"

#     if len(depth_pr.shape) == 4:
#         depth_pr = depth_pr.squeeze(1)
#         depth_gt = depth_gt.squeeze(1)

#     if nyu_crop:
#         # apply NYU crop --- commonly used in many repos for some reason
#         assert depth_pr.shape[-2] == 480
#         assert depth_pr.shape[-1] == 640
#         depth_pr = depth_pr[..., 45:471, 41:601]
#         depth_gt = depth_gt[..., 45:471, 41:601]

#     if scale_invariant:
#         depth_pr = match_scale_and_shift(depth_pr, depth_gt)

#     # zero out invalid pixels
#     valid = (depth_gt > 0).detach().float()
#     depth_pr = depth_pr * valid

#     # get num valid
#     num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

#     # get recall @ thresholds
#     thresh = torch.maximum(
#         depth_gt / depth_pr.clamp(min=1e-9), depth_pr / depth_gt.clamp(min=1e-9)
#     )
#     d1 = ((thresh < 1.25**1).float() * valid).sum(dim=(1, 2)) / num_valid
#     d2 = ((thresh < 1.25**2).float() * valid).sum(dim=(1, 2)) / num_valid
#     d3 = ((thresh < 1.25**3).float() * valid).sum(dim=(1, 2)) / num_valid

#     # compute RMSE
#     sse = (depth_gt - depth_pr).pow(2)
#     mse = (sse * valid).sum(dim=(1, 2)) / num_valid
#     rmse = mse.sqrt()
#     metrics = {"d1": d1.cpu(), "d2": d2.cpu(), "d3": d3.cpu(), "rmse": rmse.cpu()}

#     if image_average:
#         for key in metrics:
#             metrics[key] = metrics[key].mean()

#     return metrics


def evaluate_depth(
    depth_pr,
    depth_gt,
    segmentation_map,
    image_average=False,
    scale_invariant=False,
    nyu_crop=False,
    num_levels=5,
    is_navi=False,
):
    # nyu_crop must be set to False (if set to True, modify as needed)
    nyu_crop = False

    # Assert shapes match
    assert depth_pr.shape == depth_gt.shape, f"{depth_pr.shape} != {depth_gt.shape}"

    if len(depth_pr.shape) == 4:
        depth_pr = depth_pr.squeeze(1)
        depth_gt = depth_gt.squeeze(1)

    if scale_invariant:
        depth_pr = match_scale_and_shift(depth_pr, depth_gt)

    # Zero out invalid pixels
    valid = (depth_gt > 0).detach().float()
    depth_pr = depth_pr * valid
    num_valid = valid.sum(dim=(1, 2))

    # Only replace num_valid with 1e-6 if it's zero
    num_valid = torch.where(
        num_valid == 0, torch.tensor(1e-6).to(num_valid.device), num_valid
    )

    # Global metrics (across entire image)
    mean_pred = (depth_pr * valid).sum(dim=(1, 2)) / num_valid

    mean_pred = mean_pred.view(-1, 1, 1)
    variance_pred = (((depth_pr - mean_pred) ** 2) * valid).sum(dim=(1, 2)) / num_valid
    std_pred = variance_pred.sqrt()

    mean_gt = (depth_gt * valid).sum(dim=(1, 2)) / num_valid
    mean_gt = mean_gt.view(-1, 1, 1)
    variance_gt = (((depth_gt - mean_gt) ** 2) * valid).sum(dim=(1, 2)) / num_valid
    std_gt = variance_gt.sqrt()

    variance_ratio = variance_pred / torch.where(
        variance_gt == 0, torch.tensor(1e-6).to(variance_gt.device), variance_gt
    )

    thresh = torch.maximum(
        depth_gt / depth_pr.clamp(min=1e-9), depth_pr / depth_gt.clamp(min=1e-9)
    )
    d1 = ((thresh < 1.25**1).float() * valid).sum(dim=(1, 2)) / num_valid
    d2 = ((thresh < 1.25**2).float() * valid).sum(dim=(1, 2)) / num_valid
    d3 = ((thresh < 1.25**3).float() * valid).sum(dim=(1, 2)) / num_valid

    sse = (depth_gt - depth_pr).pow(2)
    mse = (sse * valid).sum(dim=(1, 2)) / num_valid
    rmse = mse.sqrt()

    global_metrics = {
        "d1": d1.cpu(),
        "d2": d2.cpu(),
        "d3": d3.cpu(),
        "rmse": rmse.cpu(),
        "mean_pred": mean_pred.cpu(),
        "std_pred": std_pred.cpu(),
        "variance_pred": variance_pred.cpu(),
        "mean_gt": mean_gt.cpu(),
        "std_gt": std_gt.cpu(),
        "variance_gt": variance_gt.cpu(),
        "variance_ratio": variance_ratio.cpu(),
    }

    if not is_navi:
        # Compute metrics for stuff and things
        stuff_mask = (
            torch.isin(
                segmentation_map, torch.tensor(STUFF).to(segmentation_map.device)
            ).float()
            * valid
        )
        things_mask = (
            torch.isin(
                segmentation_map, torch.tensor(THINGS).to(segmentation_map.device)
            ).float()
            * valid
        )

        stuff_pixels = stuff_mask.sum(dim=(1, 2))

        stuff_pixels = torch.where(
            stuff_pixels == 0, torch.tensor(1e-6).to(stuff_pixels.device), stuff_pixels
        )
        things_pixels = things_mask.sum(dim=(1, 2))

        things_pixels = torch.where(
            things_pixels == 0,
            torch.tensor(1e-6).to(things_pixels.device),
            things_pixels,
        )

        stuff_d1 = ((thresh < 1.25**1).float() * stuff_mask).sum(
            dim=(1, 2)
        ) / stuff_pixels
        things_d1 = ((thresh < 1.25**1).float() * things_mask).sum(
            dim=(1, 2)
        ) / things_pixels

        stuff_d2 = ((thresh < 1.25**2).float() * stuff_mask).sum(
            dim=(1, 2)
        ) / stuff_pixels
        things_d2 = ((thresh < 1.25**2).float() * things_mask).sum(
            dim=(1, 2)
        ) / things_pixels

        stuff_d3 = ((thresh < 1.25**3).float() * stuff_mask).sum(
            dim=(1, 2)
        ) / stuff_pixels
        things_d3 = ((thresh < 1.25**3).float() * things_mask).sum(
            dim=(1, 2)
        ) / things_pixels

        stuff_rmse = ((sse * stuff_mask).sum(dim=(1, 2)) / stuff_pixels).sqrt()
        things_rmse = ((sse * things_mask).sum(dim=(1, 2)) / things_pixels).sqrt()

        # Add stuff and things metrics to global metrics
        global_metrics.update(
            {
                "stuff_d1": stuff_d1.cpu(),
                "stuff_d2": stuff_d2.cpu(),
                "stuff_d3": stuff_d3.cpu(),
                "stuff_rmse": stuff_rmse.cpu(),
                "stuff_pixels": stuff_pixels.cpu(),
                "things_d1": things_d1.cpu(),
                "things_d2": things_d2.cpu(),
                "things_d3": things_d3.cpu(),
                "things_rmse": things_rmse.cpu(),
                "things_pixels": things_pixels.cpu(),
            }
        )
    # Compute metrics for centroid levels
    height, width = depth_pr.shape[-2], depth_pr.shape[-1]
    level_masks = []
    metrics_by_level = {}

    cumulative_mask = torch.zeros_like(valid)

    for level in range(1, num_levels + 1):
        mask = torch.zeros_like(valid)

        # Calculate the offset and size for this level
        offset = (height // num_levels) * (num_levels - level) // 2
        size = (height // num_levels) * level

        # Define the region for this level
        mask[..., offset : height - offset, offset : width - offset] = 1

        # Subtract the cumulative mask to avoid overlap with previous levels
        mask -= cumulative_mask
        mask = torch.clamp(mask, min=0)  # Ensure valid mask values

        # Only keep valid pixels in this level
        mask *= valid
        level_masks.append(mask)

        # Update the cumulative mask
        cumulative_mask += mask

        # Get valid pixels count for the level
        num_valid_level = mask.sum(dim=(1, 2))

        # Only replace num_valid_level with 1e-6 if it's zero
        num_valid_level = torch.where(
            num_valid_level == 0,
            torch.tensor(1e-6).to(num_valid_level.device),
            num_valid_level,
        )

        # Compute recall @ thresholds for the level
        thresh_level = torch.maximum(
            depth_gt / depth_pr.clamp(min=1e-9), depth_pr / depth_gt.clamp(min=1e-9)
        )
        d1_level = ((thresh_level < 1.25**1).float() * mask).sum(
            dim=(1, 2)
        ) / num_valid_level
        d2_level = ((thresh_level < 1.25**2).float() * mask).sum(
            dim=(1, 2)
        ) / num_valid_level
        d3_level = ((thresh_level < 1.25**3).float() * mask).sum(
            dim=(1, 2)
        ) / num_valid_level

        # Compute RMSE for the level
        sse_level = (depth_gt - depth_pr).pow(2) * mask
        mse_level = sse_level.sum(dim=(1, 2)) / num_valid_level
        rmse_level = mse_level.sqrt()

        # Store metrics for each level
        metrics_by_level[f"level_{level}"] = {
            "d1": d1_level.cpu(),
            "d2": d2_level.cpu(),
            "d3": d3_level.cpu(),
            "rmse": rmse_level.cpu(),
        }

    if image_average:
        # Averaging global metrics
        for key in global_metrics:
            global_metrics[key] = global_metrics[key].mean()

        # Averaging metrics by levels
        for level in range(1, num_levels + 1):
            for key in metrics_by_level[f"level_{level}"]:
                metrics_by_level[f"level_{level}"][key] = metrics_by_level[
                    f"level_{level}"
                ][key].mean()
    segment_metrics = []
    if not is_navi:
        unique_segments = torch.unique(segmentation_map)

        for segment_id in unique_segments:
            # Create a mask for this segment
            segment_mask = (segmentation_map == segment_id).float() * valid

            # Number of valid pixels for the segment
            segment_area = segment_mask.sum(dim=(1, 2))

            # Avoid division by zero
            segment_area = torch.where(
                segment_area == 0,
                torch.tensor(1e-6).to(segment_area.device),
                segment_area,
            )

            # Compute d1 accuracy for the segment
            segment_d1 = ((thresh < 1.25**1).float() * segment_mask).sum(
                dim=(1, 2)
            ) / segment_area

            # Store the segment ID, area, and d1 ratio
            for img_idx in range(depth_pr.size(0)):  # For each image in the batch
                segment_metrics.append(
                    {
                        "segment_id": segment_id.item(),
                        "image_idx": img_idx,
                        "area": segment_area[img_idx].item(),
                        "d1_ratio": segment_d1[img_idx].item(),
                    }
                )

    # Return both global metrics and metrics for each centroid level
    return global_metrics, metrics_by_level, segment_metrics


def evaluate_surface_norm_navi(snorm_pr, snorm_gt, valid, image_average=False):
    """
    Metrics to evaluate surface norm based on iDISC (and probably Fouhey et al. 2016).
    """
    snorm_pr = snorm_pr[:, :3]
    assert snorm_pr.shape == snorm_gt.shape, f"{snorm_pr.shape} != {snorm_gt.shape}"

    # compute angular error
    cos_sim = torch.cosine_similarity(snorm_pr, snorm_gt, dim=1)
    cos_sim = cos_sim.clamp(min=-1, max=1.0)
    err_deg = torch.acos(cos_sim) * 180.0 / torch.pi

    # zero out invalid errors
    assert len(valid.shape) == 4
    valid = valid.squeeze(1).float()
    err_deg = err_deg * valid
    num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

    # compute rmse
    rmse = (err_deg.pow(2).sum(dim=(1, 2)) / num_valid).sqrt()

    # compute recall at thresholds
    thresh = [11.25, 22.5, 30]
    d1 = ((err_deg < thresh[0]).float() * valid).sum(dim=(1, 2)) / num_valid
    d2 = ((err_deg < thresh[1]).float() * valid).sum(dim=(1, 2)) / num_valid
    d3 = ((err_deg < thresh[2]).float() * valid).sum(dim=(1, 2)) / num_valid

    metrics = {"d1": d1.cpu(), "d2": d2.cpu(), "d3": d3.cpu(), "rmse": rmse.cpu()}

    if image_average:
        for key in metrics:
            metrics[key] = metrics[key].mean()

    return metrics


def evaluate_surface_norm(
    snorm_pr,
    snorm_gt,
    segmentation_map,
    image_average=False,
    num_levels=5,
    thresh=[11.25, 22.5, 30.0],
    is_navi=False,
):
    """
    Metrics to evaluate surface normals based on angular error with support for different levels and segmentations.
    snorm_pr: predicted surface normals (B, 3, H, W)
    snorm_gt: ground truth surface normals (B, 3, H, W)
    valid: valid pixel mask (B, 1, H, W) or (B, H, W)
    segmentation_map: segmentation map of the scene
    num_levels: number of levels for centroid-level analysis
    image_average: whether to average the metrics across the batch
    thresh: list of thresholds in degrees for recall (default: [11.25, 22.5, 30.0])
    """
    segment_metrics = []
    snorm_pr = snorm_pr[:, :3]
    assert snorm_pr.shape == snorm_gt.shape, f"{snorm_pr.shape} != {snorm_gt.shape}"

    # Compute angular error between predicted and ground truth surface normals
    cos_sim = torch.cosine_similarity(snorm_pr, snorm_gt, dim=1).clamp(min=-1, max=1)
    err_deg = torch.acos(cos_sim) * 180.0 / torch.pi  # Convert to degrees

    valid = (
        snorm_gt.abs().sum(dim=1) > 0
    ).float()  # Valid pixel mask (non-zero ground truth normals)
    err_deg = err_deg * valid
    num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

    # Compute recall at thresholds and RMSE for the global image
    d1 = ((err_deg < thresh[0]).float() * valid).sum(dim=(1, 2)) / num_valid
    d2 = ((err_deg < thresh[1]).float() * valid).sum(dim=(1, 2)) / num_valid
    d3 = ((err_deg < thresh[2]).float() * valid).sum(dim=(1, 2)) / num_valid

    rmse = (err_deg.pow(2).sum(dim=(1, 2)) / num_valid).sqrt()

    global_metrics = {
        "d1": d1.cpu(),
        "d2": d2.cpu(),
        "d3": d3.cpu(),
        "rmse": rmse.cpu(),
    }

    # Compute metrics for centroid levels
    height, width = snorm_pr.shape[-2], snorm_pr.shape[-1]
    metrics_by_level = {}
    cumulative_mask = torch.zeros_like(valid)

    for level in range(1, num_levels + 1):
        mask = torch.zeros_like(valid)
        offset = (height // num_levels) * (num_levels - level) // 2
        size = (height // num_levels) * level

        # Define the region for this level
        mask[..., offset : height - offset, offset : width - offset] = 1
        mask -= cumulative_mask
        mask = torch.clamp(mask, min=0) * valid
        cumulative_mask += mask

        num_valid_level = mask.sum(dim=(1, 2)).clamp(min=1)
        err_deg_level = err_deg * mask

        d1_level = ((err_deg_level < thresh[0]).float() * mask).sum(
            dim=(1, 2)
        ) / num_valid_level
        d2_level = ((err_deg_level < thresh[1]).float() * mask).sum(
            dim=(1, 2)
        ) / num_valid_level
        d3_level = ((err_deg_level < thresh[2]).float() * mask).sum(
            dim=(1, 2)
        ) / num_valid_level

        rmse_level = (err_deg_level.pow(2).sum(dim=(1, 2)) / num_valid_level).sqrt()

        metrics_by_level[f"level_{level}"] = {
            "d1": d1_level.cpu(),
            "d2": d2_level.cpu(),
            "d3": d3_level.cpu(),
            "rmse": rmse_level.cpu(),
        }
    # Compute metrics for stuff and things
    if not is_navi:
        stuff_mask = (
            torch.isin(
                segmentation_map, torch.tensor(STUFF).to(segmentation_map.device)
            ).float()
            * valid
        )
        things_mask = (
            torch.isin(
                segmentation_map, torch.tensor(THINGS).to(segmentation_map.device)
            ).float()
            * valid
        )
        stuff_pixels = stuff_mask.sum(dim=(1, 2)).clamp(min=1)
        things_pixels = things_mask.sum(dim=(1, 2)).clamp(min=1)

        # Stuff metrics
        stuff_d1 = ((err_deg < thresh[0]).float() * stuff_mask).sum(
            dim=(1, 2)
        ) / stuff_pixels
        stuff_d2 = ((err_deg < thresh[1]).float() * stuff_mask).sum(
            dim=(1, 2)
        ) / stuff_pixels
        stuff_d3 = ((err_deg < thresh[2]).float() * stuff_mask).sum(
            dim=(1, 2)
        ) / stuff_pixels
        stuff_rmse = (err_deg.pow(2) * stuff_mask).sum(dim=(1, 2)).sqrt() / stuff_pixels

        # Things metrics
        things_d1 = ((err_deg < thresh[0]).float() * things_mask).sum(
            dim=(1, 2)
        ) / things_pixels
        things_d2 = ((err_deg < thresh[1]).float() * things_mask).sum(
            dim=(1, 2)
        ) / things_pixels
        things_d3 = ((err_deg < thresh[2]).float() * things_mask).sum(
            dim=(1, 2)
        ) / things_pixels
        things_rmse = (err_deg.pow(2) * things_mask).sum(
            dim=(1, 2)
        ).sqrt() / things_pixels

        global_metrics.update(
            {
                "stuff_d1": stuff_d1.cpu(),
                "stuff_d2": stuff_d2.cpu(),
                "stuff_d3": stuff_d3.cpu(),
                "stuff_rmse": stuff_rmse.cpu(),
                "stuff_pixels": stuff_pixels.cpu(),
                "things_d1": things_d1.cpu(),
                "things_d2": things_d2.cpu(),
                "things_d3": things_d3.cpu(),
                "things_rmse": things_rmse.cpu(),
                "things_pixels": things_pixels.cpu(),
            }
        )

        # Segment-level metrics
        unique_segments = torch.unique(segmentation_map)

        for segment_id in unique_segments:
            # Create a mask for the segment
            segment_mask = (segmentation_map == segment_id).float() * valid

            segment_area = segment_mask.sum(dim=(1, 2)).clamp(min=1)

            # Compute segment-specific d1 accuracy
            segment_d1 = ((err_deg < thresh[0]).float() * segment_mask).sum(
                dim=(1, 2)
            ) / segment_area

            # Store the segment ID, area, and d1 ratio
            for img_idx in range(snorm_pr.size(0)):  # For each image in the batch
                segment_metrics.append(
                    {
                        "segment_id": segment_id.item(),
                        "image_idx": img_idx,
                        "area": segment_area[img_idx].item(),
                        "d1_ratio": segment_d1[img_idx].item(),
                    }
                )

    if image_average:
        # Averaging global metrics
        for key in global_metrics:
            global_metrics[key] = global_metrics[key].mean()

        # Averaging metrics by levels
        for level in range(1, num_levels + 1):
            for key in metrics_by_level[f"level_{level}"]:
                metrics_by_level[f"level_{level}"][key] = metrics_by_level[
                    f"level_{level}"
                ][key].mean()

    # Return both global metrics and metrics for each centroid level and segment
    return global_metrics, metrics_by_level, segment_metrics


def evaluate_curvature_absrel(
    norm_curvature, norm_gt_curvature, valid, image_average=False
):
    """
    Metrics to evaluate surface curvature based on normalized curvature (two values per pixel).

    Args:
    - norm_curvature (torch.Tensor): Predicted normalized curvature with shape [B, 2, 512, 512].
    - norm_gt_curvature (torch.Tensor): Ground-truth normalized curvature with shape [B, 2, 512, 512].
    - valid (torch.Tensor): Valid mask indicating the valid regions of the curvature map [B, 1, 512, 512] or [B, 2, 512, 512].
    - image_average (bool): Whether to return metrics averaged over all images.

    Returns:
    - metrics (dict): Dictionary containing AbsRel, individual threshold accuracies (for both k1 and k2),
                      and combined threshold accuracies (δ1, δ2, δ3).
    """
    # Ensure valid has 2 channels to match the curvature tensors
    if valid.shape[1] == 1:
        valid = valid.expand(-1, 2, -1, -1)

    # Ensure curvatures have 2 channels (k1 and k2)
    norm_curvature = norm_curvature[:, :2]
    norm_curvature = torch.clamp(norm_curvature, min=-1, max=1)
    assert (
        norm_curvature.shape == norm_gt_curvature.shape
    ), f"{norm_curvature.shape} != {norm_gt_curvature.shape}"

    # Compute Absolute Relative Error (AbsRel) for both k1 and k2, applying the valid mask individually
    abs_rel_k1 = torch.abs(norm_curvature[:, 0] - norm_gt_curvature[:, 0]) / torch.abs(
        norm_gt_curvature[:, 0] + 1e-6
    )
    abs_rel_k2 = torch.abs(norm_curvature[:, 1] - norm_gt_curvature[:, 1]) / torch.abs(
        norm_gt_curvature[:, 1] + 1e-6
    )

    # Apply valid mask for each channel (k1 and k2)
    abs_rel_k1 = abs_rel_k1 * valid[:, 0]
    abs_rel_k2 = abs_rel_k2 * valid[:, 1]

    # Combine k1 and k2 absolute relative errors
    abs_rel = (abs_rel_k1 + abs_rel_k2) / 2

    # Calculate number of valid pixels per channel
    num_valid_k1 = valid[:, 0].sum(dim=(1, 2)).clamp(min=1)  # Avoid division by zero
    num_valid_k2 = valid[:, 1].sum(dim=(1, 2)).clamp(min=1)

    # Compute the final AbsRel by averaging over valid pixels
    abs_rel_k1 = abs_rel_k1.sum(dim=(1, 2)) / num_valid_k1
    abs_rel_k2 = abs_rel_k2.sum(dim=(1, 2)) / num_valid_k2
    abs_rel = (abs_rel_k1 + abs_rel_k2) / 2  # Average across k1 and k2

    # Compute threshold accuracy δ at 1.25, 1.25 * 2, and 1.25 * 3 for both k1 and k2
    ratio_k1 = torch.max(
        norm_curvature[:, 0] / norm_gt_curvature[:, 0],
        norm_gt_curvature[:, 0] / norm_curvature[:, 0],
    )
    ratio_k2 = torch.max(
        norm_curvature[:, 1] / norm_gt_curvature[:, 1],
        norm_gt_curvature[:, 1] / norm_curvature[:, 1],
    )

    # Apply valid mask to ratios
    ratio_k1 = ratio_k1 * valid[:, 0]
    ratio_k2 = ratio_k2 * valid[:, 1]

    # Calculate threshold accuracies for k1 and k2
    d1_k1 = ((ratio_k1 < 1.25).float() * valid[:, 0]).sum(dim=(1, 2)) / num_valid_k1
    d2_k1 = ((ratio_k1 < 1.25 * 2).float() * valid[:, 0]).sum(dim=(1, 2)) / num_valid_k1
    d3_k1 = ((ratio_k1 < 1.25 * 3).float() * valid[:, 0]).sum(dim=(1, 2)) / num_valid_k1

    d1_k2 = ((ratio_k2 < 1.25).float() * valid[:, 1]).sum(dim=(1, 2)) / num_valid_k2
    d2_k2 = ((ratio_k2 < 1.25 * 2).float() * valid[:, 1]).sum(dim=(1, 2)) / num_valid_k2
    d3_k2 = ((ratio_k2 < 1.25 * 3).float() * valid[:, 1]).sum(dim=(1, 2)) / num_valid_k2

    # Compute combined threshold accuracies (average of k1 and k2)
    d1 = (d1_k1 + d1_k2) / 2
    d2 = (d2_k1 + d2_k2) / 2
    d3 = (d3_k1 + d3_k2) / 2

    # Collect metrics
    metrics = {
        "AbsRel": abs_rel.cpu(),
        "δ1.25_k1": d1_k1.cpu(),
        "δ2.5_k1": d2_k1.cpu(),
        "δ3.75_k1": d3_k1.cpu(),
        "δ1.25_k2": d1_k2.cpu(),
        "δ2.5_k2": d2_k2.cpu(),
        "δ3.75_k2": d3_k2.cpu(),
        "δ1.25_avg": d1.cpu(),
        "δ2.5_avg": d2.cpu(),
        "δ3.75_avg": d3.cpu(),
    }

    # If image_average is set, compute the mean across all images
    if image_average:
        for key in metrics:
            metrics[key] = metrics[key].mean()

    return metrics


def evaluate_reshading_absrel_and_delta(
    pred, target, mask, thresholds=[1.1, 1.1**2, 1.1**3], image_average=False
):
    """
    Evaluates reshading using both Absolute Relative Difference (AbsRel) and δ (threshold-based accuracy).

    Args:
    - pred (torch.Tensor): Predicted reshading values with shape [B, 1, H, W].
    - target (torch.Tensor): Ground truth reshading values with shape [B, 1, H, W].
    - mask (torch.Tensor): Valid mask indicating where to compute the metrics with shape [B, 1, H, W].
    - thresholds (list): List of thresholds to evaluate for δ, such as [1.1, 1.1^2, 1.1^3].
    - image_average (bool): Whether to return metrics averaged over all images.

    Returns:
    - dict: A dictionary containing calculated metrics such as AbsRel and δ for various thresholds.
    """
    # Ensure pred and target are properly masked and squeezed
    pred = pred.squeeze(1)  # Remove channel dimension
    target = target.squeeze(1)  # Remove channel dimension
    mask = mask.squeeze(1)  # Remove channel dimension

    # Apply valid mask
    pred = pred * mask
    target = target * mask

    # 1. Calculate AbsRel (Absolute Relative Difference)
    absrel = torch.abs(pred - target) / (
        target + 1e-6
    )  # Add epsilon to avoid division by zero
    absrel = (absrel * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(
        min=1
    )  # Avoid zero-division

    # 2. Calculate δ (Threshold-based Accuracy) for each threshold
    delta_metrics = {}
    for threshold in thresholds:
        # Calculate the ratio
        ratio = torch.max(
            pred / (target + 1e-6), target / (pred + 1e-6)
        )  # Add epsilon to avoid division by zero

        # Calculate the percentage of pixels that meet the threshold criteria
        valid_pixels = ((ratio < threshold).float() * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1)
        delta_metrics[f"δ_{threshold}"] = valid_pixels

    # Combine both AbsRel and δ metrics
    metrics = {
        "AbsRel": absrel,
    }
    metrics.update(delta_metrics)

    # If image_average is set, compute the mean across all images
    if image_average:
        for key in metrics:
            metrics[key] = metrics[key].mean()

    return metrics


def match_scale_and_shift(prediction, target):
    # based on implementation from
    # https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0

    assert len(target.shape) == len(prediction.shape)
    if len(target.shape) == 4:
        four_chan = True
        target = target.squeeze(dim=1)
        prediction = prediction.squeeze(dim=1)
    else:
        four_chan = False

    mask = (target > 0).float()

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 *
    # a_10) . b
    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    # compute scale and shift
    scale = torch.ones_like(b_0)
    shift = torch.zeros_like(b_1)
    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    scale = scale.view(-1, 1, 1).detach()
    shift = shift.view(-1, 1, 1).detach()
    prediction = prediction * scale + shift

    return prediction[:, None, :, :] if four_chan else prediction
