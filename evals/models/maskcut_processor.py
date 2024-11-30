import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from scipy.linalg import eigh
from evals.utils import metric
from evals.models.crf import densecrf
from pycocotools import mask
import pycocotools.mask as mask_util
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.cluster import KMeans


class MaskCutProcessor:
    def __init__(
        self,
        backbone,
        feature_extractor_fn=None,
        patch_size=8,
        tau=0.2,
        fixed_size=480,
    ):
        """
        Initialize MaskCutProcessor with a backbone model and parameters.

        Args:
            backbone (nn.Module): Backbone model for feature extraction (CNN, Transformer, etc.).
            feature_extractor_fn (callable, optional): Custom feature extractor function. If None, use the default.
            patch_size (int): Patch size to use for MaskCut.
            tau (float): Threshold for the affinity matrix.
            fixed_size (int): Fixed image size for resizing.
        """
        self.backbone = backbone
        self.feature_extractor_fn = (
            feature_extractor_fn
            if feature_extractor_fn is not None
            else self.default_feature_extractor
        )
        self.patch_size = patch_size
        self.tau = tau
        self.fixed_size = fixed_size
        self.ToTensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def default_feature_extractor(self, image_tensor):
        """
        Default feature extraction method using the backbone. Can be overridden.
        Args:
            image_tensor (torch.Tensor): Preprocessed input image tensor.

        Returns:
            torch.Tensor: Feature maps extracted by the backbone.
        """
        return self.backbone(image_tensor)[0]  # Adjust based on backbone output

    def get_affinity_matrix(
        self, feats, tau, eps=1e-5, is_wandb=True, index=0, distance_threshold=0.3
    ):
        """
        Computes the affinity matrix from the feature maps.

        Args:
            feats (torch.Tensor): Feature maps.
            tau (float): Threshold for affinity matrix.
            eps (float): A small value to avoid division by zero.

        Returns:
            tuple: Affinity matrix A and diagonal matrix D.
        """
        feats = F.normalize(feats, p=2, dim=0)
        A = (feats.transpose(0, 1) @ feats).cpu().numpy()
        A_flat = A.flatten().reshape(-1, 1)
        kmeans = KMeans(n_clusters=2).fit(A_flat)
        cluster_centers = kmeans.cluster_centers_.flatten()
        distance_between_centers = abs(cluster_centers[0] - cluster_centers[1])
        # Check if the distance is too small
        if distance_between_centers < distance_threshold:
            print(
                f"Cluster centers too close: {distance_between_centers}. Using 90th percentile."
            )
            # Fall back to the 90th percentile if clusters are too close
            tau = np.percentile(A, 90)
        else:
            # If clusters are valid, use the mean of the cluster centers
            tau = np.mean(cluster_centers)
        tau = np.mean(kmeans.cluster_centers_)
        if is_wandb and index == 0:
            plt.figure(figsize=(8, 6))
            sns.histplot(A.flatten(), bins=50, kde=True)
            plt.title("Affinity Matrix Value Distribution")
            plt.xlabel("Affinity Matrix Values")
            plt.ylabel("Frequency")
            # Log the plot to W&B without saving
            wandb.log({"Affinity Matrix Distribution": wandb.Image(plt)})
            wandb.log({"Affinity Matrix Threshold": distance_between_centers})
            plt.close()

        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
        d_i = np.sum(A, axis=1)
        D = np.diag(d_i)

        return A, D

    def second_smallest_eigenvector(self, A, D):
        """
        Compute the second smallest eigenvector of the generalized eigenvalue problem.

        Args:
            A (np.ndarray): Affinity matrix.
            D (np.ndarray): Diagonal matrix.

        Returns:
            np.ndarray: The second smallest eigenvector.
        """
        _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])
        eigenvec = np.copy(eigenvectors[:, 0])
        second_smallest_vec = eigenvectors[:, 0]
        return eigenvec, second_smallest_vec

    def get_salient_areas(self, second_smallest_vec):
        """
        Extract the salient areas from the second smallest eigenvector.

        Args:
            second_smallest_vec (np.ndarray): The second smallest eigenvector.

        Returns:
            np.ndarray: Binary mask indicating salient areas.
        """
        avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
        bipartition = second_smallest_vec > avg
        return bipartition

    def resize_pil(self, I, patch_size=16):
        w, h = I.size

        new_w, new_h = (
            int(round(w / patch_size)) * patch_size,
            int(round(h / patch_size)) * patch_size,
        )
        feat_w, feat_h = new_w // patch_size, new_h // patch_size

        return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h

    def check_num_fg_corners(self, bipartition, dims):
        # check number of corners belonging to the foreground
        bipartition_ = bipartition.reshape(dims)
        top_l, top_r, bottom_l, bottom_r = (
            bipartition_[0][0],
            bipartition_[0][-1],
            bipartition_[-1][0],
            bipartition_[-1][-1],
        )
        nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
        return nc

    def detect_box(
        self,
        bipartition,
        seed,
        dims,
        initial_im_size=None,
        scales=None,
        principle_object=True,
    ):
        """
        Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
        """
        w_featmap, h_featmap = dims
        objects, num_objects = ndimage.label(bipartition)
        cc = objects[np.unravel_index(seed, dims)]

        if principle_object:
            mask = np.where(objects == cc)
            # Add +1 because excluded max
            ymin, ymax = min(mask[0]), max(mask[0]) + 1
            xmin, xmax = min(mask[1]), max(mask[1]) + 1
            # Rescale to image size
            r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
            r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
            pred = [r_xmin, r_ymin, r_xmax, r_ymax]

            # Check not out of image size (used when padding)
            if initial_im_size:
                pred[2] = min(pred[2], initial_im_size[1])
                pred[3] = min(pred[3], initial_im_size[0])

            # Coordinate predictions for the feature space
            # Axis different then in image space
            pred_feats = [ymin, xmin, ymax, xmax]

            return pred, pred_feats, objects, mask
        else:
            raise NotImplementedError

    def get_masked_affinity_matrix(self, painting, feats, mask, ps):
        # mask out affinity matrix based on the painting matrix
        dim, num_patch = feats.size()[0], feats.size()[1]
        painting = painting + mask.unsqueeze(0)
        painting[painting > 0] = 1
        painting[painting <= 0] = 0
        feats = feats.clone().view(dim, ps, ps)
        feats = ((1 - painting) * feats).view(dim, num_patch)
        return feats, painting

    def maskcut_forward(self, feats, dims, scales, init_image_size, num_pseudo_masks):
        """
        Perform the MaskCut algorithm to generate pseudo-masks.

        Args:
            feats (torch.Tensor): Feature maps.
            dims (list): Dimensions of the feature map (height, width).
            init_image_size (tuple): The original image size (height, width).
            num_pseudo_masks (int): Number of pseudo-masks to generate.

        Returns:
            list: List of binary masks.
        """
        bipartitions = []
        eigvecs = []
        for i in range(num_pseudo_masks):
            if i == 0:
                painting = torch.zeros(dims).cuda()
            else:
                feats, painting = self.get_masked_affinity_matrix(
                    painting, feats, current_mask, ps
                )
            A, D = self.get_affinity_matrix(feats, self.tau, is_wandb=True, index=i)

            eigenvec, second_smallest_vec = self.second_smallest_eigenvector(A, D)

            bipartition = self.get_salient_areas(second_smallest_vec)

            seed = np.argmax(np.abs(second_smallest_vec))
            nc = self.check_num_fg_corners(bipartition, dims)
            if nc >= 3:
                reverse = True
            else:
                reverse = bipartition[seed] != 1
            if reverse:
                # reverse bipartition, eigenvector and get new seed
                eigenvec = eigenvec * -1
                bipartition = np.logical_not(bipartition)
                seed = np.argmax(eigenvec)
            else:
                seed = np.argmax(second_smallest_vec)

            bipartition = bipartition.reshape(dims).astype(float)
            _, _, _, cc = self.detect_box(
                bipartition, seed, dims, scales=scales, initial_im_size=init_image_size
            )
            pseudo_mask = np.zeros(dims)
            pseudo_mask[cc[0], cc[1]] = 1
            pseudo_mask = torch.from_numpy(pseudo_mask).cuda()
            ps = pseudo_mask.shape[0]

            if i >= 1:
                ratio = (
                    torch.sum(pseudo_mask)
                    / pseudo_mask.size()[0]
                    / pseudo_mask.size()[1]
                )
                if metric.IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                    pseudo_mask = np.zeros(dims)
                    pseudo_mask = torch.from_numpy(pseudo_mask).cuda()
            current_mask = pseudo_mask

            # mask out foreground areas in previous stages
            masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
            bipartition = F.interpolate(
                pseudo_mask.unsqueeze(0).unsqueeze(0),
                size=init_image_size,
                mode="nearest",
            ).squeeze()
            bipartition_masked = bipartition.cpu().numpy() - masked_out
            bipartition_masked[bipartition_masked <= 0] = 0
            bipartitions.append(bipartition_masked)

            # unsample the eigenvec
            eigvec = second_smallest_vec.reshape(dims)
            eigvec = torch.from_numpy(eigvec).cuda()
            eigvec = F.interpolate(
                eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode="nearest"
            ).squeeze()
            eigvecs.append(eigvec.cpu().numpy())

        return seed, bipartitions, eigvecs

    def get_bbox_from_mask(self, mask):
        """
        Compute the bounding box from the binary mask.

        Args:
            mask (np.ndarray): Binary mask where 1 is the object and 0 is the background.

        Returns:
            tuple: Bounding box as (x_min, y_min, x_max, y_max).
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            return x_min, y_min, x_max, y_max
        else:
            return None  # No bounding box (object not detected)

    def compute_bbox(self, mask):
        """
        Compute the bounding box for a binary mask.

        Args:
            mask (np.ndarray): Binary mask where 1 represents the object.

        Returns:
            list: Bounding box [x_min, y_min, width, height] or None if the mask is empty.
        """
        # Find the non-zero pixels in the mask
        coords = np.column_stack(np.where(mask > 0))
        if coords.shape[0] == 0:
            return None

        # Calculate bounding box as [x_min, y_min, width, height]
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

        return bbox

    def process_image(self, img_path, num_pseudo_masks):
        """
        Process an image using the backbone model to output bounding boxes, individual masks, and combined mask.

        Args:
            img_path (str): Path to the input image.
            num_pseudo_masks (int): Number of pseudo-masks to generate.

        Returns:
            tuple: Bounding boxes, individual masks, and combined mask.
        """
        # I = Image.open(img_path).convert("RGB")
        bipartitions, eigvecs = [], []
        I = Image.open(img_path).convert("RGB")
        I_new = I.resize((self.fixed_size, self.fixed_size), Image.LANCZOS)
        I_resize, w, h, feat_w, feat_h = self.resize_pil(I_new, self.patch_size)
        tensor = self.ToTensor(I_resize).unsqueeze(0).cuda()

        # Use the feature extractor function (which can be customized)
        with torch.no_grad():
            feat = self.feature_extractor_fn(tensor)
        _, bipartition, eigvec = self.maskcut_forward(
            feat,
            [feat_h, feat_w],
            [self.patch_size, self.patch_size],
            [h, w],
            num_pseudo_masks,
        )

        bipartitions += bipartition
        eigvecs += eigvec

        # Initialize variables for storing results
        width, height = I.size
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = []
        masks = []
        for idx, bipartition in enumerate(bipartitions):
            # Post-process pseudo-masks using dense CRF
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)

            # Filter masks with low IoU
            mask1 = torch.from_numpy(bipartition).cuda()
            mask2 = torch.from_numpy(pseudo_mask).to(torch.float64).cuda()
            if metric.IoU(mask1, mask2) < 0.5:
                pseudo_mask = pseudo_mask * -1

            # Create binary mask and resize it to the original image size
            pseudo_mask[pseudo_mask < 0] = 0
            pseudo_mask = np.uint8(pseudo_mask * 255)
            pseudo_mask_resized = np.asarray(
                Image.fromarray(pseudo_mask).resize((width, height))
            )
            pseudo_mask_resized = pseudo_mask_resized.astype(np.uint8)
            # Add to combined mask
            combined_mask = np.maximum(combined_mask, pseudo_mask_resized)

            # Calculate bounding box and store it
            bbox = self.compute_bbox(pseudo_mask_resized)
            if bbox:
                bboxes.append(bbox)
                masks.append(pseudo_mask_resized)

        # Fill the combined mask
        combined_mask_filled = ndimage.binary_fill_holes(combined_mask)

        return bboxes, masks, combined_mask_filled
