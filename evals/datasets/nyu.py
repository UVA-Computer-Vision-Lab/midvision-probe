import os
import numpy as np
import torch
from datasets import load_dataset
import json
from .utils import get_nyu_transforms  # Assuming you have custom transforms in utils.py
from PIL import Image


def NYU(
    train_path,
    test_path,
    split,
    name="nyu",
    image_mean="imagenet",
    center_crop=False,
    rotateflip=False,
    augment_train=False,
):
    assert split in ["train", "trainval", "valid", "test"]
    if split == "test":
        return NYU_test(test_path, image_mean, center_crop)
    else:
        return NYU_geonet(
            train_path,
            split,
            image_mean,
            center_crop,
            augment_train,
            rotateflip=rotateflip,
        )


def make_serializable(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_serializable(i) for i in data]
    else:
        return data


class NYU_test(torch.utils.data.Dataset):
    """
    Dataset loader based on Ishan Misra's SSL benchmark, now updated to load
    from the local processed NYUv2 test data with segmentation and id2label.
    """

    def __init__(self, base_path, image_mean="imagenet", center_crop=False):
        super().__init__()
        self.name = "NYUv2"
        self.center_crop = center_crop
        self.max_depth = 10.0
        self.base_path = base_path

        # get transforms
        image_size = (480, 480) if center_crop else (480, 640)
        self.image_transform, self.shared_transform = get_nyu_transforms(
            image_mean,
            image_size,
            False,
            rotateflip=False,
            additional_targets={"depth": "image", "snorm": "image"},
        )

        self.num_instances = len(os.listdir(os.path.join(self.base_path, "images")))
        print(f"NYUv2 labeled test set: {self.num_instances} instances")

    def __len__(self):
        return self.num_instances

    def __getitem__(self, index):

        image_path = os.path.join(
            self.base_path, "images", f"nyuv2_test_{index}_image.png"
        )
        depth_path = os.path.join(
            self.base_path, "depths", f"nyuv2_test_{index}_depth.npy"
        )
        norm_path = os.path.join(
            self.base_path, "normals", f"nyuv2_test_{index}_norm.npy"
        )
        npz_path = os.path.join(
            self.base_path, "segmentations", f"nyuv2_test_{index}_image.npz"
        )
        metadata_path = os.path.join(
            self.base_path, "metadata", f"nyuv2_test_{index}_metadata.npy"
        )
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Load depth and surface normals
        depth = np.load(depth_path)
        snorm = np.load(norm_path)

        metadata = np.load(metadata_path, allow_pickle=True).item()

        # Load segmentation map and id2label from the npz file
        npz_data = np.load(npz_path, allow_pickle=True)
        segmentation_map = npz_data["panoptic_map"]
        id2label = npz_data["id2label"].item()  # Convert from numpy object
        id2label_serializable = make_serializable(id2label)
        id2label_json = json.dumps(id2label_serializable)
        # Convert the id2label dictionary keys to strings to ensure JSON compatibility
        # id2label_json_compatible = {str(k): v for k, v in id2label.items()}

        # Apply transforms to image
        image = self.image_transform(image)

        # Set max depth to 10
        depth[depth > self.max_depth] = 0

        # Apply center crop if needed
        if self.center_crop:
            image = image[..., 80:-80]
            depth = depth[..., 80:-80]
            snorm = snorm[..., 80:-80]
            segmentation_map = segmentation_map[..., 80:-80]

        # Convert everything to tensors
        depth = torch.tensor(depth).float()[None, :, :]
        snorm = torch.tensor(snorm).float()

        return {
            "image": image,
            "depth": depth,
            "snorm": snorm,
            "segmentation": segmentation_map,
            "metadata": metadata,
            "id2label": id2label_json,
        }


class NYU_geonet(torch.utils.data.Dataset):
    """
    Dataset loader for train/validation set using Parquet files with streaming
    """

    def __init__(
        self,
        base_path,
        split,
        image_mean="imagenet",
        center_crop=False,
        augment_train=False,
        rotateflip=False,
    ):
        super().__init__()
        self.name = "NYUv2"
        self.center_crop = center_crop
        self.max_depth = 10.0

        # get transforms
        image_size = (480, 480) if center_crop else (480, 640)
        augment = augment_train and "train" in split
        self.image_transform, self.shared_transform = get_nyu_transforms(
            image_mean,
            image_size,
            augment,
            rotateflip=rotateflip,
            additional_targets={"depth": "image", "snorm": "image"},
        )
        self.base_path = base_path

        self.image_dir = os.path.join(self.base_path, "images")
        self.depth_dir = os.path.join(self.base_path, "depths")
        self.norm_dir = os.path.join(self.base_path, "normals")
        self.segmentation_dir = os.path.join(self.base_path, "segmentations")

        self.files = [f.split("_image.png")[0] for f in os.listdir(self.image_dir)]
        print(f"NYU-GeoNet {split}: {len(self.files)} instances found.")

    def __len__(self):
        # Length is not required for streaming, return None or any valid number
        return len(self.files)

    def __getitem__(self, index):
        file_base = self.files[index]

        # File paths
        image_path = os.path.join(self.image_dir, f"{file_base}_image.png")
        depth_path = os.path.join(self.depth_dir, f"{file_base}_depth.npy")
        norm_path = os.path.join(self.norm_dir, f"{file_base}_norm.npy")
        npz_path = os.path.join(self.segmentation_dir, f"{file_base}_image.npz")

        # Construct paths to the image, depth, normals, segmentation, and metadata files
        # image_path = os.path.join(self.base_path, "images", f"nyuv2_{index}_image.png")
        # depth_path = os.path.join(self.base_path, "depths", f"nyuv2_{index}_depth.npy")
        # norm_path = os.path.join(self.base_path, "normals", f"nyuv2_{index}_norm.npy")
        # npz_path = os.path.join(
        #     self.base_path, "segmentations", f"nyuv2_{index}_image.npz"
        # )

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)[:480, :640]
        image = self.image_transform(image)
        # Load depth and surface normals
        depth = np.load(depth_path)[:480, :640]
        # set max depth to 10
        depth[depth > self.max_depth] = 0
        snorm = np.load(norm_path)[:480, :640]
        snorm = torch.tensor(snorm).permute(2, 0, 1)

        # Load segmentation map and id2label from the npz file
        npz_data = np.load(npz_path, allow_pickle=True)
        segmentation_map = npz_data["panoptic_map"][:480, :640]
        id2label = npz_data["id2label"].item()  # Convert from numpy object

        # Convert id2label to JSON serializable format
        id2label_serializable = make_serializable(id2label)
        id2label_json = json.dumps(id2label_serializable)

        # center crop
        if self.center_crop:
            image = image[..., 80:-80]
            depth = depth[..., 80:-80]
            snorm = snorm[..., 80:-80]

        if self.shared_transform:
            # put in correct format (h, w, feat)
            image = image.permute(1, 2, 0).numpy()
            snorm = snorm.permute(1, 2, 0).numpy()
            depth = depth[:, :, None]

            # transform
            transformed = self.shared_transform(image=image, depth=depth, snorm=snorm)

            # get back in (feat_dim x height x width)
            image = torch.tensor(transformed["image"]).float().permute(2, 0, 1)
            snorm = torch.tensor(transformed["snorm"]).float().permute(2, 0, 1)
            depth = torch.tensor(transformed["depth"]).float()[None, :, :, 0]
        else:
            # move to torch tensors
            depth = torch.tensor(depth).float()[None, :, :]
            snorm = torch.tensor(snorm).float()

        return {
            "image": image,
            "depth": depth,
            "snorm": snorm,
            "segmentation": segmentation_map,
            "id2label": id2label_json,
        }
