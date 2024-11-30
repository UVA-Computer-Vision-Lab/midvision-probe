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

from datasets import load_dataset
from .transforms import task_transform
from tqdm import tqdm  # Import tqdm for progress visualization
import torch


def Taskonomy(
    snorm_path,
    other_path,
    split,
    task,
    name="taskonomy",
    image_mean="imagenet",
    center_crop=False,
    rotateflip=False,
    augment_train=False,
):
    assert split in ["train", "valid", "test"]
    if split == "train":
        if task == "normal":
            dataset = load_dataset(snorm_path, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(other_path, split=split, trust_remote_code=True)
    elif split == "val":
        if task == "normal":
            dataset = load_dataset(snorm_path, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(other_path, split=split, trust_remote_code=True)
    else:
        if task == "normal":
            dataset = load_dataset(snorm_path, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(other_path, split=split, trust_remote_code=True)

    return dataset


class TaskonomyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, task):
        self.dataset = dataset
        self.task = task

    def __getitem__(self, idx):
        item = self.dataset[idx]
        task = self.task
        if self.task == "depth":
            task = "depth_euclidean"
        # Only keep 'rgb', the specified task, and 'mask_valid'
        transformed_item = {
            "rgb": task_transform(item["rgb"], "rgb"),
            self.task: task_transform(
                item[self.task], task
            ),  # Apply task_transform to the specified task
            "mask_valid": task_transform(item["mask_valid"], "mask_valid"),
        }

        return transformed_item

    def __len__(self):
        return len(self.dataset)
