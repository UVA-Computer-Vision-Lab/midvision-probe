import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
from torchvision import transforms


class VOC(torch.utils.data.Dataset):
    """
    Dataset loader based on the VOC2007 dataset.
    """

    def __init__(self, cfg, split, name="voc", image_mean="imagenet", fixed_size=480):
        super().__init__()
        assert split in ["trainval", "test"], "Invalid split! Use 'trainval' or 'test'."

        # Convert string paths to Path objects
        self.seg_path = Path(
            cfg.trainval_path if split == "trainval" else cfg.test_path
        )
        self.jpeg_dir = Path(
            cfg.trainval_jpeg_dir if split == "trainval" else cfg.test_jpeg_dir
        )
        self.xml_dir = Path(
            cfg.trainval_xml_dir if split == "trainval" else cfg.test_xml_dir
        )

        self.fixed_size = fixed_size

        # Image transformation applied to all images
        self.ToTensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # Get all .png files from the segmentation directory
        self.segmentation_list = list(self.seg_path.glob("*.png"))

        # Match .png file names with corresponding .jpg and .xml files
        self.jpeg_list = []
        self.xml_files = []
        for seg_path in self.segmentation_list:
            image_name = seg_path.stem
            jpeg_path = self.jpeg_dir / f"{image_name}.jpg"
            xml_path = self.xml_dir / f"{image_name}.xml"

            # Check if both the .jpg and .xml files exist
            if jpeg_path.exists() and xml_path.exists():
                self.jpeg_list.append(str(jpeg_path))
                self.xml_files.append(str(xml_path))

        print(f"VOC dataset ({split}): {len(self.segmentation_list)} instances")

    def __len__(self):
        return len(self.segmentation_list)

    def __getitem__(self, index):
        # Load and resize the original image
        orig_image = Image.open(self.jpeg_list[index]).convert("RGB")
        orig_image = orig_image.resize(
            (self.fixed_size, self.fixed_size), Image.LANCZOS
        )
        orig_image_normalized = self.ToTensor(orig_image)
        orig_image = transforms.ToTensor()(orig_image)
        # Load and parse the corresponding XML file
        xml_file = self.xml_files[index]
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract objects and bounding boxes from the XML
        objects = root.findall("object")
        num_objects = len(objects)

        # Extract bounding boxes
        bboxes = []
        for obj in objects:
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bboxes.append([xmin, ymin, xmax, ymax])

        # Load the segmentation image, create a binary mask, and convert it to a tensor
        segmentation_image = Image.open(self.segmentation_list[index])
        gray_image = segmentation_image.convert("L").resize(
            (self.fixed_size, self.fixed_size), Image.NEAREST
        )
        gt_binary_mask = torch.from_numpy(
            (np.array(gray_image) > 0).astype(np.float32)
        ).unsqueeze(0)

        return {
            "original_image": orig_image_normalized,
            "original_image_rgb": orig_image,
            "gt_binary_mask": gt_binary_mask,
            "num_objects": num_objects,
            # "bboxes": bboxes,  # List of bounding boxes for each object
        }
