import os
import ssl

import gdown
import timm.models as tm
import torch
import torch.nn as nn
import torchvision
import wget
from mmselfsup.models.backbones import ResNet

ckpt_dir = "/p/openvocabdustr/probing_midlevel_vision/code/probe3d/evals/models/checkpoint_weights"
os.makedirs(ckpt_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


class MMSelfSupResnet50(ResNet):
    def __init__(self):
        super().__init__(depth=50)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = super().forward(x)[0]
        return self.adaptive_pool(x).squeeze(-1).squeeze(-1)


def initialize_backbone(arch: str, **kwargs):
    if arch == "resnet50":
        model = torchvision.models.resnet50(weights=None, **kwargs)
    elif arch == "vits16":
        model = tm.vit_small_patch16_224(pretrained=False, **kwargs)
    elif arch == "vitb16":
        model = tm.vit_base_patch16_224(pretrained=False, **kwargs)
    elif arch == "vitl16":
        model = tm.vit_large_patch16_224(pretrained=False, **kwargs)
    elif arch == "vits8":
        model = tm.vision_transformer.VisionTransformer(
            patch_size=8, embed_dim=384, depth=12, num_heads=6, **kwargs
        )
    elif arch == "vitb8":
        model = tm.vision_transformer.VisionTransformer(
            patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs
        )
    else:
        raise NotImplementedError(f"Arch {arch} not implemented for now.")

    if arch.startswith("vit"):
        model.head = torch.nn.Identity()  # remove the original head
    else:
        model.fc = torch.nn.Identity()  # remove the final fc layer
    return model


def load_checkpoint(url: str, filename: str):
    path = os.path.join(ckpt_dir, filename)
    if not os.path.exists(path):
        print(f"Downloading checkpoint file: {url}")
        if url.startswith("https://drive.google.com"):
            gdown.download(url, path, quiet=False, fuzzy=True)
        else:
            ssl._create_default_https_context = ssl._create_unverified_context
            wget.download(url, path)
    ckpt = torch.load(path, map_location=device)
    return ckpt


def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def unfreeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True
    model.train()


def partially_freeze_model(model: torch.nn.Module):
    """Freeze first half of the model"""
    unfreeze_model(model)
    if model.__class__.__name__ == "VisionTransformer":
        for param in model.patch_embed.parameters():
            param.requires_grad = False
        for param in model.pos_embed.parameters():
            param.requires_grad = False
        for param in model.blocks[: len(model.blocks) // 2].parameters():
            param.requires_grad = False
    elif model.__class__.__name__ == "ResNet":
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.maxpool.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
    else:
        raise NotImplementedError(
            f"Model {model.__class__.__name__} not implemented for now."
        )


def prepare_state_dict(
    state_dict, remove_prefix=None, delete_prefixes=("head.", "fc.")
):
    for k in list(state_dict.keys()):
        if remove_prefix is not None:
            if k.startswith(remove_prefix):
                state_dict[k[len(remove_prefix) :]] = state_dict[k]
                del state_dict[k]

    if delete_prefixes is not None:
        for k in list(state_dict.keys()):
            for del_prefix in delete_prefixes:
                if k.startswith(del_prefix):
                    del state_dict[k]
    return state_dict
