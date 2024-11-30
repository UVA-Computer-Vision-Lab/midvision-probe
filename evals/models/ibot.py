from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torchvision.transforms import Resize
from .ibot_transformers import vit_base, vit_large
from .utils import center_padding, tokens_to_output

BASE_URL = "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot"


class iBOT(torch.nn.Module):
    def __init__(
        self,
        model_type="base",
        output="dense",
        layer=-1,
        return_multilayer=False,
        add_norm=False,
        return_kqv=False,  # Flag to return K, Q, V
        fixed_size=480,
        mode_selected="k",
        return_cls=False,
    ):
        super().__init__()
        self.arch = "vit"
        self.return_cls = return_cls
        assert output in ["gap", "dense", "cls", "dense-cls"]
        self.output = output
        self.return_multilayer = return_multilayer

        model_dict = {
            "base": ("ibot_vitb16", "vitb_16/checkpoint_teacher.pth"),
            "base_in22k": ("ibot_vitb16_in22k", "vitb_16_pt22k/checkpoint_student.pth"),
            "large": ("ibot_vitb16", "vitl_16/checkpoint_teacher.pth"),
            "large_in22k": (
                "ibot_vitb16_in22k",
                "vitl_16_pt22k/checkpoint_student.pth",
            ),
        }

        assert model_type in model_dict

        # Download model checkpoint
        ckpt_name, ckpt_url_path = model_dict[model_type]
        ckpt_path = Path(__file__).parent / f"checkpoint_weights/{ckpt_name}.pth"
        if not ckpt_path.exists():
            download_path = f"{BASE_URL}/{ckpt_url_path}"
            urlretrieve(download_path, ckpt_path)

        # load and cleanup state dict
        state_dict = torch.load(ckpt_path)["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # instantiate model
        model_fn = vit_base if "base" in model_type else vit_large
        feat_dim = 768 if "base" in model_type else 768
        vit = model_fn(patch_size=16, return_all_tokens=True)
        vit.load_state_dict(state_dict, strict=False)
        vit.eval()

        # set parameters
        self.vit = vit
        self.patch_size = 16
        self.checkpoint_name = "$ibot$" + ckpt_name
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        print(f"{model_type} has {num_layers} layers")
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # Define BatchNorm1d layers for each layer in the architecture
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )

        self.add_norm = add_norm  # Store flag to control batch normalization

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)
        self.return_kqv = return_kqv  # Store the flag to return K, Q, V
        self.fixed_size = fixed_size
        self.resize_transform = Resize((fixed_size, fixed_size))
        self.mode_selected = mode_selected

    def preprocess_image(self, rgb_image):
        """
        Preprocess the RGB image tensor inside the DINO class.
        Args:
        - rgb_image: Tensor of shape (C, H, W) representing the RGB image.

        Returns:
        - tensor: Processed tensor ready to be fed into ViT (1, C, H, W).
        - feat_w, feat_h: Feature width and height after patch embedding.
        """
        # Resize the image to the fixed size (fixed_size x fixed_size)
        rgb_resized = self.resize_transform(rgb_image)

        # Calculate feature map dimensions after patch embedding
        feat_w, feat_h = (
            self.fixed_size // self.patch_size,
            self.fixed_size // self.patch_size,
        )

        # Unsqueeze to add batch dimension and return
        tensor = rgb_resized.unsqueeze(0)  # Shape: (1, C, H, W)

        return tensor, feat_w, feat_h

    def extract_kqv(self, images):
        """Helper function to extract K, Q, V from the last attention layer"""
        feat_out = {}
        bs = images.shape[0]
        feat_h, feat_w = (
            images.shape[-2] // self.patch_size,
            images.shape[-1] // self.patch_size,
        )
        if images.ndim == 3:  # Check if the input is 3D (C, H, W)
            images = images.unsqueeze(0)  # Add batch dimension to make it (1, C, H, W)
        if images.ndim == 5:  # Check if the input is 3D (C, H, W)
            images = images.squeeze(0)  # Add batch dimension to make it (1, C, H, W)

        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output

        self.vit._modules["blocks"][-1]._modules["attn"]._modules[
            "qkv"
        ].register_forward_hook(hook_fn_forward_qkv)

        # Forward pass
        with torch.no_grad():
            x = self.vit.prepare_tokens(images)
            for i, blk in enumerate(self.vit.blocks):
                if i < len(self.vit.blocks):
                    x = blk(x)

        bs, nb_head, nb_token = (
            x.shape[0],
            self.vit.blocks[0].attn.num_heads,
            x.shape[1],
        )

        qkv = (
            feat_out["qkv"].reshape(bs, nb_token, 3, nb_head, -1).permute(2, 0, 3, 1, 4)
        )
        # torch.Size([1, 1, 901, 3, 256])
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k.transpose(1, 2).reshape(bs, nb_token, -1)
        q = q.transpose(1, 2).reshape(bs, nb_token, -1)
        v = v.transpose(1, 2).reshape(bs, nb_token, -1)

        if self.mode_selected == "k":
            feats = k[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
        elif self.mode_selected == "q":
            feats = q[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
        elif self.mode_selected == "v":
            feats = v[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
        elif self.mode_selected == "kqv":
            k = k[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
            q = q[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
            v = v[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
            feats = torch.cat([k, q, v], dim=1)
        return feats

    def forward(self, images):
        if self.return_kqv:
            processed_tensor, feat_w, feat_h = self.preprocess_image(images)

            feats = self.extract_kqv(processed_tensor)
            return feats
        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                if len(self.multilayers) == 1 and self.return_cls:
                    return x[:, 0]
                if self.add_norm:
                    x_batch = self.batchnorms[self.multilayers.index(i)](
                        x.permute(0, 2, 1)
                    ).permute(
                        0, 2, 1
                    )  # Exclude the class token
                    embeds.append(x_batch)
                else:
                    embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            spatial = x_i[:, 1:]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
