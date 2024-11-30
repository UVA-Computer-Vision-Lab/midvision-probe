import os

import torch
import torch.nn.functional as F
from torchvision.transforms import Resize
import torch.nn as nn
from .util import initialize_backbone, load_checkpoint, prepare_state_dict
from .utils import center_padding, tokens_to_output

# Define the checkpoints and paths
checkpoints = {
    "vitb16": {
        "url": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
        "filename": "mocov3_vitb16.pth.tar",
    }
}


class MoCoV3(torch.nn.Module):
    def __init__(
        self,
        model_name="vitb16",
        layer=-1,
        arch="vitb16",
        output="dense",
        return_multilayer=False,
        add_norm=False,  # Add flag to control batch normalization
        return_kqv=False,  # Flag to return K, Q, V
        fixed_size=480,
        mode_selected="k",
        return_cls=False,
    ):
        super().__init__()
        self.arch = "vit"
        self.return_cls = return_cls
        # Load the model within __init__
        self.model = self.load_model(arch)

        self.output = output
        self.checkpoint_name = f"$mocov3$_{arch}_{output}"
        feat_dim = 768  # ViT-B/16 has a dimension of 768
        self.patch_size = self.model.patch_embed.proj.kernel_size[0]
        self.add_norm = add_norm

        num_layers = len(self.model.blocks)
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
            self.multilayers = [multilayers[-1]]
        self.layer = "-".join(str(_x) for _x in self.multilayers)

        # Define BatchNorm2d layers for each multilayer
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )
        self.return_kqv = return_kqv  # Store the flag to return K, Q, V
        self.fixed_size = fixed_size
        self.resize_transform = Resize((fixed_size, fixed_size))
        self.mode_selected = mode_selected

    def load_model(self, arch: str):
        assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
        model = initialize_backbone(arch)
        ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
        ckpt = prepare_state_dict(
            ckpt,
            remove_prefix="module.base_encoder.",
            delete_prefixes=["module.predictor."],
        )
        ckpt = prepare_state_dict(ckpt, remove_prefix="module.momentum_encoder.")
        model.load_state_dict(ckpt)
        return model

    def extract_kqv(self, images):
        """Helper function to extract K, Q, V from the last attention layer"""
        feat_out = {}

        # Hook function to capture the qkv
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output

        # Register the hook to capture the qkv from the last attention layer
        last_layer = self.model.blocks[-1]
        last_layer.attn.qkv.register_forward_hook(hook_fn_forward_qkv)

        # Forward pass through the model to trigger the hooks
        with torch.no_grad():
            images = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )
            x = self.model.patch_embed(images)

            cls_token = self.model.cls_token.expand(images.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.model.pos_embed
            x = self.model.pos_drop(x)

            # Forward through the blocks to ensure the hook captures qkv
            for blk in self.model.blocks:
                x = blk(x)

        # Ensure the hook has captured qkv
        if "qkv" not in feat_out:
            raise RuntimeError("Failed to capture qkv from attention block")

        # Extract q, k, v from the hooked output
        qkv = feat_out["qkv"]
        bs, nb_token, _ = qkv.shape
        nb_head = self.model.blocks[-1].attn.num_heads

        # Reshape qkv: [batch_size, nb_token, num_heads, 3 * head_dim]
        qkv = qkv.reshape(bs, nb_token, 3, nb_head, self.feat_dim // nb_head).permute(
            2, 0, 3, 1, 4
        )  # [3, bs, num_heads, nb_token, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Flatten the heads dimension into the embedding dimension
        k = k.transpose(1, 2).reshape(bs, nb_token, -1)
        q = q.transpose(1, 2).reshape(bs, nb_token, -1)
        v = v.transpose(1, 2).reshape(bs, nb_token, -1)

        # Return the selected mode (k, q, v, or concatenation of kqv)
        if self.mode_selected == "k":
            feats = k[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, (nb_token - 1))
        elif self.mode_selected == "q":
            feats = q[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, (nb_token - 1))
        elif self.mode_selected == "v":
            feats = v[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, (nb_token - 1))
        elif self.mode_selected == "kqv":
            k = k[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, (nb_token - 1))
            q = q[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, (nb_token - 1))
            v = v[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, (nb_token - 1))
            feats = torch.cat([k, q, v], dim=1)

        return feats

    def forward(self, images):
        if self.return_kqv:
            return self.extract_kqv(images)
        # Processing the images through the Vision Transformer
        images = F.interpolate(
            images, size=(224, 224), mode="bilinear", align_corners=False
        )

        x = self.model.patch_embed(images)
        cls_token = self.model.cls_token.expand(images.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        embeds = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.multilayers:
                if len(self.multilayers) == 1 and self.return_cls:
                    return x[:, 0]
                if self.add_norm:
                    x_batched = self.batchnorms[self.multilayers.index(i)](
                        x.permute(0, 2, 1)
                    ).permute(
                        0, 2, 1
                    )  # Exclude the class token
                    embeds.append(x_batched[:, 1:])
                else:
                    embeds.append(x[:, 1:])  # Ignore the class token
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, x_i in enumerate(embeds):
            # Reshape to [batch_size, num_channels, height, width]
            b, n, c = x_i.shape
            h = w = int(n**0.5)  # Assuming square spatial dimensions (e.g., 14x14)
            x_i = x_i.permute(0, 2, 1).contiguous().view(b, c, h, w)
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
