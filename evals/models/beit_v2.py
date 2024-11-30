import torch
import torch.nn.functional as F
import torch.nn as nn
from .impl_utils.beit_model import beit_base_patch16_224
from .impl_utils.beit_state_dict import load_state_dict, prepare_state_dict
from .util import load_checkpoint
from torchvision.transforms import Resize

checkpoints = {
    "beit_vitb16": {
        "url": "https://drive.google.com/uc?id=1v9MzCK4GVTKiwNA0tdICAmVPZFgV8OVx",
        "filename": "beit_v2_vitb16.pth",
    }
}


class BEiTV2(torch.nn.Module):
    def __init__(
        self,
        model_name="beit_vitb16",
        layer=-1,
        arch="beit_vitb16",
        output="dense",
        return_multilayer=False,
        add_norm=False,
        return_kqv=False,  # Flag to return K, Q, V
        fixed_size=224,
        mode_selected="k",
        return_cls=False,
    ):
        super().__init__()
        self.arch = "vit"
        self.return_cls = return_cls
        # Load the model within __init__
        self.model = self.load_model(arch)

        self.output = output
        feat_dim = 768  # BEiT-B/16 also has a dimension of 768
        self.patch_size = self.model.patch_embed.proj.kernel_size[0]

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
        self.checkpoint_name = f"$beit_v2$_{model_name}_{output}_{self.layer}"

        # Define BatchNorm2d layers for each output
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )

        self.add_norm = add_norm
        self.return_kqv = return_kqv  # Store the flag to return K, Q, V
        self.fixed_size = fixed_size
        self.resize_transform = Resize((fixed_size, fixed_size))
        self.mode_selected = mode_selected

    def load_model(self, arch: str):
        assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
        model = beit_base_patch16_224(
            pretrained=False,
            num_classes=0,
            drop_rate=0.0,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=True,
            use_abs_pos_emb=False,
            init_values=0.1,
            qkv_bias=True,
        )
        model.head = torch.nn.Identity()
        ckpt = load_checkpoint(**checkpoints[arch])["model"]
        ckpt, model = prepare_state_dict(ckpt, model)
        load_state_dict(model, ckpt)
        return model

    def interpolate_pos_encoding(self, x, w, h):
        """
        Interpolate positional embeddings to match the input resolution.
        """
        npatch = x.shape[1] - 1  # Exclude the class token
        N = self.model.pos_embed.shape[1] - 1  # Exclude the class token

        if npatch == N and w == h:
            return self.model.pos_embed[:, 1:]

        # Calculate the new grid size
        class_pos_embed = self.model.pos_embed[:, 0:1]  # Class token
        patch_pos_embed = self.model.pos_embed[
            :, 1:
        ]  # Patch token positional embeddings
        dim = patch_pos_embed.shape[-1]

        # Compute the grid size for the positional embeddings
        gs_old = int(N**0.5)
        gs_new = int(npatch**0.5)

        patch_pos_embed = patch_pos_embed.reshape(1, gs_old, gs_old, dim).permute(
            0, 3, 1, 2
        )  # (1, D, H, W)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(gs_new, gs_new), mode="bilinear", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(
            1, gs_new * gs_new, dim
        )

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def extract_kqv(self, x):
        feat_h, feat_w = (
            x.shape[-2] // self.patch_size,
            x.shape[-1] // self.patch_size,
        )
        if x.ndim == 3:  # Check if the input is 3D (C, H, W)
            x = x.unsqueeze(0)  # Add batch dimension to make it (1, C, H, W)
        if x.ndim == 5:  # Check if the input is 3D (C, H, W)
            x = x.squeeze(0)  # Add batch dimension to make it (1, C, H, W)
        B, nc, w, h = x.shape
        x = self.model.patch_embed(x)  # Using self.model for patch embedding
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.model.cls_token.expand(
            batch_size, -1, -1
        )  # Expand cls token from self.model
        x = torch.cat((cls_tokens, x), dim=1)

        if self.model.pos_embed is not None:
            if x.shape[1] != self.model.pos_embed.shape[1]:
                x = x + self.model.interpolate_pos_encoding(x, w, h)
            else:
                x = x + self.model.pos_embed

        x = self.model.pos_drop(x)

        rel_pos_bias = (
            self.model.rel_pos_bias() if self.model.rel_pos_bias is not None else None
        )

        # Hook for capturing qkv from the last attention block
        feat_out = {}

        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output

        # Get the index of the last block dynamically
        last_block_index = len(self.model.blocks) - 1

        # Access the qkv layer in the last block's attention module
        attn_qkv_layer = self.model.get_submodule(f"blocks.{last_block_index}.attn.qkv")

        # Register the forward hook on the qkv layer
        attn_qkv_layer.register_forward_hook(hook_fn_forward_qkv)

        # Forward pass through the model
        for i, blk in enumerate(self.model.blocks):
            if i < last_block_index:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                if "qkv" not in feat_out:
                    qkv_output = blk.attn.qkv(x)  # Directly call qkv forward
                    qkv_output = qkv_output.reshape(
                        x.shape[0], x.shape[1], 3, -1, blk.attn.num_heads
                    ).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv_output[0], qkv_output[1], qkv_output[2]
                    feat_out["qkv"] = (q, k, v)

                x = blk(x, return_attention=True, rel_pos_bias=rel_pos_bias)

        x = self.model.norm(x)

        # If extract_kqv flag is set, return q, k, v from the last attention block
        q, k, v = feat_out["qkv"]
        # Reshape k, q, v for each patch
        k = k.transpose(1, 2).reshape(
            q.shape[0], -1, q.shape[2]
        )  # bs, head_dim, nb_token
        q = q.transpose(1, 2).reshape(q.shape[0], -1, q.shape[2])
        v = v.transpose(1, 2).reshape(v.shape[0], -1, v.shape[2])

        # Now handle the output based on the mode selected
        if self.mode_selected == "k":
            # Return only the keys, excluding the CLS token (index 0)
            return k[:, :, 1:].transpose(1, 2).reshape(k.shape[0], -1, feat_h * feat_w)

        elif self.mode_selected == "q":
            # Return only the queries, excluding the CLS token
            return q[:, :, 1:].transpose(1, 2).reshape(q.shape[0], -1, feat_h * feat_w)

        elif self.mode_selected == "v":
            # Return only the values, excluding the CLS token
            return v[:, :, 1:].transpose(1, 2).reshape(v.shape[0], -1, feat_h * feat_w)

        elif self.mode_selected == "kqv":
            # Concatenate k, q, and v along the feature dimension, excluding CLS token
            k = k[:, :, 1:].transpose(1, 2).reshape(k.shape[0], -1, feat_h * feat_w)
            q = q[:, :, 1:].transpose(1, 2).reshape(q.shape[0], -1, feat_h * feat_w)
            v = v[:, :, 1:].transpose(1, 2).reshape(v.shape[0], -1, feat_h * feat_w)
            return torch.cat([k, q, v], dim=1)

        raise ValueError(f"Invalid mode selected: {self.mode_selected}")

    def load_checkpoint(self, ckpt_path: str):
        # Load the entire checkpoint
        checkpoint = torch.load(ckpt_path)

        # Extract only the model weights
        model_state_dict = checkpoint["model"]

        # Load the model weights
        self.model.load_state_dict(model_state_dict)

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

    def forward(self, images):
        if self.return_kqv:
            processed_tensor, feat_w, feat_h = self.preprocess_image(images)

            feats = self.extract_kqv(processed_tensor)
            return feats
        # Resize images to the appropriate size for the model
        images = F.interpolate(
            images, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Extract features using the model's forward_features method
        embeds = []
        x = self.model.forward_features(images, return_all_tokens=True)

        # Process the output at specified layers
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.multilayers:
                if len(self.multilayers) == 1 and self.return_cls:
                    return x[:, 0]
                if self.add_norm:
                    x_batch = self.batchnorms[self.multilayers.index(i)](
                        x.permute(0, 2, 1)
                    ).permute(0, 2, 1)
                    embeds.append(x_batch[:, 1:])
                else:
                    embeds.append(x[:, 1:])  # Ignore the class token
                if len(embeds) == len(self.multilayers):
                    break

        # Reshape and return the outputs
        outputs = []
        for idx, x_i in enumerate(embeds):
            b, n, c = x_i.shape
            h = w = int(n**0.5)  # Assuming square spatial dimensions (e.g., 14x14)
            x_i = x_i.permute(0, 2, 1).contiguous().view(b, c, h, w)
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
