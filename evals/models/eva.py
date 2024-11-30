import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Resize
from .util import initialize_backbone, load_checkpoint, prepare_state_dict

checkpoints = {
    "vitb16": {
        "url": "https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k_20221226-26d90f07.pth",
        "filename": "eva_vitb16.pth",
    }
}


rename_dict = {
    "layers.": "blocks.",
    "patch_embed.projection": "patch_embed.proj",
    ".ln1": ".norm1",
    ".ln2": ".norm2",
    "ln1.weight": "norm.weight",
    "ln1.bias": "norm.bias",
    "ffn.blocks.0.0.": "mlp.fc1.",
    "ffn.blocks.1.": "mlp.fc2.",
}


class EVA(torch.nn.Module):
    def __init__(
        self,
        model_name="eva_vitb16",
        layer=-1,
        arch="vitb16",
        output="dense",
        return_multilayer=False,
        add_norm=False,  # Add flag to control normalization
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
        self.checkpoint_name = f"$eva$_{model_name}_{output}_{self.layer}"

        # Define BatchNorm1d layers for each layer in the architecture
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )
        self.add_norm = add_norm  # Store flag to control batch normalization
        self.return_kqv = return_kqv  # Store the flag to return K, Q, V
        self.fixed_size = fixed_size
        self.resize_transform = Resize((fixed_size, fixed_size))
        self.mode_selected = mode_selected

    def load_model(self, arch: str, **kwargs):
        assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
        model = initialize_backbone(arch, **kwargs)
        ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
        ckpt = prepare_state_dict(
            ckpt,
            remove_prefix="backbone.",
            delete_prefixes=("neck", "target_generator"),
        )

        for k in list(ckpt.keys()):
            if k == "norm1.weight":
                print()
            old_k = k
            for key, val in rename_dict.items():
                k = k.replace(key, val)
            if k != old_k:
                ckpt[k] = ckpt[old_k]
                del ckpt[old_k]

        model.load_state_dict(ckpt)
        return model

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

        self.model._modules["blocks"][-1]._modules["attn"]._modules[
            "qkv"
        ].register_forward_hook(hook_fn_forward_qkv)

        # Forward pass
        with torch.no_grad():
            x = self.model.patch_embed(images)
            x = self.model._pos_embed(x)
            for i, blk in enumerate(self.model.blocks):
                if i < len(self.model.blocks):
                    x = blk(x)

        bs, nb_head, nb_token = (
            x.shape[0],
            self.model.blocks[0].attn.num_heads,
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
        x = self.model.forward_features(images)  # Use the built-in method from timm

        # Process the output at specified layers
        for i, blk in enumerate(self.model.blocks):
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
                    embeds.append(x_batch[:, 1:])
                else:
                    embeds.append(x[:, 1:])  # Ignore the class token
                if len(embeds) == len(self.multilayers):
                    break

        # Reshape and return the outputs
        outputs = []
        for x_i in embeds:
            b, n, c = x_i.shape
            h = w = int(n**0.5)  # Assuming square spatial dimensions (e.g., 14x14)
            x_i = x_i.permute(0, 2, 1).contiguous().view(b, c, h, w)
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
