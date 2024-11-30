import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Resize
from .utils import center_padding, tokens_to_output
from evals.models.croco_models.croco import CroCoNet
from .util import load_checkpoint
import torchvision

checkpoints = {
    "vitb16": {
        "url": "https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth",  # Replace with actual URL
        "filename": "CroCo.pth",
    }
}


class CROCO(nn.Module):
    def __init__(
        self,
        model_name="vitb16",
        layer=-1,
        output="dense",
        return_multilayer=False,
        add_norm=False,
        return_kqv=False,  # Flag to return K, Q, V
        fixed_size=480,
        mode_selected="k",
        return_layers=None,
        return_cls=False,
    ):
        super().__init__()
        self.arch = "vit"
        self.return_cls = return_cls
        # Load the model within __init__
        self.model = self.load_model(model_name)
        num_layers = len(self.model.enc_blocks)
        self.output = output
        self.checkpoint_name = f"croco_{model_name}_{output}"
        self.patch_size = 16  # CroCoNet typically uses a 16x16 patch size
        self.add_norm = add_norm
        self.return_kqv = return_kqv  # Store the flag to return K, Q, V
        self.fixed_size = fixed_size
        self.resize_transform = Resize((fixed_size, fixed_size))  # Resize for input
        self.mode_selected = mode_selected

        feat_dim = 768
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        # Set up batch normalization layers, adapted to CroCo's feature dimension (512)
        if return_multilayer:
            self.feat_dim = [feat_dim] * 4
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            self.multilayers = [multilayers[-1]]

        self.layer = "-".join(str(_x) for _x in self.multilayers)

        # Define BatchNorm1d layers for each selected layer (adjusted to 512 for CroCoNet)
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )

    def load_model(self, model_name: str):
        """Load the CroCo model from checkpoint."""
        assert model_name in checkpoints.keys(), f"Invalid model: {model_name}"
        ckpt = load_checkpoint(**checkpoints[model_name])
        model = CroCoNet(
            **ckpt.get("croco_kwargs", {})
        )  # Initialize CroCoNet with arguments
        model.load_state_dict(ckpt["model"], strict=True)
        return model.eval()

    def preprocess_image(self, rgb_image):
        """
        Preprocess the RGB image tensor inside the CROCO class.
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
        """Helper function to extract K, Q, V from the last attention layer in CroCoNet."""
        feat_out = {}
        bs = images.shape[0]
        feat_h, feat_w = (
            images.shape[-2] // self.patch_size,
            images.shape[-1] // self.patch_size,
        )

        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output

        self.model.enc_blocks[-1].attn.qkv.register_forward_hook(hook_fn_forward_qkv)

        with torch.no_grad():
            x, pos = self.model.patch_embed(images)
            for blk in self.model.enc_blocks:
                x = blk(x, pos)

        qkv = (
            feat_out["qkv"]
            .reshape(bs, -1, 3, self.model.attn.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k.transpose(1, 2).reshape(bs, -1, feat_h * feat_w)
        q = q.transpose(1, 2).reshape(bs, -1, feat_h * feat_w)
        v = v.transpose(1, 2).reshape(bs, -1, feat_h * feat_w)

        if self.mode_selected == "k":
            feats = k
        elif self.mode_selected == "q":
            feats = q
        elif self.mode_selected == "v":
            feats = v
        elif self.mode_selected == "kqv":
            feats = torch.cat([k, q, v], dim=1)

        return feats

    def forward(self, images):
        if self.return_kqv:
            processed_tensor, feat_w, feat_h = self.preprocess_image(images)
            feats = self.extract_kqv(processed_tensor)
            return feats

        images = F.interpolate(
            images, size=(224, 224), mode="bilinear", align_corners=False
        )
        # pad images (if needed) to ensure it matches patch_size
        # images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        x, pos = self.model.patch_embed(images)

        if self.model.enc_pos_embed is not None:
            x = x + self.model.enc_pos_embed[None, ...]

        B, N, C = x.size()
        masks = torch.zeros((B, N), dtype=bool)
        posvis = pos
        posvis = pos[~masks].view(B, -1, 2)

        embeds = []
        for i, blk in enumerate(self.model.enc_blocks):
            x = blk(x, posvis)
            if i in self.multilayers:
                if self.add_norm:
                    x_batched = self.batchnorms[self.multilayers.index(i)](
                        x.permute(0, 2, 1)
                    ).permute(0, 2, 1)
                    embeds.append(x_batched)
                else:
                    embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = [
            tokens_to_output(self.output, embed, None, (h, w)) for embed in embeds
        ]

        if len(outputs) == 1 and self.return_cls:
            return embeds[0][:, 0]

        return outputs[0] if len(outputs) == 1 else outputs
