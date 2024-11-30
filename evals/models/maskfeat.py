import torch
import torch.nn.functional as F
from mmcls.models import VisionTransformer as MMViT
import torch.nn as nn
from .util import load_checkpoint, prepare_state_dict
from torchvision.transforms import Resize

GLOBAL_POOL = False

checkpoints = {
    "vitb16": {
        "url": "https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221101-6dfc8bf3.pth",
        "filename": "maskfeat_vitb16.pth",
    },
}


class MMSelfSupMaskFeatViT(MMViT):
    def __init__(self, global_pool=GLOBAL_POOL):
        super().__init__(img_size=224, patch_size=16)
        self.global_pool = global_pool

    def forward(self, x):
        x = super().forward(x)
        if self.global_pool:
            x = x[0][0]  # Take the CLS token
            return x.reshape(x.shape[0], 768, -1).mean(dim=-1)
        else:
            return x[0][1]  # Return the spatial features


def load_model(arch: str, global_pool=GLOBAL_POOL, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = MMSelfSupMaskFeatViT(global_pool=global_pool)
    ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
    ckpt = prepare_state_dict(
        ckpt,
        remove_prefix="backbone.",
        delete_prefixes=["target_generator.", "neck.", "mask_token"],
    )
    model.load_state_dict(ckpt)
    return model


class MASKFEAT(torch.nn.Module):
    def __init__(
        self,
        model_name="maskfeat_vitb16",
        arch="vitb16",
        output="dense",
        global_pool=GLOBAL_POOL,
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
        self.model = load_model(arch, global_pool=global_pool)

        self.output = output
        feat_dim = 768  # ViT-B/16 has a dimension of 768
        self.patch_size = self.model.patch_embed.projection.kernel_size[0]

        num_layers = len(self.model.layers)
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
        self.checkpoint_name = f"$maskfeat$_{model_name}_{output}_{self.layer}"

        # Define BatchNorm2d layers for each multilayer
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )
        self.add_norm = add_norm
        self.return_kqv = return_kqv  # Store the flag to return K, Q, V
        self.fixed_size = fixed_size
        self.resize_transform = Resize((fixed_size, fixed_size))
        self.mode_selected = mode_selected

    def extract_kqv(self, images):
        """Helper function to extract K, Q, V from the last attention layer"""
        feat_out = {}
        bs = images.shape[0]
        feat_h, feat_w = (
            images.shape[-2] // self.patch_size,
            images.shape[-1] // self.patch_size,
        )

        # Hook function to capture the qkv
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output

        # Register the hook to capture the qkv from the last attention layer
        last_layer = self.model.layers[-1]
        last_layer.attn.qkv.register_forward_hook(hook_fn_forward_qkv)

        # Forward pass through the model to trigger the hooks
        with torch.no_grad():
            images = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )
            x, patch_resolution = self.model.patch_embed(images)

            # Add class token if it exists
            if self.model.cls_token is not None:
                cls_token = self.model.cls_token.expand(images.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)

            # Add positional embedding
            x = x + self.model.resize_pos_embed(
                self.model.pos_embed,
                self.model.patch_resolution,
                patch_resolution,
                mode=self.model.interpolate_mode,
                num_extra_tokens=self.model.num_extra_tokens,
            )
            x = self.model.drop_after_pos(x)

            # Forward through the layers
            for i, blk in enumerate(self.model.layers):
                x = blk(x)
                if i == len(self.model.layers) - 1:
                    break

        # Ensure the hook has captured qkv
        if "qkv" not in feat_out:
            raise RuntimeError("Failed to capture qkv from attention block")

        # Extract q, k, v from the hooked output
        qkv = feat_out["qkv"]
        qkv = qkv.reshape(
            bs,
            -1,
            3,
            self.model.layers[0].attn.num_heads,
            self.feat_dim // self.model.layers[0].attn.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, bs, num_heads, nb_token, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Flatten the heads dimension into the embedding dimension
        k = k.transpose(1, 2).reshape(bs, -1, self.feat_dim)
        q = q.transpose(1, 2).reshape(bs, -1, self.feat_dim)
        v = v.transpose(1, 2).reshape(bs, -1, self.feat_dim)

        # Return the selected mode (k, q, v, or concatenation of kqv)
        if self.mode_selected == "k":
            feats = (
                k[:, 1:]
                .transpose(1, 2)
                .reshape(bs, self.feat_dim, patch_resolution[0] * patch_resolution[1])
            )
        elif self.mode_selected == "q":
            feats = (
                q[:, 1:]
                .transpose(1, 2)
                .reshape(bs, self.feat_dim, patch_resolution[0] * patch_resolution[1])
            )
        elif self.mode_selected == "v":
            feats = (
                v[:, 1:]
                .transpose(1, 2)
                .reshape(bs, self.feat_dim, patch_resolution[0] * patch_resolution[1])
            )
        elif self.mode_selected == "kqv":
            k = (
                k[:, 1:]
                .transpose(1, 2)
                .reshape(bs, self.feat_dim, patch_resolution[0] * patch_resolution[1])
            )
            q = (
                q[:, 1:]
                .transpose(1, 2)
                .reshape(bs, self.feat_dim, patch_resolution[0] * patch_resolution[1])
            )
            v = (
                v[:, 1:]
                .transpose(1, 2)
                .reshape(bs, self.feat_dim, patch_resolution[0] * patch_resolution[1])
            )
            feats = torch.cat([k, q, v], dim=1)

        return feats

    def forward(self, images):
        if self.return_kqv:
            processed_tensor = F.interpolate(
                images,
                size=(self.fixed_size, self.fixed_size),
                mode="bilinear",
                align_corners=False,
            )
            return self.extract_kqv(processed_tensor)

        images = F.interpolate(
            images, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Forward pass through the patch embedding and position embedding
        x, patch_resolution = self.model.patch_embed(images)

        # Add class token if it exists
        if self.model.cls_token is not None:
            cls_token = self.model.cls_token.expand(images.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = x + self.model.resize_pos_embed(
            self.model.pos_embed,
            self.model.patch_resolution,
            patch_resolution,
            mode=self.model.interpolate_mode,
            num_extra_tokens=self.model.num_extra_tokens,
        )
        x = self.model.drop_after_pos(x)

        embeds = []
        for i, blk in enumerate(self.model.layers):
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
        for x_i in embeds:
            b, n, c = x_i.shape
            h = w = int(n**0.5)  # Assuming square spatial dimensions (e.g., 14x14)
            x_i = x_i.permute(0, 2, 1).contiguous().view(b, c, h, w)
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
