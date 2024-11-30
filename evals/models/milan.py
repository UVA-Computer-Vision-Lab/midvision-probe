import torch
import torch.nn.functional as F
import torch.nn as nn
from .util import load_checkpoint, initialize_backbone, prepare_state_dict
from torchvision.transforms import Resize

checkpoints = {
    "vitb16": {
        "url": "https://drive.google.com/uc?id=18UYGG_1r5SJyAgj1ykOfoqECdVBoFLoz",  # TODO download not working via gdown
        "filename": "milan_vitb16.pth.tar",
    },
}


class MILAN(torch.nn.Module):
    def __init__(
        self,
        model_name="milan_vitb16",
        layer=-1,
        arch="vitb16",
        output="dense",
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
        # Load the model within __init__
        self.model = self.load_model(arch)

        self.output = output
        feat_dim = 768
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
        self.checkpoint_name = f"$milan$_{model_name}_{output}_{self.layer}"
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )
        self.add_norm = add_norm
        self.return_kqv = return_kqv  # Store the flag to return K, Q, V
        self.fixed_size = fixed_size
        self.resize_transform = Resize((fixed_size, fixed_size))
        self.mode_selected = mode_selected

    def load_model(self, arch: str, **kwargs):
        assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
        model = initialize_backbone(arch, **kwargs)
        ckpt = load_checkpoint(**checkpoints[arch])["model"]
        ckpt = prepare_state_dict(ckpt, delete_prefixes=("mask_token", "decoder"))
        model.load_state_dict(ckpt)
        # model.load_state_dict(ckpt, strict=False)
        return model

    def extract_kqv(self, images):
        """Helper function to extract K, Q, V from the last attention layer"""
        feat_out = {}
        breakpoint()

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
            x = self.model.forward_features(images)

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
