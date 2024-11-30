from __future__ import annotations

import torch
from torch import nn
from transformers import ViTMAEForPreTraining

from .utils import get_2d_sincos_pos_embed, tokens_to_output
from torchvision.transforms import Resize


class MAE(nn.Module):
    def __init__(
        self,
        checkpoint="facebook/vit-mae-base",
        output="dense",
        layer=-1,
        return_multilayer=False,
        add_norm=False,
        return_kqv=False,  # Flag to return K, Q, V
        fixed_size=480,
        mode_selected="k",
        return_cls=False,
    ):
        """Code based on transformer database"""
        super().__init__()
        self.arch = "vit"
        self.return_cls = return_cls
        assert output in ["cls", "gap", "dense"], "Options: [cls, gap, dense]"
        self.output = output

        self.checkpoint_name = "$mae$" + checkpoint.split("/")[1]

        self.vit = ViTMAEForPreTraining.from_pretrained(checkpoint).vit
        self.vit = self.vit.eval()

        # resize pos embedding
        # resize embedding for new size
        patch_size = self.vit.config.patch_size
        self.patch_size = patch_size
        self.layer = layer

        self.image_size = self.vit.embeddings.patch_embeddings.image_size
        self.feat_h = self.image_size[0] // self.patch_size
        self.feat_w = self.image_size[1] // self.patch_size

        feat_dim = self.vit.config.hidden_size
        num_layers = len(self.vit.encoder.layer)
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

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )
        self.add_norm = add_norm
        self.return_kqv = return_kqv  # Store the flag to return K, Q, V
        self.fixed_size = fixed_size
        self.resize_transform = Resize((fixed_size, fixed_size))
        self.mode_selected = mode_selected

    def resize_pos_embed(self, image_size):
        assert image_size[0] % self.patch_size == 0
        assert image_size[1] % self.patch_size == 0
        self.feat_h = image_size[0] // self.patch_size
        self.feat_w = image_size[1] // self.patch_size
        embed_dim = self.vit.config.hidden_size
        self.vit.embeddings.patch_embeddings.image_size = image_size
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.feat_h, self.feat_w), add_cls_token=True
        )
        # there should be an easier way ... TODO
        device = self.vit.embeddings.patch_embeddings.projection.weight.device
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0).to(device=device),
            requires_grad=False,
        )

    def embed_forward(self, embedder, pixel_values):
        # No masking here ...
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = embedder.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + embedder.position_embeddings[:, 1:, :]

        # append cls token
        cls_token = embedder.cls_token + embedder.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings

    def extract_kqv(self, images):
        """Helper function to extract K, Q, V from the last attention layer"""
        feat_out = {}
        bs = images.shape[0]
        feat_h, feat_w = (
            images.shape[-2] // self.patch_size,
            images.shape[-1] // self.patch_size,
        )

        # Hook functions to capture the query, key, and value
        def hook_fn_forward_q(module, input, output):
            feat_out["query"] = output

        def hook_fn_forward_k(module, input, output):
            feat_out["key"] = output

        def hook_fn_forward_v(module, input, output):
            feat_out["value"] = output

        # Register hooks for query, key, and value in the last attention block
        self.vit._modules["encoder"]._modules["layer"][-1]._modules[
            "attention"
        ]._modules["attention"].query.register_forward_hook(hook_fn_forward_q)

        self.vit._modules["encoder"]._modules["layer"][-1]._modules[
            "attention"
        ]._modules["attention"].key.register_forward_hook(hook_fn_forward_k)

        self.vit._modules["encoder"]._modules["layer"][-1]._modules[
            "attention"
        ]._modules["attention"].value.register_forward_hook(hook_fn_forward_v)

        # Get the head_mask for the attention layers
        head_mask = self.vit.get_head_mask(None, self.vit.config.num_hidden_layers)

        # Forward pass through the model with head_mask
        with torch.no_grad():
            x = self.embed_forward(self.vit.embeddings, images)
            for i, blk in enumerate(self.vit.encoder.layer):
                if isinstance(x, tuple):
                    x = x[0]
                x = blk(x, head_mask=head_mask[i])  # Pass head_mask to each block

        # Ensure the hooks have captured query, key, and value
        if "query" not in feat_out or "key" not in feat_out or "value" not in feat_out:
            raise RuntimeError(
                "Failed to capture query, key, or value from attention block"
            )

        # Extract q, k, v from the hooked outputs
        q = feat_out["query"]
        k = feat_out["key"]
        v = feat_out["value"]

        # Ensure the batch size and token count matches expectations
        bs, nb_token, _ = q.shape
        nb_head = self.vit.config.num_attention_heads

        # Reshape q, k, v: [batch_size, nb_token, num_heads, head_dim]
        q = q.reshape(bs, nb_token, nb_head, self.feat_dim // nb_head).permute(
            0, 2, 1, 3
        )  # [bs, num_heads, nb_token, head_dim]
        k = k.reshape(bs, nb_token, nb_head, self.feat_dim // nb_head).permute(
            0, 2, 1, 3
        )  # [bs, num_heads, nb_token, head_dim]
        v = v.reshape(bs, nb_token, nb_head, self.feat_dim // nb_head).permute(
            0, 2, 1, 3
        )  # [bs, num_heads, nb_token, head_dim]

        # Flatten the number of heads into the embedding dimension
        k = k.transpose(1, 2).reshape(bs, nb_token, -1)
        q = q.transpose(1, 2).reshape(bs, nb_token, -1)
        v = v.transpose(1, 2).reshape(bs, nb_token, -1)

        # Return the selected mode (k, q, v, or concatenation of kqv)
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
        # check if positional embeddings are correct
        if self.image_size != images.shape[-2:]:
            self.resize_pos_embed(images.shape[-2:])

        if self.return_kqv:
            return self.extract_kqv(images)
        # from MAE implementation
        head_mask = self.vit.get_head_mask(None, self.vit.config.num_hidden_layers)

        # ---- hidden ----
        embedding_output = self.embed_forward(self.vit.embeddings, images)
        encoder_outputs = self.vit.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=self.vit.config.output_attentions,
            output_hidden_states=True,
            return_dict=self.vit.config.return_dict,
        )

        outputs = []
        for idx, layer_i in enumerate(self.multilayers):
            x_i = encoder_outputs.hidden_states[layer_i]
            if len(self.multilayers) == 1 and self.return_cls:
                return x_i[:, 0]
            if self.add_norm:
                x_i_batchnorm = self.batchnorms[idx](x_i.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # Exclude the class token
                x_i_batchnorm = tokens_to_output(
                    self.output,
                    x_i_batchnorm[:, 1:],
                    x_i_batchnorm[:, 0],
                    (self.feat_h, self.feat_w),
                )
                outputs.append(x_i_batchnorm)
            else:
                x_i = tokens_to_output(
                    self.output, x_i[:, 1:], x_i[:, 0], (self.feat_h, self.feat_w)
                )
                outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
