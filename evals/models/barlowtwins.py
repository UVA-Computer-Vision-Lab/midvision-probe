import torch
from torch import nn
from torchvision.transforms import Resize
from .util import MMSelfSupResnet50 as Resnet50
from .util import initialize_backbone, load_checkpoint, prepare_state_dict

checkpoints = {
    "resnet50": {
        "url": "https://download.openmmlab.com/mmselfsup/1.x/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220825-57307488.pth",
        "filename": "barlowtwins_resnet50.pth",
    },
}


def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = Resnet50()
    ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
    ckpt = prepare_state_dict(
        ckpt,
        remove_prefix="backbone.",
        delete_prefixes=["data_preprocessor.", "neck.", "head."],
    )  # TODO what does data_preprocessor do
    model.load_state_dict(ckpt)
    return model


class BARLOWTWINS(nn.Module):
    def __init__(
        self,
        arch="resnet50",
        return_layers=None,
        output="dense",
        return_multilayer=False,
        add_norm=False,
        return_kqv=False,  # Flag to return K, Q, V
        fixed_size=480,
        mode_selected="k",
        return_cls=False,
    ):
        super().__init__()
        self.arch = arch
        self.return_cls = return_cls
        self.model = load_model(arch)
        self.output = output
        self.return_layers = (
            return_layers if return_layers is not None else [0, 1, 2, 3, 4]
        )

        # Define a list of layers based on the architecture
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.relu,
                    self.model.maxpool,
                ),
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4,
            ]
        )

        # Determine the feature dimensions
        self.feat_dims = [
            (64, 240),
            (256, 120),
            (512, 60),
            (1024, 30),
            (2048, 15),
        ]  # Example dimensions based on ResNet architecture
        feat_dims = [self.feat_dims[i] for i in self.return_layers]
        self.checkpoint_name = f"$barlowtwins$_{arch}_{output}_{self.return_layers}"
        self.patch_size = 0
        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = self.return_layers
        else:
            self.feat_dim = feat_dims[
                -1
            ]  # Get the dimension of the last layer in return_layers
            self.multilayers = [self.return_layers[-1]]

        self.layer = "-".join(str(_x) for _x in self.multilayers)
        # Define BatchNorm layers for each output layer
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm2d(feat_dim[0]) for feat_dim in self.feat_dims]
        )
        self.add_norm = add_norm
        self.return_kqv = return_kqv  # Store the flag to return K, Q, V
        self.fixed_size = fixed_size
        self.resize_transform = Resize((fixed_size, fixed_size))
        self.mode_selected = mode_selected

    def forward(self, x):
        outputs = []
        x = self.resize_transform(x)

        # Pass input through each layer up to the highest requested return layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.mode_selected == i and self.return_kqv:
                return self.batchnorms[i](x).reshape(1, x.shape[1], -1)
            if i in self.multilayers:
                if self.add_norm:
                    x_batch = self.batchnorms[i](x)
                    outputs.append(x_batch)
                else:
                    outputs.append(x)

        if len(outputs) == 1 and self.return_cls:
            return outputs[0]

        return outputs[0] if len(outputs) == 1 else outputs
