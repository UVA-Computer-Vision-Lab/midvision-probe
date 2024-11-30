import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate


class BinaryHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="dpt",
        uncertainty_aware=False,
        hidden_dim=512,
        kernel_size=1,
        output_dim=2,
        pred_type="sigmoid",
    ):
        super().__init__()
        self.uncertainty_aware = uncertainty_aware
        self.kernel_size = kernel_size
        assert head_type in ["linear", "multiscale", "dpt"]
        name = f"snorm_{head_type}_k{kernel_size}"
        self.name = f"{name}_UA" if uncertainty_aware else name
        self.pred_type = pred_type
        if pred_type == "sigmoid":
            self.batch_norm = nn.BatchNorm2d(output_dim)
        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        feats = self.head(feats)
        if self.pred_type == "sigmoid":
            feats = self.batch_norm(feats)
            return torch.sigmoid(feats)
        elif self.pred_type == "tanh":
            return torch.tanh(feats)
        return feats


class TaskonomyHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="dpt",
        uncertainty_aware=False,
        hidden_dim=512,
        kernel_size=1,
        output_dim=1,
        pred_type="sigmoid",
    ):
        super().__init__()

        self.uncertainty_aware = uncertainty_aware
        self.kernel_size = kernel_size
        assert head_type in ["linear", "multiscale", "dpt"]
        name = f"snorm_{head_type}_k{kernel_size}"
        self.name = f"{name}_UA" if uncertainty_aware else name
        self.pred_type = pred_type
        if pred_type == "sigmoid":
            self.batch_norm = nn.BatchNorm2d(output_dim)
        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        feats = self.head(feats)
        if self.pred_type == "sigmoid":
            feats = self.batch_norm(feats)
            return torch.sigmoid(feats)
        elif self.pred_type == "tanh":
            return torch.tanh(feats)
        return feats


class SurfaceNormalHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        uncertainty_aware=False,
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.uncertainty_aware = uncertainty_aware
        output_dim = 4 if uncertainty_aware else 3

        self.kernel_size = kernel_size

        assert head_type in ["linear", "multiscale", "dpt"]
        name = f"snorm_{head_type}_k{kernel_size}"
        self.name = f"{name}_UA" if uncertainty_aware else name

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        return self.head(feats)


class DepthHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        min_depth=0.001,
        max_depth=10,
        prediction_type="sigdepth",
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.name = f"{prediction_type}_{head_type}_k{kernel_size}"

        if prediction_type == "bindepth":
            output_dim = 256
            self.predict = DepthBinPrediction(min_depth, max_depth, n_bins=output_dim)
        elif prediction_type == "sigdepth":
            output_dim = 1
            self.predict = DepthSigmoidPrediction(min_depth, max_depth)
        else:
            raise ValueError()

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        """Prediction each pixel."""
        feats = self.head(feats)
        depth = self.predict(feats)
        return depth


class DepthBinPrediction(nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=10,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
    ):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_strategy = norm_strategy
        self.bins_strategy = bins_strategy

    def forward(self, prob):
        if self.bins_strategy == "UD":
            bins = torch.linspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )
        elif self.bins_strategy == "SID":
            bins = torch.logspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )

        # following Adabins, default linear
        if self.norm_strategy == "linear":
            prob = torch.relu(prob)
            eps = 0.1
            prob = prob + eps
            prob = prob / prob.sum(dim=1, keepdim=True)
        elif self.norm_strategy == "softmax":
            prob = torch.softmax(prob, dim=1)
        elif self.norm_strategy == "sigmoid":
            prob = torch.sigmoid(prob)
            prob = prob / prob.sum(dim=1, keepdim=True)

        depth = torch.einsum("ikhw,k->ihw", [prob, bins])
        depth = depth.unsqueeze(dim=1)
        return depth


class DepthSigmoidPrediction(nn.Module):
    def __init__(self, min_depth=0.001, max_depth=10):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, pred):
        depth = pred.sigmoid()
        depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        return depth


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features,
        kernel_size=3,
        with_skip=True,
        upsample=False,
        is_transformer=False,
    ):
        """
        Init.

        Args:
            features (int): number of features.
            kernel_size (int): kernel size for the residual conv units (default 3).
            with_skip (bool): if True, includes a skip connection from the input to the output.
            upsample (bool): if True, upsamples the output by a factor of 2 using bilinear interpolation.
            is_transformer (bool): if True, uses the Transformer-based implementation. If False, uses the CNN-based implementation.
        """
        super().__init__()
        self.with_skip = with_skip
        self.upsample = upsample

        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(
                features, kernel_size, is_transformer=is_transformer
            )

        self.resConfUnit2 = ResidualConvUnit(
            features, kernel_size, is_transformer=is_transformer
        )
        self.is_transformer = is_transformer

    def forward(self, x, skip_x=None):
        if skip_x is not None and self.with_skip:
            assert skip_x.shape == x.shape, "Shape of skip_x must match x"
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)

        if not self.is_transformer:
            x = nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )

        return x


class ResidualConvUnit(nn.Module):
    def __init__(
        self, features, kernel_size=3, is_transformer=False, inplace_relu=True
    ):
        """
        Init.

        Args:
            features (int): number of features.
            kernel_size (int): kernel size for convolution layers (default 3).
            is_transformer (bool): if True, uses the Transformer-based implementation. If False, uses the CNN-based implementation.
            inplace_relu (bool): if True, uses in-place ReLU operations.
        """
        super().__init__()

        if is_transformer:
            assert (
                kernel_size % 2 == 1
            ), "Kernel size needs to be odd for transformer-based implementation"
            padding = kernel_size // 2
            self.conv = nn.Sequential(
                nn.Conv2d(features, features, kernel_size, padding=padding),
                nn.ReLU(inplace=inplace_relu),
                nn.Conv2d(features, features, kernel_size, padding=padding),
                nn.ReLU(inplace=inplace_relu),
            )
        else:
            self.conv1 = nn.Conv2d(
                features, features, kernel_size=3, stride=1, padding=1, bias=True
            )
            self.conv2 = nn.Conv2d(
                features, features, kernel_size=3, stride=1, padding=1, bias=True
            )
            self.relu = nn.ReLU(inplace=inplace_relu)

    def forward(self, x):
        if hasattr(self, "conv"):
            return self.conv(x) + x
        else:
            out = self.relu(x)
            out = self.conv1(out)
            out = self.relu(out)
            out = self.conv2(out)
            return out + x


class DPT(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=3):
        super().__init__()
        assert len(input_dims) == 4

        # Determine if we're using ResNet (CNN-based) or Transformer
        self.resnet = not isinstance(input_dims[0], int)

        # Initialize convolutional layers differently based on backbone type
        if self.resnet:
            self.conv_0 = nn.Conv2d(
                input_dims[0][0],
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.conv_1 = nn.Conv2d(
                input_dims[1][0],
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.conv_2 = nn.Conv2d(
                input_dims[2][0],
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.conv_3 = nn.Conv2d(
                input_dims[3][0],
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        else:
            self.conv_0 = nn.Conv2d(input_dims[0], hidden_dim, 1, padding=0)
            self.conv_1 = nn.Conv2d(input_dims[1], hidden_dim, 1, padding=0)
            self.conv_2 = nn.Conv2d(input_dims[2], hidden_dim, 1, padding=0)
            self.conv_3 = nn.Conv2d(input_dims[3], hidden_dim, 1, padding=0)

        # Initialize FeatureFusionBlock with resnet flag to determine behavior
        self.ref_0 = FeatureFusionBlock(
            hidden_dim, kernel_size, is_transformer=not self.resnet
        )
        self.ref_1 = FeatureFusionBlock(
            hidden_dim, kernel_size, is_transformer=not self.resnet
        )
        self.ref_2 = FeatureFusionBlock(
            hidden_dim, kernel_size, is_transformer=not self.resnet
        )
        self.ref_3 = FeatureFusionBlock(
            hidden_dim, kernel_size, is_transformer=not self.resnet, with_skip=False
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, feats):
        """Prediction each pixel."""
        assert len(feats) == 4
        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        # We no longer need to resize explicitly for ResNet features
        # Let the FeatureFusionBlocks handle the dimensions
        if not self.resnet:
            feats = [F.interpolate(x, scale_factor=2) for x in feats]

        out = self.ref_3(feats[3], None)
        out = self.ref_2(feats[2], out)
        out = self.ref_1(feats[1], out)
        out = self.ref_0(feats[0], out)

        if not self.resnet:
            out = F.interpolate(out, scale_factor=4)
        out = self.out_conv(out)
        out = F.interpolate(out, scale_factor=2)
        return out


def make_conv(input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
    if num_layers == 1:
        conv = nn.Conv2d(input_dim, output_dim, kernel_size)
    else:
        assert num_layers > 1
        modules = [nn.Conv2d(input_dim, hidden_dim, kernel_size), nn.ReLU(inplace=True)]
        for i in range(num_layers - 2):
            modules.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size))
            modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(hidden_dim, output_dim, kernel_size))
        conv = nn.Sequential(*modules)

    return conv


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super().__init__()
        if type(input_dim) is not int:
            input_dim = sum(input_dim)

        assert type(input_dim) is int
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding)

    def forward(self, feats):
        if type(feats) is list:
            feats = torch.cat(feats, dim=1)

        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv(feats)


class MultiscaleHead(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=1):
        super().__init__()

        self.convs = nn.ModuleList(
            [make_conv(in_d, None, hidden_dim, 1, kernel_size) for in_d in input_dims]
        )
        interm_dim = len(input_dims) * hidden_dim
        self.conv_mid = make_conv(interm_dim, hidden_dim, hidden_dim, 3, kernel_size)
        self.conv_out = make_conv(hidden_dim, hidden_dim, output_dim, 2, kernel_size)

    def forward(self, feats):
        num_feats = len(feats)
        feats = [self.convs[i](feats[i]) for i in range(num_feats)]

        h, w = feats[-1].shape[-2:]
        feats = [interpolate(feat, (h, w), mode="bilinear") for feat in feats]
        feats = torch.cat(feats, dim=1).relu()

        # upsample
        feats = interpolate(feats, scale_factor=2, mode="bilinear")
        feats = self.conv_mid(feats).relu()
        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv_out(feats)
