"""MobileNetV2 model define"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as P

__all__ = ['MobileNetV2Backbone', 'MobileNetV2Head', 'mobilenet_v2']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class GlobalAvgPooling(nn.Module):
    """
    Global avg pooling definition.
    """

    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = torch.mean

    def forward(self, x):
        return self.mean(x, (2, 3), keepdim=False)


class ConvBNReLU(nn.Module):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        in_channels = in_planes
        out_channels = out_planes
        if groups == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        else:
            out_channels = in_planes
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                             padding=padding, groups=in_channels)

        self.features = nn.Sequential(conv, nn.BatchNorm2d(out_planes), nn.ReLU6())

    def forward(self, x):
        output = self.features(x)
        return output


class InvertedResidual(nn.Module):
    """
    Mobilenetv2 residual block definition.
    """

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.add = torch.add

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            return self.add(identity, x)
        return x


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 architecture.
    """

    def __init__(self, width_mult=1., inverted_residual_setting=None, round_nearest=8,
                 input_channel=32, last_channel=1280):
        super(MobileNetV2Backbone, self).__init__()
        block = InvertedResidual
        # setting of inverted residual blocks
        self.cfgs = inverted_residual_setting
        if inverted_residual_setting is None:
            self.cfgs = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.out_channels = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.out_channels, kernel_size=1))
        # make it nn.ModuleList
        self.features = nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        """
        Initialize weights.
        """
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(torch.tensor(np.random.normal(0, np.sqrt(2. / n),
                                                                m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        torch.tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    torch.tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    torch.tensor(np.zeros(m.beta.data.shape, dtype="float32")))

    @property
    def get_features(self):
        return self.features


class MobileNetV2Head(nn.Module):
    """
    MobileNetV2 architecture.
    """

    def __init__(self, input_channel=1280, num_classes=10, has_dropout=False, activation="None"):
        super(MobileNetV2Head, self).__init__()
        # mobilenet head
        head = ([GlobalAvgPooling()] if not has_dropout else
                [GlobalAvgPooling(), nn.Dropout(0.2)])
        self.head = nn.Sequential(*head)
        self.Linear = nn.Linear(input_channel, num_classes, bias=True)
        self.need_activation = True
        if activation == "Sigmoid":
            self.activation = P.sigmoid
        elif activation == "Softmax":
            self.activation = P.softmax
        else:
            self.need_activation = False
        # self._initialize_weights()

    def forward(self, x):
        x = self.head(x)
        x = self.Linear(x)
        if self.need_activation:
            x = self.activation(x)
        return x

    def _initialize_weights(self):
        """
        Initialize weights.
        """

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.weight.set_data(torch.tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(torch.tensor(np.zeros(m.bias.data.shape, dtype="float32")))


class MobileNetV2Combine(nn.Module):
    """
    MobileNetV2Combine architecture.
    """

    def __init__(self, backbone, head):
        super(MobileNetV2Combine, self).__init__()
        self.backbone = backbone
        self.head = head

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

        self.in_shapes = {
            'INPUT': [1, 3, 224, 224],
            'backbone.features.0.features.0': [1, 3, 224, 224],
            'backbone.features.0.features.1': [1, 32, 112, 112],
            'backbone.features.0.features.2': [1, 32, 112, 112],
            'backbone.features.1.conv.0.features.0': [1, 32, 112, 112],
            'backbone.features.1.conv.0.features.1': [1, 32, 112, 112],
            'backbone.features.1.conv.0.features.2': [1, 32, 112, 112],
            'backbone.features.1.conv.1': [1, 32, 112, 112],
            'backbone.features.1.conv.2': [1, 16, 112, 112],
            'backbone.features.2.conv.0.features.0': [1, 16, 112, 112],
            'backbone.features.2.conv.0.features.1': [1, 96, 112, 112],
            'backbone.features.2.conv.0.features.2': [1, 96, 112, 112],
            'backbone.features.2.conv.1.features.0': [1, 96, 112, 112],
            'backbone.features.2.conv.1.features.1': [1, 96, 56, 56],
            'backbone.features.2.conv.1.features.2': [1, 96, 56, 56],
            'backbone.features.2.conv.2': [1, 96, 56, 56],
            'backbone.features.2.conv.3': [1, 24, 56, 56],
            'backbone.features.3.conv.0.features.0': [1, 24, 56, 56],
            'backbone.features.3.conv.0.features.1': [1, 144, 56, 56],
            'backbone.features.3.conv.0.features.2': [1, 144, 56, 56],
            'backbone.features.3.conv.1.features.0': [1, 144, 56, 56],
            'backbone.features.3.conv.1.features.1': [1, 144, 56, 56],
            'backbone.features.3.conv.1.features.2': [1, 144, 56, 56],
            'backbone.features.3.conv.2': [1, 144, 56, 56],
            'backbone.features.3.conv.3': [1, 24, 56, 56],
            'backbone.features.4.conv.0.features.0': [1, 24, 56, 56],
            'backbone.features.4.conv.0.features.1': [1, 144, 56, 56],
            'backbone.features.4.conv.0.features.2': [1, 144, 56, 56],
            'backbone.features.4.conv.1.features.0': [1, 144, 56, 56],
            'backbone.features.4.conv.1.features.1': [1, 144, 28, 28],
            'backbone.features.4.conv.1.features.2': [1, 144, 28, 28],
            'backbone.features.4.conv.2': [1, 144, 28, 28],
            'backbone.features.4.conv.3': [1, 32, 28, 28],
            'backbone.features.5.conv.0.features.0': [1, 32, 28, 28],
            'backbone.features.5.conv.0.features.1': [1, 192, 28, 28],
            'backbone.features.5.conv.0.features.2': [1, 192, 28, 28],
            'backbone.features.5.conv.1.features.0': [1, 192, 28, 28],
            'backbone.features.5.conv.1.features.1': [1, 192, 28, 28],
            'backbone.features.5.conv.1.features.2': [1, 192, 28, 28],
            'backbone.features.5.conv.2': [1, 192, 28, 28],
            'backbone.features.5.conv.3': [1, 32, 28, 28],
            'backbone.features.6.conv.0.features.0': [1, 32, 28, 28],
            'backbone.features.6.conv.0.features.1': [1, 192, 28, 28],
            'backbone.features.6.conv.0.features.2': [1, 192, 28, 28],
            'backbone.features.6.conv.1.features.0': [1, 192, 28, 28],
            'backbone.features.6.conv.1.features.1': [1, 192, 28, 28],
            'backbone.features.6.conv.1.features.2': [1, 192, 28, 28],
            'backbone.features.6.conv.2': [1, 192, 28, 28],
            'backbone.features.6.conv.3': [1, 32, 28, 28],
            'backbone.features.7.conv.0.features.0': [1, 32, 28, 28],
            'backbone.features.7.conv.0.features.1': [1, 192, 28, 28],
            'backbone.features.7.conv.0.features.2': [1, 192, 28, 28],
            'backbone.features.7.conv.1.features.0': [1, 192, 28, 28],
            'backbone.features.7.conv.1.features.1': [1, 192, 14, 14],
            'backbone.features.7.conv.1.features.2': [1, 192, 14, 14],
            'backbone.features.7.conv.2': [1, 192, 14, 14],
            'backbone.features.7.conv.3': [1, 64, 14, 14],
            'backbone.features.8.conv.0.features.0': [1, 64, 14, 14],
            'backbone.features.8.conv.0.features.1': [1, 384, 14, 14],
            'backbone.features.8.conv.0.features.2': [1, 384, 14, 14],
            'backbone.features.8.conv.1.features.0': [1, 384, 14, 14],
            'backbone.features.8.conv.1.features.1': [1, 384, 14, 14],
            'backbone.features.8.conv.1.features.2': [1, 384, 14, 14],
            'backbone.features.8.conv.2': [1, 384, 14, 14],
            'backbone.features.8.conv.3': [1, 64, 14, 14],
            'backbone.features.9.conv.0.features.0': [1, 64, 14, 14],
            'backbone.features.9.conv.0.features.1': [1, 384, 14, 14],
            'backbone.features.9.conv.0.features.2': [1, 384, 14, 14],
            'backbone.features.9.conv.1.features.0': [1, 384, 14, 14],
            'backbone.features.9.conv.1.features.1': [1, 384, 14, 14],
            'backbone.features.9.conv.1.features.2': [1, 384, 14, 14],
            'backbone.features.9.conv.2': [1, 384, 14, 14],
            'backbone.features.9.conv.3': [1, 64, 14, 14],
            'backbone.features.10.conv.0.features.0': [1, 64, 14, 14],
            'backbone.features.10.conv.0.features.1': [1, 384, 14, 14],
            'backbone.features.10.conv.0.features.2': [1, 384, 14, 14],
            'backbone.features.10.conv.1.features.0': [1, 384, 14, 14],
            'backbone.features.10.conv.1.features.1': [1, 384, 14, 14],
            'backbone.features.10.conv.1.features.2': [1, 384, 14, 14],
            'backbone.features.10.conv.2': [1, 384, 14, 14],
            'backbone.features.10.conv.3': [1, 64, 14, 14],
            'backbone.features.11.conv.0.features.0': [1, 64, 14, 14],
            'backbone.features.11.conv.0.features.1': [1, 384, 14, 14],
            'backbone.features.11.conv.0.features.2': [1, 384, 14, 14],
            'backbone.features.11.conv.1.features.0': [1, 384, 14, 14],
            'backbone.features.11.conv.1.features.1': [1, 384, 14, 14],
            'backbone.features.11.conv.1.features.2': [1, 384, 14, 14],
            'backbone.features.11.conv.2': [1, 384, 14, 14],
            'backbone.features.11.conv.3': [1, 96, 14, 14],
            'backbone.features.12.conv.0.features.0': [1, 96, 14, 14],
            'backbone.features.12.conv.0.features.1': [1, 576, 14, 14],
            'backbone.features.12.conv.0.features.2': [1, 576, 14, 14],
            'backbone.features.12.conv.1.features.0': [1, 576, 14, 14],
            'backbone.features.12.conv.1.features.1': [1, 576, 14, 14],
            'backbone.features.12.conv.1.features.2': [1, 576, 14, 14],
            'backbone.features.12.conv.2': [1, 576, 14, 14],
            'backbone.features.12.conv.3': [1, 96, 14, 14],
            'backbone.features.13.conv.0.features.0': [1, 96, 14, 14],
            'backbone.features.13.conv.0.features.1': [1, 576, 14, 14],
            'backbone.features.13.conv.0.features.2': [1, 576, 14, 14],
            'backbone.features.13.conv.1.features.0': [1, 576, 14, 14],
            'backbone.features.13.conv.1.features.1': [1, 576, 14, 14],
            'backbone.features.13.conv.1.features.2': [1, 576, 14, 14],
            'backbone.features.13.conv.2': [1, 576, 14, 14],
            'backbone.features.13.conv.3': [1, 96, 14, 14],
            'backbone.features.14.conv.0.features.0': [1, 96, 14, 14],
            'backbone.features.14.conv.0.features.1': [1, 576, 14, 14],
            'backbone.features.14.conv.0.features.2': [1, 576, 14, 14],
            'backbone.features.14.conv.1.features.0': [1, 576, 14, 14],
            'backbone.features.14.conv.1.features.1': [1, 576, 7, 7],
            'backbone.features.14.conv.1.features.2': [1, 576, 7, 7],
            'backbone.features.14.conv.2': [1, 576, 7, 7],
            'backbone.features.14.conv.3': [1, 160, 7, 7],
            'backbone.features.15.conv.0.features.0': [1, 160, 7, 7],
            'backbone.features.15.conv.0.features.1': [1, 960, 7, 7],
            'backbone.features.15.conv.0.features.2': [1, 960, 7, 7],
            'backbone.features.15.conv.1.features.0': [1, 960, 7, 7],
            'backbone.features.15.conv.1.features.1': [1, 960, 7, 7],
            'backbone.features.15.conv.1.features.2': [1, 960, 7, 7],
            'backbone.features.15.conv.2': [1, 960, 7, 7],
            'backbone.features.15.conv.3': [1, 160, 7, 7],
            'backbone.features.16.conv.0.features.0': [1, 160, 7, 7],
            'backbone.features.16.conv.0.features.1': [1, 960, 7, 7],
            'backbone.features.16.conv.0.features.2': [1, 960, 7, 7],
            'backbone.features.16.conv.1.features.0': [1, 960, 7, 7],
            'backbone.features.16.conv.1.features.1': [1, 960, 7, 7],
            'backbone.features.16.conv.1.features.2': [1, 960, 7, 7],
            'backbone.features.16.conv.2': [1, 960, 7, 7],
            'backbone.features.16.conv.3': [1, 160, 7, 7],
            'backbone.features.17.conv.0.features.0': [1, 160, 7, 7],
            'backbone.features.17.conv.0.features.1': [1, 960, 7, 7],
            'backbone.features.17.conv.0.features.2': [1, 960, 7, 7],
            'backbone.features.17.conv.1.features.0': [1, 960, 7, 7],
            'backbone.features.17.conv.1.features.1': [1, 960, 7, 7],
            'backbone.features.17.conv.1.features.2': [1, 960, 7, 7],
            'backbone.features.17.conv.2': [1, 960, 7, 7],
            'backbone.features.17.conv.3': [1, 320, 7, 7],
            'backbone.features.18.features.0': [1, 320, 7, 7],
            'backbone.features.18.features.1': [1, 1280, 7, 7],
            'backbone.features.18.features.2': [1, 1280, 7, 7],
            'head.head.0': [1, 1280, 7, 7],
            'head.dense': [1, 1280],
            'OUTPUT': [1, 10]}
        self.orders = {
            'backbone.features.0.features.0': ["INPUT", "backbone.features.0.features.1"],
            'backbone.features.0.features.1': ["backbone.features.0.features.0", "backbone.features.0.features.2"],
            'backbone.features.0.features.2': ["backbone.features.0.features.1",
                                               "backbone.features.1.conv.0.features.0"],
            'backbone.features.1.conv.0.features.0': ["backbone.features.0.features.2",
                                                      "backbone.features.1.conv.0.features.1"],
            'backbone.features.1.conv.0.features.1': ["backbone.features.1.conv.0.features.0",
                                                      "backbone.features.1.conv.0.features.2"],
            'backbone.features.1.conv.0.features.2': ["backbone.features.1.conv.0.features.1",
                                                      "backbone.features.1.conv.1"],
            'backbone.features.1.conv.1': ["backbone.features.1.conv.0.features.2", "backbone.features.1.conv.2"],
            'backbone.features.1.conv.2': ["backbone.features.1.conv.1", "backbone.features.2.conv.0.features.0"],
            'backbone.features.2.conv.0.features.0': ["backbone.features.1.conv.2",
                                                      "backbone.features.2.conv.0.features.1"],
            'backbone.features.2.conv.0.features.1': ["backbone.features.2.conv.0.features.0",
                                                      "backbone.features.2.conv.0.features.2"],
            'backbone.features.2.conv.0.features.2': ["backbone.features.2.conv.0.features.1",
                                                      "backbone.features.2.conv.1.features.0"],
            'backbone.features.2.conv.1.features.0': ["backbone.features.2.conv.0.features.2",
                                                      "backbone.features.2.conv.1.features.1"],
            'backbone.features.2.conv.1.features.1': ["backbone.features.2.conv.1.features.0",
                                                      "backbone.features.2.conv.1.features.2"],
            'backbone.features.2.conv.1.features.2': ["backbone.features.2.conv.1.features.1",
                                                      "backbone.features.2.conv.2"],
            'backbone.features.2.conv.2': ["backbone.features.2.conv.1.features.2", "backbone.features.2.conv.3"],
            'backbone.features.2.conv.3': ["backbone.features.2.conv.2", "backbone.features.3.conv.0.features.0"],
            'backbone.features.3.conv.0.features.0': ["backbone.features.2.conv.3",
                                                      "backbone.features.3.conv.0.features.1"],
            'backbone.features.3.conv.0.features.1': ["backbone.features.3.conv.0.features.0",
                                                      "backbone.features.3.conv.0.features.2"],
            'backbone.features.3.conv.0.features.2': ["backbone.features.3.conv.0.features.1",
                                                      "backbone.features.3.conv.1.features.0"],
            'backbone.features.3.conv.1.features.0': ["backbone.features.3.conv.0.features.2",
                                                      "backbone.features.3.conv.1.features.1"],
            'backbone.features.3.conv.1.features.1': ["backbone.features.3.conv.1.features.0",
                                                      "backbone.features.3.conv.1.features.2"],
            'backbone.features.3.conv.1.features.2': ["backbone.features.3.conv.1.features.1",
                                                      "backbone.features.3.conv.2"],
            'backbone.features.3.conv.2': ["backbone.features.3.conv.1.features.2", "backbone.features.3.conv.3"],
            'backbone.features.3.conv.3': ["backbone.features.3.conv.2", "backbone.features.4.conv.0.features.0"],
            'backbone.features.4.conv.0.features.0': ["backbone.features.3.conv.3",
                                                      "backbone.features.4.conv.0.features.1"],
            'backbone.features.4.conv.0.features.1': ["backbone.features.4.conv.0.features.0",
                                                      "backbone.features.4.conv.0.features.2"],
            'backbone.features.4.conv.0.features.2': ["backbone.features.4.conv.0.features.1",
                                                      "backbone.features.4.conv.1.features.0"],
            'backbone.features.4.conv.1.features.0': ["backbone.features.4.conv.0.features.2",
                                                      "backbone.features.4.conv.1.features.1"],
            'backbone.features.4.conv.1.features.1': ["backbone.features.4.conv.1.features.0",
                                                      "backbone.features.4.conv.1.features.2"],
            'backbone.features.4.conv.1.features.2': ["backbone.features.4.conv.1.features.1",
                                                      "backbone.features.4.conv.2"],
            'backbone.features.4.conv.2': ["backbone.features.4.conv.1.features.2", "backbone.features.4.conv.3"],
            'backbone.features.4.conv.3': ["backbone.features.4.conv.2", "backbone.features.5.conv.0.features.0"],
            'backbone.features.5.conv.0.features.0': ["backbone.features.4.conv.3",
                                                      "backbone.features.5.conv.0.features.1"],
            'backbone.features.5.conv.0.features.1': ["backbone.features.5.conv.0.features.0",
                                                      "backbone.features.5.conv.0.features.2"],
            'backbone.features.5.conv.0.features.2': ["backbone.features.5.conv.0.features.1",
                                                      "backbone.features.5.conv.1.features.0"],
            'backbone.features.5.conv.1.features.0': ["backbone.features.5.conv.0.features.2",
                                                      "backbone.features.5.conv.1.features.1"],
            'backbone.features.5.conv.1.features.1': ["backbone.features.5.conv.1.features.0",
                                                      "backbone.features.5.conv.1.features.2"],
            'backbone.features.5.conv.1.features.2': ["backbone.features.5.conv.1.features.1",
                                                      "backbone.features.5.conv.2"],
            'backbone.features.5.conv.2': ["backbone.features.5.conv.1.features.2", "backbone.features.5.conv.3"],
            'backbone.features.5.conv.3': ["backbone.features.5.conv.2", "backbone.features.6.conv.0.features.0"],
            'backbone.features.6.conv.0.features.0': ["backbone.features.5.conv.3",
                                                      "backbone.features.6.conv.0.features.1"],
            'backbone.features.6.conv.0.features.1': ["backbone.features.6.conv.0.features.0",
                                                      "backbone.features.6.conv.0.features.2"],
            'backbone.features.6.conv.0.features.2': ["backbone.features.6.conv.0.features.1",
                                                      "backbone.features.6.conv.1.features.0"],
            'backbone.features.6.conv.1.features.0': ["backbone.features.6.conv.0.features.2",
                                                      "backbone.features.6.conv.1.features.1"],
            'backbone.features.6.conv.1.features.1': ["backbone.features.6.conv.1.features.0",
                                                      "backbone.features.6.conv.1.features.2"],
            'backbone.features.6.conv.1.features.2': ["backbone.features.6.conv.1.features.1",
                                                      "backbone.features.6.conv.2"],
            'backbone.features.6.conv.2': ["backbone.features.6.conv.1.features.2", "backbone.features.6.conv.3"],
            'backbone.features.6.conv.3': ["backbone.features.6.conv.2", "backbone.features.7.conv.0.features.0"],
            'backbone.features.7.conv.0.features.0': ["backbone.features.6.conv.3",
                                                      "backbone.features.7.conv.0.features.1"],
            'backbone.features.7.conv.0.features.1': ["backbone.features.7.conv.0.features.0",
                                                      "backbone.features.7.conv.0.features.2"],
            'backbone.features.7.conv.0.features.2': ["backbone.features.7.conv.0.features.1",
                                                      "backbone.features.7.conv.1.features.0"],
            'backbone.features.7.conv.1.features.0': ["backbone.features.7.conv.0.features.2",
                                                      "backbone.features.7.conv.1.features.1"],
            'backbone.features.7.conv.1.features.1': ["backbone.features.7.conv.1.features.0",
                                                      "backbone.features.7.conv.1.features.2"],
            'backbone.features.7.conv.1.features.2': ["backbone.features.7.conv.1.features.1",
                                                      "backbone.features.7.conv.2"],
            'backbone.features.7.conv.2': ["backbone.features.7.conv.1.features.2", "backbone.features.7.conv.3"],
            'backbone.features.7.conv.3': ["backbone.features.7.conv.2", "backbone.features.8.conv.0.features.0"],
            'backbone.features.8.conv.0.features.0': ["backbone.features.7.conv.3",
                                                      "backbone.features.8.conv.0.features.1"],
            'backbone.features.8.conv.0.features.1': ["backbone.features.8.conv.0.features.0",
                                                      "backbone.features.8.conv.0.features.2"],
            'backbone.features.8.conv.0.features.2': ["backbone.features.8.conv.0.features.1",
                                                      "backbone.features.8.conv.1.features.0"],
            'backbone.features.8.conv.1.features.0': ["backbone.features.8.conv.0.features.2",
                                                      "backbone.features.8.conv.1.features.1"],
            'backbone.features.8.conv.1.features.1': ["backbone.features.8.conv.1.features.0",
                                                      "backbone.features.8.conv.1.features.2"],
            'backbone.features.8.conv.1.features.2': ["backbone.features.8.conv.1.features.1",
                                                      "backbone.features.8.conv.2"],
            'backbone.features.8.conv.2': ["backbone.features.8.conv.1.features.2", "backbone.features.8.conv.3"],
            'backbone.features.8.conv.3': ["backbone.features.8.conv.2", "backbone.features.9.conv.0.features.0"],
            'backbone.features.9.conv.0.features.0': ["backbone.features.8.conv.3",
                                                      "backbone.features.9.conv.0.features.1"],
            'backbone.features.9.conv.0.features.1': ["backbone.features.9.conv.0.features.0",
                                                      "backbone.features.9.conv.0.features.2"],
            'backbone.features.9.conv.0.features.2': ["backbone.features.9.conv.0.features.1",
                                                      "backbone.features.9.conv.1.features.0"],
            'backbone.features.9.conv.1.features.0': ["backbone.features.9.conv.0.features.2",
                                                      "backbone.features.9.conv.1.features.1"],
            'backbone.features.9.conv.1.features.1': ["backbone.features.9.conv.1.features.0",
                                                      "backbone.features.9.conv.1.features.2"],
            'backbone.features.9.conv.1.features.2': ["backbone.features.9.conv.1.features.1",
                                                      "backbone.features.9.conv.2"],
            'backbone.features.9.conv.2': ["backbone.features.9.conv.1.features.2", "backbone.features.9.conv.3"],
            'backbone.features.9.conv.3': ["backbone.features.9.conv.2", "backbone.features.10.conv.0.features.0"],
            'backbone.features.10.conv.0.features.0': ["backbone.features.9.conv.3",
                                                       "backbone.features.10.conv.0.features.1"],
            'backbone.features.10.conv.0.features.1': ["backbone.features.10.conv.0.features.0",
                                                       "backbone.features.10.conv.0.features.2"],
            'backbone.features.10.conv.0.features.2': ["backbone.features.10.conv.0.features.1",
                                                       "backbone.features.10.conv.1.features.0"],
            'backbone.features.10.conv.1.features.0': ["backbone.features.10.conv.0.features.2",
                                                       "backbone.features.10.conv.1.features.1"],
            'backbone.features.10.conv.1.features.1': ["backbone.features.10.conv.1.features.0",
                                                       "backbone.features.10.conv.1.features.2"],
            'backbone.features.10.conv.1.features.2': ["backbone.features.10.conv.1.features.1",
                                                       "backbone.features.10.conv.2"],
            'backbone.features.10.conv.2': ["backbone.features.10.conv.1.features.2", "backbone.features.10.conv.3"],
            'backbone.features.10.conv.3': ["backbone.features.10.conv.2", "backbone.features.11.conv.0.features.0"],
            'backbone.features.11.conv.0.features.0': ["backbone.features.10.conv.3",
                                                       "backbone.features.11.conv.0.features.1"],
            'backbone.features.11.conv.0.features.1': ["backbone.features.11.conv.0.features.0",
                                                       "backbone.features.11.conv.0.features.2"],
            'backbone.features.11.conv.0.features.2': ["backbone.features.11.conv.0.features.1",
                                                       "backbone.features.11.conv.1.features.0"],
            'backbone.features.11.conv.1.features.0': ["backbone.features.11.conv.0.features.2",
                                                       "backbone.features.11.conv.1.features.1"],
            'backbone.features.11.conv.1.features.1': ["backbone.features.11.conv.1.features.0",
                                                       "backbone.features.11.conv.1.features.2"],
            'backbone.features.11.conv.1.features.2': ["backbone.features.11.conv.1.features.1",
                                                       "backbone.features.11.conv.2"],
            'backbone.features.11.conv.2': ["backbone.features.11.conv.1.features.2", "backbone.features.11.conv.3"],
            'backbone.features.11.conv.3': ["backbone.features.11.conv.2", "backbone.features.12.conv.0.features.0"],
            'backbone.features.12.conv.0.features.0': ["backbone.features.11.conv.3",
                                                       "backbone.features.12.conv.0.features.1"],
            'backbone.features.12.conv.0.features.1': ["backbone.features.12.conv.0.features.0",
                                                       "backbone.features.12.conv.0.features.2"],
            'backbone.features.12.conv.0.features.2': ["backbone.features.12.conv.0.features.1",
                                                       "backbone.features.12.conv.1.features.0"],
            'backbone.features.12.conv.1.features.0': ["backbone.features.12.conv.0.features.2",
                                                       "backbone.features.12.conv.1.features.1"],
            'backbone.features.12.conv.1.features.1': ["backbone.features.12.conv.1.features.0",
                                                       "backbone.features.12.conv.1.features.2"],
            'backbone.features.12.conv.1.features.2': ["backbone.features.12.conv.1.features.1",
                                                       "backbone.features.12.conv.2"],
            'backbone.features.12.conv.2': ["backbone.features.12.conv.1.features.2", "backbone.features.12.conv.3"],
            'backbone.features.12.conv.3': ["backbone.features.12.conv.2", "backbone.features.13.conv.0.features.0"],
            'backbone.features.13.conv.0.features.0': ["backbone.features.12.conv.3",
                                                       "backbone.features.13.conv.0.features.1"],
            'backbone.features.13.conv.0.features.1': ["backbone.features.13.conv.0.features.0",
                                                       "backbone.features.13.conv.0.features.2"],
            'backbone.features.13.conv.0.features.2': ["backbone.features.13.conv.0.features.1",
                                                       "backbone.features.13.conv.1.features.0"],
            'backbone.features.13.conv.1.features.0': ["backbone.features.13.conv.0.features.2",
                                                       "backbone.features.13.conv.1.features.1"],
            'backbone.features.13.conv.1.features.1': ["backbone.features.13.conv.1.features.0",
                                                       "backbone.features.13.conv.1.features.2"],
            'backbone.features.13.conv.1.features.2': ["backbone.features.13.conv.1.features.1",
                                                       "backbone.features.13.conv.2"],
            'backbone.features.13.conv.2': ["backbone.features.13.conv.1.features.2", "backbone.features.13.conv.3"],
            'backbone.features.13.conv.3': ["backbone.features.13.conv.2", "backbone.features.14.conv.0.features.0"],
            'backbone.features.14.conv.0.features.0': ["backbone.features.13.conv.3",
                                                       "backbone.features.14.conv.0.features.1"],
            'backbone.features.14.conv.0.features.1': ["backbone.features.14.conv.0.features.0",
                                                       "backbone.features.14.conv.0.features.2"],
            'backbone.features.14.conv.0.features.2': ["backbone.features.14.conv.0.features.1",
                                                       "backbone.features.14.conv.1.features.0"],
            'backbone.features.14.conv.1.features.0': ["backbone.features.14.conv.0.features.2",
                                                       "backbone.features.14.conv.1.features.1"],
            'backbone.features.14.conv.1.features.1': ["backbone.features.14.conv.1.features.0",
                                                       "backbone.features.14.conv.1.features.2"],
            'backbone.features.14.conv.1.features.2': ["backbone.features.14.conv.1.features.1",
                                                       "backbone.features.14.conv.2"],
            'backbone.features.14.conv.2': ["backbone.features.14.conv.1.features.2", "backbone.features.14.conv.3"],
            'backbone.features.14.conv.3': ["backbone.features.14.conv.2", "backbone.features.15.conv.0.features.0"],
            'backbone.features.15.conv.0.features.0': ["backbone.features.14.conv.3",
                                                       "backbone.features.15.conv.0.features.1"],
            'backbone.features.15.conv.0.features.1': ["backbone.features.15.conv.0.features.0",
                                                       "backbone.features.15.conv.0.features.2"],
            'backbone.features.15.conv.0.features.2': ["backbone.features.15.conv.0.features.1",
                                                       "backbone.features.15.conv.1.features.0"],
            'backbone.features.15.conv.1.features.0': ["backbone.features.15.conv.0.features.2",
                                                       "backbone.features.15.conv.1.features.1"],
            'backbone.features.15.conv.1.features.1': ["backbone.features.15.conv.1.features.0",
                                                       "backbone.features.15.conv.1.features.2"],
            'backbone.features.15.conv.1.features.2': ["backbone.features.15.conv.1.features.1",
                                                       "backbone.features.15.conv.2"],
            'backbone.features.15.conv.2': ["backbone.features.15.conv.1.features.2", "backbone.features.15.conv.3"],
            'backbone.features.15.conv.3': ["backbone.features.15.conv.2", "backbone.features.16.conv.0.features.0"],
            'backbone.features.16.conv.0.features.0': ["backbone.features.15.conv.3",
                                                       "backbone.features.16.conv.0.features.1"],
            'backbone.features.16.conv.0.features.1': ["backbone.features.16.conv.0.features.0",
                                                       "backbone.features.16.conv.0.features.2"],
            'backbone.features.16.conv.0.features.2': ["backbone.features.16.conv.0.features.1",
                                                       "backbone.features.16.conv.1.features.0"],
            'backbone.features.16.conv.1.features.0': ["backbone.features.16.conv.0.features.2",
                                                       "backbone.features.16.conv.1.features.1"],
            'backbone.features.16.conv.1.features.1': ["backbone.features.16.conv.1.features.0",
                                                       "backbone.features.16.conv.1.features.2"],
            'backbone.features.16.conv.1.features.2': ["backbone.features.16.conv.1.features.1",
                                                       "backbone.features.16.conv.2"],
            'backbone.features.16.conv.2': ["backbone.features.16.conv.1.features.2", "backbone.features.16.conv.3"],
            'backbone.features.16.conv.3': ["backbone.features.16.conv.2", "backbone.features.17.conv.0.features.0"],
            'backbone.features.17.conv.0.features.0': ["backbone.features.16.conv.3",
                                                       "backbone.features.17.conv.0.features.1"],
            'backbone.features.17.conv.0.features.1': ["backbone.features.17.conv.0.features.0",
                                                       "backbone.features.17.conv.0.features.2"],
            'backbone.features.17.conv.0.features.2': ["backbone.features.17.conv.0.features.1",
                                                       "backbone.features.17.conv.1.features.0"],
            'backbone.features.17.conv.1.features.0': ["backbone.features.17.conv.0.features.2",
                                                       "backbone.features.17.conv.1.features.1"],
            'backbone.features.17.conv.1.features.1': ["backbone.features.17.conv.1.features.0",
                                                       "backbone.features.17.conv.1.features.2"],
            'backbone.features.17.conv.1.features.2': ["backbone.features.17.conv.1.features.1",
                                                       "backbone.features.17.conv.2"],
            'backbone.features.17.conv.2': ["backbone.features.17.conv.1.features.2", "backbone.features.17.conv.3"],
            'backbone.features.17.conv.3': ["backbone.features.17.conv.2", "backbone.features.18.features.0"],
            'backbone.features.18.features.0': ["backbone.features.17.conv.3", "backbone.features.18.features.1"],
            'backbone.features.18.features.1': ["backbone.features.18.features.0", "backbone.features.18.features.2"],
            'backbone.features.18.features.2': ["backbone.features.18.features.1", "head.head.0"],
            'head.head.0': ["backbone.features.18.features.2", "head.dense"],
            'head.dense': ["head.head.0", "OUTPUT"]}
        self.out_shapes = {
            "INPUT": [-1, 3, 224, 224],
            'backbone.features.0.features.0': [-1, 32, 112, 112],
            'backbone.features.0.features.1': [-1, 32, 112, 112],
            'backbone.features.0.features.2': [-1, 32, 112, 112],
            'backbone.features.1.conv.0.features.0': [-1, 32, 112, 112],
            'backbone.features.1.conv.0.features.1': [-1, 32, 112, 112],
            'backbone.features.1.conv.0.features.2': [-1, 32, 112, 112],
            'backbone.features.1.conv.1': [-1, 16, 112, 112],
            'backbone.features.1.conv.2': [-1, 16, 112, 112],
            'backbone.features.2.conv.0.features.0': [-1, 96, 112, 112],
            'backbone.features.2.conv.0.features.1': [-1, 96, 112, 112],
            'backbone.features.2.conv.0.features.2': [-1, 96, 112, 112],
            'backbone.features.2.conv.1.features.0': [-1, 96, 56, 56],
            'backbone.features.2.conv.1.features.1': [-1, 96, 56, 56],
            'backbone.features.2.conv.1.features.2': [-1, 96, 56, 56],
            'backbone.features.2.conv.2': [-1, 24, 56, 56],
            'backbone.features.2.conv.3': [-1, 24, 56, 56],
            'backbone.features.3.conv.0.features.0': [-1, 144, 56, 56],
            'backbone.features.3.conv.0.features.1': [-1, 144, 56, 56],
            'backbone.features.3.conv.0.features.2': [-1, 144, 56, 56],
            'backbone.features.3.conv.1.features.0': [-1, 144, 56, 56],
            'backbone.features.3.conv.1.features.1': [-1, 144, 56, 56],
            'backbone.features.3.conv.1.features.2': [-1, 144, 56, 56],
            'backbone.features.3.conv.2': [-1, 24, 56, 56],
            'backbone.features.3.conv.3': [-1, 24, 56, 56],
            'backbone.features.4.conv.0.features.0': [-1, 144, 56, 56],
            'backbone.features.4.conv.0.features.1': [-1, 144, 56, 56],
            'backbone.features.4.conv.0.features.2': [-1, 144, 56, 56],
            'backbone.features.4.conv.1.features.0': [-1, 144, 28, 28],
            'backbone.features.4.conv.1.features.1': [-1, 144, 28, 28],
            'backbone.features.4.conv.1.features.2': [-1, 144, 28, 28],
            'backbone.features.4.conv.2': [-1, 32, 28, 28],
            'backbone.features.4.conv.3': [-1, 32, 28, 28],
            'backbone.features.5.conv.0.features.0': [-1, 192, 28, 28],
            'backbone.features.5.conv.0.features.1': [-1, 192, 28, 28],
            'backbone.features.5.conv.0.features.2': [-1, 192, 28, 28],
            'backbone.features.5.conv.1.features.0': [-1, 192, 28, 28],
            'backbone.features.5.conv.1.features.1': [-1, 192, 28, 28],
            'backbone.features.5.conv.1.features.2': [-1, 192, 28, 28],
            'backbone.features.5.conv.2': [-1, 32, 28, 28],
            'backbone.features.5.conv.3': [-1, 32, 28, 28],
            'backbone.features.6.conv.0.features.0': [-1, 192, 28, 28],
            'backbone.features.6.conv.0.features.1': [-1, 192, 28, 28],
            'backbone.features.6.conv.0.features.2': [-1, 192, 28, 28],
            'backbone.features.6.conv.1.features.0': [-1, 192, 28, 28],
            'backbone.features.6.conv.1.features.1': [-1, 192, 28, 28],
            'backbone.features.6.conv.1.features.2': [-1, 192, 28, 28],
            'backbone.features.6.conv.2': [-1, 32, 28, 28],
            'backbone.features.6.conv.3': [-1, 32, 28, 28],
            'backbone.features.7.conv.0.features.0': [-1, 192, 28, 28],
            'backbone.features.7.conv.0.features.1': [-1, 192, 28, 28],
            'backbone.features.7.conv.0.features.2': [-1, 192, 28, 28],
            'backbone.features.7.conv.1.features.0': [-1, 192, 14, 14],
            'backbone.features.7.conv.1.features.1': [-1, 192, 14, 14],
            'backbone.features.7.conv.1.features.2': [-1, 192, 14, 14],
            'backbone.features.7.conv.2': [-1, 64, 14, 14],
            'backbone.features.7.conv.3': [-1, 64, 14, 14],
            'backbone.features.8.conv.0.features.0': [-1, 384, 14, 14],
            'backbone.features.8.conv.0.features.1': [-1, 384, 14, 14],
            'backbone.features.8.conv.0.features.2': [-1, 384, 14, 14],
            'backbone.features.8.conv.1.features.0': [-1, 384, 14, 14],
            'backbone.features.8.conv.1.features.1': [-1, 384, 14, 14],
            'backbone.features.8.conv.1.features.2': [-1, 384, 14, 14],
            'backbone.features.8.conv.2': [-1, 64, 14, 14],
            'backbone.features.8.conv.3': [-1, 64, 14, 14],
            'backbone.features.9.conv.0.features.0': [-1, 384, 14, 14],
            'backbone.features.9.conv.0.features.1': [-1, 384, 14, 14],
            'backbone.features.9.conv.0.features.2': [-1, 384, 14, 14],
            'backbone.features.9.conv.1.features.0': [-1, 384, 14, 14],
            'backbone.features.9.conv.1.features.1': [-1, 384, 14, 14],
            'backbone.features.9.conv.1.features.2': [-1, 384, 14, 14],
            'backbone.features.9.conv.2': [-1, 64, 14, 14],
            'backbone.features.9.conv.3': [-1, 64, 14, 14],
            'backbone.features.10.conv.0.features.0': [-1, 384, 14, 14],
            'backbone.features.10.conv.0.features.1': [-1, 384, 14, 14],
            'backbone.features.10.conv.0.features.2': [-1, 384, 14, 14],
            'backbone.features.10.conv.1.features.0': [-1, 384, 14, 14],
            'backbone.features.10.conv.1.features.1': [-1, 384, 14, 14],
            'backbone.features.10.conv.1.features.2': [-1, 384, 14, 14],
            'backbone.features.10.conv.2': [-1, 64, 14, 14],
            'backbone.features.10.conv.3': [-1, 64, 14, 14],
            'backbone.features.11.conv.0.features.0': [-1, 384, 14, 14],
            'backbone.features.11.conv.0.features.1': [-1, 384, 14, 14],
            'backbone.features.11.conv.0.features.2': [-1, 384, 14, 14],
            'backbone.features.11.conv.1.features.0': [-1, 384, 14, 14],
            'backbone.features.11.conv.1.features.1': [-1, 384, 14, 14],
            'backbone.features.11.conv.1.features.2': [-1, 384, 14, 14],
            'backbone.features.11.conv.2': [-1, 96, 14, 14],
            'backbone.features.11.conv.3': [-1, 96, 14, 14],
            'backbone.features.12.conv.0.features.0': [-1, 576, 14, 14],
            'backbone.features.12.conv.0.features.1': [-1, 576, 14, 14],
            'backbone.features.12.conv.0.features.2': [-1, 576, 14, 14],
            'backbone.features.12.conv.1.features.0': [-1, 576, 14, 14],
            'backbone.features.12.conv.1.features.1': [-1, 576, 14, 14],
            'backbone.features.12.conv.1.features.2': [-1, 576, 14, 14],
            'backbone.features.12.conv.2': [-1, 96, 14, 14],
            'backbone.features.12.conv.3': [-1, 96, 14, 14],
            'backbone.features.13.conv.0.features.0': [-1, 576, 14, 14],
            'backbone.features.13.conv.0.features.1': [-1, 576, 14, 14],
            'backbone.features.13.conv.0.features.2': [-1, 576, 14, 14],
            'backbone.features.13.conv.1.features.0': [-1, 576, 14, 14],
            'backbone.features.13.conv.1.features.1': [-1, 576, 14, 14],
            'backbone.features.13.conv.1.features.2': [-1, 576, 14, 14],
            'backbone.features.13.conv.2': [-1, 96, 14, 14],
            'backbone.features.13.conv.3': [-1, 96, 14, 14],
            'backbone.features.14.conv.0.features.0': [-1, 576, 14, 14],
            'backbone.features.14.conv.0.features.1': [-1, 576, 14, 14],
            'backbone.features.14.conv.0.features.2': [-1, 576, 14, 14],
            'backbone.features.14.conv.1.features.0': [-1, 576, 7, 7],
            'backbone.features.14.conv.1.features.1': [-1, 576, 7, 7],
            'backbone.features.14.conv.1.features.2': [-1, 576, 7, 7],
            'backbone.features.14.conv.2': [-1, 160, 7, 7],
            'backbone.features.14.conv.3': [-1, 160, 7, 7],
            'backbone.features.15.conv.0.features.0': [-1, 960, 7, 7],
            'backbone.features.15.conv.0.features.1': [-1, 960, 7, 7],
            'backbone.features.15.conv.0.features.2': [-1, 960, 7, 7],
            'backbone.features.15.conv.1.features.0': [-1, 960, 7, 7],
            'backbone.features.15.conv.1.features.1': [-1, 960, 7, 7],
            'backbone.features.15.conv.1.features.2': [-1, 960, 7, 7],
            'backbone.features.15.conv.2': [-1, 160, 7, 7],
            'backbone.features.15.conv.3': [-1, 160, 7, 7],
            'backbone.features.16.conv.0.features.0': [-1, 960, 7, 7],
            'backbone.features.16.conv.0.features.1': [-1, 960, 7, 7],
            'backbone.features.16.conv.0.features.2': [-1, 960, 7, 7],
            'backbone.features.16.conv.1.features.0': [-1, 960, 7, 7],
            'backbone.features.16.conv.1.features.1': [-1, 960, 7, 7],
            'backbone.features.16.conv.1.features.2': [-1, 960, 7, 7],
            'backbone.features.16.conv.2': [-1, 160, 7, 7],
            'backbone.features.16.conv.3': [-1, 160, 7, 7],
            'backbone.features.17.conv.0.features.0': [-1, 960, 7, 7],
            'backbone.features.17.conv.0.features.1': [-1, 960, 7, 7],
            'backbone.features.17.conv.0.features.2': [-1, 960, 7, 7],
            'backbone.features.17.conv.1.features.0': [-1, 960, 7, 7],
            'backbone.features.17.conv.1.features.1': [-1, 960, 7, 7],
            'backbone.features.17.conv.1.features.2': [-1, 960, 7, 7],
            'backbone.features.17.conv.2': [-1, 320, 7, 7],
            'backbone.features.17.conv.3': [-1, 320, 7, 7],
            'backbone.features.18.features.0': [-1, 1280, 7, 7],
            'backbone.features.18.features.1': [-1, 1280, 7, 7],
            'backbone.features.18.features.2': [-1, 1280, 7, 7],
            'head.head.0': [-1, 1280],
            'head.dense': [-1, 10],
            "OUTPUT": [-1, 10]
        }
        self.layer_names = {
            "backbone": self.backbone,
            "backbone.features": self.backbone.features,
            "backbone.features.0": self.backbone.features[0],
            "backbone.features.0.features": self.backbone.features[0].features,
            "backbone.features.0.features.0": self.backbone.features[0].features[0],
            "backbone.features.0.features.1": self.backbone.features[0].features[1],
            "backbone.features.0.features.2": self.backbone.features[0].features[2],
            "backbone.features.1": self.backbone.features[1],
            "backbone.features.1.conv": self.backbone.features[1].conv,
            "backbone.features.1.conv.0": self.backbone.features[1].conv[0],
            "backbone.features.1.conv.0.features": self.backbone.features[1].conv[0].features,
            "backbone.features.1.conv.0.features.0": self.backbone.features[1].conv[0].features[0],
            "backbone.features.1.conv.0.features.1": self.backbone.features[1].conv[0].features[1],
            "backbone.features.1.conv.0.features.2": self.backbone.features[1].conv[0].features[2],
            "backbone.features.1.conv.1": self.backbone.features[1].conv[1],
            "backbone.features.1.conv.2": self.backbone.features[1].conv[2],
            "backbone.features.2": self.backbone.features[2],
            "backbone.features.2.conv": self.backbone.features[2].conv,
            "backbone.features.2.conv.0": self.backbone.features[2].conv[0],
            "backbone.features.2.conv.0.features": self.backbone.features[2].conv[0].features,
            "backbone.features.2.conv.0.features.0": self.backbone.features[2].conv[0].features[0],
            "backbone.features.2.conv.0.features.1": self.backbone.features[2].conv[0].features[1],
            "backbone.features.2.conv.0.features.2": self.backbone.features[2].conv[0].features[2],
            "backbone.features.2.conv.1": self.backbone.features[2].conv[1],
            "backbone.features.2.conv.1.features": self.backbone.features[2].conv[1].features,
            "backbone.features.2.conv.1.features.0": self.backbone.features[2].conv[1].features[0],
            "backbone.features.2.conv.1.features.1": self.backbone.features[2].conv[1].features[1],
            "backbone.features.2.conv.1.features.2": self.backbone.features[2].conv[1].features[2],
            "backbone.features.2.conv.2": self.backbone.features[2].conv[2],
            "backbone.features.2.conv.3": self.backbone.features[2].conv[3],
            "backbone.features.3": self.backbone.features[3],
            "backbone.features.3.conv": self.backbone.features[3].conv,
            "backbone.features.3.conv.0": self.backbone.features[3].conv[0],
            "backbone.features.3.conv.0.features": self.backbone.features[3].conv[0].features,
            "backbone.features.3.conv.0.features.0": self.backbone.features[3].conv[0].features[0],
            "backbone.features.3.conv.0.features.1": self.backbone.features[3].conv[0].features[1],
            "backbone.features.3.conv.0.features.2": self.backbone.features[3].conv[0].features[2],
            "backbone.features.3.conv.1": self.backbone.features[3].conv[1],
            "backbone.features.3.conv.1.features": self.backbone.features[3].conv[1].features,
            "backbone.features.3.conv.1.features.0": self.backbone.features[3].conv[1].features[0],
            "backbone.features.3.conv.1.features.1": self.backbone.features[3].conv[1].features[1],
            "backbone.features.3.conv.1.features.2": self.backbone.features[3].conv[1].features[2],
            "backbone.features.3.conv.2": self.backbone.features[3].conv[2],
            "backbone.features.3.conv.3": self.backbone.features[3].conv[3],
            "backbone.features.4": self.backbone.features[4],
            "backbone.features.4.conv": self.backbone.features[4].conv,
            "backbone.features.4.conv.0": self.backbone.features[4].conv[0],
            "backbone.features.4.conv.0.features": self.backbone.features[4].conv[0].features,
            "backbone.features.4.conv.0.features.0": self.backbone.features[4].conv[0].features[0],
            "backbone.features.4.conv.0.features.1": self.backbone.features[4].conv[0].features[1],
            "backbone.features.4.conv.0.features.2": self.backbone.features[4].conv[0].features[2],
            "backbone.features.4.conv.1": self.backbone.features[4].conv[1],
            "backbone.features.4.conv.1.features": self.backbone.features[4].conv[1].features,
            "backbone.features.4.conv.1.features.0": self.backbone.features[4].conv[1].features[0],
            "backbone.features.4.conv.1.features.1": self.backbone.features[4].conv[1].features[1],
            "backbone.features.4.conv.1.features.2": self.backbone.features[4].conv[1].features[2],
            "backbone.features.4.conv.2": self.backbone.features[4].conv[2],
            "backbone.features.4.conv.3": self.backbone.features[4].conv[3],
            "backbone.features.5": self.backbone.features[5],
            "backbone.features.5.conv": self.backbone.features[5].conv,
            "backbone.features.5.conv.0": self.backbone.features[5].conv[0],
            "backbone.features.5.conv.0.features": self.backbone.features[5].conv[0].features,
            "backbone.features.5.conv.0.features.0": self.backbone.features[5].conv[0].features[0],
            "backbone.features.5.conv.0.features.1": self.backbone.features[5].conv[0].features[1],
            "backbone.features.5.conv.0.features.2": self.backbone.features[5].conv[0].features[2],
            "backbone.features.5.conv.1": self.backbone.features[5].conv[1],
            "backbone.features.5.conv.1.features": self.backbone.features[5].conv[1].features,
            "backbone.features.5.conv.1.features.0": self.backbone.features[5].conv[1].features[0],
            "backbone.features.5.conv.1.features.1": self.backbone.features[5].conv[1].features[1],
            "backbone.features.5.conv.1.features.2": self.backbone.features[5].conv[1].features[2],
            "backbone.features.5.conv.2": self.backbone.features[5].conv[2],
            "backbone.features.5.conv.3": self.backbone.features[5].conv[3],
            "backbone.features.6": self.backbone.features[6],
            "backbone.features.6.conv": self.backbone.features[6].conv,
            "backbone.features.6.conv.0": self.backbone.features[6].conv[0],
            "backbone.features.6.conv.0.features": self.backbone.features[6].conv[0].features,
            "backbone.features.6.conv.0.features.0": self.backbone.features[6].conv[0].features[0],
            "backbone.features.6.conv.0.features.1": self.backbone.features[6].conv[0].features[1],
            "backbone.features.6.conv.0.features.2": self.backbone.features[6].conv[0].features[2],
            "backbone.features.6.conv.1": self.backbone.features[6].conv[1],
            "backbone.features.6.conv.1.features": self.backbone.features[6].conv[1].features,
            "backbone.features.6.conv.1.features.0": self.backbone.features[6].conv[1].features[0],
            "backbone.features.6.conv.1.features.1": self.backbone.features[6].conv[1].features[1],
            "backbone.features.6.conv.1.features.2": self.backbone.features[6].conv[1].features[2],
            "backbone.features.6.conv.2": self.backbone.features[6].conv[2],
            "backbone.features.6.conv.3": self.backbone.features[6].conv[3],
            "backbone.features.7": self.backbone.features[7],
            "backbone.features.7.conv": self.backbone.features[7].conv,
            "backbone.features.7.conv.0": self.backbone.features[7].conv[0],
            "backbone.features.7.conv.0.features": self.backbone.features[7].conv[0].features,
            "backbone.features.7.conv.0.features.0": self.backbone.features[7].conv[0].features[0],
            "backbone.features.7.conv.0.features.1": self.backbone.features[7].conv[0].features[1],
            "backbone.features.7.conv.0.features.2": self.backbone.features[7].conv[0].features[2],
            "backbone.features.7.conv.1": self.backbone.features[7].conv[1],
            "backbone.features.7.conv.1.features": self.backbone.features[7].conv[1].features,
            "backbone.features.7.conv.1.features.0": self.backbone.features[7].conv[1].features[0],
            "backbone.features.7.conv.1.features.1": self.backbone.features[7].conv[1].features[1],
            "backbone.features.7.conv.1.features.2": self.backbone.features[7].conv[1].features[2],
            "backbone.features.7.conv.2": self.backbone.features[7].conv[2],
            "backbone.features.7.conv.3": self.backbone.features[7].conv[3],
            "backbone.features.8": self.backbone.features[8],
            "backbone.features.8.conv": self.backbone.features[8].conv,
            "backbone.features.8.conv.0": self.backbone.features[8].conv[0],
            "backbone.features.8.conv.0.features": self.backbone.features[8].conv[0].features,
            "backbone.features.8.conv.0.features.0": self.backbone.features[8].conv[0].features[0],
            "backbone.features.8.conv.0.features.1": self.backbone.features[8].conv[0].features[1],
            "backbone.features.8.conv.0.features.2": self.backbone.features[8].conv[0].features[2],
            "backbone.features.8.conv.1": self.backbone.features[8].conv[1],
            "backbone.features.8.conv.1.features": self.backbone.features[8].conv[1].features,
            "backbone.features.8.conv.1.features.0": self.backbone.features[8].conv[1].features[0],
            "backbone.features.8.conv.1.features.1": self.backbone.features[8].conv[1].features[1],
            "backbone.features.8.conv.1.features.2": self.backbone.features[8].conv[1].features[2],
            "backbone.features.8.conv.2": self.backbone.features[8].conv[2],
            "backbone.features.8.conv.3": self.backbone.features[8].conv[3],
            "backbone.features.9": self.backbone.features[9],
            "backbone.features.9.conv": self.backbone.features[9].conv,
            "backbone.features.9.conv.0": self.backbone.features[9].conv[0],
            "backbone.features.9.conv.0.features": self.backbone.features[9].conv[0].features,
            "backbone.features.9.conv.0.features.0": self.backbone.features[9].conv[0].features[0],
            "backbone.features.9.conv.0.features.1": self.backbone.features[9].conv[0].features[1],
            "backbone.features.9.conv.0.features.2": self.backbone.features[9].conv[0].features[2],
            "backbone.features.9.conv.1": self.backbone.features[9].conv[1],
            "backbone.features.9.conv.1.features": self.backbone.features[9].conv[1].features,
            "backbone.features.9.conv.1.features.0": self.backbone.features[9].conv[1].features[0],
            "backbone.features.9.conv.1.features.1": self.backbone.features[9].conv[1].features[1],
            "backbone.features.9.conv.1.features.2": self.backbone.features[9].conv[1].features[2],
            "backbone.features.9.conv.2": self.backbone.features[9].conv[2],
            "backbone.features.9.conv.3": self.backbone.features[9].conv[3],
            "backbone.features.10": self.backbone.features[10],
            "backbone.features.10.conv": self.backbone.features[10].conv,
            "backbone.features.10.conv.0": self.backbone.features[10].conv[0],
            "backbone.features.10.conv.0.features": self.backbone.features[10].conv[0].features,
            "backbone.features.10.conv.0.features.0": self.backbone.features[10].conv[0].features[0],
            "backbone.features.10.conv.0.features.1": self.backbone.features[10].conv[0].features[1],
            "backbone.features.10.conv.0.features.2": self.backbone.features[10].conv[0].features[2],
            "backbone.features.10.conv.1": self.backbone.features[10].conv[1],
            "backbone.features.10.conv.1.features": self.backbone.features[10].conv[1].features,
            "backbone.features.10.conv.1.features.0": self.backbone.features[10].conv[1].features[0],
            "backbone.features.10.conv.1.features.1": self.backbone.features[10].conv[1].features[1],
            "backbone.features.10.conv.1.features.2": self.backbone.features[10].conv[1].features[2],
            "backbone.features.10.conv.2": self.backbone.features[10].conv[2],
            "backbone.features.10.conv.3": self.backbone.features[10].conv[3],
            "backbone.features.11": self.backbone.features[11],
            "backbone.features.11.conv": self.backbone.features[11].conv,
            "backbone.features.11.conv.0": self.backbone.features[11].conv[0],
            "backbone.features.11.conv.0.features": self.backbone.features[11].conv[0].features,
            "backbone.features.11.conv.0.features.0": self.backbone.features[11].conv[0].features[0],
            "backbone.features.11.conv.0.features.1": self.backbone.features[11].conv[0].features[1],
            "backbone.features.11.conv.0.features.2": self.backbone.features[11].conv[0].features[2],
            "backbone.features.11.conv.1": self.backbone.features[11].conv[1],
            "backbone.features.11.conv.1.features": self.backbone.features[11].conv[1].features,
            "backbone.features.11.conv.1.features.0": self.backbone.features[11].conv[1].features[0],
            "backbone.features.11.conv.1.features.1": self.backbone.features[11].conv[1].features[1],
            "backbone.features.11.conv.1.features.2": self.backbone.features[11].conv[1].features[2],
            "backbone.features.11.conv.2": self.backbone.features[11].conv[2],
            "backbone.features.11.conv.3": self.backbone.features[11].conv[3],
            "backbone.features.12": self.backbone.features[12],
            "backbone.features.12.conv": self.backbone.features[12].conv,
            "backbone.features.12.conv.0": self.backbone.features[12].conv[0],
            "backbone.features.12.conv.0.features": self.backbone.features[12].conv[0].features,
            "backbone.features.12.conv.0.features.0": self.backbone.features[12].conv[0].features[0],
            "backbone.features.12.conv.0.features.1": self.backbone.features[12].conv[0].features[1],
            "backbone.features.12.conv.0.features.2": self.backbone.features[12].conv[0].features[2],
            "backbone.features.12.conv.1": self.backbone.features[12].conv[1],
            "backbone.features.12.conv.1.features": self.backbone.features[12].conv[1].features,
            "backbone.features.12.conv.1.features.0": self.backbone.features[12].conv[1].features[0],
            "backbone.features.12.conv.1.features.1": self.backbone.features[12].conv[1].features[1],
            "backbone.features.12.conv.1.features.2": self.backbone.features[12].conv[1].features[2],
            "backbone.features.12.conv.2": self.backbone.features[12].conv[2],
            "backbone.features.12.conv.3": self.backbone.features[12].conv[3],
            "backbone.features.13": self.backbone.features[13],
            "backbone.features.13.conv": self.backbone.features[13].conv,
            "backbone.features.13.conv.0": self.backbone.features[13].conv[0],
            "backbone.features.13.conv.0.features": self.backbone.features[13].conv[0].features,
            "backbone.features.13.conv.0.features.0": self.backbone.features[13].conv[0].features[0],
            "backbone.features.13.conv.0.features.1": self.backbone.features[13].conv[0].features[1],
            "backbone.features.13.conv.0.features.2": self.backbone.features[13].conv[0].features[2],
            "backbone.features.13.conv.1": self.backbone.features[13].conv[1],
            "backbone.features.13.conv.1.features": self.backbone.features[13].conv[1].features,
            "backbone.features.13.conv.1.features.0": self.backbone.features[13].conv[1].features[0],
            "backbone.features.13.conv.1.features.1": self.backbone.features[13].conv[1].features[1],
            "backbone.features.13.conv.1.features.2": self.backbone.features[13].conv[1].features[2],
            "backbone.features.13.conv.2": self.backbone.features[13].conv[2],
            "backbone.features.13.conv.3": self.backbone.features[13].conv[3],
            "backbone.features.14": self.backbone.features[14],
            "backbone.features.14.conv": self.backbone.features[14].conv,
            "backbone.features.14.conv.0": self.backbone.features[14].conv[0],
            "backbone.features.14.conv.0.features": self.backbone.features[14].conv[0].features,
            "backbone.features.14.conv.0.features.0": self.backbone.features[14].conv[0].features[0],
            "backbone.features.14.conv.0.features.1": self.backbone.features[14].conv[0].features[1],
            "backbone.features.14.conv.0.features.2": self.backbone.features[14].conv[0].features[2],
            "backbone.features.14.conv.1": self.backbone.features[14].conv[1],
            "backbone.features.14.conv.1.features": self.backbone.features[14].conv[1].features,
            "backbone.features.14.conv.1.features.0": self.backbone.features[14].conv[1].features[0],
            "backbone.features.14.conv.1.features.1": self.backbone.features[14].conv[1].features[1],
            "backbone.features.14.conv.1.features.2": self.backbone.features[14].conv[1].features[2],
            "backbone.features.14.conv.2": self.backbone.features[14].conv[2],
            "backbone.features.14.conv.3": self.backbone.features[14].conv[3],
            "backbone.features.15": self.backbone.features[15],
            "backbone.features.15.conv": self.backbone.features[15].conv,
            "backbone.features.15.conv.0": self.backbone.features[15].conv[0],
            "backbone.features.15.conv.0.features": self.backbone.features[15].conv[0].features,
            "backbone.features.15.conv.0.features.0": self.backbone.features[15].conv[0].features[0],
            "backbone.features.15.conv.0.features.1": self.backbone.features[15].conv[0].features[1],
            "backbone.features.15.conv.0.features.2": self.backbone.features[15].conv[0].features[2],
            "backbone.features.15.conv.1": self.backbone.features[15].conv[1],
            "backbone.features.15.conv.1.features": self.backbone.features[15].conv[1].features,
            "backbone.features.15.conv.1.features.0": self.backbone.features[15].conv[1].features[0],
            "backbone.features.15.conv.1.features.1": self.backbone.features[15].conv[1].features[1],
            "backbone.features.15.conv.1.features.2": self.backbone.features[15].conv[1].features[2],
            "backbone.features.15.conv.2": self.backbone.features[15].conv[2],
            "backbone.features.15.conv.3": self.backbone.features[15].conv[3],
            "backbone.features.16": self.backbone.features[16],
            "backbone.features.16.conv": self.backbone.features[16].conv,
            "backbone.features.16.conv.0": self.backbone.features[16].conv[0],
            "backbone.features.16.conv.0.features": self.backbone.features[16].conv[0].features,
            "backbone.features.16.conv.0.features.0": self.backbone.features[16].conv[0].features[0],
            "backbone.features.16.conv.0.features.1": self.backbone.features[16].conv[0].features[1],
            "backbone.features.16.conv.0.features.2": self.backbone.features[16].conv[0].features[2],
            "backbone.features.16.conv.1": self.backbone.features[16].conv[1],
            "backbone.features.16.conv.1.features": self.backbone.features[16].conv[1].features,
            "backbone.features.16.conv.1.features.0": self.backbone.features[16].conv[1].features[0],
            "backbone.features.16.conv.1.features.1": self.backbone.features[16].conv[1].features[1],
            "backbone.features.16.conv.1.features.2": self.backbone.features[16].conv[1].features[2],
            "backbone.features.16.conv.2": self.backbone.features[16].conv[2],
            "backbone.features.16.conv.3": self.backbone.features[16].conv[3],
            "backbone.features.17": self.backbone.features[17],
            "backbone.features.17.conv": self.backbone.features[17].conv,
            "backbone.features.17.conv.0": self.backbone.features[17].conv[0],
            "backbone.features.17.conv.0.features": self.backbone.features[17].conv[0].features,
            "backbone.features.17.conv.0.features.0": self.backbone.features[17].conv[0].features[0],
            "backbone.features.17.conv.0.features.1": self.backbone.features[17].conv[0].features[1],
            "backbone.features.17.conv.0.features.2": self.backbone.features[17].conv[0].features[2],
            "backbone.features.17.conv.1": self.backbone.features[17].conv[1],
            "backbone.features.17.conv.1.features": self.backbone.features[17].conv[1].features,
            "backbone.features.17.conv.1.features.0": self.backbone.features[17].conv[1].features[0],
            "backbone.features.17.conv.1.features.1": self.backbone.features[17].conv[1].features[1],
            "backbone.features.17.conv.1.features.2": self.backbone.features[17].conv[1].features[2],
            "backbone.features.17.conv.2": self.backbone.features[17].conv[2],
            "backbone.features.17.conv.3": self.backbone.features[17].conv[3],
            "backbone.features.18": self.backbone.features[18],
            "backbone.features.18.features": self.backbone.features[18].features,
            "backbone.features.18.features.0": self.backbone.features[18].features[0],
            "backbone.features.18.features.1": self.backbone.features[18].features[1],
            "backbone.features.18.features.2": self.backbone.features[18].features[2],
            "head": self.head,
            "head.head": self.head.head,
            "head.head.0": self.head.head[0],
            "head.dense": self.head.Linear
        }

        self.origin_layer_names = {
            "backbone": self.backbone,
            "backbone.features": self.backbone.features,
            "backbone.features.0": self.backbone.features[0],
            "backbone.features.0.features": self.backbone.features[0].features,
            "backbone.features.0.features.0": self.backbone.features[0].features[0],
            "backbone.features.0.features.1": self.backbone.features[0].features[1],
            "backbone.features.0.features.2": self.backbone.features[0].features[2],
            "backbone.features.1": self.backbone.features[1],
            "backbone.features.1.conv": self.backbone.features[1].conv,
            "backbone.features.1.conv.0": self.backbone.features[1].conv[0],
            "backbone.features.1.conv.0.features": self.backbone.features[1].conv[0].features,
            "backbone.features.1.conv.0.features.0": self.backbone.features[1].conv[0].features[0],
            "backbone.features.1.conv.0.features.1": self.backbone.features[1].conv[0].features[1],
            "backbone.features.1.conv.0.features.2": self.backbone.features[1].conv[0].features[2],
            "backbone.features.1.conv.1": self.backbone.features[1].conv[1],
            "backbone.features.1.conv.2": self.backbone.features[1].conv[2],
            "backbone.features.2": self.backbone.features[2],
            "backbone.features.2.conv": self.backbone.features[2].conv,
            "backbone.features.2.conv.0": self.backbone.features[2].conv[0],
            "backbone.features.2.conv.0.features": self.backbone.features[2].conv[0].features,
            "backbone.features.2.conv.0.features.0": self.backbone.features[2].conv[0].features[0],
            "backbone.features.2.conv.0.features.1": self.backbone.features[2].conv[0].features[1],
            "backbone.features.2.conv.0.features.2": self.backbone.features[2].conv[0].features[2],
            "backbone.features.2.conv.1": self.backbone.features[2].conv[1],
            "backbone.features.2.conv.1.features": self.backbone.features[2].conv[1].features,
            "backbone.features.2.conv.1.features.0": self.backbone.features[2].conv[1].features[0],
            "backbone.features.2.conv.1.features.1": self.backbone.features[2].conv[1].features[1],
            "backbone.features.2.conv.1.features.2": self.backbone.features[2].conv[1].features[2],
            "backbone.features.2.conv.2": self.backbone.features[2].conv[2],
            "backbone.features.2.conv.3": self.backbone.features[2].conv[3],
            "backbone.features.3": self.backbone.features[3],
            "backbone.features.3.conv": self.backbone.features[3].conv,
            "backbone.features.3.conv.0": self.backbone.features[3].conv[0],
            "backbone.features.3.conv.0.features": self.backbone.features[3].conv[0].features,
            "backbone.features.3.conv.0.features.0": self.backbone.features[3].conv[0].features[0],
            "backbone.features.3.conv.0.features.1": self.backbone.features[3].conv[0].features[1],
            "backbone.features.3.conv.0.features.2": self.backbone.features[3].conv[0].features[2],
            "backbone.features.3.conv.1": self.backbone.features[3].conv[1],
            "backbone.features.3.conv.1.features": self.backbone.features[3].conv[1].features,
            "backbone.features.3.conv.1.features.0": self.backbone.features[3].conv[1].features[0],
            "backbone.features.3.conv.1.features.1": self.backbone.features[3].conv[1].features[1],
            "backbone.features.3.conv.1.features.2": self.backbone.features[3].conv[1].features[2],
            "backbone.features.3.conv.2": self.backbone.features[3].conv[2],
            "backbone.features.3.conv.3": self.backbone.features[3].conv[3],
            "backbone.features.4": self.backbone.features[4],
            "backbone.features.4.conv": self.backbone.features[4].conv,
            "backbone.features.4.conv.0": self.backbone.features[4].conv[0],
            "backbone.features.4.conv.0.features": self.backbone.features[4].conv[0].features,
            "backbone.features.4.conv.0.features.0": self.backbone.features[4].conv[0].features[0],
            "backbone.features.4.conv.0.features.1": self.backbone.features[4].conv[0].features[1],
            "backbone.features.4.conv.0.features.2": self.backbone.features[4].conv[0].features[2],
            "backbone.features.4.conv.1": self.backbone.features[4].conv[1],
            "backbone.features.4.conv.1.features": self.backbone.features[4].conv[1].features,
            "backbone.features.4.conv.1.features.0": self.backbone.features[4].conv[1].features[0],
            "backbone.features.4.conv.1.features.1": self.backbone.features[4].conv[1].features[1],
            "backbone.features.4.conv.1.features.2": self.backbone.features[4].conv[1].features[2],
            "backbone.features.4.conv.2": self.backbone.features[4].conv[2],
            "backbone.features.4.conv.3": self.backbone.features[4].conv[3],
            "backbone.features.5": self.backbone.features[5],
            "backbone.features.5.conv": self.backbone.features[5].conv,
            "backbone.features.5.conv.0": self.backbone.features[5].conv[0],
            "backbone.features.5.conv.0.features": self.backbone.features[5].conv[0].features,
            "backbone.features.5.conv.0.features.0": self.backbone.features[5].conv[0].features[0],
            "backbone.features.5.conv.0.features.1": self.backbone.features[5].conv[0].features[1],
            "backbone.features.5.conv.0.features.2": self.backbone.features[5].conv[0].features[2],
            "backbone.features.5.conv.1": self.backbone.features[5].conv[1],
            "backbone.features.5.conv.1.features": self.backbone.features[5].conv[1].features,
            "backbone.features.5.conv.1.features.0": self.backbone.features[5].conv[1].features[0],
            "backbone.features.5.conv.1.features.1": self.backbone.features[5].conv[1].features[1],
            "backbone.features.5.conv.1.features.2": self.backbone.features[5].conv[1].features[2],
            "backbone.features.5.conv.2": self.backbone.features[5].conv[2],
            "backbone.features.5.conv.3": self.backbone.features[5].conv[3],
            "backbone.features.6": self.backbone.features[6],
            "backbone.features.6.conv": self.backbone.features[6].conv,
            "backbone.features.6.conv.0": self.backbone.features[6].conv[0],
            "backbone.features.6.conv.0.features": self.backbone.features[6].conv[0].features,
            "backbone.features.6.conv.0.features.0": self.backbone.features[6].conv[0].features[0],
            "backbone.features.6.conv.0.features.1": self.backbone.features[6].conv[0].features[1],
            "backbone.features.6.conv.0.features.2": self.backbone.features[6].conv[0].features[2],
            "backbone.features.6.conv.1": self.backbone.features[6].conv[1],
            "backbone.features.6.conv.1.features": self.backbone.features[6].conv[1].features,
            "backbone.features.6.conv.1.features.0": self.backbone.features[6].conv[1].features[0],
            "backbone.features.6.conv.1.features.1": self.backbone.features[6].conv[1].features[1],
            "backbone.features.6.conv.1.features.2": self.backbone.features[6].conv[1].features[2],
            "backbone.features.6.conv.2": self.backbone.features[6].conv[2],
            "backbone.features.6.conv.3": self.backbone.features[6].conv[3],
            "backbone.features.7": self.backbone.features[7],
            "backbone.features.7.conv": self.backbone.features[7].conv,
            "backbone.features.7.conv.0": self.backbone.features[7].conv[0],
            "backbone.features.7.conv.0.features": self.backbone.features[7].conv[0].features,
            "backbone.features.7.conv.0.features.0": self.backbone.features[7].conv[0].features[0],
            "backbone.features.7.conv.0.features.1": self.backbone.features[7].conv[0].features[1],
            "backbone.features.7.conv.0.features.2": self.backbone.features[7].conv[0].features[2],
            "backbone.features.7.conv.1": self.backbone.features[7].conv[1],
            "backbone.features.7.conv.1.features": self.backbone.features[7].conv[1].features,
            "backbone.features.7.conv.1.features.0": self.backbone.features[7].conv[1].features[0],
            "backbone.features.7.conv.1.features.1": self.backbone.features[7].conv[1].features[1],
            "backbone.features.7.conv.1.features.2": self.backbone.features[7].conv[1].features[2],
            "backbone.features.7.conv.2": self.backbone.features[7].conv[2],
            "backbone.features.7.conv.3": self.backbone.features[7].conv[3],
            "backbone.features.8": self.backbone.features[8],
            "backbone.features.8.conv": self.backbone.features[8].conv,
            "backbone.features.8.conv.0": self.backbone.features[8].conv[0],
            "backbone.features.8.conv.0.features": self.backbone.features[8].conv[0].features,
            "backbone.features.8.conv.0.features.0": self.backbone.features[8].conv[0].features[0],
            "backbone.features.8.conv.0.features.1": self.backbone.features[8].conv[0].features[1],
            "backbone.features.8.conv.0.features.2": self.backbone.features[8].conv[0].features[2],
            "backbone.features.8.conv.1": self.backbone.features[8].conv[1],
            "backbone.features.8.conv.1.features": self.backbone.features[8].conv[1].features,
            "backbone.features.8.conv.1.features.0": self.backbone.features[8].conv[1].features[0],
            "backbone.features.8.conv.1.features.1": self.backbone.features[8].conv[1].features[1],
            "backbone.features.8.conv.1.features.2": self.backbone.features[8].conv[1].features[2],
            "backbone.features.8.conv.2": self.backbone.features[8].conv[2],
            "backbone.features.8.conv.3": self.backbone.features[8].conv[3],
            "backbone.features.9": self.backbone.features[9],
            "backbone.features.9.conv": self.backbone.features[9].conv,
            "backbone.features.9.conv.0": self.backbone.features[9].conv[0],
            "backbone.features.9.conv.0.features": self.backbone.features[9].conv[0].features,
            "backbone.features.9.conv.0.features.0": self.backbone.features[9].conv[0].features[0],
            "backbone.features.9.conv.0.features.1": self.backbone.features[9].conv[0].features[1],
            "backbone.features.9.conv.0.features.2": self.backbone.features[9].conv[0].features[2],
            "backbone.features.9.conv.1": self.backbone.features[9].conv[1],
            "backbone.features.9.conv.1.features": self.backbone.features[9].conv[1].features,
            "backbone.features.9.conv.1.features.0": self.backbone.features[9].conv[1].features[0],
            "backbone.features.9.conv.1.features.1": self.backbone.features[9].conv[1].features[1],
            "backbone.features.9.conv.1.features.2": self.backbone.features[9].conv[1].features[2],
            "backbone.features.9.conv.2": self.backbone.features[9].conv[2],
            "backbone.features.9.conv.3": self.backbone.features[9].conv[3],
            "backbone.features.10": self.backbone.features[10],
            "backbone.features.10.conv": self.backbone.features[10].conv,
            "backbone.features.10.conv.0": self.backbone.features[10].conv[0],
            "backbone.features.10.conv.0.features": self.backbone.features[10].conv[0].features,
            "backbone.features.10.conv.0.features.0": self.backbone.features[10].conv[0].features[0],
            "backbone.features.10.conv.0.features.1": self.backbone.features[10].conv[0].features[1],
            "backbone.features.10.conv.0.features.2": self.backbone.features[10].conv[0].features[2],
            "backbone.features.10.conv.1": self.backbone.features[10].conv[1],
            "backbone.features.10.conv.1.features": self.backbone.features[10].conv[1].features,
            "backbone.features.10.conv.1.features.0": self.backbone.features[10].conv[1].features[0],
            "backbone.features.10.conv.1.features.1": self.backbone.features[10].conv[1].features[1],
            "backbone.features.10.conv.1.features.2": self.backbone.features[10].conv[1].features[2],
            "backbone.features.10.conv.2": self.backbone.features[10].conv[2],
            "backbone.features.10.conv.3": self.backbone.features[10].conv[3],
            "backbone.features.11": self.backbone.features[11],
            "backbone.features.11.conv": self.backbone.features[11].conv,
            "backbone.features.11.conv.0": self.backbone.features[11].conv[0],
            "backbone.features.11.conv.0.features": self.backbone.features[11].conv[0].features,
            "backbone.features.11.conv.0.features.0": self.backbone.features[11].conv[0].features[0],
            "backbone.features.11.conv.0.features.1": self.backbone.features[11].conv[0].features[1],
            "backbone.features.11.conv.0.features.2": self.backbone.features[11].conv[0].features[2],
            "backbone.features.11.conv.1": self.backbone.features[11].conv[1],
            "backbone.features.11.conv.1.features": self.backbone.features[11].conv[1].features,
            "backbone.features.11.conv.1.features.0": self.backbone.features[11].conv[1].features[0],
            "backbone.features.11.conv.1.features.1": self.backbone.features[11].conv[1].features[1],
            "backbone.features.11.conv.1.features.2": self.backbone.features[11].conv[1].features[2],
            "backbone.features.11.conv.2": self.backbone.features[11].conv[2],
            "backbone.features.11.conv.3": self.backbone.features[11].conv[3],
            "backbone.features.12": self.backbone.features[12],
            "backbone.features.12.conv": self.backbone.features[12].conv,
            "backbone.features.12.conv.0": self.backbone.features[12].conv[0],
            "backbone.features.12.conv.0.features": self.backbone.features[12].conv[0].features,
            "backbone.features.12.conv.0.features.0": self.backbone.features[12].conv[0].features[0],
            "backbone.features.12.conv.0.features.1": self.backbone.features[12].conv[0].features[1],
            "backbone.features.12.conv.0.features.2": self.backbone.features[12].conv[0].features[2],
            "backbone.features.12.conv.1": self.backbone.features[12].conv[1],
            "backbone.features.12.conv.1.features": self.backbone.features[12].conv[1].features,
            "backbone.features.12.conv.1.features.0": self.backbone.features[12].conv[1].features[0],
            "backbone.features.12.conv.1.features.1": self.backbone.features[12].conv[1].features[1],
            "backbone.features.12.conv.1.features.2": self.backbone.features[12].conv[1].features[2],
            "backbone.features.12.conv.2": self.backbone.features[12].conv[2],
            "backbone.features.12.conv.3": self.backbone.features[12].conv[3],
            "backbone.features.13": self.backbone.features[13],
            "backbone.features.13.conv": self.backbone.features[13].conv,
            "backbone.features.13.conv.0": self.backbone.features[13].conv[0],
            "backbone.features.13.conv.0.features": self.backbone.features[13].conv[0].features,
            "backbone.features.13.conv.0.features.0": self.backbone.features[13].conv[0].features[0],
            "backbone.features.13.conv.0.features.1": self.backbone.features[13].conv[0].features[1],
            "backbone.features.13.conv.0.features.2": self.backbone.features[13].conv[0].features[2],
            "backbone.features.13.conv.1": self.backbone.features[13].conv[1],
            "backbone.features.13.conv.1.features": self.backbone.features[13].conv[1].features,
            "backbone.features.13.conv.1.features.0": self.backbone.features[13].conv[1].features[0],
            "backbone.features.13.conv.1.features.1": self.backbone.features[13].conv[1].features[1],
            "backbone.features.13.conv.1.features.2": self.backbone.features[13].conv[1].features[2],
            "backbone.features.13.conv.2": self.backbone.features[13].conv[2],
            "backbone.features.13.conv.3": self.backbone.features[13].conv[3],
            "backbone.features.14": self.backbone.features[14],
            "backbone.features.14.conv": self.backbone.features[14].conv,
            "backbone.features.14.conv.0": self.backbone.features[14].conv[0],
            "backbone.features.14.conv.0.features": self.backbone.features[14].conv[0].features,
            "backbone.features.14.conv.0.features.0": self.backbone.features[14].conv[0].features[0],
            "backbone.features.14.conv.0.features.1": self.backbone.features[14].conv[0].features[1],
            "backbone.features.14.conv.0.features.2": self.backbone.features[14].conv[0].features[2],
            "backbone.features.14.conv.1": self.backbone.features[14].conv[1],
            "backbone.features.14.conv.1.features": self.backbone.features[14].conv[1].features,
            "backbone.features.14.conv.1.features.0": self.backbone.features[14].conv[1].features[0],
            "backbone.features.14.conv.1.features.1": self.backbone.features[14].conv[1].features[1],
            "backbone.features.14.conv.1.features.2": self.backbone.features[14].conv[1].features[2],
            "backbone.features.14.conv.2": self.backbone.features[14].conv[2],
            "backbone.features.14.conv.3": self.backbone.features[14].conv[3],
            "backbone.features.15": self.backbone.features[15],
            "backbone.features.15.conv": self.backbone.features[15].conv,
            "backbone.features.15.conv.0": self.backbone.features[15].conv[0],
            "backbone.features.15.conv.0.features": self.backbone.features[15].conv[0].features,
            "backbone.features.15.conv.0.features.0": self.backbone.features[15].conv[0].features[0],
            "backbone.features.15.conv.0.features.1": self.backbone.features[15].conv[0].features[1],
            "backbone.features.15.conv.0.features.2": self.backbone.features[15].conv[0].features[2],
            "backbone.features.15.conv.1": self.backbone.features[15].conv[1],
            "backbone.features.15.conv.1.features": self.backbone.features[15].conv[1].features,
            "backbone.features.15.conv.1.features.0": self.backbone.features[15].conv[1].features[0],
            "backbone.features.15.conv.1.features.1": self.backbone.features[15].conv[1].features[1],
            "backbone.features.15.conv.1.features.2": self.backbone.features[15].conv[1].features[2],
            "backbone.features.15.conv.2": self.backbone.features[15].conv[2],
            "backbone.features.15.conv.3": self.backbone.features[15].conv[3],
            "backbone.features.16": self.backbone.features[16],
            "backbone.features.16.conv": self.backbone.features[16].conv,
            "backbone.features.16.conv.0": self.backbone.features[16].conv[0],
            "backbone.features.16.conv.0.features": self.backbone.features[16].conv[0].features,
            "backbone.features.16.conv.0.features.0": self.backbone.features[16].conv[0].features[0],
            "backbone.features.16.conv.0.features.1": self.backbone.features[16].conv[0].features[1],
            "backbone.features.16.conv.0.features.2": self.backbone.features[16].conv[0].features[2],
            "backbone.features.16.conv.1": self.backbone.features[16].conv[1],
            "backbone.features.16.conv.1.features": self.backbone.features[16].conv[1].features,
            "backbone.features.16.conv.1.features.0": self.backbone.features[16].conv[1].features[0],
            "backbone.features.16.conv.1.features.1": self.backbone.features[16].conv[1].features[1],
            "backbone.features.16.conv.1.features.2": self.backbone.features[16].conv[1].features[2],
            "backbone.features.16.conv.2": self.backbone.features[16].conv[2],
            "backbone.features.16.conv.3": self.backbone.features[16].conv[3],
            "backbone.features.17": self.backbone.features[17],
            "backbone.features.17.conv": self.backbone.features[17].conv,
            "backbone.features.17.conv.0": self.backbone.features[17].conv[0],
            "backbone.features.17.conv.0.features": self.backbone.features[17].conv[0].features,
            "backbone.features.17.conv.0.features.0": self.backbone.features[17].conv[0].features[0],
            "backbone.features.17.conv.0.features.1": self.backbone.features[17].conv[0].features[1],
            "backbone.features.17.conv.0.features.2": self.backbone.features[17].conv[0].features[2],
            "backbone.features.17.conv.1": self.backbone.features[17].conv[1],
            "backbone.features.17.conv.1.features": self.backbone.features[17].conv[1].features,
            "backbone.features.17.conv.1.features.0": self.backbone.features[17].conv[1].features[0],
            "backbone.features.17.conv.1.features.1": self.backbone.features[17].conv[1].features[1],
            "backbone.features.17.conv.1.features.2": self.backbone.features[17].conv[1].features[2],
            "backbone.features.17.conv.2": self.backbone.features[17].conv[2],
            "backbone.features.17.conv.3": self.backbone.features[17].conv[3],
            "backbone.features.18": self.backbone.features[18],
            "backbone.features.18.features": self.backbone.features[18].features,
            "backbone.features.18.features.0": self.backbone.features[18].features[0],
            "backbone.features.18.features.1": self.backbone.features[18].features[1],
            "backbone.features.18.features.2": self.backbone.features[18].features[2],
            "head": self.head,
            "head.head": self.head.head,
            "head.head.0": self.head.head[0],
            "head.dense": self.head.Linear
        }

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self, layer_name, new_layer):
        if 'backbone' == layer_name:
            self.backbone = new_layer
            self.layer_names["backbone"] = new_layer
            self.origin_layer_names["backbone"] = new_layer
        elif 'backbone.features' == layer_name:
            self.backbone.features = new_layer
            self.layer_names["backbone.features"] = new_layer
            self.origin_layer_names["backbone.features"] = new_layer
        elif 'backbone.features.0' == layer_name:
            self.backbone.features[0] = new_layer
            self.layer_names["backbone.features.0"] = new_layer
            self.origin_layer_names["backbone.features.0"] = new_layer
        elif 'backbone.features.0.features' == layer_name:
            self.backbone.features[0].features = new_layer
            self.layer_names["backbone.features.0.features"] = new_layer
            self.origin_layer_names["backbone.features.0.features"] = new_layer
        elif 'backbone.features.0.features.0' == layer_name:
            self.backbone.features[0].features[0] = new_layer
            self.layer_names["backbone.features.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.0.features.0"] = new_layer
        elif 'backbone.features.0.features.1' == layer_name:
            self.backbone.features[0].features[1] = new_layer
            self.layer_names["backbone.features.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.0.features.1"] = new_layer
        elif 'backbone.features.0.features.2' == layer_name:
            self.backbone.features[0].features[2] = new_layer
            self.layer_names["backbone.features.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.0.features.2"] = new_layer
        elif 'backbone.features.1' == layer_name:
            self.backbone.features[1] = new_layer
            self.layer_names["backbone.features.1"] = new_layer
            self.origin_layer_names["backbone.features.1"] = new_layer
        elif 'backbone.features.1.conv' == layer_name:
            self.backbone.features[1].conv = new_layer
            self.layer_names["backbone.features.1.conv"] = new_layer
            self.origin_layer_names["backbone.features.1.conv"] = new_layer
        elif 'backbone.features.1.conv.0' == layer_name:
            self.backbone.features[1].conv[0] = new_layer
            self.layer_names["backbone.features.1.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.1.conv.0"] = new_layer
        elif 'backbone.features.1.conv.0.features' == layer_name:
            self.backbone.features[1].conv[0].features = new_layer
            self.layer_names["backbone.features.1.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.1.conv.0.features"] = new_layer
        elif 'backbone.features.1.conv.0.features.0' == layer_name:
            self.backbone.features[1].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.1.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.1.conv.0.features.0"] = new_layer
        elif 'backbone.features.1.conv.0.features.1' == layer_name:
            self.backbone.features[1].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.1.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.1.conv.0.features.1"] = new_layer
        elif 'backbone.features.1.conv.0.features.2' == layer_name:
            self.backbone.features[1].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.1.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.1.conv.0.features.2"] = new_layer
        elif 'backbone.features.1.conv.1' == layer_name:
            self.backbone.features[1].conv[1] = new_layer
            self.layer_names["backbone.features.1.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.1.conv.1"] = new_layer
        elif 'backbone.features.1.conv.2' == layer_name:
            self.backbone.features[1].conv[2] = new_layer
            self.layer_names["backbone.features.1.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.1.conv.2"] = new_layer
        elif 'backbone.features.2' == layer_name:
            self.backbone.features[2] = new_layer
            self.layer_names["backbone.features.2"] = new_layer
            self.origin_layer_names["backbone.features.2"] = new_layer
        elif 'backbone.features.2.conv' == layer_name:
            self.backbone.features[2].conv = new_layer
            self.layer_names["backbone.features.2.conv"] = new_layer
            self.origin_layer_names["backbone.features.2.conv"] = new_layer
        elif 'backbone.features.2.conv.0' == layer_name:
            self.backbone.features[2].conv[0] = new_layer
            self.layer_names["backbone.features.2.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.0"] = new_layer
        elif 'backbone.features.2.conv.0.features' == layer_name:
            self.backbone.features[2].conv[0].features = new_layer
            self.layer_names["backbone.features.2.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.0.features"] = new_layer
        elif 'backbone.features.2.conv.0.features.0' == layer_name:
            self.backbone.features[2].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.2.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.0.features.0"] = new_layer
        elif 'backbone.features.2.conv.0.features.1' == layer_name:
            self.backbone.features[2].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.2.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.0.features.1"] = new_layer
        elif 'backbone.features.2.conv.0.features.2' == layer_name:
            self.backbone.features[2].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.2.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.0.features.2"] = new_layer
        elif 'backbone.features.2.conv.1' == layer_name:
            self.backbone.features[2].conv[1] = new_layer
            self.layer_names["backbone.features.2.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.1"] = new_layer
        elif 'backbone.features.2.conv.1.features' == layer_name:
            self.backbone.features[2].conv[1].features = new_layer
            self.layer_names["backbone.features.2.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.1.features"] = new_layer
        elif 'backbone.features.2.conv.1.features.0' == layer_name:
            self.backbone.features[2].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.2.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.1.features.0"] = new_layer
        elif 'backbone.features.2.conv.1.features.1' == layer_name:
            self.backbone.features[2].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.2.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.1.features.1"] = new_layer
        elif 'backbone.features.2.conv.1.features.2' == layer_name:
            self.backbone.features[2].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.2.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.1.features.2"] = new_layer
        elif 'backbone.features.2.conv.2' == layer_name:
            self.backbone.features[2].conv[2] = new_layer
            self.layer_names["backbone.features.2.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.2"] = new_layer
        elif 'backbone.features.2.conv.3' == layer_name:
            self.backbone.features[2].conv[3] = new_layer
            self.layer_names["backbone.features.2.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.2.conv.3"] = new_layer
        elif 'backbone.features.3' == layer_name:
            self.backbone.features[3] = new_layer
            self.layer_names["backbone.features.3"] = new_layer
            self.origin_layer_names["backbone.features.3"] = new_layer
        elif 'backbone.features.3.conv' == layer_name:
            self.backbone.features[3].conv = new_layer
            self.layer_names["backbone.features.3.conv"] = new_layer
            self.origin_layer_names["backbone.features.3.conv"] = new_layer
        elif 'backbone.features.3.conv.0' == layer_name:
            self.backbone.features[3].conv[0] = new_layer
            self.layer_names["backbone.features.3.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.0"] = new_layer
        elif 'backbone.features.3.conv.0.features' == layer_name:
            self.backbone.features[3].conv[0].features = new_layer
            self.layer_names["backbone.features.3.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.0.features"] = new_layer
        elif 'backbone.features.3.conv.0.features.0' == layer_name:
            self.backbone.features[3].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.3.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.0.features.0"] = new_layer
        elif 'backbone.features.3.conv.0.features.1' == layer_name:
            self.backbone.features[3].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.3.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.0.features.1"] = new_layer
        elif 'backbone.features.3.conv.0.features.2' == layer_name:
            self.backbone.features[3].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.3.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.0.features.2"] = new_layer
        elif 'backbone.features.3.conv.1' == layer_name:
            self.backbone.features[3].conv[1] = new_layer
            self.layer_names["backbone.features.3.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.1"] = new_layer
        elif 'backbone.features.3.conv.1.features' == layer_name:
            self.backbone.features[3].conv[1].features = new_layer
            self.layer_names["backbone.features.3.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.1.features"] = new_layer
        elif 'backbone.features.3.conv.1.features.0' == layer_name:
            self.backbone.features[3].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.3.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.1.features.0"] = new_layer
        elif 'backbone.features.3.conv.1.features.1' == layer_name:
            self.backbone.features[3].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.3.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.1.features.1"] = new_layer
        elif 'backbone.features.3.conv.1.features.2' == layer_name:
            self.backbone.features[3].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.3.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.1.features.2"] = new_layer
        elif 'backbone.features.3.conv.2' == layer_name:
            self.backbone.features[3].conv[2] = new_layer
            self.layer_names["backbone.features.3.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.2"] = new_layer
        elif 'backbone.features.3.conv.3' == layer_name:
            self.backbone.features[3].conv[3] = new_layer
            self.layer_names["backbone.features.3.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.3.conv.3"] = new_layer
        elif 'backbone.features.4' == layer_name:
            self.backbone.features[4] = new_layer
            self.layer_names["backbone.features.4"] = new_layer
            self.origin_layer_names["backbone.features.4"] = new_layer
        elif 'backbone.features.4.conv' == layer_name:
            self.backbone.features[4].conv = new_layer
            self.layer_names["backbone.features.4.conv"] = new_layer
            self.origin_layer_names["backbone.features.4.conv"] = new_layer
        elif 'backbone.features.4.conv.0' == layer_name:
            self.backbone.features[4].conv[0] = new_layer
            self.layer_names["backbone.features.4.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.0"] = new_layer
        elif 'backbone.features.4.conv.0.features' == layer_name:
            self.backbone.features[4].conv[0].features = new_layer
            self.layer_names["backbone.features.4.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.0.features"] = new_layer
        elif 'backbone.features.4.conv.0.features.0' == layer_name:
            self.backbone.features[4].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.4.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.0.features.0"] = new_layer
        elif 'backbone.features.4.conv.0.features.1' == layer_name:
            self.backbone.features[4].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.4.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.0.features.1"] = new_layer
        elif 'backbone.features.4.conv.0.features.2' == layer_name:
            self.backbone.features[4].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.4.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.0.features.2"] = new_layer
        elif 'backbone.features.4.conv.1' == layer_name:
            self.backbone.features[4].conv[1] = new_layer
            self.layer_names["backbone.features.4.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.1"] = new_layer
        elif 'backbone.features.4.conv.1.features' == layer_name:
            self.backbone.features[4].conv[1].features = new_layer
            self.layer_names["backbone.features.4.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.1.features"] = new_layer
        elif 'backbone.features.4.conv.1.features.0' == layer_name:
            self.backbone.features[4].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.4.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.1.features.0"] = new_layer
        elif 'backbone.features.4.conv.1.features.1' == layer_name:
            self.backbone.features[4].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.4.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.1.features.1"] = new_layer
        elif 'backbone.features.4.conv.1.features.2' == layer_name:
            self.backbone.features[4].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.4.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.1.features.2"] = new_layer
        elif 'backbone.features.4.conv.2' == layer_name:
            self.backbone.features[4].conv[2] = new_layer
            self.layer_names["backbone.features.4.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.2"] = new_layer
        elif 'backbone.features.4.conv.3' == layer_name:
            self.backbone.features[4].conv[3] = new_layer
            self.layer_names["backbone.features.4.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.4.conv.3"] = new_layer
        elif 'backbone.features.5' == layer_name:
            self.backbone.features[5] = new_layer
            self.layer_names["backbone.features.5"] = new_layer
            self.origin_layer_names["backbone.features.5"] = new_layer
        elif 'backbone.features.5.conv' == layer_name:
            self.backbone.features[5].conv = new_layer
            self.layer_names["backbone.features.5.conv"] = new_layer
            self.origin_layer_names["backbone.features.5.conv"] = new_layer
        elif 'backbone.features.5.conv.0' == layer_name:
            self.backbone.features[5].conv[0] = new_layer
            self.layer_names["backbone.features.5.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.0"] = new_layer
        elif 'backbone.features.5.conv.0.features' == layer_name:
            self.backbone.features[5].conv[0].features = new_layer
            self.layer_names["backbone.features.5.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.0.features"] = new_layer
        elif 'backbone.features.5.conv.0.features.0' == layer_name:
            self.backbone.features[5].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.5.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.0.features.0"] = new_layer
        elif 'backbone.features.5.conv.0.features.1' == layer_name:
            self.backbone.features[5].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.5.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.0.features.1"] = new_layer
        elif 'backbone.features.5.conv.0.features.2' == layer_name:
            self.backbone.features[5].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.5.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.0.features.2"] = new_layer
        elif 'backbone.features.5.conv.1' == layer_name:
            self.backbone.features[5].conv[1] = new_layer
            self.layer_names["backbone.features.5.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.1"] = new_layer
        elif 'backbone.features.5.conv.1.features' == layer_name:
            self.backbone.features[5].conv[1].features = new_layer
            self.layer_names["backbone.features.5.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.1.features"] = new_layer
        elif 'backbone.features.5.conv.1.features.0' == layer_name:
            self.backbone.features[5].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.5.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.1.features.0"] = new_layer
        elif 'backbone.features.5.conv.1.features.1' == layer_name:
            self.backbone.features[5].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.5.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.1.features.1"] = new_layer
        elif 'backbone.features.5.conv.1.features.2' == layer_name:
            self.backbone.features[5].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.5.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.1.features.2"] = new_layer
        elif 'backbone.features.5.conv.2' == layer_name:
            self.backbone.features[5].conv[2] = new_layer
            self.layer_names["backbone.features.5.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.2"] = new_layer
        elif 'backbone.features.5.conv.3' == layer_name:
            self.backbone.features[5].conv[3] = new_layer
            self.layer_names["backbone.features.5.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.5.conv.3"] = new_layer
        elif 'backbone.features.6' == layer_name:
            self.backbone.features[6] = new_layer
            self.layer_names["backbone.features.6"] = new_layer
            self.origin_layer_names["backbone.features.6"] = new_layer
        elif 'backbone.features.6.conv' == layer_name:
            self.backbone.features[6].conv = new_layer
            self.layer_names["backbone.features.6.conv"] = new_layer
            self.origin_layer_names["backbone.features.6.conv"] = new_layer
        elif 'backbone.features.6.conv.0' == layer_name:
            self.backbone.features[6].conv[0] = new_layer
            self.layer_names["backbone.features.6.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.0"] = new_layer
        elif 'backbone.features.6.conv.0.features' == layer_name:
            self.backbone.features[6].conv[0].features = new_layer
            self.layer_names["backbone.features.6.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.0.features"] = new_layer
        elif 'backbone.features.6.conv.0.features.0' == layer_name:
            self.backbone.features[6].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.6.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.0.features.0"] = new_layer
        elif 'backbone.features.6.conv.0.features.1' == layer_name:
            self.backbone.features[6].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.6.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.0.features.1"] = new_layer
        elif 'backbone.features.6.conv.0.features.2' == layer_name:
            self.backbone.features[6].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.6.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.0.features.2"] = new_layer
        elif 'backbone.features.6.conv.1' == layer_name:
            self.backbone.features[6].conv[1] = new_layer
            self.layer_names["backbone.features.6.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.1"] = new_layer
        elif 'backbone.features.6.conv.1.features' == layer_name:
            self.backbone.features[6].conv[1].features = new_layer
            self.layer_names["backbone.features.6.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.1.features"] = new_layer
        elif 'backbone.features.6.conv.1.features.0' == layer_name:
            self.backbone.features[6].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.6.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.1.features.0"] = new_layer
        elif 'backbone.features.6.conv.1.features.1' == layer_name:
            self.backbone.features[6].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.6.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.1.features.1"] = new_layer
        elif 'backbone.features.6.conv.1.features.2' == layer_name:
            self.backbone.features[6].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.6.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.1.features.2"] = new_layer
        elif 'backbone.features.6.conv.2' == layer_name:
            self.backbone.features[6].conv[2] = new_layer
            self.layer_names["backbone.features.6.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.2"] = new_layer
        elif 'backbone.features.6.conv.3' == layer_name:
            self.backbone.features[6].conv[3] = new_layer
            self.layer_names["backbone.features.6.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.6.conv.3"] = new_layer
        elif 'backbone.features.7' == layer_name:
            self.backbone.features[7] = new_layer
            self.layer_names["backbone.features.7"] = new_layer
            self.origin_layer_names["backbone.features.7"] = new_layer
        elif 'backbone.features.7.conv' == layer_name:
            self.backbone.features[7].conv = new_layer
            self.layer_names["backbone.features.7.conv"] = new_layer
            self.origin_layer_names["backbone.features.7.conv"] = new_layer
        elif 'backbone.features.7.conv.0' == layer_name:
            self.backbone.features[7].conv[0] = new_layer
            self.layer_names["backbone.features.7.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.0"] = new_layer
        elif 'backbone.features.7.conv.0.features' == layer_name:
            self.backbone.features[7].conv[0].features = new_layer
            self.layer_names["backbone.features.7.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.0.features"] = new_layer
        elif 'backbone.features.7.conv.0.features.0' == layer_name:
            self.backbone.features[7].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.7.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.0.features.0"] = new_layer
        elif 'backbone.features.7.conv.0.features.1' == layer_name:
            self.backbone.features[7].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.7.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.0.features.1"] = new_layer
        elif 'backbone.features.7.conv.0.features.2' == layer_name:
            self.backbone.features[7].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.7.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.0.features.2"] = new_layer
        elif 'backbone.features.7.conv.1' == layer_name:
            self.backbone.features[7].conv[1] = new_layer
            self.layer_names["backbone.features.7.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.1"] = new_layer
        elif 'backbone.features.7.conv.1.features' == layer_name:
            self.backbone.features[7].conv[1].features = new_layer
            self.layer_names["backbone.features.7.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.1.features"] = new_layer
        elif 'backbone.features.7.conv.1.features.0' == layer_name:
            self.backbone.features[7].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.7.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.1.features.0"] = new_layer
        elif 'backbone.features.7.conv.1.features.1' == layer_name:
            self.backbone.features[7].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.7.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.1.features.1"] = new_layer
        elif 'backbone.features.7.conv.1.features.2' == layer_name:
            self.backbone.features[7].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.7.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.1.features.2"] = new_layer
        elif 'backbone.features.7.conv.2' == layer_name:
            self.backbone.features[7].conv[2] = new_layer
            self.layer_names["backbone.features.7.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.2"] = new_layer
        elif 'backbone.features.7.conv.3' == layer_name:
            self.backbone.features[7].conv[3] = new_layer
            self.layer_names["backbone.features.7.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.7.conv.3"] = new_layer
        elif 'backbone.features.8' == layer_name:
            self.backbone.features[8] = new_layer
            self.layer_names["backbone.features.8"] = new_layer
            self.origin_layer_names["backbone.features.8"] = new_layer
        elif 'backbone.features.8.conv' == layer_name:
            self.backbone.features[8].conv = new_layer
            self.layer_names["backbone.features.8.conv"] = new_layer
            self.origin_layer_names["backbone.features.8.conv"] = new_layer
        elif 'backbone.features.8.conv.0' == layer_name:
            self.backbone.features[8].conv[0] = new_layer
            self.layer_names["backbone.features.8.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.0"] = new_layer
        elif 'backbone.features.8.conv.0.features' == layer_name:
            self.backbone.features[8].conv[0].features = new_layer
            self.layer_names["backbone.features.8.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.0.features"] = new_layer
        elif 'backbone.features.8.conv.0.features.0' == layer_name:
            self.backbone.features[8].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.8.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.0.features.0"] = new_layer
        elif 'backbone.features.8.conv.0.features.1' == layer_name:
            self.backbone.features[8].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.8.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.0.features.1"] = new_layer
        elif 'backbone.features.8.conv.0.features.2' == layer_name:
            self.backbone.features[8].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.8.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.0.features.2"] = new_layer
        elif 'backbone.features.8.conv.1' == layer_name:
            self.backbone.features[8].conv[1] = new_layer
            self.layer_names["backbone.features.8.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.1"] = new_layer
        elif 'backbone.features.8.conv.1.features' == layer_name:
            self.backbone.features[8].conv[1].features = new_layer
            self.layer_names["backbone.features.8.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.1.features"] = new_layer
        elif 'backbone.features.8.conv.1.features.0' == layer_name:
            self.backbone.features[8].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.8.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.1.features.0"] = new_layer
        elif 'backbone.features.8.conv.1.features.1' == layer_name:
            self.backbone.features[8].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.8.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.1.features.1"] = new_layer
        elif 'backbone.features.8.conv.1.features.2' == layer_name:
            self.backbone.features[8].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.8.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.1.features.2"] = new_layer
        elif 'backbone.features.8.conv.2' == layer_name:
            self.backbone.features[8].conv[2] = new_layer
            self.layer_names["backbone.features.8.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.2"] = new_layer
        elif 'backbone.features.8.conv.3' == layer_name:
            self.backbone.features[8].conv[3] = new_layer
            self.layer_names["backbone.features.8.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.8.conv.3"] = new_layer
        elif 'backbone.features.9' == layer_name:
            self.backbone.features[9] = new_layer
            self.layer_names["backbone.features.9"] = new_layer
            self.origin_layer_names["backbone.features.9"] = new_layer
        elif 'backbone.features.9.conv' == layer_name:
            self.backbone.features[9].conv = new_layer
            self.layer_names["backbone.features.9.conv"] = new_layer
            self.origin_layer_names["backbone.features.9.conv"] = new_layer
        elif 'backbone.features.9.conv.0' == layer_name:
            self.backbone.features[9].conv[0] = new_layer
            self.layer_names["backbone.features.9.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.0"] = new_layer
        elif 'backbone.features.9.conv.0.features' == layer_name:
            self.backbone.features[9].conv[0].features = new_layer
            self.layer_names["backbone.features.9.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.0.features"] = new_layer
        elif 'backbone.features.9.conv.0.features.0' == layer_name:
            self.backbone.features[9].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.9.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.0.features.0"] = new_layer
        elif 'backbone.features.9.conv.0.features.1' == layer_name:
            self.backbone.features[9].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.9.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.0.features.1"] = new_layer
        elif 'backbone.features.9.conv.0.features.2' == layer_name:
            self.backbone.features[9].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.9.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.0.features.2"] = new_layer
        elif 'backbone.features.9.conv.1' == layer_name:
            self.backbone.features[9].conv[1] = new_layer
            self.layer_names["backbone.features.9.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.1"] = new_layer
        elif 'backbone.features.9.conv.1.features' == layer_name:
            self.backbone.features[9].conv[1].features = new_layer
            self.layer_names["backbone.features.9.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.1.features"] = new_layer
        elif 'backbone.features.9.conv.1.features.0' == layer_name:
            self.backbone.features[9].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.9.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.1.features.0"] = new_layer
        elif 'backbone.features.9.conv.1.features.1' == layer_name:
            self.backbone.features[9].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.9.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.1.features.1"] = new_layer
        elif 'backbone.features.9.conv.1.features.2' == layer_name:
            self.backbone.features[9].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.9.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.1.features.2"] = new_layer
        elif 'backbone.features.9.conv.2' == layer_name:
            self.backbone.features[9].conv[2] = new_layer
            self.layer_names["backbone.features.9.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.2"] = new_layer
        elif 'backbone.features.9.conv.3' == layer_name:
            self.backbone.features[9].conv[3] = new_layer
            self.layer_names["backbone.features.9.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.9.conv.3"] = new_layer
        elif 'backbone.features.10' == layer_name:
            self.backbone.features[10] = new_layer
            self.layer_names["backbone.features.10"] = new_layer
            self.origin_layer_names["backbone.features.10"] = new_layer
        elif 'backbone.features.10.conv' == layer_name:
            self.backbone.features[10].conv = new_layer
            self.layer_names["backbone.features.10.conv"] = new_layer
            self.origin_layer_names["backbone.features.10.conv"] = new_layer
        elif 'backbone.features.10.conv.0' == layer_name:
            self.backbone.features[10].conv[0] = new_layer
            self.layer_names["backbone.features.10.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.0"] = new_layer
        elif 'backbone.features.10.conv.0.features' == layer_name:
            self.backbone.features[10].conv[0].features = new_layer
            self.layer_names["backbone.features.10.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.0.features"] = new_layer
        elif 'backbone.features.10.conv.0.features.0' == layer_name:
            self.backbone.features[10].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.10.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.0.features.0"] = new_layer
        elif 'backbone.features.10.conv.0.features.1' == layer_name:
            self.backbone.features[10].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.10.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.0.features.1"] = new_layer
        elif 'backbone.features.10.conv.0.features.2' == layer_name:
            self.backbone.features[10].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.10.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.0.features.2"] = new_layer
        elif 'backbone.features.10.conv.1' == layer_name:
            self.backbone.features[10].conv[1] = new_layer
            self.layer_names["backbone.features.10.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.1"] = new_layer
        elif 'backbone.features.10.conv.1.features' == layer_name:
            self.backbone.features[10].conv[1].features = new_layer
            self.layer_names["backbone.features.10.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.1.features"] = new_layer
        elif 'backbone.features.10.conv.1.features.0' == layer_name:
            self.backbone.features[10].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.10.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.1.features.0"] = new_layer
        elif 'backbone.features.10.conv.1.features.1' == layer_name:
            self.backbone.features[10].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.10.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.1.features.1"] = new_layer
        elif 'backbone.features.10.conv.1.features.2' == layer_name:
            self.backbone.features[10].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.10.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.1.features.2"] = new_layer
        elif 'backbone.features.10.conv.2' == layer_name:
            self.backbone.features[10].conv[2] = new_layer
            self.layer_names["backbone.features.10.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.2"] = new_layer
        elif 'backbone.features.10.conv.3' == layer_name:
            self.backbone.features[10].conv[3] = new_layer
            self.layer_names["backbone.features.10.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.10.conv.3"] = new_layer
        elif 'backbone.features.11' == layer_name:
            self.backbone.features[11] = new_layer
            self.layer_names["backbone.features.11"] = new_layer
            self.origin_layer_names["backbone.features.11"] = new_layer
        elif 'backbone.features.11.conv' == layer_name:
            self.backbone.features[11].conv = new_layer
            self.layer_names["backbone.features.11.conv"] = new_layer
            self.origin_layer_names["backbone.features.11.conv"] = new_layer
        elif 'backbone.features.11.conv.0' == layer_name:
            self.backbone.features[11].conv[0] = new_layer
            self.layer_names["backbone.features.11.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.0"] = new_layer
        elif 'backbone.features.11.conv.0.features' == layer_name:
            self.backbone.features[11].conv[0].features = new_layer
            self.layer_names["backbone.features.11.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.0.features"] = new_layer
        elif 'backbone.features.11.conv.0.features.0' == layer_name:
            self.backbone.features[11].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.11.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.0.features.0"] = new_layer
        elif 'backbone.features.11.conv.0.features.1' == layer_name:
            self.backbone.features[11].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.11.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.0.features.1"] = new_layer
        elif 'backbone.features.11.conv.0.features.2' == layer_name:
            self.backbone.features[11].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.11.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.0.features.2"] = new_layer
        elif 'backbone.features.11.conv.1' == layer_name:
            self.backbone.features[11].conv[1] = new_layer
            self.layer_names["backbone.features.11.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.1"] = new_layer
        elif 'backbone.features.11.conv.1.features' == layer_name:
            self.backbone.features[11].conv[1].features = new_layer
            self.layer_names["backbone.features.11.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.1.features"] = new_layer
        elif 'backbone.features.11.conv.1.features.0' == layer_name:
            self.backbone.features[11].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.11.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.1.features.0"] = new_layer
        elif 'backbone.features.11.conv.1.features.1' == layer_name:
            self.backbone.features[11].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.11.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.1.features.1"] = new_layer
        elif 'backbone.features.11.conv.1.features.2' == layer_name:
            self.backbone.features[11].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.11.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.1.features.2"] = new_layer
        elif 'backbone.features.11.conv.2' == layer_name:
            self.backbone.features[11].conv[2] = new_layer
            self.layer_names["backbone.features.11.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.2"] = new_layer
        elif 'backbone.features.11.conv.3' == layer_name:
            self.backbone.features[11].conv[3] = new_layer
            self.layer_names["backbone.features.11.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.11.conv.3"] = new_layer
        elif 'backbone.features.12' == layer_name:
            self.backbone.features[12] = new_layer
            self.layer_names["backbone.features.12"] = new_layer
            self.origin_layer_names["backbone.features.12"] = new_layer
        elif 'backbone.features.12.conv' == layer_name:
            self.backbone.features[12].conv = new_layer
            self.layer_names["backbone.features.12.conv"] = new_layer
            self.origin_layer_names["backbone.features.12.conv"] = new_layer
        elif 'backbone.features.12.conv.0' == layer_name:
            self.backbone.features[12].conv[0] = new_layer
            self.layer_names["backbone.features.12.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.0"] = new_layer
        elif 'backbone.features.12.conv.0.features' == layer_name:
            self.backbone.features[12].conv[0].features = new_layer
            self.layer_names["backbone.features.12.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.0.features"] = new_layer
        elif 'backbone.features.12.conv.0.features.0' == layer_name:
            self.backbone.features[12].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.12.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.0.features.0"] = new_layer
        elif 'backbone.features.12.conv.0.features.1' == layer_name:
            self.backbone.features[12].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.12.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.0.features.1"] = new_layer
        elif 'backbone.features.12.conv.0.features.2' == layer_name:
            self.backbone.features[12].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.12.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.0.features.2"] = new_layer
        elif 'backbone.features.12.conv.1' == layer_name:
            self.backbone.features[12].conv[1] = new_layer
            self.layer_names["backbone.features.12.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.1"] = new_layer
        elif 'backbone.features.12.conv.1.features' == layer_name:
            self.backbone.features[12].conv[1].features = new_layer
            self.layer_names["backbone.features.12.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.1.features"] = new_layer
        elif 'backbone.features.12.conv.1.features.0' == layer_name:
            self.backbone.features[12].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.12.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.1.features.0"] = new_layer
        elif 'backbone.features.12.conv.1.features.1' == layer_name:
            self.backbone.features[12].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.12.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.1.features.1"] = new_layer
        elif 'backbone.features.12.conv.1.features.2' == layer_name:
            self.backbone.features[12].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.12.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.1.features.2"] = new_layer
        elif 'backbone.features.12.conv.2' == layer_name:
            self.backbone.features[12].conv[2] = new_layer
            self.layer_names["backbone.features.12.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.2"] = new_layer
        elif 'backbone.features.12.conv.3' == layer_name:
            self.backbone.features[12].conv[3] = new_layer
            self.layer_names["backbone.features.12.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.12.conv.3"] = new_layer
        elif 'backbone.features.13' == layer_name:
            self.backbone.features[13] = new_layer
            self.layer_names["backbone.features.13"] = new_layer
            self.origin_layer_names["backbone.features.13"] = new_layer
        elif 'backbone.features.13.conv' == layer_name:
            self.backbone.features[13].conv = new_layer
            self.layer_names["backbone.features.13.conv"] = new_layer
            self.origin_layer_names["backbone.features.13.conv"] = new_layer
        elif 'backbone.features.13.conv.0' == layer_name:
            self.backbone.features[13].conv[0] = new_layer
            self.layer_names["backbone.features.13.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.0"] = new_layer
        elif 'backbone.features.13.conv.0.features' == layer_name:
            self.backbone.features[13].conv[0].features = new_layer
            self.layer_names["backbone.features.13.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.0.features"] = new_layer
        elif 'backbone.features.13.conv.0.features.0' == layer_name:
            self.backbone.features[13].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.13.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.0.features.0"] = new_layer
        elif 'backbone.features.13.conv.0.features.1' == layer_name:
            self.backbone.features[13].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.13.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.0.features.1"] = new_layer
        elif 'backbone.features.13.conv.0.features.2' == layer_name:
            self.backbone.features[13].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.13.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.0.features.2"] = new_layer
        elif 'backbone.features.13.conv.1' == layer_name:
            self.backbone.features[13].conv[1] = new_layer
            self.layer_names["backbone.features.13.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.1"] = new_layer
        elif 'backbone.features.13.conv.1.features' == layer_name:
            self.backbone.features[13].conv[1].features = new_layer
            self.layer_names["backbone.features.13.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.1.features"] = new_layer
        elif 'backbone.features.13.conv.1.features.0' == layer_name:
            self.backbone.features[13].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.13.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.1.features.0"] = new_layer
        elif 'backbone.features.13.conv.1.features.1' == layer_name:
            self.backbone.features[13].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.13.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.1.features.1"] = new_layer
        elif 'backbone.features.13.conv.1.features.2' == layer_name:
            self.backbone.features[13].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.13.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.1.features.2"] = new_layer
        elif 'backbone.features.13.conv.2' == layer_name:
            self.backbone.features[13].conv[2] = new_layer
            self.layer_names["backbone.features.13.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.2"] = new_layer
        elif 'backbone.features.13.conv.3' == layer_name:
            self.backbone.features[13].conv[3] = new_layer
            self.layer_names["backbone.features.13.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.13.conv.3"] = new_layer
        elif 'backbone.features.14' == layer_name:
            self.backbone.features[14] = new_layer
            self.layer_names["backbone.features.14"] = new_layer
            self.origin_layer_names["backbone.features.14"] = new_layer
        elif 'backbone.features.14.conv' == layer_name:
            self.backbone.features[14].conv = new_layer
            self.layer_names["backbone.features.14.conv"] = new_layer
            self.origin_layer_names["backbone.features.14.conv"] = new_layer
        elif 'backbone.features.14.conv.0' == layer_name:
            self.backbone.features[14].conv[0] = new_layer
            self.layer_names["backbone.features.14.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.0"] = new_layer
        elif 'backbone.features.14.conv.0.features' == layer_name:
            self.backbone.features[14].conv[0].features = new_layer
            self.layer_names["backbone.features.14.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.0.features"] = new_layer
        elif 'backbone.features.14.conv.0.features.0' == layer_name:
            self.backbone.features[14].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.14.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.0.features.0"] = new_layer
        elif 'backbone.features.14.conv.0.features.1' == layer_name:
            self.backbone.features[14].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.14.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.0.features.1"] = new_layer
        elif 'backbone.features.14.conv.0.features.2' == layer_name:
            self.backbone.features[14].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.14.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.0.features.2"] = new_layer
        elif 'backbone.features.14.conv.1' == layer_name:
            self.backbone.features[14].conv[1] = new_layer
            self.layer_names["backbone.features.14.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.1"] = new_layer
        elif 'backbone.features.14.conv.1.features' == layer_name:
            self.backbone.features[14].conv[1].features = new_layer
            self.layer_names["backbone.features.14.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.1.features"] = new_layer
        elif 'backbone.features.14.conv.1.features.0' == layer_name:
            self.backbone.features[14].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.14.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.1.features.0"] = new_layer
        elif 'backbone.features.14.conv.1.features.1' == layer_name:
            self.backbone.features[14].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.14.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.1.features.1"] = new_layer
        elif 'backbone.features.14.conv.1.features.2' == layer_name:
            self.backbone.features[14].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.14.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.1.features.2"] = new_layer
        elif 'backbone.features.14.conv.2' == layer_name:
            self.backbone.features[14].conv[2] = new_layer
            self.layer_names["backbone.features.14.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.2"] = new_layer
        elif 'backbone.features.14.conv.3' == layer_name:
            self.backbone.features[14].conv[3] = new_layer
            self.layer_names["backbone.features.14.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.14.conv.3"] = new_layer
        elif 'backbone.features.15' == layer_name:
            self.backbone.features[15] = new_layer
            self.layer_names["backbone.features.15"] = new_layer
            self.origin_layer_names["backbone.features.15"] = new_layer
        elif 'backbone.features.15.conv' == layer_name:
            self.backbone.features[15].conv = new_layer
            self.layer_names["backbone.features.15.conv"] = new_layer
            self.origin_layer_names["backbone.features.15.conv"] = new_layer
        elif 'backbone.features.15.conv.0' == layer_name:
            self.backbone.features[15].conv[0] = new_layer
            self.layer_names["backbone.features.15.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.0"] = new_layer
        elif 'backbone.features.15.conv.0.features' == layer_name:
            self.backbone.features[15].conv[0].features = new_layer
            self.layer_names["backbone.features.15.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.0.features"] = new_layer
        elif 'backbone.features.15.conv.0.features.0' == layer_name:
            self.backbone.features[15].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.15.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.0.features.0"] = new_layer
        elif 'backbone.features.15.conv.0.features.1' == layer_name:
            self.backbone.features[15].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.15.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.0.features.1"] = new_layer
        elif 'backbone.features.15.conv.0.features.2' == layer_name:
            self.backbone.features[15].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.15.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.0.features.2"] = new_layer
        elif 'backbone.features.15.conv.1' == layer_name:
            self.backbone.features[15].conv[1] = new_layer
            self.layer_names["backbone.features.15.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.1"] = new_layer
        elif 'backbone.features.15.conv.1.features' == layer_name:
            self.backbone.features[15].conv[1].features = new_layer
            self.layer_names["backbone.features.15.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.1.features"] = new_layer
        elif 'backbone.features.15.conv.1.features.0' == layer_name:
            self.backbone.features[15].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.15.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.1.features.0"] = new_layer
        elif 'backbone.features.15.conv.1.features.1' == layer_name:
            self.backbone.features[15].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.15.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.1.features.1"] = new_layer
        elif 'backbone.features.15.conv.1.features.2' == layer_name:
            self.backbone.features[15].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.15.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.1.features.2"] = new_layer
        elif 'backbone.features.15.conv.2' == layer_name:
            self.backbone.features[15].conv[2] = new_layer
            self.layer_names["backbone.features.15.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.2"] = new_layer
        elif 'backbone.features.15.conv.3' == layer_name:
            self.backbone.features[15].conv[3] = new_layer
            self.layer_names["backbone.features.15.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.15.conv.3"] = new_layer
        elif 'backbone.features.16' == layer_name:
            self.backbone.features[16] = new_layer
            self.layer_names["backbone.features.16"] = new_layer
            self.origin_layer_names["backbone.features.16"] = new_layer
        elif 'backbone.features.16.conv' == layer_name:
            self.backbone.features[16].conv = new_layer
            self.layer_names["backbone.features.16.conv"] = new_layer
            self.origin_layer_names["backbone.features.16.conv"] = new_layer
        elif 'backbone.features.16.conv.0' == layer_name:
            self.backbone.features[16].conv[0] = new_layer
            self.layer_names["backbone.features.16.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.0"] = new_layer
        elif 'backbone.features.16.conv.0.features' == layer_name:
            self.backbone.features[16].conv[0].features = new_layer
            self.layer_names["backbone.features.16.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.0.features"] = new_layer
        elif 'backbone.features.16.conv.0.features.0' == layer_name:
            self.backbone.features[16].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.16.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.0.features.0"] = new_layer
        elif 'backbone.features.16.conv.0.features.1' == layer_name:
            self.backbone.features[16].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.16.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.0.features.1"] = new_layer
        elif 'backbone.features.16.conv.0.features.2' == layer_name:
            self.backbone.features[16].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.16.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.0.features.2"] = new_layer
        elif 'backbone.features.16.conv.1' == layer_name:
            self.backbone.features[16].conv[1] = new_layer
            self.layer_names["backbone.features.16.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.1"] = new_layer
        elif 'backbone.features.16.conv.1.features' == layer_name:
            self.backbone.features[16].conv[1].features = new_layer
            self.layer_names["backbone.features.16.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.1.features"] = new_layer
        elif 'backbone.features.16.conv.1.features.0' == layer_name:
            self.backbone.features[16].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.16.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.1.features.0"] = new_layer
        elif 'backbone.features.16.conv.1.features.1' == layer_name:
            self.backbone.features[16].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.16.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.1.features.1"] = new_layer
        elif 'backbone.features.16.conv.1.features.2' == layer_name:
            self.backbone.features[16].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.16.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.1.features.2"] = new_layer
        elif 'backbone.features.16.conv.2' == layer_name:
            self.backbone.features[16].conv[2] = new_layer
            self.layer_names["backbone.features.16.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.2"] = new_layer
        elif 'backbone.features.16.conv.3' == layer_name:
            self.backbone.features[16].conv[3] = new_layer
            self.layer_names["backbone.features.16.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.16.conv.3"] = new_layer
        elif 'backbone.features.17' == layer_name:
            self.backbone.features[17] = new_layer
            self.layer_names["backbone.features.17"] = new_layer
            self.origin_layer_names["backbone.features.17"] = new_layer
        elif 'backbone.features.17.conv' == layer_name:
            self.backbone.features[17].conv = new_layer
            self.layer_names["backbone.features.17.conv"] = new_layer
            self.origin_layer_names["backbone.features.17.conv"] = new_layer
        elif 'backbone.features.17.conv.0' == layer_name:
            self.backbone.features[17].conv[0] = new_layer
            self.layer_names["backbone.features.17.conv.0"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.0"] = new_layer
        elif 'backbone.features.17.conv.0.features' == layer_name:
            self.backbone.features[17].conv[0].features = new_layer
            self.layer_names["backbone.features.17.conv.0.features"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.0.features"] = new_layer
        elif 'backbone.features.17.conv.0.features.0' == layer_name:
            self.backbone.features[17].conv[0].features[0] = new_layer
            self.layer_names["backbone.features.17.conv.0.features.0"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.0.features.0"] = new_layer
        elif 'backbone.features.17.conv.0.features.1' == layer_name:
            self.backbone.features[17].conv[0].features[1] = new_layer
            self.layer_names["backbone.features.17.conv.0.features.1"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.0.features.1"] = new_layer
        elif 'backbone.features.17.conv.0.features.2' == layer_name:
            self.backbone.features[17].conv[0].features[2] = new_layer
            self.layer_names["backbone.features.17.conv.0.features.2"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.0.features.2"] = new_layer
        elif 'backbone.features.17.conv.1' == layer_name:
            self.backbone.features[17].conv[1] = new_layer
            self.layer_names["backbone.features.17.conv.1"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.1"] = new_layer
        elif 'backbone.features.17.conv.1.features' == layer_name:
            self.backbone.features[17].conv[1].features = new_layer
            self.layer_names["backbone.features.17.conv.1.features"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.1.features"] = new_layer
        elif 'backbone.features.17.conv.1.features.0' == layer_name:
            self.backbone.features[17].conv[1].features[0] = new_layer
            self.layer_names["backbone.features.17.conv.1.features.0"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.1.features.0"] = new_layer
        elif 'backbone.features.17.conv.1.features.1' == layer_name:
            self.backbone.features[17].conv[1].features[1] = new_layer
            self.layer_names["backbone.features.17.conv.1.features.1"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.1.features.1"] = new_layer
        elif 'backbone.features.17.conv.1.features.2' == layer_name:
            self.backbone.features[17].conv[1].features[2] = new_layer
            self.layer_names["backbone.features.17.conv.1.features.2"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.1.features.2"] = new_layer
        elif 'backbone.features.17.conv.2' == layer_name:
            self.backbone.features[17].conv[2] = new_layer
            self.layer_names["backbone.features.17.conv.2"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.2"] = new_layer
        elif 'backbone.features.17.conv.3' == layer_name:
            self.backbone.features[17].conv[3] = new_layer
            self.layer_names["backbone.features.17.conv.3"] = new_layer
            self.origin_layer_names["backbone.features.17.conv.3"] = new_layer
        elif 'backbone.features.18' == layer_name:
            self.backbone.features[18] = new_layer
            self.layer_names["backbone.features.18"] = new_layer
            self.origin_layer_names["backbone.features.18"] = new_layer
        elif 'backbone.features.18.features' == layer_name:
            self.backbone.features[18].features = new_layer
            self.layer_names["backbone.features.18.features"] = new_layer
            self.origin_layer_names["backbone.features.18.features"] = new_layer
        elif 'backbone.features.18.features.0' == layer_name:
            self.backbone.features[18].features[0] = new_layer
            self.layer_names["backbone.features.18.features.0"] = new_layer
            self.origin_layer_names["backbone.features.18.features.0"] = new_layer
        elif 'backbone.features.18.features.1' == layer_name:
            self.backbone.features[18].features[1] = new_layer
            self.layer_names["backbone.features.18.features.1"] = new_layer
            self.origin_layer_names["backbone.features.18.features.1"] = new_layer
        elif 'backbone.features.18.features.2' == layer_name:
            self.backbone.features[18].features[2] = new_layer
            self.layer_names["backbone.features.18.features.2"] = new_layer
            self.origin_layer_names["backbone.features.18.features.2"] = new_layer
        elif 'head' == layer_name:
            self.head = new_layer
            self.layer_names["head"] = new_layer
            self.origin_layer_names["head"] = new_layer
        elif 'head.head' == layer_name:
            self.head.head = new_layer
            self.layer_names["head.head"] = new_layer
            self.origin_layer_names["head.head"] = new_layer
        elif 'head.head.0' == layer_name:
            self.head.head[0] = new_layer
            self.layer_names["head.head.0"] = new_layer
            self.origin_layer_names["head.head.0"] = new_layer
        elif 'head.dense' == layer_name:
            self.head.dense = new_layer
            self.layer_names["head.dense"] = new_layer
            self.origin_layer_names["head.dense"] = new_layer

    def set_origin_layers(self, layer_name, new_layer):
        if 'backbone' == layer_name:
            self.backbone = new_layer
            self.origin_layer_names["backbone"] = new_layer
        elif 'backbone.features' == layer_name:
            self.backbone.features = new_layer
            self.origin_layer_names["backbone.features"] = new_layer
        elif 'backbone.features.0' == layer_name:
            self.backbone.features[0] = new_layer
            self.origin_layer_names["backbone.features.0"] = new_layer
        elif 'backbone.features.0.features' == layer_name:
            self.backbone.features[0].features = new_layer
            self.origin_layer_names["backbone.features.0.features"] = new_layer
        elif 'backbone.features.0.features.0' == layer_name:
            self.backbone.features[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.0.features.0"] = new_layer
        elif 'backbone.features.0.features.1' == layer_name:
            self.backbone.features[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.0.features.1"] = new_layer
        elif 'backbone.features.0.features.2' == layer_name:
            self.backbone.features[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.0.features.2"] = new_layer
        elif 'backbone.features.1' == layer_name:
            self.backbone.features[1] = new_layer
            self.origin_layer_names["backbone.features.1"] = new_layer
        elif 'backbone.features.1.conv' == layer_name:
            self.backbone.features[1].conv = new_layer
            self.origin_layer_names["backbone.features.1.conv"] = new_layer
        elif 'backbone.features.1.conv.0' == layer_name:
            self.backbone.features[1].conv[0] = new_layer
            self.origin_layer_names["backbone.features.1.conv.0"] = new_layer
        elif 'backbone.features.1.conv.0.features' == layer_name:
            self.backbone.features[1].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.1.conv.0.features"] = new_layer
        elif 'backbone.features.1.conv.0.features.0' == layer_name:
            self.backbone.features[1].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.1.conv.0.features.0"] = new_layer
        elif 'backbone.features.1.conv.0.features.1' == layer_name:
            self.backbone.features[1].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.1.conv.0.features.1"] = new_layer
        elif 'backbone.features.1.conv.0.features.2' == layer_name:
            self.backbone.features[1].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.1.conv.0.features.2"] = new_layer
        elif 'backbone.features.1.conv.1' == layer_name:
            self.backbone.features[1].conv[1] = new_layer
            self.origin_layer_names["backbone.features.1.conv.1"] = new_layer
        elif 'backbone.features.1.conv.2' == layer_name:
            self.backbone.features[1].conv[2] = new_layer
            self.origin_layer_names["backbone.features.1.conv.2"] = new_layer
        elif 'backbone.features.2' == layer_name:
            self.backbone.features[2] = new_layer
            self.origin_layer_names["backbone.features.2"] = new_layer
        elif 'backbone.features.2.conv' == layer_name:
            self.backbone.features[2].conv = new_layer
            self.origin_layer_names["backbone.features.2.conv"] = new_layer
        elif 'backbone.features.2.conv.0' == layer_name:
            self.backbone.features[2].conv[0] = new_layer
            self.origin_layer_names["backbone.features.2.conv.0"] = new_layer
        elif 'backbone.features.2.conv.0.features' == layer_name:
            self.backbone.features[2].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.2.conv.0.features"] = new_layer
        elif 'backbone.features.2.conv.0.features.0' == layer_name:
            self.backbone.features[2].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.2.conv.0.features.0"] = new_layer
        elif 'backbone.features.2.conv.0.features.1' == layer_name:
            self.backbone.features[2].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.2.conv.0.features.1"] = new_layer
        elif 'backbone.features.2.conv.0.features.2' == layer_name:
            self.backbone.features[2].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.2.conv.0.features.2"] = new_layer
        elif 'backbone.features.2.conv.1' == layer_name:
            self.backbone.features[2].conv[1] = new_layer
            self.origin_layer_names["backbone.features.2.conv.1"] = new_layer
        elif 'backbone.features.2.conv.1.features' == layer_name:
            self.backbone.features[2].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.2.conv.1.features"] = new_layer
        elif 'backbone.features.2.conv.1.features.0' == layer_name:
            self.backbone.features[2].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.2.conv.1.features.0"] = new_layer
        elif 'backbone.features.2.conv.1.features.1' == layer_name:
            self.backbone.features[2].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.2.conv.1.features.1"] = new_layer
        elif 'backbone.features.2.conv.1.features.2' == layer_name:
            self.backbone.features[2].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.2.conv.1.features.2"] = new_layer
        elif 'backbone.features.2.conv.2' == layer_name:
            self.backbone.features[2].conv[2] = new_layer
            self.origin_layer_names["backbone.features.2.conv.2"] = new_layer
        elif 'backbone.features.2.conv.3' == layer_name:
            self.backbone.features[2].conv[3] = new_layer
            self.origin_layer_names["backbone.features.2.conv.3"] = new_layer
        elif 'backbone.features.3' == layer_name:
            self.backbone.features[3] = new_layer
            self.origin_layer_names["backbone.features.3"] = new_layer
        elif 'backbone.features.3.conv' == layer_name:
            self.backbone.features[3].conv = new_layer
            self.origin_layer_names["backbone.features.3.conv"] = new_layer
        elif 'backbone.features.3.conv.0' == layer_name:
            self.backbone.features[3].conv[0] = new_layer
            self.origin_layer_names["backbone.features.3.conv.0"] = new_layer
        elif 'backbone.features.3.conv.0.features' == layer_name:
            self.backbone.features[3].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.3.conv.0.features"] = new_layer
        elif 'backbone.features.3.conv.0.features.0' == layer_name:
            self.backbone.features[3].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.3.conv.0.features.0"] = new_layer
        elif 'backbone.features.3.conv.0.features.1' == layer_name:
            self.backbone.features[3].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.3.conv.0.features.1"] = new_layer
        elif 'backbone.features.3.conv.0.features.2' == layer_name:
            self.backbone.features[3].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.3.conv.0.features.2"] = new_layer
        elif 'backbone.features.3.conv.1' == layer_name:
            self.backbone.features[3].conv[1] = new_layer
            self.origin_layer_names["backbone.features.3.conv.1"] = new_layer
        elif 'backbone.features.3.conv.1.features' == layer_name:
            self.backbone.features[3].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.3.conv.1.features"] = new_layer
        elif 'backbone.features.3.conv.1.features.0' == layer_name:
            self.backbone.features[3].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.3.conv.1.features.0"] = new_layer
        elif 'backbone.features.3.conv.1.features.1' == layer_name:
            self.backbone.features[3].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.3.conv.1.features.1"] = new_layer
        elif 'backbone.features.3.conv.1.features.2' == layer_name:
            self.backbone.features[3].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.3.conv.1.features.2"] = new_layer
        elif 'backbone.features.3.conv.2' == layer_name:
            self.backbone.features[3].conv[2] = new_layer
            self.origin_layer_names["backbone.features.3.conv.2"] = new_layer
        elif 'backbone.features.3.conv.3' == layer_name:
            self.backbone.features[3].conv[3] = new_layer
            self.origin_layer_names["backbone.features.3.conv.3"] = new_layer
        elif 'backbone.features.4' == layer_name:
            self.backbone.features[4] = new_layer
            self.origin_layer_names["backbone.features.4"] = new_layer
        elif 'backbone.features.4.conv' == layer_name:
            self.backbone.features[4].conv = new_layer
            self.origin_layer_names["backbone.features.4.conv"] = new_layer
        elif 'backbone.features.4.conv.0' == layer_name:
            self.backbone.features[4].conv[0] = new_layer
            self.origin_layer_names["backbone.features.4.conv.0"] = new_layer
        elif 'backbone.features.4.conv.0.features' == layer_name:
            self.backbone.features[4].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.4.conv.0.features"] = new_layer
        elif 'backbone.features.4.conv.0.features.0' == layer_name:
            self.backbone.features[4].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.4.conv.0.features.0"] = new_layer
        elif 'backbone.features.4.conv.0.features.1' == layer_name:
            self.backbone.features[4].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.4.conv.0.features.1"] = new_layer
        elif 'backbone.features.4.conv.0.features.2' == layer_name:
            self.backbone.features[4].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.4.conv.0.features.2"] = new_layer
        elif 'backbone.features.4.conv.1' == layer_name:
            self.backbone.features[4].conv[1] = new_layer
            self.origin_layer_names["backbone.features.4.conv.1"] = new_layer
        elif 'backbone.features.4.conv.1.features' == layer_name:
            self.backbone.features[4].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.4.conv.1.features"] = new_layer
        elif 'backbone.features.4.conv.1.features.0' == layer_name:
            self.backbone.features[4].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.4.conv.1.features.0"] = new_layer
        elif 'backbone.features.4.conv.1.features.1' == layer_name:
            self.backbone.features[4].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.4.conv.1.features.1"] = new_layer
        elif 'backbone.features.4.conv.1.features.2' == layer_name:
            self.backbone.features[4].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.4.conv.1.features.2"] = new_layer
        elif 'backbone.features.4.conv.2' == layer_name:
            self.backbone.features[4].conv[2] = new_layer
            self.origin_layer_names["backbone.features.4.conv.2"] = new_layer
        elif 'backbone.features.4.conv.3' == layer_name:
            self.backbone.features[4].conv[3] = new_layer
            self.origin_layer_names["backbone.features.4.conv.3"] = new_layer
        elif 'backbone.features.5' == layer_name:
            self.backbone.features[5] = new_layer
            self.origin_layer_names["backbone.features.5"] = new_layer
        elif 'backbone.features.5.conv' == layer_name:
            self.backbone.features[5].conv = new_layer
            self.origin_layer_names["backbone.features.5.conv"] = new_layer
        elif 'backbone.features.5.conv.0' == layer_name:
            self.backbone.features[5].conv[0] = new_layer
            self.origin_layer_names["backbone.features.5.conv.0"] = new_layer
        elif 'backbone.features.5.conv.0.features' == layer_name:
            self.backbone.features[5].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.5.conv.0.features"] = new_layer
        elif 'backbone.features.5.conv.0.features.0' == layer_name:
            self.backbone.features[5].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.5.conv.0.features.0"] = new_layer
        elif 'backbone.features.5.conv.0.features.1' == layer_name:
            self.backbone.features[5].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.5.conv.0.features.1"] = new_layer
        elif 'backbone.features.5.conv.0.features.2' == layer_name:
            self.backbone.features[5].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.5.conv.0.features.2"] = new_layer
        elif 'backbone.features.5.conv.1' == layer_name:
            self.backbone.features[5].conv[1] = new_layer
            self.origin_layer_names["backbone.features.5.conv.1"] = new_layer
        elif 'backbone.features.5.conv.1.features' == layer_name:
            self.backbone.features[5].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.5.conv.1.features"] = new_layer
        elif 'backbone.features.5.conv.1.features.0' == layer_name:
            self.backbone.features[5].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.5.conv.1.features.0"] = new_layer
        elif 'backbone.features.5.conv.1.features.1' == layer_name:
            self.backbone.features[5].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.5.conv.1.features.1"] = new_layer
        elif 'backbone.features.5.conv.1.features.2' == layer_name:
            self.backbone.features[5].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.5.conv.1.features.2"] = new_layer
        elif 'backbone.features.5.conv.2' == layer_name:
            self.backbone.features[5].conv[2] = new_layer
            self.origin_layer_names["backbone.features.5.conv.2"] = new_layer
        elif 'backbone.features.5.conv.3' == layer_name:
            self.backbone.features[5].conv[3] = new_layer
            self.origin_layer_names["backbone.features.5.conv.3"] = new_layer
        elif 'backbone.features.6' == layer_name:
            self.backbone.features[6] = new_layer
            self.origin_layer_names["backbone.features.6"] = new_layer
        elif 'backbone.features.6.conv' == layer_name:
            self.backbone.features[6].conv = new_layer
            self.origin_layer_names["backbone.features.6.conv"] = new_layer
        elif 'backbone.features.6.conv.0' == layer_name:
            self.backbone.features[6].conv[0] = new_layer
            self.origin_layer_names["backbone.features.6.conv.0"] = new_layer
        elif 'backbone.features.6.conv.0.features' == layer_name:
            self.backbone.features[6].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.6.conv.0.features"] = new_layer
        elif 'backbone.features.6.conv.0.features.0' == layer_name:
            self.backbone.features[6].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.6.conv.0.features.0"] = new_layer
        elif 'backbone.features.6.conv.0.features.1' == layer_name:
            self.backbone.features[6].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.6.conv.0.features.1"] = new_layer
        elif 'backbone.features.6.conv.0.features.2' == layer_name:
            self.backbone.features[6].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.6.conv.0.features.2"] = new_layer
        elif 'backbone.features.6.conv.1' == layer_name:
            self.backbone.features[6].conv[1] = new_layer
            self.origin_layer_names["backbone.features.6.conv.1"] = new_layer
        elif 'backbone.features.6.conv.1.features' == layer_name:
            self.backbone.features[6].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.6.conv.1.features"] = new_layer
        elif 'backbone.features.6.conv.1.features.0' == layer_name:
            self.backbone.features[6].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.6.conv.1.features.0"] = new_layer
        elif 'backbone.features.6.conv.1.features.1' == layer_name:
            self.backbone.features[6].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.6.conv.1.features.1"] = new_layer
        elif 'backbone.features.6.conv.1.features.2' == layer_name:
            self.backbone.features[6].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.6.conv.1.features.2"] = new_layer
        elif 'backbone.features.6.conv.2' == layer_name:
            self.backbone.features[6].conv[2] = new_layer
            self.origin_layer_names["backbone.features.6.conv.2"] = new_layer
        elif 'backbone.features.6.conv.3' == layer_name:
            self.backbone.features[6].conv[3] = new_layer
            self.origin_layer_names["backbone.features.6.conv.3"] = new_layer
        elif 'backbone.features.7' == layer_name:
            self.backbone.features[7] = new_layer
            self.origin_layer_names["backbone.features.7"] = new_layer
        elif 'backbone.features.7.conv' == layer_name:
            self.backbone.features[7].conv = new_layer
            self.origin_layer_names["backbone.features.7.conv"] = new_layer
        elif 'backbone.features.7.conv.0' == layer_name:
            self.backbone.features[7].conv[0] = new_layer
            self.origin_layer_names["backbone.features.7.conv.0"] = new_layer
        elif 'backbone.features.7.conv.0.features' == layer_name:
            self.backbone.features[7].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.7.conv.0.features"] = new_layer
        elif 'backbone.features.7.conv.0.features.0' == layer_name:
            self.backbone.features[7].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.7.conv.0.features.0"] = new_layer
        elif 'backbone.features.7.conv.0.features.1' == layer_name:
            self.backbone.features[7].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.7.conv.0.features.1"] = new_layer
        elif 'backbone.features.7.conv.0.features.2' == layer_name:
            self.backbone.features[7].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.7.conv.0.features.2"] = new_layer
        elif 'backbone.features.7.conv.1' == layer_name:
            self.backbone.features[7].conv[1] = new_layer
            self.origin_layer_names["backbone.features.7.conv.1"] = new_layer
        elif 'backbone.features.7.conv.1.features' == layer_name:
            self.backbone.features[7].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.7.conv.1.features"] = new_layer
        elif 'backbone.features.7.conv.1.features.0' == layer_name:
            self.backbone.features[7].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.7.conv.1.features.0"] = new_layer
        elif 'backbone.features.7.conv.1.features.1' == layer_name:
            self.backbone.features[7].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.7.conv.1.features.1"] = new_layer
        elif 'backbone.features.7.conv.1.features.2' == layer_name:
            self.backbone.features[7].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.7.conv.1.features.2"] = new_layer
        elif 'backbone.features.7.conv.2' == layer_name:
            self.backbone.features[7].conv[2] = new_layer
            self.origin_layer_names["backbone.features.7.conv.2"] = new_layer
        elif 'backbone.features.7.conv.3' == layer_name:
            self.backbone.features[7].conv[3] = new_layer
            self.origin_layer_names["backbone.features.7.conv.3"] = new_layer
        elif 'backbone.features.8' == layer_name:
            self.backbone.features[8] = new_layer
            self.origin_layer_names["backbone.features.8"] = new_layer
        elif 'backbone.features.8.conv' == layer_name:
            self.backbone.features[8].conv = new_layer
            self.origin_layer_names["backbone.features.8.conv"] = new_layer
        elif 'backbone.features.8.conv.0' == layer_name:
            self.backbone.features[8].conv[0] = new_layer
            self.origin_layer_names["backbone.features.8.conv.0"] = new_layer
        elif 'backbone.features.8.conv.0.features' == layer_name:
            self.backbone.features[8].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.8.conv.0.features"] = new_layer
        elif 'backbone.features.8.conv.0.features.0' == layer_name:
            self.backbone.features[8].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.8.conv.0.features.0"] = new_layer
        elif 'backbone.features.8.conv.0.features.1' == layer_name:
            self.backbone.features[8].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.8.conv.0.features.1"] = new_layer
        elif 'backbone.features.8.conv.0.features.2' == layer_name:
            self.backbone.features[8].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.8.conv.0.features.2"] = new_layer
        elif 'backbone.features.8.conv.1' == layer_name:
            self.backbone.features[8].conv[1] = new_layer
            self.origin_layer_names["backbone.features.8.conv.1"] = new_layer
        elif 'backbone.features.8.conv.1.features' == layer_name:
            self.backbone.features[8].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.8.conv.1.features"] = new_layer
        elif 'backbone.features.8.conv.1.features.0' == layer_name:
            self.backbone.features[8].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.8.conv.1.features.0"] = new_layer
        elif 'backbone.features.8.conv.1.features.1' == layer_name:
            self.backbone.features[8].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.8.conv.1.features.1"] = new_layer
        elif 'backbone.features.8.conv.1.features.2' == layer_name:
            self.backbone.features[8].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.8.conv.1.features.2"] = new_layer
        elif 'backbone.features.8.conv.2' == layer_name:
            self.backbone.features[8].conv[2] = new_layer
            self.origin_layer_names["backbone.features.8.conv.2"] = new_layer
        elif 'backbone.features.8.conv.3' == layer_name:
            self.backbone.features[8].conv[3] = new_layer
            self.origin_layer_names["backbone.features.8.conv.3"] = new_layer
        elif 'backbone.features.9' == layer_name:
            self.backbone.features[9] = new_layer
            self.origin_layer_names["backbone.features.9"] = new_layer
        elif 'backbone.features.9.conv' == layer_name:
            self.backbone.features[9].conv = new_layer
            self.origin_layer_names["backbone.features.9.conv"] = new_layer
        elif 'backbone.features.9.conv.0' == layer_name:
            self.backbone.features[9].conv[0] = new_layer
            self.origin_layer_names["backbone.features.9.conv.0"] = new_layer
        elif 'backbone.features.9.conv.0.features' == layer_name:
            self.backbone.features[9].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.9.conv.0.features"] = new_layer
        elif 'backbone.features.9.conv.0.features.0' == layer_name:
            self.backbone.features[9].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.9.conv.0.features.0"] = new_layer
        elif 'backbone.features.9.conv.0.features.1' == layer_name:
            self.backbone.features[9].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.9.conv.0.features.1"] = new_layer
        elif 'backbone.features.9.conv.0.features.2' == layer_name:
            self.backbone.features[9].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.9.conv.0.features.2"] = new_layer
        elif 'backbone.features.9.conv.1' == layer_name:
            self.backbone.features[9].conv[1] = new_layer
            self.origin_layer_names["backbone.features.9.conv.1"] = new_layer
        elif 'backbone.features.9.conv.1.features' == layer_name:
            self.backbone.features[9].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.9.conv.1.features"] = new_layer
        elif 'backbone.features.9.conv.1.features.0' == layer_name:
            self.backbone.features[9].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.9.conv.1.features.0"] = new_layer
        elif 'backbone.features.9.conv.1.features.1' == layer_name:
            self.backbone.features[9].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.9.conv.1.features.1"] = new_layer
        elif 'backbone.features.9.conv.1.features.2' == layer_name:
            self.backbone.features[9].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.9.conv.1.features.2"] = new_layer
        elif 'backbone.features.9.conv.2' == layer_name:
            self.backbone.features[9].conv[2] = new_layer
            self.origin_layer_names["backbone.features.9.conv.2"] = new_layer
        elif 'backbone.features.9.conv.3' == layer_name:
            self.backbone.features[9].conv[3] = new_layer
            self.origin_layer_names["backbone.features.9.conv.3"] = new_layer
        elif 'backbone.features.10' == layer_name:
            self.backbone.features[10] = new_layer
            self.origin_layer_names["backbone.features.10"] = new_layer
        elif 'backbone.features.10.conv' == layer_name:
            self.backbone.features[10].conv = new_layer
            self.origin_layer_names["backbone.features.10.conv"] = new_layer
        elif 'backbone.features.10.conv.0' == layer_name:
            self.backbone.features[10].conv[0] = new_layer
            self.origin_layer_names["backbone.features.10.conv.0"] = new_layer
        elif 'backbone.features.10.conv.0.features' == layer_name:
            self.backbone.features[10].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.10.conv.0.features"] = new_layer
        elif 'backbone.features.10.conv.0.features.0' == layer_name:
            self.backbone.features[10].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.10.conv.0.features.0"] = new_layer
        elif 'backbone.features.10.conv.0.features.1' == layer_name:
            self.backbone.features[10].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.10.conv.0.features.1"] = new_layer
        elif 'backbone.features.10.conv.0.features.2' == layer_name:
            self.backbone.features[10].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.10.conv.0.features.2"] = new_layer
        elif 'backbone.features.10.conv.1' == layer_name:
            self.backbone.features[10].conv[1] = new_layer
            self.origin_layer_names["backbone.features.10.conv.1"] = new_layer
        elif 'backbone.features.10.conv.1.features' == layer_name:
            self.backbone.features[10].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.10.conv.1.features"] = new_layer
        elif 'backbone.features.10.conv.1.features.0' == layer_name:
            self.backbone.features[10].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.10.conv.1.features.0"] = new_layer
        elif 'backbone.features.10.conv.1.features.1' == layer_name:
            self.backbone.features[10].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.10.conv.1.features.1"] = new_layer
        elif 'backbone.features.10.conv.1.features.2' == layer_name:
            self.backbone.features[10].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.10.conv.1.features.2"] = new_layer
        elif 'backbone.features.10.conv.2' == layer_name:
            self.backbone.features[10].conv[2] = new_layer
            self.origin_layer_names["backbone.features.10.conv.2"] = new_layer
        elif 'backbone.features.10.conv.3' == layer_name:
            self.backbone.features[10].conv[3] = new_layer
            self.origin_layer_names["backbone.features.10.conv.3"] = new_layer
        elif 'backbone.features.11' == layer_name:
            self.backbone.features[11] = new_layer
            self.origin_layer_names["backbone.features.11"] = new_layer
        elif 'backbone.features.11.conv' == layer_name:
            self.backbone.features[11].conv = new_layer
            self.origin_layer_names["backbone.features.11.conv"] = new_layer
        elif 'backbone.features.11.conv.0' == layer_name:
            self.backbone.features[11].conv[0] = new_layer
            self.origin_layer_names["backbone.features.11.conv.0"] = new_layer
        elif 'backbone.features.11.conv.0.features' == layer_name:
            self.backbone.features[11].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.11.conv.0.features"] = new_layer
        elif 'backbone.features.11.conv.0.features.0' == layer_name:
            self.backbone.features[11].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.11.conv.0.features.0"] = new_layer
        elif 'backbone.features.11.conv.0.features.1' == layer_name:
            self.backbone.features[11].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.11.conv.0.features.1"] = new_layer
        elif 'backbone.features.11.conv.0.features.2' == layer_name:
            self.backbone.features[11].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.11.conv.0.features.2"] = new_layer
        elif 'backbone.features.11.conv.1' == layer_name:
            self.backbone.features[11].conv[1] = new_layer
            self.origin_layer_names["backbone.features.11.conv.1"] = new_layer
        elif 'backbone.features.11.conv.1.features' == layer_name:
            self.backbone.features[11].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.11.conv.1.features"] = new_layer
        elif 'backbone.features.11.conv.1.features.0' == layer_name:
            self.backbone.features[11].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.11.conv.1.features.0"] = new_layer
        elif 'backbone.features.11.conv.1.features.1' == layer_name:
            self.backbone.features[11].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.11.conv.1.features.1"] = new_layer
        elif 'backbone.features.11.conv.1.features.2' == layer_name:
            self.backbone.features[11].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.11.conv.1.features.2"] = new_layer
        elif 'backbone.features.11.conv.2' == layer_name:
            self.backbone.features[11].conv[2] = new_layer
            self.origin_layer_names["backbone.features.11.conv.2"] = new_layer
        elif 'backbone.features.11.conv.3' == layer_name:
            self.backbone.features[11].conv[3] = new_layer
            self.origin_layer_names["backbone.features.11.conv.3"] = new_layer
        elif 'backbone.features.12' == layer_name:
            self.backbone.features[12] = new_layer
            self.origin_layer_names["backbone.features.12"] = new_layer
        elif 'backbone.features.12.conv' == layer_name:
            self.backbone.features[12].conv = new_layer
            self.origin_layer_names["backbone.features.12.conv"] = new_layer
        elif 'backbone.features.12.conv.0' == layer_name:
            self.backbone.features[12].conv[0] = new_layer
            self.origin_layer_names["backbone.features.12.conv.0"] = new_layer
        elif 'backbone.features.12.conv.0.features' == layer_name:
            self.backbone.features[12].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.12.conv.0.features"] = new_layer
        elif 'backbone.features.12.conv.0.features.0' == layer_name:
            self.backbone.features[12].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.12.conv.0.features.0"] = new_layer
        elif 'backbone.features.12.conv.0.features.1' == layer_name:
            self.backbone.features[12].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.12.conv.0.features.1"] = new_layer
        elif 'backbone.features.12.conv.0.features.2' == layer_name:
            self.backbone.features[12].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.12.conv.0.features.2"] = new_layer
        elif 'backbone.features.12.conv.1' == layer_name:
            self.backbone.features[12].conv[1] = new_layer
            self.origin_layer_names["backbone.features.12.conv.1"] = new_layer
        elif 'backbone.features.12.conv.1.features' == layer_name:
            self.backbone.features[12].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.12.conv.1.features"] = new_layer
        elif 'backbone.features.12.conv.1.features.0' == layer_name:
            self.backbone.features[12].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.12.conv.1.features.0"] = new_layer
        elif 'backbone.features.12.conv.1.features.1' == layer_name:
            self.backbone.features[12].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.12.conv.1.features.1"] = new_layer
        elif 'backbone.features.12.conv.1.features.2' == layer_name:
            self.backbone.features[12].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.12.conv.1.features.2"] = new_layer
        elif 'backbone.features.12.conv.2' == layer_name:
            self.backbone.features[12].conv[2] = new_layer
            self.origin_layer_names["backbone.features.12.conv.2"] = new_layer
        elif 'backbone.features.12.conv.3' == layer_name:
            self.backbone.features[12].conv[3] = new_layer
            self.origin_layer_names["backbone.features.12.conv.3"] = new_layer
        elif 'backbone.features.13' == layer_name:
            self.backbone.features[13] = new_layer
            self.origin_layer_names["backbone.features.13"] = new_layer
        elif 'backbone.features.13.conv' == layer_name:
            self.backbone.features[13].conv = new_layer
            self.origin_layer_names["backbone.features.13.conv"] = new_layer
        elif 'backbone.features.13.conv.0' == layer_name:
            self.backbone.features[13].conv[0] = new_layer
            self.origin_layer_names["backbone.features.13.conv.0"] = new_layer
        elif 'backbone.features.13.conv.0.features' == layer_name:
            self.backbone.features[13].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.13.conv.0.features"] = new_layer
        elif 'backbone.features.13.conv.0.features.0' == layer_name:
            self.backbone.features[13].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.13.conv.0.features.0"] = new_layer
        elif 'backbone.features.13.conv.0.features.1' == layer_name:
            self.backbone.features[13].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.13.conv.0.features.1"] = new_layer
        elif 'backbone.features.13.conv.0.features.2' == layer_name:
            self.backbone.features[13].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.13.conv.0.features.2"] = new_layer
        elif 'backbone.features.13.conv.1' == layer_name:
            self.backbone.features[13].conv[1] = new_layer
            self.origin_layer_names["backbone.features.13.conv.1"] = new_layer
        elif 'backbone.features.13.conv.1.features' == layer_name:
            self.backbone.features[13].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.13.conv.1.features"] = new_layer
        elif 'backbone.features.13.conv.1.features.0' == layer_name:
            self.backbone.features[13].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.13.conv.1.features.0"] = new_layer
        elif 'backbone.features.13.conv.1.features.1' == layer_name:
            self.backbone.features[13].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.13.conv.1.features.1"] = new_layer
        elif 'backbone.features.13.conv.1.features.2' == layer_name:
            self.backbone.features[13].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.13.conv.1.features.2"] = new_layer
        elif 'backbone.features.13.conv.2' == layer_name:
            self.backbone.features[13].conv[2] = new_layer
            self.origin_layer_names["backbone.features.13.conv.2"] = new_layer
        elif 'backbone.features.13.conv.3' == layer_name:
            self.backbone.features[13].conv[3] = new_layer
            self.origin_layer_names["backbone.features.13.conv.3"] = new_layer
        elif 'backbone.features.14' == layer_name:
            self.backbone.features[14] = new_layer
            self.origin_layer_names["backbone.features.14"] = new_layer
        elif 'backbone.features.14.conv' == layer_name:
            self.backbone.features[14].conv = new_layer
            self.origin_layer_names["backbone.features.14.conv"] = new_layer
        elif 'backbone.features.14.conv.0' == layer_name:
            self.backbone.features[14].conv[0] = new_layer
            self.origin_layer_names["backbone.features.14.conv.0"] = new_layer
        elif 'backbone.features.14.conv.0.features' == layer_name:
            self.backbone.features[14].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.14.conv.0.features"] = new_layer
        elif 'backbone.features.14.conv.0.features.0' == layer_name:
            self.backbone.features[14].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.14.conv.0.features.0"] = new_layer
        elif 'backbone.features.14.conv.0.features.1' == layer_name:
            self.backbone.features[14].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.14.conv.0.features.1"] = new_layer
        elif 'backbone.features.14.conv.0.features.2' == layer_name:
            self.backbone.features[14].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.14.conv.0.features.2"] = new_layer
        elif 'backbone.features.14.conv.1' == layer_name:
            self.backbone.features[14].conv[1] = new_layer
            self.origin_layer_names["backbone.features.14.conv.1"] = new_layer
        elif 'backbone.features.14.conv.1.features' == layer_name:
            self.backbone.features[14].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.14.conv.1.features"] = new_layer
        elif 'backbone.features.14.conv.1.features.0' == layer_name:
            self.backbone.features[14].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.14.conv.1.features.0"] = new_layer
        elif 'backbone.features.14.conv.1.features.1' == layer_name:
            self.backbone.features[14].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.14.conv.1.features.1"] = new_layer
        elif 'backbone.features.14.conv.1.features.2' == layer_name:
            self.backbone.features[14].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.14.conv.1.features.2"] = new_layer
        elif 'backbone.features.14.conv.2' == layer_name:
            self.backbone.features[14].conv[2] = new_layer
            self.origin_layer_names["backbone.features.14.conv.2"] = new_layer
        elif 'backbone.features.14.conv.3' == layer_name:
            self.backbone.features[14].conv[3] = new_layer
            self.origin_layer_names["backbone.features.14.conv.3"] = new_layer
        elif 'backbone.features.15' == layer_name:
            self.backbone.features[15] = new_layer
            self.origin_layer_names["backbone.features.15"] = new_layer
        elif 'backbone.features.15.conv' == layer_name:
            self.backbone.features[15].conv = new_layer
            self.origin_layer_names["backbone.features.15.conv"] = new_layer
        elif 'backbone.features.15.conv.0' == layer_name:
            self.backbone.features[15].conv[0] = new_layer
            self.origin_layer_names["backbone.features.15.conv.0"] = new_layer
        elif 'backbone.features.15.conv.0.features' == layer_name:
            self.backbone.features[15].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.15.conv.0.features"] = new_layer
        elif 'backbone.features.15.conv.0.features.0' == layer_name:
            self.backbone.features[15].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.15.conv.0.features.0"] = new_layer
        elif 'backbone.features.15.conv.0.features.1' == layer_name:
            self.backbone.features[15].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.15.conv.0.features.1"] = new_layer
        elif 'backbone.features.15.conv.0.features.2' == layer_name:
            self.backbone.features[15].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.15.conv.0.features.2"] = new_layer
        elif 'backbone.features.15.conv.1' == layer_name:
            self.backbone.features[15].conv[1] = new_layer
            self.origin_layer_names["backbone.features.15.conv.1"] = new_layer
        elif 'backbone.features.15.conv.1.features' == layer_name:
            self.backbone.features[15].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.15.conv.1.features"] = new_layer
        elif 'backbone.features.15.conv.1.features.0' == layer_name:
            self.backbone.features[15].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.15.conv.1.features.0"] = new_layer
        elif 'backbone.features.15.conv.1.features.1' == layer_name:
            self.backbone.features[15].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.15.conv.1.features.1"] = new_layer
        elif 'backbone.features.15.conv.1.features.2' == layer_name:
            self.backbone.features[15].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.15.conv.1.features.2"] = new_layer
        elif 'backbone.features.15.conv.2' == layer_name:
            self.backbone.features[15].conv[2] = new_layer
            self.origin_layer_names["backbone.features.15.conv.2"] = new_layer
        elif 'backbone.features.15.conv.3' == layer_name:
            self.backbone.features[15].conv[3] = new_layer
            self.origin_layer_names["backbone.features.15.conv.3"] = new_layer
        elif 'backbone.features.16' == layer_name:
            self.backbone.features[16] = new_layer
            self.origin_layer_names["backbone.features.16"] = new_layer
        elif 'backbone.features.16.conv' == layer_name:
            self.backbone.features[16].conv = new_layer
            self.origin_layer_names["backbone.features.16.conv"] = new_layer
        elif 'backbone.features.16.conv.0' == layer_name:
            self.backbone.features[16].conv[0] = new_layer
            self.origin_layer_names["backbone.features.16.conv.0"] = new_layer
        elif 'backbone.features.16.conv.0.features' == layer_name:
            self.backbone.features[16].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.16.conv.0.features"] = new_layer
        elif 'backbone.features.16.conv.0.features.0' == layer_name:
            self.backbone.features[16].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.16.conv.0.features.0"] = new_layer
        elif 'backbone.features.16.conv.0.features.1' == layer_name:
            self.backbone.features[16].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.16.conv.0.features.1"] = new_layer
        elif 'backbone.features.16.conv.0.features.2' == layer_name:
            self.backbone.features[16].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.16.conv.0.features.2"] = new_layer
        elif 'backbone.features.16.conv.1' == layer_name:
            self.backbone.features[16].conv[1] = new_layer
            self.origin_layer_names["backbone.features.16.conv.1"] = new_layer
        elif 'backbone.features.16.conv.1.features' == layer_name:
            self.backbone.features[16].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.16.conv.1.features"] = new_layer
        elif 'backbone.features.16.conv.1.features.0' == layer_name:
            self.backbone.features[16].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.16.conv.1.features.0"] = new_layer
        elif 'backbone.features.16.conv.1.features.1' == layer_name:
            self.backbone.features[16].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.16.conv.1.features.1"] = new_layer
        elif 'backbone.features.16.conv.1.features.2' == layer_name:
            self.backbone.features[16].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.16.conv.1.features.2"] = new_layer
        elif 'backbone.features.16.conv.2' == layer_name:
            self.backbone.features[16].conv[2] = new_layer
            self.origin_layer_names["backbone.features.16.conv.2"] = new_layer
        elif 'backbone.features.16.conv.3' == layer_name:
            self.backbone.features[16].conv[3] = new_layer
            self.origin_layer_names["backbone.features.16.conv.3"] = new_layer
        elif 'backbone.features.17' == layer_name:
            self.backbone.features[17] = new_layer
            self.origin_layer_names["backbone.features.17"] = new_layer
        elif 'backbone.features.17.conv' == layer_name:
            self.backbone.features[17].conv = new_layer
            self.origin_layer_names["backbone.features.17.conv"] = new_layer
        elif 'backbone.features.17.conv.0' == layer_name:
            self.backbone.features[17].conv[0] = new_layer
            self.origin_layer_names["backbone.features.17.conv.0"] = new_layer
        elif 'backbone.features.17.conv.0.features' == layer_name:
            self.backbone.features[17].conv[0].features = new_layer
            self.origin_layer_names["backbone.features.17.conv.0.features"] = new_layer
        elif 'backbone.features.17.conv.0.features.0' == layer_name:
            self.backbone.features[17].conv[0].features[0] = new_layer
            self.origin_layer_names["backbone.features.17.conv.0.features.0"] = new_layer
        elif 'backbone.features.17.conv.0.features.1' == layer_name:
            self.backbone.features[17].conv[0].features[1] = new_layer
            self.origin_layer_names["backbone.features.17.conv.0.features.1"] = new_layer
        elif 'backbone.features.17.conv.0.features.2' == layer_name:
            self.backbone.features[17].conv[0].features[2] = new_layer
            self.origin_layer_names["backbone.features.17.conv.0.features.2"] = new_layer
        elif 'backbone.features.17.conv.1' == layer_name:
            self.backbone.features[17].conv[1] = new_layer
            self.origin_layer_names["backbone.features.17.conv.1"] = new_layer
        elif 'backbone.features.17.conv.1.features' == layer_name:
            self.backbone.features[17].conv[1].features = new_layer
            self.origin_layer_names["backbone.features.17.conv.1.features"] = new_layer
        elif 'backbone.features.17.conv.1.features.0' == layer_name:
            self.backbone.features[17].conv[1].features[0] = new_layer
            self.origin_layer_names["backbone.features.17.conv.1.features.0"] = new_layer
        elif 'backbone.features.17.conv.1.features.1' == layer_name:
            self.backbone.features[17].conv[1].features[1] = new_layer
            self.origin_layer_names["backbone.features.17.conv.1.features.1"] = new_layer
        elif 'backbone.features.17.conv.1.features.2' == layer_name:
            self.backbone.features[17].conv[1].features[2] = new_layer
            self.origin_layer_names["backbone.features.17.conv.1.features.2"] = new_layer
        elif 'backbone.features.17.conv.2' == layer_name:
            self.backbone.features[17].conv[2] = new_layer
            self.origin_layer_names["backbone.features.17.conv.2"] = new_layer
        elif 'backbone.features.17.conv.3' == layer_name:
            self.backbone.features[17].conv[3] = new_layer
            self.origin_layer_names["backbone.features.17.conv.3"] = new_layer
        elif 'backbone.features.18' == layer_name:
            self.backbone.features[18] = new_layer
            self.origin_layer_names["backbone.features.18"] = new_layer
        elif 'backbone.features.18.features' == layer_name:
            self.backbone.features[18].features = new_layer
            self.origin_layer_names["backbone.features.18.features"] = new_layer
        elif 'backbone.features.18.features.0' == layer_name:
            self.backbone.features[18].features[0] = new_layer
            self.origin_layer_names["backbone.features.18.features.0"] = new_layer
        elif 'backbone.features.18.features.1' == layer_name:
            self.backbone.features[18].features[1] = new_layer
            self.origin_layer_names["backbone.features.18.features.1"] = new_layer
        elif 'backbone.features.18.features.2' == layer_name:
            self.backbone.features[18].features[2] = new_layer
            self.origin_layer_names["backbone.features.18.features.2"] = new_layer
        elif 'head' == layer_name:
            self.head = new_layer
            self.origin_layer_names["head"] = new_layer
        elif 'head.head' == layer_name:
            self.head.head = new_layer
            self.origin_layer_names["head.head"] = new_layer
        elif 'head.head.0' == layer_name:
            self.head.head[0] = new_layer
            self.origin_layer_names["head.head.0"] = new_layer
        elif 'head.dense' == layer_name:
            self.head.dense = new_layer
            self.origin_layer_names["head.dense"] = new_layer

    def get_order(self, layer_name):
        if layer_name not in self.orders.keys():
            return False
        return self.orders[layer_name]

    def set_order(self, layer_name, order):
        if layer_name not in self.orders.keys():
            return False
        self.orders[layer_name] = order

    def get_outshape(self, layer_name):

        if layer_name not in self.out_shapes.keys():
            return False

        return self.out_shapes[layer_name]

    def set_outshape(self, layer_name, out):

        if layer_name not in self.out_shapes.keys():
            return False

        self.out_shapes[layer_name] = out

    def get_inshape(self, layer_name):

        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name, out):

        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name] = out

    def set_Basic_OPS(self, b):
        self.Basic_OPS = b

    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self, c):
        self.Cascade_OPs = c


def mobilenet_v2_torch():
    backbone_net = MobileNetV2Backbone()
    activation = "Softmax"
    head_net = MobileNetV2Head(input_channel=backbone_net.out_channels,
                               num_classes=10,
                               activation=activation)
    net = MobileNetV2Combine(backbone_net, head_net)

    return net
