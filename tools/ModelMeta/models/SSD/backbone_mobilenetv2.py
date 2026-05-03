import mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from ssd_utils import MultiBox


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.97,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


class ConvBNReLU(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, shared_conv=None):
        super(ConvBNReLU, self).__init__()
        padding = 0
        in_channels = in_planes
        out_channels = out_planes
        if shared_conv is None:
            if groups == 1:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same', padding=padding)
            else:
                out_channels = in_planes
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same',
                                 padding=padding, group=in_channels)
            layers = [conv, _bn(out_planes), nn.ReLU6()]
        else:
            layers = [shared_conv, _bn(out_planes), nn.ReLU6()]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class InvertedResidual(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio, last_relu=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
            _bn(oup),
        ])
        self.conv = nn.SequentialCell(layers)
        self.cast = ops.Cast()
        self.last_relu = last_relu
        self.relu = nn.ReLU6()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            x = identity + x
        if self.last_relu:
            x = self.relu(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    """nsures that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v = new_v + divisor
    return new_v


class SSDWithMobileNetV2(nn.Cell):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(SSDWithMobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        if len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        layer_index = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if layer_index == 13:
                    hidden_dim = int(round(input_channel * t))
                    self.expand_layer_conv_13 = ConvBNReLU(input_channel, hidden_dim, kernel_size=1)
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                layer_index = layer_index + 1
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        self.features_1 = nn.SequentialCell(features[:14])
        self.features_2 = nn.SequentialCell(features[14:])

        # after mobilenetv2 backbone
        in_channels = [256, 576, 1280, 512, 256, 256]
        out_channels = [576, 1280, 512, 256, 256, 128]
        ratios = [0.2, 0.2, 0.2, 0.25, 0.5, 0.25]
        strides = [1, 1, 2, 2, 2, 2]
        residual_list = []
        for i in range(2, len(in_channels)):
            residual = InvertedResidual(in_channels[i], out_channels[i], stride=strides[i],
                                        expand_ratio=ratios[i], last_relu=True)
            residual_list.append(residual)
        self.multi_residual = nn.layer.CellList(residual_list)

        self.multi_box = MultiBox(81, [576, 1280, 512, 256, 256, 128], [3, 6, 6, 6, 6, 6], 1917)

        self.layer_names = {
            "expand_layer_conv_13": self.expand_layer_conv_13,
            "expand_layer_conv_13.features": self.expand_layer_conv_13.features,
            "expand_layer_conv_13.features.0": self.expand_layer_conv_13.features[0],
            "expand_layer_conv_13.features.1": self.expand_layer_conv_13.features[1],
            "expand_layer_conv_13.features.2": self.expand_layer_conv_13.features[2],
            "features_1": self.features_1,
            "features_1.0": self.features_1[0],
            "features_1.0.features": self.features_1[0].features,
            "features_1.0.features.0": self.features_1[0].features[0],
            "features_1.0.features.1": self.features_1[0].features[1],
            "features_1.0.features.2": self.features_1[0].features[2],
            "features_1.1": self.features_1[1],
            "features_1.1.conv": self.features_1[1].conv,
            "features_1.1.conv.0": self.features_1[1].conv[0],
            "features_1.1.conv.0.features": self.features_1[1].conv[0].features,
            "features_1.1.conv.0.features.0": self.features_1[1].conv[0].features[0],
            "features_1.1.conv.0.features.1": self.features_1[1].conv[0].features[1],
            "features_1.1.conv.0.features.2": self.features_1[1].conv[0].features[2],
            "features_1.1.conv.1": self.features_1[1].conv[1],
            "features_1.1.conv.2": self.features_1[1].conv[2],
            "features_1.2": self.features_1[2],
            "features_1.2.conv": self.features_1[2].conv,
            "features_1.2.conv.0": self.features_1[2].conv[0],
            "features_1.2.conv.0.features": self.features_1[2].conv[0].features,
            "features_1.2.conv.0.features.0": self.features_1[2].conv[0].features[0],
            "features_1.2.conv.0.features.1": self.features_1[2].conv[0].features[1],
            "features_1.2.conv.0.features.2": self.features_1[2].conv[0].features[2],
            "features_1.2.conv.1": self.features_1[2].conv[1],
            "features_1.2.conv.1.features": self.features_1[2].conv[1].features,
            "features_1.2.conv.1.features.0": self.features_1[2].conv[1].features[0],
            "features_1.2.conv.1.features.1": self.features_1[2].conv[1].features[1],
            "features_1.2.conv.1.features.2": self.features_1[2].conv[1].features[2],
            "features_1.2.conv.2": self.features_1[2].conv[2],
            "features_1.2.conv.3": self.features_1[2].conv[3],
            "features_1.3": self.features_1[3],
            "features_1.3.conv": self.features_1[3].conv,
            "features_1.3.conv.0": self.features_1[3].conv[0],
            "features_1.3.conv.0.features": self.features_1[3].conv[0].features,
            "features_1.3.conv.0.features.0": self.features_1[3].conv[0].features[0],
            "features_1.3.conv.0.features.1": self.features_1[3].conv[0].features[1],
            "features_1.3.conv.0.features.2": self.features_1[3].conv[0].features[2],
            "features_1.3.conv.1": self.features_1[3].conv[1],
            "features_1.3.conv.1.features": self.features_1[3].conv[1].features,
            "features_1.3.conv.1.features.0": self.features_1[3].conv[1].features[0],
            "features_1.3.conv.1.features.1": self.features_1[3].conv[1].features[1],
            "features_1.3.conv.1.features.2": self.features_1[3].conv[1].features[2],
            "features_1.3.conv.2": self.features_1[3].conv[2],
            "features_1.3.conv.3": self.features_1[3].conv[3],
            "features_1.4": self.features_1[4],
            "features_1.4.conv": self.features_1[4].conv,
            "features_1.4.conv.0": self.features_1[4].conv[0],
            "features_1.4.conv.0.features": self.features_1[4].conv[0].features,
            "features_1.4.conv.0.features.0": self.features_1[4].conv[0].features[0],
            "features_1.4.conv.0.features.1": self.features_1[4].conv[0].features[1],
            "features_1.4.conv.0.features.2": self.features_1[4].conv[0].features[2],
            "features_1.4.conv.1": self.features_1[4].conv[1],
            "features_1.4.conv.1.features": self.features_1[4].conv[1].features,
            "features_1.4.conv.1.features.0": self.features_1[4].conv[1].features[0],
            "features_1.4.conv.1.features.1": self.features_1[4].conv[1].features[1],
            "features_1.4.conv.1.features.2": self.features_1[4].conv[1].features[2],
            "features_1.4.conv.2": self.features_1[4].conv[2],
            "features_1.4.conv.3": self.features_1[4].conv[3],
            "features_1.5": self.features_1[5],
            "features_1.5.conv": self.features_1[5].conv,
            "features_1.5.conv.0": self.features_1[5].conv[0],
            "features_1.5.conv.0.features": self.features_1[5].conv[0].features,
            "features_1.5.conv.0.features.0": self.features_1[5].conv[0].features[0],
            "features_1.5.conv.0.features.1": self.features_1[5].conv[0].features[1],
            "features_1.5.conv.0.features.2": self.features_1[5].conv[0].features[2],
            "features_1.5.conv.1": self.features_1[5].conv[1],
            "features_1.5.conv.1.features": self.features_1[5].conv[1].features,
            "features_1.5.conv.1.features.0": self.features_1[5].conv[1].features[0],
            "features_1.5.conv.1.features.1": self.features_1[5].conv[1].features[1],
            "features_1.5.conv.1.features.2": self.features_1[5].conv[1].features[2],
            "features_1.5.conv.2": self.features_1[5].conv[2],
            "features_1.5.conv.3": self.features_1[5].conv[3],
            "features_1.6": self.features_1[6],
            "features_1.6.conv": self.features_1[6].conv,
            "features_1.6.conv.0": self.features_1[6].conv[0],
            "features_1.6.conv.0.features": self.features_1[6].conv[0].features,
            "features_1.6.conv.0.features.0": self.features_1[6].conv[0].features[0],
            "features_1.6.conv.0.features.1": self.features_1[6].conv[0].features[1],
            "features_1.6.conv.0.features.2": self.features_1[6].conv[0].features[2],
            "features_1.6.conv.1": self.features_1[6].conv[1],
            "features_1.6.conv.1.features": self.features_1[6].conv[1].features,
            "features_1.6.conv.1.features.0": self.features_1[6].conv[1].features[0],
            "features_1.6.conv.1.features.1": self.features_1[6].conv[1].features[1],
            "features_1.6.conv.1.features.2": self.features_1[6].conv[1].features[2],
            "features_1.6.conv.2": self.features_1[6].conv[2],
            "features_1.6.conv.3": self.features_1[6].conv[3],
            "features_1.7": self.features_1[7],
            "features_1.7.conv": self.features_1[7].conv,
            "features_1.7.conv.0": self.features_1[7].conv[0],
            "features_1.7.conv.0.features": self.features_1[7].conv[0].features,
            "features_1.7.conv.0.features.0": self.features_1[7].conv[0].features[0],
            "features_1.7.conv.0.features.1": self.features_1[7].conv[0].features[1],
            "features_1.7.conv.0.features.2": self.features_1[7].conv[0].features[2],
            "features_1.7.conv.1": self.features_1[7].conv[1],
            "features_1.7.conv.1.features": self.features_1[7].conv[1].features,
            "features_1.7.conv.1.features.0": self.features_1[7].conv[1].features[0],
            "features_1.7.conv.1.features.1": self.features_1[7].conv[1].features[1],
            "features_1.7.conv.1.features.2": self.features_1[7].conv[1].features[2],
            "features_1.7.conv.2": self.features_1[7].conv[2],
            "features_1.7.conv.3": self.features_1[7].conv[3],
            "features_1.8": self.features_1[8],
            "features_1.8.conv": self.features_1[8].conv,
            "features_1.8.conv.0": self.features_1[8].conv[0],
            "features_1.8.conv.0.features": self.features_1[8].conv[0].features,
            "features_1.8.conv.0.features.0": self.features_1[8].conv[0].features[0],
            "features_1.8.conv.0.features.1": self.features_1[8].conv[0].features[1],
            "features_1.8.conv.0.features.2": self.features_1[8].conv[0].features[2],
            "features_1.8.conv.1": self.features_1[8].conv[1],
            "features_1.8.conv.1.features": self.features_1[8].conv[1].features,
            "features_1.8.conv.1.features.0": self.features_1[8].conv[1].features[0],
            "features_1.8.conv.1.features.1": self.features_1[8].conv[1].features[1],
            "features_1.8.conv.1.features.2": self.features_1[8].conv[1].features[2],
            "features_1.8.conv.2": self.features_1[8].conv[2],
            "features_1.8.conv.3": self.features_1[8].conv[3],
            "features_1.9": self.features_1[9],
            "features_1.9.conv": self.features_1[9].conv,
            "features_1.9.conv.0": self.features_1[9].conv[0],
            "features_1.9.conv.0.features": self.features_1[9].conv[0].features,
            "features_1.9.conv.0.features.0": self.features_1[9].conv[0].features[0],
            "features_1.9.conv.0.features.1": self.features_1[9].conv[0].features[1],
            "features_1.9.conv.0.features.2": self.features_1[9].conv[0].features[2],
            "features_1.9.conv.1": self.features_1[9].conv[1],
            "features_1.9.conv.1.features": self.features_1[9].conv[1].features,
            "features_1.9.conv.1.features.0": self.features_1[9].conv[1].features[0],
            "features_1.9.conv.1.features.1": self.features_1[9].conv[1].features[1],
            "features_1.9.conv.1.features.2": self.features_1[9].conv[1].features[2],
            "features_1.9.conv.2": self.features_1[9].conv[2],
            "features_1.9.conv.3": self.features_1[9].conv[3],
            "features_1.10": self.features_1[10],
            "features_1.10.conv": self.features_1[10].conv,
            "features_1.10.conv.0": self.features_1[10].conv[0],
            "features_1.10.conv.0.features": self.features_1[10].conv[0].features,
            "features_1.10.conv.0.features.0": self.features_1[10].conv[0].features[0],
            "features_1.10.conv.0.features.1": self.features_1[10].conv[0].features[1],
            "features_1.10.conv.0.features.2": self.features_1[10].conv[0].features[2],
            "features_1.10.conv.1": self.features_1[10].conv[1],
            "features_1.10.conv.1.features": self.features_1[10].conv[1].features,
            "features_1.10.conv.1.features.0": self.features_1[10].conv[1].features[0],
            "features_1.10.conv.1.features.1": self.features_1[10].conv[1].features[1],
            "features_1.10.conv.1.features.2": self.features_1[10].conv[1].features[2],
            "features_1.10.conv.2": self.features_1[10].conv[2],
            "features_1.10.conv.3": self.features_1[10].conv[3],
            "features_1.11": self.features_1[11],
            "features_1.11.conv": self.features_1[11].conv,
            "features_1.11.conv.0": self.features_1[11].conv[0],
            "features_1.11.conv.0.features": self.features_1[11].conv[0].features,
            "features_1.11.conv.0.features.0": self.features_1[11].conv[0].features[0],
            "features_1.11.conv.0.features.1": self.features_1[11].conv[0].features[1],
            "features_1.11.conv.0.features.2": self.features_1[11].conv[0].features[2],
            "features_1.11.conv.1": self.features_1[11].conv[1],
            "features_1.11.conv.1.features": self.features_1[11].conv[1].features,
            "features_1.11.conv.1.features.0": self.features_1[11].conv[1].features[0],
            "features_1.11.conv.1.features.1": self.features_1[11].conv[1].features[1],
            "features_1.11.conv.1.features.2": self.features_1[11].conv[1].features[2],
            "features_1.11.conv.2": self.features_1[11].conv[2],
            "features_1.11.conv.3": self.features_1[11].conv[3],
            "features_1.12": self.features_1[12],
            "features_1.12.conv": self.features_1[12].conv,
            "features_1.12.conv.0": self.features_1[12].conv[0],
            "features_1.12.conv.0.features": self.features_1[12].conv[0].features,
            "features_1.12.conv.0.features.0": self.features_1[12].conv[0].features[0],
            "features_1.12.conv.0.features.1": self.features_1[12].conv[0].features[1],
            "features_1.12.conv.0.features.2": self.features_1[12].conv[0].features[2],
            "features_1.12.conv.1": self.features_1[12].conv[1],
            "features_1.12.conv.1.features": self.features_1[12].conv[1].features,
            "features_1.12.conv.1.features.0": self.features_1[12].conv[1].features[0],
            "features_1.12.conv.1.features.1": self.features_1[12].conv[1].features[1],
            "features_1.12.conv.1.features.2": self.features_1[12].conv[1].features[2],
            "features_1.12.conv.2": self.features_1[12].conv[2],
            "features_1.12.conv.3": self.features_1[12].conv[3],
            "features_1.13": self.features_1[13],
            "features_1.13.conv": self.features_1[13].conv,
            "features_1.13.conv.0": self.features_1[13].conv[0],
            "features_1.13.conv.0.features": self.features_1[13].conv[0].features,
            "features_1.13.conv.0.features.0": self.features_1[13].conv[0].features[0],
            "features_1.13.conv.0.features.1": self.features_1[13].conv[0].features[1],
            "features_1.13.conv.0.features.2": self.features_1[13].conv[0].features[2],
            "features_1.13.conv.1": self.features_1[13].conv[1],
            "features_1.13.conv.1.features": self.features_1[13].conv[1].features,
            "features_1.13.conv.1.features.0": self.features_1[13].conv[1].features[0],
            "features_1.13.conv.1.features.1": self.features_1[13].conv[1].features[1],
            "features_1.13.conv.1.features.2": self.features_1[13].conv[1].features[2],
            "features_1.13.conv.2": self.features_1[13].conv[2],
            "features_1.13.conv.3": self.features_1[13].conv[3],
            "features_2": self.features_2,
            "features_2.0": self.features_2[0],
            "features_2.0.conv": self.features_2[0].conv,
            "features_2.0.conv.0": self.features_2[0].conv[0],
            "features_2.0.conv.0.features": self.features_2[0].conv[0].features,
            "features_2.0.conv.0.features.0": self.features_2[0].conv[0].features[0],
            "features_2.0.conv.0.features.1": self.features_2[0].conv[0].features[1],
            "features_2.0.conv.0.features.2": self.features_2[0].conv[0].features[2],
            "features_2.0.conv.1": self.features_2[0].conv[1],
            "features_2.0.conv.1.features": self.features_2[0].conv[1].features,
            "features_2.0.conv.1.features.0": self.features_2[0].conv[1].features[0],
            "features_2.0.conv.1.features.1": self.features_2[0].conv[1].features[1],
            "features_2.0.conv.1.features.2": self.features_2[0].conv[1].features[2],
            "features_2.0.conv.2": self.features_2[0].conv[2],
            "features_2.0.conv.3": self.features_2[0].conv[3],
            "features_2.1": self.features_2[1],
            "features_2.1.conv": self.features_2[1].conv,
            "features_2.1.conv.0": self.features_2[1].conv[0],
            "features_2.1.conv.0.features": self.features_2[1].conv[0].features,
            "features_2.1.conv.0.features.0": self.features_2[1].conv[0].features[0],
            "features_2.1.conv.0.features.1": self.features_2[1].conv[0].features[1],
            "features_2.1.conv.0.features.2": self.features_2[1].conv[0].features[2],
            "features_2.1.conv.1": self.features_2[1].conv[1],
            "features_2.1.conv.1.features": self.features_2[1].conv[1].features,
            "features_2.1.conv.1.features.0": self.features_2[1].conv[1].features[0],
            "features_2.1.conv.1.features.1": self.features_2[1].conv[1].features[1],
            "features_2.1.conv.1.features.2": self.features_2[1].conv[1].features[2],
            "features_2.1.conv.2": self.features_2[1].conv[2],
            "features_2.1.conv.3": self.features_2[1].conv[3],
            "features_2.2": self.features_2[2],
            "features_2.2.conv": self.features_2[2].conv,
            "features_2.2.conv.0": self.features_2[2].conv[0],
            "features_2.2.conv.0.features": self.features_2[2].conv[0].features,
            "features_2.2.conv.0.features.0": self.features_2[2].conv[0].features[0],
            "features_2.2.conv.0.features.1": self.features_2[2].conv[0].features[1],
            "features_2.2.conv.0.features.2": self.features_2[2].conv[0].features[2],
            "features_2.2.conv.1": self.features_2[2].conv[1],
            "features_2.2.conv.1.features": self.features_2[2].conv[1].features,
            "features_2.2.conv.1.features.0": self.features_2[2].conv[1].features[0],
            "features_2.2.conv.1.features.1": self.features_2[2].conv[1].features[1],
            "features_2.2.conv.1.features.2": self.features_2[2].conv[1].features[2],
            "features_2.2.conv.2": self.features_2[2].conv[2],
            "features_2.2.conv.3": self.features_2[2].conv[3],
            "features_2.3": self.features_2[3],
            "features_2.3.conv": self.features_2[3].conv,
            "features_2.3.conv.0": self.features_2[3].conv[0],
            "features_2.3.conv.0.features": self.features_2[3].conv[0].features,
            "features_2.3.conv.0.features.0": self.features_2[3].conv[0].features[0],
            "features_2.3.conv.0.features.1": self.features_2[3].conv[0].features[1],
            "features_2.3.conv.0.features.2": self.features_2[3].conv[0].features[2],
            "features_2.3.conv.1": self.features_2[3].conv[1],
            "features_2.3.conv.1.features": self.features_2[3].conv[1].features,
            "features_2.3.conv.1.features.0": self.features_2[3].conv[1].features[0],
            "features_2.3.conv.1.features.1": self.features_2[3].conv[1].features[1],
            "features_2.3.conv.1.features.2": self.features_2[3].conv[1].features[2],
            "features_2.3.conv.2": self.features_2[3].conv[2],
            "features_2.3.conv.3": self.features_2[3].conv[3],
            "features_2.4": self.features_2[4],
            "features_2.4.features": self.features_2[4].features,
            "features_2.4.features.0": self.features_2[4].features[0],
            "features_2.4.features.1": self.features_2[4].features[1],
            "features_2.4.features.2": self.features_2[4].features[2],

        }
        self.origin_layer_names = {
            "expand_layer_conv_13": self.expand_layer_conv_13,
            "expand_layer_conv_13.features": self.expand_layer_conv_13.features,
            "expand_layer_conv_13.features.0": self.expand_layer_conv_13.features[0],
            "expand_layer_conv_13.features.1": self.expand_layer_conv_13.features[1],
            "expand_layer_conv_13.features.2": self.expand_layer_conv_13.features[2],
            "features_1": self.features_1,
            "features_1.0": self.features_1[0],
            "features_1.0.features": self.features_1[0].features,
            "features_1.0.features.0": self.features_1[0].features[0],
            "features_1.0.features.1": self.features_1[0].features[1],
            "features_1.0.features.2": self.features_1[0].features[2],
            "features_1.1": self.features_1[1],
            "features_1.1.conv": self.features_1[1].conv,
            "features_1.1.conv.0": self.features_1[1].conv[0],
            "features_1.1.conv.0.features": self.features_1[1].conv[0].features,
            "features_1.1.conv.0.features.0": self.features_1[1].conv[0].features[0],
            "features_1.1.conv.0.features.1": self.features_1[1].conv[0].features[1],
            "features_1.1.conv.0.features.2": self.features_1[1].conv[0].features[2],
            "features_1.1.conv.1": self.features_1[1].conv[1],
            "features_1.1.conv.2": self.features_1[1].conv[2],
            "features_1.2": self.features_1[2],
            "features_1.2.conv": self.features_1[2].conv,
            "features_1.2.conv.0": self.features_1[2].conv[0],
            "features_1.2.conv.0.features": self.features_1[2].conv[0].features,
            "features_1.2.conv.0.features.0": self.features_1[2].conv[0].features[0],
            "features_1.2.conv.0.features.1": self.features_1[2].conv[0].features[1],
            "features_1.2.conv.0.features.2": self.features_1[2].conv[0].features[2],
            "features_1.2.conv.1": self.features_1[2].conv[1],
            "features_1.2.conv.1.features": self.features_1[2].conv[1].features,
            "features_1.2.conv.1.features.0": self.features_1[2].conv[1].features[0],
            "features_1.2.conv.1.features.1": self.features_1[2].conv[1].features[1],
            "features_1.2.conv.1.features.2": self.features_1[2].conv[1].features[2],
            "features_1.2.conv.2": self.features_1[2].conv[2],
            "features_1.2.conv.3": self.features_1[2].conv[3],
            "features_1.3": self.features_1[3],
            "features_1.3.conv": self.features_1[3].conv,
            "features_1.3.conv.0": self.features_1[3].conv[0],
            "features_1.3.conv.0.features": self.features_1[3].conv[0].features,
            "features_1.3.conv.0.features.0": self.features_1[3].conv[0].features[0],
            "features_1.3.conv.0.features.1": self.features_1[3].conv[0].features[1],
            "features_1.3.conv.0.features.2": self.features_1[3].conv[0].features[2],
            "features_1.3.conv.1": self.features_1[3].conv[1],
            "features_1.3.conv.1.features": self.features_1[3].conv[1].features,
            "features_1.3.conv.1.features.0": self.features_1[3].conv[1].features[0],
            "features_1.3.conv.1.features.1": self.features_1[3].conv[1].features[1],
            "features_1.3.conv.1.features.2": self.features_1[3].conv[1].features[2],
            "features_1.3.conv.2": self.features_1[3].conv[2],
            "features_1.3.conv.3": self.features_1[3].conv[3],
            "features_1.4": self.features_1[4],
            "features_1.4.conv": self.features_1[4].conv,
            "features_1.4.conv.0": self.features_1[4].conv[0],
            "features_1.4.conv.0.features": self.features_1[4].conv[0].features,
            "features_1.4.conv.0.features.0": self.features_1[4].conv[0].features[0],
            "features_1.4.conv.0.features.1": self.features_1[4].conv[0].features[1],
            "features_1.4.conv.0.features.2": self.features_1[4].conv[0].features[2],
            "features_1.4.conv.1": self.features_1[4].conv[1],
            "features_1.4.conv.1.features": self.features_1[4].conv[1].features,
            "features_1.4.conv.1.features.0": self.features_1[4].conv[1].features[0],
            "features_1.4.conv.1.features.1": self.features_1[4].conv[1].features[1],
            "features_1.4.conv.1.features.2": self.features_1[4].conv[1].features[2],
            "features_1.4.conv.2": self.features_1[4].conv[2],
            "features_1.4.conv.3": self.features_1[4].conv[3],
            "features_1.5": self.features_1[5],
            "features_1.5.conv": self.features_1[5].conv,
            "features_1.5.conv.0": self.features_1[5].conv[0],
            "features_1.5.conv.0.features": self.features_1[5].conv[0].features,
            "features_1.5.conv.0.features.0": self.features_1[5].conv[0].features[0],
            "features_1.5.conv.0.features.1": self.features_1[5].conv[0].features[1],
            "features_1.5.conv.0.features.2": self.features_1[5].conv[0].features[2],
            "features_1.5.conv.1": self.features_1[5].conv[1],
            "features_1.5.conv.1.features": self.features_1[5].conv[1].features,
            "features_1.5.conv.1.features.0": self.features_1[5].conv[1].features[0],
            "features_1.5.conv.1.features.1": self.features_1[5].conv[1].features[1],
            "features_1.5.conv.1.features.2": self.features_1[5].conv[1].features[2],
            "features_1.5.conv.2": self.features_1[5].conv[2],
            "features_1.5.conv.3": self.features_1[5].conv[3],
            "features_1.6": self.features_1[6],
            "features_1.6.conv": self.features_1[6].conv,
            "features_1.6.conv.0": self.features_1[6].conv[0],
            "features_1.6.conv.0.features": self.features_1[6].conv[0].features,
            "features_1.6.conv.0.features.0": self.features_1[6].conv[0].features[0],
            "features_1.6.conv.0.features.1": self.features_1[6].conv[0].features[1],
            "features_1.6.conv.0.features.2": self.features_1[6].conv[0].features[2],
            "features_1.6.conv.1": self.features_1[6].conv[1],
            "features_1.6.conv.1.features": self.features_1[6].conv[1].features,
            "features_1.6.conv.1.features.0": self.features_1[6].conv[1].features[0],
            "features_1.6.conv.1.features.1": self.features_1[6].conv[1].features[1],
            "features_1.6.conv.1.features.2": self.features_1[6].conv[1].features[2],
            "features_1.6.conv.2": self.features_1[6].conv[2],
            "features_1.6.conv.3": self.features_1[6].conv[3],
            "features_1.7": self.features_1[7],
            "features_1.7.conv": self.features_1[7].conv,
            "features_1.7.conv.0": self.features_1[7].conv[0],
            "features_1.7.conv.0.features": self.features_1[7].conv[0].features,
            "features_1.7.conv.0.features.0": self.features_1[7].conv[0].features[0],
            "features_1.7.conv.0.features.1": self.features_1[7].conv[0].features[1],
            "features_1.7.conv.0.features.2": self.features_1[7].conv[0].features[2],
            "features_1.7.conv.1": self.features_1[7].conv[1],
            "features_1.7.conv.1.features": self.features_1[7].conv[1].features,
            "features_1.7.conv.1.features.0": self.features_1[7].conv[1].features[0],
            "features_1.7.conv.1.features.1": self.features_1[7].conv[1].features[1],
            "features_1.7.conv.1.features.2": self.features_1[7].conv[1].features[2],
            "features_1.7.conv.2": self.features_1[7].conv[2],
            "features_1.7.conv.3": self.features_1[7].conv[3],
            "features_1.8": self.features_1[8],
            "features_1.8.conv": self.features_1[8].conv,
            "features_1.8.conv.0": self.features_1[8].conv[0],
            "features_1.8.conv.0.features": self.features_1[8].conv[0].features,
            "features_1.8.conv.0.features.0": self.features_1[8].conv[0].features[0],
            "features_1.8.conv.0.features.1": self.features_1[8].conv[0].features[1],
            "features_1.8.conv.0.features.2": self.features_1[8].conv[0].features[2],
            "features_1.8.conv.1": self.features_1[8].conv[1],
            "features_1.8.conv.1.features": self.features_1[8].conv[1].features,
            "features_1.8.conv.1.features.0": self.features_1[8].conv[1].features[0],
            "features_1.8.conv.1.features.1": self.features_1[8].conv[1].features[1],
            "features_1.8.conv.1.features.2": self.features_1[8].conv[1].features[2],
            "features_1.8.conv.2": self.features_1[8].conv[2],
            "features_1.8.conv.3": self.features_1[8].conv[3],
            "features_1.9": self.features_1[9],
            "features_1.9.conv": self.features_1[9].conv,
            "features_1.9.conv.0": self.features_1[9].conv[0],
            "features_1.9.conv.0.features": self.features_1[9].conv[0].features,
            "features_1.9.conv.0.features.0": self.features_1[9].conv[0].features[0],
            "features_1.9.conv.0.features.1": self.features_1[9].conv[0].features[1],
            "features_1.9.conv.0.features.2": self.features_1[9].conv[0].features[2],
            "features_1.9.conv.1": self.features_1[9].conv[1],
            "features_1.9.conv.1.features": self.features_1[9].conv[1].features,
            "features_1.9.conv.1.features.0": self.features_1[9].conv[1].features[0],
            "features_1.9.conv.1.features.1": self.features_1[9].conv[1].features[1],
            "features_1.9.conv.1.features.2": self.features_1[9].conv[1].features[2],
            "features_1.9.conv.2": self.features_1[9].conv[2],
            "features_1.9.conv.3": self.features_1[9].conv[3],
            "features_1.10": self.features_1[10],
            "features_1.10.conv": self.features_1[10].conv,
            "features_1.10.conv.0": self.features_1[10].conv[0],
            "features_1.10.conv.0.features": self.features_1[10].conv[0].features,
            "features_1.10.conv.0.features.0": self.features_1[10].conv[0].features[0],
            "features_1.10.conv.0.features.1": self.features_1[10].conv[0].features[1],
            "features_1.10.conv.0.features.2": self.features_1[10].conv[0].features[2],
            "features_1.10.conv.1": self.features_1[10].conv[1],
            "features_1.10.conv.1.features": self.features_1[10].conv[1].features,
            "features_1.10.conv.1.features.0": self.features_1[10].conv[1].features[0],
            "features_1.10.conv.1.features.1": self.features_1[10].conv[1].features[1],
            "features_1.10.conv.1.features.2": self.features_1[10].conv[1].features[2],
            "features_1.10.conv.2": self.features_1[10].conv[2],
            "features_1.10.conv.3": self.features_1[10].conv[3],
            "features_1.11": self.features_1[11],
            "features_1.11.conv": self.features_1[11].conv,
            "features_1.11.conv.0": self.features_1[11].conv[0],
            "features_1.11.conv.0.features": self.features_1[11].conv[0].features,
            "features_1.11.conv.0.features.0": self.features_1[11].conv[0].features[0],
            "features_1.11.conv.0.features.1": self.features_1[11].conv[0].features[1],
            "features_1.11.conv.0.features.2": self.features_1[11].conv[0].features[2],
            "features_1.11.conv.1": self.features_1[11].conv[1],
            "features_1.11.conv.1.features": self.features_1[11].conv[1].features,
            "features_1.11.conv.1.features.0": self.features_1[11].conv[1].features[0],
            "features_1.11.conv.1.features.1": self.features_1[11].conv[1].features[1],
            "features_1.11.conv.1.features.2": self.features_1[11].conv[1].features[2],
            "features_1.11.conv.2": self.features_1[11].conv[2],
            "features_1.11.conv.3": self.features_1[11].conv[3],
            "features_1.12": self.features_1[12],
            "features_1.12.conv": self.features_1[12].conv,
            "features_1.12.conv.0": self.features_1[12].conv[0],
            "features_1.12.conv.0.features": self.features_1[12].conv[0].features,
            "features_1.12.conv.0.features.0": self.features_1[12].conv[0].features[0],
            "features_1.12.conv.0.features.1": self.features_1[12].conv[0].features[1],
            "features_1.12.conv.0.features.2": self.features_1[12].conv[0].features[2],
            "features_1.12.conv.1": self.features_1[12].conv[1],
            "features_1.12.conv.1.features": self.features_1[12].conv[1].features,
            "features_1.12.conv.1.features.0": self.features_1[12].conv[1].features[0],
            "features_1.12.conv.1.features.1": self.features_1[12].conv[1].features[1],
            "features_1.12.conv.1.features.2": self.features_1[12].conv[1].features[2],
            "features_1.12.conv.2": self.features_1[12].conv[2],
            "features_1.12.conv.3": self.features_1[12].conv[3],
            "features_1.13": self.features_1[13],
            "features_1.13.conv": self.features_1[13].conv,
            "features_1.13.conv.0": self.features_1[13].conv[0],
            "features_1.13.conv.0.features": self.features_1[13].conv[0].features,
            "features_1.13.conv.0.features.0": self.features_1[13].conv[0].features[0],
            "features_1.13.conv.0.features.1": self.features_1[13].conv[0].features[1],
            "features_1.13.conv.0.features.2": self.features_1[13].conv[0].features[2],
            "features_1.13.conv.1": self.features_1[13].conv[1],
            "features_1.13.conv.1.features": self.features_1[13].conv[1].features,
            "features_1.13.conv.1.features.0": self.features_1[13].conv[1].features[0],
            "features_1.13.conv.1.features.1": self.features_1[13].conv[1].features[1],
            "features_1.13.conv.1.features.2": self.features_1[13].conv[1].features[2],
            "features_1.13.conv.2": self.features_1[13].conv[2],
            "features_1.13.conv.3": self.features_1[13].conv[3],
            "features_2": self.features_2,
            "features_2.0": self.features_2[0],
            "features_2.0.conv": self.features_2[0].conv,
            "features_2.0.conv.0": self.features_2[0].conv[0],
            "features_2.0.conv.0.features": self.features_2[0].conv[0].features,
            "features_2.0.conv.0.features.0": self.features_2[0].conv[0].features[0],
            "features_2.0.conv.0.features.1": self.features_2[0].conv[0].features[1],
            "features_2.0.conv.0.features.2": self.features_2[0].conv[0].features[2],
            "features_2.0.conv.1": self.features_2[0].conv[1],
            "features_2.0.conv.1.features": self.features_2[0].conv[1].features,
            "features_2.0.conv.1.features.0": self.features_2[0].conv[1].features[0],
            "features_2.0.conv.1.features.1": self.features_2[0].conv[1].features[1],
            "features_2.0.conv.1.features.2": self.features_2[0].conv[1].features[2],
            "features_2.0.conv.2": self.features_2[0].conv[2],
            "features_2.0.conv.3": self.features_2[0].conv[3],
            "features_2.1": self.features_2[1],
            "features_2.1.conv": self.features_2[1].conv,
            "features_2.1.conv.0": self.features_2[1].conv[0],
            "features_2.1.conv.0.features": self.features_2[1].conv[0].features,
            "features_2.1.conv.0.features.0": self.features_2[1].conv[0].features[0],
            "features_2.1.conv.0.features.1": self.features_2[1].conv[0].features[1],
            "features_2.1.conv.0.features.2": self.features_2[1].conv[0].features[2],
            "features_2.1.conv.1": self.features_2[1].conv[1],
            "features_2.1.conv.1.features": self.features_2[1].conv[1].features,
            "features_2.1.conv.1.features.0": self.features_2[1].conv[1].features[0],
            "features_2.1.conv.1.features.1": self.features_2[1].conv[1].features[1],
            "features_2.1.conv.1.features.2": self.features_2[1].conv[1].features[2],
            "features_2.1.conv.2": self.features_2[1].conv[2],
            "features_2.1.conv.3": self.features_2[1].conv[3],
            "features_2.2": self.features_2[2],
            "features_2.2.conv": self.features_2[2].conv,
            "features_2.2.conv.0": self.features_2[2].conv[0],
            "features_2.2.conv.0.features": self.features_2[2].conv[0].features,
            "features_2.2.conv.0.features.0": self.features_2[2].conv[0].features[0],
            "features_2.2.conv.0.features.1": self.features_2[2].conv[0].features[1],
            "features_2.2.conv.0.features.2": self.features_2[2].conv[0].features[2],
            "features_2.2.conv.1": self.features_2[2].conv[1],
            "features_2.2.conv.1.features": self.features_2[2].conv[1].features,
            "features_2.2.conv.1.features.0": self.features_2[2].conv[1].features[0],
            "features_2.2.conv.1.features.1": self.features_2[2].conv[1].features[1],
            "features_2.2.conv.1.features.2": self.features_2[2].conv[1].features[2],
            "features_2.2.conv.2": self.features_2[2].conv[2],
            "features_2.2.conv.3": self.features_2[2].conv[3],
            "features_2.3": self.features_2[3],
            "features_2.3.conv": self.features_2[3].conv,
            "features_2.3.conv.0": self.features_2[3].conv[0],
            "features_2.3.conv.0.features": self.features_2[3].conv[0].features,
            "features_2.3.conv.0.features.0": self.features_2[3].conv[0].features[0],
            "features_2.3.conv.0.features.1": self.features_2[3].conv[0].features[1],
            "features_2.3.conv.0.features.2": self.features_2[3].conv[0].features[2],
            "features_2.3.conv.1": self.features_2[3].conv[1],
            "features_2.3.conv.1.features": self.features_2[3].conv[1].features,
            "features_2.3.conv.1.features.0": self.features_2[3].conv[1].features[0],
            "features_2.3.conv.1.features.1": self.features_2[3].conv[1].features[1],
            "features_2.3.conv.1.features.2": self.features_2[3].conv[1].features[2],
            "features_2.3.conv.2": self.features_2[3].conv[2],
            "features_2.3.conv.3": self.features_2[3].conv[3],
            "features_2.4": self.features_2[4],
            "features_2.4.features": self.features_2[4].features,
            "features_2.4.features.0": self.features_2[4].features[0],
            "features_2.4.features.1": self.features_2[4].features[1],
            "features_2.4.features.2": self.features_2[4].features[2],

        }

        self.in_shapes = {
            'INPUT': [1, 3, 300, 300],
            'features_1.0.features.0': [1, 3, 300, 300],
            'features_1.0.features.1': [1, 32, 150, 150],
            'features_1.0.features.2': [1, 32, 150, 150],
            'features_1.1.conv.0.features.0': [1, 32, 150, 150],
            'features_1.1.conv.0.features.1': [1, 32, 150, 150],
            'features_1.1.conv.0.features.2': [1, 32, 150, 150],
            'features_1.1.conv.1': [1, 32, 150, 150],
            'features_1.1.conv.2': [1, 16, 150, 150],
            'features_1.2.conv.0.features.0': [1, 16, 150, 150],
            'features_1.2.conv.0.features.1': [1, 96, 150, 150],
            'features_1.2.conv.0.features.2': [1, 96, 150, 150],
            'features_1.2.conv.1.features.0': [1, 96, 150, 150],
            'features_1.2.conv.1.features.1': [1, 96, 75, 75],
            'features_1.2.conv.1.features.2': [1, 96, 75, 75],
            'features_1.2.conv.2': [1, 96, 75, 75],
            'features_1.2.conv.3': [1, 24, 75, 75],
            'features_1.3.conv.0.features.0': [1, 24, 75, 75],
            'features_1.3.conv.0.features.1': [1, 144, 75, 75],
            'features_1.3.conv.0.features.2': [1, 144, 75, 75],
            'features_1.3.conv.1.features.0': [1, 144, 75, 75],
            'features_1.3.conv.1.features.1': [1, 144, 75, 75],
            'features_1.3.conv.1.features.2': [1, 144, 75, 75],
            'features_1.3.conv.2': [1, 144, 75, 75],
            'features_1.3.conv.3': [1, 24, 75, 75],
            'features_1.4.conv.0.features.0': [1, 24, 75, 75],
            'features_1.4.conv.0.features.1': [1, 144, 75, 75],
            'features_1.4.conv.0.features.2': [1, 144, 75, 75],
            'features_1.4.conv.1.features.0': [1, 144, 75, 75],
            'features_1.4.conv.1.features.1': [1, 144, 38, 38],
            'features_1.4.conv.1.features.2': [1, 144, 38, 38],
            'features_1.4.conv.2': [1, 144, 38, 38],
            'features_1.4.conv.3': [1, 32, 38, 38],
            'features_1.5.conv.0.features.0': [1, 32, 38, 38],
            'features_1.5.conv.0.features.1': [1, 192, 38, 38],
            'features_1.5.conv.0.features.2': [1, 192, 38, 38],
            'features_1.5.conv.1.features.0': [1, 192, 38, 38],
            'features_1.5.conv.1.features.1': [1, 192, 38, 38],
            'features_1.5.conv.1.features.2': [1, 192, 38, 38],
            'features_1.5.conv.2': [1, 192, 38, 38],
            'features_1.5.conv.3': [1, 32, 38, 38],
            'features_1.6.conv.0.features.0': [1, 32, 38, 38],
            'features_1.6.conv.0.features.1': [1, 192, 38, 38],
            'features_1.6.conv.0.features.2': [1, 192, 38, 38],
            'features_1.6.conv.1.features.0': [1, 192, 38, 38],
            'features_1.6.conv.1.features.1': [1, 192, 38, 38],
            'features_1.6.conv.1.features.2': [1, 192, 38, 38],
            'features_1.6.conv.2': [1, 192, 38, 38],
            'features_1.6.conv.3': [1, 32, 38, 38],
            'features_1.7.conv.0.features.0': [1, 32, 38, 38],
            'features_1.7.conv.0.features.1': [1, 192, 38, 38],
            'features_1.7.conv.0.features.2': [1, 192, 38, 38],
            'features_1.7.conv.1.features.0': [1, 192, 38, 38],
            'features_1.7.conv.1.features.1': [1, 192, 19, 19],
            'features_1.7.conv.1.features.2': [1, 192, 19, 19],
            'features_1.7.conv.2': [1, 192, 19, 19],
            'features_1.7.conv.3': [1, 64, 19, 19],
            'features_1.8.conv.0.features.0': [1, 64, 19, 19],
            'features_1.8.conv.0.features.1': [1, 384, 19, 19],
            'features_1.8.conv.0.features.2': [1, 384, 19, 19],
            'features_1.8.conv.1.features.0': [1, 384, 19, 19],
            'features_1.8.conv.1.features.1': [1, 384, 19, 19],
            'features_1.8.conv.1.features.2': [1, 384, 19, 19],
            'features_1.8.conv.2': [1, 384, 19, 19],
            'features_1.8.conv.3': [1, 64, 19, 19],
            'features_1.9.conv.0.features.0': [1, 64, 19, 19],
            'features_1.9.conv.0.features.1': [1, 384, 19, 19],
            'features_1.9.conv.0.features.2': [1, 384, 19, 19],
            'features_1.9.conv.1.features.0': [1, 384, 19, 19],
            'features_1.9.conv.1.features.1': [1, 384, 19, 19],
            'features_1.9.conv.1.features.2': [1, 384, 19, 19],
            'features_1.9.conv.2': [1, 384, 19, 19],
            'features_1.9.conv.3': [1, 64, 19, 19],
            'features_1.10.conv.0.features.0': [1, 64, 19, 19],
            'features_1.10.conv.0.features.1': [1, 384, 19, 19],
            'features_1.10.conv.0.features.2': [1, 384, 19, 19],
            'features_1.10.conv.1.features.0': [1, 384, 19, 19],
            'features_1.10.conv.1.features.1': [1, 384, 19, 19],
            'features_1.10.conv.1.features.2': [1, 384, 19, 19],
            'features_1.10.conv.2': [1, 384, 19, 19],
            'features_1.10.conv.3': [1, 64, 19, 19],
            'features_1.11.conv.0.features.0': [1, 64, 19, 19],
            'features_1.11.conv.0.features.1': [1, 384, 19, 19],
            'features_1.11.conv.0.features.2': [1, 384, 19, 19],
            'features_1.11.conv.1.features.0': [1, 384, 19, 19],
            'features_1.11.conv.1.features.1': [1, 384, 19, 19],
            'features_1.11.conv.1.features.2': [1, 384, 19, 19],
            'features_1.11.conv.2': [1, 384, 19, 19],
            'features_1.11.conv.3': [1, 96, 19, 19],
            'features_1.12.conv.0.features.0': [1, 96, 19, 19],
            'features_1.12.conv.0.features.1': [1, 576, 19, 19],
            'features_1.12.conv.0.features.2': [1, 576, 19, 19],
            'features_1.12.conv.1.features.0': [1, 576, 19, 19],
            'features_1.12.conv.1.features.1': [1, 576, 19, 19],
            'features_1.12.conv.1.features.2': [1, 576, 19, 19],
            'features_1.12.conv.2': [1, 576, 19, 19],
            'features_1.12.conv.3': [1, 96, 19, 19],
            'features_1.13.conv.0.features.0': [1, 96, 19, 19],
            'features_1.13.conv.0.features.1': [1, 576, 19, 19],
            'features_1.13.conv.0.features.2': [1, 576, 19, 19],
            'features_1.13.conv.1.features.0': [1, 576, 19, 19],
            'features_1.13.conv.1.features.1': [1, 576, 19, 19],
            'features_1.13.conv.1.features.2': [1, 576, 19, 19],
            'features_1.13.conv.2': [1, 576, 19, 19],
            'features_1.13.conv.3': [1, 96, 19, 19],
            'features_2.0.conv.0.features.0': [1, 96, 19, 19],
            'features_2.0.conv.0.features.1': [1, 576, 19, 19],
            'features_2.0.conv.0.features.2': [1, 576, 19, 19],
            'features_2.0.conv.1.features.0': [1, 576, 19, 19],
            'features_2.0.conv.1.features.1': [1, 576, 10, 10],
            'features_2.0.conv.1.features.2': [1, 576, 10, 10],
            'features_2.0.conv.2': [1, 576, 10, 10],
            'features_2.0.conv.3': [1, 160, 10, 10],
            'features_2.1.conv.0.features.0': [1, 160, 10, 10],
            'features_2.1.conv.0.features.1': [1, 960, 10, 10],
            'features_2.1.conv.0.features.2': [1, 960, 10, 10],
            'features_2.1.conv.1.features.0': [1, 960, 10, 10],
            'features_2.1.conv.1.features.1': [1, 960, 10, 10],
            'features_2.1.conv.1.features.2': [1, 960, 10, 10],
            'features_2.1.conv.2': [1, 960, 10, 10],
            'features_2.1.conv.3': [1, 160, 10, 10],
            'features_2.2.conv.0.features.0': [1, 160, 10, 10],
            'features_2.2.conv.0.features.1': [1, 960, 10, 10],
            'features_2.2.conv.0.features.2': [1, 960, 10, 10],
            'features_2.2.conv.1.features.0': [1, 960, 10, 10],
            'features_2.2.conv.1.features.1': [1, 960, 10, 10],
            'features_2.2.conv.1.features.2': [1, 960, 10, 10],
            'features_2.2.conv.2': [1, 960, 10, 10],
            'features_2.2.conv.3': [1, 160, 10, 10],
            'features_2.3.conv.0.features.0': [1, 160, 10, 10],
            'features_2.3.conv.0.features.1': [1, 960, 10, 10],
            'features_2.3.conv.0.features.2': [1, 960, 10, 10],
            'features_2.3.conv.1.features.0': [1, 960, 10, 10],
            'features_2.3.conv.1.features.1': [1, 960, 10, 10],
            'features_2.3.conv.1.features.2': [1, 960, 10, 10],
            'features_2.3.conv.2': [1, 960, 10, 10],
            'features_2.3.conv.3': [1, 320, 10, 10],
            'features_2.4.features.0': [1, 320, 10, 10],
            'features_2.4.features.1': [1, 1280, 10, 10],
            'features_2.4.features.2': [1, 1280, 10, 10],
            'expand_layer_conv_13.features.0': [1, 96, 19, 19],
            'expand_layer_conv_13.features.1': [1, 576, 19, 19],
            'expand_layer_conv_13.features.2': [1, 576, 19, 19],
            'OUTPUT1': [1, 1280, 10, 10],
            'OUTPUT2': [1, 576, 19, 19]
        }
        self.out_shapes = {
            'INPUT': [-1, 3, 300, 300],
            'features_1.0.features.0': [1, 32, 150, 150],
            'features_1.0.features.1': [1, 32, 150, 150],
            'features_1.0.features.2': [1, 32, 150, 150],
            'features_1.1.conv.0.features.0': [1, 32, 150, 150],
            'features_1.1.conv.0.features.1': [1, 32, 150, 150],
            'features_1.1.conv.0.features.2': [1, 32, 150, 150],
            'features_1.1.conv.1': [1, 16, 150, 150],
            'features_1.1.conv.2': [1, 16, 150, 150],
            'features_1.2.conv.0.features.0': [1, 96, 150, 150],
            'features_1.2.conv.0.features.1': [1, 96, 150, 150],
            'features_1.2.conv.0.features.2': [1, 96, 150, 150],
            'features_1.2.conv.1.features.0': [1, 96, 75, 75],
            'features_1.2.conv.1.features.1': [1, 96, 75, 75],
            'features_1.2.conv.1.features.2': [1, 96, 75, 75],
            'features_1.2.conv.2': [1, 24, 75, 75],
            'features_1.2.conv.3': [1, 24, 75, 75],
            'features_1.3.conv.0.features.0': [1, 144, 75, 75],
            'features_1.3.conv.0.features.1': [1, 144, 75, 75],
            'features_1.3.conv.0.features.2': [1, 144, 75, 75],
            'features_1.3.conv.1.features.0': [1, 144, 75, 75],
            'features_1.3.conv.1.features.1': [1, 144, 75, 75],
            'features_1.3.conv.1.features.2': [1, 144, 75, 75],
            'features_1.3.conv.2': [1, 24, 75, 75],
            'features_1.3.conv.3': [1, 24, 75, 75],
            'features_1.4.conv.0.features.0': [1, 144, 75, 75],
            'features_1.4.conv.0.features.1': [1, 144, 75, 75],
            'features_1.4.conv.0.features.2': [1, 144, 75, 75],
            'features_1.4.conv.1.features.0': [1, 144, 38, 38],
            'features_1.4.conv.1.features.1': [1, 144, 38, 38],
            'features_1.4.conv.1.features.2': [1, 144, 38, 38],
            'features_1.4.conv.2': [1, 32, 38, 38],
            'features_1.4.conv.3': [1, 32, 38, 38],
            'features_1.5.conv.0.features.0': [1, 192, 38, 38],
            'features_1.5.conv.0.features.1': [1, 192, 38, 38],
            'features_1.5.conv.0.features.2': [1, 192, 38, 38],
            'features_1.5.conv.1.features.0': [1, 192, 38, 38],
            'features_1.5.conv.1.features.1': [1, 192, 38, 38],
            'features_1.5.conv.1.features.2': [1, 192, 38, 38],
            'features_1.5.conv.2': [1, 32, 38, 38],
            'features_1.5.conv.3': [1, 32, 38, 38],
            'features_1.6.conv.0.features.0': [1, 192, 38, 38],
            'features_1.6.conv.0.features.1': [1, 192, 38, 38],
            'features_1.6.conv.0.features.2': [1, 192, 38, 38],
            'features_1.6.conv.1.features.0': [1, 192, 38, 38],
            'features_1.6.conv.1.features.1': [1, 192, 38, 38],
            'features_1.6.conv.1.features.2': [1, 192, 38, 38],
            'features_1.6.conv.2': [1, 32, 38, 38],
            'features_1.6.conv.3': [1, 32, 38, 38],
            'features_1.7.conv.0.features.0': [1, 192, 38, 38],
            'features_1.7.conv.0.features.1': [1, 192, 38, 38],
            'features_1.7.conv.0.features.2': [1, 192, 38, 38],
            'features_1.7.conv.1.features.0': [1, 192, 19, 19],
            'features_1.7.conv.1.features.1': [1, 192, 19, 19],
            'features_1.7.conv.1.features.2': [1, 192, 19, 19],
            'features_1.7.conv.2': [1, 64, 19, 19],
            'features_1.7.conv.3': [1, 64, 19, 19],
            'features_1.8.conv.0.features.0': [1, 384, 19, 19],
            'features_1.8.conv.0.features.1': [1, 384, 19, 19],
            'features_1.8.conv.0.features.2': [1, 384, 19, 19],
            'features_1.8.conv.1.features.0': [1, 384, 19, 19],
            'features_1.8.conv.1.features.1': [1, 384, 19, 19],
            'features_1.8.conv.1.features.2': [1, 384, 19, 19],
            'features_1.8.conv.2': [1, 64, 19, 19],
            'features_1.8.conv.3': [1, 64, 19, 19],
            'features_1.9.conv.0.features.0': [1, 384, 19, 19],
            'features_1.9.conv.0.features.1': [1, 384, 19, 19],
            'features_1.9.conv.0.features.2': [1, 384, 19, 19],
            'features_1.9.conv.1.features.0': [1, 384, 19, 19],
            'features_1.9.conv.1.features.1': [1, 384, 19, 19],
            'features_1.9.conv.1.features.2': [1, 384, 19, 19],
            'features_1.9.conv.2': [1, 64, 19, 19],
            'features_1.9.conv.3': [1, 64, 19, 19],
            'features_1.10.conv.0.features.0': [1, 384, 19, 19],
            'features_1.10.conv.0.features.1': [1, 384, 19, 19],
            'features_1.10.conv.0.features.2': [1, 384, 19, 19],
            'features_1.10.conv.1.features.0': [1, 384, 19, 19],
            'features_1.10.conv.1.features.1': [1, 384, 19, 19],
            'features_1.10.conv.1.features.2': [1, 384, 19, 19],
            'features_1.10.conv.2': [1, 64, 19, 19],
            'features_1.10.conv.3': [1, 64, 19, 19],
            'features_1.11.conv.0.features.0': [1, 384, 19, 19],
            'features_1.11.conv.0.features.1': [1, 384, 19, 19],
            'features_1.11.conv.0.features.2': [1, 384, 19, 19],
            'features_1.11.conv.1.features.0': [1, 384, 19, 19],
            'features_1.11.conv.1.features.1': [1, 384, 19, 19],
            'features_1.11.conv.1.features.2': [1, 384, 19, 19],
            'features_1.11.conv.2': [1, 96, 19, 19],
            'features_1.11.conv.3': [1, 96, 19, 19],
            'features_1.12.conv.0.features.0': [1, 576, 19, 19],
            'features_1.12.conv.0.features.1': [1, 576, 19, 19],
            'features_1.12.conv.0.features.2': [1, 576, 19, 19],
            'features_1.12.conv.1.features.0': [1, 576, 19, 19],
            'features_1.12.conv.1.features.1': [1, 576, 19, 19],
            'features_1.12.conv.1.features.2': [1, 576, 19, 19],
            'features_1.12.conv.2': [1, 96, 19, 19],
            'features_1.12.conv.3': [1, 96, 19, 19],
            'features_1.13.conv.0.features.0': [1, 576, 19, 19],
            'features_1.13.conv.0.features.1': [1, 576, 19, 19],
            'features_1.13.conv.0.features.2': [1, 576, 19, 19],
            'features_1.13.conv.1.features.0': [1, 576, 19, 19],
            'features_1.13.conv.1.features.1': [1, 576, 19, 19],
            'features_1.13.conv.1.features.2': [1, 576, 19, 19],
            'features_1.13.conv.2': [1, 96, 19, 19],
            'features_1.13.conv.3': [1, 96, 19, 19],
            'features_2.0.conv.0.features.0': [1, 576, 19, 19],
            'features_2.0.conv.0.features.1': [1, 576, 19, 19],
            'features_2.0.conv.0.features.2': [1, 576, 19, 19],
            'features_2.0.conv.1.features.0': [1, 576, 10, 10],
            'features_2.0.conv.1.features.1': [1, 576, 10, 10],
            'features_2.0.conv.1.features.2': [1, 576, 10, 10],
            'features_2.0.conv.2': [1, 160, 10, 10],
            'features_2.0.conv.3': [1, 160, 10, 10],
            'features_2.1.conv.0.features.0': [1, 960, 10, 10],
            'features_2.1.conv.0.features.1': [1, 960, 10, 10],
            'features_2.1.conv.0.features.2': [1, 960, 10, 10],
            'features_2.1.conv.1.features.0': [1, 960, 10, 10],
            'features_2.1.conv.1.features.1': [1, 960, 10, 10],
            'features_2.1.conv.1.features.2': [1, 960, 10, 10],
            'features_2.1.conv.2': [1, 160, 10, 10],
            'features_2.1.conv.3': [1, 160, 10, 10],
            'features_2.2.conv.0.features.0': [1, 960, 10, 10],
            'features_2.2.conv.0.features.1': [1, 960, 10, 10],
            'features_2.2.conv.0.features.2': [1, 960, 10, 10],
            'features_2.2.conv.1.features.0': [1, 960, 10, 10],
            'features_2.2.conv.1.features.1': [1, 960, 10, 10],
            'features_2.2.conv.1.features.2': [1, 960, 10, 10],
            'features_2.2.conv.2': [1, 160, 10, 10],
            'features_2.2.conv.3': [1, 160, 10, 10],
            'features_2.3.conv.0.features.0': [1, 960, 10, 10],
            'features_2.3.conv.0.features.1': [1, 960, 10, 10],
            'features_2.3.conv.0.features.2': [1, 960, 10, 10],
            'features_2.3.conv.1.features.0': [1, 960, 10, 10],
            'features_2.3.conv.1.features.1': [1, 960, 10, 10],
            'features_2.3.conv.1.features.2': [1, 960, 10, 10],
            'features_2.3.conv.2': [1, 320, 10, 10],
            'features_2.3.conv.3': [1, 320, 10, 10],
            'features_2.4.features.0': [1, 1280, 10, 10],
            'features_2.4.features.1': [1, 1280, 10, 10],
            'features_2.4.features.2': [1, 1280, 10, 10],
            'expand_layer_conv_13.features.0': [1, 576, 19, 19],
            'expand_layer_conv_13.features.1': [1, 576, 19, 19],
            'expand_layer_conv_13.features.2': [1, 576, 19, 19],
            'OUTPUT1': [1, 1280, 10, 10],
            'OUTPUT2': [1, 576, 19, 19]
        }

        self.orders = {
            'features_1.0.features.0': ["INPUT", "features_1.0.features.1"],
            'features_1.0.features.1': ["features_1.0.features.0", "features_1.0.features.2"],
            'features_1.0.features.2': ["features_1.0.features.1", "features_1.1.conv.0.features.0"],

            'features_1.1.conv.0.features.0': ["features_1.0.features.2", "features_1.1.conv.0.features.1"],
            'features_1.1.conv.0.features.1': ["features_1.1.conv.0.features.0",
                                               "features_1.1.conv.0.features.2"],
            'features_1.1.conv.0.features.2': ["features_1.1.conv.0.features.1", "features_1.1.conv.1"],
            'features_1.1.conv.1': ["features_1.1.conv.0.features.2", "features_1.1.conv.2"],
            'features_1.1.conv.2': ["features_1.1.conv.1", "features_1.2.conv.0.features.0"],

            'features_1.2.conv.0.features.0': ["features_1.1.conv.2", "features_1.2.conv.0.features.1"],
            'features_1.2.conv.0.features.1': ["features_1.2.conv.0.features.0",
                                               "features_1.2.conv.0.features.2"],
            'features_1.2.conv.0.features.2': ["features_1.2.conv.0.features.1",
                                               "features_1.2.conv.1.features.0"],
            'features_1.2.conv.1.features.0': ["features_1.2.conv.0.features.2",
                                               "features_1.2.conv.1.features.1"],
            'features_1.2.conv.1.features.1': ["features_1.2.conv.1.features.0",
                                               "features_1.2.conv.1.features.2"],
            'features_1.2.conv.1.features.2': ["features_1.2.conv.1.features.1", "features_1.2.conv.2"],
            'features_1.2.conv.2': ["features_1.2.conv.1.features.2", "features_1.2.conv.3"],
            'features_1.2.conv.3': ["features_1.2.conv.2",
                                    ["features_1.3.conv.0.features.0", 'features_1.4.conv.0.features.0']],

            'features_1.3.conv.0.features.0': ["features_1.2.conv.3", "features_1.3.conv.0.features.1"],
            'features_1.3.conv.0.features.1': ["features_1.3.conv.0.features.0",
                                               "features_1.3.conv.0.features.2"],
            'features_1.3.conv.0.features.2': ["features_1.3.conv.0.features.1",
                                               "features_1.3.conv.1.features.0"],
            'features_1.3.conv.1.features.0': ["features_1.3.conv.0.features.2",
                                               "features_1.3.conv.1.features.1"],
            'features_1.3.conv.1.features.1': ["features_1.3.conv.1.features.0",
                                               "features_1.3.conv.1.features.2"],
            'features_1.3.conv.1.features.2': ["features_1.3.conv.1.features.1", "features_1.3.conv.2"],
            'features_1.3.conv.2': ["features_1.3.conv.1.features.2", "features_1.3.conv.3"],
            'features_1.3.conv.3': ["features_1.3.conv.2", "features_1.4.conv.0.features.0"],

            'features_1.4.conv.0.features.0': [["features_1.3.conv.3", 'features_1.2.conv.3'],
                                               "features_1.4.conv.0.features.1"],
            'features_1.4.conv.0.features.1': ["features_1.4.conv.0.features.0",
                                               "features_1.4.conv.0.features.2"],
            'features_1.4.conv.0.features.2': ["features_1.4.conv.0.features.1",
                                               "features_1.4.conv.1.features.0"],
            'features_1.4.conv.1.features.0': ["features_1.4.conv.0.features.2",
                                               "features_1.4.conv.1.features.1"],
            'features_1.4.conv.1.features.1': ["features_1.4.conv.1.features.0",
                                               "features_1.4.conv.1.features.2"],
            'features_1.4.conv.1.features.2': ["features_1.4.conv.1.features.1", "features_1.4.conv.2"],
            'features_1.4.conv.2': ["features_1.4.conv.1.features.2", "features_1.4.conv.3"],
            'features_1.4.conv.3': ["features_1.4.conv.2",
                                    ["features_1.5.conv.0.features.0", 'features_1.6.conv.0.features.0']],

            'features_1.5.conv.0.features.0': ["features_1.4.conv.3", "features_1.5.conv.0.features.1"],
            'features_1.5.conv.0.features.1': ["features_1.5.conv.0.features.0",
                                               "features_1.5.conv.0.features.2"],
            'features_1.5.conv.0.features.2': ["features_1.5.conv.0.features.1",
                                               "features_1.5.conv.1.features.0"],
            'features_1.5.conv.1.features.0': ["features_1.5.conv.0.features.2",
                                               "features_1.5.conv.1.features.1"],
            'features_1.5.conv.1.features.1': ["features_1.5.conv.1.features.0",
                                               "features_1.5.conv.1.features.2"],
            'features_1.5.conv.1.features.2': ["features_1.5.conv.1.features.1", "features_1.5.conv.2"],
            'features_1.5.conv.2': ["features_1.5.conv.1.features.2", "features_1.5.conv.3"],
            'features_1.5.conv.3': ["features_1.5.conv.2",
                                    ["features_1.6.conv.0.features.0", 'features_1.7.conv.0.features.0']],

            'features_1.6.conv.0.features.0': [["features_1.5.conv.3", 'features_1.4.conv.3'],
                                               "features_1.6.conv.0.features.1"],
            'features_1.6.conv.0.features.1': ["features_1.6.conv.0.features.0",
                                               "features_1.6.conv.0.features.2"],
            'features_1.6.conv.0.features.2': ["features_1.6.conv.0.features.1",
                                               "features_1.6.conv.1.features.0"],
            'features_1.6.conv.1.features.0': ["features_1.6.conv.0.features.2",
                                               "features_1.6.conv.1.features.1"],
            'features_1.6.conv.1.features.1': ["features_1.6.conv.1.features.0",
                                               "features_1.6.conv.1.features.2"],
            'features_1.6.conv.1.features.2': ["features_1.6.conv.1.features.1", "features_1.6.conv.2"],
            'features_1.6.conv.2': ["features_1.6.conv.1.features.2", "features_1.6.conv.3"],
            'features_1.6.conv.3': ["features_1.6.conv.2", "features_1.7.conv.0.features.0"],

            'features_1.7.conv.0.features.0': [["features_1.6.conv.3", 'features_1.5.conv.3'],
                                               "features_1.7.conv.0.features.1"],
            'features_1.7.conv.0.features.1': ["features_1.7.conv.0.features.0",
                                               "features_1.7.conv.0.features.2"],
            'features_1.7.conv.0.features.2': ["features_1.7.conv.0.features.1",
                                               "features_1.7.conv.1.features.0"],
            'features_1.7.conv.1.features.0': ["features_1.7.conv.0.features.2",
                                               "features_1.7.conv.1.features.1"],
            'features_1.7.conv.1.features.1': ["features_1.7.conv.1.features.0",
                                               "features_1.7.conv.1.features.2"],
            'features_1.7.conv.1.features.2': ["features_1.7.conv.1.features.1", "features_1.7.conv.2"],
            'features_1.7.conv.2': ["features_1.7.conv.1.features.2", "features_1.7.conv.3"],
            'features_1.7.conv.3': ["features_1.7.conv.2",
                                    ["features_1.8.conv.0.features.0", 'features_1.9.conv.0.features.0']],

            'features_1.8.conv.0.features.0': ["features_1.7.conv.3", "features_1.8.conv.0.features.1"],
            'features_1.8.conv.0.features.1': ["features_1.8.conv.0.features.0",
                                               "features_1.8.conv.0.features.2"],
            'features_1.8.conv.0.features.2': ["features_1.8.conv.0.features.1",
                                               "features_1.8.conv.1.features.0"],
            'features_1.8.conv.1.features.0': ["features_1.8.conv.0.features.2",
                                               "features_1.8.conv.1.features.1"],
            'features_1.8.conv.1.features.1': ["features_1.8.conv.1.features.0",
                                               "features_1.8.conv.1.features.2"],
            'features_1.8.conv.1.features.2': ["features_1.8.conv.1.features.1", "features_1.8.conv.2"],
            'features_1.8.conv.2': ["features_1.8.conv.1.features.2", "features_1.8.conv.3"],
            'features_1.8.conv.3': ["features_1.8.conv.2",
                                    ["features_1.9.conv.0.features.0", 'features_1.10.conv.0.features.0']],

            'features_1.9.conv.0.features.0': [["features_1.8.conv.3", 'features_1.7.conv.3'],
                                               "features_1.9.conv.0.features.1"],
            'features_1.9.conv.0.features.1': ["features_1.9.conv.0.features.0",
                                               "features_1.9.conv.0.features.2"],
            'features_1.9.conv.0.features.2': ["features_1.9.conv.0.features.1",
                                               "features_1.9.conv.1.features.0"],
            'features_1.9.conv.1.features.0': ["features_1.9.conv.0.features.2",
                                               "features_1.9.conv.1.features.1"],
            'features_1.9.conv.1.features.1': ["features_1.9.conv.1.features.0",
                                               "features_1.9.conv.1.features.2"],
            'features_1.9.conv.1.features.2': ["features_1.9.conv.1.features.1", "features_1.9.conv.2"],
            'features_1.9.conv.2': ["features_1.9.conv.1.features.2", "features_1.9.conv.3"],
            'features_1.9.conv.3': ["features_1.9.conv.2", "features_1.10.conv.0.features.0"],

            'features_1.10.conv.0.features.0': [["features_1.9.conv.3", 'features_1.8.conv.3'],
                                                "features_1.10.conv.0.features.1"],
            'features_1.10.conv.0.features.1': ["features_1.10.conv.0.features.0",
                                                "features_1.10.conv.0.features.2"],
            'features_1.10.conv.0.features.2': ["features_1.10.conv.0.features.1",
                                                "features_1.10.conv.1.features.0"],
            'features_1.10.conv.1.features.0': ["features_1.10.conv.0.features.2",
                                                "features_1.10.conv.1.features.1"],
            'features_1.10.conv.1.features.1': ["features_1.10.conv.1.features.0",
                                                "features_1.10.conv.1.features.2"],
            'features_1.10.conv.1.features.2': ["features_1.10.conv.1.features.1", "features_1.10.conv.2"],
            'features_1.10.conv.2': ["features_1.10.conv.1.features.2", "features_1.10.conv.3"],
            'features_1.10.conv.3': ["features_1.10.conv.2", "features_1.11.conv.0.features.0"],

            'features_1.11.conv.0.features.0': ["features_1.10.conv.3", "features_1.11.conv.0.features.1"],
            'features_1.11.conv.0.features.1': ["features_1.11.conv.0.features.0",
                                                "features_1.11.conv.0.features.2"],
            'features_1.11.conv.0.features.2': ["features_1.11.conv.0.features.1",
                                                "features_1.11.conv.1.features.0"],
            'features_1.11.conv.1.features.0': ["features_1.11.conv.0.features.2",
                                                "features_1.11.conv.1.features.1"],
            'features_1.11.conv.1.features.1': ["features_1.11.conv.1.features.0",
                                                "features_1.11.conv.1.features.2"],
            'features_1.11.conv.1.features.2': ["features_1.11.conv.1.features.1", "features_1.11.conv.2"],
            'features_1.11.conv.2': ["features_1.11.conv.1.features.2", "features_1.11.conv.3"],
            'features_1.11.conv.3': ["features_1.11.conv.2",
                                     ["features_1.12.conv.0.features.0", 'features_1.13.conv.0.features.0']],

            'features_1.12.conv.0.features.0': ["features_1.11.conv.3", "features_1.12.conv.0.features.1"],
            'features_1.12.conv.0.features.1': ["features_1.12.conv.0.features.0",
                                                "features_1.12.conv.0.features.2"],
            'features_1.12.conv.0.features.2': ["features_1.12.conv.0.features.1",
                                                "features_1.12.conv.1.features.0"],
            'features_1.12.conv.1.features.0': ["features_1.12.conv.0.features.2",
                                                "features_1.12.conv.1.features.1"],
            'features_1.12.conv.1.features.1': ["features_1.12.conv.1.features.0",
                                                "features_1.12.conv.1.features.2"],
            'features_1.12.conv.1.features.2': ["features_1.12.conv.1.features.1", "features_1.12.conv.2"],
            'features_1.12.conv.2': ["features_1.12.conv.1.features.2", "features_1.12.conv.3"],
            'features_1.12.conv.3': ["features_1.12.conv.2",
                                     ["features_1.13.conv.0.features.0", 'expand_layer_conv_13.features.0']],

            'features_1.13.conv.0.features.0': [["features_1.12.conv.3", 'features_1.11.conv.3'],
                                                "features_1.13.conv.0.features.1"],
            'features_1.13.conv.0.features.1': ["features_1.13.conv.0.features.0",
                                                "features_1.13.conv.0.features.2"],
            'features_1.13.conv.0.features.2': ["features_1.13.conv.0.features.1",
                                                "features_1.13.conv.1.features.0"],
            'features_1.13.conv.1.features.0': ["features_1.13.conv.0.features.2",
                                                "features_1.13.conv.1.features.1"],
            'features_1.13.conv.1.features.1': ["features_1.13.conv.1.features.0",
                                                "features_1.13.conv.1.features.2"],
            'features_1.13.conv.1.features.2': ["features_1.13.conv.1.features.1", "features_1.13.conv.2"],
            'features_1.13.conv.2': ["features_1.13.conv.1.features.2", "features_1.13.conv.3"],
            'features_1.13.conv.3': ["features_1.13.conv.2", "expand_layer_conv_13.features.0"],

            'expand_layer_conv_13.features.0': [["features_1.13.conv.3", 'features_1.12.conv.3'],
                                                "expand_layer_conv_13.features.1"],
            'expand_layer_conv_13.features.1': ["expand_layer_conv_13.features.0",
                                                "expand_layer_conv_13.features.2"],
            'expand_layer_conv_13.features.2': ["expand_layer_conv_13.features.1",
                                                ["features_2.0.conv.0.features.0", 'OUTPUT2']],

            'features_2.0.conv.0.features.0': ["expand_layer_conv_13.features.2",
                                               "features_2.0.conv.0.features.1"],
            'features_2.0.conv.0.features.1': ["features_2.0.conv.0.features.0",
                                               "features_2.0.conv.0.features.2"],
            'features_2.0.conv.0.features.2': ["features_2.0.conv.0.features.1",
                                               "features_2.0.conv.1.features.0"],
            'features_2.0.conv.1.features.0': ["features_2.0.conv.0.features.2",
                                               "features_2.0.conv.1.features.1"],
            'features_2.0.conv.1.features.1': ["features_2.0.conv.1.features.0",
                                               "features_2.0.conv.1.features.2"],
            'features_2.0.conv.1.features.2': ["features_2.0.conv.1.features.1", "features_2.0.conv.2"],
            'features_2.0.conv.2': ["features_2.0.conv.1.features.2", "features_2.0.conv.3"],
            'features_2.0.conv.3': ["features_2.0.conv.2",
                                    ["features_2.1.conv.0.features.0", 'features_2.2.conv.0.features.0']],

            'features_2.1.conv.0.features.0': ["features_2.0.conv.3", "features_2.1.conv.0.features.1"],
            'features_2.1.conv.0.features.1': ["features_2.1.conv.0.features.0",
                                               "features_2.1.conv.0.features.2"],
            'features_2.1.conv.0.features.2': ["features_2.1.conv.0.features.1",
                                               "features_2.1.conv.1.features.0"],
            'features_2.1.conv.1.features.0': ["features_2.1.conv.0.features.2",
                                               "features_2.1.conv.1.features.1"],
            'features_2.1.conv.1.features.1': ["features_2.1.conv.1.features.0",
                                               "features_2.1.conv.1.features.2"],
            'features_2.1.conv.1.features.2': ["features_2.1.conv.1.features.1", "features_2.1.conv.2"],
            'features_2.1.conv.2': ["features_2.1.conv.1.features.2", "features_2.1.conv.3"],
            'features_2.1.conv.3': ["features_2.1.conv.2",
                                    ["features_2.2.conv.0.features.0", 'features_2.3.conv.0.features.0']],

            'features_2.2.conv.0.features.0': [["features_2.1.conv.3", 'features_2.0.conv.3'],
                                               "features_2.2.conv.0.features.1"],
            'features_2.2.conv.0.features.1': ["features_2.2.conv.0.features.0",
                                               "features_2.2.conv.0.features.2"],
            'features_2.2.conv.0.features.2': ["features_2.2.conv.0.features.1",
                                               "features_2.2.conv.1.features.0"],
            'features_2.2.conv.1.features.0': ["features_2.2.conv.0.features.2",
                                               "features_2.2.conv.1.features.1"],
            'features_2.2.conv.1.features.1': ["features_2.2.conv.1.features.0",
                                               "features_2.2.conv.1.features.2"],
            'features_2.2.conv.1.features.2': ["features_2.2.conv.1.features.1", "features_2.2.conv.2"],
            'features_2.2.conv.2': ["features_2.2.conv.1.features.2", "features_2.2.conv.3"],
            'features_2.2.conv.3': ["features_2.2.conv.2", "features_2.3.conv.0.features.0"],

            'features_2.3.conv.0.features.0': [["features_2.2.conv.3", 'features_2.1.conv.3'],
                                               "features_2.3.conv.0.features.1"],
            'features_2.3.conv.0.features.1': ["features_2.3.conv.0.features.0",
                                               "features_2.3.conv.0.features.2"],
            'features_2.3.conv.0.features.2': ["features_2.3.conv.0.features.1",
                                               "features_2.3.conv.1.features.0"],
            'features_2.3.conv.1.features.0': ["features_2.3.conv.0.features.2",
                                               "features_2.3.conv.1.features.1"],
            'features_2.3.conv.1.features.1': ["features_2.3.conv.1.features.0",
                                               "features_2.3.conv.1.features.2"],
            'features_2.3.conv.1.features.2': ["features_2.3.conv.1.features.1", "features_2.3.conv.2"],
            'features_2.3.conv.2': ["features_2.3.conv.1.features.2", "features_2.3.conv.3"],
            'features_2.3.conv.3': ["features_2.3.conv.2", "features_2.4.features.0"],

            'features_2.4.features.0': ["features_2.3.conv.3", "features_2.4.features.1"],
            'features_2.4.features.1': ["features_2.4.features.0", "features_2.4.features.2"],
            'features_2.4.features.2': ["features_2.4.features.1", "OUTPUT1"],
        }
        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

    def construct(self, x):
        out = self.features_1(x)
        expand_layer_conv_13 = self.expand_layer_conv_13(out)
        out = self.features_2(out)

        multi_feature = (expand_layer_conv_13, out)
        feature = out
        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature = multi_feature + (feature,)

        pred_loc, pred_label = self.multi_box(multi_feature)

        if not self.training:
            pred_label = ops.Sigmoid()(pred_label)
        pred_loc = ops.cast(pred_loc, mindspore.float32)
        pred_label = ops.cast(pred_label, mindspore.float32)

        return pred_loc, pred_label

    def get_out_channels(self):
        return self.last_channel

    def set_layers(self, layer_name, new_layer):
        if 'expand_layer_conv_13' == layer_name:
            self.expand_layer_conv_13 = new_layer
            self.layer_names["expand_layer_conv_13"] = new_layer
            self.origin_layer_names["expand_layer_conv_13"] = new_layer
        elif 'expand_layer_conv_13.features' == layer_name:
            self.expand_layer_conv_13.features = new_layer
            self.layer_names["expand_layer_conv_13.features"] = new_layer
            self.origin_layer_names["expand_layer_conv_13.features"] = new_layer
        elif 'expand_layer_conv_13.features.0' == layer_name:
            self.expand_layer_conv_13.features[0] = new_layer
            self.layer_names["expand_layer_conv_13.features.0"] = new_layer
            self.origin_layer_names["expand_layer_conv_13.features.0"] = new_layer
        elif 'expand_layer_conv_13.features.1' == layer_name:
            self.expand_layer_conv_13.features[1] = new_layer
            self.layer_names["expand_layer_conv_13.features.1"] = new_layer
            self.origin_layer_names["expand_layer_conv_13.features.1"] = new_layer
        elif 'expand_layer_conv_13.features.2' == layer_name:
            self.expand_layer_conv_13.features[2] = new_layer
            self.layer_names["expand_layer_conv_13.features.2"] = new_layer
            self.origin_layer_names["expand_layer_conv_13.features.2"] = new_layer
        elif 'features_1' == layer_name:
            self.features_1 = new_layer
            self.layer_names["features_1"] = new_layer
            self.origin_layer_names["features_1"] = new_layer
        elif 'features_1.0' == layer_name:
            self.features_1[0] = new_layer
            self.layer_names["features_1.0"] = new_layer
            self.origin_layer_names["features_1.0"] = new_layer
        elif 'features_1.0.features' == layer_name:
            self.features_1[0].features = new_layer
            self.layer_names["features_1.0.features"] = new_layer
            self.origin_layer_names["features_1.0.features"] = new_layer
        elif 'features_1.0.features.0' == layer_name:
            self.features_1[0].features[0] = new_layer
            self.layer_names["features_1.0.features.0"] = new_layer
            self.origin_layer_names["features_1.0.features.0"] = new_layer
        elif 'features_1.0.features.1' == layer_name:
            self.features_1[0].features[1] = new_layer
            self.layer_names["features_1.0.features.1"] = new_layer
            self.origin_layer_names["features_1.0.features.1"] = new_layer
        elif 'features_1.0.features.2' == layer_name:
            self.features_1[0].features[2] = new_layer
            self.layer_names["features_1.0.features.2"] = new_layer
            self.origin_layer_names["features_1.0.features.2"] = new_layer
        elif 'features_1.1' == layer_name:
            self.features_1[1] = new_layer
            self.layer_names["features_1.1"] = new_layer
            self.origin_layer_names["features_1.1"] = new_layer
        elif 'features_1.1.conv' == layer_name:
            self.features_1[1].conv = new_layer
            self.layer_names["features_1.1.conv"] = new_layer
            self.origin_layer_names["features_1.1.conv"] = new_layer
        elif 'features_1.1.conv.0' == layer_name:
            self.features_1[1].conv[0] = new_layer
            self.layer_names["features_1.1.conv.0"] = new_layer
            self.origin_layer_names["features_1.1.conv.0"] = new_layer
        elif 'features_1.1.conv.0.features' == layer_name:
            self.features_1[1].conv[0].features = new_layer
            self.layer_names["features_1.1.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.1.conv.0.features"] = new_layer
        elif 'features_1.1.conv.0.features.0' == layer_name:
            self.features_1[1].conv[0].features[0] = new_layer
            self.layer_names["features_1.1.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.1.conv.0.features.0"] = new_layer
        elif 'features_1.1.conv.0.features.1' == layer_name:
            self.features_1[1].conv[0].features[1] = new_layer
            self.layer_names["features_1.1.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.1.conv.0.features.1"] = new_layer
        elif 'features_1.1.conv.0.features.2' == layer_name:
            self.features_1[1].conv[0].features[2] = new_layer
            self.layer_names["features_1.1.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.1.conv.0.features.2"] = new_layer
        elif 'features_1.1.conv.1' == layer_name:
            self.features_1[1].conv[1] = new_layer
            self.layer_names["features_1.1.conv.1"] = new_layer
            self.origin_layer_names["features_1.1.conv.1"] = new_layer
        elif 'features_1.1.conv.2' == layer_name:
            self.features_1[1].conv[2] = new_layer
            self.layer_names["features_1.1.conv.2"] = new_layer
            self.origin_layer_names["features_1.1.conv.2"] = new_layer
        elif 'features_1.2' == layer_name:
            self.features_1[2] = new_layer
            self.layer_names["features_1.2"] = new_layer
            self.origin_layer_names["features_1.2"] = new_layer
        elif 'features_1.2.conv' == layer_name:
            self.features_1[2].conv = new_layer
            self.layer_names["features_1.2.conv"] = new_layer
            self.origin_layer_names["features_1.2.conv"] = new_layer
        elif 'features_1.2.conv.0' == layer_name:
            self.features_1[2].conv[0] = new_layer
            self.layer_names["features_1.2.conv.0"] = new_layer
            self.origin_layer_names["features_1.2.conv.0"] = new_layer
        elif 'features_1.2.conv.0.features' == layer_name:
            self.features_1[2].conv[0].features = new_layer
            self.layer_names["features_1.2.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.2.conv.0.features"] = new_layer
        elif 'features_1.2.conv.0.features.0' == layer_name:
            self.features_1[2].conv[0].features[0] = new_layer
            self.layer_names["features_1.2.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.2.conv.0.features.0"] = new_layer
        elif 'features_1.2.conv.0.features.1' == layer_name:
            self.features_1[2].conv[0].features[1] = new_layer
            self.layer_names["features_1.2.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.2.conv.0.features.1"] = new_layer
        elif 'features_1.2.conv.0.features.2' == layer_name:
            self.features_1[2].conv[0].features[2] = new_layer
            self.layer_names["features_1.2.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.2.conv.0.features.2"] = new_layer
        elif 'features_1.2.conv.1' == layer_name:
            self.features_1[2].conv[1] = new_layer
            self.layer_names["features_1.2.conv.1"] = new_layer
            self.origin_layer_names["features_1.2.conv.1"] = new_layer
        elif 'features_1.2.conv.1.features' == layer_name:
            self.features_1[2].conv[1].features = new_layer
            self.layer_names["features_1.2.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.2.conv.1.features"] = new_layer
        elif 'features_1.2.conv.1.features.0' == layer_name:
            self.features_1[2].conv[1].features[0] = new_layer
            self.layer_names["features_1.2.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.2.conv.1.features.0"] = new_layer
        elif 'features_1.2.conv.1.features.1' == layer_name:
            self.features_1[2].conv[1].features[1] = new_layer
            self.layer_names["features_1.2.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.2.conv.1.features.1"] = new_layer
        elif 'features_1.2.conv.1.features.2' == layer_name:
            self.features_1[2].conv[1].features[2] = new_layer
            self.layer_names["features_1.2.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.2.conv.1.features.2"] = new_layer
        elif 'features_1.2.conv.2' == layer_name:
            self.features_1[2].conv[2] = new_layer
            self.layer_names["features_1.2.conv.2"] = new_layer
            self.origin_layer_names["features_1.2.conv.2"] = new_layer
        elif 'features_1.2.conv.3' == layer_name:
            self.features_1[2].conv[3] = new_layer
            self.layer_names["features_1.2.conv.3"] = new_layer
            self.origin_layer_names["features_1.2.conv.3"] = new_layer
        elif 'features_1.3' == layer_name:
            self.features_1[3] = new_layer
            self.layer_names["features_1.3"] = new_layer
            self.origin_layer_names["features_1.3"] = new_layer
        elif 'features_1.3.conv' == layer_name:
            self.features_1[3].conv = new_layer
            self.layer_names["features_1.3.conv"] = new_layer
            self.origin_layer_names["features_1.3.conv"] = new_layer
        elif 'features_1.3.conv.0' == layer_name:
            self.features_1[3].conv[0] = new_layer
            self.layer_names["features_1.3.conv.0"] = new_layer
            self.origin_layer_names["features_1.3.conv.0"] = new_layer
        elif 'features_1.3.conv.0.features' == layer_name:
            self.features_1[3].conv[0].features = new_layer
            self.layer_names["features_1.3.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.3.conv.0.features"] = new_layer
        elif 'features_1.3.conv.0.features.0' == layer_name:
            self.features_1[3].conv[0].features[0] = new_layer
            self.layer_names["features_1.3.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.3.conv.0.features.0"] = new_layer
        elif 'features_1.3.conv.0.features.1' == layer_name:
            self.features_1[3].conv[0].features[1] = new_layer
            self.layer_names["features_1.3.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.3.conv.0.features.1"] = new_layer
        elif 'features_1.3.conv.0.features.2' == layer_name:
            self.features_1[3].conv[0].features[2] = new_layer
            self.layer_names["features_1.3.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.3.conv.0.features.2"] = new_layer
        elif 'features_1.3.conv.1' == layer_name:
            self.features_1[3].conv[1] = new_layer
            self.layer_names["features_1.3.conv.1"] = new_layer
            self.origin_layer_names["features_1.3.conv.1"] = new_layer
        elif 'features_1.3.conv.1.features' == layer_name:
            self.features_1[3].conv[1].features = new_layer
            self.layer_names["features_1.3.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.3.conv.1.features"] = new_layer
        elif 'features_1.3.conv.1.features.0' == layer_name:
            self.features_1[3].conv[1].features[0] = new_layer
            self.layer_names["features_1.3.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.3.conv.1.features.0"] = new_layer
        elif 'features_1.3.conv.1.features.1' == layer_name:
            self.features_1[3].conv[1].features[1] = new_layer
            self.layer_names["features_1.3.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.3.conv.1.features.1"] = new_layer
        elif 'features_1.3.conv.1.features.2' == layer_name:
            self.features_1[3].conv[1].features[2] = new_layer
            self.layer_names["features_1.3.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.3.conv.1.features.2"] = new_layer
        elif 'features_1.3.conv.2' == layer_name:
            self.features_1[3].conv[2] = new_layer
            self.layer_names["features_1.3.conv.2"] = new_layer
            self.origin_layer_names["features_1.3.conv.2"] = new_layer
        elif 'features_1.3.conv.3' == layer_name:
            self.features_1[3].conv[3] = new_layer
            self.layer_names["features_1.3.conv.3"] = new_layer
            self.origin_layer_names["features_1.3.conv.3"] = new_layer
        elif 'features_1.4' == layer_name:
            self.features_1[4] = new_layer
            self.layer_names["features_1.4"] = new_layer
            self.origin_layer_names["features_1.4"] = new_layer
        elif 'features_1.4.conv' == layer_name:
            self.features_1[4].conv = new_layer
            self.layer_names["features_1.4.conv"] = new_layer
            self.origin_layer_names["features_1.4.conv"] = new_layer
        elif 'features_1.4.conv.0' == layer_name:
            self.features_1[4].conv[0] = new_layer
            self.layer_names["features_1.4.conv.0"] = new_layer
            self.origin_layer_names["features_1.4.conv.0"] = new_layer
        elif 'features_1.4.conv.0.features' == layer_name:
            self.features_1[4].conv[0].features = new_layer
            self.layer_names["features_1.4.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.4.conv.0.features"] = new_layer
        elif 'features_1.4.conv.0.features.0' == layer_name:
            self.features_1[4].conv[0].features[0] = new_layer
            self.layer_names["features_1.4.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.4.conv.0.features.0"] = new_layer
        elif 'features_1.4.conv.0.features.1' == layer_name:
            self.features_1[4].conv[0].features[1] = new_layer
            self.layer_names["features_1.4.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.4.conv.0.features.1"] = new_layer
        elif 'features_1.4.conv.0.features.2' == layer_name:
            self.features_1[4].conv[0].features[2] = new_layer
            self.layer_names["features_1.4.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.4.conv.0.features.2"] = new_layer
        elif 'features_1.4.conv.1' == layer_name:
            self.features_1[4].conv[1] = new_layer
            self.layer_names["features_1.4.conv.1"] = new_layer
            self.origin_layer_names["features_1.4.conv.1"] = new_layer
        elif 'features_1.4.conv.1.features' == layer_name:
            self.features_1[4].conv[1].features = new_layer
            self.layer_names["features_1.4.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.4.conv.1.features"] = new_layer
        elif 'features_1.4.conv.1.features.0' == layer_name:
            self.features_1[4].conv[1].features[0] = new_layer
            self.layer_names["features_1.4.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.4.conv.1.features.0"] = new_layer
        elif 'features_1.4.conv.1.features.1' == layer_name:
            self.features_1[4].conv[1].features[1] = new_layer
            self.layer_names["features_1.4.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.4.conv.1.features.1"] = new_layer
        elif 'features_1.4.conv.1.features.2' == layer_name:
            self.features_1[4].conv[1].features[2] = new_layer
            self.layer_names["features_1.4.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.4.conv.1.features.2"] = new_layer
        elif 'features_1.4.conv.2' == layer_name:
            self.features_1[4].conv[2] = new_layer
            self.layer_names["features_1.4.conv.2"] = new_layer
            self.origin_layer_names["features_1.4.conv.2"] = new_layer
        elif 'features_1.4.conv.3' == layer_name:
            self.features_1[4].conv[3] = new_layer
            self.layer_names["features_1.4.conv.3"] = new_layer
            self.origin_layer_names["features_1.4.conv.3"] = new_layer
        elif 'features_1.5' == layer_name:
            self.features_1[5] = new_layer
            self.layer_names["features_1.5"] = new_layer
            self.origin_layer_names["features_1.5"] = new_layer
        elif 'features_1.5.conv' == layer_name:
            self.features_1[5].conv = new_layer
            self.layer_names["features_1.5.conv"] = new_layer
            self.origin_layer_names["features_1.5.conv"] = new_layer
        elif 'features_1.5.conv.0' == layer_name:
            self.features_1[5].conv[0] = new_layer
            self.layer_names["features_1.5.conv.0"] = new_layer
            self.origin_layer_names["features_1.5.conv.0"] = new_layer
        elif 'features_1.5.conv.0.features' == layer_name:
            self.features_1[5].conv[0].features = new_layer
            self.layer_names["features_1.5.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.5.conv.0.features"] = new_layer
        elif 'features_1.5.conv.0.features.0' == layer_name:
            self.features_1[5].conv[0].features[0] = new_layer
            self.layer_names["features_1.5.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.5.conv.0.features.0"] = new_layer
        elif 'features_1.5.conv.0.features.1' == layer_name:
            self.features_1[5].conv[0].features[1] = new_layer
            self.layer_names["features_1.5.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.5.conv.0.features.1"] = new_layer
        elif 'features_1.5.conv.0.features.2' == layer_name:
            self.features_1[5].conv[0].features[2] = new_layer
            self.layer_names["features_1.5.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.5.conv.0.features.2"] = new_layer
        elif 'features_1.5.conv.1' == layer_name:
            self.features_1[5].conv[1] = new_layer
            self.layer_names["features_1.5.conv.1"] = new_layer
            self.origin_layer_names["features_1.5.conv.1"] = new_layer
        elif 'features_1.5.conv.1.features' == layer_name:
            self.features_1[5].conv[1].features = new_layer
            self.layer_names["features_1.5.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.5.conv.1.features"] = new_layer
        elif 'features_1.5.conv.1.features.0' == layer_name:
            self.features_1[5].conv[1].features[0] = new_layer
            self.layer_names["features_1.5.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.5.conv.1.features.0"] = new_layer
        elif 'features_1.5.conv.1.features.1' == layer_name:
            self.features_1[5].conv[1].features[1] = new_layer
            self.layer_names["features_1.5.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.5.conv.1.features.1"] = new_layer
        elif 'features_1.5.conv.1.features.2' == layer_name:
            self.features_1[5].conv[1].features[2] = new_layer
            self.layer_names["features_1.5.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.5.conv.1.features.2"] = new_layer
        elif 'features_1.5.conv.2' == layer_name:
            self.features_1[5].conv[2] = new_layer
            self.layer_names["features_1.5.conv.2"] = new_layer
            self.origin_layer_names["features_1.5.conv.2"] = new_layer
        elif 'features_1.5.conv.3' == layer_name:
            self.features_1[5].conv[3] = new_layer
            self.layer_names["features_1.5.conv.3"] = new_layer
            self.origin_layer_names["features_1.5.conv.3"] = new_layer
        elif 'features_1.6' == layer_name:
            self.features_1[6] = new_layer
            self.layer_names["features_1.6"] = new_layer
            self.origin_layer_names["features_1.6"] = new_layer
        elif 'features_1.6.conv' == layer_name:
            self.features_1[6].conv = new_layer
            self.layer_names["features_1.6.conv"] = new_layer
            self.origin_layer_names["features_1.6.conv"] = new_layer
        elif 'features_1.6.conv.0' == layer_name:
            self.features_1[6].conv[0] = new_layer
            self.layer_names["features_1.6.conv.0"] = new_layer
            self.origin_layer_names["features_1.6.conv.0"] = new_layer
        elif 'features_1.6.conv.0.features' == layer_name:
            self.features_1[6].conv[0].features = new_layer
            self.layer_names["features_1.6.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.6.conv.0.features"] = new_layer
        elif 'features_1.6.conv.0.features.0' == layer_name:
            self.features_1[6].conv[0].features[0] = new_layer
            self.layer_names["features_1.6.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.6.conv.0.features.0"] = new_layer
        elif 'features_1.6.conv.0.features.1' == layer_name:
            self.features_1[6].conv[0].features[1] = new_layer
            self.layer_names["features_1.6.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.6.conv.0.features.1"] = new_layer
        elif 'features_1.6.conv.0.features.2' == layer_name:
            self.features_1[6].conv[0].features[2] = new_layer
            self.layer_names["features_1.6.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.6.conv.0.features.2"] = new_layer
        elif 'features_1.6.conv.1' == layer_name:
            self.features_1[6].conv[1] = new_layer
            self.layer_names["features_1.6.conv.1"] = new_layer
            self.origin_layer_names["features_1.6.conv.1"] = new_layer
        elif 'features_1.6.conv.1.features' == layer_name:
            self.features_1[6].conv[1].features = new_layer
            self.layer_names["features_1.6.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.6.conv.1.features"] = new_layer
        elif 'features_1.6.conv.1.features.0' == layer_name:
            self.features_1[6].conv[1].features[0] = new_layer
            self.layer_names["features_1.6.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.6.conv.1.features.0"] = new_layer
        elif 'features_1.6.conv.1.features.1' == layer_name:
            self.features_1[6].conv[1].features[1] = new_layer
            self.layer_names["features_1.6.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.6.conv.1.features.1"] = new_layer
        elif 'features_1.6.conv.1.features.2' == layer_name:
            self.features_1[6].conv[1].features[2] = new_layer
            self.layer_names["features_1.6.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.6.conv.1.features.2"] = new_layer
        elif 'features_1.6.conv.2' == layer_name:
            self.features_1[6].conv[2] = new_layer
            self.layer_names["features_1.6.conv.2"] = new_layer
            self.origin_layer_names["features_1.6.conv.2"] = new_layer
        elif 'features_1.6.conv.3' == layer_name:
            self.features_1[6].conv[3] = new_layer
            self.layer_names["features_1.6.conv.3"] = new_layer
            self.origin_layer_names["features_1.6.conv.3"] = new_layer
        elif 'features_1.7' == layer_name:
            self.features_1[7] = new_layer
            self.layer_names["features_1.7"] = new_layer
            self.origin_layer_names["features_1.7"] = new_layer
        elif 'features_1.7.conv' == layer_name:
            self.features_1[7].conv = new_layer
            self.layer_names["features_1.7.conv"] = new_layer
            self.origin_layer_names["features_1.7.conv"] = new_layer
        elif 'features_1.7.conv.0' == layer_name:
            self.features_1[7].conv[0] = new_layer
            self.layer_names["features_1.7.conv.0"] = new_layer
            self.origin_layer_names["features_1.7.conv.0"] = new_layer
        elif 'features_1.7.conv.0.features' == layer_name:
            self.features_1[7].conv[0].features = new_layer
            self.layer_names["features_1.7.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.7.conv.0.features"] = new_layer
        elif 'features_1.7.conv.0.features.0' == layer_name:
            self.features_1[7].conv[0].features[0] = new_layer
            self.layer_names["features_1.7.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.7.conv.0.features.0"] = new_layer
        elif 'features_1.7.conv.0.features.1' == layer_name:
            self.features_1[7].conv[0].features[1] = new_layer
            self.layer_names["features_1.7.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.7.conv.0.features.1"] = new_layer
        elif 'features_1.7.conv.0.features.2' == layer_name:
            self.features_1[7].conv[0].features[2] = new_layer
            self.layer_names["features_1.7.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.7.conv.0.features.2"] = new_layer
        elif 'features_1.7.conv.1' == layer_name:
            self.features_1[7].conv[1] = new_layer
            self.layer_names["features_1.7.conv.1"] = new_layer
            self.origin_layer_names["features_1.7.conv.1"] = new_layer
        elif 'features_1.7.conv.1.features' == layer_name:
            self.features_1[7].conv[1].features = new_layer
            self.layer_names["features_1.7.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.7.conv.1.features"] = new_layer
        elif 'features_1.7.conv.1.features.0' == layer_name:
            self.features_1[7].conv[1].features[0] = new_layer
            self.layer_names["features_1.7.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.7.conv.1.features.0"] = new_layer
        elif 'features_1.7.conv.1.features.1' == layer_name:
            self.features_1[7].conv[1].features[1] = new_layer
            self.layer_names["features_1.7.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.7.conv.1.features.1"] = new_layer
        elif 'features_1.7.conv.1.features.2' == layer_name:
            self.features_1[7].conv[1].features[2] = new_layer
            self.layer_names["features_1.7.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.7.conv.1.features.2"] = new_layer
        elif 'features_1.7.conv.2' == layer_name:
            self.features_1[7].conv[2] = new_layer
            self.layer_names["features_1.7.conv.2"] = new_layer
            self.origin_layer_names["features_1.7.conv.2"] = new_layer
        elif 'features_1.7.conv.3' == layer_name:
            self.features_1[7].conv[3] = new_layer
            self.layer_names["features_1.7.conv.3"] = new_layer
            self.origin_layer_names["features_1.7.conv.3"] = new_layer
        elif 'features_1.8' == layer_name:
            self.features_1[8] = new_layer
            self.layer_names["features_1.8"] = new_layer
            self.origin_layer_names["features_1.8"] = new_layer
        elif 'features_1.8.conv' == layer_name:
            self.features_1[8].conv = new_layer
            self.layer_names["features_1.8.conv"] = new_layer
            self.origin_layer_names["features_1.8.conv"] = new_layer
        elif 'features_1.8.conv.0' == layer_name:
            self.features_1[8].conv[0] = new_layer
            self.layer_names["features_1.8.conv.0"] = new_layer
            self.origin_layer_names["features_1.8.conv.0"] = new_layer
        elif 'features_1.8.conv.0.features' == layer_name:
            self.features_1[8].conv[0].features = new_layer
            self.layer_names["features_1.8.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.8.conv.0.features"] = new_layer
        elif 'features_1.8.conv.0.features.0' == layer_name:
            self.features_1[8].conv[0].features[0] = new_layer
            self.layer_names["features_1.8.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.8.conv.0.features.0"] = new_layer
        elif 'features_1.8.conv.0.features.1' == layer_name:
            self.features_1[8].conv[0].features[1] = new_layer
            self.layer_names["features_1.8.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.8.conv.0.features.1"] = new_layer
        elif 'features_1.8.conv.0.features.2' == layer_name:
            self.features_1[8].conv[0].features[2] = new_layer
            self.layer_names["features_1.8.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.8.conv.0.features.2"] = new_layer
        elif 'features_1.8.conv.1' == layer_name:
            self.features_1[8].conv[1] = new_layer
            self.layer_names["features_1.8.conv.1"] = new_layer
            self.origin_layer_names["features_1.8.conv.1"] = new_layer
        elif 'features_1.8.conv.1.features' == layer_name:
            self.features_1[8].conv[1].features = new_layer
            self.layer_names["features_1.8.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.8.conv.1.features"] = new_layer
        elif 'features_1.8.conv.1.features.0' == layer_name:
            self.features_1[8].conv[1].features[0] = new_layer
            self.layer_names["features_1.8.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.8.conv.1.features.0"] = new_layer
        elif 'features_1.8.conv.1.features.1' == layer_name:
            self.features_1[8].conv[1].features[1] = new_layer
            self.layer_names["features_1.8.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.8.conv.1.features.1"] = new_layer
        elif 'features_1.8.conv.1.features.2' == layer_name:
            self.features_1[8].conv[1].features[2] = new_layer
            self.layer_names["features_1.8.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.8.conv.1.features.2"] = new_layer
        elif 'features_1.8.conv.2' == layer_name:
            self.features_1[8].conv[2] = new_layer
            self.layer_names["features_1.8.conv.2"] = new_layer
            self.origin_layer_names["features_1.8.conv.2"] = new_layer
        elif 'features_1.8.conv.3' == layer_name:
            self.features_1[8].conv[3] = new_layer
            self.layer_names["features_1.8.conv.3"] = new_layer
            self.origin_layer_names["features_1.8.conv.3"] = new_layer
        elif 'features_1.9' == layer_name:
            self.features_1[9] = new_layer
            self.layer_names["features_1.9"] = new_layer
            self.origin_layer_names["features_1.9"] = new_layer
        elif 'features_1.9.conv' == layer_name:
            self.features_1[9].conv = new_layer
            self.layer_names["features_1.9.conv"] = new_layer
            self.origin_layer_names["features_1.9.conv"] = new_layer
        elif 'features_1.9.conv.0' == layer_name:
            self.features_1[9].conv[0] = new_layer
            self.layer_names["features_1.9.conv.0"] = new_layer
            self.origin_layer_names["features_1.9.conv.0"] = new_layer
        elif 'features_1.9.conv.0.features' == layer_name:
            self.features_1[9].conv[0].features = new_layer
            self.layer_names["features_1.9.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.9.conv.0.features"] = new_layer
        elif 'features_1.9.conv.0.features.0' == layer_name:
            self.features_1[9].conv[0].features[0] = new_layer
            self.layer_names["features_1.9.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.9.conv.0.features.0"] = new_layer
        elif 'features_1.9.conv.0.features.1' == layer_name:
            self.features_1[9].conv[0].features[1] = new_layer
            self.layer_names["features_1.9.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.9.conv.0.features.1"] = new_layer
        elif 'features_1.9.conv.0.features.2' == layer_name:
            self.features_1[9].conv[0].features[2] = new_layer
            self.layer_names["features_1.9.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.9.conv.0.features.2"] = new_layer
        elif 'features_1.9.conv.1' == layer_name:
            self.features_1[9].conv[1] = new_layer
            self.layer_names["features_1.9.conv.1"] = new_layer
            self.origin_layer_names["features_1.9.conv.1"] = new_layer
        elif 'features_1.9.conv.1.features' == layer_name:
            self.features_1[9].conv[1].features = new_layer
            self.layer_names["features_1.9.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.9.conv.1.features"] = new_layer
        elif 'features_1.9.conv.1.features.0' == layer_name:
            self.features_1[9].conv[1].features[0] = new_layer
            self.layer_names["features_1.9.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.9.conv.1.features.0"] = new_layer
        elif 'features_1.9.conv.1.features.1' == layer_name:
            self.features_1[9].conv[1].features[1] = new_layer
            self.layer_names["features_1.9.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.9.conv.1.features.1"] = new_layer
        elif 'features_1.9.conv.1.features.2' == layer_name:
            self.features_1[9].conv[1].features[2] = new_layer
            self.layer_names["features_1.9.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.9.conv.1.features.2"] = new_layer
        elif 'features_1.9.conv.2' == layer_name:
            self.features_1[9].conv[2] = new_layer
            self.layer_names["features_1.9.conv.2"] = new_layer
            self.origin_layer_names["features_1.9.conv.2"] = new_layer
        elif 'features_1.9.conv.3' == layer_name:
            self.features_1[9].conv[3] = new_layer
            self.layer_names["features_1.9.conv.3"] = new_layer
            self.origin_layer_names["features_1.9.conv.3"] = new_layer
        elif 'features_1.10' == layer_name:
            self.features_1[10] = new_layer
            self.layer_names["features_1.10"] = new_layer
            self.origin_layer_names["features_1.10"] = new_layer
        elif 'features_1.10.conv' == layer_name:
            self.features_1[10].conv = new_layer
            self.layer_names["features_1.10.conv"] = new_layer
            self.origin_layer_names["features_1.10.conv"] = new_layer
        elif 'features_1.10.conv.0' == layer_name:
            self.features_1[10].conv[0] = new_layer
            self.layer_names["features_1.10.conv.0"] = new_layer
            self.origin_layer_names["features_1.10.conv.0"] = new_layer
        elif 'features_1.10.conv.0.features' == layer_name:
            self.features_1[10].conv[0].features = new_layer
            self.layer_names["features_1.10.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.10.conv.0.features"] = new_layer
        elif 'features_1.10.conv.0.features.0' == layer_name:
            self.features_1[10].conv[0].features[0] = new_layer
            self.layer_names["features_1.10.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.10.conv.0.features.0"] = new_layer
        elif 'features_1.10.conv.0.features.1' == layer_name:
            self.features_1[10].conv[0].features[1] = new_layer
            self.layer_names["features_1.10.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.10.conv.0.features.1"] = new_layer
        elif 'features_1.10.conv.0.features.2' == layer_name:
            self.features_1[10].conv[0].features[2] = new_layer
            self.layer_names["features_1.10.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.10.conv.0.features.2"] = new_layer
        elif 'features_1.10.conv.1' == layer_name:
            self.features_1[10].conv[1] = new_layer
            self.layer_names["features_1.10.conv.1"] = new_layer
            self.origin_layer_names["features_1.10.conv.1"] = new_layer
        elif 'features_1.10.conv.1.features' == layer_name:
            self.features_1[10].conv[1].features = new_layer
            self.layer_names["features_1.10.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.10.conv.1.features"] = new_layer
        elif 'features_1.10.conv.1.features.0' == layer_name:
            self.features_1[10].conv[1].features[0] = new_layer
            self.layer_names["features_1.10.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.10.conv.1.features.0"] = new_layer
        elif 'features_1.10.conv.1.features.1' == layer_name:
            self.features_1[10].conv[1].features[1] = new_layer
            self.layer_names["features_1.10.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.10.conv.1.features.1"] = new_layer
        elif 'features_1.10.conv.1.features.2' == layer_name:
            self.features_1[10].conv[1].features[2] = new_layer
            self.layer_names["features_1.10.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.10.conv.1.features.2"] = new_layer
        elif 'features_1.10.conv.2' == layer_name:
            self.features_1[10].conv[2] = new_layer
            self.layer_names["features_1.10.conv.2"] = new_layer
            self.origin_layer_names["features_1.10.conv.2"] = new_layer
        elif 'features_1.10.conv.3' == layer_name:
            self.features_1[10].conv[3] = new_layer
            self.layer_names["features_1.10.conv.3"] = new_layer
            self.origin_layer_names["features_1.10.conv.3"] = new_layer
        elif 'features_1.11' == layer_name:
            self.features_1[11] = new_layer
            self.layer_names["features_1.11"] = new_layer
            self.origin_layer_names["features_1.11"] = new_layer
        elif 'features_1.11.conv' == layer_name:
            self.features_1[11].conv = new_layer
            self.layer_names["features_1.11.conv"] = new_layer
            self.origin_layer_names["features_1.11.conv"] = new_layer
        elif 'features_1.11.conv.0' == layer_name:
            self.features_1[11].conv[0] = new_layer
            self.layer_names["features_1.11.conv.0"] = new_layer
            self.origin_layer_names["features_1.11.conv.0"] = new_layer
        elif 'features_1.11.conv.0.features' == layer_name:
            self.features_1[11].conv[0].features = new_layer
            self.layer_names["features_1.11.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.11.conv.0.features"] = new_layer
        elif 'features_1.11.conv.0.features.0' == layer_name:
            self.features_1[11].conv[0].features[0] = new_layer
            self.layer_names["features_1.11.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.11.conv.0.features.0"] = new_layer
        elif 'features_1.11.conv.0.features.1' == layer_name:
            self.features_1[11].conv[0].features[1] = new_layer
            self.layer_names["features_1.11.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.11.conv.0.features.1"] = new_layer
        elif 'features_1.11.conv.0.features.2' == layer_name:
            self.features_1[11].conv[0].features[2] = new_layer
            self.layer_names["features_1.11.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.11.conv.0.features.2"] = new_layer
        elif 'features_1.11.conv.1' == layer_name:
            self.features_1[11].conv[1] = new_layer
            self.layer_names["features_1.11.conv.1"] = new_layer
            self.origin_layer_names["features_1.11.conv.1"] = new_layer
        elif 'features_1.11.conv.1.features' == layer_name:
            self.features_1[11].conv[1].features = new_layer
            self.layer_names["features_1.11.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.11.conv.1.features"] = new_layer
        elif 'features_1.11.conv.1.features.0' == layer_name:
            self.features_1[11].conv[1].features[0] = new_layer
            self.layer_names["features_1.11.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.11.conv.1.features.0"] = new_layer
        elif 'features_1.11.conv.1.features.1' == layer_name:
            self.features_1[11].conv[1].features[1] = new_layer
            self.layer_names["features_1.11.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.11.conv.1.features.1"] = new_layer
        elif 'features_1.11.conv.1.features.2' == layer_name:
            self.features_1[11].conv[1].features[2] = new_layer
            self.layer_names["features_1.11.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.11.conv.1.features.2"] = new_layer
        elif 'features_1.11.conv.2' == layer_name:
            self.features_1[11].conv[2] = new_layer
            self.layer_names["features_1.11.conv.2"] = new_layer
            self.origin_layer_names["features_1.11.conv.2"] = new_layer
        elif 'features_1.11.conv.3' == layer_name:
            self.features_1[11].conv[3] = new_layer
            self.layer_names["features_1.11.conv.3"] = new_layer
            self.origin_layer_names["features_1.11.conv.3"] = new_layer
        elif 'features_1.12' == layer_name:
            self.features_1[12] = new_layer
            self.layer_names["features_1.12"] = new_layer
            self.origin_layer_names["features_1.12"] = new_layer
        elif 'features_1.12.conv' == layer_name:
            self.features_1[12].conv = new_layer
            self.layer_names["features_1.12.conv"] = new_layer
            self.origin_layer_names["features_1.12.conv"] = new_layer
        elif 'features_1.12.conv.0' == layer_name:
            self.features_1[12].conv[0] = new_layer
            self.layer_names["features_1.12.conv.0"] = new_layer
            self.origin_layer_names["features_1.12.conv.0"] = new_layer
        elif 'features_1.12.conv.0.features' == layer_name:
            self.features_1[12].conv[0].features = new_layer
            self.layer_names["features_1.12.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.12.conv.0.features"] = new_layer
        elif 'features_1.12.conv.0.features.0' == layer_name:
            self.features_1[12].conv[0].features[0] = new_layer
            self.layer_names["features_1.12.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.12.conv.0.features.0"] = new_layer
        elif 'features_1.12.conv.0.features.1' == layer_name:
            self.features_1[12].conv[0].features[1] = new_layer
            self.layer_names["features_1.12.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.12.conv.0.features.1"] = new_layer
        elif 'features_1.12.conv.0.features.2' == layer_name:
            self.features_1[12].conv[0].features[2] = new_layer
            self.layer_names["features_1.12.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.12.conv.0.features.2"] = new_layer
        elif 'features_1.12.conv.1' == layer_name:
            self.features_1[12].conv[1] = new_layer
            self.layer_names["features_1.12.conv.1"] = new_layer
            self.origin_layer_names["features_1.12.conv.1"] = new_layer
        elif 'features_1.12.conv.1.features' == layer_name:
            self.features_1[12].conv[1].features = new_layer
            self.layer_names["features_1.12.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.12.conv.1.features"] = new_layer
        elif 'features_1.12.conv.1.features.0' == layer_name:
            self.features_1[12].conv[1].features[0] = new_layer
            self.layer_names["features_1.12.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.12.conv.1.features.0"] = new_layer
        elif 'features_1.12.conv.1.features.1' == layer_name:
            self.features_1[12].conv[1].features[1] = new_layer
            self.layer_names["features_1.12.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.12.conv.1.features.1"] = new_layer
        elif 'features_1.12.conv.1.features.2' == layer_name:
            self.features_1[12].conv[1].features[2] = new_layer
            self.layer_names["features_1.12.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.12.conv.1.features.2"] = new_layer
        elif 'features_1.12.conv.2' == layer_name:
            self.features_1[12].conv[2] = new_layer
            self.layer_names["features_1.12.conv.2"] = new_layer
            self.origin_layer_names["features_1.12.conv.2"] = new_layer
        elif 'features_1.12.conv.3' == layer_name:
            self.features_1[12].conv[3] = new_layer
            self.layer_names["features_1.12.conv.3"] = new_layer
            self.origin_layer_names["features_1.12.conv.3"] = new_layer
        elif 'features_1.13' == layer_name:
            self.features_1[13] = new_layer
            self.layer_names["features_1.13"] = new_layer
            self.origin_layer_names["features_1.13"] = new_layer
        elif 'features_1.13.conv' == layer_name:
            self.features_1[13].conv = new_layer
            self.layer_names["features_1.13.conv"] = new_layer
            self.origin_layer_names["features_1.13.conv"] = new_layer
        elif 'features_1.13.conv.0' == layer_name:
            self.features_1[13].conv[0] = new_layer
            self.layer_names["features_1.13.conv.0"] = new_layer
            self.origin_layer_names["features_1.13.conv.0"] = new_layer
        elif 'features_1.13.conv.0.features' == layer_name:
            self.features_1[13].conv[0].features = new_layer
            self.layer_names["features_1.13.conv.0.features"] = new_layer
            self.origin_layer_names["features_1.13.conv.0.features"] = new_layer
        elif 'features_1.13.conv.0.features.0' == layer_name:
            self.features_1[13].conv[0].features[0] = new_layer
            self.layer_names["features_1.13.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_1.13.conv.0.features.0"] = new_layer
        elif 'features_1.13.conv.0.features.1' == layer_name:
            self.features_1[13].conv[0].features[1] = new_layer
            self.layer_names["features_1.13.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_1.13.conv.0.features.1"] = new_layer
        elif 'features_1.13.conv.0.features.2' == layer_name:
            self.features_1[13].conv[0].features[2] = new_layer
            self.layer_names["features_1.13.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_1.13.conv.0.features.2"] = new_layer
        elif 'features_1.13.conv.1' == layer_name:
            self.features_1[13].conv[1] = new_layer
            self.layer_names["features_1.13.conv.1"] = new_layer
            self.origin_layer_names["features_1.13.conv.1"] = new_layer
        elif 'features_1.13.conv.1.features' == layer_name:
            self.features_1[13].conv[1].features = new_layer
            self.layer_names["features_1.13.conv.1.features"] = new_layer
            self.origin_layer_names["features_1.13.conv.1.features"] = new_layer
        elif 'features_1.13.conv.1.features.0' == layer_name:
            self.features_1[13].conv[1].features[0] = new_layer
            self.layer_names["features_1.13.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_1.13.conv.1.features.0"] = new_layer
        elif 'features_1.13.conv.1.features.1' == layer_name:
            self.features_1[13].conv[1].features[1] = new_layer
            self.layer_names["features_1.13.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_1.13.conv.1.features.1"] = new_layer
        elif 'features_1.13.conv.1.features.2' == layer_name:
            self.features_1[13].conv[1].features[2] = new_layer
            self.layer_names["features_1.13.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_1.13.conv.1.features.2"] = new_layer
        elif 'features_1.13.conv.2' == layer_name:
            self.features_1[13].conv[2] = new_layer
            self.layer_names["features_1.13.conv.2"] = new_layer
            self.origin_layer_names["features_1.13.conv.2"] = new_layer
        elif 'features_1.13.conv.3' == layer_name:
            self.features_1[13].conv[3] = new_layer
            self.layer_names["features_1.13.conv.3"] = new_layer
            self.origin_layer_names["features_1.13.conv.3"] = new_layer
        elif 'features_2' == layer_name:
            self.features_2 = new_layer
            self.layer_names["features_2"] = new_layer
            self.origin_layer_names["features_2"] = new_layer
        elif 'features_2.0' == layer_name:
            self.features_2[0] = new_layer
            self.layer_names["features_2.0"] = new_layer
            self.origin_layer_names["features_2.0"] = new_layer
        elif 'features_2.0.conv' == layer_name:
            self.features_2[0].conv = new_layer
            self.layer_names["features_2.0.conv"] = new_layer
            self.origin_layer_names["features_2.0.conv"] = new_layer
        elif 'features_2.0.conv.0' == layer_name:
            self.features_2[0].conv[0] = new_layer
            self.layer_names["features_2.0.conv.0"] = new_layer
            self.origin_layer_names["features_2.0.conv.0"] = new_layer
        elif 'features_2.0.conv.0.features' == layer_name:
            self.features_2[0].conv[0].features = new_layer
            self.layer_names["features_2.0.conv.0.features"] = new_layer
            self.origin_layer_names["features_2.0.conv.0.features"] = new_layer
        elif 'features_2.0.conv.0.features.0' == layer_name:
            self.features_2[0].conv[0].features[0] = new_layer
            self.layer_names["features_2.0.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_2.0.conv.0.features.0"] = new_layer
        elif 'features_2.0.conv.0.features.1' == layer_name:
            self.features_2[0].conv[0].features[1] = new_layer
            self.layer_names["features_2.0.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_2.0.conv.0.features.1"] = new_layer
        elif 'features_2.0.conv.0.features.2' == layer_name:
            self.features_2[0].conv[0].features[2] = new_layer
            self.layer_names["features_2.0.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_2.0.conv.0.features.2"] = new_layer
        elif 'features_2.0.conv.1' == layer_name:
            self.features_2[0].conv[1] = new_layer
            self.layer_names["features_2.0.conv.1"] = new_layer
            self.origin_layer_names["features_2.0.conv.1"] = new_layer
        elif 'features_2.0.conv.1.features' == layer_name:
            self.features_2[0].conv[1].features = new_layer
            self.layer_names["features_2.0.conv.1.features"] = new_layer
            self.origin_layer_names["features_2.0.conv.1.features"] = new_layer
        elif 'features_2.0.conv.1.features.0' == layer_name:
            self.features_2[0].conv[1].features[0] = new_layer
            self.layer_names["features_2.0.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_2.0.conv.1.features.0"] = new_layer
        elif 'features_2.0.conv.1.features.1' == layer_name:
            self.features_2[0].conv[1].features[1] = new_layer
            self.layer_names["features_2.0.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_2.0.conv.1.features.1"] = new_layer
        elif 'features_2.0.conv.1.features.2' == layer_name:
            self.features_2[0].conv[1].features[2] = new_layer
            self.layer_names["features_2.0.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_2.0.conv.1.features.2"] = new_layer
        elif 'features_2.0.conv.2' == layer_name:
            self.features_2[0].conv[2] = new_layer
            self.layer_names["features_2.0.conv.2"] = new_layer
            self.origin_layer_names["features_2.0.conv.2"] = new_layer
        elif 'features_2.0.conv.3' == layer_name:
            self.features_2[0].conv[3] = new_layer
            self.layer_names["features_2.0.conv.3"] = new_layer
            self.origin_layer_names["features_2.0.conv.3"] = new_layer
        elif 'features_2.1' == layer_name:
            self.features_2[1] = new_layer
            self.layer_names["features_2.1"] = new_layer
            self.origin_layer_names["features_2.1"] = new_layer
        elif 'features_2.1.conv' == layer_name:
            self.features_2[1].conv = new_layer
            self.layer_names["features_2.1.conv"] = new_layer
            self.origin_layer_names["features_2.1.conv"] = new_layer
        elif 'features_2.1.conv.0' == layer_name:
            self.features_2[1].conv[0] = new_layer
            self.layer_names["features_2.1.conv.0"] = new_layer
            self.origin_layer_names["features_2.1.conv.0"] = new_layer
        elif 'features_2.1.conv.0.features' == layer_name:
            self.features_2[1].conv[0].features = new_layer
            self.layer_names["features_2.1.conv.0.features"] = new_layer
            self.origin_layer_names["features_2.1.conv.0.features"] = new_layer
        elif 'features_2.1.conv.0.features.0' == layer_name:
            self.features_2[1].conv[0].features[0] = new_layer
            self.layer_names["features_2.1.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_2.1.conv.0.features.0"] = new_layer
        elif 'features_2.1.conv.0.features.1' == layer_name:
            self.features_2[1].conv[0].features[1] = new_layer
            self.layer_names["features_2.1.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_2.1.conv.0.features.1"] = new_layer
        elif 'features_2.1.conv.0.features.2' == layer_name:
            self.features_2[1].conv[0].features[2] = new_layer
            self.layer_names["features_2.1.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_2.1.conv.0.features.2"] = new_layer
        elif 'features_2.1.conv.1' == layer_name:
            self.features_2[1].conv[1] = new_layer
            self.layer_names["features_2.1.conv.1"] = new_layer
            self.origin_layer_names["features_2.1.conv.1"] = new_layer
        elif 'features_2.1.conv.1.features' == layer_name:
            self.features_2[1].conv[1].features = new_layer
            self.layer_names["features_2.1.conv.1.features"] = new_layer
            self.origin_layer_names["features_2.1.conv.1.features"] = new_layer
        elif 'features_2.1.conv.1.features.0' == layer_name:
            self.features_2[1].conv[1].features[0] = new_layer
            self.layer_names["features_2.1.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_2.1.conv.1.features.0"] = new_layer
        elif 'features_2.1.conv.1.features.1' == layer_name:
            self.features_2[1].conv[1].features[1] = new_layer
            self.layer_names["features_2.1.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_2.1.conv.1.features.1"] = new_layer
        elif 'features_2.1.conv.1.features.2' == layer_name:
            self.features_2[1].conv[1].features[2] = new_layer
            self.layer_names["features_2.1.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_2.1.conv.1.features.2"] = new_layer
        elif 'features_2.1.conv.2' == layer_name:
            self.features_2[1].conv[2] = new_layer
            self.layer_names["features_2.1.conv.2"] = new_layer
            self.origin_layer_names["features_2.1.conv.2"] = new_layer
        elif 'features_2.1.conv.3' == layer_name:
            self.features_2[1].conv[3] = new_layer
            self.layer_names["features_2.1.conv.3"] = new_layer
            self.origin_layer_names["features_2.1.conv.3"] = new_layer
        elif 'features_2.2' == layer_name:
            self.features_2[2] = new_layer
            self.layer_names["features_2.2"] = new_layer
            self.origin_layer_names["features_2.2"] = new_layer
        elif 'features_2.2.conv' == layer_name:
            self.features_2[2].conv = new_layer
            self.layer_names["features_2.2.conv"] = new_layer
            self.origin_layer_names["features_2.2.conv"] = new_layer
        elif 'features_2.2.conv.0' == layer_name:
            self.features_2[2].conv[0] = new_layer
            self.layer_names["features_2.2.conv.0"] = new_layer
            self.origin_layer_names["features_2.2.conv.0"] = new_layer
        elif 'features_2.2.conv.0.features' == layer_name:
            self.features_2[2].conv[0].features = new_layer
            self.layer_names["features_2.2.conv.0.features"] = new_layer
            self.origin_layer_names["features_2.2.conv.0.features"] = new_layer
        elif 'features_2.2.conv.0.features.0' == layer_name:
            self.features_2[2].conv[0].features[0] = new_layer
            self.layer_names["features_2.2.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_2.2.conv.0.features.0"] = new_layer
        elif 'features_2.2.conv.0.features.1' == layer_name:
            self.features_2[2].conv[0].features[1] = new_layer
            self.layer_names["features_2.2.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_2.2.conv.0.features.1"] = new_layer
        elif 'features_2.2.conv.0.features.2' == layer_name:
            self.features_2[2].conv[0].features[2] = new_layer
            self.layer_names["features_2.2.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_2.2.conv.0.features.2"] = new_layer
        elif 'features_2.2.conv.1' == layer_name:
            self.features_2[2].conv[1] = new_layer
            self.layer_names["features_2.2.conv.1"] = new_layer
            self.origin_layer_names["features_2.2.conv.1"] = new_layer
        elif 'features_2.2.conv.1.features' == layer_name:
            self.features_2[2].conv[1].features = new_layer
            self.layer_names["features_2.2.conv.1.features"] = new_layer
            self.origin_layer_names["features_2.2.conv.1.features"] = new_layer
        elif 'features_2.2.conv.1.features.0' == layer_name:
            self.features_2[2].conv[1].features[0] = new_layer
            self.layer_names["features_2.2.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_2.2.conv.1.features.0"] = new_layer
        elif 'features_2.2.conv.1.features.1' == layer_name:
            self.features_2[2].conv[1].features[1] = new_layer
            self.layer_names["features_2.2.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_2.2.conv.1.features.1"] = new_layer
        elif 'features_2.2.conv.1.features.2' == layer_name:
            self.features_2[2].conv[1].features[2] = new_layer
            self.layer_names["features_2.2.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_2.2.conv.1.features.2"] = new_layer
        elif 'features_2.2.conv.2' == layer_name:
            self.features_2[2].conv[2] = new_layer
            self.layer_names["features_2.2.conv.2"] = new_layer
            self.origin_layer_names["features_2.2.conv.2"] = new_layer
        elif 'features_2.2.conv.3' == layer_name:
            self.features_2[2].conv[3] = new_layer
            self.layer_names["features_2.2.conv.3"] = new_layer
            self.origin_layer_names["features_2.2.conv.3"] = new_layer
        elif 'features_2.3' == layer_name:
            self.features_2[3] = new_layer
            self.layer_names["features_2.3"] = new_layer
            self.origin_layer_names["features_2.3"] = new_layer
        elif 'features_2.3.conv' == layer_name:
            self.features_2[3].conv = new_layer
            self.layer_names["features_2.3.conv"] = new_layer
            self.origin_layer_names["features_2.3.conv"] = new_layer
        elif 'features_2.3.conv.0' == layer_name:
            self.features_2[3].conv[0] = new_layer
            self.layer_names["features_2.3.conv.0"] = new_layer
            self.origin_layer_names["features_2.3.conv.0"] = new_layer
        elif 'features_2.3.conv.0.features' == layer_name:
            self.features_2[3].conv[0].features = new_layer
            self.layer_names["features_2.3.conv.0.features"] = new_layer
            self.origin_layer_names["features_2.3.conv.0.features"] = new_layer
        elif 'features_2.3.conv.0.features.0' == layer_name:
            self.features_2[3].conv[0].features[0] = new_layer
            self.layer_names["features_2.3.conv.0.features.0"] = new_layer
            self.origin_layer_names["features_2.3.conv.0.features.0"] = new_layer
        elif 'features_2.3.conv.0.features.1' == layer_name:
            self.features_2[3].conv[0].features[1] = new_layer
            self.layer_names["features_2.3.conv.0.features.1"] = new_layer
            self.origin_layer_names["features_2.3.conv.0.features.1"] = new_layer
        elif 'features_2.3.conv.0.features.2' == layer_name:
            self.features_2[3].conv[0].features[2] = new_layer
            self.layer_names["features_2.3.conv.0.features.2"] = new_layer
            self.origin_layer_names["features_2.3.conv.0.features.2"] = new_layer
        elif 'features_2.3.conv.1' == layer_name:
            self.features_2[3].conv[1] = new_layer
            self.layer_names["features_2.3.conv.1"] = new_layer
            self.origin_layer_names["features_2.3.conv.1"] = new_layer
        elif 'features_2.3.conv.1.features' == layer_name:
            self.features_2[3].conv[1].features = new_layer
            self.layer_names["features_2.3.conv.1.features"] = new_layer
            self.origin_layer_names["features_2.3.conv.1.features"] = new_layer
        elif 'features_2.3.conv.1.features.0' == layer_name:
            self.features_2[3].conv[1].features[0] = new_layer
            self.layer_names["features_2.3.conv.1.features.0"] = new_layer
            self.origin_layer_names["features_2.3.conv.1.features.0"] = new_layer
        elif 'features_2.3.conv.1.features.1' == layer_name:
            self.features_2[3].conv[1].features[1] = new_layer
            self.layer_names["features_2.3.conv.1.features.1"] = new_layer
            self.origin_layer_names["features_2.3.conv.1.features.1"] = new_layer
        elif 'features_2.3.conv.1.features.2' == layer_name:
            self.features_2[3].conv[1].features[2] = new_layer
            self.layer_names["features_2.3.conv.1.features.2"] = new_layer
            self.origin_layer_names["features_2.3.conv.1.features.2"] = new_layer
        elif 'features_2.3.conv.2' == layer_name:
            self.features_2[3].conv[2] = new_layer
            self.layer_names["features_2.3.conv.2"] = new_layer
            self.origin_layer_names["features_2.3.conv.2"] = new_layer
        elif 'features_2.3.conv.3' == layer_name:
            self.features_2[3].conv[3] = new_layer
            self.layer_names["features_2.3.conv.3"] = new_layer
            self.origin_layer_names["features_2.3.conv.3"] = new_layer
        elif 'features_2.4' == layer_name:
            self.features_2[4] = new_layer
            self.layer_names["features_2.4"] = new_layer
            self.origin_layer_names["features_2.4"] = new_layer
        elif 'features_2.4.features' == layer_name:
            self.features_2[4].features = new_layer
            self.layer_names["features_2.4.features"] = new_layer
            self.origin_layer_names["features_2.4.features"] = new_layer
        elif 'features_2.4.features.0' == layer_name:
            self.features_2[4].features[0] = new_layer
            self.layer_names["features_2.4.features.0"] = new_layer
            self.origin_layer_names["features_2.4.features.0"] = new_layer
        elif 'features_2.4.features.1' == layer_name:
            self.features_2[4].features[1] = new_layer
            self.layer_names["features_2.4.features.1"] = new_layer
            self.origin_layer_names["features_2.4.features.1"] = new_layer
        elif 'features_2.4.features.2' == layer_name:
            self.features_2[4].features[2] = new_layer
            self.layer_names["features_2.4.features.2"] = new_layer
            self.origin_layer_names["features_2.4.features.2"] = new_layer

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name, out):
        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name] = out

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

    def set_Basic_OPS(self, b):
        self.Basic_OPS = b

    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self, c):
        self.Cascade_OPs = c


if __name__ == '__main__':
    data = ms.Tensor(np.random.rand(1, 3, 300, 300), ms.float32)
    num_matched_boxes = mindspore.Tensor([[33]], mindspore.int32)
    gt_label = mindspore.Tensor(np.random.randn(1, 1917), mindspore.int32)  # np.load("./official/gt_label.npy")
    get_loc = mindspore.Tensor(np.random.randn(1, 1917, 4),
                               mindspore.float32)  # mindspore.Tensor(np.load("./official/get_loc.npy"), mindspore.float32)

    network = SSDWithMobileNetV2()
    result1, result2 = network(data)
    print(result1.shape)
    print(result2.shape)

    # # Define the learning rate
    # lr = 1e-4
    #
    # # Define the optimizer
    # opt = nn.Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), lr, 0.9, 0.00015, float(1024))
    #
    #
    # # Define the forward procedure
    # def forward_fn(x, gt_loc, gt_label, num_matched_boxes):
    #
    #     pred_loc, pred_label = network(x)
    #
    #     mask = ops.less(0, gt_label).astype(mindspore.float32)
    #     num_matched_boxes = ops.sum(num_matched_boxes.astype(mindspore.float32))
    #
    #     # Positioning loss
    #     mask_loc = ops.tile(ops.expand_dims(mask, -1), (1, 1, 4))
    #     smooth_l1 = nn.SmoothL1Loss()(pred_loc, gt_loc) * mask_loc
    #     loss_loc = ops.sum(ops.sum(smooth_l1, -1), -1)
    #
    #     # Category loss
    #     loss_cls = class_loss(pred_label, gt_label)
    #     loss_cls = ops.sum(loss_cls, (1, 2))
    #
    #     return ops.sum((loss_cls + loss_loc) / num_matched_boxes)
    #
    #
    # grad_fn = mindspore.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)
    #
    #
    # # Gradient updates
    # def train_step(x, gt_loc, gt_label, num_matched_boxes):
    #     loss, grads = grad_fn(x, gt_loc, gt_label, num_matched_boxes)
    #     opt(grads)
    #     return loss
    #
    #
    # for epoch in range(5):
    #     network.set_train(True)
    #     loss = train_step(data, get_loc, gt_label, num_matched_boxes)
    #     print("loss: " + str(loss))
