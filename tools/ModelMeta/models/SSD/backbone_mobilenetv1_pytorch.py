import math
import numpy as np
import torch
from torch import nn
from models.SSD.ssd_utils_torch import conv_bn_relu, ConvBNReLU, FeatureSelector, MultiBox


def conv_bn_relu(in_channel, out_channel, kernel_size, stride, depthwise):
    output = []
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    if isinstance(stride, tuple):
        stride = stride[0]
    padding = math.ceil((kernel_size - stride) / 2)
    # print("padding: ", padding)
    output.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding,
                            groups=1 if not depthwise else in_channel, bias=False))
    output.append(nn.BatchNorm2d(out_channel))
    output.append(nn.ReLU6())
    return nn.Sequential(*output)


class FeatureSelector(nn.Module):
    """
    Select specific layers from an entire feature list
    """

    def __init__(self, feature_idxes):
        super(FeatureSelector, self).__init__()
        self.feature_idxes = feature_idxes

    def forward(self, feature_list):
        selected = ()
        for i in self.feature_idxes:
            selected = selected + (feature_list[i],)
        return selected


class SSDWithMobileNetV1(nn.Module):
    def __init__(self, ):
        super(SSDWithMobileNetV1, self).__init__()
        cnn = [
            conv_bn_relu(3, 32, 3, 2, False),  # Conv0
            conv_bn_relu(32, 32, 3, 1, True),  # Conv1_depthwise
            conv_bn_relu(32, 64, 1, 1, False),  # Conv1_pointwise
            conv_bn_relu(64, 64, 3, 2, True),  # Conv2_depthwise
            conv_bn_relu(64, 128, 1, 1, False),  # Conv2_pointwise
            conv_bn_relu(128, 128, 3, 1, True),  # Conv3_depthwise
            conv_bn_relu(128, 128, 1, 1, False),  # Conv3_pointwise
            conv_bn_relu(128, 128, 3, 2, True),  # Conv4_depthwise
            conv_bn_relu(128, 256, 1, 1, False),  # Conv4_pointwise
            conv_bn_relu(256, 256, 3, 1, True),  # Conv5_depthwise
            conv_bn_relu(256, 256, 1, 1, False),  # Conv5_pointwise
            conv_bn_relu(256, 256, 3, 2, True),  # Conv6_depthwise
            conv_bn_relu(256, 512, 1, 1, False),  # Conv6_pointwise
            conv_bn_relu(512, 512, 3, 1, True),  # Conv7_depthwise
            conv_bn_relu(512, 512, 1, 1, False),  # Conv7_pointwise
            conv_bn_relu(512, 512, 3, 1, True),  # Conv8_depthwise
            conv_bn_relu(512, 512, 1, 1, False),  # Conv8_pointwise
            conv_bn_relu(512, 512, 3, 1, True),  # Conv9_depthwise
            conv_bn_relu(512, 512, 1, 1, False),  # Conv9_pointwise
            conv_bn_relu(512, 512, 3, 1, True),  # Conv10_depthwise
            conv_bn_relu(512, 512, 1, 1, False),  # Conv10_pointwise
            conv_bn_relu(512, 512, 3, 1, True),  # Conv11_depthwise
            conv_bn_relu(512, 512, 1, 1, False),  # Conv11_pointwise
            conv_bn_relu(512, 512, 3, 2, True),  # Conv12_depthwise
            conv_bn_relu(512, 1024, 1, 1, False),  # Conv12_pointwise
            conv_bn_relu(1024, 1024, 3, 1, True),  # Conv13_depthwise
            conv_bn_relu(1024, 1024, 1, 1, False),  # Conv13_pointwise
        ]

        self.network = nn.ModuleList(cnn)

        layer_indexs = [14, 26]
        self.selector = FeatureSelector(layer_indexs)
        in_channels = [256, 512, 1024, 512, 256, 256]
        out_channels = [512, 1024, 512, 256, 256, 128]
        strides = [1, 1, 2, 2, 2, 2]
        residual_list = []
        for i in range(2, len(in_channels)):
            residual = ConvBNReLU(in_channels[i], out_channels[i], stride=strides[i])
            residual_list.append(residual)
        self.multi_residual = nn.ModuleList(residual_list)
        self.multi_box = MultiBox(81, [512, 1024, 512, 256, 256, 128], [3, 6, 6, 6, 6, 6], 1917)

        self.layer_names = {
            "network": self.network,
            "network.0": self.network[0],
            "network.0.0": self.network[0][0],
            "network.0.1": self.network[0][1],
            "network.0.2": self.network[0][2],
            "network.1": self.network[1],
            "network.1.0": self.network[1][0],
            "network.1.1": self.network[1][1],
            "network.1.2": self.network[1][2],
            "network.2": self.network[2],
            "network.2.0": self.network[2][0],
            "network.2.1": self.network[2][1],
            "network.2.2": self.network[2][2],
            "network.3": self.network[3],
            "network.3.0": self.network[3][0],
            "network.3.1": self.network[3][1],
            "network.3.2": self.network[3][2],
            "network.4": self.network[4],
            "network.4.0": self.network[4][0],
            "network.4.1": self.network[4][1],
            "network.4.2": self.network[4][2],
            "network.5": self.network[5],
            "network.5.0": self.network[5][0],
            "network.5.1": self.network[5][1],
            "network.5.2": self.network[5][2],
            "network.6": self.network[6],
            "network.6.0": self.network[6][0],
            "network.6.1": self.network[6][1],
            "network.6.2": self.network[6][2],
            "network.7": self.network[7],
            "network.7.0": self.network[7][0],
            "network.7.1": self.network[7][1],
            "network.7.2": self.network[7][2],
            "network.8": self.network[8],
            "network.8.0": self.network[8][0],
            "network.8.1": self.network[8][1],
            "network.8.2": self.network[8][2],
            "network.9": self.network[9],
            "network.9.0": self.network[9][0],
            "network.9.1": self.network[9][1],
            "network.9.2": self.network[9][2],
            "network.10": self.network[10],
            "network.10.0": self.network[10][0],
            "network.10.1": self.network[10][1],
            "network.10.2": self.network[10][2],
            "network.11": self.network[11],
            "network.11.0": self.network[11][0],
            "network.11.1": self.network[11][1],
            "network.11.2": self.network[11][2],
            "network.12": self.network[12],
            "network.12.0": self.network[12][0],
            "network.12.1": self.network[12][1],
            "network.12.2": self.network[12][2],
            "network.13": self.network[13],
            "network.13.0": self.network[13][0],
            "network.13.1": self.network[13][1],
            "network.13.2": self.network[13][2],
            "network.14": self.network[14],
            "network.14.0": self.network[14][0],
            "network.14.1": self.network[14][1],
            "network.14.2": self.network[14][2],
            "network.15": self.network[15],
            "network.15.0": self.network[15][0],
            "network.15.1": self.network[15][1],
            "network.15.2": self.network[15][2],
            "network.16": self.network[16],
            "network.16.0": self.network[16][0],
            "network.16.1": self.network[16][1],
            "network.16.2": self.network[16][2],
            "network.17": self.network[17],
            "network.17.0": self.network[17][0],
            "network.17.1": self.network[17][1],
            "network.17.2": self.network[17][2],
            "network.18": self.network[18],
            "network.18.0": self.network[18][0],
            "network.18.1": self.network[18][1],
            "network.18.2": self.network[18][2],
            "network.19": self.network[19],
            "network.19.0": self.network[19][0],
            "network.19.1": self.network[19][1],
            "network.19.2": self.network[19][2],
            "network.20": self.network[20],
            "network.20.0": self.network[20][0],
            "network.20.1": self.network[20][1],
            "network.20.2": self.network[20][2],
            "network.21": self.network[21],
            "network.21.0": self.network[21][0],
            "network.21.1": self.network[21][1],
            "network.21.2": self.network[21][2],
            "network.22": self.network[22],
            "network.22.0": self.network[22][0],
            "network.22.1": self.network[22][1],
            "network.22.2": self.network[22][2],
            "network.23": self.network[23],
            "network.23.0": self.network[23][0],
            "network.23.1": self.network[23][1],
            "network.23.2": self.network[23][2],
            "network.24": self.network[24],
            "network.24.0": self.network[24][0],
            "network.24.1": self.network[24][1],
            "network.24.2": self.network[24][2],
            "network.25": self.network[25],
            "network.25.0": self.network[25][0],
            "network.25.1": self.network[25][1],
            "network.25.2": self.network[25][2],
            "network.26": self.network[26],
            "network.26.0": self.network[26][0],
            "network.26.1": self.network[26][1],
            "network.26.2": self.network[26][2],

        }
        self.origin_layer_names = {
            "network": self.network,
            "network.0": self.network[0],
            "network.0.0": self.network[0][0],
            "network.0.1": self.network[0][1],
            "network.0.2": self.network[0][2],
            "network.1": self.network[1],
            "network.1.0": self.network[1][0],
            "network.1.1": self.network[1][1],
            "network.1.2": self.network[1][2],
            "network.2": self.network[2],
            "network.2.0": self.network[2][0],
            "network.2.1": self.network[2][1],
            "network.2.2": self.network[2][2],
            "network.3": self.network[3],
            "network.3.0": self.network[3][0],
            "network.3.1": self.network[3][1],
            "network.3.2": self.network[3][2],
            "network.4": self.network[4],
            "network.4.0": self.network[4][0],
            "network.4.1": self.network[4][1],
            "network.4.2": self.network[4][2],
            "network.5": self.network[5],
            "network.5.0": self.network[5][0],
            "network.5.1": self.network[5][1],
            "network.5.2": self.network[5][2],
            "network.6": self.network[6],
            "network.6.0": self.network[6][0],
            "network.6.1": self.network[6][1],
            "network.6.2": self.network[6][2],
            "network.7": self.network[7],
            "network.7.0": self.network[7][0],
            "network.7.1": self.network[7][1],
            "network.7.2": self.network[7][2],
            "network.8": self.network[8],
            "network.8.0": self.network[8][0],
            "network.8.1": self.network[8][1],
            "network.8.2": self.network[8][2],
            "network.9": self.network[9],
            "network.9.0": self.network[9][0],
            "network.9.1": self.network[9][1],
            "network.9.2": self.network[9][2],
            "network.10": self.network[10],
            "network.10.0": self.network[10][0],
            "network.10.1": self.network[10][1],
            "network.10.2": self.network[10][2],
            "network.11": self.network[11],
            "network.11.0": self.network[11][0],
            "network.11.1": self.network[11][1],
            "network.11.2": self.network[11][2],
            "network.12": self.network[12],
            "network.12.0": self.network[12][0],
            "network.12.1": self.network[12][1],
            "network.12.2": self.network[12][2],
            "network.13": self.network[13],
            "network.13.0": self.network[13][0],
            "network.13.1": self.network[13][1],
            "network.13.2": self.network[13][2],
            "network.14": self.network[14],
            "network.14.0": self.network[14][0],
            "network.14.1": self.network[14][1],
            "network.14.2": self.network[14][2],
            "network.15": self.network[15],
            "network.15.0": self.network[15][0],
            "network.15.1": self.network[15][1],
            "network.15.2": self.network[15][2],
            "network.16": self.network[16],
            "network.16.0": self.network[16][0],
            "network.16.1": self.network[16][1],
            "network.16.2": self.network[16][2],
            "network.17": self.network[17],
            "network.17.0": self.network[17][0],
            "network.17.1": self.network[17][1],
            "network.17.2": self.network[17][2],
            "network.18": self.network[18],
            "network.18.0": self.network[18][0],
            "network.18.1": self.network[18][1],
            "network.18.2": self.network[18][2],
            "network.19": self.network[19],
            "network.19.0": self.network[19][0],
            "network.19.1": self.network[19][1],
            "network.19.2": self.network[19][2],
            "network.20": self.network[20],
            "network.20.0": self.network[20][0],
            "network.20.1": self.network[20][1],
            "network.20.2": self.network[20][2],
            "network.21": self.network[21],
            "network.21.0": self.network[21][0],
            "network.21.1": self.network[21][1],
            "network.21.2": self.network[21][2],
            "network.22": self.network[22],
            "network.22.0": self.network[22][0],
            "network.22.1": self.network[22][1],
            "network.22.2": self.network[22][2],
            "network.23": self.network[23],
            "network.23.0": self.network[23][0],
            "network.23.1": self.network[23][1],
            "network.23.2": self.network[23][2],
            "network.24": self.network[24],
            "network.24.0": self.network[24][0],
            "network.24.1": self.network[24][1],
            "network.24.2": self.network[24][2],
            "network.25": self.network[25],
            "network.25.0": self.network[25][0],
            "network.25.1": self.network[25][1],
            "network.25.2": self.network[25][2],
            "network.26": self.network[26],
            "network.26.0": self.network[26][0],
            "network.26.1": self.network[26][1],
            "network.26.2": self.network[26][2],

        }
        self.orders = {
            'network.0.0': ["INPUT", "network.0.1"],
            'network.0.1': ["network.0.0", "network.0.2"],
            'network.0.2': ["network.0.1", "network.1.0"],
            'network.1.0': ["network.0.2", "network.1.1"],
            'network.1.1': ["network.1.0", "network.1.2"],
            'network.1.2': ["network.1.1", "network.2.0"],
            'network.2.0': ["network.1.2", "network.2.1"],
            'network.2.1': ["network.2.0", "network.2.2"],
            'network.2.2': ["network.2.1", "network.3.0"],
            'network.3.0': ["network.2.2", "network.3.1"],
            'network.3.1': ["network.3.0", "network.3.2"],
            'network.3.2': ["network.3.1", "network.4.0"],
            'network.4.0': ["network.3.2", "network.4.1"],
            'network.4.1': ["network.4.0", "network.4.2"],
            'network.4.2': ["network.4.1", "network.5.0"],
            'network.5.0': ["network.4.2", "network.5.1"],
            'network.5.1': ["network.5.0", "network.5.2"],
            'network.5.2': ["network.5.1", "network.6.0"],
            'network.6.0': ["network.5.2", "network.6.1"],
            'network.6.1': ["network.6.0", "network.6.2"],
            'network.6.2': ["network.6.1", "network.7.0"],
            'network.7.0': ["network.6.2", "network.7.1"],
            'network.7.1': ["network.7.0", "network.7.2"],
            'network.7.2': ["network.7.1", "network.8.0"],
            'network.8.0': ["network.7.2", "network.8.1"],
            'network.8.1': ["network.8.0", "network.8.2"],
            'network.8.2': ["network.8.1", "network.9.0"],
            'network.9.0': ["network.8.2", "network.9.1"],
            'network.9.1': ["network.9.0", "network.9.2"],
            'network.9.2': ["network.9.1", "network.10.0"],
            'network.10.0': ["network.9.2", "network.10.1"],
            'network.10.1': ["network.10.0", "network.10.2"],
            'network.10.2': ["network.10.1", "network.11.0"],
            'network.11.0': ["network.10.2", "network.11.1"],
            'network.11.1': ["network.11.0", "network.11.2"],
            'network.11.2': ["network.11.1", "network.12.0"],
            'network.12.0': ["network.11.2", "network.12.1"],
            'network.12.1': ["network.12.0", "network.12.2"],
            'network.12.2': ["network.12.1", "network.13.0"],
            'network.13.0': ["network.12.2", "network.13.1"],
            'network.13.1': ["network.13.0", "network.13.2"],
            'network.13.2': ["network.13.1", "network.14.0"],
            'network.14.0': ["network.13.2", "network.14.1"],
            'network.14.1': ["network.14.0", "network.14.2"],
            'network.14.2': ["network.14.1", ["network.15.0", "OUTPUT1"]],
            'network.15.0': ["network.14.2", "network.15.1"],
            'network.15.1': ["network.15.0", "network.15.2"],
            'network.15.2': ["network.15.1", "network.16.0"],
            'network.16.0': ["network.15.2", "network.16.1"],
            'network.16.1': ["network.16.0", "network.16.2"],
            'network.16.2': ["network.16.1", "network.17.0"],
            'network.17.0': ["network.16.2", "network.17.1"],
            'network.17.1': ["network.17.0", "network.17.2"],
            'network.17.2': ["network.17.1", "network.18.0"],
            'network.18.0': ["network.17.2", "network.18.1"],
            'network.18.1': ["network.18.0", "network.18.2"],
            'network.18.2': ["network.18.1", "network.19.0"],
            'network.19.0': ["network.18.2", "network.19.1"],
            'network.19.1': ["network.19.0", "network.19.2"],
            'network.19.2': ["network.19.1", "network.20.0"],
            'network.20.0': ["network.19.2", "network.20.1"],
            'network.20.1': ["network.20.0", "network.20.2"],
            'network.20.2': ["network.20.1", "network.21.0"],
            'network.21.0': ["network.20.2", "network.21.1"],
            'network.21.1': ["network.21.0", "network.21.2"],
            'network.21.2': ["network.21.1", "network.22.0"],
            'network.22.0': ["network.21.2", "network.22.1"],
            'network.22.1': ["network.22.0", "network.22.2"],
            'network.22.2': ["network.22.1", "network.23.0"],
            'network.23.0': ["network.22.2", "network.23.1"],
            'network.23.1': ["network.23.0", "network.23.2"],
            'network.23.2': ["network.23.1", "network.24.0"],
            'network.24.0': ["network.23.2", "network.24.1"],
            'network.24.1': ["network.24.0", "network.24.2"],
            'network.24.2': ["network.24.1", "network.25.0"],
            'network.25.0': ["network.24.2", "network.25.1"],
            'network.25.1': ["network.25.0", "network.25.2"],
            'network.25.2': ["network.25.1", "network.26.0"],
            'network.26.0': ["network.25.2", "network.26.1"],
            'network.26.1': ["network.26.0", "network.26.2"],
            'network.26.2': ["network.26.1", "OUTPUT2"],

        }

        self.in_shapes = {
            'INPUT': [1, 3, 300, 300],
            'network.0.0': [1, 3, 300, 300],
            'network.0.1': [1, 32, 150, 150],
            'network.0.2': [1, 32, 150, 150],
            'network.1.0': [1, 32, 150, 150],
            'network.1.1': [1, 32, 150, 150],
            'network.1.2': [1, 32, 150, 150],
            'network.2.0': [1, 32, 150, 150],
            'network.2.1': [1, 64, 150, 150],
            'network.2.2': [1, 64, 150, 150],
            'network.3.0': [1, 64, 150, 150],
            'network.3.1': [1, 64, 75, 75],
            'network.3.2': [1, 64, 75, 75],
            'network.4.0': [1, 64, 75, 75],
            'network.4.1': [1, 128, 75, 75],
            'network.4.2': [1, 128, 75, 75],
            'network.5.0': [1, 128, 75, 75],
            'network.5.1': [1, 128, 75, 75],
            'network.5.2': [1, 128, 75, 75],
            'network.6.0': [1, 128, 75, 75],
            'network.6.1': [1, 128, 75, 75],
            'network.6.2': [1, 128, 75, 75],
            'network.7.0': [1, 128, 75, 75],
            'network.7.1': [1, 128, 38, 38],
            'network.7.2': [1, 128, 38, 38],
            'network.8.0': [1, 128, 38, 38],
            'network.8.1': [1, 256, 38, 38],
            'network.8.2': [1, 256, 38, 38],
            'network.9.0': [1, 256, 38, 38],
            'network.9.1': [1, 256, 38, 38],
            'network.9.2': [1, 256, 38, 38],
            'network.10.0': [1, 256, 38, 38],
            'network.10.1': [1, 256, 38, 38],
            'network.10.2': [1, 256, 38, 38],
            'network.11.0': [1, 256, 38, 38],
            'network.11.1': [1, 256, 19, 19],
            'network.11.2': [1, 256, 19, 19],
            'network.12.0': [1, 256, 19, 19],
            'network.12.1': [1, 512, 19, 19],
            'network.12.2': [1, 512, 19, 19],
            'network.13.0': [1, 512, 19, 19],
            'network.13.1': [1, 512, 19, 19],
            'network.13.2': [1, 512, 19, 19],
            'network.14.0': [1, 512, 19, 19],
            'network.14.1': [1, 512, 19, 19],
            'network.14.2': [1, 512, 19, 19],
            'network.15.0': [1, 512, 19, 19],
            'network.15.1': [1, 512, 19, 19],
            'network.15.2': [1, 512, 19, 19],
            'network.16.0': [1, 512, 19, 19],
            'network.16.1': [1, 512, 19, 19],
            'network.16.2': [1, 512, 19, 19],
            'network.17.0': [1, 512, 19, 19],
            'network.17.1': [1, 512, 19, 19],
            'network.17.2': [1, 512, 19, 19],
            'network.18.0': [1, 512, 19, 19],
            'network.18.1': [1, 512, 19, 19],
            'network.18.2': [1, 512, 19, 19],
            'network.19.0': [1, 512, 19, 19],
            'network.19.1': [1, 512, 19, 19],
            'network.19.2': [1, 512, 19, 19],
            'network.20.0': [1, 512, 19, 19],
            'network.20.1': [1, 512, 19, 19],
            'network.20.2': [1, 512, 19, 19],
            'network.21.0': [1, 512, 19, 19],
            'network.21.1': [1, 512, 19, 19],
            'network.21.2': [1, 512, 19, 19],
            'network.22.0': [1, 512, 19, 19],
            'network.22.1': [1, 512, 19, 19],
            'network.22.2': [1, 512, 19, 19],
            'network.23.0': [1, 512, 19, 19],
            'network.23.1': [1, 512, 10, 10],
            'network.23.2': [1, 512, 10, 10],
            'network.24.0': [1, 512, 10, 10],
            'network.24.1': [1, 1024, 10, 10],
            'network.24.2': [1, 1024, 10, 10],
            'network.25.0': [1, 1024, 10, 10],
            'network.25.1': [1, 1024, 10, 10],
            'network.25.2': [1, 1024, 10, 10],
            'network.26.0': [1, 1024, 10, 10],
            'network.26.1': [1, 1024, 10, 10],
            'network.26.2': [1, 1024, 10, 10],
            "OUTPUT1": [1, 512, 19, 19],
            "OUTPUT2": [1, 1024, 10, 10]
        }

        self.out_shapes = {
            'INPUT': [1, 3, 300, 300],
            'network.0.0': [1, 32, 150, 150],
            'network.0.1': [1, 32, 150, 150],
            'network.0.2': [1, 32, 150, 150],
            'network.1.0': [1, 32, 150, 150],
            'network.1.1': [1, 32, 150, 150],
            'network.1.2': [1, 32, 150, 150],
            'network.2.0': [1, 64, 150, 150],
            'network.2.1': [1, 64, 150, 150],
            'network.2.2': [1, 64, 150, 150],
            'network.3.0': [1, 64, 75, 75],
            'network.3.1': [1, 64, 75, 75],
            'network.3.2': [1, 64, 75, 75],
            'network.4.0': [1, 128, 75, 75],
            'network.4.1': [1, 128, 75, 75],
            'network.4.2': [1, 128, 75, 75],
            'network.5.0': [1, 128, 75, 75],
            'network.5.1': [1, 128, 75, 75],
            'network.5.2': [1, 128, 75, 75],
            'network.6.0': [1, 128, 75, 75],
            'network.6.1': [1, 128, 75, 75],
            'network.6.2': [1, 128, 75, 75],
            'network.7.0': [1, 128, 38, 38],
            'network.7.1': [1, 128, 38, 38],
            'network.7.2': [1, 128, 38, 38],
            'network.8.0': [1, 256, 38, 38],
            'network.8.1': [1, 256, 38, 38],
            'network.8.2': [1, 256, 38, 38],
            'network.9.0': [1, 256, 38, 38],
            'network.9.1': [1, 256, 38, 38],
            'network.9.2': [1, 256, 38, 38],
            'network.10.0': [1, 256, 38, 38],
            'network.10.1': [1, 256, 38, 38],
            'network.10.2': [1, 256, 38, 38],
            'network.11.0': [1, 256, 19, 19],
            'network.11.1': [1, 256, 19, 19],
            'network.11.2': [1, 256, 19, 19],
            'network.12.0': [1, 512, 19, 19],
            'network.12.1': [1, 512, 19, 19],
            'network.12.2': [1, 512, 19, 19],
            'network.13.0': [1, 512, 19, 19],
            'network.13.1': [1, 512, 19, 19],
            'network.13.2': [1, 512, 19, 19],
            'network.14.0': [1, 512, 19, 19],
            'network.14.1': [1, 512, 19, 19],
            'network.14.2': [1, 512, 19, 19],
            'network.15.0': [1, 512, 19, 19],
            'network.15.1': [1, 512, 19, 19],
            'network.15.2': [1, 512, 19, 19],
            'network.16.0': [1, 512, 19, 19],
            'network.16.1': [1, 512, 19, 19],
            'network.16.2': [1, 512, 19, 19],
            'network.17.0': [1, 512, 19, 19],
            'network.17.1': [1, 512, 19, 19],
            'network.17.2': [1, 512, 19, 19],
            'network.18.0': [1, 512, 19, 19],
            'network.18.1': [1, 512, 19, 19],
            'network.18.2': [1, 512, 19, 19],
            'network.19.0': [1, 512, 19, 19],
            'network.19.1': [1, 512, 19, 19],
            'network.19.2': [1, 512, 19, 19],
            'network.20.0': [1, 512, 19, 19],
            'network.20.1': [1, 512, 19, 19],
            'network.20.2': [1, 512, 19, 19],
            'network.21.0': [1, 512, 19, 19],
            'network.21.1': [1, 512, 19, 19],
            'network.21.2': [1, 512, 19, 19],
            'network.22.0': [1, 512, 19, 19],
            'network.22.1': [1, 512, 19, 19],
            'network.22.2': [1, 512, 19, 19],
            'network.23.0': [1, 512, 10, 10],
            'network.23.1': [1, 512, 10, 10],
            'network.23.2': [1, 512, 10, 10],
            'network.24.0': [1, 1024, 10, 10],
            'network.24.1': [1, 1024, 10, 10],
            'network.24.2': [1, 1024, 10, 10],
            'network.25.0': [1, 1024, 10, 10],
            'network.25.1': [1, 1024, 10, 10],
            'network.25.2': [1, 1024, 10, 10],
            'network.26.0': [1, 1024, 10, 10],
            'network.26.1': [1, 1024, 10, 10],
            'network.26.2': [1, 1024, 10, 10],
            "OUTPUT1": [1, 512, 19, 19],
            "OUTPUT2": [1, 1024, 10, 10]
        }

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

    def forward(self, x):
        output = x
        features = []
        for block in self.network:
            output = block(output)
            features.append(output)
        feature, output = self.selector(features)
        multi_feature = [feature, output]
        feature = output
        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature.append(feature)
        pred_loc, pred_label = self.multi_box(multi_feature)

        if not self.training:
            pred_label = torch.sigmoid(pred_label)
        pred_loc = pred_loc.to(torch.float32)
        pred_label = pred_label.to(torch.float32)

        return pred_loc, pred_label

    def set_layers(self, layer_name, new_layer):
        if 'network' == layer_name:
            self.network = new_layer
            self.layer_names["network"] = new_layer
            self.origin_layer_names["network"] = new_layer
        elif 'network.0' == layer_name:
            self.network[0] = new_layer
            self.layer_names["network.0"] = new_layer
            self.origin_layer_names["network.0"] = new_layer
        elif 'network.0.0' == layer_name:
            self.network[0][0] = new_layer
            self.layer_names["network.0.0"] = new_layer
            self.origin_layer_names["network.0.0"] = new_layer
        elif 'network.0.1' == layer_name:
            self.network[0][1] = new_layer
            self.layer_names["network.0.1"] = new_layer
            self.origin_layer_names["network.0.1"] = new_layer
        elif 'network.0.2' == layer_name:
            self.network[0][2] = new_layer
            self.layer_names["network.0.2"] = new_layer
            self.origin_layer_names["network.0.2"] = new_layer
        elif 'network.1' == layer_name:
            self.network[1] = new_layer
            self.layer_names["network.1"] = new_layer
            self.origin_layer_names["network.1"] = new_layer
        elif 'network.1.0' == layer_name:
            self.network[1][0] = new_layer
            self.layer_names["network.1.0"] = new_layer
            self.origin_layer_names["network.1.0"] = new_layer
        elif 'network.1.1' == layer_name:
            self.network[1][1] = new_layer
            self.layer_names["network.1.1"] = new_layer
            self.origin_layer_names["network.1.1"] = new_layer
        elif 'network.1.2' == layer_name:
            self.network[1][2] = new_layer
            self.layer_names["network.1.2"] = new_layer
            self.origin_layer_names["network.1.2"] = new_layer
        elif 'network.2' == layer_name:
            self.network[2] = new_layer
            self.layer_names["network.2"] = new_layer
            self.origin_layer_names["network.2"] = new_layer
        elif 'network.2.0' == layer_name:
            self.network[2][0] = new_layer
            self.layer_names["network.2.0"] = new_layer
            self.origin_layer_names["network.2.0"] = new_layer
        elif 'network.2.1' == layer_name:
            self.network[2][1] = new_layer
            self.layer_names["network.2.1"] = new_layer
            self.origin_layer_names["network.2.1"] = new_layer
        elif 'network.2.2' == layer_name:
            self.network[2][2] = new_layer
            self.layer_names["network.2.2"] = new_layer
            self.origin_layer_names["network.2.2"] = new_layer
        elif 'network.3' == layer_name:
            self.network[3] = new_layer
            self.layer_names["network.3"] = new_layer
            self.origin_layer_names["network.3"] = new_layer
        elif 'network.3.0' == layer_name:
            self.network[3][0] = new_layer
            self.layer_names["network.3.0"] = new_layer
            self.origin_layer_names["network.3.0"] = new_layer
        elif 'network.3.1' == layer_name:
            self.network[3][1] = new_layer
            self.layer_names["network.3.1"] = new_layer
            self.origin_layer_names["network.3.1"] = new_layer
        elif 'network.3.2' == layer_name:
            self.network[3][2] = new_layer
            self.layer_names["network.3.2"] = new_layer
            self.origin_layer_names["network.3.2"] = new_layer
        elif 'network.4' == layer_name:
            self.network[4] = new_layer
            self.layer_names["network.4"] = new_layer
            self.origin_layer_names["network.4"] = new_layer
        elif 'network.4.0' == layer_name:
            self.network[4][0] = new_layer
            self.layer_names["network.4.0"] = new_layer
            self.origin_layer_names["network.4.0"] = new_layer
        elif 'network.4.1' == layer_name:
            self.network[4][1] = new_layer
            self.layer_names["network.4.1"] = new_layer
            self.origin_layer_names["network.4.1"] = new_layer
        elif 'network.4.2' == layer_name:
            self.network[4][2] = new_layer
            self.layer_names["network.4.2"] = new_layer
            self.origin_layer_names["network.4.2"] = new_layer
        elif 'network.5' == layer_name:
            self.network[5] = new_layer
            self.layer_names["network.5"] = new_layer
            self.origin_layer_names["network.5"] = new_layer
        elif 'network.5.0' == layer_name:
            self.network[5][0] = new_layer
            self.layer_names["network.5.0"] = new_layer
            self.origin_layer_names["network.5.0"] = new_layer
        elif 'network.5.1' == layer_name:
            self.network[5][1] = new_layer
            self.layer_names["network.5.1"] = new_layer
            self.origin_layer_names["network.5.1"] = new_layer
        elif 'network.5.2' == layer_name:
            self.network[5][2] = new_layer
            self.layer_names["network.5.2"] = new_layer
            self.origin_layer_names["network.5.2"] = new_layer
        elif 'network.6' == layer_name:
            self.network[6] = new_layer
            self.layer_names["network.6"] = new_layer
            self.origin_layer_names["network.6"] = new_layer
        elif 'network.6.0' == layer_name:
            self.network[6][0] = new_layer
            self.layer_names["network.6.0"] = new_layer
            self.origin_layer_names["network.6.0"] = new_layer
        elif 'network.6.1' == layer_name:
            self.network[6][1] = new_layer
            self.layer_names["network.6.1"] = new_layer
            self.origin_layer_names["network.6.1"] = new_layer
        elif 'network.6.2' == layer_name:
            self.network[6][2] = new_layer
            self.layer_names["network.6.2"] = new_layer
            self.origin_layer_names["network.6.2"] = new_layer
        elif 'network.7' == layer_name:
            self.network[7] = new_layer
            self.layer_names["network.7"] = new_layer
            self.origin_layer_names["network.7"] = new_layer
        elif 'network.7.0' == layer_name:
            self.network[7][0] = new_layer
            self.layer_names["network.7.0"] = new_layer
            self.origin_layer_names["network.7.0"] = new_layer
        elif 'network.7.1' == layer_name:
            self.network[7][1] = new_layer
            self.layer_names["network.7.1"] = new_layer
            self.origin_layer_names["network.7.1"] = new_layer
        elif 'network.7.2' == layer_name:
            self.network[7][2] = new_layer
            self.layer_names["network.7.2"] = new_layer
            self.origin_layer_names["network.7.2"] = new_layer
        elif 'network.8' == layer_name:
            self.network[8] = new_layer
            self.layer_names["network.8"] = new_layer
            self.origin_layer_names["network.8"] = new_layer
        elif 'network.8.0' == layer_name:
            self.network[8][0] = new_layer
            self.layer_names["network.8.0"] = new_layer
            self.origin_layer_names["network.8.0"] = new_layer
        elif 'network.8.1' == layer_name:
            self.network[8][1] = new_layer
            self.layer_names["network.8.1"] = new_layer
            self.origin_layer_names["network.8.1"] = new_layer
        elif 'network.8.2' == layer_name:
            self.network[8][2] = new_layer
            self.layer_names["network.8.2"] = new_layer
            self.origin_layer_names["network.8.2"] = new_layer
        elif 'network.9' == layer_name:
            self.network[9] = new_layer
            self.layer_names["network.9"] = new_layer
            self.origin_layer_names["network.9"] = new_layer
        elif 'network.9.0' == layer_name:
            self.network[9][0] = new_layer
            self.layer_names["network.9.0"] = new_layer
            self.origin_layer_names["network.9.0"] = new_layer
        elif 'network.9.1' == layer_name:
            self.network[9][1] = new_layer
            self.layer_names["network.9.1"] = new_layer
            self.origin_layer_names["network.9.1"] = new_layer
        elif 'network.9.2' == layer_name:
            self.network[9][2] = new_layer
            self.layer_names["network.9.2"] = new_layer
            self.origin_layer_names["network.9.2"] = new_layer
        elif 'network.10' == layer_name:
            self.network[10] = new_layer
            self.layer_names["network.10"] = new_layer
            self.origin_layer_names["network.10"] = new_layer
        elif 'network.10.0' == layer_name:
            self.network[10][0] = new_layer
            self.layer_names["network.10.0"] = new_layer
            self.origin_layer_names["network.10.0"] = new_layer
        elif 'network.10.1' == layer_name:
            self.network[10][1] = new_layer
            self.layer_names["network.10.1"] = new_layer
            self.origin_layer_names["network.10.1"] = new_layer
        elif 'network.10.2' == layer_name:
            self.network[10][2] = new_layer
            self.layer_names["network.10.2"] = new_layer
            self.origin_layer_names["network.10.2"] = new_layer
        elif 'network.11' == layer_name:
            self.network[11] = new_layer
            self.layer_names["network.11"] = new_layer
            self.origin_layer_names["network.11"] = new_layer
        elif 'network.11.0' == layer_name:
            self.network[11][0] = new_layer
            self.layer_names["network.11.0"] = new_layer
            self.origin_layer_names["network.11.0"] = new_layer
        elif 'network.11.1' == layer_name:
            self.network[11][1] = new_layer
            self.layer_names["network.11.1"] = new_layer
            self.origin_layer_names["network.11.1"] = new_layer
        elif 'network.11.2' == layer_name:
            self.network[11][2] = new_layer
            self.layer_names["network.11.2"] = new_layer
            self.origin_layer_names["network.11.2"] = new_layer
        elif 'network.12' == layer_name:
            self.network[12] = new_layer
            self.layer_names["network.12"] = new_layer
            self.origin_layer_names["network.12"] = new_layer
        elif 'network.12.0' == layer_name:
            self.network[12][0] = new_layer
            self.layer_names["network.12.0"] = new_layer
            self.origin_layer_names["network.12.0"] = new_layer
        elif 'network.12.1' == layer_name:
            self.network[12][1] = new_layer
            self.layer_names["network.12.1"] = new_layer
            self.origin_layer_names["network.12.1"] = new_layer
        elif 'network.12.2' == layer_name:
            self.network[12][2] = new_layer
            self.layer_names["network.12.2"] = new_layer
            self.origin_layer_names["network.12.2"] = new_layer
        elif 'network.13' == layer_name:
            self.network[13] = new_layer
            self.layer_names["network.13"] = new_layer
            self.origin_layer_names["network.13"] = new_layer
        elif 'network.13.0' == layer_name:
            self.network[13][0] = new_layer
            self.layer_names["network.13.0"] = new_layer
            self.origin_layer_names["network.13.0"] = new_layer
        elif 'network.13.1' == layer_name:
            self.network[13][1] = new_layer
            self.layer_names["network.13.1"] = new_layer
            self.origin_layer_names["network.13.1"] = new_layer
        elif 'network.13.2' == layer_name:
            self.network[13][2] = new_layer
            self.layer_names["network.13.2"] = new_layer
            self.origin_layer_names["network.13.2"] = new_layer
        elif 'network.14' == layer_name:
            self.network[14] = new_layer
            self.layer_names["network.14"] = new_layer
            self.origin_layer_names["network.14"] = new_layer
        elif 'network.14.0' == layer_name:
            self.network[14][0] = new_layer
            self.layer_names["network.14.0"] = new_layer
            self.origin_layer_names["network.14.0"] = new_layer
        elif 'network.14.1' == layer_name:
            self.network[14][1] = new_layer
            self.layer_names["network.14.1"] = new_layer
            self.origin_layer_names["network.14.1"] = new_layer
        elif 'network.14.2' == layer_name:
            self.network[14][2] = new_layer
            self.layer_names["network.14.2"] = new_layer
            self.origin_layer_names["network.14.2"] = new_layer
        elif 'network.15' == layer_name:
            self.network[15] = new_layer
            self.layer_names["network.15"] = new_layer
            self.origin_layer_names["network.15"] = new_layer
        elif 'network.15.0' == layer_name:
            self.network[15][0] = new_layer
            self.layer_names["network.15.0"] = new_layer
            self.origin_layer_names["network.15.0"] = new_layer
        elif 'network.15.1' == layer_name:
            self.network[15][1] = new_layer
            self.layer_names["network.15.1"] = new_layer
            self.origin_layer_names["network.15.1"] = new_layer
        elif 'network.15.2' == layer_name:
            self.network[15][2] = new_layer
            self.layer_names["network.15.2"] = new_layer
            self.origin_layer_names["network.15.2"] = new_layer
        elif 'network.16' == layer_name:
            self.network[16] = new_layer
            self.layer_names["network.16"] = new_layer
            self.origin_layer_names["network.16"] = new_layer
        elif 'network.16.0' == layer_name:
            self.network[16][0] = new_layer
            self.layer_names["network.16.0"] = new_layer
            self.origin_layer_names["network.16.0"] = new_layer
        elif 'network.16.1' == layer_name:
            self.network[16][1] = new_layer
            self.layer_names["network.16.1"] = new_layer
            self.origin_layer_names["network.16.1"] = new_layer
        elif 'network.16.2' == layer_name:
            self.network[16][2] = new_layer
            self.layer_names["network.16.2"] = new_layer
            self.origin_layer_names["network.16.2"] = new_layer
        elif 'network.17' == layer_name:
            self.network[17] = new_layer
            self.layer_names["network.17"] = new_layer
            self.origin_layer_names["network.17"] = new_layer
        elif 'network.17.0' == layer_name:
            self.network[17][0] = new_layer
            self.layer_names["network.17.0"] = new_layer
            self.origin_layer_names["network.17.0"] = new_layer
        elif 'network.17.1' == layer_name:
            self.network[17][1] = new_layer
            self.layer_names["network.17.1"] = new_layer
            self.origin_layer_names["network.17.1"] = new_layer
        elif 'network.17.2' == layer_name:
            self.network[17][2] = new_layer
            self.layer_names["network.17.2"] = new_layer
            self.origin_layer_names["network.17.2"] = new_layer
        elif 'network.18' == layer_name:
            self.network[18] = new_layer
            self.layer_names["network.18"] = new_layer
            self.origin_layer_names["network.18"] = new_layer
        elif 'network.18.0' == layer_name:
            self.network[18][0] = new_layer
            self.layer_names["network.18.0"] = new_layer
            self.origin_layer_names["network.18.0"] = new_layer
        elif 'network.18.1' == layer_name:
            self.network[18][1] = new_layer
            self.layer_names["network.18.1"] = new_layer
            self.origin_layer_names["network.18.1"] = new_layer
        elif 'network.18.2' == layer_name:
            self.network[18][2] = new_layer
            self.layer_names["network.18.2"] = new_layer
            self.origin_layer_names["network.18.2"] = new_layer
        elif 'network.19' == layer_name:
            self.network[19] = new_layer
            self.layer_names["network.19"] = new_layer
            self.origin_layer_names["network.19"] = new_layer
        elif 'network.19.0' == layer_name:
            self.network[19][0] = new_layer
            self.layer_names["network.19.0"] = new_layer
            self.origin_layer_names["network.19.0"] = new_layer
        elif 'network.19.1' == layer_name:
            self.network[19][1] = new_layer
            self.layer_names["network.19.1"] = new_layer
            self.origin_layer_names["network.19.1"] = new_layer
        elif 'network.19.2' == layer_name:
            self.network[19][2] = new_layer
            self.layer_names["network.19.2"] = new_layer
            self.origin_layer_names["network.19.2"] = new_layer
        elif 'network.20' == layer_name:
            self.network[20] = new_layer
            self.layer_names["network.20"] = new_layer
            self.origin_layer_names["network.20"] = new_layer
        elif 'network.20.0' == layer_name:
            self.network[20][0] = new_layer
            self.layer_names["network.20.0"] = new_layer
            self.origin_layer_names["network.20.0"] = new_layer
        elif 'network.20.1' == layer_name:
            self.network[20][1] = new_layer
            self.layer_names["network.20.1"] = new_layer
            self.origin_layer_names["network.20.1"] = new_layer
        elif 'network.20.2' == layer_name:
            self.network[20][2] = new_layer
            self.layer_names["network.20.2"] = new_layer
            self.origin_layer_names["network.20.2"] = new_layer
        elif 'network.21' == layer_name:
            self.network[21] = new_layer
            self.layer_names["network.21"] = new_layer
            self.origin_layer_names["network.21"] = new_layer
        elif 'network.21.0' == layer_name:
            self.network[21][0] = new_layer
            self.layer_names["network.21.0"] = new_layer
            self.origin_layer_names["network.21.0"] = new_layer
        elif 'network.21.1' == layer_name:
            self.network[21][1] = new_layer
            self.layer_names["network.21.1"] = new_layer
            self.origin_layer_names["network.21.1"] = new_layer
        elif 'network.21.2' == layer_name:
            self.network[21][2] = new_layer
            self.layer_names["network.21.2"] = new_layer
            self.origin_layer_names["network.21.2"] = new_layer
        elif 'network.22' == layer_name:
            self.network[22] = new_layer
            self.layer_names["network.22"] = new_layer
            self.origin_layer_names["network.22"] = new_layer
        elif 'network.22.0' == layer_name:
            self.network[22][0] = new_layer
            self.layer_names["network.22.0"] = new_layer
            self.origin_layer_names["network.22.0"] = new_layer
        elif 'network.22.1' == layer_name:
            self.network[22][1] = new_layer
            self.layer_names["network.22.1"] = new_layer
            self.origin_layer_names["network.22.1"] = new_layer
        elif 'network.22.2' == layer_name:
            self.network[22][2] = new_layer
            self.layer_names["network.22.2"] = new_layer
            self.origin_layer_names["network.22.2"] = new_layer
        elif 'network.23' == layer_name:
            self.network[23] = new_layer
            self.layer_names["network.23"] = new_layer
            self.origin_layer_names["network.23"] = new_layer
        elif 'network.23.0' == layer_name:
            self.network[23][0] = new_layer
            self.layer_names["network.23.0"] = new_layer
            self.origin_layer_names["network.23.0"] = new_layer
        elif 'network.23.1' == layer_name:
            self.network[23][1] = new_layer
            self.layer_names["network.23.1"] = new_layer
            self.origin_layer_names["network.23.1"] = new_layer
        elif 'network.23.2' == layer_name:
            self.network[23][2] = new_layer
            self.layer_names["network.23.2"] = new_layer
            self.origin_layer_names["network.23.2"] = new_layer
        elif 'network.24' == layer_name:
            self.network[24] = new_layer
            self.layer_names["network.24"] = new_layer
            self.origin_layer_names["network.24"] = new_layer
        elif 'network.24.0' == layer_name:
            self.network[24][0] = new_layer
            self.layer_names["network.24.0"] = new_layer
            self.origin_layer_names["network.24.0"] = new_layer
        elif 'network.24.1' == layer_name:
            self.network[24][1] = new_layer
            self.layer_names["network.24.1"] = new_layer
            self.origin_layer_names["network.24.1"] = new_layer
        elif 'network.24.2' == layer_name:
            self.network[24][2] = new_layer
            self.layer_names["network.24.2"] = new_layer
            self.origin_layer_names["network.24.2"] = new_layer
        elif 'network.25' == layer_name:
            self.network[25] = new_layer
            self.layer_names["network.25"] = new_layer
            self.origin_layer_names["network.25"] = new_layer
        elif 'network.25.0' == layer_name:
            self.network[25][0] = new_layer
            self.layer_names["network.25.0"] = new_layer
            self.origin_layer_names["network.25.0"] = new_layer
        elif 'network.25.1' == layer_name:
            self.network[25][1] = new_layer
            self.layer_names["network.25.1"] = new_layer
            self.origin_layer_names["network.25.1"] = new_layer
        elif 'network.25.2' == layer_name:
            self.network[25][2] = new_layer
            self.layer_names["network.25.2"] = new_layer
            self.origin_layer_names["network.25.2"] = new_layer
        elif 'network.26' == layer_name:
            self.network[26] = new_layer
            self.layer_names["network.26"] = new_layer
            self.origin_layer_names["network.26"] = new_layer
        elif 'network.26.0' == layer_name:
            self.network[26][0] = new_layer
            self.layer_names["network.26.0"] = new_layer
            self.origin_layer_names["network.26.0"] = new_layer
        elif 'network.26.1' == layer_name:
            self.network[26][1] = new_layer
            self.layer_names["network.26.1"] = new_layer
            self.origin_layer_names["network.26.1"] = new_layer
        elif 'network.26.2' == layer_name:
            self.network[26][2] = new_layer
            self.layer_names["network.26.2"] = new_layer
            self.origin_layer_names["network.26.2"] = new_layer

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
    image = torch.tensor(np.random.randn(4, 3, 300, 300), dtype=torch.float32)
    num_matched_boxes = torch.tensor([[33]], dtype=torch.int32)
    gt_label = torch.tensor(np.random.rand(4, 1917), dtype=torch.int32)
    gt_loc = torch.tensor(np.random.randn(4, 1917, 4), dtype=torch.float32)

    network = SSDWithMobileNetV1()
    pred_loc, pred_label = network(image)
    # Define the learning rate
    lr = 1e-4

    # # Define the optimizer
    # opt = optim.SGD(filter(lambda x: x.requires_grad, network.parameters()), lr,
    #                 momentum=0.9, weight_decay=0.00015)
    #
    #
    # # Define the forward procedure
    # def forward_fn(x, gt_loc, gt_label, num_matched_boxes):
    #     pred_loc, pred_label = network(x)
    #     mask = (gt_label > 0).float()
    #     num_matched_boxes = num_matched_boxes.float().sum()
    #
    #     # Positioning loss
    #     mask_loc = mask.unsqueeze(-1).repeat(1, 1, 4)
    #     smooth_l1 = torch.nn.SmoothL1Loss(reduction='none')(pred_loc, gt_loc) * mask_loc
    #     loss_loc = smooth_l1.sum(dim=-1).sum(dim=-1)
    #
    #     # Category loss
    #     loss_cls = class_loss(pred_label, gt_label)
    #     loss_cls = loss_cls.sum(dim=(1, 2))
    #
    #     return ((loss_cls + loss_loc) / num_matched_boxes).sum()
    #
    #
    # # Gradient updates
    # def train_step(x, gt_loc, gt_label, num_matched_boxes):
    #     opt.zero_grad()
    #     loss = forward_fn(x, gt_loc, gt_label, num_matched_boxes)
    #     loss.backward()
    #     opt.step()
    #     return loss.item()
    #
    #
    # for epoch in range(5):
    #     network.train()
    #     loss = train_step(image, gt_loc, gt_label, num_matched_boxes)
    #     print("loss: " + str(loss))
