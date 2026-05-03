import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SSD.ssd_utils_torch import FpnTopDown, BottomUp, WeightSharedMultiBox, FeatureSelector, conv_bn_relu


class MobileNetV1(nn.Module):
    def __init__(self, class_num=1001, features_only=False):
        super(MobileNetV1, self).__init__()
        self.features_only = features_only
        self.net = nn.Sequential(
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
        )
        if not features_only:
            self.fc = nn.Linear(1024, class_num)

    def forward(self, x):
        if self.features_only:
            features = []
            for layer in self.net:
                x = layer(x)
                features.append(x)
            return features
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


# class FeatureSelector(nn.Module):
#     def __init__(self, layer_indexes):
#         super(FeatureSelector, self).__init__()
#         self.layer_indexes = layer_indexes
#
#     def forward(self, features):
#         return tuple(features[i] for i in self.layer_indexes)


class SSDMobileNetV1FPN_torch(nn.Module):
    def __init__(self):
        super(SSDMobileNetV1FPN_torch, self).__init__()
        self.network = nn.ModuleList([
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
        ])
        layer_indexes = [10, 22, 26]
        self.selector = FeatureSelector(layer_indexes)

        self.fpn = FpnTopDown([256, 512, 1024], 256)
        self.bottom_up = BottomUp(2, 256, 3, 2)
        self.num_classes, self.out_channels, self.num_default, self.num_features, self.num_addition_layers, self.num_ssd_boxes = 81, \
                                                                                                                                 256, \
                                                                                                                                 6, \
                                                                                                                                 5, \
                                                                                                                                 4, \
                                                                                                                                 51150

        self.multi_box = WeightSharedMultiBox(self.num_classes, self.out_channels, self.num_default, self.num_features,
                                              self.num_addition_layers,
                                              self.num_ssd_boxes, loc_cls_shared_addition=False)

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
            'network.10.2': ["network.10.1", ["network.11.0", "OUTPUT1"]],
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
            'network.14.2': ["network.14.1", "network.15.0"],
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
            'network.22.2': ["network.22.1", ["network.23.0", "OUTPUT2"]],
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
            'network.26.2': ["network.26.1", "OUTPUT3"],

        }

        self.in_shapes = {
            "INPUT": [1, 3, 640, 640],
            'network.0.0': [1, 3, 640, 640],
            'network.0.1': [1, 32, 320, 320],
            'network.0.2': [1, 32, 320, 320],
            'network.1.0': [1, 32, 320, 320],
            'network.1.1': [1, 32, 320, 320],
            'network.1.2': [1, 32, 320, 320],
            'network.2.0': [1, 32, 320, 320],
            'network.2.1': [1, 64, 320, 320],
            'network.2.2': [1, 64, 320, 320],
            'network.3.0': [1, 64, 320, 320],
            'network.3.1': [1, 64, 160, 160],
            'network.3.2': [1, 64, 160, 160],
            'network.4.0': [1, 64, 160, 160],
            'network.4.1': [1, 128, 160, 160],
            'network.4.2': [1, 128, 160, 160],
            'network.5.0': [1, 128, 160, 160],
            'network.5.1': [1, 128, 160, 160],
            'network.5.2': [1, 128, 160, 160],
            'network.6.0': [1, 128, 160, 160],
            'network.6.1': [1, 128, 160, 160],
            'network.6.2': [1, 128, 160, 160],
            'network.7.0': [1, 128, 160, 160],
            'network.7.1': [1, 128, 80, 80],
            'network.7.2': [1, 128, 80, 80],
            'network.8.0': [1, 128, 80, 80],
            'network.8.1': [1, 256, 80, 80],
            'network.8.2': [1, 256, 80, 80],
            'network.9.0': [1, 256, 80, 80],
            'network.9.1': [1, 256, 80, 80],
            'network.9.2': [1, 256, 80, 80],
            'network.10.0': [1, 256, 80, 80],
            'network.10.1': [1, 256, 80, 80],
            'network.10.2': [1, 256, 80, 80],
            'network.11.0': [1, 256, 80, 80],
            'network.11.1': [1, 256, 40, 40],
            'network.11.2': [1, 256, 40, 40],
            'network.12.0': [1, 256, 40, 40],
            'network.12.1': [1, 512, 40, 40],
            'network.12.2': [1, 512, 40, 40],
            'network.13.0': [1, 512, 40, 40],
            'network.13.1': [1, 512, 40, 40],
            'network.13.2': [1, 512, 40, 40],
            'network.14.0': [1, 512, 40, 40],
            'network.14.1': [1, 512, 40, 40],
            'network.14.2': [1, 512, 40, 40],
            'network.15.0': [1, 512, 40, 40],
            'network.15.1': [1, 512, 40, 40],
            'network.15.2': [1, 512, 40, 40],
            'network.16.0': [1, 512, 40, 40],
            'network.16.1': [1, 512, 40, 40],
            'network.16.2': [1, 512, 40, 40],
            'network.17.0': [1, 512, 40, 40],
            'network.17.1': [1, 512, 40, 40],
            'network.17.2': [1, 512, 40, 40],
            'network.18.0': [1, 512, 40, 40],
            'network.18.1': [1, 512, 40, 40],
            'network.18.2': [1, 512, 40, 40],
            'network.19.0': [1, 512, 40, 40],
            'network.19.1': [1, 512, 40, 40],
            'network.19.2': [1, 512, 40, 40],
            'network.20.0': [1, 512, 40, 40],
            'network.20.1': [1, 512, 40, 40],
            'network.20.2': [1, 512, 40, 40],
            'network.21.0': [1, 512, 40, 40],
            'network.21.1': [1, 512, 40, 40],
            'network.21.2': [1, 512, 40, 40],
            'network.22.0': [1, 512, 40, 40],
            'network.22.1': [1, 512, 40, 40],
            'network.22.2': [1, 512, 40, 40],
            'network.23.0': [1, 512, 40, 40],
            'network.23.1': [1, 512, 20, 20],
            'network.23.2': [1, 512, 20, 20],
            'network.24.0': [1, 512, 20, 20],
            'network.24.1': [1, 1024, 20, 20],
            'network.24.2': [1, 1024, 20, 20],
            'network.25.0': [1, 1024, 20, 20],
            'network.25.1': [1, 1024, 20, 20],
            'network.25.2': [1, 1024, 20, 20],
            'network.26.0': [1, 1024, 20, 20],
            'network.26.1': [1, 1024, 20, 20],
            'network.26.2': [1, 1024, 20, 20],
            "OUTPUT1": [1, 256, 80, 80],
            "OUTPUT2": [1, 512, 40, 40],
            "OUTPUT3": [1, 1024, 20, 20],
        }

        self.out_shapes = {
            "INPUT": [1, 3, 640, 640],
            'network.0.0': [1, 32, 320, 320],
            'network.0.1': [1, 32, 320, 320],
            'network.0.2': [1, 32, 320, 320],
            'network.1.0': [1, 32, 320, 320],
            'network.1.1': [1, 32, 320, 320],
            'network.1.2': [1, 32, 320, 320],
            'network.2.0': [1, 64, 320, 320],
            'network.2.1': [1, 64, 320, 320],
            'network.2.2': [1, 64, 320, 320],
            'network.3.0': [1, 64, 160, 160],
            'network.3.1': [1, 64, 160, 160],
            'network.3.2': [1, 64, 160, 160],
            'network.4.0': [1, 128, 160, 160],
            'network.4.1': [1, 128, 160, 160],
            'network.4.2': [1, 128, 160, 160],
            'network.5.0': [1, 128, 160, 160],
            'network.5.1': [1, 128, 160, 160],
            'network.5.2': [1, 128, 160, 160],
            'network.6.0': [1, 128, 160, 160],
            'network.6.1': [1, 128, 160, 160],
            'network.6.2': [1, 128, 160, 160],
            'network.7.0': [1, 128, 80, 80],
            'network.7.1': [1, 128, 80, 80],
            'network.7.2': [1, 128, 80, 80],
            'network.8.0': [1, 256, 80, 80],
            'network.8.1': [1, 256, 80, 80],
            'network.8.2': [1, 256, 80, 80],
            'network.9.0': [1, 256, 80, 80],
            'network.9.1': [1, 256, 80, 80],
            'network.9.2': [1, 256, 80, 80],
            'network.10.0': [1, 256, 80, 80],
            'network.10.1': [1, 256, 80, 80],
            'network.10.2': [1, 256, 80, 80],
            'network.11.0': [1, 256, 40, 40],
            'network.11.1': [1, 256, 40, 40],
            'network.11.2': [1, 256, 40, 40],
            'network.12.0': [1, 512, 40, 40],
            'network.12.1': [1, 512, 40, 40],
            'network.12.2': [1, 512, 40, 40],
            'network.13.0': [1, 512, 40, 40],
            'network.13.1': [1, 512, 40, 40],
            'network.13.2': [1, 512, 40, 40],
            'network.14.0': [1, 512, 40, 40],
            'network.14.1': [1, 512, 40, 40],
            'network.14.2': [1, 512, 40, 40],
            'network.15.0': [1, 512, 40, 40],
            'network.15.1': [1, 512, 40, 40],
            'network.15.2': [1, 512, 40, 40],
            'network.16.0': [1, 512, 40, 40],
            'network.16.1': [1, 512, 40, 40],
            'network.16.2': [1, 512, 40, 40],
            'network.17.0': [1, 512, 40, 40],
            'network.17.1': [1, 512, 40, 40],
            'network.17.2': [1, 512, 40, 40],
            'network.18.0': [1, 512, 40, 40],
            'network.18.1': [1, 512, 40, 40],
            'network.18.2': [1, 512, 40, 40],
            'network.19.0': [1, 512, 40, 40],
            'network.19.1': [1, 512, 40, 40],
            'network.19.2': [1, 512, 40, 40],
            'network.20.0': [1, 512, 40, 40],
            'network.20.1': [1, 512, 40, 40],
            'network.20.2': [1, 512, 40, 40],
            'network.21.0': [1, 512, 40, 40],
            'network.21.1': [1, 512, 40, 40],
            'network.21.2': [1, 512, 40, 40],
            'network.22.0': [1, 512, 40, 40],
            'network.22.1': [1, 512, 40, 40],
            'network.22.2': [1, 512, 40, 40],
            'network.23.0': [1, 512, 20, 20],
            'network.23.1': [1, 512, 20, 20],
            'network.23.2': [1, 512, 20, 20],
            'network.24.0': [1, 1024, 20, 20],
            'network.24.1': [1, 1024, 20, 20],
            'network.24.2': [1, 1024, 20, 20],
            'network.25.0': [1, 1024, 20, 20],
            'network.25.1': [1, 1024, 20, 20],
            'network.25.2': [1, 1024, 20, 20],
            'network.26.0': [1, 1024, 20, 20],
            'network.26.1': [1, 1024, 20, 20],
            'network.26.2': [1, 1024, 20, 20],
            "OUTPUT1": [1, 256, 80, 80],
            "OUTPUT2": [1, 512, 40, 40],
            "OUTPUT3": [1, 1024, 20, 20],
        }

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []


    def forward(self, x):
        features = []
        for i, layer in enumerate(self.network):
            x = layer(x)
            features.append(x)
        selected_features = self.selector(features)

        features = self.fpn(selected_features)
        features = self.bottom_up(features)
        pred_loc, pred_label = self.multi_box(features)

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





def mobilenet_v1(class_num=1001):
    return MobileNetV1(class_num)

# class SSDMobileNetV1FPN_torch(nn.Module):
#     def __init__(self):
#         super(SSDMobileNetV1FPN_torch, self).__init__()
#         self.network = SSDMobileNetV1FPN()
#         self.fpn = FpnTopDown([256, 512, 1024], 256)
#         self.bottom_up = BottomUp(2, 256, 3, 2)
#         self.num_classes, self.out_channels, self.num_default, self.num_features, self.num_addition_layers, self.num_ssd_boxes = 81, \
#                                                                                                                                 256, \
#                                                                                                                                 6, \
#                                                                                                                                 5, \
#                                                                                                                                 4, \
#                                                                                                                                 51150
#
#         self.multi_box = WeightSharedMultiBox(self.num_classes, self.out_channels, self.num_default, self.num_features,
#                                               self.num_addition_layers,
#                                               self.num_ssd_boxes, loc_cls_shared_addition=False)
#
#
#
#
#
#     def forward(self, x):
#         features = self.network(x)
#         features = self.fpn(features)
#         features = self.bottom_up(features)
#         result = self.multi_box(features)
#         return result




if __name__ == '__main__':
    image = torch.Tensor(np.random.rand(1, 3, 640, 640))
    model = SSDMobileNetV1FPN_torch()
    out = model(image)
    print(out[0].shape, out[1].shape)