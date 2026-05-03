# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""deeplabv3plus network"""
import mindspore.nn as nn
from mindspore.ops import operations as P


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, weight_init='xavier_uniform')


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=padding,
                     dilation=dilation, weight_init='xavier_uniform')


class Resnet(nn.Cell):
    """Resnet"""

    def __init__(self, block, block_num, output_stride, use_batch_statistics=True):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, pad_mode='pad', padding=3,
                               weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm2d(self.inplanes, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, block_num[0], use_batch_statistics=use_batch_statistics)
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2, use_batch_statistics=use_batch_statistics)

        if output_stride == 16:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=2, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)
        elif output_stride == 8:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=1, base_dilation=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=4, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)

    def _make_layer(self, block, planes, blocks, stride=1, base_dilation=1, grids=None, use_batch_statistics=True):
        """Resnet._make_layer"""
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, use_batch_statistics=use_batch_statistics)
            ])

        if grids is None:
            grids = [1] * blocks

        layers = [
            block(self.inplanes, planes, stride, downsample, dilation=base_dilation * grids[0],
                  use_batch_statistics=use_batch_statistics)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=base_dilation * grids[i],
                      use_batch_statistics=use_batch_statistics))

        return nn.SequentialCell(layers)

    def construct(self, x):
        """Resnet.construct"""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        low_level_feat = out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out, low_level_feat


class Bottleneck(nn.Cell):
    """Bottleneck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_batch_statistics=True):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        self.conv2 = conv3x3(planes, planes, stride, dilation, dilation)
        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, use_batch_statistics=use_batch_statistics)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.add = P.Add()

    def construct(self, x):
        """Bottleneck.construct"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)
        return out


class ASPPConv(nn.Cell):
    """ASPPConv"""

    def __init__(self, in_channels, out_channels, atrous_rate=1, use_batch_statistics=True):
        super(ASPPConv, self).__init__()
        if atrous_rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init='xavier_uniform')
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', padding=atrous_rate,
                             dilation=atrous_rate, weight_init='xavier_uniform')
        bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        relu = nn.ReLU()
        self.aspp_conv = nn.SequentialCell([conv, bn, relu])

    def construct(self, x):
        out = self.aspp_conv(x)
        return out


class ASPPPooling(nn.Cell):
    """ASPPPooling"""

    def __init__(self, in_channels, out_channels, use_batch_statistics=True):
        super(ASPPPooling, self).__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, weight_init='xavier_uniform'),
            nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics),
            nn.ReLU()
        ])
        self.shape = P.Shape()

    def construct(self, x):
        size = self.shape(x)
        out = nn.AvgPool2d(size[2])(x)
        out = self.conv(out)
        out = P.ResizeNearestNeighbor((size[2], size[3]), True)(out)
        return out


class ASPP(nn.Cell):
    """ASPP"""

    def __init__(self, atrous_rates, phase='train', in_channels=2048, num_classes=21,
                 use_batch_statistics=True):
        super(ASPP, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        out_channels = 256
        self.aspp1 = ASPPConv(in_channels, out_channels, atrous_rates[0], use_batch_statistics=use_batch_statistics)
        self.aspp2 = ASPPConv(in_channels, out_channels, atrous_rates[1], use_batch_statistics=use_batch_statistics)
        self.aspp3 = ASPPConv(in_channels, out_channels, atrous_rates[2], use_batch_statistics=use_batch_statistics)
        self.aspp4 = ASPPConv(in_channels, out_channels, atrous_rates[3], use_batch_statistics=use_batch_statistics)
        self.aspp_pooling = ASPPPooling(in_channels, out_channels, use_batch_statistics=use_batch_statistics)
        self.conv1 = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1,
                               weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.concat = P.Concat(axis=1)
        self.drop = nn.Dropout(0.3)

    def construct(self, x):
        """ASPP.construct"""
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp_pooling(x)

        x = self.concat((x1, x2))
        x = self.concat((x, x3))
        x = self.concat((x, x4))
        x = self.concat((x, x5))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.phase == 'train':
            x = self.drop(x)
        return x


class DeepLabV3Plus(nn.Cell):
    """DeepLabV3Plus"""

    def __init__(self, phase='train', num_classes=21, output_stride=16, freeze_bn=False):
        super(DeepLabV3Plus, self).__init__()
        use_batch_statistics = not freeze_bn
        self.resnet = Resnet(Bottleneck, [3, 4, 23, 3], output_stride=output_stride,
                             use_batch_statistics=use_batch_statistics)
        self.aspp = ASPP([1, 6, 12, 18], phase, 2048, num_classes,
                         use_batch_statistics=use_batch_statistics)
        self.shape = P.Shape()
        self.conv2 = nn.Conv2d(256, 48, kernel_size=1, weight_init='xavier_uniform')
        self.bn2 = nn.BatchNorm2d(48, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.concat = P.Concat(axis=1)
        self.last_conv = nn.SequentialCell([
            conv3x3(304, 256, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(256, use_batch_statistics=use_batch_statistics),
            nn.ReLU(),
            conv3x3(256, 256, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(256, use_batch_statistics=use_batch_statistics),
            nn.ReLU(),
            conv1x1(256, num_classes, stride=1)
        ])

        self.in_shapes = {
            'INPUT': [1, 3, 513, 513],
            'resnet.conv1': [1, 3, 513, 513],
            'resnet.bn1': [1, 64, 257, 257],
            'resnet.relu': [1, 64, 257, 257],
            'resnet.maxpool': [1, 64, 257, 257],
            'resnet.layer1.0.conv1': [1, 64, 129, 129],
            'resnet.layer1.0.bn1': [1, 64, 129, 129],
            'resnet.layer1.0.conv2': [1, 64, 129, 129],
            'resnet.layer1.0.bn2': [1, 64, 129, 129],
            'resnet.layer1.0.conv3': [1, 64, 129, 129],
            'resnet.layer1.0.bn3': [1, 256, 129, 129],
            'resnet.layer1.0.relu': [1, 256, 129, 129],
            'resnet.layer1.0.downsample.0': [1, 64, 129, 129],
            'resnet.layer1.0.downsample.1': [1, 256, 129, 129],
            'resnet.layer1.1.conv1': [1, 256, 129, 129],
            'resnet.layer1.1.bn1': [1, 64, 129, 129],
            'resnet.layer1.1.conv2': [1, 64, 129, 129],
            'resnet.layer1.1.bn2': [1, 64, 129, 129],
            'resnet.layer1.1.conv3': [1, 64, 129, 129],
            'resnet.layer1.1.bn3': [1, 256, 129, 129],
            'resnet.layer1.1.relu': [1, 256, 129, 129],
            'resnet.layer1.2.conv1': [1, 256, 129, 129],
            'resnet.layer1.2.bn1': [1, 64, 129, 129],
            'resnet.layer1.2.conv2': [1, 64, 129, 129],
            'resnet.layer1.2.bn2': [1, 64, 129, 129],
            'resnet.layer1.2.conv3': [1, 64, 129, 129],
            'resnet.layer1.2.bn3': [1, 256, 129, 129],
            'resnet.layer1.2.relu': [1, 256, 129, 129],
            'resnet.layer2.0.conv1': [1, 256, 129, 129],
            'resnet.layer2.0.bn1': [1, 128, 129, 129],
            'resnet.layer2.0.conv2': [1, 128, 129, 129],
            'resnet.layer2.0.bn2': [1, 128, 65, 65],
            'resnet.layer2.0.conv3': [1, 128, 65, 65],
            'resnet.layer2.0.bn3': [1, 512, 65, 65],
            'resnet.layer2.0.relu': [1, 512, 65, 65],
            'resnet.layer2.0.downsample.0': [1, 256, 129, 129],
            'resnet.layer2.0.downsample.1': [1, 512, 65, 65],
            'resnet.layer2.1.conv1': [1, 512, 65, 65],
            'resnet.layer2.1.bn1': [1, 128, 65, 65],
            'resnet.layer2.1.conv2': [1, 128, 65, 65],
            'resnet.layer2.1.bn2': [1, 128, 65, 65],
            'resnet.layer2.1.conv3': [1, 128, 65, 65],
            'resnet.layer2.1.bn3': [1, 512, 65, 65],
            'resnet.layer2.1.relu': [1, 512, 65, 65],
            'resnet.layer2.2.conv1': [1, 512, 65, 65],
            'resnet.layer2.2.bn1': [1, 128, 65, 65],
            'resnet.layer2.2.conv2': [1, 128, 65, 65],
            'resnet.layer2.2.bn2': [1, 128, 65, 65],
            'resnet.layer2.2.conv3': [1, 128, 65, 65],
            'resnet.layer2.2.bn3': [1, 512, 65, 65],
            'resnet.layer2.2.relu': [1, 512, 65, 65],
            'resnet.layer2.3.conv1': [1, 512, 65, 65],
            'resnet.layer2.3.bn1': [1, 128, 65, 65],
            'resnet.layer2.3.conv2': [1, 128, 65, 65],
            'resnet.layer2.3.bn2': [1, 128, 65, 65],
            'resnet.layer2.3.conv3': [1, 128, 65, 65],
            'resnet.layer2.3.bn3': [1, 512, 65, 65],
            'resnet.layer2.3.relu': [1, 512, 65, 65],
            'resnet.layer3.0.conv1': [1, 512, 65, 65],
            'resnet.layer3.0.bn1': [1, 256, 65, 65],
            'resnet.layer3.0.conv2': [1, 256, 65, 65],
            'resnet.layer3.0.bn2': [1, 256, 65, 65],
            'resnet.layer3.0.conv3': [1, 256, 65, 65],
            'resnet.layer3.0.bn3': [1, 1024, 65, 65],
            'resnet.layer3.0.relu': [1, 1024, 65, 65],
            'resnet.layer3.0.downsample.0': [1, 512, 65, 65],
            'resnet.layer3.0.downsample.1': [1, 1024, 65, 65],
            'resnet.layer3.1.conv1': [1, 1024, 65, 65],
            'resnet.layer3.1.bn1': [1, 256, 65, 65],
            'resnet.layer3.1.conv2': [1, 256, 65, 65],
            'resnet.layer3.1.bn2': [1, 256, 65, 65],
            'resnet.layer3.1.conv3': [1, 256, 65, 65],
            'resnet.layer3.1.bn3': [1, 1024, 65, 65],
            'resnet.layer3.1.relu': [1, 1024, 65, 65],
            'resnet.layer3.2.conv1': [1, 1024, 65, 65],
            'resnet.layer3.2.bn1': [1, 256, 65, 65],
            'resnet.layer3.2.conv2': [1, 256, 65, 65],
            'resnet.layer3.2.bn2': [1, 256, 65, 65],
            'resnet.layer3.2.conv3': [1, 256, 65, 65],
            'resnet.layer3.2.bn3': [1, 1024, 65, 65],
            'resnet.layer3.2.relu': [1, 1024, 65, 65],
            'resnet.layer3.3.conv1': [1, 1024, 65, 65],
            'resnet.layer3.3.bn1': [1, 256, 65, 65],
            'resnet.layer3.3.conv2': [1, 256, 65, 65],
            'resnet.layer3.3.bn2': [1, 256, 65, 65],
            'resnet.layer3.3.conv3': [1, 256, 65, 65],
            'resnet.layer3.3.bn3': [1, 1024, 65, 65],
            'resnet.layer3.3.relu': [1, 1024, 65, 65],
            'resnet.layer3.4.conv1': [1, 1024, 65, 65],
            'resnet.layer3.4.bn1': [1, 256, 65, 65],
            'resnet.layer3.4.conv2': [1, 256, 65, 65],
            'resnet.layer3.4.bn2': [1, 256, 65, 65],
            'resnet.layer3.4.conv3': [1, 256, 65, 65],
            'resnet.layer3.4.bn3': [1, 1024, 65, 65],
            'resnet.layer3.4.relu': [1, 1024, 65, 65],
            'resnet.layer3.5.conv1': [1, 1024, 65, 65],
            'resnet.layer3.5.bn1': [1, 256, 65, 65],
            'resnet.layer3.5.conv2': [1, 256, 65, 65],
            'resnet.layer3.5.bn2': [1, 256, 65, 65],
            'resnet.layer3.5.conv3': [1, 256, 65, 65],
            'resnet.layer3.5.bn3': [1, 1024, 65, 65],
            'resnet.layer3.5.relu': [1, 1024, 65, 65],
            'resnet.layer3.6.conv1': [1, 1024, 65, 65],
            'resnet.layer3.6.bn1': [1, 256, 65, 65],
            'resnet.layer3.6.conv2': [1, 256, 65, 65],
            'resnet.layer3.6.bn2': [1, 256, 65, 65],
            'resnet.layer3.6.conv3': [1, 256, 65, 65],
            'resnet.layer3.6.bn3': [1, 1024, 65, 65],
            'resnet.layer3.6.relu': [1, 1024, 65, 65],
            'resnet.layer3.7.conv1': [1, 1024, 65, 65],
            'resnet.layer3.7.bn1': [1, 256, 65, 65],
            'resnet.layer3.7.conv2': [1, 256, 65, 65],
            'resnet.layer3.7.bn2': [1, 256, 65, 65],
            'resnet.layer3.7.conv3': [1, 256, 65, 65],
            'resnet.layer3.7.bn3': [1, 1024, 65, 65],
            'resnet.layer3.7.relu': [1, 1024, 65, 65],
            'resnet.layer3.8.conv1': [1, 1024, 65, 65],
            'resnet.layer3.8.bn1': [1, 256, 65, 65],
            'resnet.layer3.8.conv2': [1, 256, 65, 65],
            'resnet.layer3.8.bn2': [1, 256, 65, 65],
            'resnet.layer3.8.conv3': [1, 256, 65, 65],
            'resnet.layer3.8.bn3': [1, 1024, 65, 65],
            'resnet.layer3.8.relu': [1, 1024, 65, 65],
            'resnet.layer3.9.conv1': [1, 1024, 65, 65],
            'resnet.layer3.9.bn1': [1, 256, 65, 65],
            'resnet.layer3.9.conv2': [1, 256, 65, 65],
            'resnet.layer3.9.bn2': [1, 256, 65, 65],
            'resnet.layer3.9.conv3': [1, 256, 65, 65],
            'resnet.layer3.9.bn3': [1, 1024, 65, 65],
            'resnet.layer3.9.relu': [1, 1024, 65, 65],
            'resnet.layer3.10.conv1': [1, 1024, 65, 65],
            'resnet.layer3.10.bn1': [1, 256, 65, 65],
            'resnet.layer3.10.conv2': [1, 256, 65, 65],
            'resnet.layer3.10.bn2': [1, 256, 65, 65],
            'resnet.layer3.10.conv3': [1, 256, 65, 65],
            'resnet.layer3.10.bn3': [1, 1024, 65, 65],
            'resnet.layer3.10.relu': [1, 1024, 65, 65],
            'resnet.layer3.11.conv1': [1, 1024, 65, 65],
            'resnet.layer3.11.bn1': [1, 256, 65, 65],
            'resnet.layer3.11.conv2': [1, 256, 65, 65],
            'resnet.layer3.11.bn2': [1, 256, 65, 65],
            'resnet.layer3.11.conv3': [1, 256, 65, 65],
            'resnet.layer3.11.bn3': [1, 1024, 65, 65],
            'resnet.layer3.11.relu': [1, 1024, 65, 65],
            'resnet.layer3.12.conv1': [1, 1024, 65, 65],
            'resnet.layer3.12.bn1': [1, 256, 65, 65],
            'resnet.layer3.12.conv2': [1, 256, 65, 65],
            'resnet.layer3.12.bn2': [1, 256, 65, 65],
            'resnet.layer3.12.conv3': [1, 256, 65, 65],
            'resnet.layer3.12.bn3': [1, 1024, 65, 65],
            'resnet.layer3.12.relu': [1, 1024, 65, 65],
            'resnet.layer3.13.conv1': [1, 1024, 65, 65],
            'resnet.layer3.13.bn1': [1, 256, 65, 65],
            'resnet.layer3.13.conv2': [1, 256, 65, 65],
            'resnet.layer3.13.bn2': [1, 256, 65, 65],
            'resnet.layer3.13.conv3': [1, 256, 65, 65],
            'resnet.layer3.13.bn3': [1, 1024, 65, 65],
            'resnet.layer3.13.relu': [1, 1024, 65, 65],
            'resnet.layer3.14.conv1': [1, 1024, 65, 65],
            'resnet.layer3.14.bn1': [1, 256, 65, 65],
            'resnet.layer3.14.conv2': [1, 256, 65, 65],
            'resnet.layer3.14.bn2': [1, 256, 65, 65],
            'resnet.layer3.14.conv3': [1, 256, 65, 65],
            'resnet.layer3.14.bn3': [1, 1024, 65, 65],
            'resnet.layer3.14.relu': [1, 1024, 65, 65],
            'resnet.layer3.15.conv1': [1, 1024, 65, 65],
            'resnet.layer3.15.bn1': [1, 256, 65, 65],
            'resnet.layer3.15.conv2': [1, 256, 65, 65],
            'resnet.layer3.15.bn2': [1, 256, 65, 65],
            'resnet.layer3.15.conv3': [1, 256, 65, 65],
            'resnet.layer3.15.bn3': [1, 1024, 65, 65],
            'resnet.layer3.15.relu': [1, 1024, 65, 65],
            'resnet.layer3.16.conv1': [1, 1024, 65, 65],
            'resnet.layer3.16.bn1': [1, 256, 65, 65],
            'resnet.layer3.16.conv2': [1, 256, 65, 65],
            'resnet.layer3.16.bn2': [1, 256, 65, 65],
            'resnet.layer3.16.conv3': [1, 256, 65, 65],
            'resnet.layer3.16.bn3': [1, 1024, 65, 65],
            'resnet.layer3.16.relu': [1, 1024, 65, 65],
            'resnet.layer3.17.conv1': [1, 1024, 65, 65],
            'resnet.layer3.17.bn1': [1, 256, 65, 65],
            'resnet.layer3.17.conv2': [1, 256, 65, 65],
            'resnet.layer3.17.bn2': [1, 256, 65, 65],
            'resnet.layer3.17.conv3': [1, 256, 65, 65],
            'resnet.layer3.17.bn3': [1, 1024, 65, 65],
            'resnet.layer3.17.relu': [1, 1024, 65, 65],
            'resnet.layer3.18.conv1': [1, 1024, 65, 65],
            'resnet.layer3.18.bn1': [1, 256, 65, 65],
            'resnet.layer3.18.conv2': [1, 256, 65, 65],
            'resnet.layer3.18.bn2': [1, 256, 65, 65],
            'resnet.layer3.18.conv3': [1, 256, 65, 65],
            'resnet.layer3.18.bn3': [1, 1024, 65, 65],
            'resnet.layer3.18.relu': [1, 1024, 65, 65],
            'resnet.layer3.19.conv1': [1, 1024, 65, 65],
            'resnet.layer3.19.bn1': [1, 256, 65, 65],
            'resnet.layer3.19.conv2': [1, 256, 65, 65],
            'resnet.layer3.19.bn2': [1, 256, 65, 65],
            'resnet.layer3.19.conv3': [1, 256, 65, 65],
            'resnet.layer3.19.bn3': [1, 1024, 65, 65],
            'resnet.layer3.19.relu': [1, 1024, 65, 65],
            'resnet.layer3.20.conv1': [1, 1024, 65, 65],
            'resnet.layer3.20.bn1': [1, 256, 65, 65],
            'resnet.layer3.20.conv2': [1, 256, 65, 65],
            'resnet.layer3.20.bn2': [1, 256, 65, 65],
            'resnet.layer3.20.conv3': [1, 256, 65, 65],
            'resnet.layer3.20.bn3': [1, 1024, 65, 65],
            'resnet.layer3.20.relu': [1, 1024, 65, 65],
            'resnet.layer3.21.conv1': [1, 1024, 65, 65],
            'resnet.layer3.21.bn1': [1, 256, 65, 65],
            'resnet.layer3.21.conv2': [1, 256, 65, 65],
            'resnet.layer3.21.bn2': [1, 256, 65, 65],
            'resnet.layer3.21.conv3': [1, 256, 65, 65],
            'resnet.layer3.21.bn3': [1, 1024, 65, 65],
            'resnet.layer3.21.relu': [1, 1024, 65, 65],
            'resnet.layer3.22.conv1': [1, 1024, 65, 65],
            'resnet.layer3.22.bn1': [1, 256, 65, 65],
            'resnet.layer3.22.conv2': [1, 256, 65, 65],
            'resnet.layer3.22.bn2': [1, 256, 65, 65],
            'resnet.layer3.22.conv3': [1, 256, 65, 65],
            'resnet.layer3.22.bn3': [1, 1024, 65, 65],
            'resnet.layer3.22.relu': [1, 1024, 65, 65],
            'resnet.layer4.0.conv1': [1, 1024, 65, 65],
            'resnet.layer4.0.bn1': [1, 512, 65, 65],
            'resnet.layer4.0.conv2': [1, 512, 65, 65],
            'resnet.layer4.0.bn2': [1, 512, 65, 65],
            'resnet.layer4.0.conv3': [1, 512, 65, 65],
            'resnet.layer4.0.bn3': [1, 2048, 65, 65],
            'resnet.layer4.0.relu': [1, 2048, 65, 65],
            'resnet.layer4.0.downsample.0': [1, 1024, 65, 65],
            'resnet.layer4.0.downsample.1': [1, 2048, 65, 65],
            'resnet.layer4.1.conv1': [1, 2048, 65, 65],
            'resnet.layer4.1.bn1': [1, 512, 65, 65],
            'resnet.layer4.1.conv2': [1, 512, 65, 65],
            'resnet.layer4.1.bn2': [1, 512, 65, 65],
            'resnet.layer4.1.conv3': [1, 512, 65, 65],
            'resnet.layer4.1.bn3': [1, 2048, 65, 65],
            'resnet.layer4.1.relu': [1, 2048, 65, 65],
            'resnet.layer4.2.conv1': [1, 2048, 65, 65],
            'resnet.layer4.2.bn1': [1, 512, 65, 65],
            'resnet.layer4.2.conv2': [1, 512, 65, 65],
            'resnet.layer4.2.bn2': [1, 512, 65, 65],
            'resnet.layer4.2.conv3': [1, 512, 65, 65],
            'resnet.layer4.2.bn3': [1, 2048, 65, 65],
            'resnet.layer4.2.relu': [1, 2048, 65, 65],
            'OUTPUT': [1, 2048, 65, 65]}
        self.out_shapes = {
            'INPUT': [1, 3, 513, 513],
            'resnet.conv1': [1, 64, 257, 257],
            'resnet.bn1': [1, 64, 257, 257],
            'resnet.relu': [1, 64, 257, 257],
            'resnet.maxpool': [1, 64, 129, 129],
            'resnet.layer1.0.conv1': [1, 64, 129, 129],
            'resnet.layer1.0.bn1': [1, 64, 129, 129],
            'resnet.layer1.0.conv2': [1, 64, 129, 129],
            'resnet.layer1.0.bn2': [1, 64, 129, 129],
            'resnet.layer1.0.conv3': [1, 256, 129, 129],
            'resnet.layer1.0.bn3': [1, 256, 129, 129],
            'resnet.layer1.0.relu': [1, 256, 129, 129],
            'resnet.layer1.0.downsample.0': [1, 256, 129, 129],
            'resnet.layer1.0.downsample.1': [1, 256, 129, 129],
            'resnet.layer1.1.conv1': [1, 64, 129, 129],
            'resnet.layer1.1.bn1': [1, 64, 129, 129],
            'resnet.layer1.1.conv2': [1, 64, 129, 129],
            'resnet.layer1.1.bn2': [1, 64, 129, 129],
            'resnet.layer1.1.conv3': [1, 256, 129, 129],
            'resnet.layer1.1.bn3': [1, 256, 129, 129],
            'resnet.layer1.1.relu': [1, 256, 129, 129],
            'resnet.layer1.2.conv1': [1, 64, 129, 129],
            'resnet.layer1.2.bn1': [1, 64, 129, 129],
            'resnet.layer1.2.conv2': [1, 64, 129, 129],
            'resnet.layer1.2.bn2': [1, 64, 129, 129],
            'resnet.layer1.2.conv3': [1, 256, 129, 129],
            'resnet.layer1.2.bn3': [1, 256, 129, 129],
            'resnet.layer1.2.relu': [1, 256, 129, 129],
            'resnet.layer2.0.conv1': [1, 128, 129, 129],
            'resnet.layer2.0.bn1': [1, 128, 129, 129],
            'resnet.layer2.0.conv2': [1, 128, 65, 65],
            'resnet.layer2.0.bn2': [1, 128, 65, 65],
            'resnet.layer2.0.conv3': [1, 512, 65, 65],
            'resnet.layer2.0.bn3': [1, 512, 65, 65],
            'resnet.layer2.0.relu': [1, 512, 65, 65],
            'resnet.layer2.0.downsample.0': [1, 512, 65, 65],
            'resnet.layer2.0.downsample.1': [1, 512, 65, 65],
            'resnet.layer2.1.conv1': [1, 128, 65, 65],
            'resnet.layer2.1.bn1': [1, 128, 65, 65],
            'resnet.layer2.1.conv2': [1, 128, 65, 65],
            'resnet.layer2.1.bn2': [1, 128, 65, 65],
            'resnet.layer2.1.conv3': [1, 512, 65, 65],
            'resnet.layer2.1.bn3': [1, 512, 65, 65],
            'resnet.layer2.1.relu': [1, 512, 65, 65],
            'resnet.layer2.2.conv1': [1, 128, 65, 65],
            'resnet.layer2.2.bn1': [1, 128, 65, 65],
            'resnet.layer2.2.conv2': [1, 128, 65, 65],
            'resnet.layer2.2.bn2': [1, 128, 65, 65],
            'resnet.layer2.2.conv3': [1, 512, 65, 65],
            'resnet.layer2.2.bn3': [1, 512, 65, 65],
            'resnet.layer2.2.relu': [1, 512, 65, 65],
            'resnet.layer2.3.conv1': [1, 128, 65, 65],
            'resnet.layer2.3.bn1': [1, 128, 65, 65],
            'resnet.layer2.3.conv2': [1, 128, 65, 65],
            'resnet.layer2.3.bn2': [1, 128, 65, 65],
            'resnet.layer2.3.conv3': [1, 512, 65, 65],
            'resnet.layer2.3.bn3': [1, 512, 65, 65],
            'resnet.layer2.3.relu': [1, 512, 65, 65],
            'resnet.layer3.0.conv1': [1, 256, 65, 65],
            'resnet.layer3.0.bn1': [1, 256, 65, 65],
            'resnet.layer3.0.conv2': [1, 256, 65, 65],
            'resnet.layer3.0.bn2': [1, 256, 65, 65],
            'resnet.layer3.0.conv3': [1, 1024, 65, 65],
            'resnet.layer3.0.bn3': [1, 1024, 65, 65],
            'resnet.layer3.0.relu': [1, 1024, 65, 65],
            'resnet.layer3.0.downsample.0': [1, 1024, 65, 65],
            'resnet.layer3.0.downsample.1': [1, 1024, 65, 65],
            'resnet.layer3.1.conv1': [1, 256, 65, 65],
            'resnet.layer3.1.bn1': [1, 256, 65, 65],
            'resnet.layer3.1.conv2': [1, 256, 65, 65],
            'resnet.layer3.1.bn2': [1, 256, 65, 65],
            'resnet.layer3.1.conv3': [1, 1024, 65, 65],
            'resnet.layer3.1.bn3': [1, 1024, 65, 65],
            'resnet.layer3.1.relu': [1, 1024, 65, 65],
            'resnet.layer3.2.conv1': [1, 256, 65, 65],
            'resnet.layer3.2.bn1': [1, 256, 65, 65],
            'resnet.layer3.2.conv2': [1, 256, 65, 65],
            'resnet.layer3.2.bn2': [1, 256, 65, 65],
            'resnet.layer3.2.conv3': [1, 1024, 65, 65],
            'resnet.layer3.2.bn3': [1, 1024, 65, 65],
            'resnet.layer3.2.relu': [1, 1024, 65, 65],
            'resnet.layer3.3.conv1': [1, 256, 65, 65],
            'resnet.layer3.3.bn1': [1, 256, 65, 65],
            'resnet.layer3.3.conv2': [1, 256, 65, 65],
            'resnet.layer3.3.bn2': [1, 256, 65, 65],
            'resnet.layer3.3.conv3': [1, 1024, 65, 65],
            'resnet.layer3.3.bn3': [1, 1024, 65, 65],
            'resnet.layer3.3.relu': [1, 1024, 65, 65],
            'resnet.layer3.4.conv1': [1, 256, 65, 65],
            'resnet.layer3.4.bn1': [1, 256, 65, 65],
            'resnet.layer3.4.conv2': [1, 256, 65, 65],
            'resnet.layer3.4.bn2': [1, 256, 65, 65],
            'resnet.layer3.4.conv3': [1, 1024, 65, 65],
            'resnet.layer3.4.bn3': [1, 1024, 65, 65],
            'resnet.layer3.4.relu': [1, 1024, 65, 65],
            'resnet.layer3.5.conv1': [1, 256, 65, 65],
            'resnet.layer3.5.bn1': [1, 256, 65, 65],
            'resnet.layer3.5.conv2': [1, 256, 65, 65],
            'resnet.layer3.5.bn2': [1, 256, 65, 65],
            'resnet.layer3.5.conv3': [1, 1024, 65, 65],
            'resnet.layer3.5.bn3': [1, 1024, 65, 65],
            'resnet.layer3.5.relu': [1, 1024, 65, 65],
            'resnet.layer3.6.conv1': [1, 256, 65, 65],
            'resnet.layer3.6.bn1': [1, 256, 65, 65],
            'resnet.layer3.6.conv2': [1, 256, 65, 65],
            'resnet.layer3.6.bn2': [1, 256, 65, 65],
            'resnet.layer3.6.conv3': [1, 1024, 65, 65],
            'resnet.layer3.6.bn3': [1, 1024, 65, 65],
            'resnet.layer3.6.relu': [1, 1024, 65, 65],
            'resnet.layer3.7.conv1': [1, 256, 65, 65],
            'resnet.layer3.7.bn1': [1, 256, 65, 65],
            'resnet.layer3.7.conv2': [1, 256, 65, 65],
            'resnet.layer3.7.bn2': [1, 256, 65, 65],
            'resnet.layer3.7.conv3': [1, 1024, 65, 65],
            'resnet.layer3.7.bn3': [1, 1024, 65, 65],
            'resnet.layer3.7.relu': [1, 1024, 65, 65],
            'resnet.layer3.8.conv1': [1, 256, 65, 65],
            'resnet.layer3.8.bn1': [1, 256, 65, 65],
            'resnet.layer3.8.conv2': [1, 256, 65, 65],
            'resnet.layer3.8.bn2': [1, 256, 65, 65],
            'resnet.layer3.8.conv3': [1, 1024, 65, 65],
            'resnet.layer3.8.bn3': [1, 1024, 65, 65],
            'resnet.layer3.8.relu': [1, 1024, 65, 65],
            'resnet.layer3.9.conv1': [1, 256, 65, 65],
            'resnet.layer3.9.bn1': [1, 256, 65, 65],
            'resnet.layer3.9.conv2': [1, 256, 65, 65],
            'resnet.layer3.9.bn2': [1, 256, 65, 65],
            'resnet.layer3.9.conv3': [1, 1024, 65, 65],
            'resnet.layer3.9.bn3': [1, 1024, 65, 65],
            'resnet.layer3.9.relu': [1, 1024, 65, 65],
            'resnet.layer3.10.conv1': [1, 256, 65, 65],
            'resnet.layer3.10.bn1': [1, 256, 65, 65],
            'resnet.layer3.10.conv2': [1, 256, 65, 65],
            'resnet.layer3.10.bn2': [1, 256, 65, 65],
            'resnet.layer3.10.conv3': [1, 1024, 65, 65],
            'resnet.layer3.10.bn3': [1, 1024, 65, 65],
            'resnet.layer3.10.relu': [1, 1024, 65, 65],
            'resnet.layer3.11.conv1': [1, 256, 65, 65],
            'resnet.layer3.11.bn1': [1, 256, 65, 65],
            'resnet.layer3.11.conv2': [1, 256, 65, 65],
            'resnet.layer3.11.bn2': [1, 256, 65, 65],
            'resnet.layer3.11.conv3': [1, 1024, 65, 65],
            'resnet.layer3.11.bn3': [1, 1024, 65, 65],
            'resnet.layer3.11.relu': [1, 1024, 65, 65],
            'resnet.layer3.12.conv1': [1, 256, 65, 65],
            'resnet.layer3.12.bn1': [1, 256, 65, 65],
            'resnet.layer3.12.conv2': [1, 256, 65, 65],
            'resnet.layer3.12.bn2': [1, 256, 65, 65],
            'resnet.layer3.12.conv3': [1, 1024, 65, 65],
            'resnet.layer3.12.bn3': [1, 1024, 65, 65],
            'resnet.layer3.12.relu': [1, 1024, 65, 65],
            'resnet.layer3.13.conv1': [1, 256, 65, 65],
            'resnet.layer3.13.bn1': [1, 256, 65, 65],
            'resnet.layer3.13.conv2': [1, 256, 65, 65],
            'resnet.layer3.13.bn2': [1, 256, 65, 65],
            'resnet.layer3.13.conv3': [1, 1024, 65, 65],
            'resnet.layer3.13.bn3': [1, 1024, 65, 65],
            'resnet.layer3.13.relu': [1, 1024, 65, 65],
            'resnet.layer3.14.conv1': [1, 256, 65, 65],
            'resnet.layer3.14.bn1': [1, 256, 65, 65],
            'resnet.layer3.14.conv2': [1, 256, 65, 65],
            'resnet.layer3.14.bn2': [1, 256, 65, 65],
            'resnet.layer3.14.conv3': [1, 1024, 65, 65],
            'resnet.layer3.14.bn3': [1, 1024, 65, 65],
            'resnet.layer3.14.relu': [1, 1024, 65, 65],
            'resnet.layer3.15.conv1': [1, 256, 65, 65],
            'resnet.layer3.15.bn1': [1, 256, 65, 65],
            'resnet.layer3.15.conv2': [1, 256, 65, 65],
            'resnet.layer3.15.bn2': [1, 256, 65, 65],
            'resnet.layer3.15.conv3': [1, 1024, 65, 65],
            'resnet.layer3.15.bn3': [1, 1024, 65, 65],
            'resnet.layer3.15.relu': [1, 1024, 65, 65],
            'resnet.layer3.16.conv1': [1, 256, 65, 65],
            'resnet.layer3.16.bn1': [1, 256, 65, 65],
            'resnet.layer3.16.conv2': [1, 256, 65, 65],
            'resnet.layer3.16.bn2': [1, 256, 65, 65],
            'resnet.layer3.16.conv3': [1, 1024, 65, 65],
            'resnet.layer3.16.bn3': [1, 1024, 65, 65],
            'resnet.layer3.16.relu': [1, 1024, 65, 65],
            'resnet.layer3.17.conv1': [1, 256, 65, 65],
            'resnet.layer3.17.bn1': [1, 256, 65, 65],
            'resnet.layer3.17.conv2': [1, 256, 65, 65],
            'resnet.layer3.17.bn2': [1, 256, 65, 65],
            'resnet.layer3.17.conv3': [1, 1024, 65, 65],
            'resnet.layer3.17.bn3': [1, 1024, 65, 65],
            'resnet.layer3.17.relu': [1, 1024, 65, 65],
            'resnet.layer3.18.conv1': [1, 256, 65, 65],
            'resnet.layer3.18.bn1': [1, 256, 65, 65],
            'resnet.layer3.18.conv2': [1, 256, 65, 65],
            'resnet.layer3.18.bn2': [1, 256, 65, 65],
            'resnet.layer3.18.conv3': [1, 1024, 65, 65],
            'resnet.layer3.18.bn3': [1, 1024, 65, 65],
            'resnet.layer3.18.relu': [1, 1024, 65, 65],
            'resnet.layer3.19.conv1': [1, 256, 65, 65],
            'resnet.layer3.19.bn1': [1, 256, 65, 65],
            'resnet.layer3.19.conv2': [1, 256, 65, 65],
            'resnet.layer3.19.bn2': [1, 256, 65, 65],
            'resnet.layer3.19.conv3': [1, 1024, 65, 65],
            'resnet.layer3.19.bn3': [1, 1024, 65, 65],
            'resnet.layer3.19.relu': [1, 1024, 65, 65],
            'resnet.layer3.20.conv1': [1, 256, 65, 65],
            'resnet.layer3.20.bn1': [1, 256, 65, 65],
            'resnet.layer3.20.conv2': [1, 256, 65, 65],
            'resnet.layer3.20.bn2': [1, 256, 65, 65],
            'resnet.layer3.20.conv3': [1, 1024, 65, 65],
            'resnet.layer3.20.bn3': [1, 1024, 65, 65],
            'resnet.layer3.20.relu': [1, 1024, 65, 65],
            'resnet.layer3.21.conv1': [1, 256, 65, 65],
            'resnet.layer3.21.bn1': [1, 256, 65, 65],
            'resnet.layer3.21.conv2': [1, 256, 65, 65],
            'resnet.layer3.21.bn2': [1, 256, 65, 65],
            'resnet.layer3.21.conv3': [1, 1024, 65, 65],
            'resnet.layer3.21.bn3': [1, 1024, 65, 65],
            'resnet.layer3.21.relu': [1, 1024, 65, 65],
            'resnet.layer3.22.conv1': [1, 256, 65, 65],
            'resnet.layer3.22.bn1': [1, 256, 65, 65],
            'resnet.layer3.22.conv2': [1, 256, 65, 65],
            'resnet.layer3.22.bn2': [1, 256, 65, 65],
            'resnet.layer3.22.conv3': [1, 1024, 65, 65],
            'resnet.layer3.22.bn3': [1, 1024, 65, 65],
            'resnet.layer3.22.relu': [1, 1024, 65, 65],
            'resnet.layer4.0.conv1': [1, 512, 65, 65],
            'resnet.layer4.0.bn1': [1, 512, 65, 65],
            'resnet.layer4.0.conv2': [1, 512, 65, 65],
            'resnet.layer4.0.bn2': [1, 512, 65, 65],
            'resnet.layer4.0.conv3': [1, 2048, 65, 65],
            'resnet.layer4.0.bn3': [1, 2048, 65, 65],
            'resnet.layer4.0.relu': [1, 2048, 65, 65],
            'resnet.layer4.0.downsample.0': [1, 2048, 65, 65],
            'resnet.layer4.0.downsample.1': [1, 2048, 65, 65],
            'resnet.layer4.1.conv1': [1, 512, 65, 65],
            'resnet.layer4.1.bn1': [1, 512, 65, 65],
            'resnet.layer4.1.conv2': [1, 512, 65, 65],
            'resnet.layer4.1.bn2': [1, 512, 65, 65],
            'resnet.layer4.1.conv3': [1, 2048, 65, 65],
            'resnet.layer4.1.bn3': [1, 2048, 65, 65],
            'resnet.layer4.1.relu': [1, 2048, 65, 65],
            'resnet.layer4.2.conv1': [1, 512, 65, 65],
            'resnet.layer4.2.bn1': [1, 512, 65, 65],
            'resnet.layer4.2.conv2': [1, 512, 65, 65],
            'resnet.layer4.2.bn2': [1, 512, 65, 65],
            'resnet.layer4.2.conv3': [1, 2048, 65, 65],
            'resnet.layer4.2.bn3': [1, 2048, 65, 65],
            'resnet.layer4.2.relu': [1, 2048, 65, 65],
            'OUTPUT': [1, 2048, 65, 65]}

        self.orders = {
            'resnet.conv1': ['INPUT', 'resnet.bn1'],
            'resnet.bn1': ['resnet.conv1', 'resnet.relu'],
            'resnet.relu': ['resnet.bn1', 'resnet.maxpool'],
            'resnet.maxpool': ['resnet.relu', ['resnet.layer1.0.conv1', 'resnet.layer1.0.downsample.0']],

            'resnet.layer1.0.conv1': ['resnet.maxpool', 'resnet.layer1.0.bn1'],
            'resnet.layer1.0.bn1': ['resnet.layer1.0.conv1', 'resnet.layer1.0.conv2'],
            'resnet.layer1.0.conv2': ['resnet.layer1.0.bn1', 'resnet.layer1.0.bn2'],
            'resnet.layer1.0.bn2': ['resnet.layer1.0.conv2', 'resnet.layer1.0.conv3'],
            'resnet.layer1.0.conv3': ['resnet.layer1.0.bn2', 'resnet.layer1.0.bn3'],
            'resnet.layer1.0.bn3': ['resnet.layer1.0.conv3', 'resnet.layer1.0.relu'],
            'resnet.layer1.0.downsample.0': ['resnet.maxpool', 'resnet.layer1.0.downsample.1'],
            'resnet.layer1.0.downsample.1': ['resnet.layer1.0.downsample.0', 'resnet.layer1.0.relu'],
            'resnet.layer1.0.relu': [['resnet.layer1.0.bn3', 'resnet.layer1.0.downsample.1'],
                                     'resnet.layer1.1.conv1'],

            'resnet.layer1.1.conv1': ['resnet.layer1.0.relu', 'resnet.layer1.1.bn1'],
            'resnet.layer1.1.bn1': ['resnet.layer1.1.conv1', 'resnet.layer1.1.conv2'],
            'resnet.layer1.1.conv2': ['resnet.layer1.1.bn1', 'resnet.layer1.1.bn2'],
            'resnet.layer1.1.bn2': ['resnet.layer1.1.conv2', 'resnet.layer1.1.conv3'],
            'resnet.layer1.1.conv3': ['resnet.layer1.1.bn2', 'resnet.layer1.1.bn3'],
            'resnet.layer1.1.bn3': ['resnet.layer1.1.conv3', 'resnet.layer1.1.relu'],
            'resnet.layer1.1.relu': ['resnet.layer1.1.bn3', 'resnet.layer1.2.conv1'],
            'resnet.layer1.2.conv1': ['resnet.layer1.1.relu', 'resnet.layer1.2.bn1'],
            'resnet.layer1.2.bn1': ['resnet.layer1.2.conv1', 'resnet.layer1.2.conv2'],
            'resnet.layer1.2.conv2': ['resnet.layer1.2.bn1', 'resnet.layer1.2.bn2'],
            'resnet.layer1.2.bn2': ['resnet.layer1.2.conv2', 'resnet.layer1.2.conv3'],
            'resnet.layer1.2.conv3': ['resnet.layer1.2.bn2', 'resnet.layer1.2.bn3'],
            'resnet.layer1.2.bn3': ['resnet.layer1.2.conv3', 'resnet.layer1.2.relu'],
            'resnet.layer1.2.relu': ['resnet.layer1.2.bn3', ['resnet.layer2.0.conv1', 'resnet.layer2.0.downsample.0']],

            'resnet.layer2.0.conv1': ['resnet.layer1.2.relu', 'resnet.layer2.0.bn1'],
            'resnet.layer2.0.bn1': ['resnet.layer2.0.conv1', 'resnet.layer2.0.conv2'],
            'resnet.layer2.0.conv2': ['resnet.layer2.0.bn1', 'resnet.layer2.0.bn2'],
            'resnet.layer2.0.bn2': ['resnet.layer2.0.conv2', 'resnet.layer2.0.conv3'],
            'resnet.layer2.0.conv3': ['resnet.layer2.0.bn2', 'resnet.layer2.0.bn3'],
            'resnet.layer2.0.bn3': ['resnet.layer2.0.conv3', 'resnet.layer2.0.relu'],
            'resnet.layer2.0.downsample.0': ['resnet.layer1.2.relu', 'resnet.layer2.0.downsample.1'],
            'resnet.layer2.0.downsample.1': ['resnet.layer2.0.downsample.0', 'resnet.layer2.0.relu'],
            'resnet.layer2.0.relu': [['resnet.layer2.0.bn3', 'resnet.layer2.0.downsample.1'],
                                     'resnet.layer2.1.conv1'],
            'resnet.layer2.1.conv1': ['resnet.layer2.0.relu', 'resnet.layer2.1.bn1'],
            'resnet.layer2.1.bn1': ['resnet.layer2.1.conv1', 'resnet.layer2.1.conv2'],
            'resnet.layer2.1.conv2': ['resnet.layer2.1.bn1', 'resnet.layer2.1.bn2'],
            'resnet.layer2.1.bn2': ['resnet.layer2.1.conv2', 'resnet.layer2.1.conv3'],
            'resnet.layer2.1.conv3': ['resnet.layer2.1.bn2', 'resnet.layer2.1.bn3'],
            'resnet.layer2.1.bn3': ['resnet.layer2.1.conv3', 'resnet.layer2.1.relu'],
            'resnet.layer2.1.relu': ['resnet.layer2.1.bn3', 'resnet.layer2.2.conv1'],
            'resnet.layer2.2.conv1': ['resnet.layer2.1.relu', 'resnet.layer2.2.bn1'],
            'resnet.layer2.2.bn1': ['resnet.layer2.2.conv1', 'resnet.layer2.2.conv2'],
            'resnet.layer2.2.conv2': ['resnet.layer2.2.bn1', 'resnet.layer2.2.bn2'],
            'resnet.layer2.2.bn2': ['resnet.layer2.2.conv2', 'resnet.layer2.2.conv3'],
            'resnet.layer2.2.conv3': ['resnet.layer2.2.bn2', 'resnet.layer2.2.bn3'],
            'resnet.layer2.2.bn3': ['resnet.layer2.2.conv3', 'resnet.layer2.2.relu'],
            'resnet.layer2.2.relu': ['resnet.layer2.2.bn3', 'resnet.layer2.3.conv1'],
            'resnet.layer2.3.conv1': ['resnet.layer2.2.relu', 'resnet.layer2.3.bn1'],
            'resnet.layer2.3.bn1': ['resnet.layer2.3.conv1', 'resnet.layer2.3.conv2'],
            'resnet.layer2.3.conv2': ['resnet.layer2.3.bn1', 'resnet.layer2.3.bn2'],
            'resnet.layer2.3.bn2': ['resnet.layer2.3.conv2', 'resnet.layer2.3.conv3'],
            'resnet.layer2.3.conv3': ['resnet.layer2.3.bn2', 'resnet.layer2.3.bn3'],
            'resnet.layer2.3.bn3': ['resnet.layer2.3.conv3', 'resnet.layer2.3.relu'],
            'resnet.layer2.3.relu': ['resnet.layer2.3.bn3', ['resnet.layer3.0.conv1', 'resnet.layer3.0.downsample.0']],

            'resnet.layer3.0.conv1': ['resnet.layer2.3.relu', 'resnet.layer3.0.bn1'],
            'resnet.layer3.0.bn1': ['resnet.layer3.0.conv1', 'resnet.layer3.0.conv2'],
            'resnet.layer3.0.conv2': ['resnet.layer3.0.bn1', 'resnet.layer3.0.bn2'],
            'resnet.layer3.0.bn2': ['resnet.layer3.0.conv2', 'resnet.layer3.0.conv3'],
            'resnet.layer3.0.conv3': ['resnet.layer3.0.bn2', 'resnet.layer3.0.bn3'],
            'resnet.layer3.0.bn3': ['resnet.layer3.0.conv3', 'resnet.layer3.0.relu'],
            'resnet.layer3.0.downsample.0': ['resnet.layer2.3.relu', 'resnet.layer3.0.downsample.1'],
            'resnet.layer3.0.downsample.1': ['resnet.layer3.0.downsample.0', 'resnet.layer3.0.relu'],
            'resnet.layer3.0.relu': [['resnet.layer3.0.bn3', 'resnet.layer3.0.downsample.1'],
                                     'resnet.layer3.1.conv1'],
            'resnet.layer3.1.conv1': ['resnet.layer3.0.relu', 'resnet.layer3.1.bn1'],
            'resnet.layer3.1.bn1': ['resnet.layer3.1.conv1', 'resnet.layer3.1.conv2'],
            'resnet.layer3.1.conv2': ['resnet.layer3.1.bn1', 'resnet.layer3.1.bn2'],
            'resnet.layer3.1.bn2': ['resnet.layer3.1.conv2', 'resnet.layer3.1.conv3'],
            'resnet.layer3.1.conv3': ['resnet.layer3.1.bn2', 'resnet.layer3.1.bn3'],
            'resnet.layer3.1.bn3': ['resnet.layer3.1.conv3', 'resnet.layer3.1.relu'],
            'resnet.layer3.1.relu': ['resnet.layer3.1.bn3', 'resnet.layer3.2.conv1'],
            'resnet.layer3.2.conv1': ['resnet.layer3.1.relu', 'resnet.layer3.2.bn1'],
            'resnet.layer3.2.bn1': ['resnet.layer3.2.conv1', 'resnet.layer3.2.conv2'],
            'resnet.layer3.2.conv2': ['resnet.layer3.2.bn1', 'resnet.layer3.2.bn2'],
            'resnet.layer3.2.bn2': ['resnet.layer3.2.conv2', 'resnet.layer3.2.conv3'],
            'resnet.layer3.2.conv3': ['resnet.layer3.2.bn2', 'resnet.layer3.2.bn3'],
            'resnet.layer3.2.bn3': ['resnet.layer3.2.conv3', 'resnet.layer3.2.relu'],
            'resnet.layer3.2.relu': ['resnet.layer3.2.bn3', 'resnet.layer3.3.conv1'],
            'resnet.layer3.3.conv1': ['resnet.layer3.2.relu', 'resnet.layer3.3.bn1'],
            'resnet.layer3.3.bn1': ['resnet.layer3.3.conv1', 'resnet.layer3.3.conv2'],
            'resnet.layer3.3.conv2': ['resnet.layer3.3.bn1', 'resnet.layer3.3.bn2'],
            'resnet.layer3.3.bn2': ['resnet.layer3.3.conv2', 'resnet.layer3.3.conv3'],
            'resnet.layer3.3.conv3': ['resnet.layer3.3.bn2', 'resnet.layer3.3.bn3'],
            'resnet.layer3.3.bn3': ['resnet.layer3.3.conv3', 'resnet.layer3.3.relu'],
            'resnet.layer3.3.relu': ['resnet.layer3.3.bn3', 'resnet.layer3.4.conv1'],
            'resnet.layer3.4.conv1': ['resnet.layer3.3.relu', 'resnet.layer3.4.bn1'],
            'resnet.layer3.4.bn1': ['resnet.layer3.4.conv1', 'resnet.layer3.4.conv2'],
            'resnet.layer3.4.conv2': ['resnet.layer3.4.bn1', 'resnet.layer3.4.bn2'],
            'resnet.layer3.4.bn2': ['resnet.layer3.4.conv2', 'resnet.layer3.4.conv3'],
            'resnet.layer3.4.conv3': ['resnet.layer3.4.bn2', 'resnet.layer3.4.bn3'],
            'resnet.layer3.4.bn3': ['resnet.layer3.4.conv3', 'resnet.layer3.4.relu'],
            'resnet.layer3.4.relu': ['resnet.layer3.4.bn3', 'resnet.layer3.5.conv1'],
            'resnet.layer3.5.conv1': ['resnet.layer3.4.relu', 'resnet.layer3.5.bn1'],
            'resnet.layer3.5.bn1': ['resnet.layer3.5.conv1', 'resnet.layer3.5.conv2'],
            'resnet.layer3.5.conv2': ['resnet.layer3.5.bn1', 'resnet.layer3.5.bn2'],
            'resnet.layer3.5.bn2': ['resnet.layer3.5.conv2', 'resnet.layer3.5.conv3'],
            'resnet.layer3.5.conv3': ['resnet.layer3.5.bn2', 'resnet.layer3.5.bn3'],
            'resnet.layer3.5.bn3': ['resnet.layer3.5.conv3', 'resnet.layer3.5.relu'],
            'resnet.layer3.5.relu': ['resnet.layer3.5.bn3', 'resnet.layer3.6.conv1'],
            'resnet.layer3.6.conv1': ['resnet.layer3.5.relu', 'resnet.layer3.6.bn1'],
            'resnet.layer3.6.bn1': ['resnet.layer3.6.conv1', 'resnet.layer3.6.conv2'],
            'resnet.layer3.6.conv2': ['resnet.layer3.6.bn1', 'resnet.layer3.6.bn2'],
            'resnet.layer3.6.bn2': ['resnet.layer3.6.conv2', 'resnet.layer3.6.conv3'],
            'resnet.layer3.6.conv3': ['resnet.layer3.6.bn2', 'resnet.layer3.6.bn3'],
            'resnet.layer3.6.bn3': ['resnet.layer3.6.conv3', 'resnet.layer3.6.relu'],
            'resnet.layer3.6.relu': ['resnet.layer3.6.bn3', 'resnet.layer3.7.conv1'],
            'resnet.layer3.7.conv1': ['resnet.layer3.6.relu', 'resnet.layer3.7.bn1'],
            'resnet.layer3.7.bn1': ['resnet.layer3.7.conv1', 'resnet.layer3.7.conv2'],
            'resnet.layer3.7.conv2': ['resnet.layer3.7.bn1', 'resnet.layer3.7.bn2'],
            'resnet.layer3.7.bn2': ['resnet.layer3.7.conv2', 'resnet.layer3.7.conv3'],
            'resnet.layer3.7.conv3': ['resnet.layer3.7.bn2', 'resnet.layer3.7.bn3'],
            'resnet.layer3.7.bn3': ['resnet.layer3.7.conv3', 'resnet.layer3.7.relu'],
            'resnet.layer3.7.relu': ['resnet.layer3.7.bn3', 'resnet.layer3.8.conv1'],
            'resnet.layer3.8.conv1': ['resnet.layer3.7.relu', 'resnet.layer3.8.bn1'],
            'resnet.layer3.8.bn1': ['resnet.layer3.8.conv1', 'resnet.layer3.8.conv2'],
            'resnet.layer3.8.conv2': ['resnet.layer3.8.bn1', 'resnet.layer3.8.bn2'],
            'resnet.layer3.8.bn2': ['resnet.layer3.8.conv2', 'resnet.layer3.8.conv3'],
            'resnet.layer3.8.conv3': ['resnet.layer3.8.bn2', 'resnet.layer3.8.bn3'],
            'resnet.layer3.8.bn3': ['resnet.layer3.8.conv3', 'resnet.layer3.8.relu'],
            'resnet.layer3.8.relu': ['resnet.layer3.8.bn3', 'resnet.layer3.9.conv1'],
            'resnet.layer3.9.conv1': ['resnet.layer3.8.relu', 'resnet.layer3.9.bn1'],
            'resnet.layer3.9.bn1': ['resnet.layer3.9.conv1', 'resnet.layer3.9.conv2'],
            'resnet.layer3.9.conv2': ['resnet.layer3.9.bn1', 'resnet.layer3.9.bn2'],
            'resnet.layer3.9.bn2': ['resnet.layer3.9.conv2', 'resnet.layer3.9.conv3'],
            'resnet.layer3.9.conv3': ['resnet.layer3.9.bn2', 'resnet.layer3.9.bn3'],
            'resnet.layer3.9.bn3': ['resnet.layer3.9.conv3', 'resnet.layer3.9.relu'],
            'resnet.layer3.9.relu': ['resnet.layer3.9.bn3', 'resnet.layer3.10.conv1'],
            'resnet.layer3.10.conv1': ['resnet.layer3.9.relu', 'resnet.layer3.10.bn1'],
            'resnet.layer3.10.bn1': ['resnet.layer3.10.conv1', 'resnet.layer3.10.conv2'],
            'resnet.layer3.10.conv2': ['resnet.layer3.10.bn1', 'resnet.layer3.10.bn2'],
            'resnet.layer3.10.bn2': ['resnet.layer3.10.conv2', 'resnet.layer3.10.conv3'],
            'resnet.layer3.10.conv3': ['resnet.layer3.10.bn2', 'resnet.layer3.10.bn3'],
            'resnet.layer3.10.bn3': ['resnet.layer3.10.conv3', 'resnet.layer3.10.relu'],
            'resnet.layer3.10.relu': ['resnet.layer3.10.bn3', 'resnet.layer3.11.conv1'],
            'resnet.layer3.11.conv1': ['resnet.layer3.10.relu', 'resnet.layer3.11.bn1'],
            'resnet.layer3.11.bn1': ['resnet.layer3.11.conv1', 'resnet.layer3.11.conv2'],
            'resnet.layer3.11.conv2': ['resnet.layer3.11.bn1', 'resnet.layer3.11.bn2'],
            'resnet.layer3.11.bn2': ['resnet.layer3.11.conv2', 'resnet.layer3.11.conv3'],
            'resnet.layer3.11.conv3': ['resnet.layer3.11.bn2', 'resnet.layer3.11.bn3'],
            'resnet.layer3.11.bn3': ['resnet.layer3.11.conv3', 'resnet.layer3.11.relu'],
            'resnet.layer3.11.relu': ['resnet.layer3.11.bn3', 'resnet.layer3.12.conv1'],
            'resnet.layer3.12.conv1': ['resnet.layer3.11.relu', 'resnet.layer3.12.bn1'],
            'resnet.layer3.12.bn1': ['resnet.layer3.12.conv1', 'resnet.layer3.12.conv2'],
            'resnet.layer3.12.conv2': ['resnet.layer3.12.bn1', 'resnet.layer3.12.bn2'],
            'resnet.layer3.12.bn2': ['resnet.layer3.12.conv2', 'resnet.layer3.12.conv3'],
            'resnet.layer3.12.conv3': ['resnet.layer3.12.bn2', 'resnet.layer3.12.bn3'],
            'resnet.layer3.12.bn3': ['resnet.layer3.12.conv3', 'resnet.layer3.12.relu'],
            'resnet.layer3.12.relu': ['resnet.layer3.12.bn3', 'resnet.layer3.13.conv1'],
            'resnet.layer3.13.conv1': ['resnet.layer3.12.relu', 'resnet.layer3.13.bn1'],
            'resnet.layer3.13.bn1': ['resnet.layer3.13.conv1', 'resnet.layer3.13.conv2'],
            'resnet.layer3.13.conv2': ['resnet.layer3.13.bn1', 'resnet.layer3.13.bn2'],
            'resnet.layer3.13.bn2': ['resnet.layer3.13.conv2', 'resnet.layer3.13.conv3'],
            'resnet.layer3.13.conv3': ['resnet.layer3.13.bn2', 'resnet.layer3.13.bn3'],
            'resnet.layer3.13.bn3': ['resnet.layer3.13.conv3', 'resnet.layer3.13.relu'],
            'resnet.layer3.13.relu': ['resnet.layer3.13.bn3', 'resnet.layer3.14.conv1'],
            'resnet.layer3.14.conv1': ['resnet.layer3.13.relu', 'resnet.layer3.14.bn1'],
            'resnet.layer3.14.bn1': ['resnet.layer3.14.conv1', 'resnet.layer3.14.conv2'],
            'resnet.layer3.14.conv2': ['resnet.layer3.14.bn1', 'resnet.layer3.14.bn2'],
            'resnet.layer3.14.bn2': ['resnet.layer3.14.conv2', 'resnet.layer3.14.conv3'],
            'resnet.layer3.14.conv3': ['resnet.layer3.14.bn2', 'resnet.layer3.14.bn3'],
            'resnet.layer3.14.bn3': ['resnet.layer3.14.conv3', 'resnet.layer3.14.relu'],
            'resnet.layer3.14.relu': ['resnet.layer3.14.bn3', 'resnet.layer3.15.conv1'],
            'resnet.layer3.15.conv1': ['resnet.layer3.14.relu', 'resnet.layer3.15.bn1'],
            'resnet.layer3.15.bn1': ['resnet.layer3.15.conv1', 'resnet.layer3.15.conv2'],
            'resnet.layer3.15.conv2': ['resnet.layer3.15.bn1', 'resnet.layer3.15.bn2'],
            'resnet.layer3.15.bn2': ['resnet.layer3.15.conv2', 'resnet.layer3.15.conv3'],
            'resnet.layer3.15.conv3': ['resnet.layer3.15.bn2', 'resnet.layer3.15.bn3'],
            'resnet.layer3.15.bn3': ['resnet.layer3.15.conv3', 'resnet.layer3.15.relu'],
            'resnet.layer3.15.relu': ['resnet.layer3.15.bn3', 'resnet.layer3.16.conv1'],
            'resnet.layer3.16.conv1': ['resnet.layer3.15.relu', 'resnet.layer3.16.bn1'],
            'resnet.layer3.16.bn1': ['resnet.layer3.16.conv1', 'resnet.layer3.16.conv2'],
            'resnet.layer3.16.conv2': ['resnet.layer3.16.bn1', 'resnet.layer3.16.bn2'],
            'resnet.layer3.16.bn2': ['resnet.layer3.16.conv2', 'resnet.layer3.16.conv3'],
            'resnet.layer3.16.conv3': ['resnet.layer3.16.bn2', 'resnet.layer3.16.bn3'],
            'resnet.layer3.16.bn3': ['resnet.layer3.16.conv3', 'resnet.layer3.16.relu'],
            'resnet.layer3.16.relu': ['resnet.layer3.16.bn3', 'resnet.layer3.17.conv1'],
            'resnet.layer3.17.conv1': ['resnet.layer3.16.relu', 'resnet.layer3.17.bn1'],
            'resnet.layer3.17.bn1': ['resnet.layer3.17.conv1', 'resnet.layer3.17.conv2'],
            'resnet.layer3.17.conv2': ['resnet.layer3.17.bn1', 'resnet.layer3.17.bn2'],
            'resnet.layer3.17.bn2': ['resnet.layer3.17.conv2', 'resnet.layer3.17.conv3'],
            'resnet.layer3.17.conv3': ['resnet.layer3.17.bn2', 'resnet.layer3.17.bn3'],
            'resnet.layer3.17.bn3': ['resnet.layer3.17.conv3', 'resnet.layer3.17.relu'],
            'resnet.layer3.17.relu': ['resnet.layer3.17.bn3', 'resnet.layer3.18.conv1'],
            'resnet.layer3.18.conv1': ['resnet.layer3.17.relu', 'resnet.layer3.18.bn1'],
            'resnet.layer3.18.bn1': ['resnet.layer3.18.conv1', 'resnet.layer3.18.conv2'],
            'resnet.layer3.18.conv2': ['resnet.layer3.18.bn1', 'resnet.layer3.18.bn2'],
            'resnet.layer3.18.bn2': ['resnet.layer3.18.conv2', 'resnet.layer3.18.conv3'],
            'resnet.layer3.18.conv3': ['resnet.layer3.18.bn2', 'resnet.layer3.18.bn3'],
            'resnet.layer3.18.bn3': ['resnet.layer3.18.conv3', 'resnet.layer3.18.relu'],
            'resnet.layer3.18.relu': ['resnet.layer3.18.bn3', 'resnet.layer3.19.conv1'],
            'resnet.layer3.19.conv1': ['resnet.layer3.18.relu', 'resnet.layer3.19.bn1'],
            'resnet.layer3.19.bn1': ['resnet.layer3.19.conv1', 'resnet.layer3.19.conv2'],
            'resnet.layer3.19.conv2': ['resnet.layer3.19.bn1', 'resnet.layer3.19.bn2'],
            'resnet.layer3.19.bn2': ['resnet.layer3.19.conv2', 'resnet.layer3.19.conv3'],
            'resnet.layer3.19.conv3': ['resnet.layer3.19.bn2', 'resnet.layer3.19.bn3'],
            'resnet.layer3.19.bn3': ['resnet.layer3.19.conv3', 'resnet.layer3.19.relu'],
            'resnet.layer3.19.relu': ['resnet.layer3.19.bn3', 'resnet.layer3.20.conv1'],
            'resnet.layer3.20.conv1': ['resnet.layer3.19.relu', 'resnet.layer3.20.bn1'],
            'resnet.layer3.20.bn1': ['resnet.layer3.20.conv1', 'resnet.layer3.20.conv2'],
            'resnet.layer3.20.conv2': ['resnet.layer3.20.bn1', 'resnet.layer3.20.bn2'],
            'resnet.layer3.20.bn2': ['resnet.layer3.20.conv2', 'resnet.layer3.20.conv3'],
            'resnet.layer3.20.conv3': ['resnet.layer3.20.bn2', 'resnet.layer3.20.bn3'],
            'resnet.layer3.20.bn3': ['resnet.layer3.20.conv3', 'resnet.layer3.20.relu'],
            'resnet.layer3.20.relu': ['resnet.layer3.20.bn3', 'resnet.layer3.21.conv1'],
            'resnet.layer3.21.conv1': ['resnet.layer3.20.relu', 'resnet.layer3.21.bn1'],
            'resnet.layer3.21.bn1': ['resnet.layer3.21.conv1', 'resnet.layer3.21.conv2'],
            'resnet.layer3.21.conv2': ['resnet.layer3.21.bn1', 'resnet.layer3.21.bn2'],
            'resnet.layer3.21.bn2': ['resnet.layer3.21.conv2', 'resnet.layer3.21.conv3'],
            'resnet.layer3.21.conv3': ['resnet.layer3.21.bn2', 'resnet.layer3.21.bn3'],
            'resnet.layer3.21.bn3': ['resnet.layer3.21.conv3', 'resnet.layer3.21.relu'],
            'resnet.layer3.21.relu': ['resnet.layer3.21.bn3', 'resnet.layer3.22.conv1'],
            'resnet.layer3.22.conv1': ['resnet.layer3.21.relu', 'resnet.layer3.22.bn1'],
            'resnet.layer3.22.bn1': ['resnet.layer3.22.conv1', 'resnet.layer3.22.conv2'],
            'resnet.layer3.22.conv2': ['resnet.layer3.22.bn1', 'resnet.layer3.22.bn2'],
            'resnet.layer3.22.bn2': ['resnet.layer3.22.conv2', 'resnet.layer3.22.conv3'],
            'resnet.layer3.22.conv3': ['resnet.layer3.22.bn2', 'resnet.layer3.22.bn3'],
            'resnet.layer3.22.bn3': ['resnet.layer3.22.conv3', 'resnet.layer3.22.relu'],
            'resnet.layer3.22.relu': ['resnet.layer3.22.bn3',
                                      ['resnet.layer4.0.conv1', 'resnet.layer4.0.downsample.0']],

            'resnet.layer4.0.conv1': ['resnet.layer3.22.relu', 'resnet.layer4.0.bn1'],
            'resnet.layer4.0.bn1': ['resnet.layer4.0.conv1', 'resnet.layer4.0.conv2'],
            'resnet.layer4.0.conv2': ['resnet.layer4.0.bn1', 'resnet.layer4.0.bn2'],
            'resnet.layer4.0.bn2': ['resnet.layer4.0.conv2', 'resnet.layer4.0.conv3'],
            'resnet.layer4.0.conv3': ['resnet.layer4.0.bn2', 'resnet.layer4.0.bn3'],
            'resnet.layer4.0.bn3': ['resnet.layer4.0.conv3', 'resnet.layer4.0.relu'],
            'resnet.layer4.0.downsample.0': ['resnet.layer3.22.relu', 'resnet.layer4.0.downsample.1'],
            'resnet.layer4.0.downsample.1': ['resnet.layer4.0.downsample.0', 'resnet.layer4.0.relu'],
            'resnet.layer4.0.relu': [['resnet.layer4.0.bn3', 'resnet.layer4.0.downsample.1'],
                                     'resnet.layer4.1.conv1'],
            'resnet.layer4.1.conv1': ['resnet.layer4.0.relu', 'resnet.layer4.1.bn1'],
            'resnet.layer4.1.bn1': ['resnet.layer4.1.conv1', 'resnet.layer4.1.conv2'],
            'resnet.layer4.1.conv2': ['resnet.layer4.1.bn1', 'resnet.layer4.1.bn2'],
            'resnet.layer4.1.bn2': ['resnet.layer4.1.conv2', 'resnet.layer4.1.conv3'],
            'resnet.layer4.1.conv3': ['resnet.layer4.1.bn2', 'resnet.layer4.1.bn3'],
            'resnet.layer4.1.bn3': ['resnet.layer4.1.conv3', 'resnet.layer4.1.relu'],
            'resnet.layer4.1.relu': ['resnet.layer4.1.bn3', 'resnet.layer4.2.conv1'],
            'resnet.layer4.2.conv1': ['resnet.layer4.1.relu', 'resnet.layer4.2.bn1'],
            'resnet.layer4.2.bn1': ['resnet.layer4.2.conv1', 'resnet.layer4.2.conv2'],
            'resnet.layer4.2.conv2': ['resnet.layer4.2.bn1', 'resnet.layer4.2.bn2'],
            'resnet.layer4.2.bn2': ['resnet.layer4.2.conv2', 'resnet.layer4.2.conv3'],
            'resnet.layer4.2.conv3': ['resnet.layer4.2.bn2', 'resnet.layer4.2.bn3'],
            'resnet.layer4.2.bn3': ['resnet.layer4.2.conv3', 'resnet.layer4.2.relu'],
            'resnet.layer4.2.relu': ['resnet.layer4.2.bn3', 'OUTPUT'],
            # 'aspp': ['resnet.layer4.2.relu', 'aspp.aspp1'],
            # 'aspp.aspp1': ['aspp', 'aspp.aspp1.aspp_conv'],
            # 'aspp.aspp1.aspp_conv': ['aspp.aspp1', 'aspp.aspp1.aspp_conv.0'],
            # 'aspp.aspp1.aspp_conv.0': ['aspp.aspp1.aspp_conv', 'aspp.aspp1.aspp_conv.1'],
            # 'aspp.aspp1.aspp_conv.1': ['aspp.aspp1.aspp_conv.0', 'aspp.aspp1.aspp_conv.2'],
            # 'aspp.aspp1.aspp_conv.2': ['aspp.aspp1.aspp_conv.1', 'aspp.aspp2'],
            # 'aspp.aspp2': ['aspp.aspp1.aspp_conv.2', 'aspp.aspp2.aspp_conv'],
            # 'aspp.aspp2.aspp_conv': ['aspp.aspp2', 'aspp.aspp2.aspp_conv.0'],
            # 'aspp.aspp2.aspp_conv.0': ['aspp.aspp2.aspp_conv', 'aspp.aspp2.aspp_conv.1'],
            # 'aspp.aspp2.aspp_conv.1': ['aspp.aspp2.aspp_conv.0', 'aspp.aspp2.aspp_conv.2'],
            # 'aspp.aspp2.aspp_conv.2': ['aspp.aspp2.aspp_conv.1', 'aspp.aspp3'],
            # 'aspp.aspp3': ['aspp.aspp2.aspp_conv.2', 'aspp.aspp3.aspp_conv'],
            # 'aspp.aspp3.aspp_conv': ['aspp.aspp3', 'aspp.aspp3.aspp_conv.0'],
            # 'aspp.aspp3.aspp_conv.0': ['aspp.aspp3.aspp_conv', 'aspp.aspp3.aspp_conv.1'],
            # 'aspp.aspp3.aspp_conv.1': ['aspp.aspp3.aspp_conv.0', 'aspp.aspp3.aspp_conv.2'],
            # 'aspp.aspp3.aspp_conv.2': ['aspp.aspp3.aspp_conv.1', 'aspp.aspp4'],
            # 'aspp.aspp4': ['aspp.aspp3.aspp_conv.2', 'aspp.aspp4.aspp_conv'],
            # 'aspp.aspp4.aspp_conv': ['aspp.aspp4', 'aspp.aspp4.aspp_conv.0'],
            # 'aspp.aspp4.aspp_conv.0': ['aspp.aspp4.aspp_conv', 'aspp.aspp4.aspp_conv.1'],
            # 'aspp.aspp4.aspp_conv.1': ['aspp.aspp4.aspp_conv.0', 'aspp.aspp4.aspp_conv.2'],
            # 'aspp.aspp4.aspp_conv.2': ['aspp.aspp4.aspp_conv.1', 'aspp.aspp_pooling'],
            # 'aspp.aspp_pooling': ['aspp.aspp4.aspp_conv.2', 'aspp.aspp_pooling.conv'],
            # 'aspp.aspp_pooling.conv': ['aspp.aspp_pooling', 'aspp.aspp_pooling.conv.0'],
            # 'aspp.aspp_pooling.conv.0': ['aspp.aspp_pooling.conv', 'aspp.aspp_pooling.conv.1'],
            # 'aspp.aspp_pooling.conv.1': ['aspp.aspp_pooling.conv.0', 'aspp.aspp_pooling.conv.2'],
            # 'aspp.aspp_pooling.conv.2': ['aspp.aspp_pooling.conv.1', 'aspp.conv1'],
            # 'aspp.conv1': ['aspp.aspp_pooling.conv.2', 'aspp.bn1'], 'aspp.bn1': ['aspp.conv1', 'aspp.relu'],
            # 'aspp.relu': ['aspp.bn1', 'aspp.conv2'], 'aspp.conv2': ['aspp.relu', 'aspp.dro'],
            # 'aspp.dro': ['aspp.conv2', '']
        }

        self.layer_names = {
            "resnet": self.resnet,
            "resnet.conv1": self.resnet.conv1,
            "resnet.bn1": self.resnet.bn1,
            "resnet.relu": self.resnet.relu,
            "resnet.maxpool": self.resnet.maxpool,
            "resnet.layer1": self.resnet.layer1,
            "resnet.layer1.0": self.resnet.layer1[0],
            "resnet.layer1.0.conv1": self.resnet.layer1[0].conv1,
            "resnet.layer1.0.bn1": self.resnet.layer1[0].bn1,
            "resnet.layer1.0.conv2": self.resnet.layer1[0].conv2,
            "resnet.layer1.0.bn2": self.resnet.layer1[0].bn2,
            "resnet.layer1.0.conv3": self.resnet.layer1[0].conv3,
            "resnet.layer1.0.bn3": self.resnet.layer1[0].bn3,
            "resnet.layer1.0.relu": self.resnet.layer1[0].relu,
            "resnet.layer1.0.downsample": self.resnet.layer1[0].downsample,
            "resnet.layer1.0.downsample.0": self.resnet.layer1[0].downsample[0],
            "resnet.layer1.0.downsample.1": self.resnet.layer1[0].downsample[1],
            "resnet.layer1.1": self.resnet.layer1[1],
            "resnet.layer1.1.conv1": self.resnet.layer1[1].conv1,
            "resnet.layer1.1.bn1": self.resnet.layer1[1].bn1,
            "resnet.layer1.1.conv2": self.resnet.layer1[1].conv2,
            "resnet.layer1.1.bn2": self.resnet.layer1[1].bn2,
            "resnet.layer1.1.conv3": self.resnet.layer1[1].conv3,
            "resnet.layer1.1.bn3": self.resnet.layer1[1].bn3,
            "resnet.layer1.1.relu": self.resnet.layer1[1].relu,
            "resnet.layer1.2": self.resnet.layer1[2],
            "resnet.layer1.2.conv1": self.resnet.layer1[2].conv1,
            "resnet.layer1.2.bn1": self.resnet.layer1[2].bn1,
            "resnet.layer1.2.conv2": self.resnet.layer1[2].conv2,
            "resnet.layer1.2.bn2": self.resnet.layer1[2].bn2,
            "resnet.layer1.2.conv3": self.resnet.layer1[2].conv3,
            "resnet.layer1.2.bn3": self.resnet.layer1[2].bn3,
            "resnet.layer1.2.relu": self.resnet.layer1[2].relu,
            "resnet.layer2": self.resnet.layer2,
            "resnet.layer2.0": self.resnet.layer2[0],
            "resnet.layer2.0.conv1": self.resnet.layer2[0].conv1,
            "resnet.layer2.0.bn1": self.resnet.layer2[0].bn1,
            "resnet.layer2.0.conv2": self.resnet.layer2[0].conv2,
            "resnet.layer2.0.bn2": self.resnet.layer2[0].bn2,
            "resnet.layer2.0.conv3": self.resnet.layer2[0].conv3,
            "resnet.layer2.0.bn3": self.resnet.layer2[0].bn3,
            "resnet.layer2.0.relu": self.resnet.layer2[0].relu,
            "resnet.layer2.0.downsample": self.resnet.layer2[0].downsample,
            "resnet.layer2.0.downsample.0": self.resnet.layer2[0].downsample[0],
            "resnet.layer2.0.downsample.1": self.resnet.layer2[0].downsample[1],
            "resnet.layer2.1": self.resnet.layer2[1],
            "resnet.layer2.1.conv1": self.resnet.layer2[1].conv1,
            "resnet.layer2.1.bn1": self.resnet.layer2[1].bn1,
            "resnet.layer2.1.conv2": self.resnet.layer2[1].conv2,
            "resnet.layer2.1.bn2": self.resnet.layer2[1].bn2,
            "resnet.layer2.1.conv3": self.resnet.layer2[1].conv3,
            "resnet.layer2.1.bn3": self.resnet.layer2[1].bn3,
            "resnet.layer2.1.relu": self.resnet.layer2[1].relu,
            "resnet.layer2.2": self.resnet.layer2[2],
            "resnet.layer2.2.conv1": self.resnet.layer2[2].conv1,
            "resnet.layer2.2.bn1": self.resnet.layer2[2].bn1,
            "resnet.layer2.2.conv2": self.resnet.layer2[2].conv2,
            "resnet.layer2.2.bn2": self.resnet.layer2[2].bn2,
            "resnet.layer2.2.conv3": self.resnet.layer2[2].conv3,
            "resnet.layer2.2.bn3": self.resnet.layer2[2].bn3,
            "resnet.layer2.2.relu": self.resnet.layer2[2].relu,
            "resnet.layer2.3": self.resnet.layer2[3],
            "resnet.layer2.3.conv1": self.resnet.layer2[3].conv1,
            "resnet.layer2.3.bn1": self.resnet.layer2[3].bn1,
            "resnet.layer2.3.conv2": self.resnet.layer2[3].conv2,
            "resnet.layer2.3.bn2": self.resnet.layer2[3].bn2,
            "resnet.layer2.3.conv3": self.resnet.layer2[3].conv3,
            "resnet.layer2.3.bn3": self.resnet.layer2[3].bn3,
            "resnet.layer2.3.relu": self.resnet.layer2[3].relu,
            "resnet.layer3": self.resnet.layer3,
            "resnet.layer3.0": self.resnet.layer3[0],
            "resnet.layer3.0.conv1": self.resnet.layer3[0].conv1,
            "resnet.layer3.0.bn1": self.resnet.layer3[0].bn1,
            "resnet.layer3.0.conv2": self.resnet.layer3[0].conv2,
            "resnet.layer3.0.bn2": self.resnet.layer3[0].bn2,
            "resnet.layer3.0.conv3": self.resnet.layer3[0].conv3,
            "resnet.layer3.0.bn3": self.resnet.layer3[0].bn3,
            "resnet.layer3.0.relu": self.resnet.layer3[0].relu,
            "resnet.layer3.0.downsample": self.resnet.layer3[0].downsample,
            "resnet.layer3.0.downsample.0": self.resnet.layer3[0].downsample[0],
            "resnet.layer3.0.downsample.1": self.resnet.layer3[0].downsample[1],
            "resnet.layer3.1": self.resnet.layer3[1],
            "resnet.layer3.1.conv1": self.resnet.layer3[1].conv1,
            "resnet.layer3.1.bn1": self.resnet.layer3[1].bn1,
            "resnet.layer3.1.conv2": self.resnet.layer3[1].conv2,
            "resnet.layer3.1.bn2": self.resnet.layer3[1].bn2,
            "resnet.layer3.1.conv3": self.resnet.layer3[1].conv3,
            "resnet.layer3.1.bn3": self.resnet.layer3[1].bn3,
            "resnet.layer3.1.relu": self.resnet.layer3[1].relu,
            "resnet.layer3.2": self.resnet.layer3[2],
            "resnet.layer3.2.conv1": self.resnet.layer3[2].conv1,
            "resnet.layer3.2.bn1": self.resnet.layer3[2].bn1,
            "resnet.layer3.2.conv2": self.resnet.layer3[2].conv2,
            "resnet.layer3.2.bn2": self.resnet.layer3[2].bn2,
            "resnet.layer3.2.conv3": self.resnet.layer3[2].conv3,
            "resnet.layer3.2.bn3": self.resnet.layer3[2].bn3,
            "resnet.layer3.2.relu": self.resnet.layer3[2].relu,
            "resnet.layer3.3": self.resnet.layer3[3],
            "resnet.layer3.3.conv1": self.resnet.layer3[3].conv1,
            "resnet.layer3.3.bn1": self.resnet.layer3[3].bn1,
            "resnet.layer3.3.conv2": self.resnet.layer3[3].conv2,
            "resnet.layer3.3.bn2": self.resnet.layer3[3].bn2,
            "resnet.layer3.3.conv3": self.resnet.layer3[3].conv3,
            "resnet.layer3.3.bn3": self.resnet.layer3[3].bn3,
            "resnet.layer3.3.relu": self.resnet.layer3[3].relu,
            "resnet.layer3.4": self.resnet.layer3[4],
            "resnet.layer3.4.conv1": self.resnet.layer3[4].conv1,
            "resnet.layer3.4.bn1": self.resnet.layer3[4].bn1,
            "resnet.layer3.4.conv2": self.resnet.layer3[4].conv2,
            "resnet.layer3.4.bn2": self.resnet.layer3[4].bn2,
            "resnet.layer3.4.conv3": self.resnet.layer3[4].conv3,
            "resnet.layer3.4.bn3": self.resnet.layer3[4].bn3,
            "resnet.layer3.4.relu": self.resnet.layer3[4].relu,
            "resnet.layer3.5": self.resnet.layer3[5],
            "resnet.layer3.5.conv1": self.resnet.layer3[5].conv1,
            "resnet.layer3.5.bn1": self.resnet.layer3[5].bn1,
            "resnet.layer3.5.conv2": self.resnet.layer3[5].conv2,
            "resnet.layer3.5.bn2": self.resnet.layer3[5].bn2,
            "resnet.layer3.5.conv3": self.resnet.layer3[5].conv3,
            "resnet.layer3.5.bn3": self.resnet.layer3[5].bn3,
            "resnet.layer3.5.relu": self.resnet.layer3[5].relu,
            "resnet.layer3.6": self.resnet.layer3[6],
            "resnet.layer3.6.conv1": self.resnet.layer3[6].conv1,
            "resnet.layer3.6.bn1": self.resnet.layer3[6].bn1,
            "resnet.layer3.6.conv2": self.resnet.layer3[6].conv2,
            "resnet.layer3.6.bn2": self.resnet.layer3[6].bn2,
            "resnet.layer3.6.conv3": self.resnet.layer3[6].conv3,
            "resnet.layer3.6.bn3": self.resnet.layer3[6].bn3,
            "resnet.layer3.6.relu": self.resnet.layer3[6].relu,
            "resnet.layer3.7": self.resnet.layer3[7],
            "resnet.layer3.7.conv1": self.resnet.layer3[7].conv1,
            "resnet.layer3.7.bn1": self.resnet.layer3[7].bn1,
            "resnet.layer3.7.conv2": self.resnet.layer3[7].conv2,
            "resnet.layer3.7.bn2": self.resnet.layer3[7].bn2,
            "resnet.layer3.7.conv3": self.resnet.layer3[7].conv3,
            "resnet.layer3.7.bn3": self.resnet.layer3[7].bn3,
            "resnet.layer3.7.relu": self.resnet.layer3[7].relu,
            "resnet.layer3.8": self.resnet.layer3[8],
            "resnet.layer3.8.conv1": self.resnet.layer3[8].conv1,
            "resnet.layer3.8.bn1": self.resnet.layer3[8].bn1,
            "resnet.layer3.8.conv2": self.resnet.layer3[8].conv2,
            "resnet.layer3.8.bn2": self.resnet.layer3[8].bn2,
            "resnet.layer3.8.conv3": self.resnet.layer3[8].conv3,
            "resnet.layer3.8.bn3": self.resnet.layer3[8].bn3,
            "resnet.layer3.8.relu": self.resnet.layer3[8].relu,
            "resnet.layer3.9": self.resnet.layer3[9],
            "resnet.layer3.9.conv1": self.resnet.layer3[9].conv1,
            "resnet.layer3.9.bn1": self.resnet.layer3[9].bn1,
            "resnet.layer3.9.conv2": self.resnet.layer3[9].conv2,
            "resnet.layer3.9.bn2": self.resnet.layer3[9].bn2,
            "resnet.layer3.9.conv3": self.resnet.layer3[9].conv3,
            "resnet.layer3.9.bn3": self.resnet.layer3[9].bn3,
            "resnet.layer3.9.relu": self.resnet.layer3[9].relu,
            "resnet.layer3.10": self.resnet.layer3[10],
            "resnet.layer3.10.conv1": self.resnet.layer3[10].conv1,
            "resnet.layer3.10.bn1": self.resnet.layer3[10].bn1,
            "resnet.layer3.10.conv2": self.resnet.layer3[10].conv2,
            "resnet.layer3.10.bn2": self.resnet.layer3[10].bn2,
            "resnet.layer3.10.conv3": self.resnet.layer3[10].conv3,
            "resnet.layer3.10.bn3": self.resnet.layer3[10].bn3,
            "resnet.layer3.10.relu": self.resnet.layer3[10].relu,
            "resnet.layer3.11": self.resnet.layer3[11],
            "resnet.layer3.11.conv1": self.resnet.layer3[11].conv1,
            "resnet.layer3.11.bn1": self.resnet.layer3[11].bn1,
            "resnet.layer3.11.conv2": self.resnet.layer3[11].conv2,
            "resnet.layer3.11.bn2": self.resnet.layer3[11].bn2,
            "resnet.layer3.11.conv3": self.resnet.layer3[11].conv3,
            "resnet.layer3.11.bn3": self.resnet.layer3[11].bn3,
            "resnet.layer3.11.relu": self.resnet.layer3[11].relu,
            "resnet.layer3.12": self.resnet.layer3[12],
            "resnet.layer3.12.conv1": self.resnet.layer3[12].conv1,
            "resnet.layer3.12.bn1": self.resnet.layer3[12].bn1,
            "resnet.layer3.12.conv2": self.resnet.layer3[12].conv2,
            "resnet.layer3.12.bn2": self.resnet.layer3[12].bn2,
            "resnet.layer3.12.conv3": self.resnet.layer3[12].conv3,
            "resnet.layer3.12.bn3": self.resnet.layer3[12].bn3,
            "resnet.layer3.12.relu": self.resnet.layer3[12].relu,
            "resnet.layer3.13": self.resnet.layer3[13],
            "resnet.layer3.13.conv1": self.resnet.layer3[13].conv1,
            "resnet.layer3.13.bn1": self.resnet.layer3[13].bn1,
            "resnet.layer3.13.conv2": self.resnet.layer3[13].conv2,
            "resnet.layer3.13.bn2": self.resnet.layer3[13].bn2,
            "resnet.layer3.13.conv3": self.resnet.layer3[13].conv3,
            "resnet.layer3.13.bn3": self.resnet.layer3[13].bn3,
            "resnet.layer3.13.relu": self.resnet.layer3[13].relu,
            "resnet.layer3.14": self.resnet.layer3[14],
            "resnet.layer3.14.conv1": self.resnet.layer3[14].conv1,
            "resnet.layer3.14.bn1": self.resnet.layer3[14].bn1,
            "resnet.layer3.14.conv2": self.resnet.layer3[14].conv2,
            "resnet.layer3.14.bn2": self.resnet.layer3[14].bn2,
            "resnet.layer3.14.conv3": self.resnet.layer3[14].conv3,
            "resnet.layer3.14.bn3": self.resnet.layer3[14].bn3,
            "resnet.layer3.14.relu": self.resnet.layer3[14].relu,
            "resnet.layer3.15": self.resnet.layer3[15],
            "resnet.layer3.15.conv1": self.resnet.layer3[15].conv1,
            "resnet.layer3.15.bn1": self.resnet.layer3[15].bn1,
            "resnet.layer3.15.conv2": self.resnet.layer3[15].conv2,
            "resnet.layer3.15.bn2": self.resnet.layer3[15].bn2,
            "resnet.layer3.15.conv3": self.resnet.layer3[15].conv3,
            "resnet.layer3.15.bn3": self.resnet.layer3[15].bn3,
            "resnet.layer3.15.relu": self.resnet.layer3[15].relu,
            "resnet.layer3.16": self.resnet.layer3[16],
            "resnet.layer3.16.conv1": self.resnet.layer3[16].conv1,
            "resnet.layer3.16.bn1": self.resnet.layer3[16].bn1,
            "resnet.layer3.16.conv2": self.resnet.layer3[16].conv2,
            "resnet.layer3.16.bn2": self.resnet.layer3[16].bn2,
            "resnet.layer3.16.conv3": self.resnet.layer3[16].conv3,
            "resnet.layer3.16.bn3": self.resnet.layer3[16].bn3,
            "resnet.layer3.16.relu": self.resnet.layer3[16].relu,
            "resnet.layer3.17": self.resnet.layer3[17],
            "resnet.layer3.17.conv1": self.resnet.layer3[17].conv1,
            "resnet.layer3.17.bn1": self.resnet.layer3[17].bn1,
            "resnet.layer3.17.conv2": self.resnet.layer3[17].conv2,
            "resnet.layer3.17.bn2": self.resnet.layer3[17].bn2,
            "resnet.layer3.17.conv3": self.resnet.layer3[17].conv3,
            "resnet.layer3.17.bn3": self.resnet.layer3[17].bn3,
            "resnet.layer3.17.relu": self.resnet.layer3[17].relu,
            "resnet.layer3.18": self.resnet.layer3[18],
            "resnet.layer3.18.conv1": self.resnet.layer3[18].conv1,
            "resnet.layer3.18.bn1": self.resnet.layer3[18].bn1,
            "resnet.layer3.18.conv2": self.resnet.layer3[18].conv2,
            "resnet.layer3.18.bn2": self.resnet.layer3[18].bn2,
            "resnet.layer3.18.conv3": self.resnet.layer3[18].conv3,
            "resnet.layer3.18.bn3": self.resnet.layer3[18].bn3,
            "resnet.layer3.18.relu": self.resnet.layer3[18].relu,
            "resnet.layer3.19": self.resnet.layer3[19],
            "resnet.layer3.19.conv1": self.resnet.layer3[19].conv1,
            "resnet.layer3.19.bn1": self.resnet.layer3[19].bn1,
            "resnet.layer3.19.conv2": self.resnet.layer3[19].conv2,
            "resnet.layer3.19.bn2": self.resnet.layer3[19].bn2,
            "resnet.layer3.19.conv3": self.resnet.layer3[19].conv3,
            "resnet.layer3.19.bn3": self.resnet.layer3[19].bn3,
            "resnet.layer3.19.relu": self.resnet.layer3[19].relu,
            "resnet.layer3.20": self.resnet.layer3[20],
            "resnet.layer3.20.conv1": self.resnet.layer3[20].conv1,
            "resnet.layer3.20.bn1": self.resnet.layer3[20].bn1,
            "resnet.layer3.20.conv2": self.resnet.layer3[20].conv2,
            "resnet.layer3.20.bn2": self.resnet.layer3[20].bn2,
            "resnet.layer3.20.conv3": self.resnet.layer3[20].conv3,
            "resnet.layer3.20.bn3": self.resnet.layer3[20].bn3,
            "resnet.layer3.20.relu": self.resnet.layer3[20].relu,
            "resnet.layer3.21": self.resnet.layer3[21],
            "resnet.layer3.21.conv1": self.resnet.layer3[21].conv1,
            "resnet.layer3.21.bn1": self.resnet.layer3[21].bn1,
            "resnet.layer3.21.conv2": self.resnet.layer3[21].conv2,
            "resnet.layer3.21.bn2": self.resnet.layer3[21].bn2,
            "resnet.layer3.21.conv3": self.resnet.layer3[21].conv3,
            "resnet.layer3.21.bn3": self.resnet.layer3[21].bn3,
            "resnet.layer3.21.relu": self.resnet.layer3[21].relu,
            "resnet.layer3.22": self.resnet.layer3[22],
            "resnet.layer3.22.conv1": self.resnet.layer3[22].conv1,
            "resnet.layer3.22.bn1": self.resnet.layer3[22].bn1,
            "resnet.layer3.22.conv2": self.resnet.layer3[22].conv2,
            "resnet.layer3.22.bn2": self.resnet.layer3[22].bn2,
            "resnet.layer3.22.conv3": self.resnet.layer3[22].conv3,
            "resnet.layer3.22.bn3": self.resnet.layer3[22].bn3,
            "resnet.layer3.22.relu": self.resnet.layer3[22].relu,
            "resnet.layer4": self.resnet.layer4,
            "resnet.layer4.0": self.resnet.layer4[0],
            "resnet.layer4.0.conv1": self.resnet.layer4[0].conv1,
            "resnet.layer4.0.bn1": self.resnet.layer4[0].bn1,
            "resnet.layer4.0.conv2": self.resnet.layer4[0].conv2,
            "resnet.layer4.0.bn2": self.resnet.layer4[0].bn2,
            "resnet.layer4.0.conv3": self.resnet.layer4[0].conv3,
            "resnet.layer4.0.bn3": self.resnet.layer4[0].bn3,
            "resnet.layer4.0.relu": self.resnet.layer4[0].relu,
            "resnet.layer4.0.downsample": self.resnet.layer4[0].downsample,
            "resnet.layer4.0.downsample.0": self.resnet.layer4[0].downsample[0],
            "resnet.layer4.0.downsample.1": self.resnet.layer4[0].downsample[1],
            "resnet.layer4.1": self.resnet.layer4[1],
            "resnet.layer4.1.conv1": self.resnet.layer4[1].conv1,
            "resnet.layer4.1.bn1": self.resnet.layer4[1].bn1,
            "resnet.layer4.1.conv2": self.resnet.layer4[1].conv2,
            "resnet.layer4.1.bn2": self.resnet.layer4[1].bn2,
            "resnet.layer4.1.conv3": self.resnet.layer4[1].conv3,
            "resnet.layer4.1.bn3": self.resnet.layer4[1].bn3,
            "resnet.layer4.1.relu": self.resnet.layer4[1].relu,
            "resnet.layer4.2": self.resnet.layer4[2],
            "resnet.layer4.2.conv1": self.resnet.layer4[2].conv1,
            "resnet.layer4.2.bn1": self.resnet.layer4[2].bn1,
            "resnet.layer4.2.conv2": self.resnet.layer4[2].conv2,
            "resnet.layer4.2.bn2": self.resnet.layer4[2].bn2,
            "resnet.layer4.2.conv3": self.resnet.layer4[2].conv3,
            "resnet.layer4.2.bn3": self.resnet.layer4[2].bn3,
            "resnet.layer4.2.relu": self.resnet.layer4[2].relu,
            # "aspp": self.aspp,
            # "aspp.aspp1": self.aspp.aspp1,
            # "aspp.aspp1.aspp_conv": self.aspp.aspp1.aspp_conv,
            # "aspp.aspp1.aspp_conv.0": self.aspp.aspp1.aspp_conv[0],
            # "aspp.aspp1.aspp_conv.1": self.aspp.aspp1.aspp_conv[1],
            # "aspp.aspp1.aspp_conv.2": self.aspp.aspp1.aspp_conv[2],
            # "aspp.aspp2": self.aspp.aspp2,
            # "aspp.aspp2.aspp_conv": self.aspp.aspp2.aspp_conv,
            # "aspp.aspp2.aspp_conv.0": self.aspp.aspp2.aspp_conv[0],
            # "aspp.aspp2.aspp_conv.1": self.aspp.aspp2.aspp_conv[1],
            # "aspp.aspp2.aspp_conv.2": self.aspp.aspp2.aspp_conv[2],
            # "aspp.aspp3": self.aspp.aspp3,
            # "aspp.aspp3.aspp_conv": self.aspp.aspp3.aspp_conv,
            # "aspp.aspp3.aspp_conv.0": self.aspp.aspp3.aspp_conv[0],
            # "aspp.aspp3.aspp_conv.1": self.aspp.aspp3.aspp_conv[1],
            # "aspp.aspp3.aspp_conv.2": self.aspp.aspp3.aspp_conv[2],
            # "aspp.aspp4": self.aspp.aspp4,
            # "aspp.aspp4.aspp_conv": self.aspp.aspp4.aspp_conv,
            # "aspp.aspp4.aspp_conv.0": self.aspp.aspp4.aspp_conv[0],
            # "aspp.aspp4.aspp_conv.1": self.aspp.aspp4.aspp_conv[1],
            # "aspp.aspp4.aspp_conv.2": self.aspp.aspp4.aspp_conv[2],
            # "aspp.aspp_pooling": self.aspp.aspp_pooling,
            # "aspp.aspp_pooling.conv": self.aspp.aspp_pooling.conv,
            # "aspp.aspp_pooling.conv.0": self.aspp.aspp_pooling.conv[0],
            # "aspp.aspp_pooling.conv.1": self.aspp.aspp_pooling.conv[1],
            # "aspp.aspp_pooling.conv.2": self.aspp.aspp_pooling.conv[2],
            # "aspp.conv1": self.aspp.conv1,
            # "aspp.bn1": self.aspp.bn1,
            # "aspp.relu": self.aspp.relu,
            # "aspp.conv2": self.aspp.conv2,
            # "aspp.drop": self.aspp.drop,
        }
        self.origin_layer_names = {
            "resnet": self.resnet,
            "resnet.conv1": self.resnet.conv1,
            "resnet.bn1": self.resnet.bn1,
            "resnet.relu": self.resnet.relu,
            "resnet.maxpool": self.resnet.maxpool,
            "resnet.layer1": self.resnet.layer1,
            "resnet.layer1.0": self.resnet.layer1[0],
            "resnet.layer1.0.conv1": self.resnet.layer1[0].conv1,
            "resnet.layer1.0.bn1": self.resnet.layer1[0].bn1,
            "resnet.layer1.0.conv2": self.resnet.layer1[0].conv2,
            "resnet.layer1.0.bn2": self.resnet.layer1[0].bn2,
            "resnet.layer1.0.conv3": self.resnet.layer1[0].conv3,
            "resnet.layer1.0.bn3": self.resnet.layer1[0].bn3,
            "resnet.layer1.0.relu": self.resnet.layer1[0].relu,
            "resnet.layer1.0.downsample": self.resnet.layer1[0].downsample,
            "resnet.layer1.0.downsample.0": self.resnet.layer1[0].downsample[0],
            "resnet.layer1.0.downsample.1": self.resnet.layer1[0].downsample[1],
            "resnet.layer1.1": self.resnet.layer1[1],
            "resnet.layer1.1.conv1": self.resnet.layer1[1].conv1,
            "resnet.layer1.1.bn1": self.resnet.layer1[1].bn1,
            "resnet.layer1.1.conv2": self.resnet.layer1[1].conv2,
            "resnet.layer1.1.bn2": self.resnet.layer1[1].bn2,
            "resnet.layer1.1.conv3": self.resnet.layer1[1].conv3,
            "resnet.layer1.1.bn3": self.resnet.layer1[1].bn3,
            "resnet.layer1.1.relu": self.resnet.layer1[1].relu,
            "resnet.layer1.2": self.resnet.layer1[2],
            "resnet.layer1.2.conv1": self.resnet.layer1[2].conv1,
            "resnet.layer1.2.bn1": self.resnet.layer1[2].bn1,
            "resnet.layer1.2.conv2": self.resnet.layer1[2].conv2,
            "resnet.layer1.2.bn2": self.resnet.layer1[2].bn2,
            "resnet.layer1.2.conv3": self.resnet.layer1[2].conv3,
            "resnet.layer1.2.bn3": self.resnet.layer1[2].bn3,
            "resnet.layer1.2.relu": self.resnet.layer1[2].relu,
            "resnet.layer2": self.resnet.layer2,
            "resnet.layer2.0": self.resnet.layer2[0],
            "resnet.layer2.0.conv1": self.resnet.layer2[0].conv1,
            "resnet.layer2.0.bn1": self.resnet.layer2[0].bn1,
            "resnet.layer2.0.conv2": self.resnet.layer2[0].conv2,
            "resnet.layer2.0.bn2": self.resnet.layer2[0].bn2,
            "resnet.layer2.0.conv3": self.resnet.layer2[0].conv3,
            "resnet.layer2.0.bn3": self.resnet.layer2[0].bn3,
            "resnet.layer2.0.relu": self.resnet.layer2[0].relu,
            "resnet.layer2.0.downsample": self.resnet.layer2[0].downsample,
            "resnet.layer2.0.downsample.0": self.resnet.layer2[0].downsample[0],
            "resnet.layer2.0.downsample.1": self.resnet.layer2[0].downsample[1],
            "resnet.layer2.1": self.resnet.layer2[1],
            "resnet.layer2.1.conv1": self.resnet.layer2[1].conv1,
            "resnet.layer2.1.bn1": self.resnet.layer2[1].bn1,
            "resnet.layer2.1.conv2": self.resnet.layer2[1].conv2,
            "resnet.layer2.1.bn2": self.resnet.layer2[1].bn2,
            "resnet.layer2.1.conv3": self.resnet.layer2[1].conv3,
            "resnet.layer2.1.bn3": self.resnet.layer2[1].bn3,
            "resnet.layer2.1.relu": self.resnet.layer2[1].relu,
            "resnet.layer2.2": self.resnet.layer2[2],
            "resnet.layer2.2.conv1": self.resnet.layer2[2].conv1,
            "resnet.layer2.2.bn1": self.resnet.layer2[2].bn1,
            "resnet.layer2.2.conv2": self.resnet.layer2[2].conv2,
            "resnet.layer2.2.bn2": self.resnet.layer2[2].bn2,
            "resnet.layer2.2.conv3": self.resnet.layer2[2].conv3,
            "resnet.layer2.2.bn3": self.resnet.layer2[2].bn3,
            "resnet.layer2.2.relu": self.resnet.layer2[2].relu,
            "resnet.layer2.3": self.resnet.layer2[3],
            "resnet.layer2.3.conv1": self.resnet.layer2[3].conv1,
            "resnet.layer2.3.bn1": self.resnet.layer2[3].bn1,
            "resnet.layer2.3.conv2": self.resnet.layer2[3].conv2,
            "resnet.layer2.3.bn2": self.resnet.layer2[3].bn2,
            "resnet.layer2.3.conv3": self.resnet.layer2[3].conv3,
            "resnet.layer2.3.bn3": self.resnet.layer2[3].bn3,
            "resnet.layer2.3.relu": self.resnet.layer2[3].relu,
            "resnet.layer3": self.resnet.layer3,
            "resnet.layer3.0": self.resnet.layer3[0],
            "resnet.layer3.0.conv1": self.resnet.layer3[0].conv1,
            "resnet.layer3.0.bn1": self.resnet.layer3[0].bn1,
            "resnet.layer3.0.conv2": self.resnet.layer3[0].conv2,
            "resnet.layer3.0.bn2": self.resnet.layer3[0].bn2,
            "resnet.layer3.0.conv3": self.resnet.layer3[0].conv3,
            "resnet.layer3.0.bn3": self.resnet.layer3[0].bn3,
            "resnet.layer3.0.relu": self.resnet.layer3[0].relu,
            "resnet.layer3.0.downsample": self.resnet.layer3[0].downsample,
            "resnet.layer3.0.downsample.0": self.resnet.layer3[0].downsample[0],
            "resnet.layer3.0.downsample.1": self.resnet.layer3[0].downsample[1],
            "resnet.layer3.1": self.resnet.layer3[1],
            "resnet.layer3.1.conv1": self.resnet.layer3[1].conv1,
            "resnet.layer3.1.bn1": self.resnet.layer3[1].bn1,
            "resnet.layer3.1.conv2": self.resnet.layer3[1].conv2,
            "resnet.layer3.1.bn2": self.resnet.layer3[1].bn2,
            "resnet.layer3.1.conv3": self.resnet.layer3[1].conv3,
            "resnet.layer3.1.bn3": self.resnet.layer3[1].bn3,
            "resnet.layer3.1.relu": self.resnet.layer3[1].relu,
            "resnet.layer3.2": self.resnet.layer3[2],
            "resnet.layer3.2.conv1": self.resnet.layer3[2].conv1,
            "resnet.layer3.2.bn1": self.resnet.layer3[2].bn1,
            "resnet.layer3.2.conv2": self.resnet.layer3[2].conv2,
            "resnet.layer3.2.bn2": self.resnet.layer3[2].bn2,
            "resnet.layer3.2.conv3": self.resnet.layer3[2].conv3,
            "resnet.layer3.2.bn3": self.resnet.layer3[2].bn3,
            "resnet.layer3.2.relu": self.resnet.layer3[2].relu,
            "resnet.layer3.3": self.resnet.layer3[3],
            "resnet.layer3.3.conv1": self.resnet.layer3[3].conv1,
            "resnet.layer3.3.bn1": self.resnet.layer3[3].bn1,
            "resnet.layer3.3.conv2": self.resnet.layer3[3].conv2,
            "resnet.layer3.3.bn2": self.resnet.layer3[3].bn2,
            "resnet.layer3.3.conv3": self.resnet.layer3[3].conv3,
            "resnet.layer3.3.bn3": self.resnet.layer3[3].bn3,
            "resnet.layer3.3.relu": self.resnet.layer3[3].relu,
            "resnet.layer3.4": self.resnet.layer3[4],
            "resnet.layer3.4.conv1": self.resnet.layer3[4].conv1,
            "resnet.layer3.4.bn1": self.resnet.layer3[4].bn1,
            "resnet.layer3.4.conv2": self.resnet.layer3[4].conv2,
            "resnet.layer3.4.bn2": self.resnet.layer3[4].bn2,
            "resnet.layer3.4.conv3": self.resnet.layer3[4].conv3,
            "resnet.layer3.4.bn3": self.resnet.layer3[4].bn3,
            "resnet.layer3.4.relu": self.resnet.layer3[4].relu,
            "resnet.layer3.5": self.resnet.layer3[5],
            "resnet.layer3.5.conv1": self.resnet.layer3[5].conv1,
            "resnet.layer3.5.bn1": self.resnet.layer3[5].bn1,
            "resnet.layer3.5.conv2": self.resnet.layer3[5].conv2,
            "resnet.layer3.5.bn2": self.resnet.layer3[5].bn2,
            "resnet.layer3.5.conv3": self.resnet.layer3[5].conv3,
            "resnet.layer3.5.bn3": self.resnet.layer3[5].bn3,
            "resnet.layer3.5.relu": self.resnet.layer3[5].relu,
            "resnet.layer3.6": self.resnet.layer3[6],
            "resnet.layer3.6.conv1": self.resnet.layer3[6].conv1,
            "resnet.layer3.6.bn1": self.resnet.layer3[6].bn1,
            "resnet.layer3.6.conv2": self.resnet.layer3[6].conv2,
            "resnet.layer3.6.bn2": self.resnet.layer3[6].bn2,
            "resnet.layer3.6.conv3": self.resnet.layer3[6].conv3,
            "resnet.layer3.6.bn3": self.resnet.layer3[6].bn3,
            "resnet.layer3.6.relu": self.resnet.layer3[6].relu,
            "resnet.layer3.7": self.resnet.layer3[7],
            "resnet.layer3.7.conv1": self.resnet.layer3[7].conv1,
            "resnet.layer3.7.bn1": self.resnet.layer3[7].bn1,
            "resnet.layer3.7.conv2": self.resnet.layer3[7].conv2,
            "resnet.layer3.7.bn2": self.resnet.layer3[7].bn2,
            "resnet.layer3.7.conv3": self.resnet.layer3[7].conv3,
            "resnet.layer3.7.bn3": self.resnet.layer3[7].bn3,
            "resnet.layer3.7.relu": self.resnet.layer3[7].relu,
            "resnet.layer3.8": self.resnet.layer3[8],
            "resnet.layer3.8.conv1": self.resnet.layer3[8].conv1,
            "resnet.layer3.8.bn1": self.resnet.layer3[8].bn1,
            "resnet.layer3.8.conv2": self.resnet.layer3[8].conv2,
            "resnet.layer3.8.bn2": self.resnet.layer3[8].bn2,
            "resnet.layer3.8.conv3": self.resnet.layer3[8].conv3,
            "resnet.layer3.8.bn3": self.resnet.layer3[8].bn3,
            "resnet.layer3.8.relu": self.resnet.layer3[8].relu,
            "resnet.layer3.9": self.resnet.layer3[9],
            "resnet.layer3.9.conv1": self.resnet.layer3[9].conv1,
            "resnet.layer3.9.bn1": self.resnet.layer3[9].bn1,
            "resnet.layer3.9.conv2": self.resnet.layer3[9].conv2,
            "resnet.layer3.9.bn2": self.resnet.layer3[9].bn2,
            "resnet.layer3.9.conv3": self.resnet.layer3[9].conv3,
            "resnet.layer3.9.bn3": self.resnet.layer3[9].bn3,
            "resnet.layer3.9.relu": self.resnet.layer3[9].relu,
            "resnet.layer3.10": self.resnet.layer3[10],
            "resnet.layer3.10.conv1": self.resnet.layer3[10].conv1,
            "resnet.layer3.10.bn1": self.resnet.layer3[10].bn1,
            "resnet.layer3.10.conv2": self.resnet.layer3[10].conv2,
            "resnet.layer3.10.bn2": self.resnet.layer3[10].bn2,
            "resnet.layer3.10.conv3": self.resnet.layer3[10].conv3,
            "resnet.layer3.10.bn3": self.resnet.layer3[10].bn3,
            "resnet.layer3.10.relu": self.resnet.layer3[10].relu,
            "resnet.layer3.11": self.resnet.layer3[11],
            "resnet.layer3.11.conv1": self.resnet.layer3[11].conv1,
            "resnet.layer3.11.bn1": self.resnet.layer3[11].bn1,
            "resnet.layer3.11.conv2": self.resnet.layer3[11].conv2,
            "resnet.layer3.11.bn2": self.resnet.layer3[11].bn2,
            "resnet.layer3.11.conv3": self.resnet.layer3[11].conv3,
            "resnet.layer3.11.bn3": self.resnet.layer3[11].bn3,
            "resnet.layer3.11.relu": self.resnet.layer3[11].relu,
            "resnet.layer3.12": self.resnet.layer3[12],
            "resnet.layer3.12.conv1": self.resnet.layer3[12].conv1,
            "resnet.layer3.12.bn1": self.resnet.layer3[12].bn1,
            "resnet.layer3.12.conv2": self.resnet.layer3[12].conv2,
            "resnet.layer3.12.bn2": self.resnet.layer3[12].bn2,
            "resnet.layer3.12.conv3": self.resnet.layer3[12].conv3,
            "resnet.layer3.12.bn3": self.resnet.layer3[12].bn3,
            "resnet.layer3.12.relu": self.resnet.layer3[12].relu,
            "resnet.layer3.13": self.resnet.layer3[13],
            "resnet.layer3.13.conv1": self.resnet.layer3[13].conv1,
            "resnet.layer3.13.bn1": self.resnet.layer3[13].bn1,
            "resnet.layer3.13.conv2": self.resnet.layer3[13].conv2,
            "resnet.layer3.13.bn2": self.resnet.layer3[13].bn2,
            "resnet.layer3.13.conv3": self.resnet.layer3[13].conv3,
            "resnet.layer3.13.bn3": self.resnet.layer3[13].bn3,
            "resnet.layer3.13.relu": self.resnet.layer3[13].relu,
            "resnet.layer3.14": self.resnet.layer3[14],
            "resnet.layer3.14.conv1": self.resnet.layer3[14].conv1,
            "resnet.layer3.14.bn1": self.resnet.layer3[14].bn1,
            "resnet.layer3.14.conv2": self.resnet.layer3[14].conv2,
            "resnet.layer3.14.bn2": self.resnet.layer3[14].bn2,
            "resnet.layer3.14.conv3": self.resnet.layer3[14].conv3,
            "resnet.layer3.14.bn3": self.resnet.layer3[14].bn3,
            "resnet.layer3.14.relu": self.resnet.layer3[14].relu,
            "resnet.layer3.15": self.resnet.layer3[15],
            "resnet.layer3.15.conv1": self.resnet.layer3[15].conv1,
            "resnet.layer3.15.bn1": self.resnet.layer3[15].bn1,
            "resnet.layer3.15.conv2": self.resnet.layer3[15].conv2,
            "resnet.layer3.15.bn2": self.resnet.layer3[15].bn2,
            "resnet.layer3.15.conv3": self.resnet.layer3[15].conv3,
            "resnet.layer3.15.bn3": self.resnet.layer3[15].bn3,
            "resnet.layer3.15.relu": self.resnet.layer3[15].relu,
            "resnet.layer3.16": self.resnet.layer3[16],
            "resnet.layer3.16.conv1": self.resnet.layer3[16].conv1,
            "resnet.layer3.16.bn1": self.resnet.layer3[16].bn1,
            "resnet.layer3.16.conv2": self.resnet.layer3[16].conv2,
            "resnet.layer3.16.bn2": self.resnet.layer3[16].bn2,
            "resnet.layer3.16.conv3": self.resnet.layer3[16].conv3,
            "resnet.layer3.16.bn3": self.resnet.layer3[16].bn3,
            "resnet.layer3.16.relu": self.resnet.layer3[16].relu,
            "resnet.layer3.17": self.resnet.layer3[17],
            "resnet.layer3.17.conv1": self.resnet.layer3[17].conv1,
            "resnet.layer3.17.bn1": self.resnet.layer3[17].bn1,
            "resnet.layer3.17.conv2": self.resnet.layer3[17].conv2,
            "resnet.layer3.17.bn2": self.resnet.layer3[17].bn2,
            "resnet.layer3.17.conv3": self.resnet.layer3[17].conv3,
            "resnet.layer3.17.bn3": self.resnet.layer3[17].bn3,
            "resnet.layer3.17.relu": self.resnet.layer3[17].relu,
            "resnet.layer3.18": self.resnet.layer3[18],
            "resnet.layer3.18.conv1": self.resnet.layer3[18].conv1,
            "resnet.layer3.18.bn1": self.resnet.layer3[18].bn1,
            "resnet.layer3.18.conv2": self.resnet.layer3[18].conv2,
            "resnet.layer3.18.bn2": self.resnet.layer3[18].bn2,
            "resnet.layer3.18.conv3": self.resnet.layer3[18].conv3,
            "resnet.layer3.18.bn3": self.resnet.layer3[18].bn3,
            "resnet.layer3.18.relu": self.resnet.layer3[18].relu,
            "resnet.layer3.19": self.resnet.layer3[19],
            "resnet.layer3.19.conv1": self.resnet.layer3[19].conv1,
            "resnet.layer3.19.bn1": self.resnet.layer3[19].bn1,
            "resnet.layer3.19.conv2": self.resnet.layer3[19].conv2,
            "resnet.layer3.19.bn2": self.resnet.layer3[19].bn2,
            "resnet.layer3.19.conv3": self.resnet.layer3[19].conv3,
            "resnet.layer3.19.bn3": self.resnet.layer3[19].bn3,
            "resnet.layer3.19.relu": self.resnet.layer3[19].relu,
            "resnet.layer3.20": self.resnet.layer3[20],
            "resnet.layer3.20.conv1": self.resnet.layer3[20].conv1,
            "resnet.layer3.20.bn1": self.resnet.layer3[20].bn1,
            "resnet.layer3.20.conv2": self.resnet.layer3[20].conv2,
            "resnet.layer3.20.bn2": self.resnet.layer3[20].bn2,
            "resnet.layer3.20.conv3": self.resnet.layer3[20].conv3,
            "resnet.layer3.20.bn3": self.resnet.layer3[20].bn3,
            "resnet.layer3.20.relu": self.resnet.layer3[20].relu,
            "resnet.layer3.21": self.resnet.layer3[21],
            "resnet.layer3.21.conv1": self.resnet.layer3[21].conv1,
            "resnet.layer3.21.bn1": self.resnet.layer3[21].bn1,
            "resnet.layer3.21.conv2": self.resnet.layer3[21].conv2,
            "resnet.layer3.21.bn2": self.resnet.layer3[21].bn2,
            "resnet.layer3.21.conv3": self.resnet.layer3[21].conv3,
            "resnet.layer3.21.bn3": self.resnet.layer3[21].bn3,
            "resnet.layer3.21.relu": self.resnet.layer3[21].relu,
            "resnet.layer3.22": self.resnet.layer3[22],
            "resnet.layer3.22.conv1": self.resnet.layer3[22].conv1,
            "resnet.layer3.22.bn1": self.resnet.layer3[22].bn1,
            "resnet.layer3.22.conv2": self.resnet.layer3[22].conv2,
            "resnet.layer3.22.bn2": self.resnet.layer3[22].bn2,
            "resnet.layer3.22.conv3": self.resnet.layer3[22].conv3,
            "resnet.layer3.22.bn3": self.resnet.layer3[22].bn3,
            "resnet.layer3.22.relu": self.resnet.layer3[22].relu,
            "resnet.layer4": self.resnet.layer4,
            "resnet.layer4.0": self.resnet.layer4[0],
            "resnet.layer4.0.conv1": self.resnet.layer4[0].conv1,
            "resnet.layer4.0.bn1": self.resnet.layer4[0].bn1,
            "resnet.layer4.0.conv2": self.resnet.layer4[0].conv2,
            "resnet.layer4.0.bn2": self.resnet.layer4[0].bn2,
            "resnet.layer4.0.conv3": self.resnet.layer4[0].conv3,
            "resnet.layer4.0.bn3": self.resnet.layer4[0].bn3,
            "resnet.layer4.0.relu": self.resnet.layer4[0].relu,
            "resnet.layer4.0.downsample": self.resnet.layer4[0].downsample,
            "resnet.layer4.0.downsample.0": self.resnet.layer4[0].downsample[0],
            "resnet.layer4.0.downsample.1": self.resnet.layer4[0].downsample[1],
            "resnet.layer4.1": self.resnet.layer4[1],
            "resnet.layer4.1.conv1": self.resnet.layer4[1].conv1,
            "resnet.layer4.1.bn1": self.resnet.layer4[1].bn1,
            "resnet.layer4.1.conv2": self.resnet.layer4[1].conv2,
            "resnet.layer4.1.bn2": self.resnet.layer4[1].bn2,
            "resnet.layer4.1.conv3": self.resnet.layer4[1].conv3,
            "resnet.layer4.1.bn3": self.resnet.layer4[1].bn3,
            "resnet.layer4.1.relu": self.resnet.layer4[1].relu,
            "resnet.layer4.2": self.resnet.layer4[2],
            "resnet.layer4.2.conv1": self.resnet.layer4[2].conv1,
            "resnet.layer4.2.bn1": self.resnet.layer4[2].bn1,
            "resnet.layer4.2.conv2": self.resnet.layer4[2].conv2,
            "resnet.layer4.2.bn2": self.resnet.layer4[2].bn2,
            "resnet.layer4.2.conv3": self.resnet.layer4[2].conv3,
            "resnet.layer4.2.bn3": self.resnet.layer4[2].bn3,
            "resnet.layer4.2.relu": self.resnet.layer4[2].relu,
            # "aspp": self.aspp,
            # "aspp.aspp1": self.aspp.aspp1,
            # "aspp.aspp1.aspp_conv": self.aspp.aspp1.aspp_conv,
            # "aspp.aspp1.aspp_conv.0": self.aspp.aspp1.aspp_conv[0],
            # "aspp.aspp1.aspp_conv.1": self.aspp.aspp1.aspp_conv[1],
            # "aspp.aspp1.aspp_conv.2": self.aspp.aspp1.aspp_conv[2],
            # "aspp.aspp2": self.aspp.aspp2,
            # "aspp.aspp2.aspp_conv": self.aspp.aspp2.aspp_conv,
            # "aspp.aspp2.aspp_conv.0": self.aspp.aspp2.aspp_conv[0],
            # "aspp.aspp2.aspp_conv.1": self.aspp.aspp2.aspp_conv[1],
            # "aspp.aspp2.aspp_conv.2": self.aspp.aspp2.aspp_conv[2],
            # "aspp.aspp3": self.aspp.aspp3,
            # "aspp.aspp3.aspp_conv": self.aspp.aspp3.aspp_conv,
            # "aspp.aspp3.aspp_conv.0": self.aspp.aspp3.aspp_conv[0],
            # "aspp.aspp3.aspp_conv.1": self.aspp.aspp3.aspp_conv[1],
            # "aspp.aspp3.aspp_conv.2": self.aspp.aspp3.aspp_conv[2],
            # "aspp.aspp4": self.aspp.aspp4,
            # "aspp.aspp4.aspp_conv": self.aspp.aspp4.aspp_conv,
            # "aspp.aspp4.aspp_conv.0": self.aspp.aspp4.aspp_conv[0],
            # "aspp.aspp4.aspp_conv.1": self.aspp.aspp4.aspp_conv[1],
            # "aspp.aspp4.aspp_conv.2": self.aspp.aspp4.aspp_conv[2],
            # "aspp.aspp_pooling": self.aspp.aspp_pooling,
            # "aspp.aspp_pooling.conv": self.aspp.aspp_pooling.conv,
            # "aspp.aspp_pooling.conv.0": self.aspp.aspp_pooling.conv[0],
            # "aspp.aspp_pooling.conv.1": self.aspp.aspp_pooling.conv[1],
            # "aspp.aspp_pooling.conv.2": self.aspp.aspp_pooling.conv[2],
            # "aspp.conv1": self.aspp.conv1,
            # "aspp.bn1": self.aspp.bn1,
            # "aspp.relu": self.aspp.relu,
            # "aspp.conv2": self.aspp.conv2,
            # "aspp.drop": self.aspp.drop,
        }

        self.Cascade_OPs = None
        self.Basic_OPS = None
        self.add_Cascade_OPs = []

    def construct(self, x):
        """DeepLabV3Plus.construct"""
        size = self.shape(x)
        out, low_level_features = self.resnet(x)
        size2 = self.shape(low_level_features)
        out = self.aspp(out)
        out = P.ResizeNearestNeighbor((size2[2], size2[3]), True)(out)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        out = self.concat((out, low_level_features))
        out = self.last_conv(out)
        out = P.ResizeBilinear((size[2], size[3]), True)(out)
        return out

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self, layer_name, new_layer):
        if 'resnet' == layer_name:
            self.resnet = new_layer
            self.layer_names["resnet"] = new_layer
            self.origin_layer_names["resnet"] = new_layer
        elif 'resnet.conv1' == layer_name:
            self.resnet.conv1 = new_layer
            self.layer_names["resnet.conv1"] = new_layer
            self.origin_layer_names["resnet.conv1"] = new_layer
        elif 'resnet.bn1' == layer_name:
            self.resnet.bn1 = new_layer
            self.layer_names["resnet.bn1"] = new_layer
            self.origin_layer_names["resnet.bn1"] = new_layer
        elif 'resnet.relu' == layer_name:
            self.resnet.relu = new_layer
            self.layer_names["resnet.relu"] = new_layer
            self.origin_layer_names["resnet.relu"] = new_layer
        elif 'resnet.maxpool' == layer_name:
            self.resnet.maxpool = new_layer
            self.layer_names["resnet.maxpool"] = new_layer
            self.origin_layer_names["resnet.maxpool"] = new_layer
        elif 'resnet.layer1' == layer_name:
            self.resnet.layer1 = new_layer
            self.layer_names["resnet.layer1"] = new_layer
            self.origin_layer_names["resnet.layer1"] = new_layer
        elif 'resnet.layer1.0' == layer_name:
            self.resnet.layer1[0] = new_layer
            self.layer_names["resnet.layer1.0"] = new_layer
            self.origin_layer_names["resnet.layer1.0"] = new_layer
        elif 'resnet.layer1.0.conv1' == layer_name:
            self.resnet.layer1[0].conv1 = new_layer
            self.layer_names["resnet.layer1.0.conv1"] = new_layer
            self.origin_layer_names["resnet.layer1.0.conv1"] = new_layer
        elif 'resnet.layer1.0.bn1' == layer_name:
            self.resnet.layer1[0].bn1 = new_layer
            self.layer_names["resnet.layer1.0.bn1"] = new_layer
            self.origin_layer_names["resnet.layer1.0.bn1"] = new_layer
        elif 'resnet.layer1.0.conv2' == layer_name:
            self.resnet.layer1[0].conv2 = new_layer
            self.layer_names["resnet.layer1.0.conv2"] = new_layer
            self.origin_layer_names["resnet.layer1.0.conv2"] = new_layer
        elif 'resnet.layer1.0.bn2' == layer_name:
            self.resnet.layer1[0].bn2 = new_layer
            self.layer_names["resnet.layer1.0.bn2"] = new_layer
            self.origin_layer_names["resnet.layer1.0.bn2"] = new_layer
        elif 'resnet.layer1.0.conv3' == layer_name:
            self.resnet.layer1[0].conv3 = new_layer
            self.layer_names["resnet.layer1.0.conv3"] = new_layer
            self.origin_layer_names["resnet.layer1.0.conv3"] = new_layer
        elif 'resnet.layer1.0.bn3' == layer_name:
            self.resnet.layer1[0].bn3 = new_layer
            self.layer_names["resnet.layer1.0.bn3"] = new_layer
            self.origin_layer_names["resnet.layer1.0.bn3"] = new_layer
        elif 'resnet.layer1.0.relu' == layer_name:
            self.resnet.layer1[0].relu = new_layer
            self.layer_names["resnet.layer1.0.relu"] = new_layer
            self.origin_layer_names["resnet.layer1.0.relu"] = new_layer
        elif 'resnet.layer1.0.downsample' == layer_name:
            self.resnet.layer1[0].downsample = new_layer
            self.layer_names["resnet.layer1.0.downsample"] = new_layer
            self.origin_layer_names["resnet.layer1.0.downsample"] = new_layer
        elif 'resnet.layer1.0.downsample.0' == layer_name:
            self.resnet.layer1[0].downsample[0] = new_layer
            self.layer_names["resnet.layer1.0.downsample.0"] = new_layer
            self.origin_layer_names["resnet.layer1.0.downsample.0"] = new_layer
        elif 'resnet.layer1.0.downsample.1' == layer_name:
            self.resnet.layer1[0].downsample[1] = new_layer
            self.layer_names["resnet.layer1.0.downsample.1"] = new_layer
            self.origin_layer_names["resnet.layer1.0.downsample.1"] = new_layer
        elif 'resnet.layer1.1' == layer_name:
            self.resnet.layer1[1] = new_layer
            self.layer_names["resnet.layer1.1"] = new_layer
            self.origin_layer_names["resnet.layer1.1"] = new_layer
        elif 'resnet.layer1.1.conv1' == layer_name:
            self.resnet.layer1[1].conv1 = new_layer
            self.layer_names["resnet.layer1.1.conv1"] = new_layer
            self.origin_layer_names["resnet.layer1.1.conv1"] = new_layer
        elif 'resnet.layer1.1.bn1' == layer_name:
            self.resnet.layer1[1].bn1 = new_layer
            self.layer_names["resnet.layer1.1.bn1"] = new_layer
            self.origin_layer_names["resnet.layer1.1.bn1"] = new_layer
        elif 'resnet.layer1.1.conv2' == layer_name:
            self.resnet.layer1[1].conv2 = new_layer
            self.layer_names["resnet.layer1.1.conv2"] = new_layer
            self.origin_layer_names["resnet.layer1.1.conv2"] = new_layer
        elif 'resnet.layer1.1.bn2' == layer_name:
            self.resnet.layer1[1].bn2 = new_layer
            self.layer_names["resnet.layer1.1.bn2"] = new_layer
            self.origin_layer_names["resnet.layer1.1.bn2"] = new_layer
        elif 'resnet.layer1.1.conv3' == layer_name:
            self.resnet.layer1[1].conv3 = new_layer
            self.layer_names["resnet.layer1.1.conv3"] = new_layer
            self.origin_layer_names["resnet.layer1.1.conv3"] = new_layer
        elif 'resnet.layer1.1.bn3' == layer_name:
            self.resnet.layer1[1].bn3 = new_layer
            self.layer_names["resnet.layer1.1.bn3"] = new_layer
            self.origin_layer_names["resnet.layer1.1.bn3"] = new_layer
        elif 'resnet.layer1.1.relu' == layer_name:
            self.resnet.layer1[1].relu = new_layer
            self.layer_names["resnet.layer1.1.relu"] = new_layer
            self.origin_layer_names["resnet.layer1.1.relu"] = new_layer
        elif 'resnet.layer1.2' == layer_name:
            self.resnet.layer1[2] = new_layer
            self.layer_names["resnet.layer1.2"] = new_layer
            self.origin_layer_names["resnet.layer1.2"] = new_layer
        elif 'resnet.layer1.2.conv1' == layer_name:
            self.resnet.layer1[2].conv1 = new_layer
            self.layer_names["resnet.layer1.2.conv1"] = new_layer
            self.origin_layer_names["resnet.layer1.2.conv1"] = new_layer
        elif 'resnet.layer1.2.bn1' == layer_name:
            self.resnet.layer1[2].bn1 = new_layer
            self.layer_names["resnet.layer1.2.bn1"] = new_layer
            self.origin_layer_names["resnet.layer1.2.bn1"] = new_layer
        elif 'resnet.layer1.2.conv2' == layer_name:
            self.resnet.layer1[2].conv2 = new_layer
            self.layer_names["resnet.layer1.2.conv2"] = new_layer
            self.origin_layer_names["resnet.layer1.2.conv2"] = new_layer
        elif 'resnet.layer1.2.bn2' == layer_name:
            self.resnet.layer1[2].bn2 = new_layer
            self.layer_names["resnet.layer1.2.bn2"] = new_layer
            self.origin_layer_names["resnet.layer1.2.bn2"] = new_layer
        elif 'resnet.layer1.2.conv3' == layer_name:
            self.resnet.layer1[2].conv3 = new_layer
            self.layer_names["resnet.layer1.2.conv3"] = new_layer
            self.origin_layer_names["resnet.layer1.2.conv3"] = new_layer
        elif 'resnet.layer1.2.bn3' == layer_name:
            self.resnet.layer1[2].bn3 = new_layer
            self.layer_names["resnet.layer1.2.bn3"] = new_layer
            self.origin_layer_names["resnet.layer1.2.bn3"] = new_layer
        elif 'resnet.layer1.2.relu' == layer_name:
            self.resnet.layer1[2].relu = new_layer
            self.layer_names["resnet.layer1.2.relu"] = new_layer
            self.origin_layer_names["resnet.layer1.2.relu"] = new_layer
        elif 'resnet.layer2' == layer_name:
            self.resnet.layer2 = new_layer
            self.layer_names["resnet.layer2"] = new_layer
            self.origin_layer_names["resnet.layer2"] = new_layer
        elif 'resnet.layer2.0' == layer_name:
            self.resnet.layer2[0] = new_layer
            self.layer_names["resnet.layer2.0"] = new_layer
            self.origin_layer_names["resnet.layer2.0"] = new_layer
        elif 'resnet.layer2.0.conv1' == layer_name:
            self.resnet.layer2[0].conv1 = new_layer
            self.layer_names["resnet.layer2.0.conv1"] = new_layer
            self.origin_layer_names["resnet.layer2.0.conv1"] = new_layer
        elif 'resnet.layer2.0.bn1' == layer_name:
            self.resnet.layer2[0].bn1 = new_layer
            self.layer_names["resnet.layer2.0.bn1"] = new_layer
            self.origin_layer_names["resnet.layer2.0.bn1"] = new_layer
        elif 'resnet.layer2.0.conv2' == layer_name:
            self.resnet.layer2[0].conv2 = new_layer
            self.layer_names["resnet.layer2.0.conv2"] = new_layer
            self.origin_layer_names["resnet.layer2.0.conv2"] = new_layer
        elif 'resnet.layer2.0.bn2' == layer_name:
            self.resnet.layer2[0].bn2 = new_layer
            self.layer_names["resnet.layer2.0.bn2"] = new_layer
            self.origin_layer_names["resnet.layer2.0.bn2"] = new_layer
        elif 'resnet.layer2.0.conv3' == layer_name:
            self.resnet.layer2[0].conv3 = new_layer
            self.layer_names["resnet.layer2.0.conv3"] = new_layer
            self.origin_layer_names["resnet.layer2.0.conv3"] = new_layer
        elif 'resnet.layer2.0.bn3' == layer_name:
            self.resnet.layer2[0].bn3 = new_layer
            self.layer_names["resnet.layer2.0.bn3"] = new_layer
            self.origin_layer_names["resnet.layer2.0.bn3"] = new_layer
        elif 'resnet.layer2.0.relu' == layer_name:
            self.resnet.layer2[0].relu = new_layer
            self.layer_names["resnet.layer2.0.relu"] = new_layer
            self.origin_layer_names["resnet.layer2.0.relu"] = new_layer
        elif 'resnet.layer2.0.downsample' == layer_name:
            self.resnet.layer2[0].downsample = new_layer
            self.layer_names["resnet.layer2.0.downsample"] = new_layer
            self.origin_layer_names["resnet.layer2.0.downsample"] = new_layer
        elif 'resnet.layer2.0.downsample.0' == layer_name:
            self.resnet.layer2[0].downsample[0] = new_layer
            self.layer_names["resnet.layer2.0.downsample.0"] = new_layer
            self.origin_layer_names["resnet.layer2.0.downsample.0"] = new_layer
        elif 'resnet.layer2.0.downsample.1' == layer_name:
            self.resnet.layer2[0].downsample[1] = new_layer
            self.layer_names["resnet.layer2.0.downsample.1"] = new_layer
            self.origin_layer_names["resnet.layer2.0.downsample.1"] = new_layer
        elif 'resnet.layer2.1' == layer_name:
            self.resnet.layer2[1] = new_layer
            self.layer_names["resnet.layer2.1"] = new_layer
            self.origin_layer_names["resnet.layer2.1"] = new_layer
        elif 'resnet.layer2.1.conv1' == layer_name:
            self.resnet.layer2[1].conv1 = new_layer
            self.layer_names["resnet.layer2.1.conv1"] = new_layer
            self.origin_layer_names["resnet.layer2.1.conv1"] = new_layer
        elif 'resnet.layer2.1.bn1' == layer_name:
            self.resnet.layer2[1].bn1 = new_layer
            self.layer_names["resnet.layer2.1.bn1"] = new_layer
            self.origin_layer_names["resnet.layer2.1.bn1"] = new_layer
        elif 'resnet.layer2.1.conv2' == layer_name:
            self.resnet.layer2[1].conv2 = new_layer
            self.layer_names["resnet.layer2.1.conv2"] = new_layer
            self.origin_layer_names["resnet.layer2.1.conv2"] = new_layer
        elif 'resnet.layer2.1.bn2' == layer_name:
            self.resnet.layer2[1].bn2 = new_layer
            self.layer_names["resnet.layer2.1.bn2"] = new_layer
            self.origin_layer_names["resnet.layer2.1.bn2"] = new_layer
        elif 'resnet.layer2.1.conv3' == layer_name:
            self.resnet.layer2[1].conv3 = new_layer
            self.layer_names["resnet.layer2.1.conv3"] = new_layer
            self.origin_layer_names["resnet.layer2.1.conv3"] = new_layer
        elif 'resnet.layer2.1.bn3' == layer_name:
            self.resnet.layer2[1].bn3 = new_layer
            self.layer_names["resnet.layer2.1.bn3"] = new_layer
            self.origin_layer_names["resnet.layer2.1.bn3"] = new_layer
        elif 'resnet.layer2.1.relu' == layer_name:
            self.resnet.layer2[1].relu = new_layer
            self.layer_names["resnet.layer2.1.relu"] = new_layer
            self.origin_layer_names["resnet.layer2.1.relu"] = new_layer
        elif 'resnet.layer2.2' == layer_name:
            self.resnet.layer2[2] = new_layer
            self.layer_names["resnet.layer2.2"] = new_layer
            self.origin_layer_names["resnet.layer2.2"] = new_layer
        elif 'resnet.layer2.2.conv1' == layer_name:
            self.resnet.layer2[2].conv1 = new_layer
            self.layer_names["resnet.layer2.2.conv1"] = new_layer
            self.origin_layer_names["resnet.layer2.2.conv1"] = new_layer
        elif 'resnet.layer2.2.bn1' == layer_name:
            self.resnet.layer2[2].bn1 = new_layer
            self.layer_names["resnet.layer2.2.bn1"] = new_layer
            self.origin_layer_names["resnet.layer2.2.bn1"] = new_layer
        elif 'resnet.layer2.2.conv2' == layer_name:
            self.resnet.layer2[2].conv2 = new_layer
            self.layer_names["resnet.layer2.2.conv2"] = new_layer
            self.origin_layer_names["resnet.layer2.2.conv2"] = new_layer
        elif 'resnet.layer2.2.bn2' == layer_name:
            self.resnet.layer2[2].bn2 = new_layer
            self.layer_names["resnet.layer2.2.bn2"] = new_layer
            self.origin_layer_names["resnet.layer2.2.bn2"] = new_layer
        elif 'resnet.layer2.2.conv3' == layer_name:
            self.resnet.layer2[2].conv3 = new_layer
            self.layer_names["resnet.layer2.2.conv3"] = new_layer
            self.origin_layer_names["resnet.layer2.2.conv3"] = new_layer
        elif 'resnet.layer2.2.bn3' == layer_name:
            self.resnet.layer2[2].bn3 = new_layer
            self.layer_names["resnet.layer2.2.bn3"] = new_layer
            self.origin_layer_names["resnet.layer2.2.bn3"] = new_layer
        elif 'resnet.layer2.2.relu' == layer_name:
            self.resnet.layer2[2].relu = new_layer
            self.layer_names["resnet.layer2.2.relu"] = new_layer
            self.origin_layer_names["resnet.layer2.2.relu"] = new_layer
        elif 'resnet.layer2.3' == layer_name:
            self.resnet.layer2[3] = new_layer
            self.layer_names["resnet.layer2.3"] = new_layer
            self.origin_layer_names["resnet.layer2.3"] = new_layer
        elif 'resnet.layer2.3.conv1' == layer_name:
            self.resnet.layer2[3].conv1 = new_layer
            self.layer_names["resnet.layer2.3.conv1"] = new_layer
            self.origin_layer_names["resnet.layer2.3.conv1"] = new_layer
        elif 'resnet.layer2.3.bn1' == layer_name:
            self.resnet.layer2[3].bn1 = new_layer
            self.layer_names["resnet.layer2.3.bn1"] = new_layer
            self.origin_layer_names["resnet.layer2.3.bn1"] = new_layer
        elif 'resnet.layer2.3.conv2' == layer_name:
            self.resnet.layer2[3].conv2 = new_layer
            self.layer_names["resnet.layer2.3.conv2"] = new_layer
            self.origin_layer_names["resnet.layer2.3.conv2"] = new_layer
        elif 'resnet.layer2.3.bn2' == layer_name:
            self.resnet.layer2[3].bn2 = new_layer
            self.layer_names["resnet.layer2.3.bn2"] = new_layer
            self.origin_layer_names["resnet.layer2.3.bn2"] = new_layer
        elif 'resnet.layer2.3.conv3' == layer_name:
            self.resnet.layer2[3].conv3 = new_layer
            self.layer_names["resnet.layer2.3.conv3"] = new_layer
            self.origin_layer_names["resnet.layer2.3.conv3"] = new_layer
        elif 'resnet.layer2.3.bn3' == layer_name:
            self.resnet.layer2[3].bn3 = new_layer
            self.layer_names["resnet.layer2.3.bn3"] = new_layer
            self.origin_layer_names["resnet.layer2.3.bn3"] = new_layer
        elif 'resnet.layer2.3.relu' == layer_name:
            self.resnet.layer2[3].relu = new_layer
            self.layer_names["resnet.layer2.3.relu"] = new_layer
            self.origin_layer_names["resnet.layer2.3.relu"] = new_layer
        elif 'resnet.layer3' == layer_name:
            self.resnet.layer3 = new_layer
            self.layer_names["resnet.layer3"] = new_layer
            self.origin_layer_names["resnet.layer3"] = new_layer
        elif 'resnet.layer3.0' == layer_name:
            self.resnet.layer3[0] = new_layer
            self.layer_names["resnet.layer3.0"] = new_layer
            self.origin_layer_names["resnet.layer3.0"] = new_layer
        elif 'resnet.layer3.0.conv1' == layer_name:
            self.resnet.layer3[0].conv1 = new_layer
            self.layer_names["resnet.layer3.0.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.0.conv1"] = new_layer
        elif 'resnet.layer3.0.bn1' == layer_name:
            self.resnet.layer3[0].bn1 = new_layer
            self.layer_names["resnet.layer3.0.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.0.bn1"] = new_layer
        elif 'resnet.layer3.0.conv2' == layer_name:
            self.resnet.layer3[0].conv2 = new_layer
            self.layer_names["resnet.layer3.0.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.0.conv2"] = new_layer
        elif 'resnet.layer3.0.bn2' == layer_name:
            self.resnet.layer3[0].bn2 = new_layer
            self.layer_names["resnet.layer3.0.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.0.bn2"] = new_layer
        elif 'resnet.layer3.0.conv3' == layer_name:
            self.resnet.layer3[0].conv3 = new_layer
            self.layer_names["resnet.layer3.0.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.0.conv3"] = new_layer
        elif 'resnet.layer3.0.bn3' == layer_name:
            self.resnet.layer3[0].bn3 = new_layer
            self.layer_names["resnet.layer3.0.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.0.bn3"] = new_layer
        elif 'resnet.layer3.0.relu' == layer_name:
            self.resnet.layer3[0].relu = new_layer
            self.layer_names["resnet.layer3.0.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.0.relu"] = new_layer
        elif 'resnet.layer3.0.downsample' == layer_name:
            self.resnet.layer3[0].downsample = new_layer
            self.layer_names["resnet.layer3.0.downsample"] = new_layer
            self.origin_layer_names["resnet.layer3.0.downsample"] = new_layer
        elif 'resnet.layer3.0.downsample.0' == layer_name:
            self.resnet.layer3[0].downsample[0] = new_layer
            self.layer_names["resnet.layer3.0.downsample.0"] = new_layer
            self.origin_layer_names["resnet.layer3.0.downsample.0"] = new_layer
        elif 'resnet.layer3.0.downsample.1' == layer_name:
            self.resnet.layer3[0].downsample[1] = new_layer
            self.layer_names["resnet.layer3.0.downsample.1"] = new_layer
            self.origin_layer_names["resnet.layer3.0.downsample.1"] = new_layer
        elif 'resnet.layer3.1' == layer_name:
            self.resnet.layer3[1] = new_layer
            self.layer_names["resnet.layer3.1"] = new_layer
            self.origin_layer_names["resnet.layer3.1"] = new_layer
        elif 'resnet.layer3.1.conv1' == layer_name:
            self.resnet.layer3[1].conv1 = new_layer
            self.layer_names["resnet.layer3.1.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.1.conv1"] = new_layer
        elif 'resnet.layer3.1.bn1' == layer_name:
            self.resnet.layer3[1].bn1 = new_layer
            self.layer_names["resnet.layer3.1.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.1.bn1"] = new_layer
        elif 'resnet.layer3.1.conv2' == layer_name:
            self.resnet.layer3[1].conv2 = new_layer
            self.layer_names["resnet.layer3.1.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.1.conv2"] = new_layer
        elif 'resnet.layer3.1.bn2' == layer_name:
            self.resnet.layer3[1].bn2 = new_layer
            self.layer_names["resnet.layer3.1.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.1.bn2"] = new_layer
        elif 'resnet.layer3.1.conv3' == layer_name:
            self.resnet.layer3[1].conv3 = new_layer
            self.layer_names["resnet.layer3.1.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.1.conv3"] = new_layer
        elif 'resnet.layer3.1.bn3' == layer_name:
            self.resnet.layer3[1].bn3 = new_layer
            self.layer_names["resnet.layer3.1.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.1.bn3"] = new_layer
        elif 'resnet.layer3.1.relu' == layer_name:
            self.resnet.layer3[1].relu = new_layer
            self.layer_names["resnet.layer3.1.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.1.relu"] = new_layer
        elif 'resnet.layer3.2' == layer_name:
            self.resnet.layer3[2] = new_layer
            self.layer_names["resnet.layer3.2"] = new_layer
            self.origin_layer_names["resnet.layer3.2"] = new_layer
        elif 'resnet.layer3.2.conv1' == layer_name:
            self.resnet.layer3[2].conv1 = new_layer
            self.layer_names["resnet.layer3.2.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.2.conv1"] = new_layer
        elif 'resnet.layer3.2.bn1' == layer_name:
            self.resnet.layer3[2].bn1 = new_layer
            self.layer_names["resnet.layer3.2.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.2.bn1"] = new_layer
        elif 'resnet.layer3.2.conv2' == layer_name:
            self.resnet.layer3[2].conv2 = new_layer
            self.layer_names["resnet.layer3.2.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.2.conv2"] = new_layer
        elif 'resnet.layer3.2.bn2' == layer_name:
            self.resnet.layer3[2].bn2 = new_layer
            self.layer_names["resnet.layer3.2.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.2.bn2"] = new_layer
        elif 'resnet.layer3.2.conv3' == layer_name:
            self.resnet.layer3[2].conv3 = new_layer
            self.layer_names["resnet.layer3.2.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.2.conv3"] = new_layer
        elif 'resnet.layer3.2.bn3' == layer_name:
            self.resnet.layer3[2].bn3 = new_layer
            self.layer_names["resnet.layer3.2.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.2.bn3"] = new_layer
        elif 'resnet.layer3.2.relu' == layer_name:
            self.resnet.layer3[2].relu = new_layer
            self.layer_names["resnet.layer3.2.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.2.relu"] = new_layer
        elif 'resnet.layer3.3' == layer_name:
            self.resnet.layer3[3] = new_layer
            self.layer_names["resnet.layer3.3"] = new_layer
            self.origin_layer_names["resnet.layer3.3"] = new_layer
        elif 'resnet.layer3.3.conv1' == layer_name:
            self.resnet.layer3[3].conv1 = new_layer
            self.layer_names["resnet.layer3.3.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.3.conv1"] = new_layer
        elif 'resnet.layer3.3.bn1' == layer_name:
            self.resnet.layer3[3].bn1 = new_layer
            self.layer_names["resnet.layer3.3.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.3.bn1"] = new_layer
        elif 'resnet.layer3.3.conv2' == layer_name:
            self.resnet.layer3[3].conv2 = new_layer
            self.layer_names["resnet.layer3.3.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.3.conv2"] = new_layer
        elif 'resnet.layer3.3.bn2' == layer_name:
            self.resnet.layer3[3].bn2 = new_layer
            self.layer_names["resnet.layer3.3.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.3.bn2"] = new_layer
        elif 'resnet.layer3.3.conv3' == layer_name:
            self.resnet.layer3[3].conv3 = new_layer
            self.layer_names["resnet.layer3.3.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.3.conv3"] = new_layer
        elif 'resnet.layer3.3.bn3' == layer_name:
            self.resnet.layer3[3].bn3 = new_layer
            self.layer_names["resnet.layer3.3.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.3.bn3"] = new_layer
        elif 'resnet.layer3.3.relu' == layer_name:
            self.resnet.layer3[3].relu = new_layer
            self.layer_names["resnet.layer3.3.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.3.relu"] = new_layer
        elif 'resnet.layer3.4' == layer_name:
            self.resnet.layer3[4] = new_layer
            self.layer_names["resnet.layer3.4"] = new_layer
            self.origin_layer_names["resnet.layer3.4"] = new_layer
        elif 'resnet.layer3.4.conv1' == layer_name:
            self.resnet.layer3[4].conv1 = new_layer
            self.layer_names["resnet.layer3.4.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.4.conv1"] = new_layer
        elif 'resnet.layer3.4.bn1' == layer_name:
            self.resnet.layer3[4].bn1 = new_layer
            self.layer_names["resnet.layer3.4.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.4.bn1"] = new_layer
        elif 'resnet.layer3.4.conv2' == layer_name:
            self.resnet.layer3[4].conv2 = new_layer
            self.layer_names["resnet.layer3.4.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.4.conv2"] = new_layer
        elif 'resnet.layer3.4.bn2' == layer_name:
            self.resnet.layer3[4].bn2 = new_layer
            self.layer_names["resnet.layer3.4.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.4.bn2"] = new_layer
        elif 'resnet.layer3.4.conv3' == layer_name:
            self.resnet.layer3[4].conv3 = new_layer
            self.layer_names["resnet.layer3.4.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.4.conv3"] = new_layer
        elif 'resnet.layer3.4.bn3' == layer_name:
            self.resnet.layer3[4].bn3 = new_layer
            self.layer_names["resnet.layer3.4.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.4.bn3"] = new_layer
        elif 'resnet.layer3.4.relu' == layer_name:
            self.resnet.layer3[4].relu = new_layer
            self.layer_names["resnet.layer3.4.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.4.relu"] = new_layer
        elif 'resnet.layer3.5' == layer_name:
            self.resnet.layer3[5] = new_layer
            self.layer_names["resnet.layer3.5"] = new_layer
            self.origin_layer_names["resnet.layer3.5"] = new_layer
        elif 'resnet.layer3.5.conv1' == layer_name:
            self.resnet.layer3[5].conv1 = new_layer
            self.layer_names["resnet.layer3.5.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.5.conv1"] = new_layer
        elif 'resnet.layer3.5.bn1' == layer_name:
            self.resnet.layer3[5].bn1 = new_layer
            self.layer_names["resnet.layer3.5.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.5.bn1"] = new_layer
        elif 'resnet.layer3.5.conv2' == layer_name:
            self.resnet.layer3[5].conv2 = new_layer
            self.layer_names["resnet.layer3.5.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.5.conv2"] = new_layer
        elif 'resnet.layer3.5.bn2' == layer_name:
            self.resnet.layer3[5].bn2 = new_layer
            self.layer_names["resnet.layer3.5.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.5.bn2"] = new_layer
        elif 'resnet.layer3.5.conv3' == layer_name:
            self.resnet.layer3[5].conv3 = new_layer
            self.layer_names["resnet.layer3.5.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.5.conv3"] = new_layer
        elif 'resnet.layer3.5.bn3' == layer_name:
            self.resnet.layer3[5].bn3 = new_layer
            self.layer_names["resnet.layer3.5.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.5.bn3"] = new_layer
        elif 'resnet.layer3.5.relu' == layer_name:
            self.resnet.layer3[5].relu = new_layer
            self.layer_names["resnet.layer3.5.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.5.relu"] = new_layer
        elif 'resnet.layer3.6' == layer_name:
            self.resnet.layer3[6] = new_layer
            self.layer_names["resnet.layer3.6"] = new_layer
            self.origin_layer_names["resnet.layer3.6"] = new_layer
        elif 'resnet.layer3.6.conv1' == layer_name:
            self.resnet.layer3[6].conv1 = new_layer
            self.layer_names["resnet.layer3.6.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.6.conv1"] = new_layer
        elif 'resnet.layer3.6.bn1' == layer_name:
            self.resnet.layer3[6].bn1 = new_layer
            self.layer_names["resnet.layer3.6.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.6.bn1"] = new_layer
        elif 'resnet.layer3.6.conv2' == layer_name:
            self.resnet.layer3[6].conv2 = new_layer
            self.layer_names["resnet.layer3.6.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.6.conv2"] = new_layer
        elif 'resnet.layer3.6.bn2' == layer_name:
            self.resnet.layer3[6].bn2 = new_layer
            self.layer_names["resnet.layer3.6.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.6.bn2"] = new_layer
        elif 'resnet.layer3.6.conv3' == layer_name:
            self.resnet.layer3[6].conv3 = new_layer
            self.layer_names["resnet.layer3.6.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.6.conv3"] = new_layer
        elif 'resnet.layer3.6.bn3' == layer_name:
            self.resnet.layer3[6].bn3 = new_layer
            self.layer_names["resnet.layer3.6.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.6.bn3"] = new_layer
        elif 'resnet.layer3.6.relu' == layer_name:
            self.resnet.layer3[6].relu = new_layer
            self.layer_names["resnet.layer3.6.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.6.relu"] = new_layer
        elif 'resnet.layer3.7' == layer_name:
            self.resnet.layer3[7] = new_layer
            self.layer_names["resnet.layer3.7"] = new_layer
            self.origin_layer_names["resnet.layer3.7"] = new_layer
        elif 'resnet.layer3.7.conv1' == layer_name:
            self.resnet.layer3[7].conv1 = new_layer
            self.layer_names["resnet.layer3.7.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.7.conv1"] = new_layer
        elif 'resnet.layer3.7.bn1' == layer_name:
            self.resnet.layer3[7].bn1 = new_layer
            self.layer_names["resnet.layer3.7.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.7.bn1"] = new_layer
        elif 'resnet.layer3.7.conv2' == layer_name:
            self.resnet.layer3[7].conv2 = new_layer
            self.layer_names["resnet.layer3.7.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.7.conv2"] = new_layer
        elif 'resnet.layer3.7.bn2' == layer_name:
            self.resnet.layer3[7].bn2 = new_layer
            self.layer_names["resnet.layer3.7.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.7.bn2"] = new_layer
        elif 'resnet.layer3.7.conv3' == layer_name:
            self.resnet.layer3[7].conv3 = new_layer
            self.layer_names["resnet.layer3.7.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.7.conv3"] = new_layer
        elif 'resnet.layer3.7.bn3' == layer_name:
            self.resnet.layer3[7].bn3 = new_layer
            self.layer_names["resnet.layer3.7.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.7.bn3"] = new_layer
        elif 'resnet.layer3.7.relu' == layer_name:
            self.resnet.layer3[7].relu = new_layer
            self.layer_names["resnet.layer3.7.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.7.relu"] = new_layer
        elif 'resnet.layer3.8' == layer_name:
            self.resnet.layer3[8] = new_layer
            self.layer_names["resnet.layer3.8"] = new_layer
            self.origin_layer_names["resnet.layer3.8"] = new_layer
        elif 'resnet.layer3.8.conv1' == layer_name:
            self.resnet.layer3[8].conv1 = new_layer
            self.layer_names["resnet.layer3.8.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.8.conv1"] = new_layer
        elif 'resnet.layer3.8.bn1' == layer_name:
            self.resnet.layer3[8].bn1 = new_layer
            self.layer_names["resnet.layer3.8.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.8.bn1"] = new_layer
        elif 'resnet.layer3.8.conv2' == layer_name:
            self.resnet.layer3[8].conv2 = new_layer
            self.layer_names["resnet.layer3.8.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.8.conv2"] = new_layer
        elif 'resnet.layer3.8.bn2' == layer_name:
            self.resnet.layer3[8].bn2 = new_layer
            self.layer_names["resnet.layer3.8.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.8.bn2"] = new_layer
        elif 'resnet.layer3.8.conv3' == layer_name:
            self.resnet.layer3[8].conv3 = new_layer
            self.layer_names["resnet.layer3.8.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.8.conv3"] = new_layer
        elif 'resnet.layer3.8.bn3' == layer_name:
            self.resnet.layer3[8].bn3 = new_layer
            self.layer_names["resnet.layer3.8.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.8.bn3"] = new_layer
        elif 'resnet.layer3.8.relu' == layer_name:
            self.resnet.layer3[8].relu = new_layer
            self.layer_names["resnet.layer3.8.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.8.relu"] = new_layer
        elif 'resnet.layer3.9' == layer_name:
            self.resnet.layer3[9] = new_layer
            self.layer_names["resnet.layer3.9"] = new_layer
            self.origin_layer_names["resnet.layer3.9"] = new_layer
        elif 'resnet.layer3.9.conv1' == layer_name:
            self.resnet.layer3[9].conv1 = new_layer
            self.layer_names["resnet.layer3.9.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.9.conv1"] = new_layer
        elif 'resnet.layer3.9.bn1' == layer_name:
            self.resnet.layer3[9].bn1 = new_layer
            self.layer_names["resnet.layer3.9.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.9.bn1"] = new_layer
        elif 'resnet.layer3.9.conv2' == layer_name:
            self.resnet.layer3[9].conv2 = new_layer
            self.layer_names["resnet.layer3.9.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.9.conv2"] = new_layer
        elif 'resnet.layer3.9.bn2' == layer_name:
            self.resnet.layer3[9].bn2 = new_layer
            self.layer_names["resnet.layer3.9.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.9.bn2"] = new_layer
        elif 'resnet.layer3.9.conv3' == layer_name:
            self.resnet.layer3[9].conv3 = new_layer
            self.layer_names["resnet.layer3.9.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.9.conv3"] = new_layer
        elif 'resnet.layer3.9.bn3' == layer_name:
            self.resnet.layer3[9].bn3 = new_layer
            self.layer_names["resnet.layer3.9.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.9.bn3"] = new_layer
        elif 'resnet.layer3.9.relu' == layer_name:
            self.resnet.layer3[9].relu = new_layer
            self.layer_names["resnet.layer3.9.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.9.relu"] = new_layer
        elif 'resnet.layer3.10' == layer_name:
            self.resnet.layer3[10] = new_layer
            self.layer_names["resnet.layer3.10"] = new_layer
            self.origin_layer_names["resnet.layer3.10"] = new_layer
        elif 'resnet.layer3.10.conv1' == layer_name:
            self.resnet.layer3[10].conv1 = new_layer
            self.layer_names["resnet.layer3.10.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.10.conv1"] = new_layer
        elif 'resnet.layer3.10.bn1' == layer_name:
            self.resnet.layer3[10].bn1 = new_layer
            self.layer_names["resnet.layer3.10.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.10.bn1"] = new_layer
        elif 'resnet.layer3.10.conv2' == layer_name:
            self.resnet.layer3[10].conv2 = new_layer
            self.layer_names["resnet.layer3.10.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.10.conv2"] = new_layer
        elif 'resnet.layer3.10.bn2' == layer_name:
            self.resnet.layer3[10].bn2 = new_layer
            self.layer_names["resnet.layer3.10.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.10.bn2"] = new_layer
        elif 'resnet.layer3.10.conv3' == layer_name:
            self.resnet.layer3[10].conv3 = new_layer
            self.layer_names["resnet.layer3.10.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.10.conv3"] = new_layer
        elif 'resnet.layer3.10.bn3' == layer_name:
            self.resnet.layer3[10].bn3 = new_layer
            self.layer_names["resnet.layer3.10.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.10.bn3"] = new_layer
        elif 'resnet.layer3.10.relu' == layer_name:
            self.resnet.layer3[10].relu = new_layer
            self.layer_names["resnet.layer3.10.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.10.relu"] = new_layer
        elif 'resnet.layer3.11' == layer_name:
            self.resnet.layer3[11] = new_layer
            self.layer_names["resnet.layer3.11"] = new_layer
            self.origin_layer_names["resnet.layer3.11"] = new_layer
        elif 'resnet.layer3.11.conv1' == layer_name:
            self.resnet.layer3[11].conv1 = new_layer
            self.layer_names["resnet.layer3.11.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.11.conv1"] = new_layer
        elif 'resnet.layer3.11.bn1' == layer_name:
            self.resnet.layer3[11].bn1 = new_layer
            self.layer_names["resnet.layer3.11.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.11.bn1"] = new_layer
        elif 'resnet.layer3.11.conv2' == layer_name:
            self.resnet.layer3[11].conv2 = new_layer
            self.layer_names["resnet.layer3.11.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.11.conv2"] = new_layer
        elif 'resnet.layer3.11.bn2' == layer_name:
            self.resnet.layer3[11].bn2 = new_layer
            self.layer_names["resnet.layer3.11.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.11.bn2"] = new_layer
        elif 'resnet.layer3.11.conv3' == layer_name:
            self.resnet.layer3[11].conv3 = new_layer
            self.layer_names["resnet.layer3.11.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.11.conv3"] = new_layer
        elif 'resnet.layer3.11.bn3' == layer_name:
            self.resnet.layer3[11].bn3 = new_layer
            self.layer_names["resnet.layer3.11.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.11.bn3"] = new_layer
        elif 'resnet.layer3.11.relu' == layer_name:
            self.resnet.layer3[11].relu = new_layer
            self.layer_names["resnet.layer3.11.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.11.relu"] = new_layer
        elif 'resnet.layer3.12' == layer_name:
            self.resnet.layer3[12] = new_layer
            self.layer_names["resnet.layer3.12"] = new_layer
            self.origin_layer_names["resnet.layer3.12"] = new_layer
        elif 'resnet.layer3.12.conv1' == layer_name:
            self.resnet.layer3[12].conv1 = new_layer
            self.layer_names["resnet.layer3.12.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.12.conv1"] = new_layer
        elif 'resnet.layer3.12.bn1' == layer_name:
            self.resnet.layer3[12].bn1 = new_layer
            self.layer_names["resnet.layer3.12.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.12.bn1"] = new_layer
        elif 'resnet.layer3.12.conv2' == layer_name:
            self.resnet.layer3[12].conv2 = new_layer
            self.layer_names["resnet.layer3.12.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.12.conv2"] = new_layer
        elif 'resnet.layer3.12.bn2' == layer_name:
            self.resnet.layer3[12].bn2 = new_layer
            self.layer_names["resnet.layer3.12.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.12.bn2"] = new_layer
        elif 'resnet.layer3.12.conv3' == layer_name:
            self.resnet.layer3[12].conv3 = new_layer
            self.layer_names["resnet.layer3.12.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.12.conv3"] = new_layer
        elif 'resnet.layer3.12.bn3' == layer_name:
            self.resnet.layer3[12].bn3 = new_layer
            self.layer_names["resnet.layer3.12.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.12.bn3"] = new_layer
        elif 'resnet.layer3.12.relu' == layer_name:
            self.resnet.layer3[12].relu = new_layer
            self.layer_names["resnet.layer3.12.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.12.relu"] = new_layer
        elif 'resnet.layer3.13' == layer_name:
            self.resnet.layer3[13] = new_layer
            self.layer_names["resnet.layer3.13"] = new_layer
            self.origin_layer_names["resnet.layer3.13"] = new_layer
        elif 'resnet.layer3.13.conv1' == layer_name:
            self.resnet.layer3[13].conv1 = new_layer
            self.layer_names["resnet.layer3.13.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.13.conv1"] = new_layer
        elif 'resnet.layer3.13.bn1' == layer_name:
            self.resnet.layer3[13].bn1 = new_layer
            self.layer_names["resnet.layer3.13.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.13.bn1"] = new_layer
        elif 'resnet.layer3.13.conv2' == layer_name:
            self.resnet.layer3[13].conv2 = new_layer
            self.layer_names["resnet.layer3.13.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.13.conv2"] = new_layer
        elif 'resnet.layer3.13.bn2' == layer_name:
            self.resnet.layer3[13].bn2 = new_layer
            self.layer_names["resnet.layer3.13.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.13.bn2"] = new_layer
        elif 'resnet.layer3.13.conv3' == layer_name:
            self.resnet.layer3[13].conv3 = new_layer
            self.layer_names["resnet.layer3.13.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.13.conv3"] = new_layer
        elif 'resnet.layer3.13.bn3' == layer_name:
            self.resnet.layer3[13].bn3 = new_layer
            self.layer_names["resnet.layer3.13.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.13.bn3"] = new_layer
        elif 'resnet.layer3.13.relu' == layer_name:
            self.resnet.layer3[13].relu = new_layer
            self.layer_names["resnet.layer3.13.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.13.relu"] = new_layer
        elif 'resnet.layer3.14' == layer_name:
            self.resnet.layer3[14] = new_layer
            self.layer_names["resnet.layer3.14"] = new_layer
            self.origin_layer_names["resnet.layer3.14"] = new_layer
        elif 'resnet.layer3.14.conv1' == layer_name:
            self.resnet.layer3[14].conv1 = new_layer
            self.layer_names["resnet.layer3.14.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.14.conv1"] = new_layer
        elif 'resnet.layer3.14.bn1' == layer_name:
            self.resnet.layer3[14].bn1 = new_layer
            self.layer_names["resnet.layer3.14.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.14.bn1"] = new_layer
        elif 'resnet.layer3.14.conv2' == layer_name:
            self.resnet.layer3[14].conv2 = new_layer
            self.layer_names["resnet.layer3.14.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.14.conv2"] = new_layer
        elif 'resnet.layer3.14.bn2' == layer_name:
            self.resnet.layer3[14].bn2 = new_layer
            self.layer_names["resnet.layer3.14.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.14.bn2"] = new_layer
        elif 'resnet.layer3.14.conv3' == layer_name:
            self.resnet.layer3[14].conv3 = new_layer
            self.layer_names["resnet.layer3.14.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.14.conv3"] = new_layer
        elif 'resnet.layer3.14.bn3' == layer_name:
            self.resnet.layer3[14].bn3 = new_layer
            self.layer_names["resnet.layer3.14.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.14.bn3"] = new_layer
        elif 'resnet.layer3.14.relu' == layer_name:
            self.resnet.layer3[14].relu = new_layer
            self.layer_names["resnet.layer3.14.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.14.relu"] = new_layer
        elif 'resnet.layer3.15' == layer_name:
            self.resnet.layer3[15] = new_layer
            self.layer_names["resnet.layer3.15"] = new_layer
            self.origin_layer_names["resnet.layer3.15"] = new_layer
        elif 'resnet.layer3.15.conv1' == layer_name:
            self.resnet.layer3[15].conv1 = new_layer
            self.layer_names["resnet.layer3.15.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.15.conv1"] = new_layer
        elif 'resnet.layer3.15.bn1' == layer_name:
            self.resnet.layer3[15].bn1 = new_layer
            self.layer_names["resnet.layer3.15.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.15.bn1"] = new_layer
        elif 'resnet.layer3.15.conv2' == layer_name:
            self.resnet.layer3[15].conv2 = new_layer
            self.layer_names["resnet.layer3.15.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.15.conv2"] = new_layer
        elif 'resnet.layer3.15.bn2' == layer_name:
            self.resnet.layer3[15].bn2 = new_layer
            self.layer_names["resnet.layer3.15.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.15.bn2"] = new_layer
        elif 'resnet.layer3.15.conv3' == layer_name:
            self.resnet.layer3[15].conv3 = new_layer
            self.layer_names["resnet.layer3.15.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.15.conv3"] = new_layer
        elif 'resnet.layer3.15.bn3' == layer_name:
            self.resnet.layer3[15].bn3 = new_layer
            self.layer_names["resnet.layer3.15.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.15.bn3"] = new_layer
        elif 'resnet.layer3.15.relu' == layer_name:
            self.resnet.layer3[15].relu = new_layer
            self.layer_names["resnet.layer3.15.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.15.relu"] = new_layer
        elif 'resnet.layer3.16' == layer_name:
            self.resnet.layer3[16] = new_layer
            self.layer_names["resnet.layer3.16"] = new_layer
            self.origin_layer_names["resnet.layer3.16"] = new_layer
        elif 'resnet.layer3.16.conv1' == layer_name:
            self.resnet.layer3[16].conv1 = new_layer
            self.layer_names["resnet.layer3.16.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.16.conv1"] = new_layer
        elif 'resnet.layer3.16.bn1' == layer_name:
            self.resnet.layer3[16].bn1 = new_layer
            self.layer_names["resnet.layer3.16.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.16.bn1"] = new_layer
        elif 'resnet.layer3.16.conv2' == layer_name:
            self.resnet.layer3[16].conv2 = new_layer
            self.layer_names["resnet.layer3.16.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.16.conv2"] = new_layer
        elif 'resnet.layer3.16.bn2' == layer_name:
            self.resnet.layer3[16].bn2 = new_layer
            self.layer_names["resnet.layer3.16.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.16.bn2"] = new_layer
        elif 'resnet.layer3.16.conv3' == layer_name:
            self.resnet.layer3[16].conv3 = new_layer
            self.layer_names["resnet.layer3.16.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.16.conv3"] = new_layer
        elif 'resnet.layer3.16.bn3' == layer_name:
            self.resnet.layer3[16].bn3 = new_layer
            self.layer_names["resnet.layer3.16.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.16.bn3"] = new_layer
        elif 'resnet.layer3.16.relu' == layer_name:
            self.resnet.layer3[16].relu = new_layer
            self.layer_names["resnet.layer3.16.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.16.relu"] = new_layer
        elif 'resnet.layer3.17' == layer_name:
            self.resnet.layer3[17] = new_layer
            self.layer_names["resnet.layer3.17"] = new_layer
            self.origin_layer_names["resnet.layer3.17"] = new_layer
        elif 'resnet.layer3.17.conv1' == layer_name:
            self.resnet.layer3[17].conv1 = new_layer
            self.layer_names["resnet.layer3.17.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.17.conv1"] = new_layer
        elif 'resnet.layer3.17.bn1' == layer_name:
            self.resnet.layer3[17].bn1 = new_layer
            self.layer_names["resnet.layer3.17.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.17.bn1"] = new_layer
        elif 'resnet.layer3.17.conv2' == layer_name:
            self.resnet.layer3[17].conv2 = new_layer
            self.layer_names["resnet.layer3.17.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.17.conv2"] = new_layer
        elif 'resnet.layer3.17.bn2' == layer_name:
            self.resnet.layer3[17].bn2 = new_layer
            self.layer_names["resnet.layer3.17.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.17.bn2"] = new_layer
        elif 'resnet.layer3.17.conv3' == layer_name:
            self.resnet.layer3[17].conv3 = new_layer
            self.layer_names["resnet.layer3.17.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.17.conv3"] = new_layer
        elif 'resnet.layer3.17.bn3' == layer_name:
            self.resnet.layer3[17].bn3 = new_layer
            self.layer_names["resnet.layer3.17.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.17.bn3"] = new_layer
        elif 'resnet.layer3.17.relu' == layer_name:
            self.resnet.layer3[17].relu = new_layer
            self.layer_names["resnet.layer3.17.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.17.relu"] = new_layer
        elif 'resnet.layer3.18' == layer_name:
            self.resnet.layer3[18] = new_layer
            self.layer_names["resnet.layer3.18"] = new_layer
            self.origin_layer_names["resnet.layer3.18"] = new_layer
        elif 'resnet.layer3.18.conv1' == layer_name:
            self.resnet.layer3[18].conv1 = new_layer
            self.layer_names["resnet.layer3.18.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.18.conv1"] = new_layer
        elif 'resnet.layer3.18.bn1' == layer_name:
            self.resnet.layer3[18].bn1 = new_layer
            self.layer_names["resnet.layer3.18.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.18.bn1"] = new_layer
        elif 'resnet.layer3.18.conv2' == layer_name:
            self.resnet.layer3[18].conv2 = new_layer
            self.layer_names["resnet.layer3.18.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.18.conv2"] = new_layer
        elif 'resnet.layer3.18.bn2' == layer_name:
            self.resnet.layer3[18].bn2 = new_layer
            self.layer_names["resnet.layer3.18.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.18.bn2"] = new_layer
        elif 'resnet.layer3.18.conv3' == layer_name:
            self.resnet.layer3[18].conv3 = new_layer
            self.layer_names["resnet.layer3.18.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.18.conv3"] = new_layer
        elif 'resnet.layer3.18.bn3' == layer_name:
            self.resnet.layer3[18].bn3 = new_layer
            self.layer_names["resnet.layer3.18.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.18.bn3"] = new_layer
        elif 'resnet.layer3.18.relu' == layer_name:
            self.resnet.layer3[18].relu = new_layer
            self.layer_names["resnet.layer3.18.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.18.relu"] = new_layer
        elif 'resnet.layer3.19' == layer_name:
            self.resnet.layer3[19] = new_layer
            self.layer_names["resnet.layer3.19"] = new_layer
            self.origin_layer_names["resnet.layer3.19"] = new_layer
        elif 'resnet.layer3.19.conv1' == layer_name:
            self.resnet.layer3[19].conv1 = new_layer
            self.layer_names["resnet.layer3.19.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.19.conv1"] = new_layer
        elif 'resnet.layer3.19.bn1' == layer_name:
            self.resnet.layer3[19].bn1 = new_layer
            self.layer_names["resnet.layer3.19.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.19.bn1"] = new_layer
        elif 'resnet.layer3.19.conv2' == layer_name:
            self.resnet.layer3[19].conv2 = new_layer
            self.layer_names["resnet.layer3.19.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.19.conv2"] = new_layer
        elif 'resnet.layer3.19.bn2' == layer_name:
            self.resnet.layer3[19].bn2 = new_layer
            self.layer_names["resnet.layer3.19.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.19.bn2"] = new_layer
        elif 'resnet.layer3.19.conv3' == layer_name:
            self.resnet.layer3[19].conv3 = new_layer
            self.layer_names["resnet.layer3.19.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.19.conv3"] = new_layer
        elif 'resnet.layer3.19.bn3' == layer_name:
            self.resnet.layer3[19].bn3 = new_layer
            self.layer_names["resnet.layer3.19.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.19.bn3"] = new_layer
        elif 'resnet.layer3.19.relu' == layer_name:
            self.resnet.layer3[19].relu = new_layer
            self.layer_names["resnet.layer3.19.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.19.relu"] = new_layer
        elif 'resnet.layer3.20' == layer_name:
            self.resnet.layer3[20] = new_layer
            self.layer_names["resnet.layer3.20"] = new_layer
            self.origin_layer_names["resnet.layer3.20"] = new_layer
        elif 'resnet.layer3.20.conv1' == layer_name:
            self.resnet.layer3[20].conv1 = new_layer
            self.layer_names["resnet.layer3.20.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.20.conv1"] = new_layer
        elif 'resnet.layer3.20.bn1' == layer_name:
            self.resnet.layer3[20].bn1 = new_layer
            self.layer_names["resnet.layer3.20.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.20.bn1"] = new_layer
        elif 'resnet.layer3.20.conv2' == layer_name:
            self.resnet.layer3[20].conv2 = new_layer
            self.layer_names["resnet.layer3.20.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.20.conv2"] = new_layer
        elif 'resnet.layer3.20.bn2' == layer_name:
            self.resnet.layer3[20].bn2 = new_layer
            self.layer_names["resnet.layer3.20.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.20.bn2"] = new_layer
        elif 'resnet.layer3.20.conv3' == layer_name:
            self.resnet.layer3[20].conv3 = new_layer
            self.layer_names["resnet.layer3.20.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.20.conv3"] = new_layer
        elif 'resnet.layer3.20.bn3' == layer_name:
            self.resnet.layer3[20].bn3 = new_layer
            self.layer_names["resnet.layer3.20.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.20.bn3"] = new_layer
        elif 'resnet.layer3.20.relu' == layer_name:
            self.resnet.layer3[20].relu = new_layer
            self.layer_names["resnet.layer3.20.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.20.relu"] = new_layer
        elif 'resnet.layer3.21' == layer_name:
            self.resnet.layer3[21] = new_layer
            self.layer_names["resnet.layer3.21"] = new_layer
            self.origin_layer_names["resnet.layer3.21"] = new_layer
        elif 'resnet.layer3.21.conv1' == layer_name:
            self.resnet.layer3[21].conv1 = new_layer
            self.layer_names["resnet.layer3.21.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.21.conv1"] = new_layer
        elif 'resnet.layer3.21.bn1' == layer_name:
            self.resnet.layer3[21].bn1 = new_layer
            self.layer_names["resnet.layer3.21.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.21.bn1"] = new_layer
        elif 'resnet.layer3.21.conv2' == layer_name:
            self.resnet.layer3[21].conv2 = new_layer
            self.layer_names["resnet.layer3.21.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.21.conv2"] = new_layer
        elif 'resnet.layer3.21.bn2' == layer_name:
            self.resnet.layer3[21].bn2 = new_layer
            self.layer_names["resnet.layer3.21.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.21.bn2"] = new_layer
        elif 'resnet.layer3.21.conv3' == layer_name:
            self.resnet.layer3[21].conv3 = new_layer
            self.layer_names["resnet.layer3.21.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.21.conv3"] = new_layer
        elif 'resnet.layer3.21.bn3' == layer_name:
            self.resnet.layer3[21].bn3 = new_layer
            self.layer_names["resnet.layer3.21.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.21.bn3"] = new_layer
        elif 'resnet.layer3.21.relu' == layer_name:
            self.resnet.layer3[21].relu = new_layer
            self.layer_names["resnet.layer3.21.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.21.relu"] = new_layer
        elif 'resnet.layer3.22' == layer_name:
            self.resnet.layer3[22] = new_layer
            self.layer_names["resnet.layer3.22"] = new_layer
            self.origin_layer_names["resnet.layer3.22"] = new_layer
        elif 'resnet.layer3.22.conv1' == layer_name:
            self.resnet.layer3[22].conv1 = new_layer
            self.layer_names["resnet.layer3.22.conv1"] = new_layer
            self.origin_layer_names["resnet.layer3.22.conv1"] = new_layer
        elif 'resnet.layer3.22.bn1' == layer_name:
            self.resnet.layer3[22].bn1 = new_layer
            self.layer_names["resnet.layer3.22.bn1"] = new_layer
            self.origin_layer_names["resnet.layer3.22.bn1"] = new_layer
        elif 'resnet.layer3.22.conv2' == layer_name:
            self.resnet.layer3[22].conv2 = new_layer
            self.layer_names["resnet.layer3.22.conv2"] = new_layer
            self.origin_layer_names["resnet.layer3.22.conv2"] = new_layer
        elif 'resnet.layer3.22.bn2' == layer_name:
            self.resnet.layer3[22].bn2 = new_layer
            self.layer_names["resnet.layer3.22.bn2"] = new_layer
            self.origin_layer_names["resnet.layer3.22.bn2"] = new_layer
        elif 'resnet.layer3.22.conv3' == layer_name:
            self.resnet.layer3[22].conv3 = new_layer
            self.layer_names["resnet.layer3.22.conv3"] = new_layer
            self.origin_layer_names["resnet.layer3.22.conv3"] = new_layer
        elif 'resnet.layer3.22.bn3' == layer_name:
            self.resnet.layer3[22].bn3 = new_layer
            self.layer_names["resnet.layer3.22.bn3"] = new_layer
            self.origin_layer_names["resnet.layer3.22.bn3"] = new_layer
        elif 'resnet.layer3.22.relu' == layer_name:
            self.resnet.layer3[22].relu = new_layer
            self.layer_names["resnet.layer3.22.relu"] = new_layer
            self.origin_layer_names["resnet.layer3.22.relu"] = new_layer
        elif 'resnet.layer4' == layer_name:
            self.resnet.layer4 = new_layer
            self.layer_names["resnet.layer4"] = new_layer
            self.origin_layer_names["resnet.layer4"] = new_layer
        elif 'resnet.layer4.0' == layer_name:
            self.resnet.layer4[0] = new_layer
            self.layer_names["resnet.layer4.0"] = new_layer
            self.origin_layer_names["resnet.layer4.0"] = new_layer
        elif 'resnet.layer4.0.conv1' == layer_name:
            self.resnet.layer4[0].conv1 = new_layer
            self.layer_names["resnet.layer4.0.conv1"] = new_layer
            self.origin_layer_names["resnet.layer4.0.conv1"] = new_layer
        elif 'resnet.layer4.0.bn1' == layer_name:
            self.resnet.layer4[0].bn1 = new_layer
            self.layer_names["resnet.layer4.0.bn1"] = new_layer
            self.origin_layer_names["resnet.layer4.0.bn1"] = new_layer
        elif 'resnet.layer4.0.conv2' == layer_name:
            self.resnet.layer4[0].conv2 = new_layer
            self.layer_names["resnet.layer4.0.conv2"] = new_layer
            self.origin_layer_names["resnet.layer4.0.conv2"] = new_layer
        elif 'resnet.layer4.0.bn2' == layer_name:
            self.resnet.layer4[0].bn2 = new_layer
            self.layer_names["resnet.layer4.0.bn2"] = new_layer
            self.origin_layer_names["resnet.layer4.0.bn2"] = new_layer
        elif 'resnet.layer4.0.conv3' == layer_name:
            self.resnet.layer4[0].conv3 = new_layer
            self.layer_names["resnet.layer4.0.conv3"] = new_layer
            self.origin_layer_names["resnet.layer4.0.conv3"] = new_layer
        elif 'resnet.layer4.0.bn3' == layer_name:
            self.resnet.layer4[0].bn3 = new_layer
            self.layer_names["resnet.layer4.0.bn3"] = new_layer
            self.origin_layer_names["resnet.layer4.0.bn3"] = new_layer
        elif 'resnet.layer4.0.relu' == layer_name:
            self.resnet.layer4[0].relu = new_layer
            self.layer_names["resnet.layer4.0.relu"] = new_layer
            self.origin_layer_names["resnet.layer4.0.relu"] = new_layer
        elif 'resnet.layer4.0.downsample' == layer_name:
            self.resnet.layer4[0].downsample = new_layer
            self.layer_names["resnet.layer4.0.downsample"] = new_layer
            self.origin_layer_names["resnet.layer4.0.downsample"] = new_layer
        elif 'resnet.layer4.0.downsample.0' == layer_name:
            self.resnet.layer4[0].downsample[0] = new_layer
            self.layer_names["resnet.layer4.0.downsample.0"] = new_layer
            self.origin_layer_names["resnet.layer4.0.downsample.0"] = new_layer
        elif 'resnet.layer4.0.downsample.1' == layer_name:
            self.resnet.layer4[0].downsample[1] = new_layer
            self.layer_names["resnet.layer4.0.downsample.1"] = new_layer
            self.origin_layer_names["resnet.layer4.0.downsample.1"] = new_layer
        elif 'resnet.layer4.1' == layer_name:
            self.resnet.layer4[1] = new_layer
            self.layer_names["resnet.layer4.1"] = new_layer
            self.origin_layer_names["resnet.layer4.1"] = new_layer
        elif 'resnet.layer4.1.conv1' == layer_name:
            self.resnet.layer4[1].conv1 = new_layer
            self.layer_names["resnet.layer4.1.conv1"] = new_layer
            self.origin_layer_names["resnet.layer4.1.conv1"] = new_layer
        elif 'resnet.layer4.1.bn1' == layer_name:
            self.resnet.layer4[1].bn1 = new_layer
            self.layer_names["resnet.layer4.1.bn1"] = new_layer
            self.origin_layer_names["resnet.layer4.1.bn1"] = new_layer
        elif 'resnet.layer4.1.conv2' == layer_name:
            self.resnet.layer4[1].conv2 = new_layer
            self.layer_names["resnet.layer4.1.conv2"] = new_layer
            self.origin_layer_names["resnet.layer4.1.conv2"] = new_layer
        elif 'resnet.layer4.1.bn2' == layer_name:
            self.resnet.layer4[1].bn2 = new_layer
            self.layer_names["resnet.layer4.1.bn2"] = new_layer
            self.origin_layer_names["resnet.layer4.1.bn2"] = new_layer
        elif 'resnet.layer4.1.conv3' == layer_name:
            self.resnet.layer4[1].conv3 = new_layer
            self.layer_names["resnet.layer4.1.conv3"] = new_layer
            self.origin_layer_names["resnet.layer4.1.conv3"] = new_layer
        elif 'resnet.layer4.1.bn3' == layer_name:
            self.resnet.layer4[1].bn3 = new_layer
            self.layer_names["resnet.layer4.1.bn3"] = new_layer
            self.origin_layer_names["resnet.layer4.1.bn3"] = new_layer
        elif 'resnet.layer4.1.relu' == layer_name:
            self.resnet.layer4[1].relu = new_layer
            self.layer_names["resnet.layer4.1.relu"] = new_layer
            self.origin_layer_names["resnet.layer4.1.relu"] = new_layer
        elif 'resnet.layer4.2' == layer_name:
            self.resnet.layer4[2] = new_layer
            self.layer_names["resnet.layer4.2"] = new_layer
            self.origin_layer_names["resnet.layer4.2"] = new_layer
        elif 'resnet.layer4.2.conv1' == layer_name:
            self.resnet.layer4[2].conv1 = new_layer
            self.layer_names["resnet.layer4.2.conv1"] = new_layer
            self.origin_layer_names["resnet.layer4.2.conv1"] = new_layer
        elif 'resnet.layer4.2.bn1' == layer_name:
            self.resnet.layer4[2].bn1 = new_layer
            self.layer_names["resnet.layer4.2.bn1"] = new_layer
            self.origin_layer_names["resnet.layer4.2.bn1"] = new_layer
        elif 'resnet.layer4.2.conv2' == layer_name:
            self.resnet.layer4[2].conv2 = new_layer
            self.layer_names["resnet.layer4.2.conv2"] = new_layer
            self.origin_layer_names["resnet.layer4.2.conv2"] = new_layer
        elif 'resnet.layer4.2.bn2' == layer_name:
            self.resnet.layer4[2].bn2 = new_layer
            self.layer_names["resnet.layer4.2.bn2"] = new_layer
            self.origin_layer_names["resnet.layer4.2.bn2"] = new_layer
        elif 'resnet.layer4.2.conv3' == layer_name:
            self.resnet.layer4[2].conv3 = new_layer
            self.layer_names["resnet.layer4.2.conv3"] = new_layer
            self.origin_layer_names["resnet.layer4.2.conv3"] = new_layer
        elif 'resnet.layer4.2.bn3' == layer_name:
            self.resnet.layer4[2].bn3 = new_layer
            self.layer_names["resnet.layer4.2.bn3"] = new_layer
            self.origin_layer_names["resnet.layer4.2.bn3"] = new_layer
        elif 'resnet.layer4.2.relu' == layer_name:
            self.resnet.layer4[2].relu = new_layer
            self.layer_names["resnet.layer4.2.relu"] = new_layer
            self.origin_layer_names["resnet.layer4.2.relu"] = new_layer

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
