import torch
from torch import nn

from models.SSD.ssd_utils_torch import WeightSharedMultiBox, BottomUp, FpnTopDown


def _conv7x7(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=stride, padding=(7 - 1) // 2, bias=False)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.97)


def _conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=(1 - 1) // 2, bias=False)


def _conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=(3 - 1) // 2, bias=False)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.997)


class ResNet(nn.Module):
    def __init__(self, block, layer_nums, in_channels, out_channels, strides):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = torch.nn.functional.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layer_nums[0], in_channels[0], out_channels[0], strides[0])
        self.layer2 = self._make_layer(block, layer_nums[1], in_channels[1], out_channels[1], strides[1])
        self.layer3 = self._make_layer(block, layer_nums[2], in_channels[2], out_channels[2], strides[2])
        self.layer4 = self._make_layer(block, layer_nums[3], in_channels[3], out_channels[3], strides[3])

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        layers = []
        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c1, c2, c3, c4, c5


class ResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)
        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn_last(out_channel)
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.Sequential(_conv1x1(in_channel, out_channel, stride), _bn(out_channel))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out += identity
        out = self.relu(out)

        return out


def resnet50():
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2])


class ssd_resnet50fpn_torch(nn.Module):
    def __init__(self):
        super(ssd_resnet50fpn_torch, self).__init__()
        self.network = resnet50()
        self.fpn = FpnTopDown([512, 1024, 2048], 256)
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

        self.in_shapes = {
            'INPUT': [1, 3, 640, 640],
            'network.conv1': [1, 3, 640, 640],
            'network.bn1': [1, 64, 320, 320],
            'network.maxpool': [1, 64, 320, 320],
            'network.layer1.0.conv1': [1, 64, 160, 160],
            'network.layer1.0.bn1': [1, 64, 160, 160],
            'network.layer1.0.conv2': [1, 64, 160, 160],
            'network.layer1.0.bn2': [1, 64, 160, 160],
            'network.layer1.0.conv3': [1, 64, 160, 160],
            'network.layer1.0.bn3': [1, 256, 160, 160],
            'network.layer1.0.relu': [1, 256, 160, 160],
            'network.layer1.0.down_sample_layer.0': [1, 64, 160, 160],
            'network.layer1.0.down_sample_layer.1': [1, 256, 160, 160],
            'network.layer1.1.conv1': [1, 256, 160, 160],
            'network.layer1.1.bn1': [1, 64, 160, 160],
            'network.layer1.1.conv2': [1, 64, 160, 160],
            'network.layer1.1.bn2': [1, 64, 160, 160],
            'network.layer1.1.conv3': [1, 64, 160, 160],
            'network.layer1.1.bn3': [1, 256, 160, 160],
            'network.layer1.1.relu': [1, 256, 160, 160],
            'network.layer1.2.conv1': [1, 256, 160, 160],
            'network.layer1.2.bn1': [1, 64, 160, 160],
            'network.layer1.2.conv2': [1, 64, 160, 160],
            'network.layer1.2.bn2': [1, 64, 160, 160],
            'network.layer1.2.conv3': [1, 64, 160, 160],
            'network.layer1.2.bn3': [1, 256, 160, 160],
            'network.layer1.2.relu': [1, 256, 160, 160],
            'network.layer2.0.conv1': [1, 256, 160, 160],
            'network.layer2.0.bn1': [1, 128, 160, 160],
            'network.layer2.0.conv2': [1, 128, 160, 160],
            'network.layer2.0.bn2': [1, 128, 80, 80],
            'network.layer2.0.conv3': [1, 128, 80, 80],
            'network.layer2.0.bn3': [1, 512, 80, 80],
            'network.layer2.0.relu': [1, 512, 80, 80],
            'network.layer2.0.down_sample_layer.0': [1, 256, 160, 160],
            'network.layer2.0.down_sample_layer.1': [1, 512, 80, 80],
            'network.layer2.1.conv1': [1, 512, 80, 80],
            'network.layer2.1.bn1': [1, 128, 80, 80],
            'network.layer2.1.conv2': [1, 128, 80, 80],
            'network.layer2.1.bn2': [1, 128, 80, 80],
            'network.layer2.1.conv3': [1, 128, 80, 80],
            'network.layer2.1.bn3': [1, 512, 80, 80],
            'network.layer2.1.relu': [1, 512, 80, 80],
            'network.layer2.2.conv1': [1, 512, 80, 80],
            'network.layer2.2.bn1': [1, 128, 80, 80],
            'network.layer2.2.conv2': [1, 128, 80, 80],
            'network.layer2.2.bn2': [1, 128, 80, 80],
            'network.layer2.2.conv3': [1, 128, 80, 80],
            'network.layer2.2.bn3': [1, 512, 80, 80],
            'network.layer2.2.relu': [1, 512, 80, 80],
            'network.layer2.3.conv1': [1, 512, 80, 80],
            'network.layer2.3.bn1': [1, 128, 80, 80],
            'network.layer2.3.conv2': [1, 128, 80, 80],
            'network.layer2.3.bn2': [1, 128, 80, 80],
            'network.layer2.3.conv3': [1, 128, 80, 80],
            'network.layer2.3.bn3': [1, 512, 80, 80],
            'network.layer2.3.relu': [1, 512, 80, 80],
            'network.layer3.0.conv1': [1, 512, 80, 80],
            'network.layer3.0.bn1': [1, 256, 80, 80],
            'network.layer3.0.conv2': [1, 256, 80, 80],
            'network.layer3.0.bn2': [1, 256, 40, 40],
            'network.layer3.0.conv3': [1, 256, 40, 40],
            'network.layer3.0.bn3': [1, 1024, 40, 40],
            'network.layer3.0.relu': [1, 1024, 40, 40],
            'network.layer3.0.down_sample_layer.0': [1, 512, 80, 80],
            'network.layer3.0.down_sample_layer.1': [1, 1024, 40, 40],
            'network.layer3.1.conv1': [1, 1024, 40, 40],
            'network.layer3.1.bn1': [1, 256, 40, 40],
            'network.layer3.1.conv2': [1, 256, 40, 40],
            'network.layer3.1.bn2': [1, 256, 40, 40],
            'network.layer3.1.conv3': [1, 256, 40, 40],
            'network.layer3.1.bn3': [1, 1024, 40, 40],
            'network.layer3.1.relu': [1, 1024, 40, 40],
            'network.layer3.2.conv1': [1, 1024, 40, 40],
            'network.layer3.2.bn1': [1, 256, 40, 40],
            'network.layer3.2.conv2': [1, 256, 40, 40],
            'network.layer3.2.bn2': [1, 256, 40, 40],
            'network.layer3.2.conv3': [1, 256, 40, 40],
            'network.layer3.2.bn3': [1, 1024, 40, 40],
            'network.layer3.2.relu': [1, 1024, 40, 40],
            'network.layer3.3.conv1': [1, 1024, 40, 40],
            'network.layer3.3.bn1': [1, 256, 40, 40],
            'network.layer3.3.conv2': [1, 256, 40, 40],
            'network.layer3.3.bn2': [1, 256, 40, 40],
            'network.layer3.3.conv3': [1, 256, 40, 40],
            'network.layer3.3.bn3': [1, 1024, 40, 40],
            'network.layer3.3.relu': [1, 1024, 40, 40],
            'network.layer3.4.conv1': [1, 1024, 40, 40],
            'network.layer3.4.bn1': [1, 256, 40, 40],
            'network.layer3.4.conv2': [1, 256, 40, 40],
            'network.layer3.4.bn2': [1, 256, 40, 40],
            'network.layer3.4.conv3': [1, 256, 40, 40],
            'network.layer3.4.bn3': [1, 1024, 40, 40],
            'network.layer3.4.relu': [1, 1024, 40, 40],
            'network.layer3.5.conv1': [1, 1024, 40, 40],
            'network.layer3.5.bn1': [1, 256, 40, 40],
            'network.layer3.5.conv2': [1, 256, 40, 40],
            'network.layer3.5.bn2': [1, 256, 40, 40],
            'network.layer3.5.conv3': [1, 256, 40, 40],
            'network.layer3.5.bn3': [1, 1024, 40, 40],
            'network.layer3.5.relu': [1, 1024, 40, 40],
            'network.layer4.0.conv1': [1, 1024, 40, 40],
            'network.layer4.0.bn1': [1, 512, 40, 40],
            'network.layer4.0.conv2': [1, 512, 40, 40],
            'network.layer4.0.bn2': [1, 512, 20, 20],
            'network.layer4.0.conv3': [1, 512, 20, 20],
            'network.layer4.0.bn3': [1, 2048, 20, 20],
            'network.layer4.0.relu': [1, 2048, 20, 20],
            'network.layer4.0.down_sample_layer.0': [1, 1024, 40, 40],
            'network.layer4.0.down_sample_layer.1': [1, 2048, 20, 20],
            'network.layer4.1.conv1': [1, 2048, 20, 20],
            'network.layer4.1.bn1': [1, 512, 20, 20],
            'network.layer4.1.conv2': [1, 512, 20, 20],
            'network.layer4.1.bn2': [1, 512, 20, 20],
            'network.layer4.1.conv3': [1, 512, 20, 20],
            'network.layer4.1.bn3': [1, 2048, 20, 20],
            'network.layer4.1.relu': [1, 2048, 20, 20],
            'network.layer4.2.conv1': [1, 2048, 20, 20],
            'network.layer4.2.bn1': [1, 512, 20, 20],
            'network.layer4.2.conv2': [1, 512, 20, 20],
            'network.layer4.2.bn2': [1, 512, 20, 20],
            'network.layer4.2.conv3': [1, 512, 20, 20],
            'network.layer4.2.bn3': [1, 2048, 20, 20],
            'network.layer4.2.relu': [1, 2048, 20, 20],
            'OUTPUT1': [1, 64, 160, 160],
            'OUTPUT2': [1, 256, 160, 160],
            'OUTPUT3': [1, 512, 80, 80],
            'OUTPUT4': [1, 1024, 40, 40],
            'OUTPUT5': [1, 2048, 20, 20]
        }

        self.out_shapes = {
            'INPUT': [1, 3, 640, 640],
            'network.conv1': [1, 64, 320, 320],
            'network.bn1': [1, 64, 320, 320],
            'network.maxpool': [1, 64, 160, 160],
            'network.layer1.0.conv1': [1, 64, 160, 160],
            'network.layer1.0.bn1': [1, 64, 160, 160],
            'network.layer1.0.conv2': [1, 64, 160, 160],
            'network.layer1.0.bn2': [1, 64, 160, 160],
            'network.layer1.0.conv3': [1, 256, 160, 160],
            'network.layer1.0.bn3': [1, 256, 160, 160],
            'network.layer1.0.relu': [1, 256, 160, 160],
            'network.layer1.0.down_sample_layer.0': [1, 256, 160, 160],
            'network.layer1.0.down_sample_layer.1': [1, 256, 160, 160],
            'network.layer1.1.conv1': [1, 64, 160, 160],
            'network.layer1.1.bn1': [1, 64, 160, 160],
            'network.layer1.1.conv2': [1, 64, 160, 160],
            'network.layer1.1.bn2': [1, 64, 160, 160],
            'network.layer1.1.conv3': [1, 256, 160, 160],
            'network.layer1.1.bn3': [1, 256, 160, 160],
            'network.layer1.1.relu': [1, 256, 160, 160],
            'network.layer1.2.conv1': [1, 64, 160, 160],
            'network.layer1.2.bn1': [1, 64, 160, 160],
            'network.layer1.2.conv2': [1, 64, 160, 160],
            'network.layer1.2.bn2': [1, 64, 160, 160],
            'network.layer1.2.conv3': [1, 256, 160, 160],
            'network.layer1.2.bn3': [1, 256, 160, 160],
            'network.layer1.2.relu': [1, 256, 160, 160],
            'network.layer2.0.conv1': [1, 128, 160, 160],
            'network.layer2.0.bn1': [1, 128, 160, 160],
            'network.layer2.0.conv2': [1, 128, 80, 80],
            'network.layer2.0.bn2': [1, 128, 80, 80],
            'network.layer2.0.conv3': [1, 512, 80, 80],
            'network.layer2.0.bn3': [1, 512, 80, 80],
            'network.layer2.0.relu': [1, 512, 80, 80],
            'network.layer2.0.down_sample_layer.0': [1, 512, 80, 80],
            'network.layer2.0.down_sample_layer.1': [1, 512, 80, 80],
            'network.layer2.1.conv1': [1, 128, 80, 80],
            'network.layer2.1.bn1': [1, 128, 80, 80],
            'network.layer2.1.conv2': [1, 128, 80, 80],
            'network.layer2.1.bn2': [1, 128, 80, 80],
            'network.layer2.1.conv3': [1, 512, 80, 80],
            'network.layer2.1.bn3': [1, 512, 80, 80],
            'network.layer2.1.relu': [1, 512, 80, 80],
            'network.layer2.2.conv1': [1, 128, 80, 80],
            'network.layer2.2.bn1': [1, 128, 80, 80],
            'network.layer2.2.conv2': [1, 128, 80, 80],
            'network.layer2.2.bn2': [1, 128, 80, 80],
            'network.layer2.2.conv3': [1, 512, 80, 80],
            'network.layer2.2.bn3': [1, 512, 80, 80],
            'network.layer2.2.relu': [1, 512, 80, 80],
            'network.layer2.3.conv1': [1, 128, 80, 80],
            'network.layer2.3.bn1': [1, 128, 80, 80],
            'network.layer2.3.conv2': [1, 128, 80, 80],
            'network.layer2.3.bn2': [1, 128, 80, 80],
            'network.layer2.3.conv3': [1, 512, 80, 80],
            'network.layer2.3.bn3': [1, 512, 80, 80],
            'network.layer2.3.relu': [1, 512, 80, 80],
            'network.layer3.0.conv1': [1, 256, 80, 80],
            'network.layer3.0.bn1': [1, 256, 80, 80],
            'network.layer3.0.conv2': [1, 256, 40, 40],
            'network.layer3.0.bn2': [1, 256, 40, 40],
            'network.layer3.0.conv3': [1, 1024, 40, 40],
            'network.layer3.0.bn3': [1, 1024, 40, 40],
            'network.layer3.0.relu': [1, 1024, 40, 40],
            'network.layer3.0.down_sample_layer.0': [1, 1024, 40, 40],
            'network.layer3.0.down_sample_layer.1': [1, 1024, 40, 40],
            'network.layer3.1.conv1': [1, 256, 40, 40],
            'network.layer3.1.bn1': [1, 256, 40, 40],
            'network.layer3.1.conv2': [1, 256, 40, 40],
            'network.layer3.1.bn2': [1, 256, 40, 40],
            'network.layer3.1.conv3': [1, 1024, 40, 40],
            'network.layer3.1.bn3': [1, 1024, 40, 40],
            'network.layer3.1.relu': [1, 1024, 40, 40],
            'network.layer3.2.conv1': [1, 256, 40, 40],
            'network.layer3.2.bn1': [1, 256, 40, 40],
            'network.layer3.2.conv2': [1, 256, 40, 40],
            'network.layer3.2.bn2': [1, 256, 40, 40],
            'network.layer3.2.conv3': [1, 1024, 40, 40],
            'network.layer3.2.bn3': [1, 1024, 40, 40],
            'network.layer3.2.relu': [1, 1024, 40, 40],
            'network.layer3.3.conv1': [1, 256, 40, 40],
            'network.layer3.3.bn1': [1, 256, 40, 40],
            'network.layer3.3.conv2': [1, 256, 40, 40],
            'network.layer3.3.bn2': [1, 256, 40, 40],
            'network.layer3.3.conv3': [1, 1024, 40, 40],
            'network.layer3.3.bn3': [1, 1024, 40, 40],
            'network.layer3.3.relu': [1, 1024, 40, 40],
            'network.layer3.4.conv1': [1, 256, 40, 40],
            'network.layer3.4.bn1': [1, 256, 40, 40],
            'network.layer3.4.conv2': [1, 256, 40, 40],
            'network.layer3.4.bn2': [1, 256, 40, 40],
            'network.layer3.4.conv3': [1, 1024, 40, 40],
            'network.layer3.4.bn3': [1, 1024, 40, 40],
            'network.layer3.4.relu': [1, 1024, 40, 40],
            'network.layer3.5.conv1': [1, 256, 40, 40],
            'network.layer3.5.bn1': [1, 256, 40, 40],
            'network.layer3.5.conv2': [1, 256, 40, 40],
            'network.layer3.5.bn2': [1, 256, 40, 40],
            'network.layer3.5.conv3': [1, 1024, 40, 40],
            'network.layer3.5.bn3': [1, 1024, 40, 40],
            'network.layer3.5.relu': [1, 1024, 40, 40],
            'network.layer4.0.conv1': [1, 512, 40, 40],
            'network.layer4.0.bn1': [1, 512, 40, 40],
            'network.layer4.0.conv2': [1, 512, 20, 20],
            'network.layer4.0.bn2': [1, 512, 20, 20],
            'network.layer4.0.conv3': [1, 2048, 20, 20],
            'network.layer4.0.bn3': [1, 2048, 20, 20],
            'network.layer4.0.relu': [1, 2048, 20, 20],
            'network.layer4.0.down_sample_layer.0': [1, 2048, 20, 20],
            'network.layer4.0.down_sample_layer.1': [1, 2048, 20, 20],
            'network.layer4.1.conv1': [1, 512, 20, 20],
            'network.layer4.1.bn1': [1, 512, 20, 20],
            'network.layer4.1.conv2': [1, 512, 20, 20],
            'network.layer4.1.bn2': [1, 512, 20, 20],
            'network.layer4.1.conv3': [1, 2048, 20, 20],
            'network.layer4.1.bn3': [1, 2048, 20, 20],
            'network.layer4.1.relu': [1, 2048, 20, 20],
            'network.layer4.2.conv1': [1, 512, 20, 20],
            'network.layer4.2.bn1': [1, 512, 20, 20],
            'network.layer4.2.conv2': [1, 512, 20, 20],
            'network.layer4.2.bn2': [1, 512, 20, 20],
            'network.layer4.2.conv3': [1, 2048, 20, 20],
            'network.layer4.2.bn3': [1, 2048, 20, 20],
            'network.layer4.2.relu': [1, 2048, 20, 20],
            'OUTPUT1': [1, 64, 160, 160],
            'OUTPUT2': [1, 256, 160, 160],
            'OUTPUT3': [1, 512, 80, 80],
            'OUTPUT4': [1, 1024, 40, 40],
            'OUTPUT5': [1, 2048, 20, 20]
        }

        self.orders = {
            'network.conv1': ["INPUT", "network.bn1"],
            'network.bn1': ["network.conv1", "network.maxpool"],
            'network.maxpool': ["network.bn1",
                                ["network.layer1.0.conv1", 'network.layer1.0.down_sample_layer.0', "OUTPUT1"]],
            'network.layer1.0.conv1': ["network.maxpool", "network.layer1.0.bn1"],
            'network.layer1.0.bn1': ["network.layer1.0.conv1", "network.layer1.0.conv2"],
            'network.layer1.0.conv2': ["network.layer1.0.bn1", "network.layer1.0.bn2"],
            'network.layer1.0.bn2': ["network.layer1.0.conv2", "network.layer1.0.conv3"],
            'network.layer1.0.conv3': ["network.layer1.0.bn2", "network.layer1.0.bn3"],
            'network.layer1.0.bn3': ["network.layer1.0.conv3", "network.layer1.0.relu"],
            'network.layer1.0.relu': ["network.layer1.0.bn3", "network.layer1.1.conv1"],
            'network.layer1.0.down_sample_layer.0': ["network.maxpool", "network.layer1.0.down_sample_layer.1"],
            'network.layer1.0.down_sample_layer.1': ["network.layer1.0.down_sample_layer.0", "network.layer1.1.conv1"],
            'network.layer1.1.conv1': [['network.layer1.0.relu', 'network.layer1.0.down_sample_layer.1'],
                                       "network.layer1.1.bn1"],
            'network.layer1.1.bn1': ["network.layer1.1.conv1", "network.layer1.1.conv2"],
            'network.layer1.1.conv2': ["network.layer1.1.bn1", "network.layer1.1.bn2"],
            'network.layer1.1.bn2': ["network.layer1.1.conv2", "network.layer1.1.conv3"],
            'network.layer1.1.conv3': ["network.layer1.1.bn2", "network.layer1.1.bn3"],
            'network.layer1.1.bn3': ["network.layer1.1.conv3", "network.layer1.1.relu"],
            'network.layer1.1.relu': ["network.layer1.1.bn3", "network.layer1.2.conv1"],
            'network.layer1.2.conv1': ["network.layer1.1.relu", "network.layer1.2.bn1"],
            'network.layer1.2.bn1': ["network.layer1.2.conv1", "network.layer1.2.conv2"],
            'network.layer1.2.conv2': ["network.layer1.2.bn1", "network.layer1.2.bn2"],
            'network.layer1.2.bn2': ["network.layer1.2.conv2", "network.layer1.2.conv3"],
            'network.layer1.2.conv3': ["network.layer1.2.bn2", "network.layer1.2.bn3"],
            'network.layer1.2.bn3': ["network.layer1.2.conv3", "network.layer1.2.relu"],
            'network.layer1.2.relu': ["network.layer1.2.bn3",
                                      ["network.layer2.0.conv1", 'network.layer2.0.down_sample_layer.0', "OUTPUT2"]],
            'network.layer2.0.conv1': ["network.layer1.2.relu", "network.layer2.0.bn1"],
            'network.layer2.0.bn1': ["network.layer2.0.conv1", "network.layer2.0.conv2"],
            'network.layer2.0.conv2': ["network.layer2.0.bn1", "network.layer2.0.bn2"],
            'network.layer2.0.bn2': ["network.layer2.0.conv2", "network.layer2.0.conv3"],
            'network.layer2.0.conv3': ["network.layer2.0.bn2", "network.layer2.0.bn3"],
            'network.layer2.0.bn3': ["network.layer2.0.conv3", "network.layer2.0.relu"],
            'network.layer2.0.relu': ["network.layer2.0.bn3", "network.layer2.1.conv1"],
            'network.layer2.0.down_sample_layer.0': ["network.layer1.2.relu", "network.layer2.0.down_sample_layer.1"],
            'network.layer2.0.down_sample_layer.1': ["network.layer2.0.down_sample_layer.0", "network.layer2.1.conv1"],
            'network.layer2.1.conv1': [['network.layer2.0.relu', "network.layer2.0.down_sample_layer.1"],
                                       "network.layer2.1.bn1"],
            'network.layer2.1.bn1': ["network.layer2.1.conv1", "network.layer2.1.conv2"],
            'network.layer2.1.conv2': ["network.layer2.1.bn1", "network.layer2.1.bn2"],
            'network.layer2.1.bn2': ["network.layer2.1.conv2", "network.layer2.1.conv3"],
            'network.layer2.1.conv3': ["network.layer2.1.bn2", "network.layer2.1.bn3"],
            'network.layer2.1.bn3': ["network.layer2.1.conv3", "network.layer2.1.relu"],
            'network.layer2.1.relu': ["network.layer2.1.bn3", "network.layer2.2.conv1"],
            'network.layer2.2.conv1': ["network.layer2.1.relu", "network.layer2.2.bn1"],
            'network.layer2.2.bn1': ["network.layer2.2.conv1", "network.layer2.2.conv2"],
            'network.layer2.2.conv2': ["network.layer2.2.bn1", "network.layer2.2.bn2"],
            'network.layer2.2.bn2': ["network.layer2.2.conv2", "network.layer2.2.conv3"],
            'network.layer2.2.conv3': ["network.layer2.2.bn2", "network.layer2.2.bn3"],
            'network.layer2.2.bn3': ["network.layer2.2.conv3", "network.layer2.2.relu"],
            'network.layer2.2.relu': ["network.layer2.2.bn3", "network.layer2.3.conv1"],
            'network.layer2.3.conv1': ["network.layer2.2.relu", "network.layer2.3.bn1"],
            'network.layer2.3.bn1': ["network.layer2.3.conv1", "network.layer2.3.conv2"],
            'network.layer2.3.conv2': ["network.layer2.3.bn1", "network.layer2.3.bn2"],
            'network.layer2.3.bn2': ["network.layer2.3.conv2", "network.layer2.3.conv3"],
            'network.layer2.3.conv3': ["network.layer2.3.bn2", "network.layer2.3.bn3"],
            'network.layer2.3.bn3': ["network.layer2.3.conv3", "network.layer2.3.relu"],
            'network.layer2.3.relu': ["network.layer2.3.bn3",
                                      ["network.layer3.0.conv1", 'network.layer3.0.down_sample_layer.0', "OUTPUT3"]],

            'network.layer3.0.conv1': ["network.layer2.3.relu", "network.layer3.0.bn1"],
            'network.layer3.0.bn1': ["network.layer3.0.conv1", "network.layer3.0.conv2"],
            'network.layer3.0.conv2': ["network.layer3.0.bn1", "network.layer3.0.bn2"],
            'network.layer3.0.bn2': ["network.layer3.0.conv2", "network.layer3.0.conv3"],
            'network.layer3.0.conv3': ["network.layer3.0.bn2", "network.layer3.0.bn3"],
            'network.layer3.0.bn3': ["network.layer3.0.conv3", "network.layer3.0.relu"],
            'network.layer3.0.relu': ["network.layer3.0.bn3", "network.layer3.1.conv1"],
            'network.layer3.0.down_sample_layer.0': ["network.layer2.3.relu", "network.layer3.0.down_sample_layer.1"],
            'network.layer3.0.down_sample_layer.1': ["network.layer3.0.down_sample_layer.0", "network.layer3.1.conv1"],

            'network.layer3.1.conv1': [["network.layer3.0.down_sample_layer.1", 'network.layer3.0.relu'],
                                       "network.layer3.1.bn1"],
            'network.layer3.1.bn1': ["network.layer3.1.conv1", "network.layer3.1.conv2"],
            'network.layer3.1.conv2': ["network.layer3.1.bn1", "network.layer3.1.bn2"],
            'network.layer3.1.bn2': ["network.layer3.1.conv2", "network.layer3.1.conv3"],
            'network.layer3.1.conv3': ["network.layer3.1.bn2", "network.layer3.1.bn3"],
            'network.layer3.1.bn3': ["network.layer3.1.conv3", "network.layer3.1.relu"],
            'network.layer3.1.relu': ["network.layer3.1.bn3", "network.layer3.2.conv1"],
            'network.layer3.2.conv1': ["network.layer3.1.relu", "network.layer3.2.bn1"],
            'network.layer3.2.bn1': ["network.layer3.2.conv1", "network.layer3.2.conv2"],
            'network.layer3.2.conv2': ["network.layer3.2.bn1", "network.layer3.2.bn2"],
            'network.layer3.2.bn2': ["network.layer3.2.conv2", "network.layer3.2.conv3"],
            'network.layer3.2.conv3': ["network.layer3.2.bn2", "network.layer3.2.bn3"],
            'network.layer3.2.bn3': ["network.layer3.2.conv3", "network.layer3.2.relu"],
            'network.layer3.2.relu': ["network.layer3.2.bn3", "network.layer3.3.conv1"],
            'network.layer3.3.conv1': ["network.layer3.2.relu", "network.layer3.3.bn1"],
            'network.layer3.3.bn1': ["network.layer3.3.conv1", "network.layer3.3.conv2"],
            'network.layer3.3.conv2': ["network.layer3.3.bn1", "network.layer3.3.bn2"],
            'network.layer3.3.bn2': ["network.layer3.3.conv2", "network.layer3.3.conv3"],
            'network.layer3.3.conv3': ["network.layer3.3.bn2", "network.layer3.3.bn3"],
            'network.layer3.3.bn3': ["network.layer3.3.conv3", "network.layer3.3.relu"],
            'network.layer3.3.relu': ["network.layer3.3.bn3", "network.layer3.4.conv1"],
            'network.layer3.4.conv1': ["network.layer3.3.relu", "network.layer3.4.bn1"],
            'network.layer3.4.bn1': ["network.layer3.4.conv1", "network.layer3.4.conv2"],
            'network.layer3.4.conv2': ["network.layer3.4.bn1", "network.layer3.4.bn2"],
            'network.layer3.4.bn2': ["network.layer3.4.conv2", "network.layer3.4.conv3"],
            'network.layer3.4.conv3': ["network.layer3.4.bn2", "network.layer3.4.bn3"],
            'network.layer3.4.bn3': ["network.layer3.4.conv3", "network.layer3.4.relu"],
            'network.layer3.4.relu': ["network.layer3.4.bn3", "network.layer3.5.conv1"],
            'network.layer3.5.conv1': ["network.layer3.4.relu", "network.layer3.5.bn1"],
            'network.layer3.5.bn1': ["network.layer3.5.conv1", "network.layer3.5.conv2"],
            'network.layer3.5.conv2': ["network.layer3.5.bn1", "network.layer3.5.bn2"],
            'network.layer3.5.bn2': ["network.layer3.5.conv2", "network.layer3.5.conv3"],
            'network.layer3.5.conv3': ["network.layer3.5.bn2", "network.layer3.5.bn3"],
            'network.layer3.5.bn3': ["network.layer3.5.conv3", "network.layer3.5.relu"],
            'network.layer3.5.relu': ["network.layer3.5.bn3",
                                      ["network.layer4.0.conv1", 'network.layer4.0.down_sample_layer.0', "OUTPUT4"]],

            'network.layer4.0.conv1': ["network.layer3.5.relu", "network.layer4.0.bn1"],
            'network.layer4.0.bn1': ["network.layer4.0.conv1", "network.layer4.0.conv2"],
            'network.layer4.0.conv2': ["network.layer4.0.bn1", "network.layer4.0.bn2"],
            'network.layer4.0.bn2': ["network.layer4.0.conv2", "network.layer4.0.conv3"],
            'network.layer4.0.conv3': ["network.layer4.0.bn2", "network.layer4.0.bn3"],
            'network.layer4.0.bn3': ["network.layer4.0.conv3", "network.layer4.0.relu"],
            'network.layer4.0.relu': ["network.layer4.0.bn3", "network.layer4.1.conv1"],
            'network.layer4.0.down_sample_layer.0': ["network.layer3.5.relu", "network.layer4.0.down_sample_layer.1"],
            'network.layer4.0.down_sample_layer.1': ["network.layer4.0.down_sample_layer.0", "network.layer4.1.conv1"],

            'network.layer4.1.conv1': [["network.layer4.0.down_sample_layer.1", 'network.layer4.0.relu'],
                                       "network.layer4.1.bn1"],
            'network.layer4.1.bn1': ["network.layer4.1.conv1", "network.layer4.1.conv2"],
            'network.layer4.1.conv2': ["network.layer4.1.bn1", "network.layer4.1.bn2"],
            'network.layer4.1.bn2': ["network.layer4.1.conv2", "network.layer4.1.conv3"],
            'network.layer4.1.conv3': ["network.layer4.1.bn2", "network.layer4.1.bn3"],
            'network.layer4.1.bn3': ["network.layer4.1.conv3", "network.layer4.1.relu"],
            'network.layer4.1.relu': ["network.layer4.1.bn3", "network.layer4.2.conv1"],
            'network.layer4.2.conv1': ["network.layer4.1.relu", "network.layer4.2.bn1"],
            'network.layer4.2.bn1': ["network.layer4.2.conv1", "network.layer4.2.conv2"],
            'network.layer4.2.conv2': ["network.layer4.2.bn1", "network.layer4.2.bn2"],
            'network.layer4.2.bn2': ["network.layer4.2.conv2", "network.layer4.2.conv3"],
            'network.layer4.2.conv3': ["network.layer4.2.bn2", "network.layer4.2.bn3"],
            'network.layer4.2.bn3': ["network.layer4.2.conv3", "network.layer4.2.relu"],
            'network.layer4.2.relu': ["network.layer4.2.bn3", "OUTPUT5"],
        }

        self.layer_names = {
            "network": self.network,
            "network.conv1": self.network.conv1,
            "network.bn1": self.network.bn1,
            "network.maxpool": self.network.maxpool,
            "network.layer1": self.network.layer1,
            "network.layer1.0": self.network.layer1[0],
            "network.layer1.0.conv1": self.network.layer1[0].conv1,
            "network.layer1.0.bn1": self.network.layer1[0].bn1,
            "network.layer1.0.conv2": self.network.layer1[0].conv2,
            "network.layer1.0.bn2": self.network.layer1[0].bn2,
            "network.layer1.0.conv3": self.network.layer1[0].conv3,
            "network.layer1.0.bn3": self.network.layer1[0].bn3,
            "network.layer1.0.relu": self.network.layer1[0].relu,
            "network.layer1.0.down_sample_layer": self.network.layer1[0].down_sample_layer,
            "network.layer1.0.down_sample_layer.0": self.network.layer1[0].down_sample_layer[0],
            "network.layer1.0.down_sample_layer.1": self.network.layer1[0].down_sample_layer[1],
            "network.layer1.1": self.network.layer1[1],
            "network.layer1.1.conv1": self.network.layer1[1].conv1,
            "network.layer1.1.bn1": self.network.layer1[1].bn1,
            "network.layer1.1.conv2": self.network.layer1[1].conv2,
            "network.layer1.1.bn2": self.network.layer1[1].bn2,
            "network.layer1.1.conv3": self.network.layer1[1].conv3,
            "network.layer1.1.bn3": self.network.layer1[1].bn3,
            "network.layer1.1.relu": self.network.layer1[1].relu,
            "network.layer1.2": self.network.layer1[2],
            "network.layer1.2.conv1": self.network.layer1[2].conv1,
            "network.layer1.2.bn1": self.network.layer1[2].bn1,
            "network.layer1.2.conv2": self.network.layer1[2].conv2,
            "network.layer1.2.bn2": self.network.layer1[2].bn2,
            "network.layer1.2.conv3": self.network.layer1[2].conv3,
            "network.layer1.2.bn3": self.network.layer1[2].bn3,
            "network.layer1.2.relu": self.network.layer1[2].relu,
            "network.layer2": self.network.layer2,
            "network.layer2.0": self.network.layer2[0],
            "network.layer2.0.conv1": self.network.layer2[0].conv1,
            "network.layer2.0.bn1": self.network.layer2[0].bn1,
            "network.layer2.0.conv2": self.network.layer2[0].conv2,
            "network.layer2.0.bn2": self.network.layer2[0].bn2,
            "network.layer2.0.conv3": self.network.layer2[0].conv3,
            "network.layer2.0.bn3": self.network.layer2[0].bn3,
            "network.layer2.0.relu": self.network.layer2[0].relu,
            "network.layer2.0.down_sample_layer": self.network.layer2[0].down_sample_layer,
            "network.layer2.0.down_sample_layer.0": self.network.layer2[0].down_sample_layer[0],
            "network.layer2.0.down_sample_layer.1": self.network.layer2[0].down_sample_layer[1],
            "network.layer2.1": self.network.layer2[1],
            "network.layer2.1.conv1": self.network.layer2[1].conv1,
            "network.layer2.1.bn1": self.network.layer2[1].bn1,
            "network.layer2.1.conv2": self.network.layer2[1].conv2,
            "network.layer2.1.bn2": self.network.layer2[1].bn2,
            "network.layer2.1.conv3": self.network.layer2[1].conv3,
            "network.layer2.1.bn3": self.network.layer2[1].bn3,
            "network.layer2.1.relu": self.network.layer2[1].relu,
            "network.layer2.2": self.network.layer2[2],
            "network.layer2.2.conv1": self.network.layer2[2].conv1,
            "network.layer2.2.bn1": self.network.layer2[2].bn1,
            "network.layer2.2.conv2": self.network.layer2[2].conv2,
            "network.layer2.2.bn2": self.network.layer2[2].bn2,
            "network.layer2.2.conv3": self.network.layer2[2].conv3,
            "network.layer2.2.bn3": self.network.layer2[2].bn3,
            "network.layer2.2.relu": self.network.layer2[2].relu,
            "network.layer2.3": self.network.layer2[3],
            "network.layer2.3.conv1": self.network.layer2[3].conv1,
            "network.layer2.3.bn1": self.network.layer2[3].bn1,
            "network.layer2.3.conv2": self.network.layer2[3].conv2,
            "network.layer2.3.bn2": self.network.layer2[3].bn2,
            "network.layer2.3.conv3": self.network.layer2[3].conv3,
            "network.layer2.3.bn3": self.network.layer2[3].bn3,
            "network.layer2.3.relu": self.network.layer2[3].relu,
            "network.layer3": self.network.layer3,
            "network.layer3.0": self.network.layer3[0],
            "network.layer3.0.conv1": self.network.layer3[0].conv1,
            "network.layer3.0.bn1": self.network.layer3[0].bn1,
            "network.layer3.0.conv2": self.network.layer3[0].conv2,
            "network.layer3.0.bn2": self.network.layer3[0].bn2,
            "network.layer3.0.conv3": self.network.layer3[0].conv3,
            "network.layer3.0.bn3": self.network.layer3[0].bn3,
            "network.layer3.0.relu": self.network.layer3[0].relu,
            "network.layer3.0.down_sample_layer": self.network.layer3[0].down_sample_layer,
            "network.layer3.0.down_sample_layer.0": self.network.layer3[0].down_sample_layer[0],
            "network.layer3.0.down_sample_layer.1": self.network.layer3[0].down_sample_layer[1],
            "network.layer3.1": self.network.layer3[1],
            "network.layer3.1.conv1": self.network.layer3[1].conv1,
            "network.layer3.1.bn1": self.network.layer3[1].bn1,
            "network.layer3.1.conv2": self.network.layer3[1].conv2,
            "network.layer3.1.bn2": self.network.layer3[1].bn2,
            "network.layer3.1.conv3": self.network.layer3[1].conv3,
            "network.layer3.1.bn3": self.network.layer3[1].bn3,
            "network.layer3.1.relu": self.network.layer3[1].relu,
            "network.layer3.2": self.network.layer3[2],
            "network.layer3.2.conv1": self.network.layer3[2].conv1,
            "network.layer3.2.bn1": self.network.layer3[2].bn1,
            "network.layer3.2.conv2": self.network.layer3[2].conv2,
            "network.layer3.2.bn2": self.network.layer3[2].bn2,
            "network.layer3.2.conv3": self.network.layer3[2].conv3,
            "network.layer3.2.bn3": self.network.layer3[2].bn3,
            "network.layer3.2.relu": self.network.layer3[2].relu,
            "network.layer3.3": self.network.layer3[3],
            "network.layer3.3.conv1": self.network.layer3[3].conv1,
            "network.layer3.3.bn1": self.network.layer3[3].bn1,
            "network.layer3.3.conv2": self.network.layer3[3].conv2,
            "network.layer3.3.bn2": self.network.layer3[3].bn2,
            "network.layer3.3.conv3": self.network.layer3[3].conv3,
            "network.layer3.3.bn3": self.network.layer3[3].bn3,
            "network.layer3.3.relu": self.network.layer3[3].relu,
            "network.layer3.4": self.network.layer3[4],
            "network.layer3.4.conv1": self.network.layer3[4].conv1,
            "network.layer3.4.bn1": self.network.layer3[4].bn1,
            "network.layer3.4.conv2": self.network.layer3[4].conv2,
            "network.layer3.4.bn2": self.network.layer3[4].bn2,
            "network.layer3.4.conv3": self.network.layer3[4].conv3,
            "network.layer3.4.bn3": self.network.layer3[4].bn3,
            "network.layer3.4.relu": self.network.layer3[4].relu,
            "network.layer3.5": self.network.layer3[5],
            "network.layer3.5.conv1": self.network.layer3[5].conv1,
            "network.layer3.5.bn1": self.network.layer3[5].bn1,
            "network.layer3.5.conv2": self.network.layer3[5].conv2,
            "network.layer3.5.bn2": self.network.layer3[5].bn2,
            "network.layer3.5.conv3": self.network.layer3[5].conv3,
            "network.layer3.5.bn3": self.network.layer3[5].bn3,
            "network.layer3.5.relu": self.network.layer3[5].relu,
            "network.layer4": self.network.layer4,
            "network.layer4.0": self.network.layer4[0],
            "network.layer4.0.conv1": self.network.layer4[0].conv1,
            "network.layer4.0.bn1": self.network.layer4[0].bn1,
            "network.layer4.0.conv2": self.network.layer4[0].conv2,
            "network.layer4.0.bn2": self.network.layer4[0].bn2,
            "network.layer4.0.conv3": self.network.layer4[0].conv3,
            "network.layer4.0.bn3": self.network.layer4[0].bn3,
            "network.layer4.0.relu": self.network.layer4[0].relu,
            "network.layer4.0.down_sample_layer": self.network.layer4[0].down_sample_layer,
            "network.layer4.0.down_sample_layer.0": self.network.layer4[0].down_sample_layer[0],
            "network.layer4.0.down_sample_layer.1": self.network.layer4[0].down_sample_layer[1],
            "network.layer4.1": self.network.layer4[1],
            "network.layer4.1.conv1": self.network.layer4[1].conv1,
            "network.layer4.1.bn1": self.network.layer4[1].bn1,
            "network.layer4.1.conv2": self.network.layer4[1].conv2,
            "network.layer4.1.bn2": self.network.layer4[1].bn2,
            "network.layer4.1.conv3": self.network.layer4[1].conv3,
            "network.layer4.1.bn3": self.network.layer4[1].bn3,
            "network.layer4.1.relu": self.network.layer4[1].relu,
            "network.layer4.2": self.network.layer4[2],
            "network.layer4.2.conv1": self.network.layer4[2].conv1,
            "network.layer4.2.bn1": self.network.layer4[2].bn1,
            "network.layer4.2.conv2": self.network.layer4[2].conv2,
            "network.layer4.2.bn2": self.network.layer4[2].bn2,
            "network.layer4.2.conv3": self.network.layer4[2].conv3,
            "network.layer4.2.bn3": self.network.layer4[2].bn3,
            "network.layer4.2.relu": self.network.layer4[2].relu,
        }
        self.origin_layer_names = {
            "network": self.network,
            "network.conv1": self.network.conv1,
            "network.bn1": self.network.bn1,
            "network.maxpool": self.network.maxpool,
            "network.layer1": self.network.layer1,
            "network.layer1.0": self.network.layer1[0],
            "network.layer1.0.conv1": self.network.layer1[0].conv1,
            "network.layer1.0.bn1": self.network.layer1[0].bn1,
            "network.layer1.0.conv2": self.network.layer1[0].conv2,
            "network.layer1.0.bn2": self.network.layer1[0].bn2,
            "network.layer1.0.conv3": self.network.layer1[0].conv3,
            "network.layer1.0.bn3": self.network.layer1[0].bn3,
            "network.layer1.0.relu": self.network.layer1[0].relu,
            "network.layer1.0.down_sample_layer": self.network.layer1[0].down_sample_layer,
            "network.layer1.0.down_sample_layer.0": self.network.layer1[0].down_sample_layer[0],
            "network.layer1.0.down_sample_layer.1": self.network.layer1[0].down_sample_layer[1],
            "network.layer1.1": self.network.layer1[1],
            "network.layer1.1.conv1": self.network.layer1[1].conv1,
            "network.layer1.1.bn1": self.network.layer1[1].bn1,
            "network.layer1.1.conv2": self.network.layer1[1].conv2,
            "network.layer1.1.bn2": self.network.layer1[1].bn2,
            "network.layer1.1.conv3": self.network.layer1[1].conv3,
            "network.layer1.1.bn3": self.network.layer1[1].bn3,
            "network.layer1.1.relu": self.network.layer1[1].relu,
            "network.layer1.2": self.network.layer1[2],
            "network.layer1.2.conv1": self.network.layer1[2].conv1,
            "network.layer1.2.bn1": self.network.layer1[2].bn1,
            "network.layer1.2.conv2": self.network.layer1[2].conv2,
            "network.layer1.2.bn2": self.network.layer1[2].bn2,
            "network.layer1.2.conv3": self.network.layer1[2].conv3,
            "network.layer1.2.bn3": self.network.layer1[2].bn3,
            "network.layer1.2.relu": self.network.layer1[2].relu,
            "network.layer2": self.network.layer2,
            "network.layer2.0": self.network.layer2[0],
            "network.layer2.0.conv1": self.network.layer2[0].conv1,
            "network.layer2.0.bn1": self.network.layer2[0].bn1,
            "network.layer2.0.conv2": self.network.layer2[0].conv2,
            "network.layer2.0.bn2": self.network.layer2[0].bn2,
            "network.layer2.0.conv3": self.network.layer2[0].conv3,
            "network.layer2.0.bn3": self.network.layer2[0].bn3,
            "network.layer2.0.relu": self.network.layer2[0].relu,
            "network.layer2.0.down_sample_layer": self.network.layer2[0].down_sample_layer,
            "network.layer2.0.down_sample_layer.0": self.network.layer2[0].down_sample_layer[0],
            "network.layer2.0.down_sample_layer.1": self.network.layer2[0].down_sample_layer[1],
            "network.layer2.1": self.network.layer2[1],
            "network.layer2.1.conv1": self.network.layer2[1].conv1,
            "network.layer2.1.bn1": self.network.layer2[1].bn1,
            "network.layer2.1.conv2": self.network.layer2[1].conv2,
            "network.layer2.1.bn2": self.network.layer2[1].bn2,
            "network.layer2.1.conv3": self.network.layer2[1].conv3,
            "network.layer2.1.bn3": self.network.layer2[1].bn3,
            "network.layer2.1.relu": self.network.layer2[1].relu,
            "network.layer2.2": self.network.layer2[2],
            "network.layer2.2.conv1": self.network.layer2[2].conv1,
            "network.layer2.2.bn1": self.network.layer2[2].bn1,
            "network.layer2.2.conv2": self.network.layer2[2].conv2,
            "network.layer2.2.bn2": self.network.layer2[2].bn2,
            "network.layer2.2.conv3": self.network.layer2[2].conv3,
            "network.layer2.2.bn3": self.network.layer2[2].bn3,
            "network.layer2.2.relu": self.network.layer2[2].relu,
            "network.layer2.3": self.network.layer2[3],
            "network.layer2.3.conv1": self.network.layer2[3].conv1,
            "network.layer2.3.bn1": self.network.layer2[3].bn1,
            "network.layer2.3.conv2": self.network.layer2[3].conv2,
            "network.layer2.3.bn2": self.network.layer2[3].bn2,
            "network.layer2.3.conv3": self.network.layer2[3].conv3,
            "network.layer2.3.bn3": self.network.layer2[3].bn3,
            "network.layer2.3.relu": self.network.layer2[3].relu,
            "network.layer3": self.network.layer3,
            "network.layer3.0": self.network.layer3[0],
            "network.layer3.0.conv1": self.network.layer3[0].conv1,
            "network.layer3.0.bn1": self.network.layer3[0].bn1,
            "network.layer3.0.conv2": self.network.layer3[0].conv2,
            "network.layer3.0.bn2": self.network.layer3[0].bn2,
            "network.layer3.0.conv3": self.network.layer3[0].conv3,
            "network.layer3.0.bn3": self.network.layer3[0].bn3,
            "network.layer3.0.relu": self.network.layer3[0].relu,
            "network.layer3.0.down_sample_layer": self.network.layer3[0].down_sample_layer,
            "network.layer3.0.down_sample_layer.0": self.network.layer3[0].down_sample_layer[0],
            "network.layer3.0.down_sample_layer.1": self.network.layer3[0].down_sample_layer[1],
            "network.layer3.1": self.network.layer3[1],
            "network.layer3.1.conv1": self.network.layer3[1].conv1,
            "network.layer3.1.bn1": self.network.layer3[1].bn1,
            "network.layer3.1.conv2": self.network.layer3[1].conv2,
            "network.layer3.1.bn2": self.network.layer3[1].bn2,
            "network.layer3.1.conv3": self.network.layer3[1].conv3,
            "network.layer3.1.bn3": self.network.layer3[1].bn3,
            "network.layer3.1.relu": self.network.layer3[1].relu,
            "network.layer3.2": self.network.layer3[2],
            "network.layer3.2.conv1": self.network.layer3[2].conv1,
            "network.layer3.2.bn1": self.network.layer3[2].bn1,
            "network.layer3.2.conv2": self.network.layer3[2].conv2,
            "network.layer3.2.bn2": self.network.layer3[2].bn2,
            "network.layer3.2.conv3": self.network.layer3[2].conv3,
            "network.layer3.2.bn3": self.network.layer3[2].bn3,
            "network.layer3.2.relu": self.network.layer3[2].relu,
            "network.layer3.3": self.network.layer3[3],
            "network.layer3.3.conv1": self.network.layer3[3].conv1,
            "network.layer3.3.bn1": self.network.layer3[3].bn1,
            "network.layer3.3.conv2": self.network.layer3[3].conv2,
            "network.layer3.3.bn2": self.network.layer3[3].bn2,
            "network.layer3.3.conv3": self.network.layer3[3].conv3,
            "network.layer3.3.bn3": self.network.layer3[3].bn3,
            "network.layer3.3.relu": self.network.layer3[3].relu,
            "network.layer3.4": self.network.layer3[4],
            "network.layer3.4.conv1": self.network.layer3[4].conv1,
            "network.layer3.4.bn1": self.network.layer3[4].bn1,
            "network.layer3.4.conv2": self.network.layer3[4].conv2,
            "network.layer3.4.bn2": self.network.layer3[4].bn2,
            "network.layer3.4.conv3": self.network.layer3[4].conv3,
            "network.layer3.4.bn3": self.network.layer3[4].bn3,
            "network.layer3.4.relu": self.network.layer3[4].relu,
            "network.layer3.5": self.network.layer3[5],
            "network.layer3.5.conv1": self.network.layer3[5].conv1,
            "network.layer3.5.bn1": self.network.layer3[5].bn1,
            "network.layer3.5.conv2": self.network.layer3[5].conv2,
            "network.layer3.5.bn2": self.network.layer3[5].bn2,
            "network.layer3.5.conv3": self.network.layer3[5].conv3,
            "network.layer3.5.bn3": self.network.layer3[5].bn3,
            "network.layer3.5.relu": self.network.layer3[5].relu,
            "network.layer4": self.network.layer4,
            "network.layer4.0": self.network.layer4[0],
            "network.layer4.0.conv1": self.network.layer4[0].conv1,
            "network.layer4.0.bn1": self.network.layer4[0].bn1,
            "network.layer4.0.conv2": self.network.layer4[0].conv2,
            "network.layer4.0.bn2": self.network.layer4[0].bn2,
            "network.layer4.0.conv3": self.network.layer4[0].conv3,
            "network.layer4.0.bn3": self.network.layer4[0].bn3,
            "network.layer4.0.relu": self.network.layer4[0].relu,
            "network.layer4.0.down_sample_layer": self.network.layer4[0].down_sample_layer,
            "network.layer4.0.down_sample_layer.0": self.network.layer4[0].down_sample_layer[0],
            "network.layer4.0.down_sample_layer.1": self.network.layer4[0].down_sample_layer[1],
            "network.layer4.1": self.network.layer4[1],
            "network.layer4.1.conv1": self.network.layer4[1].conv1,
            "network.layer4.1.bn1": self.network.layer4[1].bn1,
            "network.layer4.1.conv2": self.network.layer4[1].conv2,
            "network.layer4.1.bn2": self.network.layer4[1].bn2,
            "network.layer4.1.conv3": self.network.layer4[1].conv3,
            "network.layer4.1.bn3": self.network.layer4[1].bn3,
            "network.layer4.1.relu": self.network.layer4[1].relu,
            "network.layer4.2": self.network.layer4[2],
            "network.layer4.2.conv1": self.network.layer4[2].conv1,
            "network.layer4.2.bn1": self.network.layer4[2].bn1,
            "network.layer4.2.conv2": self.network.layer4[2].conv2,
            "network.layer4.2.bn2": self.network.layer4[2].bn2,
            "network.layer4.2.conv3": self.network.layer4[2].conv3,
            "network.layer4.2.bn3": self.network.layer4[2].bn3,
            "network.layer4.2.relu": self.network.layer4[2].relu,
        }

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

    def forward(self, x):
        _, _, c3, c4, c5 = self.network(x)
        features = self.fpn((c3, c4, c5))
        features = self.bottom_up(features)
        pred_loc, pred_label = self.multi_box(features)

        if not self.training:
            pred_label = torch.sigmoid(pred_label)
        pred_loc = pred_loc.to(torch.float32)
        pred_label = pred_label.to(torch.float32)

        return pred_loc, pred_label

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self, layer_name, new_layer):
        if 'network' == layer_name:
            self.network = new_layer
            self.layer_names["network"] = new_layer
            self.origin_layer_names["network"] = new_layer
        elif 'network.conv1' == layer_name:
            self.network.conv1 = new_layer
            self.layer_names["network.conv1"] = new_layer
            self.origin_layer_names["network.conv1"] = new_layer
        elif 'network.bn1' == layer_name:
            self.network.bn1 = new_layer
            self.layer_names["network.bn1"] = new_layer
            self.origin_layer_names["network.bn1"] = new_layer
        elif 'network.maxpool' == layer_name:
            self.network.maxpool = new_layer
            self.layer_names["network.maxpool"] = new_layer
            self.origin_layer_names["network.maxpool"] = new_layer
        elif 'network.layer1' == layer_name:
            self.network.layer1 = new_layer
            self.layer_names["network.layer1"] = new_layer
            self.origin_layer_names["network.layer1"] = new_layer
        elif 'network.layer1.0' == layer_name:
            self.network.layer1[0] = new_layer
            self.layer_names["network.layer1.0"] = new_layer
            self.origin_layer_names["network.layer1.0"] = new_layer
        elif 'network.layer1.0.conv1' == layer_name:
            self.network.layer1[0].conv1 = new_layer
            self.layer_names["network.layer1.0.conv1"] = new_layer
            self.origin_layer_names["network.layer1.0.conv1"] = new_layer
        elif 'network.layer1.0.bn1' == layer_name:
            self.network.layer1[0].bn1 = new_layer
            self.layer_names["network.layer1.0.bn1"] = new_layer
            self.origin_layer_names["network.layer1.0.bn1"] = new_layer
        elif 'network.layer1.0.conv2' == layer_name:
            self.network.layer1[0].conv2 = new_layer
            self.layer_names["network.layer1.0.conv2"] = new_layer
            self.origin_layer_names["network.layer1.0.conv2"] = new_layer
        elif 'network.layer1.0.bn2' == layer_name:
            self.network.layer1[0].bn2 = new_layer
            self.layer_names["network.layer1.0.bn2"] = new_layer
            self.origin_layer_names["network.layer1.0.bn2"] = new_layer
        elif 'network.layer1.0.conv3' == layer_name:
            self.network.layer1[0].conv3 = new_layer
            self.layer_names["network.layer1.0.conv3"] = new_layer
            self.origin_layer_names["network.layer1.0.conv3"] = new_layer
        elif 'network.layer1.0.bn3' == layer_name:
            self.network.layer1[0].bn3 = new_layer
            self.layer_names["network.layer1.0.bn3"] = new_layer
            self.origin_layer_names["network.layer1.0.bn3"] = new_layer
        elif 'network.layer1.0.relu' == layer_name:
            self.network.layer1[0].relu = new_layer
            self.layer_names["network.layer1.0.relu"] = new_layer
            self.origin_layer_names["network.layer1.0.relu"] = new_layer
        elif 'network.layer1.0.down_sample_layer' == layer_name:
            self.network.layer1[0].down_sample_layer = new_layer
            self.layer_names["network.layer1.0.down_sample_layer"] = new_layer
            self.origin_layer_names["network.layer1.0.down_sample_layer"] = new_layer
        elif 'network.layer1.0.down_sample_layer.0' == layer_name:
            self.network.layer1[0].down_sample_layer[0] = new_layer
            self.layer_names["network.layer1.0.down_sample_layer.0"] = new_layer
            self.origin_layer_names["network.layer1.0.down_sample_layer.0"] = new_layer
        elif 'network.layer1.0.down_sample_layer.1' == layer_name:
            self.network.layer1[0].down_sample_layer[1] = new_layer
            self.layer_names["network.layer1.0.down_sample_layer.1"] = new_layer
            self.origin_layer_names["network.layer1.0.down_sample_layer.1"] = new_layer
        elif 'network.layer1.1' == layer_name:
            self.network.layer1[1] = new_layer
            self.layer_names["network.layer1.1"] = new_layer
            self.origin_layer_names["network.layer1.1"] = new_layer
        elif 'network.layer1.1.conv1' == layer_name:
            self.network.layer1[1].conv1 = new_layer
            self.layer_names["network.layer1.1.conv1"] = new_layer
            self.origin_layer_names["network.layer1.1.conv1"] = new_layer
        elif 'network.layer1.1.bn1' == layer_name:
            self.network.layer1[1].bn1 = new_layer
            self.layer_names["network.layer1.1.bn1"] = new_layer
            self.origin_layer_names["network.layer1.1.bn1"] = new_layer
        elif 'network.layer1.1.conv2' == layer_name:
            self.network.layer1[1].conv2 = new_layer
            self.layer_names["network.layer1.1.conv2"] = new_layer
            self.origin_layer_names["network.layer1.1.conv2"] = new_layer
        elif 'network.layer1.1.bn2' == layer_name:
            self.network.layer1[1].bn2 = new_layer
            self.layer_names["network.layer1.1.bn2"] = new_layer
            self.origin_layer_names["network.layer1.1.bn2"] = new_layer
        elif 'network.layer1.1.conv3' == layer_name:
            self.network.layer1[1].conv3 = new_layer
            self.layer_names["network.layer1.1.conv3"] = new_layer
            self.origin_layer_names["network.layer1.1.conv3"] = new_layer
        elif 'network.layer1.1.bn3' == layer_name:
            self.network.layer1[1].bn3 = new_layer
            self.layer_names["network.layer1.1.bn3"] = new_layer
            self.origin_layer_names["network.layer1.1.bn3"] = new_layer
        elif 'network.layer1.1.relu' == layer_name:
            self.network.layer1[1].relu = new_layer
            self.layer_names["network.layer1.1.relu"] = new_layer
            self.origin_layer_names["network.layer1.1.relu"] = new_layer
        elif 'network.layer1.2' == layer_name:
            self.network.layer1[2] = new_layer
            self.layer_names["network.layer1.2"] = new_layer
            self.origin_layer_names["network.layer1.2"] = new_layer
        elif 'network.layer1.2.conv1' == layer_name:
            self.network.layer1[2].conv1 = new_layer
            self.layer_names["network.layer1.2.conv1"] = new_layer
            self.origin_layer_names["network.layer1.2.conv1"] = new_layer
        elif 'network.layer1.2.bn1' == layer_name:
            self.network.layer1[2].bn1 = new_layer
            self.layer_names["network.layer1.2.bn1"] = new_layer
            self.origin_layer_names["network.layer1.2.bn1"] = new_layer
        elif 'network.layer1.2.conv2' == layer_name:
            self.network.layer1[2].conv2 = new_layer
            self.layer_names["network.layer1.2.conv2"] = new_layer
            self.origin_layer_names["network.layer1.2.conv2"] = new_layer
        elif 'network.layer1.2.bn2' == layer_name:
            self.network.layer1[2].bn2 = new_layer
            self.layer_names["network.layer1.2.bn2"] = new_layer
            self.origin_layer_names["network.layer1.2.bn2"] = new_layer
        elif 'network.layer1.2.conv3' == layer_name:
            self.network.layer1[2].conv3 = new_layer
            self.layer_names["network.layer1.2.conv3"] = new_layer
            self.origin_layer_names["network.layer1.2.conv3"] = new_layer
        elif 'network.layer1.2.bn3' == layer_name:
            self.network.layer1[2].bn3 = new_layer
            self.layer_names["network.layer1.2.bn3"] = new_layer
            self.origin_layer_names["network.layer1.2.bn3"] = new_layer
        elif 'network.layer1.2.relu' == layer_name:
            self.network.layer1[2].relu = new_layer
            self.layer_names["network.layer1.2.relu"] = new_layer
            self.origin_layer_names["network.layer1.2.relu"] = new_layer
        elif 'network.layer2' == layer_name:
            self.network.layer2 = new_layer
            self.layer_names["network.layer2"] = new_layer
            self.origin_layer_names["network.layer2"] = new_layer
        elif 'network.layer2.0' == layer_name:
            self.network.layer2[0] = new_layer
            self.layer_names["network.layer2.0"] = new_layer
            self.origin_layer_names["network.layer2.0"] = new_layer
        elif 'network.layer2.0.conv1' == layer_name:
            self.network.layer2[0].conv1 = new_layer
            self.layer_names["network.layer2.0.conv1"] = new_layer
            self.origin_layer_names["network.layer2.0.conv1"] = new_layer
        elif 'network.layer2.0.bn1' == layer_name:
            self.network.layer2[0].bn1 = new_layer
            self.layer_names["network.layer2.0.bn1"] = new_layer
            self.origin_layer_names["network.layer2.0.bn1"] = new_layer
        elif 'network.layer2.0.conv2' == layer_name:
            self.network.layer2[0].conv2 = new_layer
            self.layer_names["network.layer2.0.conv2"] = new_layer
            self.origin_layer_names["network.layer2.0.conv2"] = new_layer
        elif 'network.layer2.0.bn2' == layer_name:
            self.network.layer2[0].bn2 = new_layer
            self.layer_names["network.layer2.0.bn2"] = new_layer
            self.origin_layer_names["network.layer2.0.bn2"] = new_layer
        elif 'network.layer2.0.conv3' == layer_name:
            self.network.layer2[0].conv3 = new_layer
            self.layer_names["network.layer2.0.conv3"] = new_layer
            self.origin_layer_names["network.layer2.0.conv3"] = new_layer
        elif 'network.layer2.0.bn3' == layer_name:
            self.network.layer2[0].bn3 = new_layer
            self.layer_names["network.layer2.0.bn3"] = new_layer
            self.origin_layer_names["network.layer2.0.bn3"] = new_layer
        elif 'network.layer2.0.relu' == layer_name:
            self.network.layer2[0].relu = new_layer
            self.layer_names["network.layer2.0.relu"] = new_layer
            self.origin_layer_names["network.layer2.0.relu"] = new_layer
        elif 'network.layer2.0.down_sample_layer' == layer_name:
            self.network.layer2[0].down_sample_layer = new_layer
            self.layer_names["network.layer2.0.down_sample_layer"] = new_layer
            self.origin_layer_names["network.layer2.0.down_sample_layer"] = new_layer
        elif 'network.layer2.0.down_sample_layer.0' == layer_name:
            self.network.layer2[0].down_sample_layer[0] = new_layer
            self.layer_names["network.layer2.0.down_sample_layer.0"] = new_layer
            self.origin_layer_names["network.layer2.0.down_sample_layer.0"] = new_layer
        elif 'network.layer2.0.down_sample_layer.1' == layer_name:
            self.network.layer2[0].down_sample_layer[1] = new_layer
            self.layer_names["network.layer2.0.down_sample_layer.1"] = new_layer
            self.origin_layer_names["network.layer2.0.down_sample_layer.1"] = new_layer
        elif 'network.layer2.1' == layer_name:
            self.network.layer2[1] = new_layer
            self.layer_names["network.layer2.1"] = new_layer
            self.origin_layer_names["network.layer2.1"] = new_layer
        elif 'network.layer2.1.conv1' == layer_name:
            self.network.layer2[1].conv1 = new_layer
            self.layer_names["network.layer2.1.conv1"] = new_layer
            self.origin_layer_names["network.layer2.1.conv1"] = new_layer
        elif 'network.layer2.1.bn1' == layer_name:
            self.network.layer2[1].bn1 = new_layer
            self.layer_names["network.layer2.1.bn1"] = new_layer
            self.origin_layer_names["network.layer2.1.bn1"] = new_layer
        elif 'network.layer2.1.conv2' == layer_name:
            self.network.layer2[1].conv2 = new_layer
            self.layer_names["network.layer2.1.conv2"] = new_layer
            self.origin_layer_names["network.layer2.1.conv2"] = new_layer
        elif 'network.layer2.1.bn2' == layer_name:
            self.network.layer2[1].bn2 = new_layer
            self.layer_names["network.layer2.1.bn2"] = new_layer
            self.origin_layer_names["network.layer2.1.bn2"] = new_layer
        elif 'network.layer2.1.conv3' == layer_name:
            self.network.layer2[1].conv3 = new_layer
            self.layer_names["network.layer2.1.conv3"] = new_layer
            self.origin_layer_names["network.layer2.1.conv3"] = new_layer
        elif 'network.layer2.1.bn3' == layer_name:
            self.network.layer2[1].bn3 = new_layer
            self.layer_names["network.layer2.1.bn3"] = new_layer
            self.origin_layer_names["network.layer2.1.bn3"] = new_layer
        elif 'network.layer2.1.relu' == layer_name:
            self.network.layer2[1].relu = new_layer
            self.layer_names["network.layer2.1.relu"] = new_layer
            self.origin_layer_names["network.layer2.1.relu"] = new_layer
        elif 'network.layer2.2' == layer_name:
            self.network.layer2[2] = new_layer
            self.layer_names["network.layer2.2"] = new_layer
            self.origin_layer_names["network.layer2.2"] = new_layer
        elif 'network.layer2.2.conv1' == layer_name:
            self.network.layer2[2].conv1 = new_layer
            self.layer_names["network.layer2.2.conv1"] = new_layer
            self.origin_layer_names["network.layer2.2.conv1"] = new_layer
        elif 'network.layer2.2.bn1' == layer_name:
            self.network.layer2[2].bn1 = new_layer
            self.layer_names["network.layer2.2.bn1"] = new_layer
            self.origin_layer_names["network.layer2.2.bn1"] = new_layer
        elif 'network.layer2.2.conv2' == layer_name:
            self.network.layer2[2].conv2 = new_layer
            self.layer_names["network.layer2.2.conv2"] = new_layer
            self.origin_layer_names["network.layer2.2.conv2"] = new_layer
        elif 'network.layer2.2.bn2' == layer_name:
            self.network.layer2[2].bn2 = new_layer
            self.layer_names["network.layer2.2.bn2"] = new_layer
            self.origin_layer_names["network.layer2.2.bn2"] = new_layer
        elif 'network.layer2.2.conv3' == layer_name:
            self.network.layer2[2].conv3 = new_layer
            self.layer_names["network.layer2.2.conv3"] = new_layer
            self.origin_layer_names["network.layer2.2.conv3"] = new_layer
        elif 'network.layer2.2.bn3' == layer_name:
            self.network.layer2[2].bn3 = new_layer
            self.layer_names["network.layer2.2.bn3"] = new_layer
            self.origin_layer_names["network.layer2.2.bn3"] = new_layer
        elif 'network.layer2.2.relu' == layer_name:
            self.network.layer2[2].relu = new_layer
            self.layer_names["network.layer2.2.relu"] = new_layer
            self.origin_layer_names["network.layer2.2.relu"] = new_layer
        elif 'network.layer2.3' == layer_name:
            self.network.layer2[3] = new_layer
            self.layer_names["network.layer2.3"] = new_layer
            self.origin_layer_names["network.layer2.3"] = new_layer
        elif 'network.layer2.3.conv1' == layer_name:
            self.network.layer2[3].conv1 = new_layer
            self.layer_names["network.layer2.3.conv1"] = new_layer
            self.origin_layer_names["network.layer2.3.conv1"] = new_layer
        elif 'network.layer2.3.bn1' == layer_name:
            self.network.layer2[3].bn1 = new_layer
            self.layer_names["network.layer2.3.bn1"] = new_layer
            self.origin_layer_names["network.layer2.3.bn1"] = new_layer
        elif 'network.layer2.3.conv2' == layer_name:
            self.network.layer2[3].conv2 = new_layer
            self.layer_names["network.layer2.3.conv2"] = new_layer
            self.origin_layer_names["network.layer2.3.conv2"] = new_layer
        elif 'network.layer2.3.bn2' == layer_name:
            self.network.layer2[3].bn2 = new_layer
            self.layer_names["network.layer2.3.bn2"] = new_layer
            self.origin_layer_names["network.layer2.3.bn2"] = new_layer
        elif 'network.layer2.3.conv3' == layer_name:
            self.network.layer2[3].conv3 = new_layer
            self.layer_names["network.layer2.3.conv3"] = new_layer
            self.origin_layer_names["network.layer2.3.conv3"] = new_layer
        elif 'network.layer2.3.bn3' == layer_name:
            self.network.layer2[3].bn3 = new_layer
            self.layer_names["network.layer2.3.bn3"] = new_layer
            self.origin_layer_names["network.layer2.3.bn3"] = new_layer
        elif 'network.layer2.3.relu' == layer_name:
            self.network.layer2[3].relu = new_layer
            self.layer_names["network.layer2.3.relu"] = new_layer
            self.origin_layer_names["network.layer2.3.relu"] = new_layer
        elif 'network.layer3' == layer_name:
            self.network.layer3 = new_layer
            self.layer_names["network.layer3"] = new_layer
            self.origin_layer_names["network.layer3"] = new_layer
        elif 'network.layer3.0' == layer_name:
            self.network.layer3[0] = new_layer
            self.layer_names["network.layer3.0"] = new_layer
            self.origin_layer_names["network.layer3.0"] = new_layer
        elif 'network.layer3.0.conv1' == layer_name:
            self.network.layer3[0].conv1 = new_layer
            self.layer_names["network.layer3.0.conv1"] = new_layer
            self.origin_layer_names["network.layer3.0.conv1"] = new_layer
        elif 'network.layer3.0.bn1' == layer_name:
            self.network.layer3[0].bn1 = new_layer
            self.layer_names["network.layer3.0.bn1"] = new_layer
            self.origin_layer_names["network.layer3.0.bn1"] = new_layer
        elif 'network.layer3.0.conv2' == layer_name:
            self.network.layer3[0].conv2 = new_layer
            self.layer_names["network.layer3.0.conv2"] = new_layer
            self.origin_layer_names["network.layer3.0.conv2"] = new_layer
        elif 'network.layer3.0.bn2' == layer_name:
            self.network.layer3[0].bn2 = new_layer
            self.layer_names["network.layer3.0.bn2"] = new_layer
            self.origin_layer_names["network.layer3.0.bn2"] = new_layer
        elif 'network.layer3.0.conv3' == layer_name:
            self.network.layer3[0].conv3 = new_layer
            self.layer_names["network.layer3.0.conv3"] = new_layer
            self.origin_layer_names["network.layer3.0.conv3"] = new_layer
        elif 'network.layer3.0.bn3' == layer_name:
            self.network.layer3[0].bn3 = new_layer
            self.layer_names["network.layer3.0.bn3"] = new_layer
            self.origin_layer_names["network.layer3.0.bn3"] = new_layer
        elif 'network.layer3.0.relu' == layer_name:
            self.network.layer3[0].relu = new_layer
            self.layer_names["network.layer3.0.relu"] = new_layer
            self.origin_layer_names["network.layer3.0.relu"] = new_layer
        elif 'network.layer3.0.down_sample_layer' == layer_name:
            self.network.layer3[0].down_sample_layer = new_layer
            self.layer_names["network.layer3.0.down_sample_layer"] = new_layer
            self.origin_layer_names["network.layer3.0.down_sample_layer"] = new_layer
        elif 'network.layer3.0.down_sample_layer.0' == layer_name:
            self.network.layer3[0].down_sample_layer[0] = new_layer
            self.layer_names["network.layer3.0.down_sample_layer.0"] = new_layer
            self.origin_layer_names["network.layer3.0.down_sample_layer.0"] = new_layer
        elif 'network.layer3.0.down_sample_layer.1' == layer_name:
            self.network.layer3[0].down_sample_layer[1] = new_layer
            self.layer_names["network.layer3.0.down_sample_layer.1"] = new_layer
            self.origin_layer_names["network.layer3.0.down_sample_layer.1"] = new_layer
        elif 'network.layer3.1' == layer_name:
            self.network.layer3[1] = new_layer
            self.layer_names["network.layer3.1"] = new_layer
            self.origin_layer_names["network.layer3.1"] = new_layer
        elif 'network.layer3.1.conv1' == layer_name:
            self.network.layer3[1].conv1 = new_layer
            self.layer_names["network.layer3.1.conv1"] = new_layer
            self.origin_layer_names["network.layer3.1.conv1"] = new_layer
        elif 'network.layer3.1.bn1' == layer_name:
            self.network.layer3[1].bn1 = new_layer
            self.layer_names["network.layer3.1.bn1"] = new_layer
            self.origin_layer_names["network.layer3.1.bn1"] = new_layer
        elif 'network.layer3.1.conv2' == layer_name:
            self.network.layer3[1].conv2 = new_layer
            self.layer_names["network.layer3.1.conv2"] = new_layer
            self.origin_layer_names["network.layer3.1.conv2"] = new_layer
        elif 'network.layer3.1.bn2' == layer_name:
            self.network.layer3[1].bn2 = new_layer
            self.layer_names["network.layer3.1.bn2"] = new_layer
            self.origin_layer_names["network.layer3.1.bn2"] = new_layer
        elif 'network.layer3.1.conv3' == layer_name:
            self.network.layer3[1].conv3 = new_layer
            self.layer_names["network.layer3.1.conv3"] = new_layer
            self.origin_layer_names["network.layer3.1.conv3"] = new_layer
        elif 'network.layer3.1.bn3' == layer_name:
            self.network.layer3[1].bn3 = new_layer
            self.layer_names["network.layer3.1.bn3"] = new_layer
            self.origin_layer_names["network.layer3.1.bn3"] = new_layer
        elif 'network.layer3.1.relu' == layer_name:
            self.network.layer3[1].relu = new_layer
            self.layer_names["network.layer3.1.relu"] = new_layer
            self.origin_layer_names["network.layer3.1.relu"] = new_layer
        elif 'network.layer3.2' == layer_name:
            self.network.layer3[2] = new_layer
            self.layer_names["network.layer3.2"] = new_layer
            self.origin_layer_names["network.layer3.2"] = new_layer
        elif 'network.layer3.2.conv1' == layer_name:
            self.network.layer3[2].conv1 = new_layer
            self.layer_names["network.layer3.2.conv1"] = new_layer
            self.origin_layer_names["network.layer3.2.conv1"] = new_layer
        elif 'network.layer3.2.bn1' == layer_name:
            self.network.layer3[2].bn1 = new_layer
            self.layer_names["network.layer3.2.bn1"] = new_layer
            self.origin_layer_names["network.layer3.2.bn1"] = new_layer
        elif 'network.layer3.2.conv2' == layer_name:
            self.network.layer3[2].conv2 = new_layer
            self.layer_names["network.layer3.2.conv2"] = new_layer
            self.origin_layer_names["network.layer3.2.conv2"] = new_layer
        elif 'network.layer3.2.bn2' == layer_name:
            self.network.layer3[2].bn2 = new_layer
            self.layer_names["network.layer3.2.bn2"] = new_layer
            self.origin_layer_names["network.layer3.2.bn2"] = new_layer
        elif 'network.layer3.2.conv3' == layer_name:
            self.network.layer3[2].conv3 = new_layer
            self.layer_names["network.layer3.2.conv3"] = new_layer
            self.origin_layer_names["network.layer3.2.conv3"] = new_layer
        elif 'network.layer3.2.bn3' == layer_name:
            self.network.layer3[2].bn3 = new_layer
            self.layer_names["network.layer3.2.bn3"] = new_layer
            self.origin_layer_names["network.layer3.2.bn3"] = new_layer
        elif 'network.layer3.2.relu' == layer_name:
            self.network.layer3[2].relu = new_layer
            self.layer_names["network.layer3.2.relu"] = new_layer
            self.origin_layer_names["network.layer3.2.relu"] = new_layer
        elif 'network.layer3.3' == layer_name:
            self.network.layer3[3] = new_layer
            self.layer_names["network.layer3.3"] = new_layer
            self.origin_layer_names["network.layer3.3"] = new_layer
        elif 'network.layer3.3.conv1' == layer_name:
            self.network.layer3[3].conv1 = new_layer
            self.layer_names["network.layer3.3.conv1"] = new_layer
            self.origin_layer_names["network.layer3.3.conv1"] = new_layer
        elif 'network.layer3.3.bn1' == layer_name:
            self.network.layer3[3].bn1 = new_layer
            self.layer_names["network.layer3.3.bn1"] = new_layer
            self.origin_layer_names["network.layer3.3.bn1"] = new_layer
        elif 'network.layer3.3.conv2' == layer_name:
            self.network.layer3[3].conv2 = new_layer
            self.layer_names["network.layer3.3.conv2"] = new_layer
            self.origin_layer_names["network.layer3.3.conv2"] = new_layer
        elif 'network.layer3.3.bn2' == layer_name:
            self.network.layer3[3].bn2 = new_layer
            self.layer_names["network.layer3.3.bn2"] = new_layer
            self.origin_layer_names["network.layer3.3.bn2"] = new_layer
        elif 'network.layer3.3.conv3' == layer_name:
            self.network.layer3[3].conv3 = new_layer
            self.layer_names["network.layer3.3.conv3"] = new_layer
            self.origin_layer_names["network.layer3.3.conv3"] = new_layer
        elif 'network.layer3.3.bn3' == layer_name:
            self.network.layer3[3].bn3 = new_layer
            self.layer_names["network.layer3.3.bn3"] = new_layer
            self.origin_layer_names["network.layer3.3.bn3"] = new_layer
        elif 'network.layer3.3.relu' == layer_name:
            self.network.layer3[3].relu = new_layer
            self.layer_names["network.layer3.3.relu"] = new_layer
            self.origin_layer_names["network.layer3.3.relu"] = new_layer
        elif 'network.layer3.4' == layer_name:
            self.network.layer3[4] = new_layer
            self.layer_names["network.layer3.4"] = new_layer
            self.origin_layer_names["network.layer3.4"] = new_layer
        elif 'network.layer3.4.conv1' == layer_name:
            self.network.layer3[4].conv1 = new_layer
            self.layer_names["network.layer3.4.conv1"] = new_layer
            self.origin_layer_names["network.layer3.4.conv1"] = new_layer
        elif 'network.layer3.4.bn1' == layer_name:
            self.network.layer3[4].bn1 = new_layer
            self.layer_names["network.layer3.4.bn1"] = new_layer
            self.origin_layer_names["network.layer3.4.bn1"] = new_layer
        elif 'network.layer3.4.conv2' == layer_name:
            self.network.layer3[4].conv2 = new_layer
            self.layer_names["network.layer3.4.conv2"] = new_layer
            self.origin_layer_names["network.layer3.4.conv2"] = new_layer
        elif 'network.layer3.4.bn2' == layer_name:
            self.network.layer3[4].bn2 = new_layer
            self.layer_names["network.layer3.4.bn2"] = new_layer
            self.origin_layer_names["network.layer3.4.bn2"] = new_layer
        elif 'network.layer3.4.conv3' == layer_name:
            self.network.layer3[4].conv3 = new_layer
            self.layer_names["network.layer3.4.conv3"] = new_layer
            self.origin_layer_names["network.layer3.4.conv3"] = new_layer
        elif 'network.layer3.4.bn3' == layer_name:
            self.network.layer3[4].bn3 = new_layer
            self.layer_names["network.layer3.4.bn3"] = new_layer
            self.origin_layer_names["network.layer3.4.bn3"] = new_layer
        elif 'network.layer3.4.relu' == layer_name:
            self.network.layer3[4].relu = new_layer
            self.layer_names["network.layer3.4.relu"] = new_layer
            self.origin_layer_names["network.layer3.4.relu"] = new_layer
        elif 'network.layer3.5' == layer_name:
            self.network.layer3[5] = new_layer
            self.layer_names["network.layer3.5"] = new_layer
            self.origin_layer_names["network.layer3.5"] = new_layer
        elif 'network.layer3.5.conv1' == layer_name:
            self.network.layer3[5].conv1 = new_layer
            self.layer_names["network.layer3.5.conv1"] = new_layer
            self.origin_layer_names["network.layer3.5.conv1"] = new_layer
        elif 'network.layer3.5.bn1' == layer_name:
            self.network.layer3[5].bn1 = new_layer
            self.layer_names["network.layer3.5.bn1"] = new_layer
            self.origin_layer_names["network.layer3.5.bn1"] = new_layer
        elif 'network.layer3.5.conv2' == layer_name:
            self.network.layer3[5].conv2 = new_layer
            self.layer_names["network.layer3.5.conv2"] = new_layer
            self.origin_layer_names["network.layer3.5.conv2"] = new_layer
        elif 'network.layer3.5.bn2' == layer_name:
            self.network.layer3[5].bn2 = new_layer
            self.layer_names["network.layer3.5.bn2"] = new_layer
            self.origin_layer_names["network.layer3.5.bn2"] = new_layer
        elif 'network.layer3.5.conv3' == layer_name:
            self.network.layer3[5].conv3 = new_layer
            self.layer_names["network.layer3.5.conv3"] = new_layer
            self.origin_layer_names["network.layer3.5.conv3"] = new_layer
        elif 'network.layer3.5.bn3' == layer_name:
            self.network.layer3[5].bn3 = new_layer
            self.layer_names["network.layer3.5.bn3"] = new_layer
            self.origin_layer_names["network.layer3.5.bn3"] = new_layer
        elif 'network.layer3.5.relu' == layer_name:
            self.network.layer3[5].relu = new_layer
            self.layer_names["network.layer3.5.relu"] = new_layer
            self.origin_layer_names["network.layer3.5.relu"] = new_layer
        elif 'network.layer4' == layer_name:
            self.network.layer4 = new_layer
            self.layer_names["network.layer4"] = new_layer
            self.origin_layer_names["network.layer4"] = new_layer
        elif 'network.layer4.0' == layer_name:
            self.network.layer4[0] = new_layer
            self.layer_names["network.layer4.0"] = new_layer
            self.origin_layer_names["network.layer4.0"] = new_layer
        elif 'network.layer4.0.conv1' == layer_name:
            self.network.layer4[0].conv1 = new_layer
            self.layer_names["network.layer4.0.conv1"] = new_layer
            self.origin_layer_names["network.layer4.0.conv1"] = new_layer
        elif 'network.layer4.0.bn1' == layer_name:
            self.network.layer4[0].bn1 = new_layer
            self.layer_names["network.layer4.0.bn1"] = new_layer
            self.origin_layer_names["network.layer4.0.bn1"] = new_layer
        elif 'network.layer4.0.conv2' == layer_name:
            self.network.layer4[0].conv2 = new_layer
            self.layer_names["network.layer4.0.conv2"] = new_layer
            self.origin_layer_names["network.layer4.0.conv2"] = new_layer
        elif 'network.layer4.0.bn2' == layer_name:
            self.network.layer4[0].bn2 = new_layer
            self.layer_names["network.layer4.0.bn2"] = new_layer
            self.origin_layer_names["network.layer4.0.bn2"] = new_layer
        elif 'network.layer4.0.conv3' == layer_name:
            self.network.layer4[0].conv3 = new_layer
            self.layer_names["network.layer4.0.conv3"] = new_layer
            self.origin_layer_names["network.layer4.0.conv3"] = new_layer
        elif 'network.layer4.0.bn3' == layer_name:
            self.network.layer4[0].bn3 = new_layer
            self.layer_names["network.layer4.0.bn3"] = new_layer
            self.origin_layer_names["network.layer4.0.bn3"] = new_layer
        elif 'network.layer4.0.relu' == layer_name:
            self.network.layer4[0].relu = new_layer
            self.layer_names["network.layer4.0.relu"] = new_layer
            self.origin_layer_names["network.layer4.0.relu"] = new_layer
        elif 'network.layer4.0.down_sample_layer' == layer_name:
            self.network.layer4[0].down_sample_layer = new_layer
            self.layer_names["network.layer4.0.down_sample_layer"] = new_layer
            self.origin_layer_names["network.layer4.0.down_sample_layer"] = new_layer
        elif 'network.layer4.0.down_sample_layer.0' == layer_name:
            self.network.layer4[0].down_sample_layer[0] = new_layer
            self.layer_names["network.layer4.0.down_sample_layer.0"] = new_layer
            self.origin_layer_names["network.layer4.0.down_sample_layer.0"] = new_layer
        elif 'network.layer4.0.down_sample_layer.1' == layer_name:
            self.network.layer4[0].down_sample_layer[1] = new_layer
            self.layer_names["network.layer4.0.down_sample_layer.1"] = new_layer
            self.origin_layer_names["network.layer4.0.down_sample_layer.1"] = new_layer
        elif 'network.layer4.1' == layer_name:
            self.network.layer4[1] = new_layer
            self.layer_names["network.layer4.1"] = new_layer
            self.origin_layer_names["network.layer4.1"] = new_layer
        elif 'network.layer4.1.conv1' == layer_name:
            self.network.layer4[1].conv1 = new_layer
            self.layer_names["network.layer4.1.conv1"] = new_layer
            self.origin_layer_names["network.layer4.1.conv1"] = new_layer
        elif 'network.layer4.1.bn1' == layer_name:
            self.network.layer4[1].bn1 = new_layer
            self.layer_names["network.layer4.1.bn1"] = new_layer
            self.origin_layer_names["network.layer4.1.bn1"] = new_layer
        elif 'network.layer4.1.conv2' == layer_name:
            self.network.layer4[1].conv2 = new_layer
            self.layer_names["network.layer4.1.conv2"] = new_layer
            self.origin_layer_names["network.layer4.1.conv2"] = new_layer
        elif 'network.layer4.1.bn2' == layer_name:
            self.network.layer4[1].bn2 = new_layer
            self.layer_names["network.layer4.1.bn2"] = new_layer
            self.origin_layer_names["network.layer4.1.bn2"] = new_layer
        elif 'network.layer4.1.conv3' == layer_name:
            self.network.layer4[1].conv3 = new_layer
            self.layer_names["network.layer4.1.conv3"] = new_layer
            self.origin_layer_names["network.layer4.1.conv3"] = new_layer
        elif 'network.layer4.1.bn3' == layer_name:
            self.network.layer4[1].bn3 = new_layer
            self.layer_names["network.layer4.1.bn3"] = new_layer
            self.origin_layer_names["network.layer4.1.bn3"] = new_layer
        elif 'network.layer4.1.relu' == layer_name:
            self.network.layer4[1].relu = new_layer
            self.layer_names["network.layer4.1.relu"] = new_layer
            self.origin_layer_names["network.layer4.1.relu"] = new_layer
        elif 'network.layer4.2' == layer_name:
            self.network.layer4[2] = new_layer
            self.layer_names["network.layer4.2"] = new_layer
            self.origin_layer_names["network.layer4.2"] = new_layer
        elif 'network.layer4.2.conv1' == layer_name:
            self.network.layer4[2].conv1 = new_layer
            self.layer_names["network.layer4.2.conv1"] = new_layer
            self.origin_layer_names["network.layer4.2.conv1"] = new_layer
        elif 'network.layer4.2.bn1' == layer_name:
            self.network.layer4[2].bn1 = new_layer
            self.layer_names["network.layer4.2.bn1"] = new_layer
            self.origin_layer_names["network.layer4.2.bn1"] = new_layer
        elif 'network.layer4.2.conv2' == layer_name:
            self.network.layer4[2].conv2 = new_layer
            self.layer_names["network.layer4.2.conv2"] = new_layer
            self.origin_layer_names["network.layer4.2.conv2"] = new_layer
        elif 'network.layer4.2.bn2' == layer_name:
            self.network.layer4[2].bn2 = new_layer
            self.layer_names["network.layer4.2.bn2"] = new_layer
            self.origin_layer_names["network.layer4.2.bn2"] = new_layer
        elif 'network.layer4.2.conv3' == layer_name:
            self.network.layer4[2].conv3 = new_layer
            self.layer_names["network.layer4.2.conv3"] = new_layer
            self.origin_layer_names["network.layer4.2.conv3"] = new_layer
        elif 'network.layer4.2.bn3' == layer_name:
            self.network.layer4[2].bn3 = new_layer
            self.layer_names["network.layer4.2.bn3"] = new_layer
            self.origin_layer_names["network.layer4.2.bn3"] = new_layer
        elif 'network.layer4.2.relu' == layer_name:
            self.network.layer4[2].relu = new_layer
            self.layer_names["network.layer4.2.relu"] = new_layer
            self.origin_layer_names["network.layer4.2.relu"] = new_layer

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

