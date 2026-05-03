"""model"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comparer import compare_models
# from models.PatchCore.src import model_ms
import model_ms


class Bottleneck(nn.Module):
    """Bottleneck"""
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """construct"""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet"""

    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.avgpool_same = nn.AvgPool2d(kernel_size=3, stride=1)

        self.mean = torch.mean
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer"""
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return [c2, c3]


def wide_resnet50_2():
    return ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2)


if __name__ == '__main__':
    import troubleshooter as ts
    net_torch = wide_resnet50_2()
    net_torch.eval()
    # inpu = torch.randn(1, 3, 224, 224)
    # out_torch = net_torch(inpu)
    # print("type", type(out_torch))
    # print(out_torch[0].shape, out_torch[1].shape)
    net_ms = model_ms.wide_resnet50_2()
    net_ms.set_train(False)
    input_size = (1, 3, 224, 224)
    diff_finder = ts.migrator.NetDifferenceFinder(pt_net=net_torch, ms_net=net_ms, fix_seed=0, auto_conv_ckpt=1)  #
    diff_finder.compare(auto_inputs=((input_size, np.float32),))
    compare_models(net_ms, net_torch, np_data=[np.ones([1, 3, 224, 224])])
