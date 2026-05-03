"""Se_ResNet."""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm


def _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return torch.tensor(weight, dtype=torch.float32)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return torch.tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=1)


def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=1)


def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=1)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9, )


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9, )


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Linear(in_channel, out_channel, bias=True)


class SELayer(nn.Module):
    """SE Layer"""

    def __init__(self, out_channel, reduction=16):
        super(SELayer, self).__init__()
        self.se_global_pool = torch.mean
        self.se_dense_0 = _fc(out_channel, int(out_channel / 4))
        self.se_dense_1 = _fc(int(out_channel / 4), out_channel)
        self.se_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.se_mul = torch.mul

    def forward(self, out):
        out_se = out
        out = self.se_global_pool(out, (2, 3))
        out = self.se_dense_0(out)
        out = self.relu(out)
        out = self.se_dense_1(out)
        out = self.se_sigmoid(out)
        out = torch.reshape(out, out.size() + (1, 1))
        out = self.se_mul(out, out_se)
        return out


class Se_ResidualBlock(nn.Module):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        use_se (bool): enable SE-ResNet50 net. Default: False.
        se_block(bool): use se block in SE-ResNet50 net. Default: False.

    Returns:
        Tensor, output tensor.

    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 reduction=16):
        super(Se_ResidualBlock, self).__init__()
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
            self.down_sample_layer = nn.Sequential(_conv1x1(in_channel, out_channel, stride),
                                                   _bn(out_channel))  # use_se=self.use_se
        self.add = torch.add
        self.se = SELayer(out_channel, reduction)

    def forward(self, x):
        """se_block"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.down_sample:
            identity = self.down_sample_layer(identity)
        out = self.add(out, identity)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
        use_se (bool): enable SE-ResNet50 net. Default: False.
        se_block(bool): use se block in SE-ResNet50 net in layer 3 and layer 4. Default: False.
    Returns:
        Tensor, output tensor.

    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.mean = torch.mean
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            se_block(bool): use se block in SE-ResNet50 net. Default: False.
        Returns:
            Sequential, the output layer.

        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)
        return nn.Sequential(*layers)

    def forward(self, x):
        """forward network"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out


def se_resnet50(class_num=10):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    """
    return ResNet(Se_ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


# def se_resnet101(class_num=10):
#     """
#     Get ResNet101 neural network.
#
#     Args:
#         class_num (int): Class number.
#
#     Returns:
#         Cell, cell instance of ResNet101 neural network.
#
#     """
#     return ResNet(Se_ResidualBlock,
#                   [3, 4, 23, 3],
#                   [64, 256, 512, 1024],
#                   [256, 512, 1024, 2048],
#                   [1, 2, 2, 2],
#                   class_num)

if __name__ == '__main__':
    model = se_resnet50(class_num=1000)
    data = torch.tensor(np.random.randn(1, 3, 224, 224), dtype=torch.float32)
    print(model(data).shape)
