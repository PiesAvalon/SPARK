import math
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# Assuming `config` is a similar configuration object in your PyTorch code
from network.cv.resnet.src.model_utils.config import config


def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    if config.net_name == "resnet152":
        stddev = (scale ** 0.5)
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return torch.Tensor(weight)


def _weight_variable(shape, factor=0.01):
    init_value = torch.randn(shape) * factor
    return init_value


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            neg_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            neg_slope = param
        else:
            raise ValueError("neg_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + neg_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    tensor = torch.zeros(inputs_shape)
    fan = init._calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return torch.normal(0, std, size=inputs_shape)


# def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
#     tensor = torch.zeros(inputs_shape)
#     fan = init._calculate_correct_fan(tensor, mode)
#     gain = calculate_gain(nonlinearity, a)
#     std = gain / math.sqrt(fan)
#     bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
#     return torch.uniform(-bound, bound, size=inputs_shape)


def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=3)
    else:
        weight_shape = (out_channel, in_channel, 3, 3)
        weight = kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu')
        if config.net_name == "resnet152":
            weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
    else:
        weight_shape = (out_channel, in_channel, 1, 1)
        weight = kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu')
        if config.net_name == "resnet152":
            weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)


def _conv7x7(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=7)
    else:
        weight_shape = (out_channel, in_channel, 7, 7)
        weight = kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu')
        if config.net_name == "resnet152":
            weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=stride, padding=3)


def _bn(channel, res_base=False):
    if res_base:
        return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.1)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9)


def _bn_last(channel):
    bn = nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9)
    nn.init.zeros_(bn.weight)
    nn.init.zeros_(bn.bias)
    return bn


def _fc(in_channel, out_channel, use_se=False):
    if use_se:
        weight = np.random.normal(loc=0, scale=0.01, size=(out_channel, in_channel))
        weight = torch.Tensor(weight)
    else:
        weight_shape = (out_channel, in_channel)
        # weight = kaiming_uniform(weight_shape, a=math.sqrt(5))
        if config.net_name == "resnet152":
            weight = _weight_variable(weight_shape)
    fc = nn.Linear(in_channel, out_channel)
    # fc.weight = nn.Parameter(weight)
    nn.init.zeros_(fc.bias)
    return fc


class ResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, use_se=False, se_block=False):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.se_block = se_block
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1, use_se=self.use_se)
        self.bn1 = _bn(channel)
        if self.use_se and self.stride != 1:
            self.e2 = nn.Sequential(_conv3x3(channel, channel, stride=1, use_se=True), _bn(channel),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        else:
            self.conv2 = _conv3x3(channel, channel, stride=stride, use_se=self.use_se)
            self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1, use_se=self.use_se)
        self.bn3 = _bn(out_channel)
        if config.optimizer == "Thor" or config.net_name == "resnet152":
            self.bn3 = _bn_last(out_channel)
        if self.se_block:
            self.se_dense_0 = _fc(out_channel, int(out_channel / 4), use_se=self.use_se)
            self.se_dense_1 = _fc(int(out_channel / 4), out_channel, use_se=self.use_se)
            self.se_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None
        if self.down_sample:
            if self.use_se:
                if stride == 1:
                    self.down_sample_layer = nn.Sequential(_conv1x1(in_channel, out_channel, stride, use_se=self.use_se), _bn(out_channel))
                else:
                    self.down_sample_layer = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                                           _conv1x1(in_channel, out_channel, 1, use_se=self.use_se), _bn(out_channel))
            else:
                self.down_sample_layer = nn.Sequential(_conv1x1(in_channel, out_channel, stride, use_se=self.use_se), _bn(out_channel))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)
        if self.use_se and self.stride != 1:
            out = self.e2(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            #out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se_block:
            out_se = out
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = self.se_dense_0(out)
            out = self.relu(out)
            out = self.se_dense_1(out)
            out = self.se_sigmoid(out)
            out = out.view(out.size(0), out.size(1), 1, 1)
            out = out * out_se
        if self.down_sample:
            identity = self.down_sample_layer(identity)
        out += identity
        out = self.relu(out)
        return out


class ResidualBlockBase(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, use_se=False, se_block=False, res_base=True):
        super(ResidualBlockBase, self).__init__()
        self.res_base = res_base
        self.conv1 = _conv3x3(in_channel, out_channel, stride=stride, res_base=self.res_base)
        self.bn1 = _bn(out_channel)
        self.conv2 = _conv3x3(out_channel, out_channel, stride=1, res_base=self.res_base)
        self.bn2 = _bn(out_channel)
        self.relu = nn.ReLU()

        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None
        if self.down_sample:
            self.down_sample_layer = nn.Sequential(_conv1x1(in_channel, out_channel, stride, use_se=use_se, res_base=self.res_base),
                                                   _bn(out_channel, res_base))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample:
            identity = self.down_sample_layer(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layer_nums, in_channels, out_channels, strides, num_classes, use_se=False, res_base=False):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.use_se = use_se
        self.res_base = res_base
        self.se_block = False
        if self.use_se:
            self.se_block = True

        if self.use_se:
            self.conv1_0 = _conv3x3(3, 32, stride=2, use_se=self.use_se)
            self.bn1_0 = _bn(32)
            self.conv1_1 = _conv3x3(32, 32, stride=1, use_se=self.use_se)
            self.bn1_1 = _bn(32)
            self.conv1_2 = _conv3x3(32, 64, stride=1, use_se=self.use_se)
        else:
            self.conv1 = _conv7x7(3, 64, stride=2, res_base=self.res_base)
        self.bn1 = _bn(64, self.res_base)
        self.relu = nn.ReLU()

        if self.res_base:
            self.pad = nn.ConstantPad2d((1, 1, 1, 1), 0)  # pad (left, right, top, bottom)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # "valid" padding in PyTorch
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # "same" padding in PyTorch

        self.layer1 = self._make_layer(block, layer_nums[0], in_channels[0], out_channels[0], strides[0], self.use_se)
        self.layer2 = self._make_layer(block, layer_nums[1], in_channels[1], out_channels[1], strides[1], self.use_se)
        self.layer3 = self._make_layer(block, layer_nums[2], in_channels[2], out_channels[2], strides[2], self.use_se, self.se_block)
        self.layer4 = self._make_layer(block, layer_nums[3], in_channels[3], out_channels[3], strides[3], self.use_se, self.se_block)

        self.mean = torch.std_mean
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes, use_se=self.use_se)

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

        self.origin_layer_names = {
            "conv1": self.conv1,
            "bn1": self.bn1,
            "relu": self.relu,
            "maxpool": self.maxpool,
            "layer1": self.layer1,
            "layer1.0": self.layer1[0],
            "layer1.0.conv1": self.layer1[0].conv1,
            "layer1.0.bn1": self.layer1[0].bn1,
            "layer1.0.conv2": self.layer1[0].conv2,
            "layer1.0.bn2": self.layer1[0].bn2,
            "layer1.0.conv3": self.layer1[0].conv3,
            "layer1.0.bn3": self.layer1[0].bn3,
            "layer1.0.relu": self.layer1[0].relu,
            "layer1.0.down_sample_layer": self.layer1[0].down_sample_layer,
            "layer1.0.down_sample_layer.0": self.layer1[0].down_sample_layer[0],
            "layer1.0.down_sample_layer.1": self.layer1[0].down_sample_layer[1],
            "layer1.1": self.layer1[1],
            "layer1.1.conv1": self.layer1[1].conv1,
            "layer1.1.bn1": self.layer1[1].bn1,
            "layer1.1.conv2": self.layer1[1].conv2,
            "layer1.1.bn2": self.layer1[1].bn2,
            "layer1.1.conv3": self.layer1[1].conv3,
            "layer1.1.bn3": self.layer1[1].bn3,
            "layer1.1.relu": self.layer1[1].relu,
            "layer1.2": self.layer1[2],
            "layer1.2.conv1": self.layer1[2].conv1,
            "layer1.2.bn1": self.layer1[2].bn1,
            "layer1.2.conv2": self.layer1[2].conv2,
            "layer1.2.bn2": self.layer1[2].bn2,
            "layer1.2.conv3": self.layer1[2].conv3,
            "layer1.2.bn3": self.layer1[2].bn3,
            "layer1.2.relu": self.layer1[2].relu,
            "layer2": self.layer2,
            "layer2.0": self.layer2[0],
            "layer2.0.conv1": self.layer2[0].conv1,
            "layer2.0.bn1": self.layer2[0].bn1,
            "layer2.0.conv2": self.layer2[0].conv2,
            "layer2.0.bn2": self.layer2[0].bn2,
            "layer2.0.conv3": self.layer2[0].conv3,
            "layer2.0.bn3": self.layer2[0].bn3,
            "layer2.0.relu": self.layer2[0].relu,
            "layer2.0.down_sample_layer": self.layer2[0].down_sample_layer,
            "layer2.0.down_sample_layer.0": self.layer2[0].down_sample_layer[0],
            "layer2.0.down_sample_layer.1": self.layer2[0].down_sample_layer[1],
            "layer2.1": self.layer2[1],
            "layer2.1.conv1": self.layer2[1].conv1,
            "layer2.1.bn1": self.layer2[1].bn1,
            "layer2.1.conv2": self.layer2[1].conv2,
            "layer2.1.bn2": self.layer2[1].bn2,
            "layer2.1.conv3": self.layer2[1].conv3,
            "layer2.1.bn3": self.layer2[1].bn3,
            "layer2.1.relu": self.layer2[1].relu,
            "layer2.2": self.layer2[2],
            "layer2.2.conv1": self.layer2[2].conv1,
            "layer2.2.bn1": self.layer2[2].bn1,
            "layer2.2.conv2": self.layer2[2].conv2,
            "layer2.2.bn2": self.layer2[2].bn2,
            "layer2.2.conv3": self.layer2[2].conv3,
            "layer2.2.bn3": self.layer2[2].bn3,
            "layer2.2.relu": self.layer2[2].relu,
            "layer2.3": self.layer2[3],
            "layer2.3.conv1": self.layer2[3].conv1,
            "layer2.3.bn1": self.layer2[3].bn1,
            "layer2.3.conv2": self.layer2[3].conv2,
            "layer2.3.bn2": self.layer2[3].bn2,
            "layer2.3.conv3": self.layer2[3].conv3,
            "layer2.3.bn3": self.layer2[3].bn3,
            "layer2.3.relu": self.layer2[3].relu,
            "layer3": self.layer3,
            "layer3.0": self.layer3[0],
            "layer3.0.conv1": self.layer3[0].conv1,
            "layer3.0.bn1": self.layer3[0].bn1,
            "layer3.0.conv2": self.layer3[0].conv2,
            "layer3.0.bn2": self.layer3[0].bn2,
            "layer3.0.conv3": self.layer3[0].conv3,
            "layer3.0.bn3": self.layer3[0].bn3,
            "layer3.0.relu": self.layer3[0].relu,
            "layer3.0.down_sample_layer": self.layer3[0].down_sample_layer,
            "layer3.0.down_sample_layer.0": self.layer3[0].down_sample_layer[0],
            "layer3.0.down_sample_layer.1": self.layer3[0].down_sample_layer[1],
            "layer3.1": self.layer3[1],
            "layer3.1.conv1": self.layer3[1].conv1,
            "layer3.1.bn1": self.layer3[1].bn1,
            "layer3.1.conv2": self.layer3[1].conv2,
            "layer3.1.bn2": self.layer3[1].bn2,
            "layer3.1.conv3": self.layer3[1].conv3,
            "layer3.1.bn3": self.layer3[1].bn3,
            "layer3.1.relu": self.layer3[1].relu,
            "layer3.2": self.layer3[2],
            "layer3.2.conv1": self.layer3[2].conv1,
            "layer3.2.bn1": self.layer3[2].bn1,
            "layer3.2.conv2": self.layer3[2].conv2,
            "layer3.2.bn2": self.layer3[2].bn2,
            "layer3.2.conv3": self.layer3[2].conv3,
            "layer3.2.bn3": self.layer3[2].bn3,
            "layer3.2.relu": self.layer3[2].relu,
            "layer3.3": self.layer3[3],
            "layer3.3.conv1": self.layer3[3].conv1,
            "layer3.3.bn1": self.layer3[3].bn1,
            "layer3.3.conv2": self.layer3[3].conv2,
            "layer3.3.bn2": self.layer3[3].bn2,
            "layer3.3.conv3": self.layer3[3].conv3,
            "layer3.3.bn3": self.layer3[3].bn3,
            "layer3.3.relu": self.layer3[3].relu,
            "layer3.4": self.layer3[4],
            "layer3.4.conv1": self.layer3[4].conv1,
            "layer3.4.bn1": self.layer3[4].bn1,
            "layer3.4.conv2": self.layer3[4].conv2,
            "layer3.4.bn2": self.layer3[4].bn2,
            "layer3.4.conv3": self.layer3[4].conv3,
            "layer3.4.bn3": self.layer3[4].bn3,
            "layer3.4.relu": self.layer3[4].relu,
            "layer3.5": self.layer3[5],
            "layer3.5.conv1": self.layer3[5].conv1,
            "layer3.5.bn1": self.layer3[5].bn1,
            "layer3.5.conv2": self.layer3[5].conv2,
            "layer3.5.bn2": self.layer3[5].bn2,
            "layer3.5.conv3": self.layer3[5].conv3,
            "layer3.5.bn3": self.layer3[5].bn3,
            "layer3.5.relu": self.layer3[5].relu,
            "layer4": self.layer4,
            "layer4.0": self.layer4[0],
            "layer4.0.conv1": self.layer4[0].conv1,
            "layer4.0.bn1": self.layer4[0].bn1,
            "layer4.0.conv2": self.layer4[0].conv2,
            "layer4.0.bn2": self.layer4[0].bn2,
            "layer4.0.conv3": self.layer4[0].conv3,
            "layer4.0.bn3": self.layer4[0].bn3,
            "layer4.0.relu": self.layer4[0].relu,
            "layer4.0.down_sample_layer": self.layer4[0].down_sample_layer,
            "layer4.0.down_sample_layer.0": self.layer4[0].down_sample_layer[0],
            "layer4.0.down_sample_layer.1": self.layer4[0].down_sample_layer[1],
            "layer4.1": self.layer4[1],
            "layer4.1.conv1": self.layer4[1].conv1,
            "layer4.1.bn1": self.layer4[1].bn1,
            "layer4.1.conv2": self.layer4[1].conv2,
            "layer4.1.bn2": self.layer4[1].bn2,
            "layer4.1.conv3": self.layer4[1].conv3,
            "layer4.1.bn3": self.layer4[1].bn3,
            "layer4.1.relu": self.layer4[1].relu,
            "layer4.2": self.layer4[2],
            "layer4.2.conv1": self.layer4[2].conv1,
            "layer4.2.bn1": self.layer4[2].bn1,
            "layer4.2.conv2": self.layer4[2].conv2,
            "layer4.2.bn2": self.layer4[2].bn2,
            "layer4.2.conv3": self.layer4[2].conv3,
            "layer4.2.bn3": self.layer4[2].bn3,
            "layer4.2.relu": self.layer4[2].relu,
            "flatten": self.flatten,
            "end_point": self.end_point,
        }
        self.layer_names = {
            "conv1": self.conv1,
            "bn1": self.bn1,
            "relu": self.relu,
            "maxpool": self.maxpool,
            "layer1": self.layer1,
            "layer1.0": self.layer1[0],
            "layer1.0.conv1": self.layer1[0].conv1,
            "layer1.0.bn1": self.layer1[0].bn1,
            "layer1.0.conv2": self.layer1[0].conv2,
            "layer1.0.bn2": self.layer1[0].bn2,
            "layer1.0.conv3": self.layer1[0].conv3,
            "layer1.0.bn3": self.layer1[0].bn3,
            "layer1.0.relu": self.layer1[0].relu,
            "layer1.0.down_sample_layer": self.layer1[0].down_sample_layer,
            "layer1.0.down_sample_layer.0": self.layer1[0].down_sample_layer[0],
            "layer1.0.down_sample_layer.1": self.layer1[0].down_sample_layer[1],
            "layer1.1": self.layer1[1],
            "layer1.1.conv1": self.layer1[1].conv1,
            "layer1.1.bn1": self.layer1[1].bn1,
            "layer1.1.conv2": self.layer1[1].conv2,
            "layer1.1.bn2": self.layer1[1].bn2,
            "layer1.1.conv3": self.layer1[1].conv3,
            "layer1.1.bn3": self.layer1[1].bn3,
            "layer1.1.relu": self.layer1[1].relu,
            "layer1.2": self.layer1[2],
            "layer1.2.conv1": self.layer1[2].conv1,
            "layer1.2.bn1": self.layer1[2].bn1,
            "layer1.2.conv2": self.layer1[2].conv2,
            "layer1.2.bn2": self.layer1[2].bn2,
            "layer1.2.conv3": self.layer1[2].conv3,
            "layer1.2.bn3": self.layer1[2].bn3,
            "layer1.2.relu": self.layer1[2].relu,
            "layer2": self.layer2,
            "layer2.0": self.layer2[0],
            "layer2.0.conv1": self.layer2[0].conv1,
            "layer2.0.bn1": self.layer2[0].bn1,
            "layer2.0.conv2": self.layer2[0].conv2,
            "layer2.0.bn2": self.layer2[0].bn2,
            "layer2.0.conv3": self.layer2[0].conv3,
            "layer2.0.bn3": self.layer2[0].bn3,
            "layer2.0.relu": self.layer2[0].relu,
            "layer2.0.down_sample_layer": self.layer2[0].down_sample_layer,
            "layer2.0.down_sample_layer.0": self.layer2[0].down_sample_layer[0],
            "layer2.0.down_sample_layer.1": self.layer2[0].down_sample_layer[1],
            "layer2.1": self.layer2[1],
            "layer2.1.conv1": self.layer2[1].conv1,
            "layer2.1.bn1": self.layer2[1].bn1,
            "layer2.1.conv2": self.layer2[1].conv2,
            "layer2.1.bn2": self.layer2[1].bn2,
            "layer2.1.conv3": self.layer2[1].conv3,
            "layer2.1.bn3": self.layer2[1].bn3,
            "layer2.1.relu": self.layer2[1].relu,
            "layer2.2": self.layer2[2],
            "layer2.2.conv1": self.layer2[2].conv1,
            "layer2.2.bn1": self.layer2[2].bn1,
            "layer2.2.conv2": self.layer2[2].conv2,
            "layer2.2.bn2": self.layer2[2].bn2,
            "layer2.2.conv3": self.layer2[2].conv3,
            "layer2.2.bn3": self.layer2[2].bn3,
            "layer2.2.relu": self.layer2[2].relu,
            "layer2.3": self.layer2[3],
            "layer2.3.conv1": self.layer2[3].conv1,
            "layer2.3.bn1": self.layer2[3].bn1,
            "layer2.3.conv2": self.layer2[3].conv2,
            "layer2.3.bn2": self.layer2[3].bn2,
            "layer2.3.conv3": self.layer2[3].conv3,
            "layer2.3.bn3": self.layer2[3].bn3,
            "layer2.3.relu": self.layer2[3].relu,
            "layer3": self.layer3,
            "layer3.0": self.layer3[0],
            "layer3.0.conv1": self.layer3[0].conv1,
            "layer3.0.bn1": self.layer3[0].bn1,
            "layer3.0.conv2": self.layer3[0].conv2,
            "layer3.0.bn2": self.layer3[0].bn2,
            "layer3.0.conv3": self.layer3[0].conv3,
            "layer3.0.bn3": self.layer3[0].bn3,
            "layer3.0.relu": self.layer3[0].relu,
            "layer3.0.down_sample_layer": self.layer3[0].down_sample_layer,
            "layer3.0.down_sample_layer.0": self.layer3[0].down_sample_layer[0],
            "layer3.0.down_sample_layer.1": self.layer3[0].down_sample_layer[1],
            "layer3.1": self.layer3[1],
            "layer3.1.conv1": self.layer3[1].conv1,
            "layer3.1.bn1": self.layer3[1].bn1,
            "layer3.1.conv2": self.layer3[1].conv2,
            "layer3.1.bn2": self.layer3[1].bn2,
            "layer3.1.conv3": self.layer3[1].conv3,
            "layer3.1.bn3": self.layer3[1].bn3,
            "layer3.1.relu": self.layer3[1].relu,
            "layer3.2": self.layer3[2],
            "layer3.2.conv1": self.layer3[2].conv1,
            "layer3.2.bn1": self.layer3[2].bn1,
            "layer3.2.conv2": self.layer3[2].conv2,
            "layer3.2.bn2": self.layer3[2].bn2,
            "layer3.2.conv3": self.layer3[2].conv3,
            "layer3.2.bn3": self.layer3[2].bn3,
            "layer3.2.relu": self.layer3[2].relu,
            "layer3.3": self.layer3[3],
            "layer3.3.conv1": self.layer3[3].conv1,
            "layer3.3.bn1": self.layer3[3].bn1,
            "layer3.3.conv2": self.layer3[3].conv2,
            "layer3.3.bn2": self.layer3[3].bn2,
            "layer3.3.conv3": self.layer3[3].conv3,
            "layer3.3.bn3": self.layer3[3].bn3,
            "layer3.3.relu": self.layer3[3].relu,
            "layer3.4": self.layer3[4],
            "layer3.4.conv1": self.layer3[4].conv1,
            "layer3.4.bn1": self.layer3[4].bn1,
            "layer3.4.conv2": self.layer3[4].conv2,
            "layer3.4.bn2": self.layer3[4].bn2,
            "layer3.4.conv3": self.layer3[4].conv3,
            "layer3.4.bn3": self.layer3[4].bn3,
            "layer3.4.relu": self.layer3[4].relu,
            "layer3.5": self.layer3[5],
            "layer3.5.conv1": self.layer3[5].conv1,
            "layer3.5.bn1": self.layer3[5].bn1,
            "layer3.5.conv2": self.layer3[5].conv2,
            "layer3.5.bn2": self.layer3[5].bn2,
            "layer3.5.conv3": self.layer3[5].conv3,
            "layer3.5.bn3": self.layer3[5].bn3,
            "layer3.5.relu": self.layer3[5].relu,
            "layer4": self.layer4,
            "layer4.0": self.layer4[0],
            "layer4.0.conv1": self.layer4[0].conv1,
            "layer4.0.bn1": self.layer4[0].bn1,
            "layer4.0.conv2": self.layer4[0].conv2,
            "layer4.0.bn2": self.layer4[0].bn2,
            "layer4.0.conv3": self.layer4[0].conv3,
            "layer4.0.bn3": self.layer4[0].bn3,
            "layer4.0.relu": self.layer4[0].relu,
            "layer4.0.down_sample_layer": self.layer4[0].down_sample_layer,
            "layer4.0.down_sample_layer.0": self.layer4[0].down_sample_layer[0],
            "layer4.0.down_sample_layer.1": self.layer4[0].down_sample_layer[1],
            "layer4.1": self.layer4[1],
            "layer4.1.conv1": self.layer4[1].conv1,
            "layer4.1.bn1": self.layer4[1].bn1,
            "layer4.1.conv2": self.layer4[1].conv2,
            "layer4.1.bn2": self.layer4[1].bn2,
            "layer4.1.conv3": self.layer4[1].conv3,
            "layer4.1.bn3": self.layer4[1].bn3,
            "layer4.1.relu": self.layer4[1].relu,
            "layer4.2": self.layer4[2],
            "layer4.2.conv1": self.layer4[2].conv1,
            "layer4.2.bn1": self.layer4[2].bn1,
            "layer4.2.conv2": self.layer4[2].conv2,
            "layer4.2.bn2": self.layer4[2].bn2,
            "layer4.2.conv3": self.layer4[2].conv3,
            "layer4.2.bn3": self.layer4[2].bn3,
            "layer4.2.relu": self.layer4[2].relu,
            "flatten": self.flatten,
            "end_point": self.end_point,
        }

        self.in_shapes={
                        'INPUT': [1, 3, 224, 224],
                        'conv1': [1, 3, 224, 224],
                         'bn1': [1, 64, 112, 112],
                         'relu': [1, 64, 112, 112],
                         'maxpool': [1, 64, 112, 112],
                         'layer1.0.conv1': [1, 64, 56, 56],
                         'layer1.0.bn1': [1, 64, 56, 56],
                         'layer1.0.conv2': [1, 64, 56, 56],
                         'layer1.0.bn2': [1, 64, 56, 56],
                         'layer1.0.conv3': [1, 64, 56, 56],
                         'layer1.0.bn3': [1, 256, 56, 56],
                         'layer1.0.relu': [1, 256, 56, 56],
                         'layer1.0.down_sample_layer.0': [1, 64, 56, 56],
                         'layer1.0.down_sample_layer.1': [1, 256, 56, 56],
                         'layer1.1.conv1': [1, 256, 56, 56],
                         'layer1.1.bn1': [1, 64, 56, 56],
                         'layer1.1.conv2': [1, 64, 56, 56],
                         'layer1.1.bn2': [1, 64, 56, 56],
                         'layer1.1.conv3': [1, 64, 56, 56],
                         'layer1.1.bn3': [1, 256, 56, 56],
                         'layer1.1.relu': [1, 256, 56, 56],
                         'layer1.2.conv1': [1, 256, 56, 56],
                         'layer1.2.bn1': [1, 64, 56, 56],
                         'layer1.2.conv2': [1, 64, 56, 56],
                         'layer1.2.bn2': [1, 64, 56, 56],
                         'layer1.2.conv3': [1, 64, 56, 56],
                         'layer1.2.bn3': [1, 256, 56, 56],
                         'layer1.2.relu': [1, 256, 56, 56],
                         'layer2.0.conv1': [1, 256, 56, 56],
                         'layer2.0.bn1': [1, 128, 56, 56],
                         'layer2.0.conv2': [1, 128, 56, 56],
                         'layer2.0.bn2': [1, 128, 28, 28],
                         'layer2.0.conv3': [1, 128, 28, 28],
                         'layer2.0.bn3': [1, 512, 28, 28],
                         'layer2.0.relu': [1, 512, 28, 28],
                         'layer2.0.down_sample_layer.0': [1, 256, 56, 56],
                         'layer2.0.down_sample_layer.1': [1, 512, 28, 28],
                         'layer2.1.conv1': [1, 512, 28, 28],
                         'layer2.1.bn1': [1, 128, 28, 28],
                         'layer2.1.conv2': [1, 128, 28, 28],
                         'layer2.1.bn2': [1, 128, 28, 28],
                         'layer2.1.conv3': [1, 128, 28, 28],
                         'layer2.1.bn3': [1, 512, 28, 28],
                         'layer2.1.relu': [1, 512, 28, 28],
                         'layer2.2.conv1': [1, 512, 28, 28],
                         'layer2.2.bn1': [1, 128, 28, 28],
                         'layer2.2.conv2': [1, 128, 28, 28],
                         'layer2.2.bn2': [1, 128, 28, 28],
                         'layer2.2.conv3': [1, 128, 28, 28],
                         'layer2.2.bn3': [1, 512, 28, 28],
                         'layer2.2.relu': [1, 512, 28, 28],
                         'layer2.3.conv1': [1, 512, 28, 28],
                         'layer2.3.bn1': [1, 128, 28, 28],
                         'layer2.3.conv2': [1, 128, 28, 28],
                         'layer2.3.bn2': [1, 128, 28, 28],
                         'layer2.3.conv3': [1, 128, 28, 28],
                         'layer2.3.bn3': [1, 512, 28, 28],
                         'layer2.3.relu': [1, 512, 28, 28],
                         'layer3.0.conv1': [1, 512, 28, 28],
                         'layer3.0.bn1': [1, 256, 28, 28],
                         'layer3.0.conv2': [1, 256, 28, 28],
                         'layer3.0.bn2': [1, 256, 14, 14],
                         'layer3.0.conv3': [1, 256, 14, 14],
                         'layer3.0.bn3': [1, 1024, 14, 14],
                         'layer3.0.relu': [1, 1024, 14, 14],
                         'layer3.0.down_sample_layer.0': [1, 512, 28, 28],
                         'layer3.0.down_sample_layer.1': [1, 1024, 14, 14],
                         'layer3.1.conv1': [1, 1024, 14, 14],
                         'layer3.1.bn1': [1, 256, 14, 14],
                         'layer3.1.conv2': [1, 256, 14, 14],
                         'layer3.1.bn2': [1, 256, 14, 14],
                         'layer3.1.conv3': [1, 256, 14, 14],
                         'layer3.1.bn3': [1, 1024, 14, 14],
                         'layer3.1.relu': [1, 1024, 14, 14],
                         'layer3.2.conv1': [1, 1024, 14, 14],
                         'layer3.2.bn1': [1, 256, 14, 14],
                         'layer3.2.conv2': [1, 256, 14, 14],
                         'layer3.2.bn2': [1, 256, 14, 14],
                         'layer3.2.conv3': [1, 256, 14, 14],
                         'layer3.2.bn3': [1, 1024, 14, 14],
                         'layer3.2.relu': [1, 1024, 14, 14],
                         'layer3.3.conv1': [1, 1024, 14, 14],
                         'layer3.3.bn1': [1, 256, 14, 14],
                         'layer3.3.conv2': [1, 256, 14, 14],
                         'layer3.3.bn2': [1, 256, 14, 14],
                         'layer3.3.conv3': [1, 256, 14, 14],
                         'layer3.3.bn3': [1, 1024, 14, 14],
                         'layer3.3.relu': [1, 1024, 14, 14],
                         'layer3.4.conv1': [1, 1024, 14, 14],
                         'layer3.4.bn1': [1, 256, 14, 14],
                         'layer3.4.conv2': [1, 256, 14, 14],
                         'layer3.4.bn2': [1, 256, 14, 14],
                         'layer3.4.conv3': [1, 256, 14, 14],
                         'layer3.4.bn3': [1, 1024, 14, 14],
                         'layer3.4.relu': [1, 1024, 14, 14],
                         'layer3.5.conv1': [1, 1024, 14, 14],
                         'layer3.5.bn1': [1, 256, 14, 14],
                         'layer3.5.conv2': [1, 256, 14, 14],
                         'layer3.5.bn2': [1, 256, 14, 14],
                         'layer3.5.conv3': [1, 256, 14, 14],
                         'layer3.5.bn3': [1, 1024, 14, 14],
                         'layer3.5.relu': [1, 1024, 14, 14],
                         'layer4.0.conv1': [1, 1024, 14, 14],
                         'layer4.0.bn1': [1, 512, 14, 14],
                         'layer4.0.conv2': [1, 512, 14, 14],
                         'layer4.0.bn2': [1, 512, 7, 7],
                         'layer4.0.conv3': [1, 512, 7, 7],
                         'layer4.0.bn3': [1, 2048, 7, 7],
                         'layer4.0.relu': [1, 2048, 7, 7],
                         'layer4.0.down_sample_layer.0': [1, 1024, 14, 14],
                         'layer4.0.down_sample_layer.1': [1, 2048, 7, 7],
                         'layer4.1.conv1': [1, 2048, 7, 7],
                         'layer4.1.bn1': [1, 512, 7, 7],
                         'layer4.1.conv2': [1, 512, 7, 7],
                         'layer4.1.bn2': [1, 512, 7, 7],
                         'layer4.1.conv3': [1, 512, 7, 7],
                         'layer4.1.bn3': [1, 2048, 7, 7],
                         'layer4.1.relu': [1, 2048, 7, 7],
                         'layer4.2.conv1': [1, 2048, 7, 7],
                         'layer4.2.bn1': [1, 512, 7, 7],
                         'layer4.2.conv2': [1, 512, 7, 7],
                         'layer4.2.bn2': [1, 512, 7, 7],
                         'layer4.2.conv3': [1, 512, 7, 7],
                         'layer4.2.bn3': [1, 2048, 7, 7],
                         'layer4.2.relu': [1, 2048, 7, 7],
                         'flatten': [1, 2048, 1, 1],
                         'end_point': [1, 2048],
                         'OUTPUT': [1, 10]}

        self.out_shapes={
            'INPUT': [1, 3, 224, 224],
            'conv1': [1, 64, 112, 112],
             'bn1': [1, 64, 112, 112],
             'relu': [1, 64, 112, 112],
             'maxpool': [1, 64, 56, 56],
             'layer1.0.conv1': [1, 64, 56, 56],
             'layer1.0.bn1': [1, 64, 56, 56],
             'layer1.0.conv2': [1, 64, 56, 56],
             'layer1.0.bn2': [1, 64, 56, 56],
             'layer1.0.conv3': [1, 256, 56, 56],
             'layer1.0.bn3': [1, 256, 56, 56],
             'layer1.0.relu': [1, 256, 56, 56],
             'layer1.0.down_sample_layer.0': [1, 256, 56, 56],
             'layer1.0.down_sample_layer.1': [1, 256, 56, 56],
             'layer1.1.conv1': [1, 64, 56, 56],
             'layer1.1.bn1': [1, 64, 56, 56],
             'layer1.1.conv2': [1, 64, 56, 56],
             'layer1.1.bn2': [1, 64, 56, 56],
             'layer1.1.conv3': [1, 256, 56, 56],
             'layer1.1.bn3': [1, 256, 56, 56],
             'layer1.1.relu': [1, 256, 56, 56],
             'layer1.2.conv1': [1, 64, 56, 56],
             'layer1.2.bn1': [1, 64, 56, 56],
             'layer1.2.conv2': [1, 64, 56, 56],
             'layer1.2.bn2': [1, 64, 56, 56],
             'layer1.2.conv3': [1, 256, 56, 56],
             'layer1.2.bn3': [1, 256, 56, 56],
             'layer1.2.relu': [1, 256, 56, 56],
             'layer2.0.conv1': [1, 128, 56, 56],
             'layer2.0.bn1': [1, 128, 56, 56],
             'layer2.0.conv2': [1, 128, 28, 28],
             'layer2.0.bn2': [1, 128, 28, 28],
             'layer2.0.conv3': [1, 512, 28, 28],
             'layer2.0.bn3': [1, 512, 28, 28],
             'layer2.0.relu': [1, 512, 28, 28],
             'layer2.0.down_sample_layer.0': [1, 512, 28, 28],
             'layer2.0.down_sample_layer.1': [1, 512, 28, 28],
             'layer2.1.conv1': [1, 128, 28, 28],
             'layer2.1.bn1': [1, 128, 28, 28],
             'layer2.1.conv2': [1, 128, 28, 28],
             'layer2.1.bn2': [1, 128, 28, 28],
             'layer2.1.conv3': [1, 512, 28, 28],
             'layer2.1.bn3': [1, 512, 28, 28],
             'layer2.1.relu': [1, 512, 28, 28],
             'layer2.2.conv1': [1, 128, 28, 28],
             'layer2.2.bn1': [1, 128, 28, 28],
             'layer2.2.conv2': [1, 128, 28, 28],
             'layer2.2.bn2': [1, 128, 28, 28],
             'layer2.2.conv3': [1, 512, 28, 28],
             'layer2.2.bn3': [1, 512, 28, 28],
             'layer2.2.relu': [1, 512, 28, 28],
             'layer2.3.conv1': [1, 128, 28, 28],
             'layer2.3.bn1': [1, 128, 28, 28],
             'layer2.3.conv2': [1, 128, 28, 28],
             'layer2.3.bn2': [1, 128, 28, 28],
             'layer2.3.conv3': [1, 512, 28, 28],
             'layer2.3.bn3': [1, 512, 28, 28],
             'layer2.3.relu': [1, 512, 28, 28],
             'layer3.0.conv1': [1, 256, 28, 28],
             'layer3.0.bn1': [1, 256, 28, 28],
             'layer3.0.conv2': [1, 256, 14, 14],
             'layer3.0.bn2': [1, 256, 14, 14],
             'layer3.0.conv3': [1, 1024, 14, 14],
             'layer3.0.bn3': [1, 1024, 14, 14],
             'layer3.0.relu': [1, 1024, 14, 14],
             'layer3.0.down_sample_layer.0': [1, 1024, 14, 14],
             'layer3.0.down_sample_layer.1': [1, 1024, 14, 14],
             'layer3.1.conv1': [1, 256, 14, 14],
             'layer3.1.bn1': [1, 256, 14, 14],
             'layer3.1.conv2': [1, 256, 14, 14],
             'layer3.1.bn2': [1, 256, 14, 14],
             'layer3.1.conv3': [1, 1024, 14, 14],
             'layer3.1.bn3': [1, 1024, 14, 14],
             'layer3.1.relu': [1, 1024, 14, 14],
             'layer3.2.conv1': [1, 256, 14, 14],
             'layer3.2.bn1': [1, 256, 14, 14],
             'layer3.2.conv2': [1, 256, 14, 14],
             'layer3.2.bn2': [1, 256, 14, 14],
             'layer3.2.conv3': [1, 1024, 14, 14],
             'layer3.2.bn3': [1, 1024, 14, 14],
             'layer3.2.relu': [1, 1024, 14, 14],
             'layer3.3.conv1': [1, 256, 14, 14],
             'layer3.3.bn1': [1, 256, 14, 14],
             'layer3.3.conv2': [1, 256, 14, 14],
             'layer3.3.bn2': [1, 256, 14, 14],
             'layer3.3.conv3': [1, 1024, 14, 14],
             'layer3.3.bn3': [1, 1024, 14, 14],
             'layer3.3.relu': [1, 1024, 14, 14],
             'layer3.4.conv1': [1, 256, 14, 14],
             'layer3.4.bn1': [1, 256, 14, 14],
             'layer3.4.conv2': [1, 256, 14, 14],
             'layer3.4.bn2': [1, 256, 14, 14],
             'layer3.4.conv3': [1, 1024, 14, 14],
             'layer3.4.bn3': [1, 1024, 14, 14],
             'layer3.4.relu': [1, 1024, 14, 14],
             'layer3.5.conv1': [1, 256, 14, 14],
             'layer3.5.bn1': [1, 256, 14, 14],
             'layer3.5.conv2': [1, 256, 14, 14],
             'layer3.5.bn2': [1, 256, 14, 14],
             'layer3.5.conv3': [1, 1024, 14, 14],
             'layer3.5.bn3': [1, 1024, 14, 14],
             'layer3.5.relu': [1, 1024, 14, 14],
             'layer4.0.conv1': [1, 512, 14, 14],
             'layer4.0.bn1': [1, 512, 14, 14],
             'layer4.0.conv2': [1, 512, 7, 7],
             'layer4.0.bn2': [1, 512, 7, 7],
             'layer4.0.conv3': [1, 2048, 7, 7],
             'layer4.0.bn3': [1, 2048, 7, 7],
             'layer4.0.relu': [1, 2048, 7, 7],
             'layer4.0.down_sample_layer.0': [1, 2048, 7, 7],
             'layer4.0.down_sample_layer.1': [1, 2048, 7, 7],
             'layer4.1.conv1': [1, 512, 7, 7],
             'layer4.1.bn1': [1, 512, 7, 7],
             'layer4.1.conv2': [1, 512, 7, 7],
             'layer4.1.bn2': [1, 512, 7, 7],
             'layer4.1.conv3': [1, 2048, 7, 7],
             'layer4.1.bn3': [1, 2048, 7, 7],
             'layer4.1.relu': [1, 2048, 7, 7],
             'layer4.2.conv1': [1, 512, 7, 7],
             'layer4.2.bn1': [1, 512, 7, 7],
             'layer4.2.conv2': [1, 512, 7, 7],
             'layer4.2.bn2': [1, 512, 7, 7],
             'layer4.2.conv3': [1, 2048, 7, 7],
             'layer4.2.bn3': [1, 2048, 7, 7],
             'layer4.2.relu': [1, 2048, 7, 7],
             'flatten': [1, 2048],
             'end_point': [1, 10],
             'OUTPUT': [1, 10]}



        self.orders = {
            'conv1': ["INPUT", "bn1"],
            'bn1': ["conv1", "relu"],
            'relu': ["bn1", "maxpool"],
            'maxpool': ["relu", ["layer1.0.conv1", "layer1.0.down_sample_layer.0"]],
            'layer1.0.conv1': ["maxpool", "layer1.0.bn1"],
            'layer1.0.bn1': ["layer1.0.conv1", "layer1.0.conv2"],
            # 'layer1.0.relu1': ["layer1.0.bn1", "layer1.0.conv2"],
            'layer1.0.conv2': ["layer1.0.bn1", "layer1.0.bn2"],
            'layer1.0.bn2': ["layer1.0.conv2", "layer1.0.conv3"],
            # 'layer1.0.relu2': ["layer1.0.bn2", "layer1.0.conv3"],
            'layer1.0.conv3': ["layer1.0.bn2", "layer1.0.bn3"],
            'layer1.0.bn3': ["layer1.0.conv3", "layer1.0.relu"],
            'layer1.0.down_sample_layer.0': ["maxpool", "layer1.0.down_sample_layer.1"],
            'layer1.0.down_sample_layer.1': ["layer1.0.down_sample_layer.0", "layer1.0.relu"],
            'layer1.0.relu': [["layer1.0.bn3", "layer1.0.down_sample_layer.1"], "layer1.1.conv1"],
            'layer1.1.conv1': ["layer1.0.relu", "layer1.1.bn1"],
            'layer1.1.bn1': ["layer1.1.conv1", "layer1.1.conv2"],
            # 'layer1.1.relu1': ["layer1.1.bn1", "layer1.1.conv2"],
            'layer1.1.conv2': ["layer1.1.bn1", "layer1.1.bn2"],
            'layer1.1.bn2': ["layer1.1.conv2", "layer1.1.conv3"],
            # 'layer1.1.relu2': ["layer1.1.bn2", "layer1.1.conv3"],
            'layer1.1.conv3': ["layer1.1.bn2", "layer1.1.bn3"],
            'layer1.1.bn3': ["layer1.1.conv3", "layer1.1.relu"],
            'layer1.1.relu': ["layer1.1.bn3", "layer1.2.conv1"],
            'layer1.2.conv1': ["layer1.1.relu", "layer1.2.bn1"],
            'layer1.2.bn1': ["layer1.2.conv1", "layer1.2.conv2"],
            # 'layer1.2.relu1': ["layer1.2.bn1", "layer1.2.conv2"],
            'layer1.2.conv2': ["layer1.2.bn1", "layer1.2.bn2"],
            'layer1.2.bn2': ["layer1.2.conv2", "layer1.2.conv3"],
            # 'layer1.2.relu2': ["layer1.2.bn2", "layer1.2.conv3"],
            'layer1.2.conv3': ["layer1.2.bn2", "layer1.2.bn3"],
            'layer1.2.bn3': ["layer1.2.conv3", "layer1.2.relu"],
            'layer1.2.relu': ["layer1.2.bn3", ["layer2.0.conv1", "layer2.0.down_sample_layer.0"]],
            'layer2.0.conv1': ["layer1.2.relu", "layer2.0.bn1"],
            'layer2.0.bn1': ["layer2.0.conv1", "layer2.0.conv2"],
            # 'layer2.0.relu1': ["layer2.0.bn1", "layer2.0.conv2"],
            'layer2.0.conv2': ["layer2.0.bn1", "layer2.0.bn2"],
            'layer2.0.bn2': ["layer2.0.conv2", "layer2.0.conv3"],
            # 'layer2.0.relu2': ["layer2.0.bn2", "layer2.0.conv3"],
            'layer2.0.conv3': ["layer2.0.bn2", "layer2.0.bn3"],
            'layer2.0.bn3': ["layer2.0.conv3", "layer2.0.relu"],
            'layer2.0.down_sample_layer.0': ["layer1.2.relu", "layer2.0.down_sample_layer.1"],
            'layer2.0.down_sample_layer.1': ["layer2.0.down_sample_layer.0", "layer2.0.relu"],
            'layer2.0.relu': [["layer2.0.bn3", "layer2.0.down_sample_layer.1"], "layer2.1.conv1"],
            'layer2.1.conv1': ["layer2.0.relu", "layer2.1.bn1"],
            'layer2.1.bn1': ["layer2.1.conv1", "layer2.1.conv2"],
            # 'layer2.1.relu1': ["layer2.1.bn1", "layer2.1.conv2"],
            'layer2.1.conv2': ["layer2.1.bn1", "layer2.1.bn2"],
            'layer2.1.bn2': ["layer2.1.conv2", "layer2.1.conv3"],
            # 'layer2.1.relu2': ["layer2.1.bn2", "layer2.1.conv3"],
            'layer2.1.conv3': ["layer2.1.bn2", "layer2.1.bn3"],
            'layer2.1.bn3': ["layer2.1.conv3", "layer2.1.relu"],
            'layer2.1.relu': ["layer2.1.bn3", "layer2.2.conv1"],
            'layer2.2.conv1': ["layer2.1.relu", "layer2.2.bn1"],
            'layer2.2.bn1': ["layer2.2.conv1", "layer2.2.conv2"],
            # 'layer2.2.relu1': ["layer2.2.bn1", "layer2.2.conv2"],
            'layer2.2.conv2': ["layer2.2.bn1", "layer2.2.bn2"],
            'layer2.2.bn2': ["layer2.2.conv2", "layer2.2.conv3"],
            # 'layer2.2.relu2': ["layer2.2.bn2", "layer2.2.conv3"],
            'layer2.2.conv3': ["layer2.2.bn2", "layer2.2.bn3"],
            'layer2.2.bn3': ["layer2.2.conv3", "layer2.2.relu"],
            'layer2.2.relu': ["layer2.2.bn3", "layer2.3.conv1"],
            'layer2.3.conv1': ["layer2.2.relu", "layer2.3.bn1"],
            'layer2.3.bn1': ["layer2.3.conv1", "layer2.3.conv2"],
            # 'layer2.3.relu1': ["layer2.3.bn1", "layer2.3.conv2"],
            'layer2.3.conv2': ["layer2.3.bn1", "layer2.3.bn2"],
            'layer2.3.bn2': ["layer2.3.conv2", "layer2.3.conv3"],
            # 'layer2.3.relu2': ["layer2.3.bn2", "layer2.3.conv3"],
            'layer2.3.conv3': ["layer2.3.bn2", "layer2.3.bn3"],
            'layer2.3.bn3': ["layer2.3.conv3", "layer2.3.relu"],
            'layer2.3.relu': ["layer2.3.bn3", ["layer3.0.conv1", "layer3.0.down_sample_layer.0"]],
            'layer3.0.conv1': ["layer2.3.relu", "layer3.0.bn1"],
            'layer3.0.bn1': ["layer3.0.conv1", "layer3.0.conv2"],
            # 'layer3.0.relu1': ["layer3.0.bn1", "layer3.0.conv2"],
            'layer3.0.conv2': ["layer3.0.bn1", "layer3.0.bn2"],
            'layer3.0.bn2': ["layer3.0.conv2", "layer3.0.conv3"],
            # 'layer3.0.relu2': ["layer3.0.bn2", "layer3.0.conv3"],
            'layer3.0.conv3': ["layer3.0.bn2", "layer3.0.bn3"],
            'layer3.0.bn3': ["layer3.0.conv3", "layer3.0.relu"],
            'layer3.0.down_sample_layer.0': ["layer2.3.relu", "layer3.0.down_sample_layer.1"],
            'layer3.0.down_sample_layer.1': ["layer3.0.down_sample_layer.0", "layer3.0.relu"],
            'layer3.0.relu': [["layer3.0.bn3", "layer3.0.down_sample_layer.1"], "layer3.1.conv1"],
            'layer3.1.conv1': ["layer3.0.relu", "layer3.1.bn1"],
            'layer3.1.bn1': ["layer3.1.conv1", "layer3.1.conv2"],
            # 'layer3.1.relu1': ["layer3.1.bn1", "layer3.1.conv2"],
            'layer3.1.conv2': ["layer3.1.bn1", "layer3.1.bn2"],
            'layer3.1.bn2': ["layer3.1.conv2", "layer3.1.conv3"],
            # 'layer3.1.relu2': ["layer3.1.bn2", "layer3.1.conv3"],
            'layer3.1.conv3': ["layer3.1.bn2", "layer3.1.bn3"],
            'layer3.1.bn3': ["layer3.1.conv3", "layer3.1.relu"],
            'layer3.1.relu': ["layer3.1.bn3", "layer3.2.conv1"],
            'layer3.2.conv1': ["layer3.1.relu", "layer3.2.bn1"],
            'layer3.2.bn1': ["layer3.2.conv1", "layer3.2.conv2"],
            # 'layer3.2.relu1': ["layer3.2.bn1", "layer3.2.conv2"],
            'layer3.2.conv2': ["layer3.2.bn1", "layer3.2.bn2"],
            'layer3.2.bn2': ["layer3.2.conv2", "layer3.2.conv3"],
            # 'layer3.2.relu2': ["layer3.2.bn2", "layer3.2.conv3"],
            'layer3.2.conv3': ["layer3.2.bn2", "layer3.2.bn3"],
            'layer3.2.bn3': ["layer3.2.conv3", "layer3.2.relu"],
            'layer3.2.relu': ["layer3.2.bn3", "layer3.3.conv1"],
            'layer3.3.conv1': ["layer3.2.relu", "layer3.3.bn1"],
            'layer3.3.bn1': ["layer3.3.conv1", "layer3.3.conv2"],
            # 'layer3.3.relu1': ["layer3.3.bn1", "layer3.3.conv2"],
            'layer3.3.conv2': ["layer3.3.bn1", "layer3.3.bn2"],
            'layer3.3.bn2': ["layer3.3.conv2", "layer3.3.conv3"],
            # 'layer3.3.relu2': ["layer3.3.bn2", "layer3.3.conv3"],
            'layer3.3.conv3': ["layer3.3.bn2", "layer3.3.bn3"],
            'layer3.3.bn3': ["layer3.3.conv3", "layer3.3.relu"],
            'layer3.3.relu': ["layer3.3.bn3", "layer3.4.conv1"],
            'layer3.4.conv1': ["layer3.3.relu", "layer3.4.bn1"],
            'layer3.4.bn1': ["layer3.4.conv1", "layer3.4.conv2"],
            # 'layer3.4.relu1': ["layer3.4.bn1", "layer3.4.conv2"],
            'layer3.4.conv2': ["layer3.4.bn1", "layer3.4.bn2"],
            'layer3.4.bn2': ["layer3.4.conv2", "layer3.4.conv3"],
            # 'layer3.4.relu2': ["layer3.4.bn2", "layer3.4.conv3"],
            'layer3.4.conv3': ["layer3.4.bn2", "layer3.4.bn3"],
            'layer3.4.bn3': ["layer3.4.conv3", "layer3.4.relu"],
            'layer3.4.relu': ["layer3.4.bn3", "layer3.5.conv1"],
            'layer3.5.conv1': ["layer3.4.relu", "layer3.5.bn1"],
            'layer3.5.bn1': ["layer3.5.conv1", "layer3.5.conv2"],
            # 'layer3.5.relu1': ["layer3.5.bn1", "layer3.5.conv2"],
            'layer3.5.conv2': ["layer3.5.bn1", "layer3.5.bn2"],
            'layer3.5.bn2': ["layer3.5.conv2", "layer3.5.conv3"],
            # 'layer3.5.relu2': ["layer3.5.bn2", "layer3.5.conv3"],
            'layer3.5.conv3': ["layer3.5.bn2", "layer3.5.bn3"],
            'layer3.5.bn3': ["layer3.5.conv3", "layer3.5.relu"],
            'layer3.5.relu': ["layer3.5.bn3", ["layer4.0.conv1", "layer4.0.down_sample_layer.0"]],
            'layer4.0.conv1': ["layer3.5.relu", "layer4.0.bn1"],
            'layer4.0.bn1': ["layer4.0.conv1", "layer4.0.conv2"],
            'layer4.0.conv2': ["layer4.0.bn1", "layer4.0.bn2"],
            'layer4.0.bn2': ["layer4.0.conv2", "layer4.0.conv3"],
            'layer4.0.conv3': ["layer4.0.bn2", "layer4.0.bn3"],
            'layer4.0.bn3': ["layer4.0.conv3", "layer4.0.relu"],
            'layer4.0.down_sample_layer.0': ["layer3.5.relu", "layer4.0.down_sample_layer.1"],
            'layer4.0.down_sample_layer.1': ["layer4.0.down_sample_layer.0", "layer4.0.relu"],
            'layer4.0.relu': [["layer4.0.bn3", "layer4.0.down_sample_layer.1"], "layer4.1.conv1"],
            'layer4.1.conv1': ["layer4.0.relu", "layer4.1.bn1"],
            'layer4.1.bn1': ["layer4.1.conv1", "layer4.1.conv2"],
            # 'layer4.1.relu1': ["layer4.1.bn1", "layer4.1.conv2"],
            'layer4.1.conv2': ["layer4.1.bn1", "layer4.1.bn2"],
            'layer4.1.bn2': ["layer4.1.conv2", "layer4.1.conv3"],
            # 'layer4.1.relu2': ["layer4.1.bn2", "layer4.1.conv3"],
            'layer4.1.conv3': ["layer4.1.bn2", "layer4.1.bn3"],
            'layer4.1.bn3': ["layer4.1.conv3", "layer4.1.relu"],
            'layer4.1.relu': ["layer4.1.bn3", "layer4.2.conv1"],
            'layer4.2.conv1': ["layer4.1.relu", "layer4.2.bn1"],
            'layer4.2.bn1': ["layer4.2.conv1", "layer4.2.conv2"],
            # 'layer4.2.relu1': ["layer4.2.bn1", "layer4.2.conv2"],
            'layer4.2.conv2': ["layer4.2.bn1", "layer4.2.bn2"],
            'layer4.2.bn2': ["layer4.2.conv2", "layer4.2.conv3"],
            # 'layer4.2.relu2': ["layer4.2.bn2", "layer4.2.conv3"],
            'layer4.2.conv3': ["layer4.2.bn2", "layer4.2.bn3"],
            'layer4.2.bn3': ["layer4.2.conv3", "layer4.2.relu"],
            'layer4.2.relu': ["layer4.2.bn3", "flatten"],
            'flatten': ["layer4.2.relu", "end_point"],
            'end_point': ["flatten", "OUTPUT"],
        }




    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, use_se=False, se_block=False):
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride, use_se=use_se)
        layers.append(resnet_block)
        if se_block:
            for _ in range(1, layer_num - 1):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
            layers.append(resnet_block)
        else:
            for _ in range(1, layer_num):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_se:
            x = self.conv1_0(x)
            x = self.bn1_0(x)
            x = self.relu(x)
            x = self.conv1_1(x)
            x = self.bn1_1(x)
            x = self.relu(x)
            x = self.conv1_2(x)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.res_base:
            x = self.pad(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5,dim=(2, 3),keepdim=True)[1]
        out = self.flatten(out)
        out = self.end_point(out)

        return out

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self, layer_name, new_layer):
        if 'conv1' == layer_name:
            self.conv1 = new_layer
            self.layer_names["conv1"] = new_layer
            self.origin_layer_names["conv1"] = new_layer
        elif 'bn1' == layer_name:
            self.bn1 = new_layer
            self.layer_names["bn1"] = new_layer
            self.origin_layer_names["bn1"] = new_layer
        elif 'relu' == layer_name:
            self.relu = new_layer
            self.layer_names["relu"] = new_layer
            self.origin_layer_names["relu"] = new_layer
        elif 'maxpool' == layer_name:
            self.maxpool = new_layer
            self.layer_names["maxpool"] = new_layer
            self.origin_layer_names["maxpool"] = new_layer
        elif 'layer1' == layer_name:
            self.layer1 = new_layer
            self.layer_names["layer1"] = new_layer
            self.origin_layer_names["layer1"] = new_layer
        elif 'layer1.0' == layer_name:
            self.layer1[0] = new_layer
            self.layer_names["layer1.0"] = new_layer
            self.origin_layer_names["layer1.0"] = new_layer
        elif 'layer1.0.conv1' == layer_name:
            self.layer1[0].conv1 = new_layer
            self.layer_names["layer1.0.conv1"] = new_layer
            self.origin_layer_names["layer1.0.conv1"] = new_layer
        elif 'layer1.0.bn1' == layer_name:
            self.layer1[0].bn1 = new_layer
            self.layer_names["layer1.0.bn1"] = new_layer
            self.origin_layer_names["layer1.0.bn1"] = new_layer
        elif 'layer1.0.conv2' == layer_name:
            self.layer1[0].conv2 = new_layer
            self.layer_names["layer1.0.conv2"] = new_layer
            self.origin_layer_names["layer1.0.conv2"] = new_layer
        elif 'layer1.0.bn2' == layer_name:
            self.layer1[0].bn2 = new_layer
            self.layer_names["layer1.0.bn2"] = new_layer
            self.origin_layer_names["layer1.0.bn2"] = new_layer
        elif 'layer1.0.conv3' == layer_name:
            self.layer1[0].conv3 = new_layer
            self.layer_names["layer1.0.conv3"] = new_layer
            self.origin_layer_names["layer1.0.conv3"] = new_layer
        elif 'layer1.0.bn3' == layer_name:
            self.layer1[0].bn3 = new_layer
            self.layer_names["layer1.0.bn3"] = new_layer
            self.origin_layer_names["layer1.0.bn3"] = new_layer
        elif 'layer1.0.relu' == layer_name:
            self.layer1[0].relu = new_layer
            self.layer_names["layer1.0.relu"] = new_layer
            self.origin_layer_names["layer1.0.relu"] = new_layer
        elif 'layer1.0.down_sample_layer' == layer_name:
            self.layer1[0].down_sample_layer = new_layer
            self.layer_names["layer1.0.down_sample_layer"] = new_layer
            self.origin_layer_names["layer1.0.down_sample_layer"] = new_layer
        elif 'layer1.0.down_sample_layer.0' == layer_name:
            self.layer1[0].down_sample_layer[0] = new_layer
            self.layer_names["layer1.0.down_sample_layer.0"] = new_layer
            self.origin_layer_names["layer1.0.down_sample_layer.0"] = new_layer
        elif 'layer1.0.down_sample_layer.1' == layer_name:
            self.layer1[0].down_sample_layer[1] = new_layer
            self.layer_names["layer1.0.down_sample_layer.1"] = new_layer
            self.origin_layer_names["layer1.0.down_sample_layer.1"] = new_layer
        elif 'layer1.1' == layer_name:
            self.layer1[1] = new_layer
            self.layer_names["layer1.1"] = new_layer
            self.origin_layer_names["layer1.1"] = new_layer
        elif 'layer1.1.conv1' == layer_name:
            self.layer1[1].conv1 = new_layer
            self.layer_names["layer1.1.conv1"] = new_layer
            self.origin_layer_names["layer1.1.conv1"] = new_layer
        elif 'layer1.1.bn1' == layer_name:
            self.layer1[1].bn1 = new_layer
            self.layer_names["layer1.1.bn1"] = new_layer
            self.origin_layer_names["layer1.1.bn1"] = new_layer
        elif 'layer1.1.conv2' == layer_name:
            self.layer1[1].conv2 = new_layer
            self.layer_names["layer1.1.conv2"] = new_layer
            self.origin_layer_names["layer1.1.conv2"] = new_layer
        elif 'layer1.1.bn2' == layer_name:
            self.layer1[1].bn2 = new_layer
            self.layer_names["layer1.1.bn2"] = new_layer
            self.origin_layer_names["layer1.1.bn2"] = new_layer
        elif 'layer1.1.conv3' == layer_name:
            self.layer1[1].conv3 = new_layer
            self.layer_names["layer1.1.conv3"] = new_layer
            self.origin_layer_names["layer1.1.conv3"] = new_layer
        elif 'layer1.1.bn3' == layer_name:
            self.layer1[1].bn3 = new_layer
            self.layer_names["layer1.1.bn3"] = new_layer
            self.origin_layer_names["layer1.1.bn3"] = new_layer
        elif 'layer1.1.relu' == layer_name:
            self.layer1[1].relu = new_layer
            self.layer_names["layer1.1.relu"] = new_layer
            self.origin_layer_names["layer1.1.relu"] = new_layer
        elif 'layer1.2' == layer_name:
            self.layer1[2] = new_layer
            self.layer_names["layer1.2"] = new_layer
            self.origin_layer_names["layer1.2"] = new_layer
        elif 'layer1.2.conv1' == layer_name:
            self.layer1[2].conv1 = new_layer
            self.layer_names["layer1.2.conv1"] = new_layer
            self.origin_layer_names["layer1.2.conv1"] = new_layer
        elif 'layer1.2.bn1' == layer_name:
            self.layer1[2].bn1 = new_layer
            self.layer_names["layer1.2.bn1"] = new_layer
            self.origin_layer_names["layer1.2.bn1"] = new_layer
        elif 'layer1.2.conv2' == layer_name:
            self.layer1[2].conv2 = new_layer
            self.layer_names["layer1.2.conv2"] = new_layer
            self.origin_layer_names["layer1.2.conv2"] = new_layer
        elif 'layer1.2.bn2' == layer_name:
            self.layer1[2].bn2 = new_layer
            self.layer_names["layer1.2.bn2"] = new_layer
            self.origin_layer_names["layer1.2.bn2"] = new_layer
        elif 'layer1.2.conv3' == layer_name:
            self.layer1[2].conv3 = new_layer
            self.layer_names["layer1.2.conv3"] = new_layer
            self.origin_layer_names["layer1.2.conv3"] = new_layer
        elif 'layer1.2.bn3' == layer_name:
            self.layer1[2].bn3 = new_layer
            self.layer_names["layer1.2.bn3"] = new_layer
            self.origin_layer_names["layer1.2.bn3"] = new_layer
        elif 'layer1.2.relu' == layer_name:
            self.layer1[2].relu = new_layer
            self.layer_names["layer1.2.relu"] = new_layer
            self.origin_layer_names["layer1.2.relu"] = new_layer
        elif 'layer2' == layer_name:
            self.layer2 = new_layer
            self.layer_names["layer2"] = new_layer
            self.origin_layer_names["layer2"] = new_layer
        elif 'layer2.0' == layer_name:
            self.layer2[0] = new_layer
            self.layer_names["layer2.0"] = new_layer
            self.origin_layer_names["layer2.0"] = new_layer
        elif 'layer2.0.conv1' == layer_name:
            self.layer2[0].conv1 = new_layer
            self.layer_names["layer2.0.conv1"] = new_layer
            self.origin_layer_names["layer2.0.conv1"] = new_layer
        elif 'layer2.0.bn1' == layer_name:
            self.layer2[0].bn1 = new_layer
            self.layer_names["layer2.0.bn1"] = new_layer
            self.origin_layer_names["layer2.0.bn1"] = new_layer
        elif 'layer2.0.conv2' == layer_name:
            self.layer2[0].conv2 = new_layer
            self.layer_names["layer2.0.conv2"] = new_layer
            self.origin_layer_names["layer2.0.conv2"] = new_layer
        elif 'layer2.0.bn2' == layer_name:
            self.layer2[0].bn2 = new_layer
            self.layer_names["layer2.0.bn2"] = new_layer
            self.origin_layer_names["layer2.0.bn2"] = new_layer
        elif 'layer2.0.conv3' == layer_name:
            self.layer2[0].conv3 = new_layer
            self.layer_names["layer2.0.conv3"] = new_layer
            self.origin_layer_names["layer2.0.conv3"] = new_layer
        elif 'layer2.0.bn3' == layer_name:
            self.layer2[0].bn3 = new_layer
            self.layer_names["layer2.0.bn3"] = new_layer
            self.origin_layer_names["layer2.0.bn3"] = new_layer
        elif 'layer2.0.relu' == layer_name:
            self.layer2[0].relu = new_layer
            self.layer_names["layer2.0.relu"] = new_layer
            self.origin_layer_names["layer2.0.relu"] = new_layer
        elif 'layer2.0.down_sample_layer' == layer_name:
            self.layer2[0].down_sample_layer = new_layer
            self.layer_names["layer2.0.down_sample_layer"] = new_layer
            self.origin_layer_names["layer2.0.down_sample_layer"] = new_layer
        elif 'layer2.0.down_sample_layer.0' == layer_name:
            self.layer2[0].down_sample_layer[0] = new_layer
            self.layer_names["layer2.0.down_sample_layer.0"] = new_layer
            self.origin_layer_names["layer2.0.down_sample_layer.0"] = new_layer
        elif 'layer2.0.down_sample_layer.1' == layer_name:
            self.layer2[0].down_sample_layer[1] = new_layer
            self.layer_names["layer2.0.down_sample_layer.1"] = new_layer
            self.origin_layer_names["layer2.0.down_sample_layer.1"] = new_layer
        elif 'layer2.1' == layer_name:
            self.layer2[1] = new_layer
            self.layer_names["layer2.1"] = new_layer
            self.origin_layer_names["layer2.1"] = new_layer
        elif 'layer2.1.conv1' == layer_name:
            self.layer2[1].conv1 = new_layer
            self.layer_names["layer2.1.conv1"] = new_layer
            self.origin_layer_names["layer2.1.conv1"] = new_layer
        elif 'layer2.1.bn1' == layer_name:
            self.layer2[1].bn1 = new_layer
            self.layer_names["layer2.1.bn1"] = new_layer
            self.origin_layer_names["layer2.1.bn1"] = new_layer
        elif 'layer2.1.conv2' == layer_name:
            self.layer2[1].conv2 = new_layer
            self.layer_names["layer2.1.conv2"] = new_layer
            self.origin_layer_names["layer2.1.conv2"] = new_layer
        elif 'layer2.1.bn2' == layer_name:
            self.layer2[1].bn2 = new_layer
            self.layer_names["layer2.1.bn2"] = new_layer
            self.origin_layer_names["layer2.1.bn2"] = new_layer
        elif 'layer2.1.conv3' == layer_name:
            self.layer2[1].conv3 = new_layer
            self.layer_names["layer2.1.conv3"] = new_layer
            self.origin_layer_names["layer2.1.conv3"] = new_layer
        elif 'layer2.1.bn3' == layer_name:
            self.layer2[1].bn3 = new_layer
            self.layer_names["layer2.1.bn3"] = new_layer
            self.origin_layer_names["layer2.1.bn3"] = new_layer
        elif 'layer2.1.relu' == layer_name:
            self.layer2[1].relu = new_layer
            self.layer_names["layer2.1.relu"] = new_layer
            self.origin_layer_names["layer2.1.relu"] = new_layer
        elif 'layer2.2' == layer_name:
            self.layer2[2] = new_layer
            self.layer_names["layer2.2"] = new_layer
            self.origin_layer_names["layer2.2"] = new_layer
        elif 'layer2.2.conv1' == layer_name:
            self.layer2[2].conv1 = new_layer
            self.layer_names["layer2.2.conv1"] = new_layer
            self.origin_layer_names["layer2.2.conv1"] = new_layer
        elif 'layer2.2.bn1' == layer_name:
            self.layer2[2].bn1 = new_layer
            self.layer_names["layer2.2.bn1"] = new_layer
            self.origin_layer_names["layer2.2.bn1"] = new_layer
        elif 'layer2.2.conv2' == layer_name:
            self.layer2[2].conv2 = new_layer
            self.layer_names["layer2.2.conv2"] = new_layer
            self.origin_layer_names["layer2.2.conv2"] = new_layer
        elif 'layer2.2.bn2' == layer_name:
            self.layer2[2].bn2 = new_layer
            self.layer_names["layer2.2.bn2"] = new_layer
            self.origin_layer_names["layer2.2.bn2"] = new_layer
        elif 'layer2.2.conv3' == layer_name:
            self.layer2[2].conv3 = new_layer
            self.layer_names["layer2.2.conv3"] = new_layer
            self.origin_layer_names["layer2.2.conv3"] = new_layer
        elif 'layer2.2.bn3' == layer_name:
            self.layer2[2].bn3 = new_layer
            self.layer_names["layer2.2.bn3"] = new_layer
            self.origin_layer_names["layer2.2.bn3"] = new_layer
        elif 'layer2.2.relu' == layer_name:
            self.layer2[2].relu = new_layer
            self.layer_names["layer2.2.relu"] = new_layer
            self.origin_layer_names["layer2.2.relu"] = new_layer
        elif 'layer2.3' == layer_name:
            self.layer2[3] = new_layer
            self.layer_names["layer2.3"] = new_layer
            self.origin_layer_names["layer2.3"] = new_layer
        elif 'layer2.3.conv1' == layer_name:
            self.layer2[3].conv1 = new_layer
            self.layer_names["layer2.3.conv1"] = new_layer
            self.origin_layer_names["layer2.3.conv1"] = new_layer
        elif 'layer2.3.bn1' == layer_name:
            self.layer2[3].bn1 = new_layer
            self.layer_names["layer2.3.bn1"] = new_layer
            self.origin_layer_names["layer2.3.bn1"] = new_layer
        elif 'layer2.3.conv2' == layer_name:
            self.layer2[3].conv2 = new_layer
            self.layer_names["layer2.3.conv2"] = new_layer
            self.origin_layer_names["layer2.3.conv2"] = new_layer
        elif 'layer2.3.bn2' == layer_name:
            self.layer2[3].bn2 = new_layer
            self.layer_names["layer2.3.bn2"] = new_layer
            self.origin_layer_names["layer2.3.bn2"] = new_layer
        elif 'layer2.3.conv3' == layer_name:
            self.layer2[3].conv3 = new_layer
            self.layer_names["layer2.3.conv3"] = new_layer
            self.origin_layer_names["layer2.3.conv3"] = new_layer
        elif 'layer2.3.bn3' == layer_name:
            self.layer2[3].bn3 = new_layer
            self.layer_names["layer2.3.bn3"] = new_layer
            self.origin_layer_names["layer2.3.bn3"] = new_layer
        elif 'layer2.3.relu' == layer_name:
            self.layer2[3].relu = new_layer
            self.layer_names["layer2.3.relu"] = new_layer
            self.origin_layer_names["layer2.3.relu"] = new_layer
        elif 'layer3' == layer_name:
            self.layer3 = new_layer
            self.layer_names["layer3"] = new_layer
            self.origin_layer_names["layer3"] = new_layer
        elif 'layer3.0' == layer_name:
            self.layer3[0] = new_layer
            self.layer_names["layer3.0"] = new_layer
            self.origin_layer_names["layer3.0"] = new_layer
        elif 'layer3.0.conv1' == layer_name:
            self.layer3[0].conv1 = new_layer
            self.layer_names["layer3.0.conv1"] = new_layer
            self.origin_layer_names["layer3.0.conv1"] = new_layer
        elif 'layer3.0.bn1' == layer_name:
            self.layer3[0].bn1 = new_layer
            self.layer_names["layer3.0.bn1"] = new_layer
            self.origin_layer_names["layer3.0.bn1"] = new_layer
        elif 'layer3.0.conv2' == layer_name:
            self.layer3[0].conv2 = new_layer
            self.layer_names["layer3.0.conv2"] = new_layer
            self.origin_layer_names["layer3.0.conv2"] = new_layer
        elif 'layer3.0.bn2' == layer_name:
            self.layer3[0].bn2 = new_layer
            self.layer_names["layer3.0.bn2"] = new_layer
            self.origin_layer_names["layer3.0.bn2"] = new_layer
        elif 'layer3.0.conv3' == layer_name:
            self.layer3[0].conv3 = new_layer
            self.layer_names["layer3.0.conv3"] = new_layer
            self.origin_layer_names["layer3.0.conv3"] = new_layer
        elif 'layer3.0.bn3' == layer_name:
            self.layer3[0].bn3 = new_layer
            self.layer_names["layer3.0.bn3"] = new_layer
            self.origin_layer_names["layer3.0.bn3"] = new_layer
        elif 'layer3.0.relu' == layer_name:
            self.layer3[0].relu = new_layer
            self.layer_names["layer3.0.relu"] = new_layer
            self.origin_layer_names["layer3.0.relu"] = new_layer
        elif 'layer3.0.down_sample_layer' == layer_name:
            self.layer3[0].down_sample_layer = new_layer
            self.layer_names["layer3.0.down_sample_layer"] = new_layer
            self.origin_layer_names["layer3.0.down_sample_layer"] = new_layer
        elif 'layer3.0.down_sample_layer.0' == layer_name:
            self.layer3[0].down_sample_layer[0] = new_layer
            self.layer_names["layer3.0.down_sample_layer.0"] = new_layer
            self.origin_layer_names["layer3.0.down_sample_layer.0"] = new_layer
        elif 'layer3.0.down_sample_layer.1' == layer_name:
            self.layer3[0].down_sample_layer[1] = new_layer
            self.layer_names["layer3.0.down_sample_layer.1"] = new_layer
            self.origin_layer_names["layer3.0.down_sample_layer.1"] = new_layer
        elif 'layer3.1' == layer_name:
            self.layer3[1] = new_layer
            self.layer_names["layer3.1"] = new_layer
            self.origin_layer_names["layer3.1"] = new_layer
        elif 'layer3.1.conv1' == layer_name:
            self.layer3[1].conv1 = new_layer
            self.layer_names["layer3.1.conv1"] = new_layer
            self.origin_layer_names["layer3.1.conv1"] = new_layer
        elif 'layer3.1.bn1' == layer_name:
            self.layer3[1].bn1 = new_layer
            self.layer_names["layer3.1.bn1"] = new_layer
            self.origin_layer_names["layer3.1.bn1"] = new_layer
        elif 'layer3.1.conv2' == layer_name:
            self.layer3[1].conv2 = new_layer
            self.layer_names["layer3.1.conv2"] = new_layer
            self.origin_layer_names["layer3.1.conv2"] = new_layer
        elif 'layer3.1.bn2' == layer_name:
            self.layer3[1].bn2 = new_layer
            self.layer_names["layer3.1.bn2"] = new_layer
            self.origin_layer_names["layer3.1.bn2"] = new_layer
        elif 'layer3.1.conv3' == layer_name:
            self.layer3[1].conv3 = new_layer
            self.layer_names["layer3.1.conv3"] = new_layer
            self.origin_layer_names["layer3.1.conv3"] = new_layer
        elif 'layer3.1.bn3' == layer_name:
            self.layer3[1].bn3 = new_layer
            self.layer_names["layer3.1.bn3"] = new_layer
            self.origin_layer_names["layer3.1.bn3"] = new_layer
        elif 'layer3.1.relu' == layer_name:
            self.layer3[1].relu = new_layer
            self.layer_names["layer3.1.relu"] = new_layer
            self.origin_layer_names["layer3.1.relu"] = new_layer
        elif 'layer3.2' == layer_name:
            self.layer3[2] = new_layer
            self.layer_names["layer3.2"] = new_layer
            self.origin_layer_names["layer3.2"] = new_layer
        elif 'layer3.2.conv1' == layer_name:
            self.layer3[2].conv1 = new_layer
            self.layer_names["layer3.2.conv1"] = new_layer
            self.origin_layer_names["layer3.2.conv1"] = new_layer
        elif 'layer3.2.bn1' == layer_name:
            self.layer3[2].bn1 = new_layer
            self.layer_names["layer3.2.bn1"] = new_layer
            self.origin_layer_names["layer3.2.bn1"] = new_layer
        elif 'layer3.2.conv2' == layer_name:
            self.layer3[2].conv2 = new_layer
            self.layer_names["layer3.2.conv2"] = new_layer
            self.origin_layer_names["layer3.2.conv2"] = new_layer
        elif 'layer3.2.bn2' == layer_name:
            self.layer3[2].bn2 = new_layer
            self.layer_names["layer3.2.bn2"] = new_layer
            self.origin_layer_names["layer3.2.bn2"] = new_layer
        elif 'layer3.2.conv3' == layer_name:
            self.layer3[2].conv3 = new_layer
            self.layer_names["layer3.2.conv3"] = new_layer
            self.origin_layer_names["layer3.2.conv3"] = new_layer
        elif 'layer3.2.bn3' == layer_name:
            self.layer3[2].bn3 = new_layer
            self.layer_names["layer3.2.bn3"] = new_layer
            self.origin_layer_names["layer3.2.bn3"] = new_layer
        elif 'layer3.2.relu' == layer_name:
            self.layer3[2].relu = new_layer
            self.layer_names["layer3.2.relu"] = new_layer
            self.origin_layer_names["layer3.2.relu"] = new_layer
        elif 'layer3.3' == layer_name:
            self.layer3[3] = new_layer
            self.layer_names["layer3.3"] = new_layer
            self.origin_layer_names["layer3.3"] = new_layer
        elif 'layer3.3.conv1' == layer_name:
            self.layer3[3].conv1 = new_layer
            self.layer_names["layer3.3.conv1"] = new_layer
            self.origin_layer_names["layer3.3.conv1"] = new_layer
        elif 'layer3.3.bn1' == layer_name:
            self.layer3[3].bn1 = new_layer
            self.layer_names["layer3.3.bn1"] = new_layer
            self.origin_layer_names["layer3.3.bn1"] = new_layer
        elif 'layer3.3.conv2' == layer_name:
            self.layer3[3].conv2 = new_layer
            self.layer_names["layer3.3.conv2"] = new_layer
            self.origin_layer_names["layer3.3.conv2"] = new_layer
        elif 'layer3.3.bn2' == layer_name:
            self.layer3[3].bn2 = new_layer
            self.layer_names["layer3.3.bn2"] = new_layer
            self.origin_layer_names["layer3.3.bn2"] = new_layer
        elif 'layer3.3.conv3' == layer_name:
            self.layer3[3].conv3 = new_layer
            self.layer_names["layer3.3.conv3"] = new_layer
            self.origin_layer_names["layer3.3.conv3"] = new_layer
        elif 'layer3.3.bn3' == layer_name:
            self.layer3[3].bn3 = new_layer
            self.layer_names["layer3.3.bn3"] = new_layer
            self.origin_layer_names["layer3.3.bn3"] = new_layer
        elif 'layer3.3.relu' == layer_name:
            self.layer3[3].relu = new_layer
            self.layer_names["layer3.3.relu"] = new_layer
            self.origin_layer_names["layer3.3.relu"] = new_layer
        elif 'layer3.4' == layer_name:
            self.layer3[4] = new_layer
            self.layer_names["layer3.4"] = new_layer
            self.origin_layer_names["layer3.4"] = new_layer
        elif 'layer3.4.conv1' == layer_name:
            self.layer3[4].conv1 = new_layer
            self.layer_names["layer3.4.conv1"] = new_layer
            self.origin_layer_names["layer3.4.conv1"] = new_layer
        elif 'layer3.4.bn1' == layer_name:
            self.layer3[4].bn1 = new_layer
            self.layer_names["layer3.4.bn1"] = new_layer
            self.origin_layer_names["layer3.4.bn1"] = new_layer
        elif 'layer3.4.conv2' == layer_name:
            self.layer3[4].conv2 = new_layer
            self.layer_names["layer3.4.conv2"] = new_layer
            self.origin_layer_names["layer3.4.conv2"] = new_layer
        elif 'layer3.4.bn2' == layer_name:
            self.layer3[4].bn2 = new_layer
            self.layer_names["layer3.4.bn2"] = new_layer
            self.origin_layer_names["layer3.4.bn2"] = new_layer
        elif 'layer3.4.conv3' == layer_name:
            self.layer3[4].conv3 = new_layer
            self.layer_names["layer3.4.conv3"] = new_layer
            self.origin_layer_names["layer3.4.conv3"] = new_layer
        elif 'layer3.4.bn3' == layer_name:
            self.layer3[4].bn3 = new_layer
            self.layer_names["layer3.4.bn3"] = new_layer
            self.origin_layer_names["layer3.4.bn3"] = new_layer
        elif 'layer3.4.relu' == layer_name:
            self.layer3[4].relu = new_layer
            self.layer_names["layer3.4.relu"] = new_layer
            self.origin_layer_names["layer3.4.relu"] = new_layer
        elif 'layer3.5' == layer_name:
            self.layer3[5] = new_layer
            self.layer_names["layer3.5"] = new_layer
            self.origin_layer_names["layer3.5"] = new_layer
        elif 'layer3.5.conv1' == layer_name:
            self.layer3[5].conv1 = new_layer
            self.layer_names["layer3.5.conv1"] = new_layer
            self.origin_layer_names["layer3.5.conv1"] = new_layer
        elif 'layer3.5.bn1' == layer_name:
            self.layer3[5].bn1 = new_layer
            self.layer_names["layer3.5.bn1"] = new_layer
            self.origin_layer_names["layer3.5.bn1"] = new_layer
        elif 'layer3.5.conv2' == layer_name:
            self.layer3[5].conv2 = new_layer
            self.layer_names["layer3.5.conv2"] = new_layer
            self.origin_layer_names["layer3.5.conv2"] = new_layer
        elif 'layer3.5.bn2' == layer_name:
            self.layer3[5].bn2 = new_layer
            self.layer_names["layer3.5.bn2"] = new_layer
            self.origin_layer_names["layer3.5.bn2"] = new_layer
        elif 'layer3.5.conv3' == layer_name:
            self.layer3[5].conv3 = new_layer
            self.layer_names["layer3.5.conv3"] = new_layer
            self.origin_layer_names["layer3.5.conv3"] = new_layer
        elif 'layer3.5.bn3' == layer_name:
            self.layer3[5].bn3 = new_layer
            self.layer_names["layer3.5.bn3"] = new_layer
            self.origin_layer_names["layer3.5.bn3"] = new_layer
        elif 'layer3.5.relu' == layer_name:
            self.layer3[5].relu = new_layer
            self.layer_names["layer3.5.relu"] = new_layer
            self.origin_layer_names["layer3.5.relu"] = new_layer
        elif 'layer4' == layer_name:
            self.layer4 = new_layer
            self.layer_names["layer4"] = new_layer
            self.origin_layer_names["layer4"] = new_layer
        elif 'layer4.0' == layer_name:
            self.layer4[0] = new_layer
            self.layer_names["layer4.0"] = new_layer
            self.origin_layer_names["layer4.0"] = new_layer
        elif 'layer4.0.conv1' == layer_name:
            self.layer4[0].conv1 = new_layer
            self.layer_names["layer4.0.conv1"] = new_layer
            self.origin_layer_names["layer4.0.conv1"] = new_layer
        elif 'layer4.0.bn1' == layer_name:
            self.layer4[0].bn1 = new_layer
            self.layer_names["layer4.0.bn1"] = new_layer
            self.origin_layer_names["layer4.0.bn1"] = new_layer
        elif 'layer4.0.conv2' == layer_name:
            self.layer4[0].conv2 = new_layer
            self.layer_names["layer4.0.conv2"] = new_layer
            self.origin_layer_names["layer4.0.conv2"] = new_layer
        elif 'layer4.0.bn2' == layer_name:
            self.layer4[0].bn2 = new_layer
            self.layer_names["layer4.0.bn2"] = new_layer
            self.origin_layer_names["layer4.0.bn2"] = new_layer
        elif 'layer4.0.conv3' == layer_name:
            self.layer4[0].conv3 = new_layer
            self.layer_names["layer4.0.conv3"] = new_layer
            self.origin_layer_names["layer4.0.conv3"] = new_layer
        elif 'layer4.0.bn3' == layer_name:
            self.layer4[0].bn3 = new_layer
            self.layer_names["layer4.0.bn3"] = new_layer
            self.origin_layer_names["layer4.0.bn3"] = new_layer
        elif 'layer4.0.relu' == layer_name:
            self.layer4[0].relu = new_layer
            self.layer_names["layer4.0.relu"] = new_layer
            self.origin_layer_names["layer4.0.relu"] = new_layer
        elif 'layer4.0.down_sample_layer' == layer_name:
            self.layer4[0].down_sample_layer = new_layer
            self.layer_names["layer4.0.down_sample_layer"] = new_layer
            self.origin_layer_names["layer4.0.down_sample_layer"] = new_layer
        elif 'layer4.0.down_sample_layer.0' == layer_name:
            self.layer4[0].down_sample_layer[0] = new_layer
            self.layer_names["layer4.0.down_sample_layer.0"] = new_layer
            self.origin_layer_names["layer4.0.down_sample_layer.0"] = new_layer
        elif 'layer4.0.down_sample_layer.1' == layer_name:
            self.layer4[0].down_sample_layer[1] = new_layer
            self.layer_names["layer4.0.down_sample_layer.1"] = new_layer
            self.origin_layer_names["layer4.0.down_sample_layer.1"] = new_layer
        elif 'layer4.1' == layer_name:
            self.layer4[1] = new_layer
            self.layer_names["layer4.1"] = new_layer
            self.origin_layer_names["layer4.1"] = new_layer
        elif 'layer4.1.conv1' == layer_name:
            self.layer4[1].conv1 = new_layer
            self.layer_names["layer4.1.conv1"] = new_layer
            self.origin_layer_names["layer4.1.conv1"] = new_layer
        elif 'layer4.1.bn1' == layer_name:
            self.layer4[1].bn1 = new_layer
            self.layer_names["layer4.1.bn1"] = new_layer
            self.origin_layer_names["layer4.1.bn1"] = new_layer
        elif 'layer4.1.conv2' == layer_name:
            self.layer4[1].conv2 = new_layer
            self.layer_names["layer4.1.conv2"] = new_layer
            self.origin_layer_names["layer4.1.conv2"] = new_layer
        elif 'layer4.1.bn2' == layer_name:
            self.layer4[1].bn2 = new_layer
            self.layer_names["layer4.1.bn2"] = new_layer
            self.origin_layer_names["layer4.1.bn2"] = new_layer
        elif 'layer4.1.conv3' == layer_name:
            self.layer4[1].conv3 = new_layer
            self.layer_names["layer4.1.conv3"] = new_layer
            self.origin_layer_names["layer4.1.conv3"] = new_layer
        elif 'layer4.1.bn3' == layer_name:
            self.layer4[1].bn3 = new_layer
            self.layer_names["layer4.1.bn3"] = new_layer
            self.origin_layer_names["layer4.1.bn3"] = new_layer
        elif 'layer4.1.relu' == layer_name:
            self.layer4[1].relu = new_layer
            self.layer_names["layer4.1.relu"] = new_layer
            self.origin_layer_names["layer4.1.relu"] = new_layer
        elif 'layer4.2' == layer_name:
            self.layer4[2] = new_layer
            self.layer_names["layer4.2"] = new_layer
            self.origin_layer_names["layer4.2"] = new_layer
        elif 'layer4.2.conv1' == layer_name:
            self.layer4[2].conv1 = new_layer
            self.layer_names["layer4.2.conv1"] = new_layer
            self.origin_layer_names["layer4.2.conv1"] = new_layer
        elif 'layer4.2.bn1' == layer_name:
            self.layer4[2].bn1 = new_layer
            self.layer_names["layer4.2.bn1"] = new_layer
            self.origin_layer_names["layer4.2.bn1"] = new_layer
        elif 'layer4.2.conv2' == layer_name:
            self.layer4[2].conv2 = new_layer
            self.layer_names["layer4.2.conv2"] = new_layer
            self.origin_layer_names["layer4.2.conv2"] = new_layer
        elif 'layer4.2.bn2' == layer_name:
            self.layer4[2].bn2 = new_layer
            self.layer_names["layer4.2.bn2"] = new_layer
            self.origin_layer_names["layer4.2.bn2"] = new_layer
        elif 'layer4.2.conv3' == layer_name:
            self.layer4[2].conv3 = new_layer
            self.layer_names["layer4.2.conv3"] = new_layer
            self.origin_layer_names["layer4.2.conv3"] = new_layer
        elif 'layer4.2.bn3' == layer_name:
            self.layer4[2].bn3 = new_layer
            self.layer_names["layer4.2.bn3"] = new_layer
            self.origin_layer_names["layer4.2.bn3"] = new_layer
        elif 'layer4.2.relu' == layer_name:
            self.layer4[2].relu = new_layer
            self.layer_names["layer4.2.relu"] = new_layer
            self.origin_layer_names["layer4.2.relu"] = new_layer
        elif 'flatten' == layer_name:
            self.flatten = new_layer
            self.layer_names["flatten"] = new_layer
            self.origin_layer_names["flatten"] = new_layer
        elif 'end_point' == layer_name:
            self.end_point = new_layer
            self.layer_names["end_point"] = new_layer
            self.origin_layer_names["end_point"] = new_layer


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


def resnet50(class_num=10):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = resnet(10)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


if __name__ == '__main__':
    # import numpy as np
    #
    # x, y = np.load("")
    net = resnet50(10)
    data = torch.Tensor(np.random.randn(1, 3, 224, 224))
    print(net(data).shape)
