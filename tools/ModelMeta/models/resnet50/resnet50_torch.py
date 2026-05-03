import math

import mindspore
import numpy as np

from infoplus.TorchInfoPlus import torchinfoplus
from models.resnet50.resnet50 import create_cifar10_dataset

np.random.seed(6)
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

device = "CPU"


# Assuming `config` is a similar configuration object in your PyTorch code
# from models.resnet.src.model_utils.config import config


def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
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
    return np.random.normal(0, std, size=inputs_shape)


# def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
#     tensor = torch.zeros(inputs_shape)
#     fan = init._calculate_correct_fan(tensor, mode)
#     gain = calculate_gain(nonlinearity, a)
#     std = gain / math.sqrt(fan)
#     bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
#     return torch.uniform(-bound, bound, size=inputs_shape)


def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    # if use_se:
    #     # weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=3)
    #     pass
    # else:
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = torch.tensor(np.zeros(weight_shape), dtype=torch.float32)
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
    conv.weight = nn.Parameter(weight)
    return conv


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    # if use_se:
    #     # weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
    #     pass
    # else:
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = torch.tensor(np.zeros(weight_shape), dtype=torch.float32)
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)
    conv.weight = nn.Parameter(weight)
    return conv


def _conv7x7(in_channel, out_channel, stride=1, use_se=False):
    # if use_se:
    #     weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=7)
        # pass
    # else:
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = torch.tensor(np.zeros(weight_shape), dtype=torch.float32)
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=stride, padding=3, bias=False)
    conv.weight = nn.Parameter(weight)
    return conv


def _bn(channel, res_base=False):
    # if res_base:
    #     return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.9)
    # print("channel: ", channel)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.1)


def _bn_last(channel):
    bn = nn.BatchNorm2d(channel, eps=1e-4, momentum=0.1)
    # nn.init.zeros_(bn.weight)
    # nn.init.zeros_(bn.bias)
    return bn


def _fc(in_channel, out_channel, use_se=False):
    if use_se:
        # weight = np.random.normal(loc=0, scale=0.01, size=(out_channel, in_channel))
        # # weight = torch.Tensor(weight)
        pass
    else:
        weight_shape = (out_channel, in_channel)
        weight = torch.tensor(np.zeros(weight_shape), dtype=torch.float32)
    fc = nn.Linear(in_channel, out_channel)
    fc.weight = nn.Parameter(weight)
    # nn.init.zeros_(fc.bias)
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
                    self.down_sample_layer = nn.Sequential(
                        _conv1x1(in_channel, out_channel, stride, use_se=self.use_se), _bn(out_channel))
                else:
                    self.down_sample_layer = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                                           _conv1x1(in_channel, out_channel, 1, use_se=self.use_se),
                                                           _bn(out_channel))
            else:
                self.down_sample_layer = nn.Sequential(_conv1x1(in_channel, out_channel, stride, use_se=self.use_se),
                                                       _bn(out_channel))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # if self.use_se and self.stride != 1:
        #     out = self.e2(out)
        # else:
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layer_nums, in_channels, out_channels, strides, num_classes):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        # self.use_se = use_se
        # self.se_block = False
        # if self.use_se:
        #     self.se_block = True

        # if self.use_se:
        #     self.conv1_0 = _conv3x3(3, 32, stride=2, use_se=self.use_se)
        #     self.bn1_0 = _bn(32)
        #     self.conv1_1 = _conv3x3(32, 32, stride=1, use_se=self.use_se)
        #     self.bn1_1 = _bn(32)
        #     self.conv1_2 = _conv3x3(32, 64, stride=1, use_se=self.use_se)
        # else:
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64, False)
        self.relu = nn.ReLU()

        # if self.res_base:
        #     self.pad = nn.ConstantPad2d((1, 1, 1, 1), 0)  # pad (left, right, top, bottom)
        #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # "valid" padding in PyTorch
        # else:
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # "same" padding in PyTorch

        self.layer1 = self._make_layer(block, layer_nums[0], in_channels[0], out_channels[0], strides[0], False)
        self.layer2 = self._make_layer(block, layer_nums[1], in_channels[1], out_channels[1], strides[1], False)
        self.layer3 = self._make_layer(block, layer_nums[2], in_channels[2], out_channels[2], strides[2], False)
        self.layer4 = self._make_layer(block, layer_nums[3], in_channels[3], out_channels[3], strides[3], False)

        self.mean = torch.mean
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes, use_se=False)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, use_se=False):
        layers = []
        resnet_block = block(in_channel, out_channel, stride=stride, use_se=use_se)
        layers.append(resnet_block)
        # if se_block:
        #     for _ in range(1, layer_num - 1):
        #         resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
        #         layers.append(resnet_block)
        #     resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
        #     layers.append(resnet_block)
        # else:
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
            layers.append(resnet_block)
        return nn.Sequential(*layers)

    def forward(self, x):
        # if self.use_se:
        #     x = self.conv1_0(x)
        #     x = self.bn1_0(x)
        #     x = self.relu(x)
        #     x = self.conv1_1(x)
        #     x = self.bn1_1(x)
        #     x = self.relu(x)
        #     x = self.conv1_2(x)
        # else:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # if self.res_base:
        #     x = self.pad(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, dim=(2, 3), keepdim=True)
        out = self.flatten(out)
        out = self.end_point(out)
        return out
        # return c2


def resnet50(class_num=10):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


def eval_torch(model_old):
    img_size = [224, 224]
    test_set = create_cifar10_dataset("../../datasets/cifar10", image_size=img_size,
                                      batch_size=1, training=False)
    test_iter = test_set.create_tuple_iterator(output_numpy=True)
    model_old.eval()
    correct_old = 0
    test_data_size = 0
    with torch.no_grad():
        for data, target in test_iter:
            d = torch.tensor(data, dtype=torch.float32)
            t = torch.tensor(target, dtype=torch.int64)
            test_data_size += len(d)
            d, t = d.to(device), t.to(device)
            output_old = model_old(d)
            pred_old = output_old.argmax(dim=1, keepdim=True)
            correct_old += pred_old.eq(t.view_as(pred_old)).sum().item()
    accuracy_old = 100. * correct_old / test_data_size
    # evaluation step for new model

    print(f'Accuracy_old: {accuracy_old}%')


def eval_ms(model_old):
    img_size = [224, 224]
    test_set = create_cifar10_dataset("../../datasets/cifar10", image_size=img_size,
                                      batch_size=1, training=False)
    test_iter = test_set.create_tuple_iterator(output_numpy=True)
    model_old.set_train(False)
    test_data_size = 0
    correct_ms = 0
    for data, target in test_iter:
        d = mindspore.Tensor(data, dtype=mindspore.float32)
        t = mindspore.Tensor(target, dtype=mindspore.int64)
        test_data_size += len(d)
        output_old = model_old(d)
        correct_ms += (output_old.argmax(1) == t).asnumpy().sum()
        if test_data_size == 1000:
            break
    correct_ms /= test_data_size
    # evaluation step for new model

    print(f'Accuracy_old: {100 * correct_ms}%')


if __name__ == '__main__':
    # import numpy as np
    #
    # x, y = np.load("")
    net = resnet50(10).to(device)
    a = torch.randn(1, 3, 224, 224)
    print(net(a).shape)
    np_data = [np.random.randn(1, 3, 224, 224)]
    # torch.onnx.export(net, torch.randn(1, 3, 224, 224).to(device), "resnet50.onnx", verbose=True)
    dtypes = [torch.float32]
    torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, device)
    result, global_layer_info = torchinfoplus.summary(
        model=net,
        input_data=torch_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=1)
    # print("result", result)
    input_datas = torchinfoplus.get_input_datas(global_layer_info)
