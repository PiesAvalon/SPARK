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
"""
Image classifiation.
"""
from pprint import pformat
import math
import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import numpy as np
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer
from models.vgg19.utils.var_init import default_recurisive_init, KaimingNormal


def _make_layer(base, args, batch_norm):
    args.padding = 0
    args.pad_mode = "same"
    args.has_bias = False
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    args.initialize_mode = "XavierUniform"
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            weight = 'ones'
            if args.initialize_mode == "XavierUniform":
                weight_shape = (v, in_channels, 3, 3)
                weight = initializer('XavierUniform', shape=weight_shape, dtype=mstype.float32)

            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=args.padding,
                               pad_mode=args.pad_mode,
                               has_bias=args.has_bias,
                               weight_init=weight)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)


class Vgg19(nn.Cell):
    """
    VGG network definition.

    Args:
        base (list): Configuration for different layers, mainly the channel number of Conv layer.
        num_classes (int): Class numbers. Default: 1000.
        batch_norm (bool): Whether to do the batchnorm. Default: False.
        batch_size (int): Batch size. Default: 1.
        include_top(bool): Whether to include the 3 fully-connected layers at the top of the network. Default: True.

    Returns:
        Tensor, infer output tensor.
    """

    def __init__(self, base, num_classes=1000, batch_norm=False, batch_size=1, args=None, phase="train",
                 include_top=True):
        super(Vgg19, self).__init__()
        _ = batch_size
        # print("base:", base)
        # print("args:", args)
        self.layers = _make_layer(base, args, batch_norm=batch_norm)
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               padding=args.padding,
                               pad_mode=args.pad_mode,
                               has_bias=args.has_bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=64, kernel_size=3,
                               padding=args.padding,
                               pad_mode=args.pad_mode,
                               has_bias=args.has_bias)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                               padding=args.padding, pad_mode=args.pad_mode,
                               has_bias=args.has_bias)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                               padding=args.padding, pad_mode=args.pad_mode,
                               has_bias=args.has_bias)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                               padding=args.padding, pad_mode=args.pad_mode,
                               has_bias=args.has_bias)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                               padding=args.padding, pad_mode=args.pad_mode,
                               has_bias=args.has_bias)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                               padding=args.padding, pad_mode=args.pad_mode,
                               has_bias=args.has_bias)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                               padding=args.padding, pad_mode=args.pad_mode,
                               has_bias=args.has_bias)
        self.bn8 = nn.BatchNorm2d(256)
        self.relu8 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                               padding=args.padding, pad_mode=args.pad_mode,
                               has_bias=args.has_bias)
        self.bn9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                padding=args.padding, pad_mode=args.pad_mode,
                                has_bias=args.has_bias)
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                padding=args.padding, pad_mode=args.pad_mode,
                                has_bias=args.has_bias)
        self.bn11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                padding=args.padding, pad_mode=args.pad_mode,
                                has_bias=args.has_bias)
        self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                padding=args.padding, pad_mode=args.pad_mode,
                                has_bias=args.has_bias)
        self.bn13 = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                padding=args.padding, pad_mode=args.pad_mode,
                                has_bias=args.has_bias)
        self.bn14 = nn.BatchNorm2d(512)
        self.relu14 = nn.ReLU()
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                padding=args.padding, pad_mode=args.pad_mode,
                                has_bias=args.has_bias)
        self.bn15 = nn.BatchNorm2d(512)
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                padding=args.padding, pad_mode=args.pad_mode,
                                has_bias=args.has_bias)
        self.bn16 = nn.BatchNorm2d(512)
        self.relu16 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.include_top = include_top
        self.flatten = nn.Flatten()
        dropout_ratio = 0.5
        args.has_dropout = False
        self.num_classes = num_classes
        if not args.has_dropout or phase == "test":
            dropout_ratio = 1.0
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=1 - dropout_ratio),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=1 - dropout_ratio),
            nn.Dense(4096, num_classes)])
        if args.initialize_mode == "KaimingNormal":
            default_recurisive_init(self)
            self.custom_init_weight()
        self.relu = nn.ReLU()
        self.dropout_ratio = 0.5
        self.dense1 = nn.Dense(512 * 7 * 7, 4096)
        self.dense2 = nn.Dense(4096, 4096)
        self.dense3 = nn.Dense(4096, self.num_classes)
        self.dropout = nn.Dropout(p=1 - self.dropout_ratio)

    def construct(self, x):
        # x = self.layers(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.maxpool3(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        x = self.maxpool4(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu13(x)
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu14(x)
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.relu15(x)
        x = self.conv16(x)
        x = self.bn16(x)
        x = self.relu16(x)
        x = self.maxpool5(x)

        # if self.include_top:
        x = self.flatten(x)
        # x = self.classifier(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense3(x)
        return x

    def custom_init_weight(self):
        """
        Init the weight of Conv2d and Dense in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(
                    KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(
                    init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))


cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg19(num_classes=1000, args=None, phase="train", **kwargs):
    """
    Get Vgg19 neural network with Batch Normalization.
    Args:
        num_classes (int): Class numbers. Default: 1000.
        args(namespace): param for net init.
        phase(str): train or test mode.

    Returns:
        Cell, cell instance of Vgg19 neural network with Batch Normalization.
    """
    net = Vgg19(cfg['19'], num_classes=num_classes, args=args, batch_norm=True, phase=phase, **kwargs)
    return net


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """

    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    # mindspore.set_context(device_target="GPU", device_id=1, enable_graph_kernel=True, mode=context.GRAPH_MODE)
    input_np = np.load("../../Expired/cifar_224_bs56.npy")
    inputs = mindspore.Tensor(input_np, mindspore.float32)
    net = vgg19(10, args=Config({}))
    # mindspore.export(net, inputs, file_name="../onnx/vgg19.onnx", file_format='ONNX')
    print(net(inputs))
    shapes = [(1, 3, 224, 224)]
    dtypes = [mindspore.float32]
    np_data = [np.ones(val) for val in shapes]
