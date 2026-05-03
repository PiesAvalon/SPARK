from functools import reduce
import mindspore
import numpy as np
from mindspore import nn


def _calculate_in_and_out(arr):
    dim = len(arr.shape)
    if dim < 2:
        raise ValueError("If initialize data with xavier uniform, the dimension of data must greater than 1.")

    n_in = arr.shape[1]
    n_out = arr.shape[0]

    if dim > 2:
        counter = reduce(lambda x, y: x * y, arr.shape[2:])
        n_in *= counter
        n_out *= counter
    return n_in, n_out


def _make_layer(base, args, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            weight = 'ones'
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=0,
                               pad_mode='same',
                               has_bias=False,
                               weight_init=weight)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)


class Vgg19(nn.Cell):
    def __init__(self, base, num_classes=1000, batch_norm=False, batch_size=1, args=None, phase="train",
                 include_top=True):
        super(Vgg19, self).__init__()
        _ = batch_size
        self.layers = _make_layer(base, args, batch_norm=batch_norm)
        self.include_top = include_top
        self.flatten = nn.Flatten()
        self.dropout_ratio = 0.5
        self.num_classes = num_classes
        if phase == "test":
            self.dropout_ratio = 1.0
        self.relu = nn.ReLU()
        self.dense1 = nn.Dense(512 * 7 * 7, 4096)
        self.dense2 = nn.Dense(4096, 4096)
        self.dense3 = nn.Dense(4096, self.num_classes)
        self.dropout = nn.Dropout(p=1 - self.dropout_ratio)

    def construct(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense3(x)
        return x


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


if __name__ == '__main__':
    input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
    inputs = mindspore.Tensor(input_np, mindspore.float32)
    net = vgg19(10)
    # print(net(inputs))
    # mindspore.export(net, mindspore.Tensor(input_np, mindspore.float32), file_name="vgg19.onnx", file_format='ONNX')
