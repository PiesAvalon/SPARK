import mindspore
import torch.nn as nn
import numpy as np
import torch.fx
from torch.fx import symbolic_trace


@torch.fx.wrap
def _make_layer(in_channels, base, batch_norm):
    """Make stage network of VGG."""
    layers = []
    for v in base:
        # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Vgg19(nn.Module):
    def __init__(self, base, num_classes=1000, batch_norm=False, include_top=True):
        super(Vgg19, self).__init__()
        self.include_top = include_top
        self.layers = _make_layer(3, base, batch_norm=batch_norm)
        self.flatten = nn.Flatten()
        self.dropout_ratio = 0.5
        self.num_classes = num_classes
        self.relu = nn.ReLU(inplace=True)
        self.dense1 = nn.Linear(512 * 7 * 7, 4096)
        self.dense2 = nn.Linear(4096, 4096)
        self.dense3 = nn.Linear(4096, self.num_classes)
        self.dropout = nn.Dropout(p=1 - self.dropout_ratio)

    def forward(self, x):
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


def vgg19(num_classes=1000, batch_norm=True):
    """
    Get Vgg19 neural network with Batch Normalization.

    Args:
        num_classes (int): Class numbers. Default: 1000.
        batch_norm(bool): Apply Batch Normalization or not.

    Returns:
        Cell, cell instance of Vgg19 neural network with Batch Normalization.
    """
    net = Vgg19(cfg['19'], num_classes=num_classes, batch_norm=batch_norm)
    return net


def param_convert(ms_params, pt_params):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}
    new_params_list = []
    for ms_param in ms_params.keys():
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数
        if "bn" in ms_param or "downsample.1" in ms_param:
            ms_param_item = ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": mindspore.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表
            if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms_param]
                new_params_list.append({"name": ms_param, "data": ms_value})
            else:
                print(ms_param, "not match in pt_params")


class MyCustomTracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        # Check if the current module is '_make_layer'
        # print("module_qualified_name", module_qualified_name)
        if module_qualified_name.endswith('_make_layer'):
            return True
        if module_qualified_name.startswith('layers'):
            return True
        return super().is_leaf_module(m, module_qualified_name)


if __name__ == '__main__':
    input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
    inputs = torch.from_numpy(input_np).float()
    net = vgg19(10)
    symbolic_traced = MyCustomTracer().trace(net)
    traced = torch.fx.GraphModule(net, symbolic_traced)
    for node in symbolic_traced.nodes:
        print("mutable_op_name", node.name, "op_type", node.op)
        # pass