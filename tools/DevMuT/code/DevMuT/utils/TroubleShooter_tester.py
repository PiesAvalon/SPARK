from common.model_utils import get_model
import troubleshooter as ts
import numpy as np
import mindspore
import mindspore.nn as nn
import sys
import os
import torch
from mindspore.common.initializer import Normal

seed=20230729
# torch.manual_seed(seed)
# mindspore.set_seed(seed)
# np.random.seed(seed)

class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

    """

    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        self.transpose = mindspore.ops.Transpose()
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    # 跟踪construct函数执行
    @ts.tracking()
    def construct(self, x):
        x = self.transpose(x, (0, 3, 1, 2))
        x = self.conv1(x)
        # x = self.relu(x)
        # x = self.max_pool2d(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.max_pool2d(x)
        # if not self.include_top:
        #     return x
        # x = self.flatten(x)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


def test_compare_final_results():
    model_name = "resnet"
    device = "cpu"
    input_size = (1, 3, 224, 224)
    model_ms, model_t = get_model(model_name, device, device_id=0, input_size=input_size)
    diff_finder = ts.migrator.NetDifferenceFinder(pt_net=model_t, ms_net=model_ms)

    diff_finder.fix_random(seed=seed)
    diff_finder.compare(auto_inputs=((input_size, np.float32),))

    # inputs = None
    # auto_inputs = ((input_size, np.float32),)
    # rtol = 1e-4
    # atol = 1e-4
    # equal_nan = False
    #
    # diff_finder._check_auto_input(auto_inputs)
    # diff_finder._handle_weights(diff_finder.pt_params_path, diff_finder.ms_params_path, diff_finder.auto_conv_ckpt)
    # diff_finder._compare_ckpt()
    # if auto_inputs:
    #     inputs = diff_finder._build_auto_input(auto_inputs)
    # diff_finder._check_input(inputs)
    # inputs = diff_finder._convert_data_format(inputs)
    # compare_results = diff_finder._inference(inputs, rtol, atol, equal_nan)

def test_grad_fn():
    model_name = "resnet"
    device = "cpu"
    input_size = (10, 3, 224, 224)
    model_ms, model_t = get_model(model_name, device, device_id=0, input_size=input_size)

    data = mindspore.Tensor(np.random.randn(*input_size), mindspore.float32)

    labels = mindspore.Tensor([3, 5, 4, 2, 8, 1, 5, 7, 7, 6], mindspore.float32)

    optimizer_ms = mindspore.nn.Adam(params=model_ms.trainable_params(), learning_rate=0.0005, momentum=0.9,
                                     weight_decay=0.0001)
    loss_fun_ms = mindspore.nn.CrossEntropyLoss()

    def forward_fn(data, label):
        # logits = model_ms(data)
        logits = mindspore.Tensor([2, 1, 5, 2, 8, 3, 5, 4, 9, 5], mindspore.float32)
        loss = loss_fun_ms(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer_ms.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, optimizer_ms(grads))
        return loss

    result = train_step(data, labels)

if __name__ == '__main__':
    model_name = "resnet"
    device = "cpu"
    input_size = (1, 3, 224, 224)

    ts.widget.fix_random(seed)
    model_ms, model_t = get_model(model_name, device, device_id=0, input_size=input_size)
    diff_finder = ts.migrator.NetDifferenceFinder(pt_net=model_t, ms_net=model_ms,fix_seed=seed)#

    diff_finder.compare(auto_inputs=((input_size, np.float32),))






