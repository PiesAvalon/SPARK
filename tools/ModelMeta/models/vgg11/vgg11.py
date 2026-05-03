import mindspore
import numpy as np
from mindspore import nn


class vgg11(nn.Cell):
    def __init__(self):
        super(vgg11, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, pad_mode='same')
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, pad_mode='same')
        self.conv3 = nn.Conv2d(128, 256, 3, 1, pad_mode='same')
        self.conv4 = nn.Conv2d(256, 256, 3, 1, pad_mode='same')
        self.conv5 = nn.Conv2d(256, 512, 3, 1, pad_mode='same')
        self.conv6 = nn.Conv2d(512, 512, 3, 1, pad_mode='same')
        self.conv7 = nn.Conv2d(512, 512, 3, 1, pad_mode='same')
        self.conv8 = nn.Conv2d(512, 512, 3, 1, pad_mode='same')
        self.averagepool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.reshape = mindspore.ops.Reshape()
        self.fc1 = nn.Dense(512, 4096)
        self.fc2 = nn.Dense(4096, 4096)
        self.fc3 = nn.Dense(4096, 10)
        self.shape = 512

    def construct(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu1(x)
        x = self.conv4(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.relu1(x)
        x = self.conv6(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv7(x)
        x = self.relu1(x)
        x = self.conv8(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.averagepool(x)
        x_1 = mindspore.ops.flatten(x, start_dim=1)
        x = self.fc1(x_1)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = vgg11()
    # print(net)
    # a = mindspore.Tensor(np.random.randn(1, 3, 32, 32), mindspore.float32)
    # print(net(a)[0].shape)
    mindspore.export(net, mindspore.Tensor(np.random.randn(1, 3, 32, 32), mindspore.float32), file_name='vgg11.onnx', file_format='ONNX')
