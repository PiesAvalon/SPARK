import torch
import torch.nn as nn


class vgg11(nn.Module):
    def __init__(self):
        super(vgg11, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.averagepool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.relu1(self.conv1(inputs))
        x = self.maxpool(x)
        x = self.relu1(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu1(self.conv3(x))
        x = self.relu1(self.conv4(x))
        x = self.maxpool(x)
        x = self.relu1(self.conv5(x))
        x = self.relu1(self.conv6(x))
        x = self.maxpool(x)
        x = self.relu1(self.conv7(x))
        x = self.relu1(self.conv8(x))
        x = self.maxpool(x)
        x = self.averagepool(x)
        x = torch.flatten(x, 1)
        x = self.relu1(self.fc1(x))
        x = self.relu1(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = vgg11()
    a = torch.randn(1, 3, 32, 32)
    print(net(a).shape)
    # torch.onnx.export(net, a, "vgg11.onnx", verbose=True, opset_version=11)
