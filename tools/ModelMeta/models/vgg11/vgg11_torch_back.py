import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, inputs):
        x = self.fc(inputs)
        return x


class Vgg11(nn.Module):

    def __init__(self):
        super().__init__()

        # vgg11的卷积通道变化
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        conv_blks = []
        in_channels = 3

        # 卷积部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_blks)

        self.fc = nn.Sequential(
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10))

        self.fn = nn.Flatten()

    def forward(self, x):
        out = self.conv(x)
        out = self.fn(out)
        out = self.fc(out)
        return out


# vgg块：num_convs个卷积层 + 1个最大汇聚层
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


if __name__ == '__main__':
    net = Vgg11()
    a = torch.randn(1, 3, 224, 224)
    print(net(a).shape)
