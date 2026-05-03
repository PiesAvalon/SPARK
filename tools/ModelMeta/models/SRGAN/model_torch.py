import math
import torch
import torch.nn as nn

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    init_type: normal | xavier | kaiming
    init_gain: scaling factor for normal, xavier and kaiming.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

class ResidualBlock(nn.Module):
    """Structure of ResidualBlock"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + x

class SubpixelConvolutionLayer(nn.Module):
    """Structure of SubpixelConvolutionLayer"""
    def __init__(self, channels):
        super(SubpixelConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU(num_parameters=channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out

class Generator(nn.Module):
    """Structure of Generator"""
    def __init__(self, upscale_factor):
        super(Generator, self).__init__()
        # 计算子像素卷积层的数量
        num_subpixel_convolution_layers = int(math.log(upscale_factor, 2))
        # 第一层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, bias=True),
            nn.PReLU(num_parameters=64)
        )

        # 16个残差块
        trunk = []
        for _ in range(16):
            trunk.append(ResidualBlock(64))
        self.trunk = nn.Sequential(*trunk)

        # 残差块后的第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(num_parameters=64)
        )

        # 子像素卷积层
        subpixel_conv_layers = []
        for _ in range(num_subpixel_convolution_layers):
            subpixel_conv_layers.append(SubpixelConvolutionLayer(64))
        self.subpixel_conv = nn.Sequential(*subpixel_conv_layers)

        # 最后的输出层
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        conv1 = self.conv1(x)
        trunk = self.trunk(conv1)
        conv2 = self.conv2(trunk)
        out = conv1 + conv2
        out = self.subpixel_conv(out)
        out = self.conv3(out)
        out = self.tanh(out)
        return out

def get_generator(upscale_factor, init_gain):
    """Return generator network by args."""
    net = Generator(upscale_factor)
    init_weights(net, init_type='normal', init_gain=init_gain)
    return net

import torch
import torch.nn as nn

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    init_type: normal | xavier | kaiming
    init_gain: scaling factor for normal, xavier and kaiming.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
    """Structure of Discriminator"""
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        feature_map_size = int(image_size // 16)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, eps=1e-05),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # state size: (256) x (image_size/8) x (image_size/8)
            nn.BatchNorm2d(256, eps=1e-05),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # state size: (512) x (image_size/16) x (image_size/16)
            nn.BatchNorm2d(512, eps=1e-05),
            nn.LeakyReLU(0.2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512 * feature_map_size * feature_map_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

def get_discriminator(image_size, init_gain):
    """Return discriminator by args."""
    net = Discriminator(image_size)
    init_weights(net, 'normal', init_gain)
    return net

if __name__ == "__main__":

    print(get_generator(4, 0.02))
    print(get_discriminator(4, 0.02))