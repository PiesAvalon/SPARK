import numpy
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels, out_channels, use_bn=True, kernel_size=3, stride=1, padding=1, activation='relu'):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(getattr(nn, activation)())
    return nn.Sequential(*layers)


class UnetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, num_layers=2, kernel_size=3, stride=1, padding=1):
        super(UnetConv2d, self).__init__()
        convs = []
        for _ in range(num_layers):
            convs.append(conv_bn_relu(in_channels, out_channels, use_bn, kernel_size, stride, padding, "ReLU"))
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv, n_concat=2):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2d(in_channels + (n_concat - 2) * out_channels, out_channels, False)
        self.use_deconv = use_deconv
        if use_deconv:
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        else:
            self.up_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, high_feature, *low_features):
        if self.use_deconv:
            output = self.up_conv(high_feature)
        else:
            output = nn.functional.interpolate(high_feature, scale_factor=2, mode='bilinear', align_corners=False)
            output = self.up_conv(output)
        for feature in low_features:
            try:
                output = torch.cat((output, feature), dim=1)
            except RuntimeError as e:
                print(e)
                print("================================")
                print(output.shape)
                print(feature.shape)
                print("================================")
        return self.conv(output)


class NestedUNet(nn.Module):
    """
    Nested unet
    """

    def __init__(self, in_channels, n_classes=2, feature_scale=2, use_deconv=True, use_bn=True, use_ds=True):
        super(NestedUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.feature_scale = feature_scale
        self.use_deconv = use_deconv
        self.use_bn = use_bn
        self.use_ds = use_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Down Sample
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv00 = UnetConv2d(self.in_channels, filters[0], self.use_bn)
        self.conv10 = UnetConv2d(filters[0], filters[1], self.use_bn)
        self.conv20 = UnetConv2d(filters[1], filters[2], self.use_bn)
        self.conv30 = UnetConv2d(filters[2], filters[3], self.use_bn)
        self.conv40 = UnetConv2d(filters[3], filters[4], self.use_bn)

        # Up Sample
        self.up_concat01 = UnetUp(filters[1], filters[0], self.use_deconv, 2)
        self.up_concat11 = UnetUp(filters[2], filters[1], self.use_deconv, 2)
        self.up_concat21 = UnetUp(filters[3], filters[2], self.use_deconv, 2)
        self.up_concat31 = UnetUp(filters[4], filters[3], self.use_deconv, 2)

        self.up_concat02 = UnetUp(filters[1], filters[0], self.use_deconv, 3)
        self.up_concat12 = UnetUp(filters[2], filters[1], self.use_deconv, 3)
        self.up_concat22 = UnetUp(filters[3], filters[2], self.use_deconv, 3)

        self.up_concat03 = UnetUp(filters[1], filters[0], self.use_deconv, 4)
        self.up_concat13 = UnetUp(filters[2], filters[1], self.use_deconv, 4)

        self.up_concat04 = UnetUp(filters[1], filters[0], self.use_deconv, 5)

        # Finale Convolution
        self.final1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final4 = nn.Conv2d(filters[0], n_classes, 1)

        self.out_shapes = {
            'INPUT': [-1, 1, 96, 96],
            'conv00.convs.0.0': [-1, 32, 96, 96], 'conv00.convs.0.1': [-1, 32, 96, 96],
            'conv00.convs.0.2': [-1, 32, 96, 96], 'conv00.convs.1.0': [-1, 32, 96, 96],
            'conv00.convs.1.1': [-1, 32, 96, 96], 'conv00.convs.1.2': [-1, 32, 96, 96],
            'maxpool0': [-1, 32, 48, 48], 'conv10.convs.0.0': [-1, 64, 48, 48],
            'conv10.convs.0.1': [-1, 64, 48, 48], 'conv10.convs.0.2': [-1, 64, 48, 48],
            'conv10.convs.1.0': [-1, 64, 48, 48], 'conv10.convs.1.1': [-1, 64, 48, 48],
            'conv10.convs.1.2': [-1, 64, 48, 48], 'maxpool1': [-1, 64, 24, 24],
            'conv20.convs.0.0': [-1, 128, 24, 24],
            'conv20.convs.0.1': [-1, 128, 24, 24], 'conv20.convs.0.2': [-1, 128, 24, 24],
            'conv20.convs.1.0': [-1, 128, 24, 24], 'conv20.convs.1.1': [-1, 128, 24, 24],
            'conv20.convs.1.2': [-1, 128, 24, 24], 'maxpool2': [-1, 128, 12, 12],
            'conv30.convs.0.0': [-1, 256, 12, 12],
            'conv30.convs.0.1': [-1, 256, 12, 12], 'conv30.convs.0.2': [-1, 256, 12, 12],
            'conv30.convs.1.0': [-1, 256, 12, 12], 'conv30.convs.1.1': [-1, 256, 12, 12],
            'conv30.convs.1.2': [-1, 256, 12, 12], 'maxpool3': [-1, 256, 6, 6],
            'conv40.convs.0.0': [-1, 512, 6, 6],
            'conv40.convs.0.1': [-1, 512, 6, 6], 'conv40.convs.0.2': [-1, 512, 6, 6],
            'conv40.convs.1.0': [-1, 512, 6, 6], 'conv40.convs.1.1': [-1, 512, 6, 6],
            'conv40.convs.1.2': [-1, 512, 6, 6], 'up_concat01.up_conv': [-1, 32, 96, 96],
            'up_concat01.conv.convs.0.0': [-1, 32, 96, 96],
            'up_concat01.conv.convs.0.1': [-1, 32, 96, 96],
            'up_concat01.conv.convs.1.0': [-1, 32, 96, 96],
            'up_concat01.conv.convs.1.1': [-1, 32, 96, 96], 'up_concat11.up_conv': [-1, 64, 48, 48],
            'up_concat11.conv.convs.0.0': [-1, 64, 48, 48],
            'up_concat11.conv.convs.0.1': [-1, 64, 48, 48],
            'up_concat11.conv.convs.1.0': [-1, 64, 48, 48],
            'up_concat11.conv.convs.1.1': [-1, 64, 48, 48], 'up_concat21.up_conv': [-1, 128, 24, 24],
            'up_concat21.conv.convs.0.0': [-1, 128, 24, 24],
            'up_concat21.conv.convs.0.1': [-1, 128, 24, 24],
            'up_concat21.conv.convs.1.0': [-1, 128, 24, 24],
            'up_concat21.conv.convs.1.1': [-1, 128, 24, 24], 'up_concat31.up_conv': [-1, 256, 12, 12],
            'up_concat31.conv.convs.0.0': [-1, 256, 12, 12],
            'up_concat31.conv.convs.0.1': [-1, 256, 12, 12],
            'up_concat31.conv.convs.1.0': [-1, 256, 12, 12],
            'up_concat31.conv.convs.1.1': [-1, 256, 12, 12], 'up_concat02.up_conv': [-1, 32, 96, 96],
            'up_concat02.conv.convs.0.0': [-1, 32, 96, 96],
            'up_concat02.conv.convs.0.1': [-1, 32, 96, 96],
            'up_concat02.conv.convs.1.0': [-1, 32, 96, 96],
            'up_concat02.conv.convs.1.1': [-1, 32, 96, 96], 'up_concat12.up_conv': [-1, 64, 48, 48],
            'up_concat12.conv.convs.0.0': [-1, 64, 48, 48],
            'up_concat12.conv.convs.0.1': [-1, 64, 48, 48],
            'up_concat12.conv.convs.1.0': [-1, 64, 48, 48],
            'up_concat12.conv.convs.1.1': [-1, 64, 48, 48], 'up_concat22.up_conv': [-1, 128, 24, 24],
            'up_concat22.conv.convs.0.0': [-1, 128, 24, 24],
            'up_concat22.conv.convs.0.1': [-1, 128, 24, 24],
            'up_concat22.conv.convs.1.0': [-1, 128, 24, 24],
            'up_concat22.conv.convs.1.1': [-1, 128, 24, 24], 'up_concat03.up_conv': [-1, 32, 96, 96],
            'up_concat03.conv.convs.0.0': [-1, 32, 96, 96],
            'up_concat03.conv.convs.0.1': [-1, 32, 96, 96],
            'up_concat03.conv.convs.1.0': [-1, 32, 96, 96],
            'up_concat03.conv.convs.1.1': [-1, 32, 96, 96], 'up_concat13.up_conv': [-1, 64, 48, 48],
            'up_concat13.conv.convs.0.0': [-1, 64, 48, 48],
            'up_concat13.conv.convs.0.1': [-1, 64, 48, 48],
            'up_concat13.conv.convs.1.0': [-1, 64, 48, 48],
            'up_concat13.conv.convs.1.1': [-1, 64, 48, 48], 'up_concat04.up_conv': [-1, 32, 96, 96],
            'up_concat04.conv.convs.0.0': [-1, 32, 96, 96],
            'up_concat04.conv.convs.0.1': [-1, 32, 96, 96],
            'up_concat04.conv.convs.1.0': [-1, 32, 96, 96],
            'up_concat04.conv.convs.1.1': [-1, 32, 96, 96], 'final1': [-1, 2, 96, 96],
            'final2': [-1, 2, 96, 96], 'final3': [-1, 2, 96, 96], 'final4': [-1, 2, 96, 96],
            'OUTPUT': [4, 2, 2, 96, 96]}
        self.in_shapes = {
            'INPUT': [-1, 1, 96, 96],
            'conv00.convs.0.0': [-1, 1, 96, 96], 'conv00.convs.0.1': [-1, 32, 96, 96],
            'conv00.convs.0.2': [-1, 32, 96, 96], 'conv00.convs.1.0': [-1, 32, 96, 96],
            'conv00.convs.1.1': [-1, 32, 96, 96], 'conv00.convs.1.2': [-1, 32, 96, 96],
            'maxpool0': [-1, 32, 96, 96], 'conv10.convs.0.0': [-1, 32, 48, 48],
            'conv10.convs.0.1': [-1, 64, 48, 48], 'conv10.convs.0.2': [-1, 64, 48, 48],
            'conv10.convs.1.0': [-1, 64, 48, 48], 'conv10.convs.1.1': [-1, 64, 48, 48],
            'conv10.convs.1.2': [-1, 64, 48, 48], 'maxpool1': [-1, 64, 48, 48],
            'conv20.convs.0.0': [-1, 64, 24, 24],
            'conv20.convs.0.1': [-1, 128, 24, 24], 'conv20.convs.0.2': [-1, 128, 24, 24],
            'conv20.convs.1.0': [-1, 128, 24, 24], 'conv20.convs.1.1': [-1, 128, 24, 24],
            'conv20.convs.1.2': [-1, 128, 24, 24], 'maxpool2': [-1, 128, 24, 24],
            'conv30.convs.0.0': [-1, 128, 12, 12],
            'conv30.convs.0.1': [-1, 256, 12, 12], 'conv30.convs.0.2': [-1, 256, 12, 12],
            'conv30.convs.1.0': [-1, 256, 12, 12], 'conv30.convs.1.1': [-1, 256, 12, 12],
            'conv30.convs.1.2': [-1, 256, 12, 12], 'maxpool3': [-1, 256, 12, 12],
            'conv40.convs.0.0': [-1, 256, 6, 6],
            'conv40.convs.0.1': [-1, 512, 6, 6], 'conv40.convs.0.2': [-1, 512, 6, 6],
            'conv40.convs.1.0': [-1, 512, 6, 6], 'conv40.convs.1.1': [-1, 512, 6, 6],
            'conv40.convs.1.2': [-1, 512, 6, 6], 'up_concat01.up_conv': [-1, 64, 48, 48],
            'up_concat01.conv.convs.0.0': [-1, 64, 96, 96],
            'up_concat01.conv.convs.0.1': [-1, 32, 96, 96],
            'up_concat01.conv.convs.1.0': [-1, 32, 96, 96],
            'up_concat01.conv.convs.1.1': [-1, 32, 96, 96], 'up_concat11.up_conv': [-1, 128, 24, 24],
            'up_concat11.conv.convs.0.0': [-1, 128, 48, 48],
            'up_concat11.conv.convs.0.1': [-1, 64, 48, 48],
            'up_concat11.conv.convs.1.0': [-1, 64, 48, 48],
            'up_concat11.conv.convs.1.1': [-1, 64, 48, 48], 'up_concat21.up_conv': [-1, 256, 12, 12],
            'up_concat21.conv.convs.0.0': [-1, 256, 24, 24],
            'up_concat21.conv.convs.0.1': [-1, 128, 24, 24],
            'up_concat21.conv.convs.1.0': [-1, 128, 24, 24],
            'up_concat21.conv.convs.1.1': [-1, 128, 24, 24], 'up_concat31.up_conv': [-1, 512, 6, 6],
            'up_concat31.conv.convs.0.0': [-1, 512, 12, 12],
            'up_concat31.conv.convs.0.1': [-1, 256, 12, 12],
            'up_concat31.conv.convs.1.0': [-1, 256, 12, 12],
            'up_concat31.conv.convs.1.1': [-1, 256, 12, 12], 'up_concat02.up_conv': [-1, 64, 48, 48],
            'up_concat02.conv.convs.0.0': [-1, 96, 96, 96],
            'up_concat02.conv.convs.0.1': [-1, 32, 96, 96],
            'up_concat02.conv.convs.1.0': [-1, 32, 96, 96],
            'up_concat02.conv.convs.1.1': [-1, 32, 96, 96], 'up_concat12.up_conv': [-1, 128, 24, 24],
            'up_concat12.conv.convs.0.0': [-1, 192, 48, 48],
            'up_concat12.conv.convs.0.1': [-1, 64, 48, 48],
            'up_concat12.conv.convs.1.0': [-1, 64, 48, 48],
            'up_concat12.conv.convs.1.1': [-1, 64, 48, 48], 'up_concat22.up_conv': [-1, 256, 12, 12],
            'up_concat22.conv.convs.0.0': [-1, 384, 24, 24],
            'up_concat22.conv.convs.0.1': [-1, 128, 24, 24],
            'up_concat22.conv.convs.1.0': [-1, 128, 24, 24],
            'up_concat22.conv.convs.1.1': [-1, 128, 24, 24], 'up_concat03.up_conv': [-1, 64, 48, 48],
            'up_concat03.conv.convs.0.0': [-1, 128, 96, 96],
            'up_concat03.conv.convs.0.1': [-1, 32, 96, 96],
            'up_concat03.conv.convs.1.0': [-1, 32, 96, 96],
            'up_concat03.conv.convs.1.1': [-1, 32, 96, 96], 'up_concat13.up_conv': [-1, 128, 24, 24],
            'up_concat13.conv.convs.0.0': [-1, 256, 48, 48],
            'up_concat13.conv.convs.0.1': [-1, 64, 48, 48],
            'up_concat13.conv.convs.1.0': [-1, 64, 48, 48],
            'up_concat13.conv.convs.1.1': [-1, 64, 48, 48], 'up_concat04.up_conv': [-1, 64, 48, 48],
            'up_concat04.conv.convs.0.0': [-1, 160, 96, 96],
            'up_concat04.conv.convs.0.1': [-1, 32, 96, 96],
            'up_concat04.conv.convs.1.0': [-1, 32, 96, 96],
            'up_concat04.conv.convs.1.1': [-1, 32, 96, 96], 'final1': [-1, 32, 96, 96],
            'final2': [-1, 32, 96, 96], 'final3': [-1, 32, 96, 96], 'final4': [-1, 32, 96, 96],
            'OUTPUT': [4, 2, 2, 96, 96]}

        self.orders = {
            'conv00.convs.0.0': ['INPUT', 'conv00.convs.0.1'],
            'conv00.convs.0.1': ['conv00.convs.0.0', 'conv00.convs.0.2'],
            'conv00.convs.0.2': ['conv00.convs.0.1', 'conv00.convs.1.0'],
            'conv00.convs.1.0': ['conv00.convs.0.2', 'conv00.convs.1.1'],
            'conv00.convs.1.1': ['conv00.convs.1.0', 'conv00.convs.1.2'],
            'conv00.convs.1.2': ['conv00.convs.1.1',
                                 ['maxpool0', 'up_concat01.conv.convs.0.0', 'up_concat02.conv.convs.0.0',
                                  'up_concat03.conv.convs.0.0', 'up_concat04.conv.convs.0.0']],

            'maxpool0': ['conv00.convs.1.2', 'conv10.convs.0.0'],
            'conv10.convs.0.0': ['maxpool0', 'conv10.convs.0.1'],
            'conv10.convs.0.1': ['conv10.convs.0.0', 'conv10.convs.0.2'],
            'conv10.convs.0.2': ['conv10.convs.0.1', 'conv10.convs.1.0'],
            'conv10.convs.1.0': ['conv10.convs.0.2', 'conv10.convs.1.1'],
            'conv10.convs.1.1': ['conv10.convs.1.0', 'conv10.convs.1.2'],
            'conv10.convs.1.2': ['conv10.convs.1.1', ['maxpool1', 'up_concat11.conv.convs.0.0', 'up_concat01.up_conv',
                                                      'up_concat12.conv.convs.0.0', 'up_concat13.conv.convs.0.0']],
            'maxpool1': ['conv10.convs.1.2', 'conv20.convs.0.0'],

            'conv20.convs.0.0': ['maxpool1', 'conv20.convs.0.1'],
            'conv20.convs.0.1': ['conv20.convs.0.0', 'conv20.convs.0.2'],
            'conv20.convs.0.2': ['conv20.convs.0.1', 'conv20.convs.1.0'],
            'conv20.convs.1.0': ['conv20.convs.0.2', 'conv20.convs.1.1'],
            'conv20.convs.1.1': ['conv20.convs.1.0', 'conv20.convs.1.2'],
            'conv20.convs.1.2': ['conv20.convs.1.1', ['maxpool2', 'up_concat21.conv.convs.0.0', 'up_concat11.up_conv',
                                                      'up_concat22.conv.convs.0.0']],

            'maxpool2': ['conv20.convs.1.2', 'conv30.convs.0.0'],
            'conv30.convs.0.0': ['maxpool2', 'conv30.convs.0.1'],
            'conv30.convs.0.1': ['conv30.convs.0.0', 'conv30.convs.0.2'],
            'conv30.convs.0.2': ['conv30.convs.0.1', 'conv30.convs.1.0'],
            'conv30.convs.1.0': ['conv30.convs.0.2', 'conv30.convs.1.1'],
            'conv30.convs.1.1': ['conv30.convs.1.0', 'conv30.convs.1.2'],
            'conv30.convs.1.2': ['conv30.convs.1.1', ['maxpool3', 'up_concat31.conv.convs.0.0', 'up_concat21.up_conv']],

            'maxpool3': ['conv30.convs.1.2', 'conv40.convs.0.0'],
            'conv40.convs.0.0': ['maxpool3', 'conv40.convs.0.1'],
            'conv40.convs.0.1': ['conv40.convs.0.0', 'conv40.convs.0.2'],
            'conv40.convs.0.2': ['conv40.convs.0.1', 'conv40.convs.1.0'],
            'conv40.convs.1.0': ['conv40.convs.0.2', 'conv40.convs.1.1'],
            'conv40.convs.1.1': ['conv40.convs.1.0', 'conv40.convs.1.2'],
            'conv40.convs.1.2': ['conv40.convs.1.1', 'up_concat31.up_conv'],

            'up_concat01.up_conv': ['conv10.convs.1.2', 'up_concat01.conv.convs.0.0'],
            'up_concat01.conv.convs.0.0': [['conv00.convs.1.2', 'up_concat01.up_conv'], 'up_concat01.conv.convs.0.1'],
            'up_concat01.conv.convs.0.1': ['up_concat01.conv.convs.0.0', 'up_concat01.conv.convs.1.0'],
            'up_concat01.conv.convs.1.0': ['up_concat01.conv.convs.0.1', 'up_concat01.conv.convs.1.1'],
            'up_concat01.conv.convs.1.1': ['up_concat01.conv.convs.1.0',
                                           ['up_concat02.conv.convs.0.0', 'up_concat03.conv.convs.0.0',
                                            'up_concat04.conv.convs.0.0', 'final1']],

            'up_concat11.up_conv': ['conv20.convs.1.2', 'up_concat11.conv.convs.0.0'],
            'up_concat11.conv.convs.0.0': [['conv10.convs.1.2', 'up_concat11.up_conv'], 'up_concat11.conv.convs.0.1'],
            'up_concat11.conv.convs.0.1': ['up_concat11.conv.convs.0.0', 'up_concat11.conv.convs.1.0'],
            'up_concat11.conv.convs.1.0': ['up_concat11.conv.convs.0.1', 'up_concat11.conv.convs.1.1'],
            'up_concat11.conv.convs.1.1': ['up_concat11.conv.convs.1.0',
                                           ['up_concat02.up_conv', 'up_concat12.conv.convs.0.0',
                                            'up_concat13.conv.convs.0.0']],

            'up_concat21.up_conv': ['conv30.convs.1.2', 'up_concat21.conv.convs.0.0'],
            'up_concat21.conv.convs.0.0': [['conv20.convs.1.2', 'up_concat21.up_conv'], 'up_concat21.conv.convs.0.1'],
            'up_concat21.conv.convs.0.1': ['up_concat21.conv.convs.0.0', 'up_concat21.conv.convs.1.0'],
            'up_concat21.conv.convs.1.0': ['up_concat21.conv.convs.0.1', 'up_concat21.conv.convs.1.1'],
            'up_concat21.conv.convs.1.1': ['up_concat21.conv.convs.1.0',
                                           ['up_concat12.up_conv', 'up_concat22.conv.convs.0.0']],

            'up_concat31.up_conv': ['conv40.convs.1.2', 'up_concat31.conv.convs.0.0'],
            'up_concat31.conv.convs.0.0': [['up_concat31.up_conv', 'conv30.convs.1.2'], 'up_concat31.conv.convs.0.1'],
            'up_concat31.conv.convs.0.1': ['up_concat31.conv.convs.0.0', 'up_concat31.conv.convs.1.0'],
            'up_concat31.conv.convs.1.0': ['up_concat31.conv.convs.0.1', 'up_concat31.conv.convs.1.1'],
            'up_concat31.conv.convs.1.1': ['up_concat31.conv.convs.1.0', 'up_concat22.up_conv'],

            'up_concat02.up_conv': ['up_concat11.conv.convs.1.1', 'up_concat02.conv.convs.0.0'],
            'up_concat02.conv.convs.0.0': [['up_concat02.up_conv', 'conv00.convs.1.2', 'up_concat01.conv.convs.1.1'],
                                           'up_concat02.conv.convs.0.1'],
            'up_concat02.conv.convs.0.1': ['up_concat02.conv.convs.0.0', 'up_concat02.conv.convs.1.0'],
            'up_concat02.conv.convs.1.0': ['up_concat02.conv.convs.0.1', 'up_concat02.conv.convs.1.1'],
            'up_concat02.conv.convs.1.1': ['up_concat02.conv.convs.1.0',
                                           ['up_concat03.conv.convs.0.0', 'up_concat04.conv.convs.0.0', 'final2']],

            'up_concat12.up_conv': ['up_concat21.conv.convs.1.1', 'up_concat12.conv.convs.0.0'],
            'up_concat12.conv.convs.0.0': [['up_concat12.up_conv', 'conv10.convs.1.2', 'up_concat11.conv.convs.1.1'],
                                           'up_concat12.conv.convs.0.1'],
            'up_concat12.conv.convs.0.1': ['up_concat12.conv.convs.0.0', 'up_concat12.conv.convs.1.0'],
            'up_concat12.conv.convs.1.0': ['up_concat12.conv.convs.0.1', 'up_concat12.conv.convs.1.1'],
            'up_concat12.conv.convs.1.1': ['up_concat12.conv.convs.1.0',
                                           ['up_concat03.up_conv', 'up_concat13.conv.convs.0.0']],

            'up_concat22.up_conv': ['up_concat31.conv.convs.1.1', 'up_concat22.conv.convs.0.0'],
            'up_concat22.conv.convs.0.0': [['up_concat22.up_conv', 'conv20.convs.1.2', 'up_concat21.conv.convs.1.1'],
                                           'up_concat22.conv.convs.0.1'],
            'up_concat22.conv.convs.0.1': ['up_concat22.conv.convs.0.0', 'up_concat22.conv.convs.1.0'],
            'up_concat22.conv.convs.1.0': ['up_concat22.conv.convs.0.1', 'up_concat22.conv.convs.1.1'],
            'up_concat22.conv.convs.1.1': ['up_concat22.conv.convs.1.0', 'up_concat13.up_conv'],

            'up_concat03.up_conv': ['up_concat12.conv.convs.1.1', 'up_concat03.conv.convs.0.0'],
            'up_concat03.conv.convs.0.0': [
                ['up_concat03.up_conv', 'conv00.convs.1.2', 'up_concat01.conv.convs.1.1', 'up_concat02.conv.convs.1.1'],
                'up_concat03.conv.convs.0.1'],
            'up_concat03.conv.convs.0.1': ['up_concat03.conv.convs.0.0', 'up_concat03.conv.convs.1.0'],
            'up_concat03.conv.convs.1.0': ['up_concat03.conv.convs.0.1', 'up_concat03.conv.convs.1.1'],
            'up_concat03.conv.convs.1.1': ['up_concat03.conv.convs.1.0', ['up_concat04.conv.convs.0.0', 'final3']],

            'up_concat13.up_conv': ['up_concat22.conv.convs.1.1', 'up_concat13.conv.convs.0.0'],
            'up_concat13.conv.convs.0.0': [
                ['up_concat13.up_conv', 'conv10.convs.1.2', 'up_concat11.conv.convs.1.1', 'up_concat12.conv.convs.1.1'],
                'up_concat13.conv.convs.0.1'],
            'up_concat13.conv.convs.0.1': ['up_concat13.conv.convs.0.0', 'up_concat13.conv.convs.1.0'],
            'up_concat13.conv.convs.1.0': ['up_concat13.conv.convs.0.1', 'up_concat13.conv.convs.1.1'],
            'up_concat13.conv.convs.1.1': ['up_concat13.conv.convs.1.0', 'up_concat04.up_conv'],

            'up_concat04.up_conv': ['up_concat13.conv.convs.1.1', 'up_concat04.conv.convs.0.0'],
            'up_concat04.conv.convs.0.0': [
                ['up_concat04.up_conv', 'conv00.convs.1.2', 'up_concat01.conv.convs.1.1', 'up_concat02.conv.convs.1.1',
                 'up_concat03.conv.convs.1.1'], 'up_concat04.conv.convs.0.1'],
            'up_concat04.conv.convs.0.1': ['up_concat04.conv.convs.0.0', 'up_concat04.conv.convs.1.0'],
            'up_concat04.conv.convs.1.0': ['up_concat04.conv.convs.0.1', 'up_concat04.conv.convs.1.1'],
            'up_concat04.conv.convs.1.1': ['up_concat04.conv.convs.1.0', 'final4'],

            'final1': ['up_concat01.conv.convs.1.1', 'OUTPUT'],
            'final2': ['up_concat02.conv.convs.1.1', 'OUTPUT'],
            'final3': ['up_concat03.conv.convs.1.1', 'OUTPUT'],
            'final4': ['up_concat04.conv.convs.1.1', 'OUTPUT'],
        }

        self.layer_names = {
            "maxpool1": self.maxpool1,
            "maxpool2": self.maxpool2,
            "maxpool3": self.maxpool3,
            "maxpool0": self.maxpool0,
            "conv00": self.conv00,
            "conv00.convs": self.conv00.convs,
            "conv00.convs.0": self.conv00.convs[0],
            "conv00.convs.0.0": self.conv00.convs[0][0],
            "conv00.convs.0.1": self.conv00.convs[0][1],
            "conv00.convs.0.2": self.conv00.convs[0][2],
            "conv00.convs.1": self.conv00.convs[1],
            "conv00.convs.1.0": self.conv00.convs[1][0],
            "conv00.convs.1.1": self.conv00.convs[1][1],
            "conv00.convs.1.2": self.conv00.convs[1][2],
            "conv10": self.conv10,
            "conv10.convs": self.conv10.convs,
            "conv10.convs.0": self.conv10.convs[0],
            "conv10.convs.0.0": self.conv10.convs[0][0],
            "conv10.convs.0.1": self.conv10.convs[0][1],
            "conv10.convs.0.2": self.conv10.convs[0][2],
            "conv10.convs.1": self.conv10.convs[1],
            "conv10.convs.1.0": self.conv10.convs[1][0],
            "conv10.convs.1.1": self.conv10.convs[1][1],
            "conv10.convs.1.2": self.conv10.convs[1][2],
            "conv20": self.conv20,
            "conv20.convs": self.conv20.convs,
            "conv20.convs.0": self.conv20.convs[0],
            "conv20.convs.0.0": self.conv20.convs[0][0],
            "conv20.convs.0.1": self.conv20.convs[0][1],
            "conv20.convs.0.2": self.conv20.convs[0][2],
            "conv20.convs.1": self.conv20.convs[1],
            "conv20.convs.1.0": self.conv20.convs[1][0],
            "conv20.convs.1.1": self.conv20.convs[1][1],
            "conv20.convs.1.2": self.conv20.convs[1][2],
            "conv30": self.conv30,
            "conv30.convs": self.conv30.convs,
            "conv30.convs.0": self.conv30.convs[0],
            "conv30.convs.0.0": self.conv30.convs[0][0],
            "conv30.convs.0.1": self.conv30.convs[0][1],
            "conv30.convs.0.2": self.conv30.convs[0][2],
            "conv30.convs.1": self.conv30.convs[1],
            "conv30.convs.1.0": self.conv30.convs[1][0],
            "conv30.convs.1.1": self.conv30.convs[1][1],
            "conv30.convs.1.2": self.conv30.convs[1][2],
            "conv40": self.conv40,
            "conv40.convs": self.conv40.convs,
            "conv40.convs.0": self.conv40.convs[0],
            "conv40.convs.0.0": self.conv40.convs[0][0],
            "conv40.convs.0.1": self.conv40.convs[0][1],
            "conv40.convs.0.2": self.conv40.convs[0][2],
            "conv40.convs.1": self.conv40.convs[1],
            "conv40.convs.1.0": self.conv40.convs[1][0],
            "conv40.convs.1.1": self.conv40.convs[1][1],
            "conv40.convs.1.2": self.conv40.convs[1][2],
            "up_concat01": self.up_concat01,
            "up_concat01.conv": self.up_concat01.conv,
            "up_concat01.conv.convs": self.up_concat01.conv.convs,
            "up_concat01.conv.convs.0": self.up_concat01.conv.convs[0],
            "up_concat01.conv.convs.0.0": self.up_concat01.conv.convs[0][0],
            "up_concat01.conv.convs.0.1": self.up_concat01.conv.convs[0][1],
            "up_concat01.conv.convs.1": self.up_concat01.conv.convs[1],
            "up_concat01.conv.convs.1.0": self.up_concat01.conv.convs[1][0],
            "up_concat01.conv.convs.1.1": self.up_concat01.conv.convs[1][1],
            "up_concat01.up_conv": self.up_concat01.up_conv,
            "up_concat11": self.up_concat11,
            "up_concat11.conv": self.up_concat11.conv,
            "up_concat11.conv.convs": self.up_concat11.conv.convs,
            "up_concat11.conv.convs.0": self.up_concat11.conv.convs[0],
            "up_concat11.conv.convs.0.0": self.up_concat11.conv.convs[0][0],
            "up_concat11.conv.convs.0.1": self.up_concat11.conv.convs[0][1],
            "up_concat11.conv.convs.1": self.up_concat11.conv.convs[1],
            "up_concat11.conv.convs.1.0": self.up_concat11.conv.convs[1][0],
            "up_concat11.conv.convs.1.1": self.up_concat11.conv.convs[1][1],
            "up_concat11.up_conv": self.up_concat11.up_conv,
            "up_concat21": self.up_concat21,
            "up_concat21.conv": self.up_concat21.conv,
            "up_concat21.conv.convs": self.up_concat21.conv.convs,
            "up_concat21.conv.convs.0": self.up_concat21.conv.convs[0],
            "up_concat21.conv.convs.0.0": self.up_concat21.conv.convs[0][0],
            "up_concat21.conv.convs.0.1": self.up_concat21.conv.convs[0][1],
            "up_concat21.conv.convs.1": self.up_concat21.conv.convs[1],
            "up_concat21.conv.convs.1.0": self.up_concat21.conv.convs[1][0],
            "up_concat21.conv.convs.1.1": self.up_concat21.conv.convs[1][1],
            "up_concat21.up_conv": self.up_concat21.up_conv,
            "up_concat31": self.up_concat31,
            "up_concat31.conv": self.up_concat31.conv,
            "up_concat31.conv.convs": self.up_concat31.conv.convs,
            "up_concat31.conv.convs.0": self.up_concat31.conv.convs[0],
            "up_concat31.conv.convs.0.0": self.up_concat31.conv.convs[0][0],
            "up_concat31.conv.convs.0.1": self.up_concat31.conv.convs[0][1],
            "up_concat31.conv.convs.1": self.up_concat31.conv.convs[1],
            "up_concat31.conv.convs.1.0": self.up_concat31.conv.convs[1][0],
            "up_concat31.conv.convs.1.1": self.up_concat31.conv.convs[1][1],
            "up_concat31.up_conv": self.up_concat31.up_conv,
            "up_concat02": self.up_concat02,
            "up_concat02.conv": self.up_concat02.conv,
            "up_concat02.conv.convs": self.up_concat02.conv.convs,
            "up_concat02.conv.convs.0": self.up_concat02.conv.convs[0],
            "up_concat02.conv.convs.0.0": self.up_concat02.conv.convs[0][0],
            "up_concat02.conv.convs.0.1": self.up_concat02.conv.convs[0][1],
            "up_concat02.conv.convs.1": self.up_concat02.conv.convs[1],
            "up_concat02.conv.convs.1.0": self.up_concat02.conv.convs[1][0],
            "up_concat02.conv.convs.1.1": self.up_concat02.conv.convs[1][1],
            "up_concat02.up_conv": self.up_concat02.up_conv,
            "up_concat12": self.up_concat12,
            "up_concat12.conv": self.up_concat12.conv,
            "up_concat12.conv.convs": self.up_concat12.conv.convs,
            "up_concat12.conv.convs.0": self.up_concat12.conv.convs[0],
            "up_concat12.conv.convs.0.0": self.up_concat12.conv.convs[0][0],
            "up_concat12.conv.convs.0.1": self.up_concat12.conv.convs[0][1],
            "up_concat12.conv.convs.1": self.up_concat12.conv.convs[1],
            "up_concat12.conv.convs.1.0": self.up_concat12.conv.convs[1][0],
            "up_concat12.conv.convs.1.1": self.up_concat12.conv.convs[1][1],
            "up_concat12.up_conv": self.up_concat12.up_conv,
            "up_concat22": self.up_concat22,
            "up_concat22.conv": self.up_concat22.conv,
            "up_concat22.conv.convs": self.up_concat22.conv.convs,
            "up_concat22.conv.convs.0": self.up_concat22.conv.convs[0],
            "up_concat22.conv.convs.0.0": self.up_concat22.conv.convs[0][0],
            "up_concat22.conv.convs.0.1": self.up_concat22.conv.convs[0][1],
            "up_concat22.conv.convs.1": self.up_concat22.conv.convs[1],
            "up_concat22.conv.convs.1.0": self.up_concat22.conv.convs[1][0],
            "up_concat22.conv.convs.1.1": self.up_concat22.conv.convs[1][1],
            "up_concat22.up_conv": self.up_concat22.up_conv,
            "up_concat03": self.up_concat03,
            "up_concat03.conv": self.up_concat03.conv,
            "up_concat03.conv.convs": self.up_concat03.conv.convs,
            "up_concat03.conv.convs.0": self.up_concat03.conv.convs[0],
            "up_concat03.conv.convs.0.0": self.up_concat03.conv.convs[0][0],
            "up_concat03.conv.convs.0.1": self.up_concat03.conv.convs[0][1],
            "up_concat03.conv.convs.1": self.up_concat03.conv.convs[1],
            "up_concat03.conv.convs.1.0": self.up_concat03.conv.convs[1][0],
            "up_concat03.conv.convs.1.1": self.up_concat03.conv.convs[1][1],
            "up_concat03.up_conv": self.up_concat03.up_conv,
            "up_concat13": self.up_concat13,
            "up_concat13.conv": self.up_concat13.conv,
            "up_concat13.conv.convs": self.up_concat13.conv.convs,
            "up_concat13.conv.convs.0": self.up_concat13.conv.convs[0],
            "up_concat13.conv.convs.0.0": self.up_concat13.conv.convs[0][0],
            "up_concat13.conv.convs.0.1": self.up_concat13.conv.convs[0][1],
            "up_concat13.conv.convs.1": self.up_concat13.conv.convs[1],
            "up_concat13.conv.convs.1.0": self.up_concat13.conv.convs[1][0],
            "up_concat13.conv.convs.1.1": self.up_concat13.conv.convs[1][1],
            "up_concat13.up_conv": self.up_concat13.up_conv,
            "up_concat04": self.up_concat04,
            "up_concat04.conv": self.up_concat04.conv,
            "up_concat04.conv.convs": self.up_concat04.conv.convs,
            "up_concat04.conv.convs.0": self.up_concat04.conv.convs[0],
            "up_concat04.conv.convs.0.0": self.up_concat04.conv.convs[0][0],
            "up_concat04.conv.convs.0.1": self.up_concat04.conv.convs[0][1],
            "up_concat04.conv.convs.1": self.up_concat04.conv.convs[1],
            "up_concat04.conv.convs.1.0": self.up_concat04.conv.convs[1][0],
            "up_concat04.conv.convs.1.1": self.up_concat04.conv.convs[1][1],
            "up_concat04.up_conv": self.up_concat04.up_conv,
            "final1": self.final1,
            "final2": self.final2,
            "final3": self.final3,
            "final4": self.final4,
        }
        self.origin_layer_names = {
            "maxpool1": self.maxpool1,
            "maxpool2": self.maxpool2,
            "maxpool3": self.maxpool3,
            "maxpool0": self.maxpool0,
            "conv00": self.conv00,
            "conv00.convs": self.conv00.convs,
            "conv00.convs.0": self.conv00.convs[0],
            "conv00.convs.0.0": self.conv00.convs[0][0],
            "conv00.convs.0.1": self.conv00.convs[0][1],
            "conv00.convs.0.2": self.conv00.convs[0][2],
            "conv00.convs.1": self.conv00.convs[1],
            "conv00.convs.1.0": self.conv00.convs[1][0],
            "conv00.convs.1.1": self.conv00.convs[1][1],
            "conv00.convs.1.2": self.conv00.convs[1][2],
            "conv10": self.conv10,
            "conv10.convs": self.conv10.convs,
            "conv10.convs.0": self.conv10.convs[0],
            "conv10.convs.0.0": self.conv10.convs[0][0],
            "conv10.convs.0.1": self.conv10.convs[0][1],
            "conv10.convs.0.2": self.conv10.convs[0][2],
            "conv10.convs.1": self.conv10.convs[1],
            "conv10.convs.1.0": self.conv10.convs[1][0],
            "conv10.convs.1.1": self.conv10.convs[1][1],
            "conv10.convs.1.2": self.conv10.convs[1][2],
            "conv20": self.conv20,
            "conv20.convs": self.conv20.convs,
            "conv20.convs.0": self.conv20.convs[0],
            "conv20.convs.0.0": self.conv20.convs[0][0],
            "conv20.convs.0.1": self.conv20.convs[0][1],
            "conv20.convs.0.2": self.conv20.convs[0][2],
            "conv20.convs.1": self.conv20.convs[1],
            "conv20.convs.1.0": self.conv20.convs[1][0],
            "conv20.convs.1.1": self.conv20.convs[1][1],
            "conv20.convs.1.2": self.conv20.convs[1][2],
            "conv30": self.conv30,
            "conv30.convs": self.conv30.convs,
            "conv30.convs.0": self.conv30.convs[0],
            "conv30.convs.0.0": self.conv30.convs[0][0],
            "conv30.convs.0.1": self.conv30.convs[0][1],
            "conv30.convs.0.2": self.conv30.convs[0][2],
            "conv30.convs.1": self.conv30.convs[1],
            "conv30.convs.1.0": self.conv30.convs[1][0],
            "conv30.convs.1.1": self.conv30.convs[1][1],
            "conv30.convs.1.2": self.conv30.convs[1][2],
            "conv40": self.conv40,
            "conv40.convs": self.conv40.convs,
            "conv40.convs.0": self.conv40.convs[0],
            "conv40.convs.0.0": self.conv40.convs[0][0],
            "conv40.convs.0.1": self.conv40.convs[0][1],
            "conv40.convs.0.2": self.conv40.convs[0][2],
            "conv40.convs.1": self.conv40.convs[1],
            "conv40.convs.1.0": self.conv40.convs[1][0],
            "conv40.convs.1.1": self.conv40.convs[1][1],
            "conv40.convs.1.2": self.conv40.convs[1][2],
            "up_concat01": self.up_concat01,
            "up_concat01.conv": self.up_concat01.conv,
            "up_concat01.conv.convs": self.up_concat01.conv.convs,
            "up_concat01.conv.convs.0": self.up_concat01.conv.convs[0],
            "up_concat01.conv.convs.0.0": self.up_concat01.conv.convs[0][0],
            "up_concat01.conv.convs.0.1": self.up_concat01.conv.convs[0][1],
            "up_concat01.conv.convs.1": self.up_concat01.conv.convs[1],
            "up_concat01.conv.convs.1.0": self.up_concat01.conv.convs[1][0],
            "up_concat01.conv.convs.1.1": self.up_concat01.conv.convs[1][1],
            "up_concat01.up_conv": self.up_concat01.up_conv,
            "up_concat11": self.up_concat11,
            "up_concat11.conv": self.up_concat11.conv,
            "up_concat11.conv.convs": self.up_concat11.conv.convs,
            "up_concat11.conv.convs.0": self.up_concat11.conv.convs[0],
            "up_concat11.conv.convs.0.0": self.up_concat11.conv.convs[0][0],
            "up_concat11.conv.convs.0.1": self.up_concat11.conv.convs[0][1],
            "up_concat11.conv.convs.1": self.up_concat11.conv.convs[1],
            "up_concat11.conv.convs.1.0": self.up_concat11.conv.convs[1][0],
            "up_concat11.conv.convs.1.1": self.up_concat11.conv.convs[1][1],
            "up_concat11.up_conv": self.up_concat11.up_conv,
            "up_concat21": self.up_concat21,
            "up_concat21.conv": self.up_concat21.conv,
            "up_concat21.conv.convs": self.up_concat21.conv.convs,
            "up_concat21.conv.convs.0": self.up_concat21.conv.convs[0],
            "up_concat21.conv.convs.0.0": self.up_concat21.conv.convs[0][0],
            "up_concat21.conv.convs.0.1": self.up_concat21.conv.convs[0][1],
            "up_concat21.conv.convs.1": self.up_concat21.conv.convs[1],
            "up_concat21.conv.convs.1.0": self.up_concat21.conv.convs[1][0],
            "up_concat21.conv.convs.1.1": self.up_concat21.conv.convs[1][1],
            "up_concat21.up_conv": self.up_concat21.up_conv,
            "up_concat31": self.up_concat31,
            "up_concat31.conv": self.up_concat31.conv,
            "up_concat31.conv.convs": self.up_concat31.conv.convs,
            "up_concat31.conv.convs.0": self.up_concat31.conv.convs[0],
            "up_concat31.conv.convs.0.0": self.up_concat31.conv.convs[0][0],
            "up_concat31.conv.convs.0.1": self.up_concat31.conv.convs[0][1],
            "up_concat31.conv.convs.1": self.up_concat31.conv.convs[1],
            "up_concat31.conv.convs.1.0": self.up_concat31.conv.convs[1][0],
            "up_concat31.conv.convs.1.1": self.up_concat31.conv.convs[1][1],
            "up_concat31.up_conv": self.up_concat31.up_conv,
            "up_concat02": self.up_concat02,
            "up_concat02.conv": self.up_concat02.conv,
            "up_concat02.conv.convs": self.up_concat02.conv.convs,
            "up_concat02.conv.convs.0": self.up_concat02.conv.convs[0],
            "up_concat02.conv.convs.0.0": self.up_concat02.conv.convs[0][0],
            "up_concat02.conv.convs.0.1": self.up_concat02.conv.convs[0][1],
            "up_concat02.conv.convs.1": self.up_concat02.conv.convs[1],
            "up_concat02.conv.convs.1.0": self.up_concat02.conv.convs[1][0],
            "up_concat02.conv.convs.1.1": self.up_concat02.conv.convs[1][1],
            "up_concat02.up_conv": self.up_concat02.up_conv,
            "up_concat12": self.up_concat12,
            "up_concat12.conv": self.up_concat12.conv,
            "up_concat12.conv.convs": self.up_concat12.conv.convs,
            "up_concat12.conv.convs.0": self.up_concat12.conv.convs[0],
            "up_concat12.conv.convs.0.0": self.up_concat12.conv.convs[0][0],
            "up_concat12.conv.convs.0.1": self.up_concat12.conv.convs[0][1],
            "up_concat12.conv.convs.1": self.up_concat12.conv.convs[1],
            "up_concat12.conv.convs.1.0": self.up_concat12.conv.convs[1][0],
            "up_concat12.conv.convs.1.1": self.up_concat12.conv.convs[1][1],
            "up_concat12.up_conv": self.up_concat12.up_conv,
            "up_concat22": self.up_concat22,
            "up_concat22.conv": self.up_concat22.conv,
            "up_concat22.conv.convs": self.up_concat22.conv.convs,
            "up_concat22.conv.convs.0": self.up_concat22.conv.convs[0],
            "up_concat22.conv.convs.0.0": self.up_concat22.conv.convs[0][0],
            "up_concat22.conv.convs.0.1": self.up_concat22.conv.convs[0][1],
            "up_concat22.conv.convs.1": self.up_concat22.conv.convs[1],
            "up_concat22.conv.convs.1.0": self.up_concat22.conv.convs[1][0],
            "up_concat22.conv.convs.1.1": self.up_concat22.conv.convs[1][1],
            "up_concat22.up_conv": self.up_concat22.up_conv,
            "up_concat03": self.up_concat03,
            "up_concat03.conv": self.up_concat03.conv,
            "up_concat03.conv.convs": self.up_concat03.conv.convs,
            "up_concat03.conv.convs.0": self.up_concat03.conv.convs[0],
            "up_concat03.conv.convs.0.0": self.up_concat03.conv.convs[0][0],
            "up_concat03.conv.convs.0.1": self.up_concat03.conv.convs[0][1],
            "up_concat03.conv.convs.1": self.up_concat03.conv.convs[1],
            "up_concat03.conv.convs.1.0": self.up_concat03.conv.convs[1][0],
            "up_concat03.conv.convs.1.1": self.up_concat03.conv.convs[1][1],
            "up_concat03.up_conv": self.up_concat03.up_conv,
            "up_concat13": self.up_concat13,
            "up_concat13.conv": self.up_concat13.conv,
            "up_concat13.conv.convs": self.up_concat13.conv.convs,
            "up_concat13.conv.convs.0": self.up_concat13.conv.convs[0],
            "up_concat13.conv.convs.0.0": self.up_concat13.conv.convs[0][0],
            "up_concat13.conv.convs.0.1": self.up_concat13.conv.convs[0][1],
            "up_concat13.conv.convs.1": self.up_concat13.conv.convs[1],
            "up_concat13.conv.convs.1.0": self.up_concat13.conv.convs[1][0],
            "up_concat13.conv.convs.1.1": self.up_concat13.conv.convs[1][1],
            "up_concat13.up_conv": self.up_concat13.up_conv,
            "up_concat04": self.up_concat04,
            "up_concat04.conv": self.up_concat04.conv,
            "up_concat04.conv.convs": self.up_concat04.conv.convs,
            "up_concat04.conv.convs.0": self.up_concat04.conv.convs[0],
            "up_concat04.conv.convs.0.0": self.up_concat04.conv.convs[0][0],
            "up_concat04.conv.convs.0.1": self.up_concat04.conv.convs[0][1],
            "up_concat04.conv.convs.1": self.up_concat04.conv.convs[1],
            "up_concat04.conv.convs.1.0": self.up_concat04.conv.convs[1][0],
            "up_concat04.conv.convs.1.1": self.up_concat04.conv.convs[1][1],
            "up_concat04.up_conv": self.up_concat04.up_conv,
            "final1": self.final1,
            "final2": self.final2,
            "final3": self.final3,
            "final4": self.final4,
        }

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []




    def forward(self, x):
        x00 = self.conv00(x)  # channel = filters[0]
        x10 = self.conv10(self.maxpool0(x00))  # channel = filters[1]
        x20 = self.conv20(self.maxpool1(x10))  # channel = filters[2]
        x30 = self.conv30(self.maxpool2(x20))  # channel = filters[3]
        x40 = self.conv40(self.maxpool3(x30))  # channel = filters[4]

        x01 = self.up_concat01(x10, x00)  # channel = filters[0]
        x11 = self.up_concat11(x20, x10)  # channel = filters[1]
        x21 = self.up_concat21(x30, x20)  # channel = filters[2]
        x31 = self.up_concat31(x40, x30)  # channel = filters[3]

        x02 = self.up_concat02(x11, x00, x01)  # channel = filters[0]
        x12 = self.up_concat12(x21, x10, x11)  # channel = filters[1]
        x22 = self.up_concat22(x31, x20, x21)  # channel = filters[2]

        x03 = self.up_concat03(x12, x00, x01, x02)  # channel = filters[0]
        x13 = self.up_concat13(x22, x10, x11, x12)  # channel = filters[1]

        x04 = self.up_concat04(x13, x00, x01, x02, x03)  # channel = filters[0]

        final1 = self.final1(x01)
        final2 = self.final2(x02)
        final3 = self.final3(x03)
        final4 = self.final4(x04)

        if self.use_ds:
            final = torch.stack((final1, final2, final3, final4), dim=0)
            return final
        return final4

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]


    def set_layers(self, layer_name, new_layer):
        if 'maxpool1' == layer_name:
            self.maxpool1 = new_layer
            self.layer_names["maxpool1"] = new_layer
            self.origin_layer_names["maxpool1"] = new_layer
        if 'maxpool2' == layer_name:
            self.maxpool2 = new_layer
            self.layer_names["maxpool2"] = new_layer
            self.origin_layer_names["maxpool2"] = new_layer
        if 'maxpool3' == layer_name:
            self.maxpool3 = new_layer
            self.layer_names["maxpool3"] = new_layer
            self.origin_layer_names["maxpool3"] = new_layer
        if 'maxpool0' == layer_name:
            self.maxpool0 = new_layer
            self.layer_names["maxpool0"] = new_layer
            self.origin_layer_names["maxpool0"] = new_layer
        elif 'conv00' == layer_name:
            self.conv00 = new_layer
            self.layer_names["conv00"] = new_layer
            self.origin_layer_names["conv00"] = new_layer
        elif 'conv00.convs' == layer_name:
            self.conv00.convs = new_layer
            self.layer_names["conv00.convs"] = new_layer
            self.origin_layer_names["conv00.convs"] = new_layer
        elif 'conv00.convs.0' == layer_name:
            self.conv00.convs[0] = new_layer
            self.layer_names["conv00.convs.0"] = new_layer
            self.origin_layer_names["conv00.convs.0"] = new_layer
        elif 'conv00.convs.0.0' == layer_name:
            self.conv00.convs[0][0] = new_layer
            self.layer_names["conv00.convs.0.0"] = new_layer
            self.origin_layer_names["conv00.convs.0.0"] = new_layer
        elif 'conv00.convs.0.1' == layer_name:
            self.conv00.convs[0][1] = new_layer
            self.layer_names["conv00.convs.0.1"] = new_layer
            self.origin_layer_names["conv00.convs.0.1"] = new_layer
        elif 'conv00.convs.0.2' == layer_name:
            self.conv00.convs[0][2] = new_layer
            self.layer_names["conv00.convs.0.2"] = new_layer
            self.origin_layer_names["conv00.convs.0.2"] = new_layer
        elif 'conv00.convs.1' == layer_name:
            self.conv00.convs[1] = new_layer
            self.layer_names["conv00.convs.1"] = new_layer
            self.origin_layer_names["conv00.convs.1"] = new_layer
        elif 'conv00.convs.1.0' == layer_name:
            self.conv00.convs[1][0] = new_layer
            self.layer_names["conv00.convs.1.0"] = new_layer
            self.origin_layer_names["conv00.convs.1.0"] = new_layer
        elif 'conv00.convs.1.1' == layer_name:
            self.conv00.convs[1][1] = new_layer
            self.layer_names["conv00.convs.1.1"] = new_layer
            self.origin_layer_names["conv00.convs.1.1"] = new_layer
        elif 'conv00.convs.1.2' == layer_name:
            self.conv00.convs[1][2] = new_layer
            self.layer_names["conv00.convs.1.2"] = new_layer
            self.origin_layer_names["conv00.convs.1.2"] = new_layer
        elif 'conv10' == layer_name:
            self.conv10 = new_layer
            self.layer_names["conv10"] = new_layer
            self.origin_layer_names["conv10"] = new_layer
        elif 'conv10.convs' == layer_name:
            self.conv10.convs = new_layer
            self.layer_names["conv10.convs"] = new_layer
            self.origin_layer_names["conv10.convs"] = new_layer
        elif 'conv10.convs.0' == layer_name:
            self.conv10.convs[0] = new_layer
            self.layer_names["conv10.convs.0"] = new_layer
            self.origin_layer_names["conv10.convs.0"] = new_layer
        elif 'conv10.convs.0.0' == layer_name:
            self.conv10.convs[0][0] = new_layer
            self.layer_names["conv10.convs.0.0"] = new_layer
            self.origin_layer_names["conv10.convs.0.0"] = new_layer
        elif 'conv10.convs.0.1' == layer_name:
            self.conv10.convs[0][1] = new_layer
            self.layer_names["conv10.convs.0.1"] = new_layer
            self.origin_layer_names["conv10.convs.0.1"] = new_layer
        elif 'conv10.convs.0.2' == layer_name:
            self.conv10.convs[0][2] = new_layer
            self.layer_names["conv10.convs.0.2"] = new_layer
            self.origin_layer_names["conv10.convs.0.2"] = new_layer
        elif 'conv10.convs.1' == layer_name:
            self.conv10.convs[1] = new_layer
            self.layer_names["conv10.convs.1"] = new_layer
            self.origin_layer_names["conv10.convs.1"] = new_layer
        elif 'conv10.convs.1.0' == layer_name:
            self.conv10.convs[1][0] = new_layer
            self.layer_names["conv10.convs.1.0"] = new_layer
            self.origin_layer_names["conv10.convs.1.0"] = new_layer
        elif 'conv10.convs.1.1' == layer_name:
            self.conv10.convs[1][1] = new_layer
            self.layer_names["conv10.convs.1.1"] = new_layer
            self.origin_layer_names["conv10.convs.1.1"] = new_layer
        elif 'conv10.convs.1.2' == layer_name:
            self.conv10.convs[1][2] = new_layer
            self.layer_names["conv10.convs.1.2"] = new_layer
            self.origin_layer_names["conv10.convs.1.2"] = new_layer
        elif 'conv20' == layer_name:
            self.conv20 = new_layer
            self.layer_names["conv20"] = new_layer
            self.origin_layer_names["conv20"] = new_layer
        elif 'conv20.convs' == layer_name:
            self.conv20.convs = new_layer
            self.layer_names["conv20.convs"] = new_layer
            self.origin_layer_names["conv20.convs"] = new_layer
        elif 'conv20.convs.0' == layer_name:
            self.conv20.convs[0] = new_layer
            self.layer_names["conv20.convs.0"] = new_layer
            self.origin_layer_names["conv20.convs.0"] = new_layer
        elif 'conv20.convs.0.0' == layer_name:
            self.conv20.convs[0][0] = new_layer
            self.layer_names["conv20.convs.0.0"] = new_layer
            self.origin_layer_names["conv20.convs.0.0"] = new_layer
        elif 'conv20.convs.0.1' == layer_name:
            self.conv20.convs[0][1] = new_layer
            self.layer_names["conv20.convs.0.1"] = new_layer
            self.origin_layer_names["conv20.convs.0.1"] = new_layer
        elif 'conv20.convs.0.2' == layer_name:
            self.conv20.convs[0][2] = new_layer
            self.layer_names["conv20.convs.0.2"] = new_layer
            self.origin_layer_names["conv20.convs.0.2"] = new_layer
        elif 'conv20.convs.1' == layer_name:
            self.conv20.convs[1] = new_layer
            self.layer_names["conv20.convs.1"] = new_layer
            self.origin_layer_names["conv20.convs.1"] = new_layer
        elif 'conv20.convs.1.0' == layer_name:
            self.conv20.convs[1][0] = new_layer
            self.layer_names["conv20.convs.1.0"] = new_layer
            self.origin_layer_names["conv20.convs.1.0"] = new_layer
        elif 'conv20.convs.1.1' == layer_name:
            self.conv20.convs[1][1] = new_layer
            self.layer_names["conv20.convs.1.1"] = new_layer
            self.origin_layer_names["conv20.convs.1.1"] = new_layer
        elif 'conv20.convs.1.2' == layer_name:
            self.conv20.convs[1][2] = new_layer
            self.layer_names["conv20.convs.1.2"] = new_layer
            self.origin_layer_names["conv20.convs.1.2"] = new_layer
        elif 'conv30' == layer_name:
            self.conv30 = new_layer
            self.layer_names["conv30"] = new_layer
            self.origin_layer_names["conv30"] = new_layer
        elif 'conv30.convs' == layer_name:
            self.conv30.convs = new_layer
            self.layer_names["conv30.convs"] = new_layer
            self.origin_layer_names["conv30.convs"] = new_layer
        elif 'conv30.convs.0' == layer_name:
            self.conv30.convs[0] = new_layer
            self.layer_names["conv30.convs.0"] = new_layer
            self.origin_layer_names["conv30.convs.0"] = new_layer
        elif 'conv30.convs.0.0' == layer_name:
            self.conv30.convs[0][0] = new_layer
            self.layer_names["conv30.convs.0.0"] = new_layer
            self.origin_layer_names["conv30.convs.0.0"] = new_layer
        elif 'conv30.convs.0.1' == layer_name:
            self.conv30.convs[0][1] = new_layer
            self.layer_names["conv30.convs.0.1"] = new_layer
            self.origin_layer_names["conv30.convs.0.1"] = new_layer
        elif 'conv30.convs.0.2' == layer_name:
            self.conv30.convs[0][2] = new_layer
            self.layer_names["conv30.convs.0.2"] = new_layer
            self.origin_layer_names["conv30.convs.0.2"] = new_layer
        elif 'conv30.convs.1' == layer_name:
            self.conv30.convs[1] = new_layer
            self.layer_names["conv30.convs.1"] = new_layer
            self.origin_layer_names["conv30.convs.1"] = new_layer
        elif 'conv30.convs.1.0' == layer_name:
            self.conv30.convs[1][0] = new_layer
            self.layer_names["conv30.convs.1.0"] = new_layer
            self.origin_layer_names["conv30.convs.1.0"] = new_layer
        elif 'conv30.convs.1.1' == layer_name:
            self.conv30.convs[1][1] = new_layer
            self.layer_names["conv30.convs.1.1"] = new_layer
            self.origin_layer_names["conv30.convs.1.1"] = new_layer
        elif 'conv30.convs.1.2' == layer_name:
            self.conv30.convs[1][2] = new_layer
            self.layer_names["conv30.convs.1.2"] = new_layer
            self.origin_layer_names["conv30.convs.1.2"] = new_layer
        elif 'conv40' == layer_name:
            self.conv40 = new_layer
            self.layer_names["conv40"] = new_layer
            self.origin_layer_names["conv40"] = new_layer
        elif 'conv40.convs' == layer_name:
            self.conv40.convs = new_layer
            self.layer_names["conv40.convs"] = new_layer
            self.origin_layer_names["conv40.convs"] = new_layer
        elif 'conv40.convs.0' == layer_name:
            self.conv40.convs[0] = new_layer
            self.layer_names["conv40.convs.0"] = new_layer
            self.origin_layer_names["conv40.convs.0"] = new_layer
        elif 'conv40.convs.0.0' == layer_name:
            self.conv40.convs[0][0] = new_layer
            self.layer_names["conv40.convs.0.0"] = new_layer
            self.origin_layer_names["conv40.convs.0.0"] = new_layer
        elif 'conv40.convs.0.1' == layer_name:
            self.conv40.convs[0][1] = new_layer
            self.layer_names["conv40.convs.0.1"] = new_layer
            self.origin_layer_names["conv40.convs.0.1"] = new_layer
        elif 'conv40.convs.0.2' == layer_name:
            self.conv40.convs[0][2] = new_layer
            self.layer_names["conv40.convs.0.2"] = new_layer
            self.origin_layer_names["conv40.convs.0.2"] = new_layer
        elif 'conv40.convs.1' == layer_name:
            self.conv40.convs[1] = new_layer
            self.layer_names["conv40.convs.1"] = new_layer
            self.origin_layer_names["conv40.convs.1"] = new_layer
        elif 'conv40.convs.1.0' == layer_name:
            self.conv40.convs[1][0] = new_layer
            self.layer_names["conv40.convs.1.0"] = new_layer
            self.origin_layer_names["conv40.convs.1.0"] = new_layer
        elif 'conv40.convs.1.1' == layer_name:
            self.conv40.convs[1][1] = new_layer
            self.layer_names["conv40.convs.1.1"] = new_layer
            self.origin_layer_names["conv40.convs.1.1"] = new_layer
        elif 'conv40.convs.1.2' == layer_name:
            self.conv40.convs[1][2] = new_layer
            self.layer_names["conv40.convs.1.2"] = new_layer
            self.origin_layer_names["conv40.convs.1.2"] = new_layer
        elif 'up_concat01' == layer_name:
            self.up_concat01 = new_layer
            self.layer_names["up_concat01"] = new_layer
            self.origin_layer_names["up_concat01"] = new_layer
        elif 'up_concat01.conv' == layer_name:
            self.up_concat01.conv = new_layer
            self.layer_names["up_concat01.conv"] = new_layer
            self.origin_layer_names["up_concat01.conv"] = new_layer
        elif 'up_concat01.conv.convs' == layer_name:
            self.up_concat01.conv.convs = new_layer
            self.layer_names["up_concat01.conv.convs"] = new_layer
            self.origin_layer_names["up_concat01.conv.convs"] = new_layer
        elif 'up_concat01.conv.convs.0' == layer_name:
            self.up_concat01.conv.convs[0] = new_layer
            self.layer_names["up_concat01.conv.convs.0"] = new_layer
            self.origin_layer_names["up_concat01.conv.convs.0"] = new_layer
        elif 'up_concat01.conv.convs.0.0' == layer_name:
            self.up_concat01.conv.convs[0][0] = new_layer
            self.layer_names["up_concat01.conv.convs.0.0"] = new_layer
            self.origin_layer_names["up_concat01.conv.convs.0.0"] = new_layer
        elif 'up_concat01.conv.convs.0.1' == layer_name:
            self.up_concat01.conv.convs[0][1] = new_layer
            self.layer_names["up_concat01.conv.convs.0.1"] = new_layer
            self.origin_layer_names["up_concat01.conv.convs.0.1"] = new_layer
        elif 'up_concat01.conv.convs.1' == layer_name:
            self.up_concat01.conv.convs[1] = new_layer
            self.layer_names["up_concat01.conv.convs.1"] = new_layer
            self.origin_layer_names["up_concat01.conv.convs.1"] = new_layer
        elif 'up_concat01.conv.convs.1.0' == layer_name:
            self.up_concat01.conv.convs[1][0] = new_layer
            self.layer_names["up_concat01.conv.convs.1.0"] = new_layer
            self.origin_layer_names["up_concat01.conv.convs.1.0"] = new_layer
        elif 'up_concat01.conv.convs.1.1' == layer_name:
            self.up_concat01.conv.convs[1][1] = new_layer
            self.layer_names["up_concat01.conv.convs.1.1"] = new_layer
            self.origin_layer_names["up_concat01.conv.convs.1.1"] = new_layer
        elif 'up_concat01.up_conv' == layer_name:
            self.up_concat01.up_conv = new_layer
            self.layer_names["up_concat01.up_conv"] = new_layer
            self.origin_layer_names["up_concat01.up_conv"] = new_layer
        elif 'up_concat11' == layer_name:
            self.up_concat11 = new_layer
            self.layer_names["up_concat11"] = new_layer
            self.origin_layer_names["up_concat11"] = new_layer
        elif 'up_concat11.conv' == layer_name:
            self.up_concat11.conv = new_layer
            self.layer_names["up_concat11.conv"] = new_layer
            self.origin_layer_names["up_concat11.conv"] = new_layer
        elif 'up_concat11.conv.convs' == layer_name:
            self.up_concat11.conv.convs = new_layer
            self.layer_names["up_concat11.conv.convs"] = new_layer
            self.origin_layer_names["up_concat11.conv.convs"] = new_layer
        elif 'up_concat11.conv.convs.0' == layer_name:
            self.up_concat11.conv.convs[0] = new_layer
            self.layer_names["up_concat11.conv.convs.0"] = new_layer
            self.origin_layer_names["up_concat11.conv.convs.0"] = new_layer
        elif 'up_concat11.conv.convs.0.0' == layer_name:
            self.up_concat11.conv.convs[0][0] = new_layer
            self.layer_names["up_concat11.conv.convs.0.0"] = new_layer
            self.origin_layer_names["up_concat11.conv.convs.0.0"] = new_layer
        elif 'up_concat11.conv.convs.0.1' == layer_name:
            self.up_concat11.conv.convs[0][1] = new_layer
            self.layer_names["up_concat11.conv.convs.0.1"] = new_layer
            self.origin_layer_names["up_concat11.conv.convs.0.1"] = new_layer
        elif 'up_concat11.conv.convs.1' == layer_name:
            self.up_concat11.conv.convs[1] = new_layer
            self.layer_names["up_concat11.conv.convs.1"] = new_layer
            self.origin_layer_names["up_concat11.conv.convs.1"] = new_layer
        elif 'up_concat11.conv.convs.1.0' == layer_name:
            self.up_concat11.conv.convs[1][0] = new_layer
            self.layer_names["up_concat11.conv.convs.1.0"] = new_layer
            self.origin_layer_names["up_concat11.conv.convs.1.0"] = new_layer
        elif 'up_concat11.conv.convs.1.1' == layer_name:
            self.up_concat11.conv.convs[1][1] = new_layer
            self.layer_names["up_concat11.conv.convs.1.1"] = new_layer
            self.origin_layer_names["up_concat11.conv.convs.1.1"] = new_layer
        elif 'up_concat11.up_conv' == layer_name:
            self.up_concat11.up_conv = new_layer
            self.layer_names["up_concat11.up_conv"] = new_layer
            self.origin_layer_names["up_concat11.up_conv"] = new_layer
        elif 'up_concat21' == layer_name:
            self.up_concat21 = new_layer
            self.layer_names["up_concat21"] = new_layer
            self.origin_layer_names["up_concat21"] = new_layer
        elif 'up_concat21.conv' == layer_name:
            self.up_concat21.conv = new_layer
            self.layer_names["up_concat21.conv"] = new_layer
            self.origin_layer_names["up_concat21.conv"] = new_layer
        elif 'up_concat21.conv.convs' == layer_name:
            self.up_concat21.conv.convs = new_layer
            self.layer_names["up_concat21.conv.convs"] = new_layer
            self.origin_layer_names["up_concat21.conv.convs"] = new_layer
        elif 'up_concat21.conv.convs.0' == layer_name:
            self.up_concat21.conv.convs[0] = new_layer
            self.layer_names["up_concat21.conv.convs.0"] = new_layer
            self.origin_layer_names["up_concat21.conv.convs.0"] = new_layer
        elif 'up_concat21.conv.convs.0.0' == layer_name:
            self.up_concat21.conv.convs[0][0] = new_layer
            self.layer_names["up_concat21.conv.convs.0.0"] = new_layer
            self.origin_layer_names["up_concat21.conv.convs.0.0"] = new_layer
        elif 'up_concat21.conv.convs.0.1' == layer_name:
            self.up_concat21.conv.convs[0][1] = new_layer
            self.layer_names["up_concat21.conv.convs.0.1"] = new_layer
            self.origin_layer_names["up_concat21.conv.convs.0.1"] = new_layer
        elif 'up_concat21.conv.convs.1' == layer_name:
            self.up_concat21.conv.convs[1] = new_layer
            self.layer_names["up_concat21.conv.convs.1"] = new_layer
            self.origin_layer_names["up_concat21.conv.convs.1"] = new_layer
        elif 'up_concat21.conv.convs.1.0' == layer_name:
            self.up_concat21.conv.convs[1][0] = new_layer
            self.layer_names["up_concat21.conv.convs.1.0"] = new_layer
            self.origin_layer_names["up_concat21.conv.convs.1.0"] = new_layer
        elif 'up_concat21.conv.convs.1.1' == layer_name:
            self.up_concat21.conv.convs[1][1] = new_layer
            self.layer_names["up_concat21.conv.convs.1.1"] = new_layer
            self.origin_layer_names["up_concat21.conv.convs.1.1"] = new_layer
        elif 'up_concat21.up_conv' == layer_name:
            self.up_concat21.up_conv = new_layer
            self.layer_names["up_concat21.up_conv"] = new_layer
            self.origin_layer_names["up_concat21.up_conv"] = new_layer
        elif 'up_concat31' == layer_name:
            self.up_concat31 = new_layer
            self.layer_names["up_concat31"] = new_layer
            self.origin_layer_names["up_concat31"] = new_layer
        elif 'up_concat31.conv' == layer_name:
            self.up_concat31.conv = new_layer
            self.layer_names["up_concat31.conv"] = new_layer
            self.origin_layer_names["up_concat31.conv"] = new_layer
        elif 'up_concat31.conv.convs' == layer_name:
            self.up_concat31.conv.convs = new_layer
            self.layer_names["up_concat31.conv.convs"] = new_layer
            self.origin_layer_names["up_concat31.conv.convs"] = new_layer
        elif 'up_concat31.conv.convs.0' == layer_name:
            self.up_concat31.conv.convs[0] = new_layer
            self.layer_names["up_concat31.conv.convs.0"] = new_layer
            self.origin_layer_names["up_concat31.conv.convs.0"] = new_layer
        elif 'up_concat31.conv.convs.0.0' == layer_name:
            self.up_concat31.conv.convs[0][0] = new_layer
            self.layer_names["up_concat31.conv.convs.0.0"] = new_layer
            self.origin_layer_names["up_concat31.conv.convs.0.0"] = new_layer
        elif 'up_concat31.conv.convs.0.1' == layer_name:
            self.up_concat31.conv.convs[0][1] = new_layer
            self.layer_names["up_concat31.conv.convs.0.1"] = new_layer
            self.origin_layer_names["up_concat31.conv.convs.0.1"] = new_layer
        elif 'up_concat31.conv.convs.1' == layer_name:
            self.up_concat31.conv.convs[1] = new_layer
            self.layer_names["up_concat31.conv.convs.1"] = new_layer
            self.origin_layer_names["up_concat31.conv.convs.1"] = new_layer
        elif 'up_concat31.conv.convs.1.0' == layer_name:
            self.up_concat31.conv.convs[1][0] = new_layer
            self.layer_names["up_concat31.conv.convs.1.0"] = new_layer
            self.origin_layer_names["up_concat31.conv.convs.1.0"] = new_layer
        elif 'up_concat31.conv.convs.1.1' == layer_name:
            self.up_concat31.conv.convs[1][1] = new_layer
            self.layer_names["up_concat31.conv.convs.1.1"] = new_layer
            self.origin_layer_names["up_concat31.conv.convs.1.1"] = new_layer
        elif 'up_concat31.up_conv' == layer_name:
            self.up_concat31.up_conv = new_layer
            self.layer_names["up_concat31.up_conv"] = new_layer
            self.origin_layer_names["up_concat31.up_conv"] = new_layer
        elif 'up_concat02' == layer_name:
            self.up_concat02 = new_layer
            self.layer_names["up_concat02"] = new_layer
            self.origin_layer_names["up_concat02"] = new_layer
        elif 'up_concat02.conv' == layer_name:
            self.up_concat02.conv = new_layer
            self.layer_names["up_concat02.conv"] = new_layer
            self.origin_layer_names["up_concat02.conv"] = new_layer
        elif 'up_concat02.conv.convs' == layer_name:
            self.up_concat02.conv.convs = new_layer
            self.layer_names["up_concat02.conv.convs"] = new_layer
            self.origin_layer_names["up_concat02.conv.convs"] = new_layer
        elif 'up_concat02.conv.convs.0' == layer_name:
            self.up_concat02.conv.convs[0] = new_layer
            self.layer_names["up_concat02.conv.convs.0"] = new_layer
            self.origin_layer_names["up_concat02.conv.convs.0"] = new_layer
        elif 'up_concat02.conv.convs.0.0' == layer_name:
            self.up_concat02.conv.convs[0][0] = new_layer
            self.layer_names["up_concat02.conv.convs.0.0"] = new_layer
            self.origin_layer_names["up_concat02.conv.convs.0.0"] = new_layer
        elif 'up_concat02.conv.convs.0.1' == layer_name:
            self.up_concat02.conv.convs[0][1] = new_layer
            self.layer_names["up_concat02.conv.convs.0.1"] = new_layer
            self.origin_layer_names["up_concat02.conv.convs.0.1"] = new_layer
        elif 'up_concat02.conv.convs.1' == layer_name:
            self.up_concat02.conv.convs[1] = new_layer
            self.layer_names["up_concat02.conv.convs.1"] = new_layer
            self.origin_layer_names["up_concat02.conv.convs.1"] = new_layer
        elif 'up_concat02.conv.convs.1.0' == layer_name:
            self.up_concat02.conv.convs[1][0] = new_layer
            self.layer_names["up_concat02.conv.convs.1.0"] = new_layer
            self.origin_layer_names["up_concat02.conv.convs.1.0"] = new_layer
        elif 'up_concat02.conv.convs.1.1' == layer_name:
            self.up_concat02.conv.convs[1][1] = new_layer
            self.layer_names["up_concat02.conv.convs.1.1"] = new_layer
            self.origin_layer_names["up_concat02.conv.convs.1.1"] = new_layer
        elif 'up_concat02.up_conv' == layer_name:
            self.up_concat02.up_conv = new_layer
            self.layer_names["up_concat02.up_conv"] = new_layer
            self.origin_layer_names["up_concat02.up_conv"] = new_layer
        elif 'up_concat12' == layer_name:
            self.up_concat12 = new_layer
            self.layer_names["up_concat12"] = new_layer
            self.origin_layer_names["up_concat12"] = new_layer
        elif 'up_concat12.conv' == layer_name:
            self.up_concat12.conv = new_layer
            self.layer_names["up_concat12.conv"] = new_layer
            self.origin_layer_names["up_concat12.conv"] = new_layer
        elif 'up_concat12.conv.convs' == layer_name:
            self.up_concat12.conv.convs = new_layer
            self.layer_names["up_concat12.conv.convs"] = new_layer
            self.origin_layer_names["up_concat12.conv.convs"] = new_layer
        elif 'up_concat12.conv.convs.0' == layer_name:
            self.up_concat12.conv.convs[0] = new_layer
            self.layer_names["up_concat12.conv.convs.0"] = new_layer
            self.origin_layer_names["up_concat12.conv.convs.0"] = new_layer
        elif 'up_concat12.conv.convs.0.0' == layer_name:
            self.up_concat12.conv.convs[0][0] = new_layer
            self.layer_names["up_concat12.conv.convs.0.0"] = new_layer
            self.origin_layer_names["up_concat12.conv.convs.0.0"] = new_layer
        elif 'up_concat12.conv.convs.0.1' == layer_name:
            self.up_concat12.conv.convs[0][1] = new_layer
            self.layer_names["up_concat12.conv.convs.0.1"] = new_layer
            self.origin_layer_names["up_concat12.conv.convs.0.1"] = new_layer
        elif 'up_concat12.conv.convs.1' == layer_name:
            self.up_concat12.conv.convs[1] = new_layer
            self.layer_names["up_concat12.conv.convs.1"] = new_layer
            self.origin_layer_names["up_concat12.conv.convs.1"] = new_layer
        elif 'up_concat12.conv.convs.1.0' == layer_name:
            self.up_concat12.conv.convs[1][0] = new_layer
            self.layer_names["up_concat12.conv.convs.1.0"] = new_layer
            self.origin_layer_names["up_concat12.conv.convs.1.0"] = new_layer
        elif 'up_concat12.conv.convs.1.1' == layer_name:
            self.up_concat12.conv.convs[1][1] = new_layer
            self.layer_names["up_concat12.conv.convs.1.1"] = new_layer
            self.origin_layer_names["up_concat12.conv.convs.1.1"] = new_layer
        elif 'up_concat12.up_conv' == layer_name:
            self.up_concat12.up_conv = new_layer
            self.layer_names["up_concat12.up_conv"] = new_layer
            self.origin_layer_names["up_concat12.up_conv"] = new_layer
        elif 'up_concat22' == layer_name:
            self.up_concat22 = new_layer
            self.layer_names["up_concat22"] = new_layer
            self.origin_layer_names["up_concat22"] = new_layer
        elif 'up_concat22.conv' == layer_name:
            self.up_concat22.conv = new_layer
            self.layer_names["up_concat22.conv"] = new_layer
            self.origin_layer_names["up_concat22.conv"] = new_layer
        elif 'up_concat22.conv.convs' == layer_name:
            self.up_concat22.conv.convs = new_layer
            self.layer_names["up_concat22.conv.convs"] = new_layer
            self.origin_layer_names["up_concat22.conv.convs"] = new_layer
        elif 'up_concat22.conv.convs.0' == layer_name:
            self.up_concat22.conv.convs[0] = new_layer
            self.layer_names["up_concat22.conv.convs.0"] = new_layer
            self.origin_layer_names["up_concat22.conv.convs.0"] = new_layer
        elif 'up_concat22.conv.convs.0.0' == layer_name:
            self.up_concat22.conv.convs[0][0] = new_layer
            self.layer_names["up_concat22.conv.convs.0.0"] = new_layer
            self.origin_layer_names["up_concat22.conv.convs.0.0"] = new_layer
        elif 'up_concat22.conv.convs.0.1' == layer_name:
            self.up_concat22.conv.convs[0][1] = new_layer
            self.layer_names["up_concat22.conv.convs.0.1"] = new_layer
            self.origin_layer_names["up_concat22.conv.convs.0.1"] = new_layer
        elif 'up_concat22.conv.convs.1' == layer_name:
            self.up_concat22.conv.convs[1] = new_layer
            self.layer_names["up_concat22.conv.convs.1"] = new_layer
            self.origin_layer_names["up_concat22.conv.convs.1"] = new_layer
        elif 'up_concat22.conv.convs.1.0' == layer_name:
            self.up_concat22.conv.convs[1][0] = new_layer
            self.layer_names["up_concat22.conv.convs.1.0"] = new_layer
            self.origin_layer_names["up_concat22.conv.convs.1.0"] = new_layer
        elif 'up_concat22.conv.convs.1.1' == layer_name:
            self.up_concat22.conv.convs[1][1] = new_layer
            self.layer_names["up_concat22.conv.convs.1.1"] = new_layer
            self.origin_layer_names["up_concat22.conv.convs.1.1"] = new_layer
        elif 'up_concat22.up_conv' == layer_name:
            self.up_concat22.up_conv = new_layer
            self.layer_names["up_concat22.up_conv"] = new_layer
            self.origin_layer_names["up_concat22.up_conv"] = new_layer
        elif 'up_concat03' == layer_name:
            self.up_concat03 = new_layer
            self.layer_names["up_concat03"] = new_layer
            self.origin_layer_names["up_concat03"] = new_layer
        elif 'up_concat03.conv' == layer_name:
            self.up_concat03.conv = new_layer
            self.layer_names["up_concat03.conv"] = new_layer
            self.origin_layer_names["up_concat03.conv"] = new_layer
        elif 'up_concat03.conv.convs' == layer_name:
            self.up_concat03.conv.convs = new_layer
            self.layer_names["up_concat03.conv.convs"] = new_layer
            self.origin_layer_names["up_concat03.conv.convs"] = new_layer
        elif 'up_concat03.conv.convs.0' == layer_name:
            self.up_concat03.conv.convs[0] = new_layer
            self.layer_names["up_concat03.conv.convs.0"] = new_layer
            self.origin_layer_names["up_concat03.conv.convs.0"] = new_layer
        elif 'up_concat03.conv.convs.0.0' == layer_name:
            self.up_concat03.conv.convs[0][0] = new_layer
            self.layer_names["up_concat03.conv.convs.0.0"] = new_layer
            self.origin_layer_names["up_concat03.conv.convs.0.0"] = new_layer
        elif 'up_concat03.conv.convs.0.1' == layer_name:
            self.up_concat03.conv.convs[0][1] = new_layer
            self.layer_names["up_concat03.conv.convs.0.1"] = new_layer
            self.origin_layer_names["up_concat03.conv.convs.0.1"] = new_layer
        elif 'up_concat03.conv.convs.1' == layer_name:
            self.up_concat03.conv.convs[1] = new_layer
            self.layer_names["up_concat03.conv.convs.1"] = new_layer
            self.origin_layer_names["up_concat03.conv.convs.1"] = new_layer
        elif 'up_concat03.conv.convs.1.0' == layer_name:
            self.up_concat03.conv.convs[1][0] = new_layer
            self.layer_names["up_concat03.conv.convs.1.0"] = new_layer
            self.origin_layer_names["up_concat03.conv.convs.1.0"] = new_layer
        elif 'up_concat03.conv.convs.1.1' == layer_name:
            self.up_concat03.conv.convs[1][1] = new_layer
            self.layer_names["up_concat03.conv.convs.1.1"] = new_layer
            self.origin_layer_names["up_concat03.conv.convs.1.1"] = new_layer
        elif 'up_concat03.up_conv' == layer_name:
            self.up_concat03.up_conv = new_layer
            self.layer_names["up_concat03.up_conv"] = new_layer
            self.origin_layer_names["up_concat03.up_conv"] = new_layer
        elif 'up_concat13' == layer_name:
            self.up_concat13 = new_layer
            self.layer_names["up_concat13"] = new_layer
            self.origin_layer_names["up_concat13"] = new_layer
        elif 'up_concat13.conv' == layer_name:
            self.up_concat13.conv = new_layer
            self.layer_names["up_concat13.conv"] = new_layer
            self.origin_layer_names["up_concat13.conv"] = new_layer
        elif 'up_concat13.conv.convs' == layer_name:
            self.up_concat13.conv.convs = new_layer
            self.layer_names["up_concat13.conv.convs"] = new_layer
            self.origin_layer_names["up_concat13.conv.convs"] = new_layer
        elif 'up_concat13.conv.convs.0' == layer_name:
            self.up_concat13.conv.convs[0] = new_layer
            self.layer_names["up_concat13.conv.convs.0"] = new_layer
            self.origin_layer_names["up_concat13.conv.convs.0"] = new_layer
        elif 'up_concat13.conv.convs.0.0' == layer_name:
            self.up_concat13.conv.convs[0][0] = new_layer
            self.layer_names["up_concat13.conv.convs.0.0"] = new_layer
            self.origin_layer_names["up_concat13.conv.convs.0.0"] = new_layer
        elif 'up_concat13.conv.convs.0.1' == layer_name:
            self.up_concat13.conv.convs[0][1] = new_layer
            self.layer_names["up_concat13.conv.convs.0.1"] = new_layer
            self.origin_layer_names["up_concat13.conv.convs.0.1"] = new_layer
        elif 'up_concat13.conv.convs.1' == layer_name:
            self.up_concat13.conv.convs[1] = new_layer
            self.layer_names["up_concat13.conv.convs.1"] = new_layer
            self.origin_layer_names["up_concat13.conv.convs.1"] = new_layer
        elif 'up_concat13.conv.convs.1.0' == layer_name:
            self.up_concat13.conv.convs[1][0] = new_layer
            self.layer_names["up_concat13.conv.convs.1.0"] = new_layer
            self.origin_layer_names["up_concat13.conv.convs.1.0"] = new_layer
        elif 'up_concat13.conv.convs.1.1' == layer_name:
            self.up_concat13.conv.convs[1][1] = new_layer
            self.layer_names["up_concat13.conv.convs.1.1"] = new_layer
            self.origin_layer_names["up_concat13.conv.convs.1.1"] = new_layer
        elif 'up_concat13.up_conv' == layer_name:
            self.up_concat13.up_conv = new_layer
            self.layer_names["up_concat13.up_conv"] = new_layer
            self.origin_layer_names["up_concat13.up_conv"] = new_layer
        elif 'up_concat04' == layer_name:
            self.up_concat04 = new_layer
            self.layer_names["up_concat04"] = new_layer
            self.origin_layer_names["up_concat04"] = new_layer
        elif 'up_concat04.conv' == layer_name:
            self.up_concat04.conv = new_layer
            self.layer_names["up_concat04.conv"] = new_layer
            self.origin_layer_names["up_concat04.conv"] = new_layer
        elif 'up_concat04.conv.convs' == layer_name:
            self.up_concat04.conv.convs = new_layer
            self.layer_names["up_concat04.conv.convs"] = new_layer
            self.origin_layer_names["up_concat04.conv.convs"] = new_layer
        elif 'up_concat04.conv.convs.0' == layer_name:
            self.up_concat04.conv.convs[0] = new_layer
            self.layer_names["up_concat04.conv.convs.0"] = new_layer
            self.origin_layer_names["up_concat04.conv.convs.0"] = new_layer
        elif 'up_concat04.conv.convs.0.0' == layer_name:
            self.up_concat04.conv.convs[0][0] = new_layer
            self.layer_names["up_concat04.conv.convs.0.0"] = new_layer
            self.origin_layer_names["up_concat04.conv.convs.0.0"] = new_layer
        elif 'up_concat04.conv.convs.0.1' == layer_name:
            self.up_concat04.conv.convs[0][1] = new_layer
            self.layer_names["up_concat04.conv.convs.0.1"] = new_layer
            self.origin_layer_names["up_concat04.conv.convs.0.1"] = new_layer
        elif 'up_concat04.conv.convs.1' == layer_name:
            self.up_concat04.conv.convs[1] = new_layer
            self.layer_names["up_concat04.conv.convs.1"] = new_layer
            self.origin_layer_names["up_concat04.conv.convs.1"] = new_layer
        elif 'up_concat04.conv.convs.1.0' == layer_name:
            self.up_concat04.conv.convs[1][0] = new_layer
            self.layer_names["up_concat04.conv.convs.1.0"] = new_layer
            self.origin_layer_names["up_concat04.conv.convs.1.0"] = new_layer
        elif 'up_concat04.conv.convs.1.1' == layer_name:
            self.up_concat04.conv.convs[1][1] = new_layer
            self.layer_names["up_concat04.conv.convs.1.1"] = new_layer
            self.origin_layer_names["up_concat04.conv.convs.1.1"] = new_layer
        elif 'up_concat04.up_conv' == layer_name:
            self.up_concat04.up_conv = new_layer
            self.layer_names["up_concat04.up_conv"] = new_layer
            self.origin_layer_names["up_concat04.up_conv"] = new_layer
        elif 'final1' == layer_name:
            self.final1 = new_layer
            self.layer_names["final1"] = new_layer
            self.origin_layer_names["final1"] = new_layer
        elif 'final2' == layer_name:
            self.final2 = new_layer
            self.layer_names["final2"] = new_layer
            self.origin_layer_names["final2"] = new_layer
        elif 'final3' == layer_name:
            self.final3 = new_layer
            self.layer_names["final3"] = new_layer
            self.origin_layer_names["final3"] = new_layer
        elif 'final4' == layer_name:
            self.final4 = new_layer
            self.layer_names["final4"] = new_layer
            self.origin_layer_names["final4"] = new_layer

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



class MyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__()

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.reduction = reduction

    def forward(self, x, weights=1.0):
        input_dtype = x.dtype
        x = x.float() * weights.float()

        if self.reduction == 'mean':
            x = torch.mean(x)
        elif self.reduction == 'sum':
            x = torch.sum(x)

        x = x.to(input_dtype)
        return x


class CrossEntropyWithLogits(MyLoss):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()

    def forward(self, logits, label):
        logits = logits.permute(0, 2, 3, 1)
        label = label.permute(0, 2, 3, 1)

        logits_shape = logits.shape
        label_shape = label.shape
        logits = logits.view(-1, logits_shape[-1])
        label = label.view(-1, label_shape[-1])

        loss = F.cross_entropy(logits, torch.argmax(label, dim=1), reduction=self.reduction)
        return loss


class Losser(nn.Module):
    def __init__(self, network, criterion):
        super(Losser, self).__init__()
        self.network = network
        self.criterion = criterion

    def forward(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss


if __name__ == '__main__':
    from src.model_utils.config import config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.use_deconv = True
    config.use_ds = False
    config.use_bn = False
    net = NestedUNet(in_channels=1).to(device)
    nt = numpy.random.randn(1, 1, 96, 96)
    na = numpy.random.randn(1, 2, 96, 96)
    t = torch.tensor(nt, dtype=torch.float32)
    a = torch.tensor(na, dtype=torch.float32)

    # a = torch.tensor(a, dtype=torch.float32)
    # net(t)
    # print("done")
    config.batch_size = 1
    #net = NestedUNet(in_channels=1, n_classes=2, use_deconv=config.use_deconv,use_bn=config.use_bn, use_ds=config.use_ds).to(device)
    net = NestedUNet(in_channels=1, n_classes=2, feature_scale = 2,use_deconv=True, use_bn=True, use_ds=False)

    losser = CrossEntropyWithLogits()
    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def train_step(data, label):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        logits = net(data)
        loss = losser(logits, label)
        loss.backward()
        optimizer.step()
        return loss.item()

    print("================================================================")
    loss_pytorch = train_step(t, a)
    print(loss_pytorch)
    print("================================================================")
