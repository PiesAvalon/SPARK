import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def _get_bbox(rank, shape, central_fraction):
    """get bbox start and size for slice"""
    if rank == 3:
        c, h, w = shape
    else:
        n, c, h, w = shape

    bbox_h_start = int((float(h) - np.float32(h * central_fraction)) / 2)
    bbox_w_start = int((float(w) - np.float32(w * central_fraction)) / 2)
    bbox_h_size = h - bbox_h_start * 2
    bbox_w_size = w - bbox_w_start * 2

    if rank == 3:
        bbox_begin = (0, bbox_h_start, bbox_w_start)
        bbox_size = (c, bbox_h_size, bbox_w_size)
    else:
        bbox_begin = (0, 0, bbox_h_start, bbox_w_start)
        bbox_size = (n, c, bbox_h_size, bbox_w_size)

    return bbox_begin, bbox_size


class CentralCrop(nn.Module):
    """
    Crops the central region of the images with the central_fraction.

    Args:
        central_fraction (float): Fraction of size to crop. It must be float and in range (0.0, 1.0].

    Inputs:
        - **image** (Tensor) - A 3-D tensor of shape [C, H, W], or a 4-D tensor of shape [N, C, H, W].

    Outputs:
        Tensor, 3-D or 4-D float tensor, according to the input.

    Raises:
        TypeError: If `central_fraction` is not a float.
        ValueError: If `central_fraction` is not in range (0.0, 1.0].
    """

    def __init__(self, central_fraction):
        super(CentralCrop, self).__init__()
        if not isinstance(central_fraction, float):
            raise TypeError(f"central_fraction must be a float, but got {type(central_fraction)}")
        if not 0.0 < central_fraction <= 1.0:
            raise ValueError(f"central_fraction must be in range (0.0, 1.0], but got {central_fraction}")
        self.central_fraction = central_fraction

    def forward(self, image):
        image_shape = image.shape
        rank = len(image_shape)
        if rank not in (3, 4):
            raise ValueError(f"Expected input rank to be 3 or 4, but got {rank}")

        if self.central_fraction == 1.0:
            return image

        if rank == 3:
            c, h, w = image_shape
            image = image.unsqueeze(0)
        else:
            n, c, h, w = image_shape

        start_h = int((h - h * self.central_fraction) / 2)
        end_h = start_h + int(h * self.central_fraction)
        start_w = int((w - w * self.central_fraction) / 2)
        end_w = start_w + int(w * self.central_fraction)

        cropped_image = image[:, :, start_h:end_h, start_w:end_w]

        return cropped_image.squeeze(0) if rank == 3 else cropped_image


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.factor = 56.0 / 64.0
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.cat = torch.cat

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.cat([x1, x2], dim=1)
        return self.conv(x)


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.factor = 104.0 / 136.0
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.cat = torch.cat

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.cat([x1, x2], dim=1)
        return self.conv(x)


class Up3(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.factor = 200 / 280
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.cat = torch.cat

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.cat([x1, x2], dim=1)
        return self.conv(x)


class Up4(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.factor = 392 / 568
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.cat = torch.cat

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetMedical(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up1(1024, 512, bilinear=True)
        self.up2 = Up2(512, 256, bilinear=True)
        self.up3 = Up3(256, 128, bilinear=True)
        self.up4 = Up4(128, 64, bilinear=True)
        self.outc = OutConv(64, n_classes)

        self.in_shapes = {
             'INPUT': [-1, 1, 572, 572],
             'inc.double_conv.0': [1, 1, 572, 572],
             'inc.double_conv.1': [1, 64, 570, 570],
             'inc.double_conv.2': [1, 64, 570, 570],
             'inc.double_conv.3': [1, 64, 568, 568],
             'down1.maxpool_conv.0': [1, 64, 568, 568],
             'down1.maxpool_conv.1.double_conv.0': [1, 64, 284, 284],
             'down1.maxpool_conv.1.double_conv.1': [1, 128, 282, 282],
             'down1.maxpool_conv.1.double_conv.2': [1, 128, 282, 282],
             'down1.maxpool_conv.1.double_conv.3': [1, 128, 280, 280],
             'down2.maxpool_conv.0': [1, 128, 280, 280],
             'down2.maxpool_conv.1.double_conv.0': [1, 128, 140, 140],
             'down2.maxpool_conv.1.double_conv.1': [1, 256, 138, 138],
             'down2.maxpool_conv.1.double_conv.2': [1, 256, 138, 138],
             'down2.maxpool_conv.1.double_conv.3': [1, 256, 136, 136],
             'down3.maxpool_conv.0': [1, 256, 136, 136],
             'down3.maxpool_conv.1.double_conv.0': [1, 256, 68, 68],
             'down3.maxpool_conv.1.double_conv.1': [1, 512, 66, 66],
             'down3.maxpool_conv.1.double_conv.2': [1, 512, 66, 66],
             'down3.maxpool_conv.1.double_conv.3': [1, 512, 64, 64],
             'down4.maxpool_conv.0': [1, 512, 64, 64],
             'down4.maxpool_conv.1.double_conv.0': [1, 512, 32, 32],
             'down4.maxpool_conv.1.double_conv.1': [1, 1024, 30, 30],
             'down4.maxpool_conv.1.double_conv.2': [1, 1024, 30, 30],
             'down4.maxpool_conv.1.double_conv.3': [1, 1024, 28, 28],
             'up1.up': [1, 1024, 28, 28],
             'up1.center_crop': [1, 512, 64, 64],
             'up1.conv.double_conv.0': [1, 1024, 56, 56],
             'up1.conv.double_conv.1': [1, 512, 54, 54],
             'up1.conv.double_conv.2': [1, 512, 54, 54],
             'up1.conv.double_conv.3': [1, 512, 52, 52],
             'up1.relu': [1, 512, 56, 56],
             'up2.up': [1, 512, 52, 52],
             'up2.center_crop': [1, 256, 136, 136],
             'up2.conv.double_conv.0': [1, 512, 104, 104],
             'up2.conv.double_conv.1': [1, 256, 102, 102],
             'up2.conv.double_conv.2': [1, 256, 102, 102],
             'up2.conv.double_conv.3': [1, 256, 100, 100],
             'up2.relu': [1, 256, 104, 104],
             'up3.up': [1, 256, 100, 100],
             'up3.center_crop': [1, 128, 280, 280],
             'up3.conv.double_conv.0': [1, 256, 200, 200],
             'up3.conv.double_conv.1': [1, 128, 198, 198],
             'up3.conv.double_conv.2': [1, 128, 198, 198],
             'up3.conv.double_conv.3': [1, 128, 196, 196],
             'up3.relu': [1, 128, 200, 200],
             'up4.up': [1, 128, 196, 196],
             'up4.center_crop': [1, 64, 568, 568],
             'up4.conv.double_conv.0': [1, 128, 392, 392],
             'up4.conv.double_conv.1': [1, 64, 390, 390],
             'up4.conv.double_conv.2': [1, 64, 390, 390],
             'up4.conv.double_conv.3': [1, 64, 388, 388],
             'up4.relu': [1, 64, 392, 392],
             'outc.conv': [1, 64, 388, 388],
             'OUTPUT': [-1, 64, 388, 388]
        }
        self.out_shapes = {
            'INPUT': [-1, 1, 572, 572],
            'inc.double_conv.0': [1, 64, 570, 570],
             'inc.double_conv.1': [1, 64, 570, 570],
             'inc.double_conv.2': [1, 64, 568, 568],
             'inc.double_conv.3': [1, 64, 568, 568],
             'down1.maxpool_conv.0': [1, 64, 284, 284],
             'down1.maxpool_conv.1.double_conv.0': [1, 128, 282, 282],
             'down1.maxpool_conv.1.double_conv.1': [1, 128, 282, 282],
             'down1.maxpool_conv.1.double_conv.2': [1, 128, 280, 280],
             'down1.maxpool_conv.1.double_conv.3': [1, 128, 280, 280],
             'down2.maxpool_conv.0': [1, 128, 140, 140],
             'down2.maxpool_conv.1.double_conv.0': [1, 256, 138, 138],
             'down2.maxpool_conv.1.double_conv.1': [1, 256, 138, 138],
             'down2.maxpool_conv.1.double_conv.2': [1, 256, 136, 136],
             'down2.maxpool_conv.1.double_conv.3': [1, 256, 136, 136],
             'down3.maxpool_conv.0': [1, 256, 68, 68],
             'down3.maxpool_conv.1.double_conv.0': [1, 512, 66, 66],
             'down3.maxpool_conv.1.double_conv.1': [1, 512, 66, 66],
             'down3.maxpool_conv.1.double_conv.2': [1, 512, 64, 64],
             'down3.maxpool_conv.1.double_conv.3': [1, 512, 64, 64],
             'down4.maxpool_conv.0': [1, 512, 32, 32],
             'down4.maxpool_conv.1.double_conv.0': [1, 1024, 30, 30],
             'down4.maxpool_conv.1.double_conv.1': [1, 1024, 30, 30],
             'down4.maxpool_conv.1.double_conv.2': [1, 1024, 28, 28],
             'down4.maxpool_conv.1.double_conv.3': [1, 1024, 28, 28],
             'up1.up': [1, 512, 56, 56],
             'up1.center_crop': [1, 512, 56, 56],
             'up1.conv.double_conv.0': [1, 512, 54, 54],
             'up1.conv.double_conv.1': [1, 512, 54, 54],
             'up1.conv.double_conv.2': [1, 512, 52, 52],
             'up1.conv.double_conv.3': [1, 512, 52, 52],
             'up1.relu': [1, 512, 56, 56],
             'up2.up': [1, 256, 104, 104],
             'up2.center_crop': [1, 256, 104, 104],
             'up2.conv.double_conv.0': [1, 256, 102, 102],
             'up2.conv.double_conv.1': [1, 256, 102, 102],
             'up2.conv.double_conv.2': [1, 256, 100, 100],
             'up2.conv.double_conv.3': [1, 256, 100, 100],
             'up2.relu': [1, 256, 104, 104],
             'up3.up': [1, 128, 200, 200],
             'up3.center_crop': [1, 128, 200, 200],
             'up3.conv.double_conv.0': [1, 128, 198, 198],
             'up3.conv.double_conv.1': [1, 128, 198, 198],
             'up3.conv.double_conv.2': [1, 128, 196, 196],
             'up3.conv.double_conv.3': [1, 128, 196, 196],
             'up3.relu': [1, 128, 200, 200],
             'up4.up': [1, 64, 392, 392],
             'up4.center_crop': [1, 64, 392, 392],
             'up4.conv.double_conv.0': [1, 64, 390, 390],
             'up4.conv.double_conv.1': [1, 64, 390, 390],
             'up4.conv.double_conv.2': [1, 64, 388, 388],
             'up4.conv.double_conv.3': [1, 64, 388, 388],
             'up4.relu': [1, 64, 392, 392],
             'outc.conv': [1, 2, 388, 388],
            'OUTPUT': [-1, 2, 388, 388]
        }


        self.orders = {
            "inc.double_conv.0": ["INPUT", "inc.double_conv.1"],
            "inc.double_conv.1": ["inc.double_conv.0", "inc.double_conv.2"],
            "inc.double_conv.2": ["inc.double_conv.1", "inc.double_conv.3"],
            "inc.double_conv.3": ["inc.double_conv.2", ["down1.maxpool_conv.0", "up4.center_crop"]],

            "down1.maxpool_conv.0": ["inc.double_conv.3", "down1.maxpool_conv.1.double_conv.0"],
            "down1.maxpool_conv.1.double_conv.0": ["down1.maxpool_conv.0", "down1.maxpool_conv.1.double_conv.1"],
            "down1.maxpool_conv.1.double_conv.1": ["down1.maxpool_conv.1.double_conv.0",
                                                   "down1.maxpool_conv.1.double_conv.2"],
            "down1.maxpool_conv.1.double_conv.2": ["down1.maxpool_conv.1.double_conv.1",
                                                   "down1.maxpool_conv.1.double_conv.3"],
            "down1.maxpool_conv.1.double_conv.3": ["down1.maxpool_conv.1.double_conv.2",
                                                   ["down2.maxpool_conv.0", "up3.center_crop"]],

            "down2.maxpool_conv.0": ["down1.maxpool_conv.1.double_conv.3", "down2.maxpool_conv.1.double_conv.0"],
            "down2.maxpool_conv.1.double_conv.0": ["down2.maxpool_conv.0", "down2.maxpool_conv.1.double_conv.1"],
            "down2.maxpool_conv.1.double_conv.1": ["down2.maxpool_conv.1.double_conv.0",
                                                   "down2.maxpool_conv.1.double_conv.2"],
            "down2.maxpool_conv.1.double_conv.2": ["down2.maxpool_conv.1.double_conv.1",
                                                   "down2.maxpool_conv.1.double_conv.3"],
            "down2.maxpool_conv.1.double_conv.3": ["down2.maxpool_conv.1.double_conv.2",
                                                   ["down3.maxpool_conv.0", "up2.center_crop"]],

            "down3.maxpool_conv.0": ["down2.maxpool_conv.1.double_conv.3", "down3.maxpool_conv.1.double_conv.0"],
            "down3.maxpool_conv.1.double_conv.0": ["down3.maxpool_conv.0", "down3.maxpool_conv.1.double_conv.1"],
            "down3.maxpool_conv.1.double_conv.1": ["down3.maxpool_conv.1.double_conv.0",
                                                   "down3.maxpool_conv.1.double_conv.2"],
            "down3.maxpool_conv.1.double_conv.2": ["down3.maxpool_conv.1.double_conv.1",
                                                   "down3.maxpool_conv.1.double_conv.3"],
            "down3.maxpool_conv.1.double_conv.3": ["down3.maxpool_conv.1.double_conv.2",
                                                   ["down4.maxpool_conv.0", "up1.center_crop"]],

            "down4.maxpool_conv.0": ["down3.maxpool_conv.1.double_conv.3", "down4.maxpool_conv.1.double_conv.0"],
            "down4.maxpool_conv.1.double_conv.0": ["down4.maxpool_conv.0", "down4.maxpool_conv.1.double_conv.1"],
            "down4.maxpool_conv.1.double_conv.1": ["down4.maxpool_conv.1.double_conv.0",
                                                   "down4.maxpool_conv.1.double_conv.2"],
            "down4.maxpool_conv.1.double_conv.2": ["down4.maxpool_conv.1.double_conv.1",
                                                   "down4.maxpool_conv.1.double_conv.3"],
            "down4.maxpool_conv.1.double_conv.3": ["down4.maxpool_conv.1.double_conv.2", "up1.up"],

            "up1.up": ["down4.maxpool_conv.1.double_conv.3", "up1.relu"],
            "up1.relu": ["up1.up", "up1.conv.double_conv.0"],
            "up1.center_crop": ["down3.maxpool_conv.1.double_conv.3", "up1.conv.double_conv.0"],
            "up1.conv.double_conv.0": [["up1.relu", "up1.center_crop"], "up1.conv.double_conv.1"],
            "up1.conv.double_conv.1": ["up1.conv.double_conv.0", "up1.conv.double_conv.2"],
            "up1.conv.double_conv.2": ["up1.conv.double_conv.1", "up1.conv.double_conv.3"],
            "up1.conv.double_conv.3": ["up1.conv.double_conv.2", "up2.up"],

            "up2.up": ["up1.conv.double_conv.3", "up2.relu"],
            "up2.relu": ["up2.up", "up2.conv.double_conv.0"],
            "up2.center_crop": ["down2.maxpool_conv.1.double_conv.3", "up2.conv.double_conv.0"],
            "up2.conv.double_conv.0": [["up2.relu", "up2.center_crop"], "up2.conv.double_conv.1"],
            "up2.conv.double_conv.1": ["up2.conv.double_conv.0", "up2.conv.double_conv.2"],
            "up2.conv.double_conv.2": ["up2.conv.double_conv.1", "up2.conv.double_conv.3"],
            "up2.conv.double_conv.3": ["up2.conv.double_conv.2", "up3.up"],

            "up3.up": ["up2.conv.double_conv.3", "up3.relu"],
            "up3.relu": ["up3.up", "up3.conv.double_conv.0"],
            "up3.center_crop": ["down1.maxpool_conv.1.double_conv.3", "up3.conv.double_conv.0"],
            "up3.conv.double_conv.0": [["up3.relu", "up3.center_crop"], "up3.conv.double_conv.1"],
            "up3.conv.double_conv.1": ["up3.conv.double_conv.0", "up3.conv.double_conv.2"],
            "up3.conv.double_conv.2": ["up3.conv.double_conv.1", "up3.conv.double_conv.3"],
            "up3.conv.double_conv.3": ["up3.conv.double_conv.2", "up4.up"],

            "up4.up": ["up3.conv.double_conv.3", "up4.relu"],
            "up4.relu": ["up4.up", "up4.conv.double_conv.0"],
            "up4.center_crop": ["inc.double_conv.3", "up4.conv.double_conv.0"],
            "up4.conv.double_conv.0": [["up4.relu", "up4.center_crop"], "up4.conv.double_conv.1"],
            "up4.conv.double_conv.1": ["up4.conv.double_conv.0", "up4.conv.double_conv.2"],
            "up4.conv.double_conv.2": ["up4.conv.double_conv.1", "up4.conv.double_conv.3"],
            "up4.conv.double_conv.3": ["up4.conv.double_conv.2", "outc.conv"],

            "outc.conv": ["up4.conv.double_conv.3", "OUTPUT"],
        }

        self.layer_names = {
            "inc": self.inc,
            "inc.double_conv": self.inc.double_conv,
            "inc.double_conv.0": self.inc.double_conv[0],
            "inc.double_conv.1": self.inc.double_conv[1],
            "inc.double_conv.2": self.inc.double_conv[2],
            "inc.double_conv.3": self.inc.double_conv[3],
            "down1": self.down1,
            "down1.maxpool_conv": self.down1.maxpool_conv,
            "down1.maxpool_conv.0": self.down1.maxpool_conv[0],
            "down1.maxpool_conv.1": self.down1.maxpool_conv[1],
            "down1.maxpool_conv.1.double_conv": self.down1.maxpool_conv[1].double_conv,
            "down1.maxpool_conv.1.double_conv.0": self.down1.maxpool_conv[1].double_conv[0],
            "down1.maxpool_conv.1.double_conv.1": self.down1.maxpool_conv[1].double_conv[1],
            "down1.maxpool_conv.1.double_conv.2": self.down1.maxpool_conv[1].double_conv[2],
            "down1.maxpool_conv.1.double_conv.3": self.down1.maxpool_conv[1].double_conv[3],
            "down2": self.down2,
            "down2.maxpool_conv": self.down2.maxpool_conv,
            "down2.maxpool_conv.0": self.down2.maxpool_conv[0],
            "down2.maxpool_conv.1": self.down2.maxpool_conv[1],
            "down2.maxpool_conv.1.double_conv": self.down2.maxpool_conv[1].double_conv,
            "down2.maxpool_conv.1.double_conv.0": self.down2.maxpool_conv[1].double_conv[0],
            "down2.maxpool_conv.1.double_conv.1": self.down2.maxpool_conv[1].double_conv[1],
            "down2.maxpool_conv.1.double_conv.2": self.down2.maxpool_conv[1].double_conv[2],
            "down2.maxpool_conv.1.double_conv.3": self.down2.maxpool_conv[1].double_conv[3],
            "down3": self.down3,
            "down3.maxpool_conv": self.down3.maxpool_conv,
            "down3.maxpool_conv.0": self.down3.maxpool_conv[0],
            "down3.maxpool_conv.1": self.down3.maxpool_conv[1],
            "down3.maxpool_conv.1.double_conv": self.down3.maxpool_conv[1].double_conv,
            "down3.maxpool_conv.1.double_conv.0": self.down3.maxpool_conv[1].double_conv[0],
            "down3.maxpool_conv.1.double_conv.1": self.down3.maxpool_conv[1].double_conv[1],
            "down3.maxpool_conv.1.double_conv.2": self.down3.maxpool_conv[1].double_conv[2],
            "down3.maxpool_conv.1.double_conv.3": self.down3.maxpool_conv[1].double_conv[3],
            "down4": self.down4,
            "down4.maxpool_conv": self.down4.maxpool_conv,
            "down4.maxpool_conv.0": self.down4.maxpool_conv[0],
            "down4.maxpool_conv.1": self.down4.maxpool_conv[1],
            "down4.maxpool_conv.1.double_conv": self.down4.maxpool_conv[1].double_conv,
            "down4.maxpool_conv.1.double_conv.0": self.down4.maxpool_conv[1].double_conv[0],
            "down4.maxpool_conv.1.double_conv.1": self.down4.maxpool_conv[1].double_conv[1],
            "down4.maxpool_conv.1.double_conv.2": self.down4.maxpool_conv[1].double_conv[2],
            "down4.maxpool_conv.1.double_conv.3": self.down4.maxpool_conv[1].double_conv[3],
            "up1": self.up1,
            "up1.center_crop": self.up1.center_crop,
            "up1.conv": self.up1.conv,
            "up1.conv.double_conv": self.up1.conv.double_conv,
            "up1.conv.double_conv.0": self.up1.conv.double_conv[0],
            "up1.conv.double_conv.1": self.up1.conv.double_conv[1],
            "up1.conv.double_conv.2": self.up1.conv.double_conv[2],
            "up1.conv.double_conv.3": self.up1.conv.double_conv[3],
            "up1.up": self.up1.up,
            "up1.relu": self.up1.relu,
            "up2": self.up2,
            "up2.center_crop": self.up2.center_crop,
            "up2.conv": self.up2.conv,
            "up2.conv.double_conv": self.up2.conv.double_conv,
            "up2.conv.double_conv.0": self.up2.conv.double_conv[0],
            "up2.conv.double_conv.1": self.up2.conv.double_conv[1],
            "up2.conv.double_conv.2": self.up2.conv.double_conv[2],
            "up2.conv.double_conv.3": self.up2.conv.double_conv[3],
            "up2.up": self.up2.up,
            "up2.relu": self.up2.relu,
            "up3": self.up3,
            "up3.center_crop": self.up3.center_crop,
            "up3.conv": self.up3.conv,
            "up3.conv.double_conv": self.up3.conv.double_conv,
            "up3.conv.double_conv.0": self.up3.conv.double_conv[0],
            "up3.conv.double_conv.1": self.up3.conv.double_conv[1],
            "up3.conv.double_conv.2": self.up3.conv.double_conv[2],
            "up3.conv.double_conv.3": self.up3.conv.double_conv[3],
            "up3.up": self.up3.up,
            "up3.relu": self.up3.relu,
            "up4": self.up4,
            "up4.center_crop": self.up4.center_crop,
            "up4.conv": self.up4.conv,
            "up4.conv.double_conv": self.up4.conv.double_conv,
            "up4.conv.double_conv.0": self.up4.conv.double_conv[0],
            "up4.conv.double_conv.1": self.up4.conv.double_conv[1],
            "up4.conv.double_conv.2": self.up4.conv.double_conv[2],
            "up4.conv.double_conv.3": self.up4.conv.double_conv[3],
            "up4.up": self.up4.up,
            "up4.relu": self.up4.relu,
            "outc": self.outc,
            "outc.conv": self.outc.conv,
        }

        self.origin_layer_names = {
            "inc": self.inc,
            "inc.double_conv": self.inc.double_conv,
            "inc.double_conv.0": self.inc.double_conv[0],
            "inc.double_conv.1": self.inc.double_conv[1],
            "inc.double_conv.2": self.inc.double_conv[2],
            "inc.double_conv.3": self.inc.double_conv[3],
            "down1": self.down1,
            "down1.maxpool_conv": self.down1.maxpool_conv,
            "down1.maxpool_conv.0": self.down1.maxpool_conv[0],
            "down1.maxpool_conv.1": self.down1.maxpool_conv[1],
            "down1.maxpool_conv.1.double_conv": self.down1.maxpool_conv[1].double_conv,
            "down1.maxpool_conv.1.double_conv.0": self.down1.maxpool_conv[1].double_conv[0],
            "down1.maxpool_conv.1.double_conv.1": self.down1.maxpool_conv[1].double_conv[1],
            "down1.maxpool_conv.1.double_conv.2": self.down1.maxpool_conv[1].double_conv[2],
            "down1.maxpool_conv.1.double_conv.3": self.down1.maxpool_conv[1].double_conv[3],
            "down2": self.down2,
            "down2.maxpool_conv": self.down2.maxpool_conv,
            "down2.maxpool_conv.0": self.down2.maxpool_conv[0],
            "down2.maxpool_conv.1": self.down2.maxpool_conv[1],
            "down2.maxpool_conv.1.double_conv": self.down2.maxpool_conv[1].double_conv,
            "down2.maxpool_conv.1.double_conv.0": self.down2.maxpool_conv[1].double_conv[0],
            "down2.maxpool_conv.1.double_conv.1": self.down2.maxpool_conv[1].double_conv[1],
            "down2.maxpool_conv.1.double_conv.2": self.down2.maxpool_conv[1].double_conv[2],
            "down2.maxpool_conv.1.double_conv.3": self.down2.maxpool_conv[1].double_conv[3],
            "down3": self.down3,
            "down3.maxpool_conv": self.down3.maxpool_conv,
            "down3.maxpool_conv.0": self.down3.maxpool_conv[0],
            "down3.maxpool_conv.1": self.down3.maxpool_conv[1],
            "down3.maxpool_conv.1.double_conv": self.down3.maxpool_conv[1].double_conv,
            "down3.maxpool_conv.1.double_conv.0": self.down3.maxpool_conv[1].double_conv[0],
            "down3.maxpool_conv.1.double_conv.1": self.down3.maxpool_conv[1].double_conv[1],
            "down3.maxpool_conv.1.double_conv.2": self.down3.maxpool_conv[1].double_conv[2],
            "down3.maxpool_conv.1.double_conv.3": self.down3.maxpool_conv[1].double_conv[3],
            "down4": self.down4,
            "down4.maxpool_conv": self.down4.maxpool_conv,
            "down4.maxpool_conv.0": self.down4.maxpool_conv[0],
            "down4.maxpool_conv.1": self.down4.maxpool_conv[1],
            "down4.maxpool_conv.1.double_conv": self.down4.maxpool_conv[1].double_conv,
            "down4.maxpool_conv.1.double_conv.0": self.down4.maxpool_conv[1].double_conv[0],
            "down4.maxpool_conv.1.double_conv.1": self.down4.maxpool_conv[1].double_conv[1],
            "down4.maxpool_conv.1.double_conv.2": self.down4.maxpool_conv[1].double_conv[2],
            "down4.maxpool_conv.1.double_conv.3": self.down4.maxpool_conv[1].double_conv[3],
            "up1": self.up1,
            "up1.center_crop": self.up1.center_crop,
            "up1.conv": self.up1.conv,
            "up1.conv.double_conv": self.up1.conv.double_conv,
            "up1.conv.double_conv.0": self.up1.conv.double_conv[0],
            "up1.conv.double_conv.1": self.up1.conv.double_conv[1],
            "up1.conv.double_conv.2": self.up1.conv.double_conv[2],
            "up1.conv.double_conv.3": self.up1.conv.double_conv[3],
            "up1.up": self.up1.up,
            "up1.relu": self.up1.relu,
            "up2": self.up2,
            "up2.center_crop": self.up2.center_crop,
            "up2.conv": self.up2.conv,
            "up2.conv.double_conv": self.up2.conv.double_conv,
            "up2.conv.double_conv.0": self.up2.conv.double_conv[0],
            "up2.conv.double_conv.1": self.up2.conv.double_conv[1],
            "up2.conv.double_conv.2": self.up2.conv.double_conv[2],
            "up2.conv.double_conv.3": self.up2.conv.double_conv[3],
            "up2.up": self.up2.up,
            "up2.relu": self.up2.relu,
            "up3": self.up3,
            "up3.center_crop": self.up3.center_crop,
            "up3.conv": self.up3.conv,
            "up3.conv.double_conv": self.up3.conv.double_conv,
            "up3.conv.double_conv.0": self.up3.conv.double_conv[0],
            "up3.conv.double_conv.1": self.up3.conv.double_conv[1],
            "up3.conv.double_conv.2": self.up3.conv.double_conv[2],
            "up3.conv.double_conv.3": self.up3.conv.double_conv[3],
            "up3.up": self.up3.up,
            "up3.relu": self.up3.relu,
            "up4": self.up4,
            "up4.center_crop": self.up4.center_crop,
            "up4.conv": self.up4.conv,
            "up4.conv.double_conv": self.up4.conv.double_conv,
            "up4.conv.double_conv.0": self.up4.conv.double_conv[0],
            "up4.conv.double_conv.1": self.up4.conv.double_conv[1],
            "up4.conv.double_conv.2": self.up4.conv.double_conv[2],
            "up4.conv.double_conv.3": self.up4.conv.double_conv[3],
            "up4.up": self.up4.up,
            "up4.relu": self.up4.relu,
            "outc": self.outc,
            "outc.conv": self.outc.conv,
        }

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def set_layers(self, layer_name, new_layer):
        if 'inc' == layer_name:
            self.inc = new_layer
            self.layer_names["inc"] = new_layer
            self.origin_layer_names["inc"] = new_layer
        elif 'inc.double_conv' == layer_name:
            self.inc.double_conv = new_layer
            self.layer_names["inc.double_conv"] = new_layer
            self.origin_layer_names["inc.double_conv"] = new_layer
        elif 'inc.double_conv.0' == layer_name:
            self.inc.double_conv[0] = new_layer
            self.layer_names["inc.double_conv.0"] = new_layer
            self.origin_layer_names["inc.double_conv.0"] = new_layer
        elif 'inc.double_conv.1' == layer_name:
            self.inc.double_conv[1] = new_layer
            self.layer_names["inc.double_conv.1"] = new_layer
            self.origin_layer_names["inc.double_conv.1"] = new_layer
        elif 'inc.double_conv.2' == layer_name:
            self.inc.double_conv[2] = new_layer
            self.layer_names["inc.double_conv.2"] = new_layer
            self.origin_layer_names["inc.double_conv.2"] = new_layer
        elif 'inc.double_conv.3' == layer_name:
            self.inc.double_conv[3] = new_layer
            self.layer_names["inc.double_conv.3"] = new_layer
            self.origin_layer_names["inc.double_conv.3"] = new_layer
        elif 'down1' == layer_name:
            self.down1 = new_layer
            self.layer_names["down1"] = new_layer
            self.origin_layer_names["down1"] = new_layer
        elif 'down1.maxpool_conv' == layer_name:
            self.down1.maxpool_conv = new_layer
            self.layer_names["down1.maxpool_conv"] = new_layer
            self.origin_layer_names["down1.maxpool_conv"] = new_layer
        elif 'down1.maxpool_conv.0' == layer_name:
            self.down1.maxpool_conv[0] = new_layer
            self.layer_names["down1.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.0"] = new_layer
        elif 'down1.maxpool_conv.1' == layer_name:
            self.down1.maxpool_conv[1] = new_layer
            self.layer_names["down1.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv' == layer_name:
            self.down1.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.0' == layer_name:
            self.down1.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.1' == layer_name:
            self.down1.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.2' == layer_name:
            self.down1.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.3' == layer_name:
            self.down1.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'down2' == layer_name:
            self.down2 = new_layer
            self.layer_names["down2"] = new_layer
            self.origin_layer_names["down2"] = new_layer
        elif 'down2.maxpool_conv' == layer_name:
            self.down2.maxpool_conv = new_layer
            self.layer_names["down2.maxpool_conv"] = new_layer
            self.origin_layer_names["down2.maxpool_conv"] = new_layer
        elif 'down2.maxpool_conv.0' == layer_name:
            self.down2.maxpool_conv[0] = new_layer
            self.layer_names["down2.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.0"] = new_layer
        elif 'down2.maxpool_conv.1' == layer_name:
            self.down2.maxpool_conv[1] = new_layer
            self.layer_names["down2.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv' == layer_name:
            self.down2.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.0' == layer_name:
            self.down2.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.1' == layer_name:
            self.down2.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.2' == layer_name:
            self.down2.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.3' == layer_name:
            self.down2.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'down3' == layer_name:
            self.down3 = new_layer
            self.layer_names["down3"] = new_layer
            self.origin_layer_names["down3"] = new_layer
        elif 'down3.maxpool_conv' == layer_name:
            self.down3.maxpool_conv = new_layer
            self.layer_names["down3.maxpool_conv"] = new_layer
            self.origin_layer_names["down3.maxpool_conv"] = new_layer
        elif 'down3.maxpool_conv.0' == layer_name:
            self.down3.maxpool_conv[0] = new_layer
            self.layer_names["down3.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.0"] = new_layer
        elif 'down3.maxpool_conv.1' == layer_name:
            self.down3.maxpool_conv[1] = new_layer
            self.layer_names["down3.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv' == layer_name:
            self.down3.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.0' == layer_name:
            self.down3.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.1' == layer_name:
            self.down3.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.2' == layer_name:
            self.down3.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.3' == layer_name:
            self.down3.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'down4' == layer_name:
            self.down4 = new_layer
            self.layer_names["down4"] = new_layer
            self.origin_layer_names["down4"] = new_layer
        elif 'down4.maxpool_conv' == layer_name:
            self.down4.maxpool_conv = new_layer
            self.layer_names["down4.maxpool_conv"] = new_layer
            self.origin_layer_names["down4.maxpool_conv"] = new_layer
        elif 'down4.maxpool_conv.0' == layer_name:
            self.down4.maxpool_conv[0] = new_layer
            self.layer_names["down4.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.0"] = new_layer
        elif 'down4.maxpool_conv.1' == layer_name:
            self.down4.maxpool_conv[1] = new_layer
            self.layer_names["down4.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv' == layer_name:
            self.down4.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.0' == layer_name:
            self.down4.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.1' == layer_name:
            self.down4.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.2' == layer_name:
            self.down4.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.3' == layer_name:
            self.down4.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'up1' == layer_name:
            self.up1 = new_layer
            self.layer_names["up1"] = new_layer
            self.origin_layer_names["up1"] = new_layer
        elif 'up1.center_crop' == layer_name:
            self.up1.center_crop = new_layer
            self.layer_names["up1.center_crop"] = new_layer
            self.origin_layer_names["up1.center_crop"] = new_layer
        elif 'up1.conv' == layer_name:
            self.up1.conv = new_layer
            self.layer_names["up1.conv"] = new_layer
            self.origin_layer_names["up1.conv"] = new_layer
        elif 'up1.conv.double_conv' == layer_name:
            self.up1.conv.double_conv = new_layer
            self.layer_names["up1.conv.double_conv"] = new_layer
            self.origin_layer_names["up1.conv.double_conv"] = new_layer
        elif 'up1.conv.double_conv.0' == layer_name:
            self.up1.conv.double_conv[0] = new_layer
            self.layer_names["up1.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.0"] = new_layer
        elif 'up1.conv.double_conv.1' == layer_name:
            self.up1.conv.double_conv[1] = new_layer
            self.layer_names["up1.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.1"] = new_layer
        elif 'up1.conv.double_conv.2' == layer_name:
            self.up1.conv.double_conv[2] = new_layer
            self.layer_names["up1.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.2"] = new_layer
        elif 'up1.conv.double_conv.3' == layer_name:
            self.up1.conv.double_conv[3] = new_layer
            self.layer_names["up1.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.3"] = new_layer
        elif 'up1.up' == layer_name:
            self.up1.up = new_layer
            self.layer_names["up1.up"] = new_layer
            self.origin_layer_names["up1.up"] = new_layer
        elif 'up1.relu' == layer_name:
            self.up1.relu = new_layer
            self.layer_names["up1.relu"] = new_layer
            self.origin_layer_names["up1.relu"] = new_layer
        elif 'up2' == layer_name:
            self.up2 = new_layer
            self.layer_names["up2"] = new_layer
            self.origin_layer_names["up2"] = new_layer
        elif 'up2.center_crop' == layer_name:
            self.up2.center_crop = new_layer
            self.layer_names["up2.center_crop"] = new_layer
            self.origin_layer_names["up2.center_crop"] = new_layer
        elif 'up2.conv' == layer_name:
            self.up2.conv = new_layer
            self.layer_names["up2.conv"] = new_layer
            self.origin_layer_names["up2.conv"] = new_layer
        elif 'up2.conv.double_conv' == layer_name:
            self.up2.conv.double_conv = new_layer
            self.layer_names["up2.conv.double_conv"] = new_layer
            self.origin_layer_names["up2.conv.double_conv"] = new_layer
        elif 'up2.conv.double_conv.0' == layer_name:
            self.up2.conv.double_conv[0] = new_layer
            self.layer_names["up2.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.0"] = new_layer
        elif 'up2.conv.double_conv.1' == layer_name:
            self.up2.conv.double_conv[1] = new_layer
            self.layer_names["up2.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.1"] = new_layer
        elif 'up2.conv.double_conv.2' == layer_name:
            self.up2.conv.double_conv[2] = new_layer
            self.layer_names["up2.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.2"] = new_layer
        elif 'up2.conv.double_conv.3' == layer_name:
            self.up2.conv.double_conv[3] = new_layer
            self.layer_names["up2.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.3"] = new_layer
        elif 'up2.up' == layer_name:
            self.up2.up = new_layer
            self.layer_names["up2.up"] = new_layer
            self.origin_layer_names["up2.up"] = new_layer
        elif 'up2.relu' == layer_name:
            self.up2.relu = new_layer
            self.layer_names["up2.relu"] = new_layer
            self.origin_layer_names["up2.relu"] = new_layer
        elif 'up3' == layer_name:
            self.up3 = new_layer
            self.layer_names["up3"] = new_layer
            self.origin_layer_names["up3"] = new_layer
        elif 'up3.center_crop' == layer_name:
            self.up3.center_crop = new_layer
            self.layer_names["up3.center_crop"] = new_layer
            self.origin_layer_names["up3.center_crop"] = new_layer
        elif 'up3.conv' == layer_name:
            self.up3.conv = new_layer
            self.layer_names["up3.conv"] = new_layer
            self.origin_layer_names["up3.conv"] = new_layer
        elif 'up3.conv.double_conv' == layer_name:
            self.up3.conv.double_conv = new_layer
            self.layer_names["up3.conv.double_conv"] = new_layer
            self.origin_layer_names["up3.conv.double_conv"] = new_layer
        elif 'up3.conv.double_conv.0' == layer_name:
            self.up3.conv.double_conv[0] = new_layer
            self.layer_names["up3.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.0"] = new_layer
        elif 'up3.conv.double_conv.1' == layer_name:
            self.up3.conv.double_conv[1] = new_layer
            self.layer_names["up3.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.1"] = new_layer
        elif 'up3.conv.double_conv.2' == layer_name:
            self.up3.conv.double_conv[2] = new_layer
            self.layer_names["up3.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.2"] = new_layer
        elif 'up3.conv.double_conv.3' == layer_name:
            self.up3.conv.double_conv[3] = new_layer
            self.layer_names["up3.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.3"] = new_layer
        elif 'up3.up' == layer_name:
            self.up3.up = new_layer
            self.layer_names["up3.up"] = new_layer
            self.origin_layer_names["up3.up"] = new_layer
        elif 'up3.relu' == layer_name:
            self.up3.relu = new_layer
            self.layer_names["up3.relu"] = new_layer
            self.origin_layer_names["up3.relu"] = new_layer
        elif 'up4' == layer_name:
            self.up4 = new_layer
            self.layer_names["up4"] = new_layer
            self.origin_layer_names["up4"] = new_layer
        elif 'up4.center_crop' == layer_name:
            self.up4.center_crop = new_layer
            self.layer_names["up4.center_crop"] = new_layer
            self.origin_layer_names["up4.center_crop"] = new_layer
        elif 'up4.conv' == layer_name:
            self.up4.conv = new_layer
            self.layer_names["up4.conv"] = new_layer
            self.origin_layer_names["up4.conv"] = new_layer
        elif 'up4.conv.double_conv' == layer_name:
            self.up4.conv.double_conv = new_layer
            self.layer_names["up4.conv.double_conv"] = new_layer
            self.origin_layer_names["up4.conv.double_conv"] = new_layer
        elif 'up4.conv.double_conv.0' == layer_name:
            self.up4.conv.double_conv[0] = new_layer
            self.layer_names["up4.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.0"] = new_layer
        elif 'up4.conv.double_conv.1' == layer_name:
            self.up4.conv.double_conv[1] = new_layer
            self.layer_names["up4.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.1"] = new_layer
        elif 'up4.conv.double_conv.2' == layer_name:
            self.up4.conv.double_conv[2] = new_layer
            self.layer_names["up4.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.2"] = new_layer
        elif 'up4.conv.double_conv.3' == layer_name:
            self.up4.conv.double_conv[3] = new_layer
            self.layer_names["up4.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.3"] = new_layer
        elif 'up4.up' == layer_name:
            self.up4.up = new_layer
            self.layer_names["up4.up"] = new_layer
            self.origin_layer_names["up4.up"] = new_layer
        elif 'up4.relu' == layer_name:
            self.up4.relu = new_layer
            self.layer_names["up4.relu"] = new_layer
            self.origin_layer_names["up4.relu"] = new_layer
        elif 'outc' == layer_name:
            self.outc = new_layer
            self.layer_names["outc"] = new_layer
            self.origin_layer_names["outc"] = new_layer
        elif 'outc.conv' == layer_name:
            self.outc.conv = new_layer
            self.layer_names["outc.conv"] = new_layer
            self.origin_layer_names["outc.conv"] = new_layer

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]


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
        logits = torch.reshape(logits, (-1, logits_shape[-1]))
        label = torch.reshape(label, (-1, label_shape[-1]))

        loss = F.cross_entropy(logits, torch.argmax(label, dim=1), reduction=self.reduction)
        return loss



if __name__ == '__main__':
    input_size = (1, 1, 572, 572)
    data = torch.tensor(np.random.randn(*input_size), dtype=torch.float32)
    model = UNetMedical(n_channels=1, n_classes=2)
    print(model(data).shape)

    t = numpy.random.randn(1, 1, 572, 572)
    a = numpy.random.randn(1, 2, 388, 388)
    t = torch.tensor(t, dtype=torch.float32)
    a = torch.tensor(a, dtype=torch.float32)
    losser = CrossEntropyWithLogits()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0005)


    # print(net(t))
    def train_step(data, label):
        data, label = data, label
        optimizer.zero_grad()
        logits = model(data)
        loss = losser(logits, label)
        loss.backward()
        optimizer.step()
        return loss.item()


    print("================================================================")
    loss_pytorch = train_step(t, a)
    print(loss_pytorch)












