import os
import shutil

import numpy
import numpy as np
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
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
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

    def forward(self, x):
        x00 = self.conv00(x)  # channel = filters[0]
        x10 = self.conv10(self.maxpool(x00))  # channel = filters[1]
        x20 = self.conv20(self.maxpool(x10))  # channel = filters[2]
        x30 = self.conv30(self.maxpool(x20))  # channel = filters[3]
        x40 = self.conv40(self.maxpool(x30))  # channel = filters[4]

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
        logits = logits.reshape(-1, logits_shape[-1])
        label = label.reshape(-1, label_shape[-1])

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


from torchmetrics import Metric


class DiceCoeff(Metric):
    """Unet Metric, return dice coefficient and IOU."""

    def __init__(self, print_res=True, show_eval=False):
        super().__init__()
        self.show_eval = show_eval
        self.print_res = print_res
        self.img_num = 0
        self.clear()

    def clear(self):
        self._dice_coeff_sum = 0
        self._iou_sum = 0
        self._samples_num = 0
        self.img_num = 0
        if self.show_eval:
            self.eval_images_path = "./draw_eval"
            if os.path.exists(self.eval_images_path):
                shutil.rmtree(self.eval_images_path)
            os.mkdir(self.eval_images_path)

    def draw_img(self, gray, index):
        """
        black：rgb(0,0,0)
        red：rgb(255,0,0)
        green：rgb(0,255,0)
        blue：rgb(0,0,255)
        cyan：rgb(0,255,255)
        cyan purple：rgb(255,0,255)
        white：rgb(255,255,255)
        """
        color = config.color
        color = np.array(color)
        np_draw = np.uint8(color[gray.astype(int)])
        return np_draw

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.view(-1).detach().cpu().numpy()
        target = target.view(-1).detach().cpu().numpy()

        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        iou = intersection / (union + 1e-6)
        dice_coeff = 2 * intersection / (np.sum(pred) + np.sum(target) + 1e-6)

        self._dice_coeff_sum += dice_coeff
        self._iou_sum += iou
        self._samples_num += 1

    def compute(self):
        dice_coeff_avg = self._dice_coeff_sum / self._samples_num
        iou_avg = self._iou_sum / self._samples_num
        return dice_coeff_avg, iou_avg


class UnetEval(nn.Module):
    """
    Add Unet evaluation activation.
    """

    def __init__(self, net, need_slice=False, eval_activate="softmax"):
        super(UnetEval, self).__init__()
        self.net = net
        self.need_slice = need_slice
        if eval_activate.lower() not in ("softmax", "argmax"):
            raise ValueError("eval_activate only support 'softmax' or 'argmax'")
        self.is_softmax = True
        if eval_activate == "argmax":
            self.is_softmax = False

    def forward(self, x):
        out = self.net(x)
        if self.need_slice:
            out = out[..., -1:]
        out = out.permute(0, 2, 3, 1)
        if self.is_softmax:
            softmax_out = F.softmax(out, dim=-1)
            return softmax_out
        argmax_out = torch.argmax(out, dim=-1)
        return argmax_out


if __name__ == '__main__':
    from src.model_utils.config import config
    config.use_deconv = True
    config.use_ds = False
    config.use_bn = False
    net = NestedUNet(in_channels=1).to("cuda")
    nt = numpy.random.randn(1, 1, 96, 96)
    na = numpy.random.randn(1, 2, 96, 96)
    t = torch.tensor(nt, dtype=torch.float32)
    a = torch.tensor(na, dtype=torch.float32)
    # from torchsummary import summary
    # summary(net, (1, 96, 96))
    # a = torch.tensor(a, dtype=torch.float32)
    # net(t)
    # print("done")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.batch_size = 1
    net = NestedUNet(in_channels=1, n_classes=2, use_deconv=config.use_deconv,
                     use_bn=config.use_bn, use_ds=config.use_ds).to(device)
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
