import os
import shutil
import time
import numpy
import numpy as np
import torch
from torch.fx import symbolic_trace

torch.fx.wrap('len')
torch.fx.wrap('int')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.UNet.Unet import create_Unet_dataset
from configs.Unetconfig import config


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


# class CentralCrop(nn.Module):
#     """
#     Crops the central region of the images with the central_fraction.
#
#     Args:
#         central_fraction (float): Fraction of size to crop. It must be float and in range (0.0, 1.0].
#
#     Inputs:
#         - **image** (Tensor) - A 3-D tensor of shape [C, H, W], or a 4-D tensor of shape [N, C, H, W].
#
#     Outputs:
#         Tensor, 3-D or 4-D float tensor, according to the input.
#
#     Raises:
#         TypeError: If `central_fraction` is not a float.
#         ValueError: If `central_fraction` is not in range (0.0, 1.0].
#     """
#
#     def __init__(self, central_fraction):
#         super(CentralCrop, self).__init__()
#         if not isinstance(central_fraction, float):
#             raise TypeError(f"central_fraction must be a float, but got {type(central_fraction)}")
#         if not 0.0 < central_fraction <= 1.0:
#             raise ValueError(f"central_fraction must be in range (0.0, 1.0], but got {central_fraction}")
#         self.central_fraction = central_fraction
#
#     def forward(self, image):
#         image_shape = image.shape
#         # rank = len(image_shape)
#         # if rank not in (3, 4):
#         #     raise ValueError(f"Expected input rank to be 3 or 4, but got {rank}")
#
#         if self.central_fraction == 1.0:
#             return image
#
#         # if rank == 3:
#         #     c, h, w = image_shape
#         #     image = image.unsqueeze(0)
#         # else:
#         n, c, h, w = image_shape
#
#         start_h = ((h - h * self.central_fraction) / 2)
#         # start_h = int((h - h * self.central_fraction) / 2)
#         # end_h = start_h + int(h * self.central_fraction)
#         end_h = start_h + (h * self.central_fraction)
#         start_w = ((w - w * self.central_fraction) / 2)
#         # start_w = int((w - w * self.central_fraction) / 2)
#         end_w = start_w + (w * self.central_fraction)
#         # end_w = start_w + int(w * self.central_fraction)
#
#         cropped_image = image[:, :, int(start_h):int(end_h), int(start_w):int(end_w)]
#
#         # return cropped_image.squeeze(0) if rank == 3 else cropped_image
#         return cropped_image


def CentralCrop(image, central_fraction=56.0 / 64.0):
    if not isinstance(central_fraction, float):
        raise TypeError(f"central_fraction must be a float, but got {type(central_fraction)}")
    if not 0.0 < central_fraction <= 1.0:
        raise ValueError(f"central_fraction must be in range (0.0, 1.0], but got {central_fraction}")

    if central_fraction == 1.0:
        return image

    image_shape = image.shape
    n, c, h, w = image_shape

    start_h = (h - h * central_fraction) / 2
    end_h = start_h + (h * central_fraction)
    start_w = (w - w * central_fraction) / 2
    end_w = start_w + (w * central_fraction)

    cropped_image = image[:, :, int(start_h):int(end_h), int(start_w):int(end_w)]

    return cropped_image


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
        self.center_crop = CentralCrop
        self.conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.cat = torch.cat

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        with torch.no_grad():
            x2 = self.center_crop(x2, self.factor)
        x = self.cat([x1, x2], dim=1)
        return self.conv(x)


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.factor = 104.0 / 136.0
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.center_crop = CentralCrop
        self.conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.cat = torch.cat

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        with torch.no_grad():
            x2 = self.center_crop(x2, self.factor)
        x = self.cat([x1, x2], dim=1)
        return self.conv(x)


class Up3(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.factor = 200 / 280
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.center_crop = CentralCrop
        self.conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.cat = torch.cat

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        with torch.no_grad():
            x2 = self.center_crop(x2, self.factor)
        x = self.cat([x1, x2], dim=1)
        return self.conv(x)


class Up4(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.factor = 392 / 568
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.center_crop = CentralCrop
        self.conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.cat = torch.cat

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        with torch.no_grad():
            x2 = self.center_crop(x2, self.factor)
        x = self.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetMedical_torch(nn.Module):
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


# from torchmetrics import Metric


class DiceCoeff():
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


class UnetEval_torch(nn.Module):
    """
    Add Unet evaluation activation.
    """

    def __init__(self, net, need_slice=False, eval_activate="softmax"):
        super(UnetEval_torch, self).__init__()
        self.net = net
        self.need_slice = need_slice
        if eval_activate.lower() not in ("softmax", "argmax"):
            raise ValueError("eval_activate only support 'softmax' or 'argmax'")
        self.is_softmax = True
        if eval_activate == "argmax":
            self.is_softmax = False
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.net(x)
        if self.need_slice:
            out = out[-1:]
            out = out.squeeze(0)
        out = out.permute(0, 2, 3, 1)
        if self.is_softmax:
            softmax_out = self.softmax(out)
            return softmax_out
        argmax_out = torch.argmax(out, dim=-1)
        print("torch argmax_out", argmax_out.shape)
        return argmax_out


def train_eval_Unet_torch(model_old, model_new, data_dir, batch_size=1):
    # Unet Version
    config.use_deconv = True
    config.use_ds = False
    config.use_bn = False
    device = torch.device("cuda:6" if torch.cuda.is_available() else "CPU")
    config.batch_size = batch_size
    config.data_path = data_dir
    # t = numpy.random.randn(1, 1, 572, 572)
    # a = numpy.random.randn(1, 2, 388, 388)
    # t = torch.tensor(t, dtype=torch.float32)
    # a = torch.tensor(a, dtype=torch.float32)
    losser = CrossEntropyWithLogits()
    if device == "CPU":
        config.repeat = 1
    train_dataset, valid_dataset = create_Unet_dataset(config.data_path, config.repeat, config.batch_size, True,
                                                       config.cross_valid_ind,
                                                       # do_crop=config.crop,
                                                       image_size=config.image_size)
    print("train dataset size is:", train_dataset.get_dataset_size())
    train_ds = train_dataset.create_tuple_iterator(output_numpy=True)
    valid_ds = valid_dataset.create_tuple_iterator(output_numpy=True)
    optimizer1 = optim.Adam(model_old.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer2 = optim.Adam(model_new.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    testnet1 = UnetEval_torch(model_old, eval_activate=config.eval_activate.lower())
    testnet2 = UnetEval_torch(model_new, eval_activate=config.eval_activate.lower())
    time_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    f = open("log/loss_Unet_torch_" + str(device) + str(time_time) + ".txt", "w")

    metric1 = DiceCoeff()
    metric2 = DiceCoeff()
    epoch_num = 6
    per_batch = 200
    losses_ms_avg = []
    losses_ms_avg_new = []

    def train_step_old(data, label):
        data, label = data.to(device), label.to(device)
        optimizer1.zero_grad()
        logits = model_old(data)
        loss = losser(logits, label)
        loss.backward()
        optimizer1.step()
        return loss.item()

    def train_step_new(data, label):
        data, label = data.to(device), label.to(device)
        optimizer2.zero_grad()
        logits = model_new(data)
        loss = losser(logits, label)
        loss.backward()
        optimizer2.step()
        return loss.item()

    for epoch in range(epoch_num):
        nums = 0
        losses_ms = []
        losses_ms_new = []
        for data in train_ds:
            nums += data[0].shape[0]
            ta = torch.tensor(data[0], dtype=torch.float32).to(device)
            tb = torch.tensor(data[1], dtype=torch.float32).to(device)
            loss_old = train_step_old(ta, tb)
            loss_new = train_step_new(ta, tb)
            losses_ms.append(loss_old.cpu.detach.numpy())
            losses_ms_new.append(loss_new.cpu.detach.numpy())
            if nums % per_batch == 0:
                print("batch:", nums, "loss_old:", loss_old, "loss_new:", loss_new)
                f.write("batch:" + str(nums) + "loss_old:" + str(loss_old.cpu.detach.numpy()) +
                        "loss_new:" + str(loss_new.cpu.detach.numpy()) + "\n")
        losses_ms_avg.append(np.mean(losses_ms))
        losses_ms_avg_new.append(np.mean(losses_ms_new))
        print("epoch:", epoch, "loss_old:", str(np.mean(losses_ms)), "loss_new:", str(np.mean(losses_ms_new)) + "\n")
        f.write("epoch {}: ".format(epoch) + "losses_ms_avg: " + str(np.mean(losses_ms_avg)) + " losses_ms_avg_new: "
                + str(np.mean(losses_ms_avg_new)) + "\n")
        metric1.clear()
        metric2.clear()
        for tdata in valid_ds:
            t = torch.tensor(tdata[0], dtype=torch.float32).to(device)
            a = torch.tensor(tdata[1], dtype=torch.float32).to(device)
            inputs, labels = t, a
            inputs, labels = inputs.to(device), labels.to(device)  # Send tensors to the appropriate device (CPU or GPU)
            logits1 = testnet1(inputs)
            logits2 = testnet2(inputs)
            metric1.update(logits1, labels)
            metric2.update(logits2, labels)
        dice_coeff_avg1, iou_avg1 = metric1.compute()
        dice_coeff_avg2, iou_avg2 = metric2.compute()
        print("old Dice Coefficient:", dice_coeff_avg1, "IOU:", iou_avg1)
        print("new Dice Coefficient:", dice_coeff_avg2, "IOU:", iou_avg2)
        f.write("old Dice Coefficient:" + str(dice_coeff_avg1) + "IOU:" + str(iou_avg1) + "\n")
        f.write("new Dice Coefficient:" + str(dice_coeff_avg2) + "IOU:" + str(iou_avg2) + "\n")
        f.write("=============================================================\n")
    f.close()


if __name__ == '__main__':
    net = UNetMedical_torch(n_channels=1, n_classes=2)
    torch.onnx.export(net, torch.randn(1, 1, 572, 572), "unet.onnx", verbose=True)
    # symbolic_traced: torch.fx.GraphModule = symbolic_trace(net)
    # # print(symbolic_traced.graph)
    # # # print(net)
    # graph: torch.fx.Graph = torch.fx.Tracer().trace(symbolic_traced)
    # for node in graph.nodes:
    #     print("node.name", node.name)
    #     print("node.op", node.op)
    # print(symbolic_traced.code)
