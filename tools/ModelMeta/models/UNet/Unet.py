import os
import platform
import shutil
import time
from collections import deque
from mindspore.nn.cell import Cell
import mindspore
import mindspore.dataset.vision as c_vision
import cv2
import multiprocessing
import mindspore.dataset as ds
from PIL import Image, ImageSequence
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F2
from mindspore.dataset.vision.utils import Inter
import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore.common.initializer import TruncatedNormal
from mindspore.rewrite import SymbolTree

from configs.Unetconfig import config
from mindspore import context, ops
from mindspore.nn import CentralCrop

from models.UNet import main


def preprocess_img_mask(img, mask, num_classes, img_size, augment=False, eval_resize=False):
    """
    Preprocess for multi-class dataset.
    Random crop and flip images and masks when augment is True.
    """
    if augment:
        img_size_w = int(np.random.randint(img_size[0], img_size[0] * 1.5, 1))
        img_size_h = int(np.random.randint(img_size[1], img_size[1] * 1.5, 1))
        img = cv2.resize(img, (img_size_w, img_size_h))
        mask = cv2.resize(mask, (img_size_w, img_size_h))
        dw = int(np.random.randint(0, img_size_w - img_size[0] + 1, 1))
        dh = int(np.random.randint(0, img_size_h - img_size[1] + 1, 1))
        img = img[dh:dh + img_size[1], dw:dw + img_size[0], :]
        mask = mask[dh:dh + img_size[1], dw:dw + img_size[0]]
        if np.random.random() > 0.5:
            flip_code = int(np.random.randint(-1, 2, 1))
            img = cv2.flip(img, flip_code)
            mask = cv2.flip(mask, flip_code)
    else:
        img = cv2.resize(img, img_size)
        if not eval_resize:
            mask = cv2.resize(mask, img_size)
    img = (img.astype(np.float32) - 127.5) / 127.5
    img = img.transpose(2, 0, 1)
    if num_classes == 2:
        mask = mask.astype(np.float32) / mask.max()
        mask = (mask > 0.5).astype(np.int64)
    else:
        mask = mask.astype(np.int64)
    mask = (np.arange(num_classes) == mask[..., None]).astype(int)
    mask = mask.transpose(2, 0, 1).astype(np.float32)
    return img, mask


class MultiClassDataset:
    """
    Read image and mask from original images, and split all data into train_dataset and val_dataset by `split`.
    Get image path and mask path from a tree of directories,
    images within one folder is an image, the image file named `"image.png"`, the mask file named `"mask.png"`.
    """

    def __init__(self, data_dir, repeat, is_train=False, split=0.8, shuffle=False):
        self.data_dir = data_dir
        self.is_train = is_train
        self.split = (split != 1.0)
        if self.split:
            self.img_ids = sorted(next(os.walk(self.data_dir))[1])
            self.train_ids = self.img_ids[:int(len(self.img_ids) * split)] * repeat
            self.val_ids = self.img_ids[int(len(self.img_ids) * split):]
        else:
            self.train_ids = sorted(next(os.walk(os.path.join(self.data_dir, "train")))[1])
            self.val_ids = sorted(next(os.walk(os.path.join(self.data_dir, "val")))[1])
        if shuffle:
            np.random.shuffle(self.train_ids)

    def _read_img_mask(self, img_id):
        if self.split:
            path = os.path.join(self.data_dir, img_id)
        elif self.is_train:
            path = os.path.join(self.data_dir, "train", img_id)
        else:
            path = os.path.join(self.data_dir, "val", img_id)
        img = cv2.imread(os.path.join(path, "image.png"))
        mask = cv2.imread(os.path.join(path, "mask.png"), cv2.IMREAD_GRAYSCALE)
        return img, mask

    def __getitem__(self, index):
        if self.is_train:
            return self._read_img_mask(self.train_ids[index])
        return self._read_img_mask(self.val_ids[index])

    @property
    def column_names(self):
        column_names = ['image', 'mask']
        return column_names

    def __len__(self):
        if self.is_train:
            return len(self.train_ids)
        return len(self.val_ids)


def create_multi_class_dataset(data_dir, img_size, repeat, batch_size, num_classes=2, is_train=False, augment=False,
                               eval_resize=False, split=0.8, rank=0, group_size=1, shuffle=True):
    """
    Get generator dataset for multi-class dataset.
    """
    cv2.setNumThreads(0)
    ds.config.set_enable_shared_mem(True)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(4, cores // group_size)
    mc_dataset = MultiClassDataset(data_dir, repeat, is_train, split, shuffle)
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=True,
                                  num_shards=group_size, shard_id=rank,
                                  num_parallel_workers=num_parallel_workers, python_multiprocessing=is_train)
    compose_map_func = (lambda image, mask: preprocess_img_mask(image, mask, num_classes, tuple(img_size),
                                                                augment and is_train, eval_resize))
    dataset = dataset.map(operations=compose_map_func, input_columns=mc_dataset.column_names,
                          output_columns=mc_dataset.column_names,
                          num_parallel_workers=num_parallel_workers)
    dataset = dataset.batch(batch_size, drop_remainder=is_train, num_parallel_workers=num_parallel_workers)
    return dataset


def get_axis(x):
    shape = F2.shape(x)
    length = F2.tuple_len(shape)
    perm = F2.make_range(0, length)
    return perm


def _load_multipage_tiff(path):
    """Load tiff images containing many images in the channel dimension"""
    return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])


def _get_val_train_indices(length, fold, ratio=0.8):
    assert 0 < ratio <= 1, "Train/total data ratio must be in range (0.0, 1.0]"
    np.random.seed(0)
    indices = np.arange(0, length, 1, dtype=np.int64)
    np.random.shuffle(indices)

    if fold is not None:
        indices = deque(indices)
        indices.rotate(fold * round((1.0 - ratio) * length))
        indices = np.array(indices)
        train_indices = indices[:round(ratio * len(indices))]
        val_indices = indices[round(ratio * len(indices)):]
    else:
        train_indices = indices
        val_indices = []
    return train_indices, val_indices


def train_data_augmentation(img, mask):
    h_flip = np.random.random()
    if h_flip > 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)
    v_flip = np.random.random()
    if v_flip > 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)

    left = int(np.random.uniform() * 0.3 * 572)
    right = int((1 - np.random.uniform() * 0.3) * 572)
    top = int(np.random.uniform() * 0.3 * 572)
    bottom = int((1 - np.random.uniform() * 0.3) * 572)

    img = img[top:bottom, left:right]
    mask = mask[top:bottom, left:right]

    # adjust brightness
    brightness = np.random.uniform(-0.2, 0.2)
    img = np.float32(img + brightness * np.ones(img.shape))
    img = np.clip(img, -1.0, 1.0)

    return img, mask


def data_post_process(img, mask):
    img = np.expand_dims(img, axis=0)
    mask = (mask > 0.5).astype(np.int64)
    mask = (np.arange(mask.max() + 1) == mask[..., None]).astype(int)
    mask = mask.transpose(2, 0, 1).astype(np.float32)
    return img, mask


def create_Unet_dataset(data_dir, repeat=1, batch_size=16, augment=False, cross_val_ind=1,
                        image_size=None, training=False):
    # print("image_size", image_size)
    do_crop = [388, 388]
    images = _load_multipage_tiff(os.path.join(data_dir, 'train-volume.tif'))
    masks = _load_multipage_tiff(os.path.join(data_dir, 'train-labels.tif'))

    train_indices, val_indices = _get_val_train_indices(len(images), cross_val_ind)
    train_images = images[train_indices]
    train_masks = masks[train_indices]
    train_images = np.repeat(train_images, repeat, axis=0)
    train_masks = np.repeat(train_masks, repeat, axis=0)
    val_images = images[val_indices]
    val_masks = masks[val_indices]

    train_image_data = {"image": train_images}
    train_mask_data = {"mask": train_masks}
    valid_image_data = {"image": val_images}
    valid_mask_data = {"mask": val_masks}
    ds_train_images = ds.NumpySlicesDataset(data=train_image_data, sampler=None, shuffle=False)
    ds_train_masks = ds.NumpySlicesDataset(data=train_mask_data, sampler=None, shuffle=False)
    ds_valid_images = ds.NumpySlicesDataset(data=valid_image_data, sampler=None, shuffle=False)
    ds_valid_masks = ds.NumpySlicesDataset(data=valid_mask_data, sampler=None, shuffle=False)

    if do_crop != "None":
        resize_size = [int(image_size[x] * do_crop[x] / 572) for x in range(len(image_size))]
    else:
        resize_size = image_size
    c_resize_op = c_vision.Resize(size=(resize_size[0], resize_size[1]), interpolation=Inter.BILINEAR)
    c_pad = c_vision.Pad(padding=(image_size[0] - resize_size[0]) // 2)
    c_rescale_image = c_vision.Rescale(1.0 / 127.5, -1)
    c_rescale_mask = c_vision.Rescale(1.0 / 255.0, 0)

    c_trans_normalize_img = [c_rescale_image, c_resize_op, c_pad]
    c_trans_normalize_mask = [c_rescale_mask, c_resize_op, c_pad]
    c_center_crop = c_vision.CenterCrop(size=388)

    train_image_ds = ds_train_images.map(input_columns="image", operations=c_trans_normalize_img)
    train_mask_ds = ds_train_masks.map(input_columns="mask", operations=c_trans_normalize_mask)
    train_ds = ds.zip((train_image_ds, train_mask_ds))
    train_ds = train_ds.project(columns=["image", "mask"])
    if augment:
        augment_process = train_data_augmentation
        c_resize_op = c_vision.Resize(size=(image_size[0], image_size[1]), interpolation=Inter.BILINEAR)
        train_ds = train_ds.map(input_columns=["image", "mask"], operations=augment_process)
        train_ds = train_ds.map(input_columns="image", operations=c_resize_op)
        train_ds = train_ds.map(input_columns="mask", operations=c_resize_op)

    if do_crop != "None":
        train_ds = train_ds.map(input_columns="mask", operations=c_center_crop)
    post_process = data_post_process
    train_ds = train_ds.map(input_columns=["image", "mask"], operations=post_process)
    train_ds = train_ds.shuffle(repeat * 24)
    train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)

    valid_image_ds = ds_valid_images.map(input_columns="image", operations=c_trans_normalize_img)
    valid_mask_ds = ds_valid_masks.map(input_columns="mask", operations=c_trans_normalize_mask)
    valid_ds = ds.zip((valid_image_ds, valid_mask_ds))
    valid_ds = valid_ds.project(columns=["image", "mask"])
    if do_crop != "None":
        valid_ds = valid_ds.map(input_columns="mask", operations=c_center_crop)
    post_process = data_post_process
    valid_ds = valid_ds.map(input_columns=["image", "mask"], operations=post_process)
    valid_ds = valid_ds.batch(batch_size=1, drop_remainder=True)

    return train_ds, valid_ds


def get_loss(x, weights=1.0):
    """
    Computes the weighted loss
    Args:
        weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
            inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
    """
    input_dtype = x.dtype
    x = F.Cast()(x, mstype.float32)
    weights = F.Cast()(weights, mstype.float32)
    x = F.Mul()(weights, x)
    if True and True:
        x = F.ReduceMean()(x, get_axis(x))
    # if True and not True:
    #     x = self.reduce_sum(x, self.get_axis(x))
    x = F.Cast()(x, input_dtype)
    return x


# class DoubleConv(nn.Cell):
#
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         init_value_0 = TruncatedNormal(0.06)
#         init_value_1 = TruncatedNormal(0.06)
#         if not mid_channels:
#             mid_channels = out_channels
#         self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, has_bias=True, weight_init=init_value_0,
#                                pad_mode="valid")
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, has_bias=True, weight_init=init_value_1,
#                                pad_mode="valid")
#         # self.double_conv = nn.SequentialCell(
#         #     [nn.Conv2d(in_channels, mid_channels, kernel_size=3, has_bias=True,
#         #                weight_init=init_value_0, pad_mode="valid"),
#         #      nn.ReLU(),
#         #      nn.Conv2d(mid_channels, out_channels, kernel_size=3, has_bias=True,
#         #                weight_init=init_value_1, pad_mode="valid"),
#         #      nn.ReLU()]
#         # )
#
#     def construct(self, x):
#         print("x", x.shape)
#         x = self.conv1(x)
#         print("conv1", x.shape)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         return x


class DoubleConv(nn.Cell):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        init_value_0 = TruncatedNormal(0.06)
        init_value_1 = TruncatedNormal(0.06)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.SequentialCell(
            [nn.Conv2d(in_channels, mid_channels, kernel_size=3, has_bias=True,
                       weight_init=init_value_0, pad_mode="valid"),
             nn.ReLU(),
             nn.Conv2d(mid_channels, out_channels, kernel_size=3, has_bias=True,
                       weight_init=init_value_1, pad_mode="valid"),
             nn.ReLU()]
        )

    def construct(self, x):
        return self.double_conv(x)


class Down(nn.Cell):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.SequentialCell(
            [nn.MaxPool2d(kernel_size=2, stride=2),
             DoubleConv(in_channels, out_channels)]
        )

    def construct(self, x):
        return self.maxpool_conv(x)


class Up1(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = ops.concat
        self.factor = 56.0 / 64.0
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.print_fn = F.Print()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        # print("x1.shape = ", x1.shape)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2), axis=1)
        # print("x.shape = ", x.shape)
        return self.conv(x)


class Up2(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = ops.concat
        self.factor = 104.0 / 136.0
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2), axis=1)
        return self.conv(x)


class Up3(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = ops.concat
        self.factor = 200 / 280
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.print_fn = F.Print()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2), axis=1)
        return self.conv(x)


class Up4(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = ops.concat
        self.factor = 392 / 568
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2), axis=1)
        return self.conv(x)


class OutConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        init_value = TruncatedNormal(0.06)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=True, weight_init=init_value)

    def construct(self, x):
        x = self.conv(x)
        return x


class UNetMedical(nn.Cell):
    def __init__(self, n_channels, n_classes):
        super(UNetMedical, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        # self.secell = SequentialCell([nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(), nn.Sigmoid()])
        self.down2 = Down(128, 256)
        self.down1 = Down(64, 128)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up1(1024, 512)
        self.up2 = Up2(512, 256)
        self.up3 = Up3(256, 128)
        self.up4 = Up4(128, 64)
        self.outc = OutConv(64, n_classes)
        self.relu = nn.ReLU()

    def construct(self, x):
        # x = self.secell(x)
        # print("x.shape = ", x.shape)
        x1 = self.inc(x)
        # print("x1.shape = ", x1.shape)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # print("x3.shape = ", x3.shape)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print("x5.shape = ", x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class MyLoss(Cell):
    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__()
        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = F.ReduceMean()
        self.reduce_sum = F.ReduceSum()
        self.mul = F.Mul()
        self.cast = F.Cast()

    def get_axis(self, x):
        shape = F2.shape(x)
        length = F2.tuple_len(shape)
        perm = F2.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        """
        Computes the weighted loss
        Args:
            weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
                inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
        """
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, base, target):
        raise NotImplementedError


class CrossEntropyWithLogits(MyLoss):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.softmax_cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.cast = F.Cast()

    def construct(self, logits, label):
        # NCHW->NHWC
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, mindspore.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))
        _, _, _, c = F.Shape()(label)
        loss = self.reduce_mean(
            self.softmax_cross_entropy_loss(self.reshape_fn(logits, (-1, c)), self.reshape_fn(label, (-1, c))))
        return self.get_loss(loss)


class Losser(nn.Cell):  # deprecated since we are no longer needing to use this for gradient descent
    def __init__(self, network, criterion):
        super(Losser, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss


def lose(logits, label, network):
    logits = network(logits)
    logits = F.Transpose()(logits, (0, 2, 3, 1))
    logits = F.Cast()(logits, mindspore.float32)
    label = F.Transpose()(label, (0, 2, 3, 1))
    _, _, _, c = F.Shape()(label)
    loss = F.ReduceMean()(
        nn.SoftmaxCrossEntropyWithLogits()(F.Reshape()(logits, (-1, c)), F.Reshape()(label, (-1, c))))
    return get_loss(loss)


class UnetEval(nn.Cell):
    """
    Add Unet evaluation activation.
    """

    def __init__(self, net, need_slice=False, eval_activate="softmax"):
        super(UnetEval, self).__init__()
        self.net = net
        self.need_slice = need_slice
        self.transpose = ops.Transpose()
        self.softmax = ops.Softmax(axis=-1)
        self.argmax = ops.Argmax(axis=-1)
        self.squeeze = ops.Squeeze(axis=0)
        if eval_activate.lower() not in ("softmax", "argmax"):
            raise ValueError("eval_activate only support 'softmax' or 'argmax'")
        self.is_softmax = True
        if eval_activate == "argmax":
            self.is_softmax = False

    def construct(self, x):
        # print("x", x.shape)
        out = self.net(x)
        if self.need_slice:
            out = self.squeeze(out[-1:])
        out = self.transpose(out, (0, 2, 3, 1))
        if self.is_softmax:
            softmax_out = self.softmax(out)
            return softmax_out
        argmax_out = self.argmax(out)
        return argmax_out


class dice_coeff(nn.Metric):
    """Unet Metric, return dice coefficient and IOU."""

    def __init__(self, print_res=True, show_eval=False):
        super(dice_coeff, self).__init__()
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

    def update(self, *inputs):
        # print(inputs[0].shape, inputs[1].shape)
        if len(inputs) != 2:
            raise ValueError('Need 2 inputs (y_predict, y), but got {}'.format(len(inputs)))
        y = self._convert_data(inputs[1])
        self._samples_num += y.shape[0]
        y = y.transpose(0, 2, 3, 1)
        b, h, w, c = y.shape
        if b != 1:
            raise ValueError('Batch size should be 1 when in evaluation.')
        y = y.reshape((h, w, c))
        start_index = 0
        if not config.include_background:
            y = y[:, :, 1:]
            start_index = 1

        if config.eval_activate.lower() == "softmax":
            y_softmax = np.squeeze(self._convert_data(inputs[0]), axis=0)
            if config.eval_resize:
                y_pred = []
                for i in range(start_index, config.num_classes):
                    y_pred.append(cv2.resize(np.uint8(y_softmax[:, :, i] * 255), (w, h)) / 255)
                y_pred = np.stack(y_pred, axis=-1)
            else:
                y_pred = y_softmax
                if not config.include_background:
                    y_pred = y_softmax[:, :, start_index:]

        elif config.eval_activate.lower() == "argmax":
            y_argmax = np.squeeze(self._convert_data(inputs[0]), axis=0)
            y_pred = []
            for i in range(start_index, config.num_classes):
                if config.eval_resize:
                    y_pred.append(cv2.resize(np.uint8(y_argmax == i), (w, h), interpolation=cv2.INTER_NEAREST))
                else:
                    y_pred.append(np.float32(y_argmax == i))
            y_pred = np.stack(y_pred, axis=-1)
        else:
            raise ValueError('config eval_activate should be softmax or argmax.')

        if self.show_eval:
            self.img_num += 1
            if not config.include_background:
                y_pred_draw = np.ones((h, w, c)) * 0.5
                y_pred_draw[:, :, 1:] = y_pred
                y_draw = np.ones((h, w, c)) * 0.5
                y_draw[:, :, 1:] = y
            else:
                y_pred_draw = y_pred
                y_draw = y
            y_pred_draw = y_pred_draw.argmax(-1)
            y_draw = y_draw.argmax(-1)
            cv2.imwrite(os.path.join(self.eval_images_path, "predict-" + str(self.img_num) + ".png"),
                        self.draw_img(y_pred_draw, 2))
            cv2.imwrite(os.path.join(self.eval_images_path, "mask-" + str(self.img_num) + ".png"),
                        self.draw_img(y_draw, 2))

        y_pred = y_pred.astype(np.float32)
        inter = np.dot(y_pred.flatten(), y.flatten())
        union = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2 * float(inter) / float(union + 1e-6)
        single_iou = single_dice_coeff / (2 - single_dice_coeff)
        if self.print_res:
            print("single dice coeff is: {}, IOU is: {}".format(single_dice_coeff, single_iou))
        self._dice_coeff_sum += single_dice_coeff
        self._iou_sum += single_iou

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._dice_coeff_sum / float(self._samples_num), self._iou_sum / float(self._samples_num)


def train_eval_Unet(model1, model2, data_dir, batch_size, now_time):
    config.data_path = data_dir
    config.batch_size = batch_size
    losser = CrossEntropyWithLogits()
    # t = numpy.random.randn(1, 1, 572, 572)
    # a = numpy.random.randn(1, 2, 388, 388)
    # t = ms.Tensor(t, dtype=ms.float32)
    # a = ms.Tensor(a, dtype=ms.float32)
    if mindspore.get_context("device_target") == "CPU":
        config.repeat = 1
    train_dataset, valid_dataset = create_Unet_dataset(config.data_path, config.repeat, config.batch_size, True,
                                                       config.cross_valid_ind,
                                                       # do_crop=config.crop,
                                                       image_size=config.image_size)
    print("train dataset size is:", train_dataset.get_dataset_size())
    valid_ds = valid_dataset.create_dict_iterator(output_numpy=False)

    get_optimizer(model1, model2)

    optimizer1 = nn.Adam(filter(lambda x: x.requires_grad, model1.get_parameters()), learning_rate=1e-5,
                         weight_decay=float(3e-5))
    optimizer2 = nn.Adam(filter(lambda x: x.requires_grad, model2.get_parameters()), learning_rate=1e-5,
                         weight_decay=float(3e-5))
    f = open(os.path.join("mutated_net/" + str(model1.__class__.__name__) + "/", str(now_time),
                          "/loss_" + str(platform.platform()) + str(mindspore.get_context('device_target')) + ".txt"),
             "w")

    def forward_fn1(data, label):
        logits = model1(data)
        loss = losser(logits, label)
        return loss, logits

    def forward_fn2(data, label):
        logits = model2(data)
        loss = losser(logits, label)
        return loss, logits

    grad_fn1 = mindspore.ops.value_and_grad(forward_fn1, None, optimizer1.parameters, has_aux=True)
    grad_fn2 = mindspore.ops.value_and_grad(forward_fn2, None, optimizer2.parameters, has_aux=True)

    def train_step1(data, label):
        (loss, _), grads = grad_fn1(data, label)
        loss = mindspore.ops.depend(loss, optimizer1(grads))
        return loss

    def train_step2(data, label):
        (loss, _), grads = grad_fn2(data, label)
        loss = mindspore.ops.depend(loss, optimizer2(grads))
        return loss

    epoch_num = 6
    per_batch = 200
    # Trying to fix no log for loss per batches
    losses_ms_avg = []
    losses_ms_avg_new = []
    metric1 = main.dice_coeff(show_eval=False, print_res=False)
    metric2 = main.dice_coeff(show_eval=False, print_res=False)
    testnet1 = main.UnetEval(model1, eval_activate=config.eval_activate.lower())
    testnet2 = main.UnetEval(model2, eval_activate=config.eval_activate.lower())

    for epoch in range(epoch_num):
        nums = 0
        losses_ms = []
        losses_ms_new = []
        for data in train_dataset:
            nums += data[0].shape[0]
            loss_ms1 = train_step1(data[0], data[1])
            loss_ms2 = train_step2(data[0], data[1])
            losses_ms.append(loss_ms1.asnumpy())
            losses_ms_new.append(loss_ms2.asnumpy())
            if nums % per_batch == 0:
                print("batch:" + str(nums) + " loss_ms1:" + str(loss_ms1.asnumpy()) + " loss_ms2:" + str(
                    loss_ms2.asnumpy()))
                f.write("batch:" + str(nums) + " loss_ms1:" + str(loss_ms1.asnumpy()) + " loss_ms2:" + str(
                    loss_ms2.asnumpy()) + "\n")
            # break
        losses_ms_avg.append(np.mean(losses_ms))
        losses_ms_avg_new.append(np.mean(losses_ms_new))
        print("epoch {}: ".format(epoch), "losses_ms_avg: ", str(np.mean(losses_ms_avg)), " losses_ms_avg_new: ",
              str(np.mean(losses_ms_avg_new)) + "\n")
        f.write("epoch {}: ".format(epoch) + "losses_ms_avg: " + str(np.mean(losses_ms_avg)) + " losses_ms_avg_new: "
                + str(np.mean(losses_ms_avg_new)) + "\n")
        metric1.clear()
        metric2.clear()
        for tdata in valid_ds:
            metric1.update(testnet1(mindspore.Tensor(tdata['image'], mindspore.float32)),
                           mindspore.Tensor(tdata['mask'], mindspore.float32))
            # indexes is [0, 2], using x as logits, y2 as label.
            metric2.update(testnet2(mindspore.Tensor(tdata['image'], mindspore.float32)),
                           mindspore.Tensor(tdata['mask'], mindspore.float32))
        accuracy1 = metric1.eval()
        accuracy2 = metric2.eval()
        # Send tensors to the appropriate device (CPU or GPU)
        print("old ms_Dice Coefficient", accuracy1)
        print("new ms_Dice Coefficient", accuracy2)
        f.write("old ms_Dice Coefficient" + str(accuracy1) + "\n")
        f.write("new ms_Dice Coefficient" + str(accuracy2) + "\n")


def get_optimizer(model1, model2):
    model_name = "TextCNN"
    model_old_trainable_params = model1.trainable_params()
    new_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in model_old_trainable_params:
        modelms_trainable_param.name = model_name + str(
            layer_nums) + "_" + modelms_trainable_param.name
        new_trainable_params.append(modelms_trainable_param)
        layer_nums += 1
    model_mutant_trainable_params = model2.trainable_params()
    mutant_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in model_mutant_trainable_params:
        modelms_trainable_param.name = model_name + str(
            layer_nums) + "_" + modelms_trainable_param.name
        mutant_trainable_params.append(modelms_trainable_param)
        layer_nums += 1
