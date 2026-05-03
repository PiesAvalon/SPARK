import os
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
import mindspore.ops.operations as P
from network.cv.unetplus.src.model_utils.config import config
from mindspore import context


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


def create_dataset(data_dir, repeat=400, train_batch_size=16, augment=False, cross_val_ind=1, run_distribute=False,
                   do_crop=None, img_size=None):
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
        resize_size = [int(img_size[x] * do_crop[x] / 572) for x in range(len(img_size))]
    else:
        resize_size = img_size
    c_resize_op = c_vision.Resize(size=(resize_size[0], resize_size[1]), interpolation=Inter.BILINEAR)
    c_pad = c_vision.Pad(padding=(img_size[0] - resize_size[0]) // 2)
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
        c_resize_op = c_vision.Resize(size=(img_size[0], img_size[1]), interpolation=Inter.BILINEAR)
        train_ds = train_ds.map(input_columns=["image", "mask"], operations=augment_process)
        train_ds = train_ds.map(input_columns="image", operations=c_resize_op)
        train_ds = train_ds.map(input_columns="mask", operations=c_resize_op)

    if do_crop != "None":
        train_ds = train_ds.map(input_columns="mask", operations=c_center_crop)
    post_process = data_post_process
    train_ds = train_ds.map(input_columns=["image", "mask"], operations=post_process)
    train_ds = train_ds.shuffle(repeat * 24)
    train_ds = train_ds.batch(batch_size=train_batch_size, drop_remainder=True)

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
    if True:
        x = F.ReduceMean()(x, get_axis(x))
    x = F.Cast()(x, input_dtype)
    return x


def conv_bn_relu(in_channel, out_channel, use_bn=True, kernel_size=3, stride=1, pad_mode="same", activation='relu'):
    output = [nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode=pad_mode)]
    if use_bn:
        output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)


class UnetConv2d(nn.Cell):
    """
    Convolution block in Unet, usually double conv.
    """

    def __init__(self, in_channel, out_channel, use_bn=True, num_layer=2, kernel_size=3, stride=1, padding='same'):
        super(UnetConv2d, self).__init__()
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channel = in_channel
        self.out_channel = out_channel

        convs = []
        for _ in range(num_layer):
            convs.append(conv_bn_relu(in_channel, out_channel, use_bn, kernel_size, stride, padding, "relu"))
            in_channel = out_channel

        self.convs = nn.SequentialCell(convs)

    def construct(self, inputs):
        x = self.convs(inputs)
        return x


class UnetUp(nn.Cell):
    """
    Upsampling high_feature with factor=2 and concat with low feature
    """

    def __init__(self, in_channel, out_channel, use_deconv, n_concat=2):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2d(in_channel + (n_concat - 2) * out_channel, out_channel, False)
        self.concat = P.Concat(axis=1)
        self.use_deconv = use_deconv
        if use_deconv:
            self.up_conv = nn.Conv2dTranspose(in_channel, out_channel, kernel_size=2, stride=2, pad_mode="same")
        else:
            self.up_conv = nn.Conv2d(in_channel, out_channel, 1)

    def construct(self, high_feature, *low_feature):
        if self.use_deconv:
            output = self.up_conv(high_feature)
        else:
            _, _, h, w = F.shape(high_feature)
            output = P.ResizeBilinear((h * 2, w * 2))(high_feature)
            output = self.up_conv(output)
        for feature in low_feature:
            output = self.concat((output, feature))
        return self.conv(output)


class NestedUNet(nn.Cell):
    """
    Nested unet
    """

    def __init__(self, in_channel, n_class=2, feature_scale=2, use_deconv=True, use_bn=True, use_ds=True):
        super(NestedUNet, self).__init__()
        self.in_channel = in_channel
        self.n_class = n_class
        self.feature_scale = feature_scale
        self.use_deconv = use_deconv
        self.use_bn = use_bn
        self.use_ds = use_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Down Sample
        self.maxpool0= nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.conv00 = UnetConv2d(self.in_channel, filters[0], self.use_bn)
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
        self.final1 = nn.Conv2d(filters[0], n_class, 1)
        self.final2 = nn.Conv2d(filters[0], n_class, 1)
        self.final3 = nn.Conv2d(filters[0], n_class, 1)
        self.final4 = nn.Conv2d(filters[0], n_class, 1)
        self.stack = P.Stack(axis=0)

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





    def construct(self, inputs):
        x00 = self.conv00(inputs)  # channel = filters[0]
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
            final = self.stack((final1, final2, final3, final4))
            return final
        return final4

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


def loss_ms(output, label):
    losser = CrossEntropyWithLogits()
    net_loss = losser(output, label)
    return net_loss


# def lose(logits, label, network):
#     logits = network(logits)
#     logits = F.Transpose()(logits, (0, 2, 3, 1))
#     logits = F.Cast()(logits, mindspore.float32)
#     label = F.Transpose()(label, (0, 2, 3, 1))
#     _, _, _, c = F.Shape()(label)
#     loss = F.ReduceMean()(
#         nn.SoftmaxCrossEntropyWithLogits()(F.Reshape()(logits, (-1, c)), F.Reshape()(label, (-1, c))))
#     return get_loss(loss)


if __name__ == '__main__':
    # Unet++ version
    device_id = 0
    config.batch_size = 1
    config.device_target="CPU"
    context.set_context(device_target=config.device_target, save_graphs=False, op_timeout=600, device_id=device_id)
    net = NestedUNet(in_channel=1, n_class=2, use_deconv=True,use_bn=True, use_ds=False)
    losser = CrossEntropyWithLogits()

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=config.lr, weight_decay=config.weight_decay,loss_scale=config.loss_scale)


    def forward_fn(data, label):
        logits = net(data)
        loss = losser(logits, label)
        return loss, logits


    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)


    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, optimizer(grads))
        return loss


    print("================================================================")
    nt = np.random.randn(1, 1, 96, 96)
    na = np.random.randn(1, 2, 96, 96)
    t = mindspore.Tensor(nt, dtype=mindspore.float32)
    a = mindspore.Tensor(na, dtype=mindspore.float32)

    out=net(t)
    loss_result=loss_ms(out,a)
    print(loss_result)
    print("================================================================")
