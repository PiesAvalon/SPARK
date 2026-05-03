import os
import shutil
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
from mindspore import context, ops
import os
import ast
import argparse
from pprint import pprint, pformat
import yaml


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """

    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


# def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="unet_simple_config.yaml"):
#     """
#     Parse command line arguments to the configuration according to the default yaml.
#
#     Args:
#         parser: Parent parser.
#         cfg: Base configuration.
#         helper: Helper description.
#         cfg_path: Path to the default yaml config.
#     """
#     parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
#                                      parents=[parser])
#     helper = {} if helper is None else helper
#     choices = {} if choices is None else choices
#     for item in cfg:
#         if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
#             help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
#             choice = choices[item] if item in choices else None
#             if isinstance(cfg[item], bool):
#                 parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice,
#                                     help=help_description)
#             else:
#                 parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice,
#                                     help=help_description)
#     args = parser.parse_args()
#     return args
#
#
# def parse_yaml(yaml_path):
#     """
#     Parse the yaml config file.
#
#     Args:
#         yaml_path: Path to the yaml config.
#     """
#     with open(yaml_path, 'r') as fin:
#         try:
#             cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
#             cfgs = [x for x in cfgs]
#             if len(cfgs) == 1:
#                 cfg_helper = {}
#                 cfg = cfgs[0]
#                 cfg_choices = {}
#             elif len(cfgs) == 2:
#                 cfg, cfg_helper = cfgs
#                 cfg_choices = {}
#             elif len(cfgs) == 3:
#                 cfg, cfg_helper, cfg_choices = cfgs
#             else:
#                 raise ValueError(
#                     "At most 3 docs (config description for help, choices) are supported in config yaml")
#             print(cfg_helper)
#         except:
#
#             raise ValueError("Failed to parse yaml")
#     return cfg, cfg_helper, cfg_choices
#
#
# def merge(args, cfg):
#     """
#     Merge the base config from yaml file and command line arguments.
#
#     Args:
#         args: Command line arguments.
#         cfg: Base configuration.
#     """
#     args_var = vars(args)
#     for item in args_var:
#         cfg[item] = args_var[item]
#     return cfg
#
#
# def get_config():
#     """
#     Get Config according to the yaml file and cli arguments.
#     """
#     parser = argparse.ArgumentParser(description="default name", add_help=False)
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parser.add_argument("--config_path", type=str,
#                         default=os.path.join(current_dir, "unet_nested_cell_config.yaml"),
#                         help="Config file path")
#     path_args, _ = parser.parse_known_args()
#     default, helper, choices = parse_yaml(path_args.config_path)
#     args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices,
#                              cfg_path=path_args.config_path)
#     final_config = merge(args, default)
#     pprint(final_config)
#     print("Please check the above information for the configurations", flush=True)
#     return Config(final_config)

config_plus = Config({
    # Url for modelarts
    'data_url': "",
    'train_url': "",
    'checkpoint_url': "",
    # Path for local
    'data_path': "/data/pzy/Unet/archive",
    'output_path': "/cache/train",
    'load_path': "/cache/checkpoint_path/",
    'device_target': "GPU",
    'enable_profiling': False,

    # ==============================================================================
    # Training options
    'model_name': "unet_nested",
    'include_background': True,
    'run_eval': True,
    'run_distribute': False,
    'dataset': "ISBI",
    'crop': "None",
    'image_size': [96, 96],
    'train_augment': False,
    'lr': 0.0003,
    'epochs': 200,
    'repeat': 10,
    'distribute_epochs': 1600,
    'batch_size': 16,
    'distribute_batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 3,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,
    'use_ds': False,
    'use_bn': False,
    'use_deconv': True,
    'resume': False,
    'resume_ckpt': "./",
    'transfer_training': False,
    'filter_weight': ["final1.weight", "final2.weight", "final3.weight", "final4.weight"],
    'show_eval': False,
    'color': [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 255]],

    # Eval options
    'eval_metrics': "dice_coeff",
    'eval_start_epoch': 0,
    'eval_interval': 1,
    'keep_checkpoint_max': 10,
    'eval_activate': "Softmax",
    'eval_resize': False,
    'checkpoint_path': "./checkpoint/",
    'checkpoint_file_path': "ckpt_unet_nested_adam-4-75.ckpt",
    'rst_path': "./result_Files/",
    'result_path': "./preprocess_Result",

    # Export options
    'width': 96,
    'height': 96,
    'file_name': "unetplusplus",
    'file_format': "MINDIR",
})
config = config_plus


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


def create_dataset(data_dir, repeat=400, train_batch_size=16, augment=False, cross_val_ind=1, run_distribute=False, do_crop=None, img_size=None):
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
    if True and True:
        x = F.ReduceMean()(x, get_axis(x))
    # if True and not True:
    #     x = self.reduce_sum(x, self.get_axis(x))
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
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
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

    def construct(self, inputs):
        x00 = self.conv00(inputs)  # channel = filters[0]
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
            final = self.stack((final1, final2, final3, final4))
            return final
        return final4


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


class Losser(nn.Cell):
    def __init__(self, network, criterion):
        super(Losser, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
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


if __name__ == '__main__':
    # Unet++ version
    device_id = 0
    epoch_num = 9
    print("config.crop", config.crop)
    config.batch_size = 8
    context.set_context(device_target=config.device_target, save_graphs=False, op_timeout=600, device_id=device_id)
    net = NestedUNet(in_channel=1, n_class=2, use_deconv=config.use_deconv,
                     use_bn=config.use_bn, use_ds=config.use_ds)
    losser = CrossEntropyWithLogits()
    train_dataset, valid_dataset = create_dataset(config.data_path, config.repeat, config.batch_size, False,
                                                  config.cross_valid_ind,
                                                  run_distribute=False, do_crop=config.crop, img_size=config.image_size)
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=config.lr, weight_decay=config.weight_decay)
    testnet = UnetEval(net, eval_activate=config.eval_activate.lower())
    metric = dice_coeff(show_eval=True)


    # loser = Losser(net, losser)
    def forward_fn(data, label):
        logits = net(data)
        loss = losser(logits, label)
        return loss, logits


    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)


    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, optimizer(grads))
        return loss


    for epoch in range(epoch_num):
        for idx, data in enumerate(train_dataset):

            # print("data:", data[0].dtype)
            # print("label:", data[1].dtype)
            loss_ms = train_step(data[0], data[1])
            if idx % 100 == 0:
                # print("================================================================")
                print("epoch:", epoch, "idx", idx, "loss_ms", loss_ms)
            # print("================================================================")
        for tdata in valid_dataset:
            # print("y_pre", testnet(tdata[0]).shape)
            metric.clear()
            # indexes is [0, 2], using x as logits, y2 as label.
            metric.update(testnet(tdata[0]), tdata[1])
            accuracy = metric.eval()
            # print("accuracy", accuracy)
