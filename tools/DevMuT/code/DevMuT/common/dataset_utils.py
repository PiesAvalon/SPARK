import os
import numpy as np
import cv2

import mindspore.dataset as de
import mindspore.dataset.transforms as deC
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision as c_vision
import mindspore.dataset.vision.c_transforms as vision
from mindspore.common import dtype as mstype
from mindspore.dataset.transforms.transforms import Compose
from mindspore.dataset.vision.utils import Inter


class SegDataset:
    """SegDataset"""

    def __init__(self,
                 image_mean=[103.53, 116.28, 123.675],
                 image_std=[57.375, 57.120, 58.395],
                 data_file="./Pascal_VOC_deeplab/datamind/traindata0",
                 batch_size=4,
                 crop_size=513,
                 max_scale=2.0,
                 min_scale=0.5,
                 ignore_label=255,
                 num_classes=21,
                 num_readers=2,
                 num_parallel_calls=4,
                 shard_id=0,
                 shard_num=1):

        self.data_file = data_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.num_readers = num_readers
        self.num_parallel_calls = num_parallel_calls
        self.shard_id = shard_id
        self.shard_num = shard_num
        assert max_scale > min_scale

    def preprocess_(self, image, label):
        """SegDataset.preprocess_"""
        # bgr image
        image_out = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        label_out = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image_out.shape[0]), int(sc * image_out.shape[1])
        image_out = cv2.resize(image_out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label_out, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        image_out = (image_out - self.image_mean) / self.image_std
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size]

        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]

        image_out = image_out.transpose((2, 0, 1))
        image_out = image_out.copy()
        label_out = label_out.copy()
        return image_out, label_out

    def get_dataset(self, repeat=1):
        """SegDataset.get_dataset"""
        data_set = de.MindDataset(self.data_file, columns_list=["data", "label"],
                                  shuffle=True, num_parallel_workers=self.num_readers,
                                  num_shards=self.shard_num, shard_id=self.shard_id)
        transforms_list = self.preprocess_
        data_set = data_set.map(operations=transforms_list, input_columns=["data", "label"],
                                output_columns=["data", "label"],
                                num_parallel_workers=self.num_parallel_calls)
        data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
        data_set = data_set.batch(self.batch_size, drop_remainder=True)
        data_set = data_set.repeat(repeat)
        return data_set


def get_cifar10(data_dir="./cifar10", batch_size=1, is_train=True):
    if is_train:
        data_dir = os.path.join(data_dir, "cifar-10-batches-bin")
    else:
        data_dir = os.path.join(data_dir, "cifar-10-verify-bin")
    data_set = de.Cifar10Dataset(data_dir, num_shards=1, shard_id=0)
    image_size = (224, 224)
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize(image_size)  # interpolation default BILINEAR
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)
    return data_set


def get_rtpolarity(data_dir="./rt-polarity/rt-polaritydata", batch_size=1, is_train=True):
    from network.nlp.textcnn.src import dataset
    instance = dataset.MovieReview(root_dir=data_dir, maxlen=51, split=0.9)
    if is_train:
        train_dataset = instance.create_train_dataset(batch_size=batch_size)
        return train_dataset
    else:
        test_dataset = instance.create_test_dataset(batch_size=batch_size)
        return test_dataset

def get_coco2017yolov4_dataset(data_dir=r"./coco2017", batch_size=32, is_train=True):
    from network.cv.yolov4.model_utils.config import config as yolov4config
    from network.cv.yolov4.src.yolo_dataset import COCOYoloDataset
    from network.cv.yolov4.src.distributed_sampler import DistributedSampler
    from network.cv.yolov4.src.transforms import reshape_fn, MultiScaleTrans
    import multiprocessing

    yolov4config.data_dir = data_dir
    yolov4config.train_img_dir = yolov4config.data_dir + r"/train2017"
    yolov4config.val_img_dir = yolov4config.data_dir + r"/val2017"
    yolov4config.train_ann_file = yolov4config.data_dir + "/annotations/instances_train2017.json"
    yolov4config.val_ann_file = yolov4config.data_dir + "/annotations/instances_val2017.json"

    default_config = yolov4config

    if is_train:
        image_dir = yolov4config.train_img_dir
        anno_path = yolov4config.train_ann_file
        rank = yolov4config.rank
        is_training = True
        filter_crowd = True
        device_num = yolov4config.group_size
        remove_empty_anno = True
        shuffle = True
    else:
        image_dir = yolov4config.val_img_dir
        anno_path = yolov4config.val_ann_file
        rank = yolov4config.rank
        is_training = False
        device_num = yolov4config.group_size
        shuffle = False
        filter_crowd = False
        remove_empty_anno = False

    cv2.setNumThreads(0)
    yolo_dataset = COCOYoloDataset(root=image_dir, ann_file=anno_path, filter_crowd_anno=filter_crowd,
                                   remove_images_without_annotations=remove_empty_anno, is_training=is_training)
    distributed_sampler = DistributedSampler(len(yolo_dataset), device_num, rank, shuffle=shuffle)
    hwc_to_chw = c_vision.HWC2CHW()

    default_config.dataset_size = len(yolo_dataset)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / device_num)
    if is_training:
        each_multiscale = default_config.each_multiscale
        multi_scale_trans = MultiScaleTrans(default_config, device_num, each_multiscale)
        dataset_column_names = ["image", "annotation", "bbox1", "bbox2", "bbox3",
                                "gt_box1", "gt_box2", "gt_box3"]
        if device_num != 8:
            ds = de.GeneratorDataset(yolo_dataset, column_names=dataset_column_names,
                                     num_parallel_workers=min(32, num_parallel_workers),
                                     sampler=distributed_sampler)
            ds = ds.batch(batch_size, per_batch_map=multi_scale_trans, input_columns=dataset_column_names,
                          num_parallel_workers=min(32, num_parallel_workers), drop_remainder=True)
        else:
            ds = de.GeneratorDataset(yolo_dataset, column_names=dataset_column_names, sampler=distributed_sampler)
            ds = ds.batch(batch_size, per_batch_map=multi_scale_trans, input_columns=dataset_column_names,
                          num_parallel_workers=min(8, num_parallel_workers), drop_remainder=True)
    else:
        ds = de.GeneratorDataset(yolo_dataset, column_names=["image", "img_id"],
                                 sampler=distributed_sampler)
        compose_map_func = (lambda image, img_id: reshape_fn(image, img_id, default_config))
        ds = ds.map(operations=compose_map_func, input_columns=["image", "img_id"],
                    output_columns=["image", "image_shape", "img_id"],
                    num_parallel_workers=8)
        ds = ds.map(operations=hwc_to_chw, input_columns=["image"], num_parallel_workers=8)
        ds = ds.batch(batch_size, drop_remainder=True)

    return ds


def get_unetplus_dataset(data_dir="./ischanllge", batch_size=16, is_train=True):
    repeat = 40
    augment = False
    cross_val_ind = 1

    is_plus = True
    if is_plus:
        do_crop = "None"
        img_size = [96, 96]
    else:
        do_crop = [388, 388]
        img_size = [572, 572]

    from network.cv.unet.mainplus import _load_multipage_tiff
    from network.cv.unet.mainplus import _get_val_train_indices
    from network.cv.unet.mainplus import data_post_process
    from network.cv.unet.mainplus import train_data_augmentation

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
    ds_train_images = de.NumpySlicesDataset(data=train_image_data, sampler=None, shuffle=False)
    ds_train_masks = de.NumpySlicesDataset(data=train_mask_data, sampler=None, shuffle=False)
    ds_valid_images = de.NumpySlicesDataset(data=valid_image_data, sampler=None, shuffle=False)
    ds_valid_masks = de.NumpySlicesDataset(data=valid_mask_data, sampler=None, shuffle=False)
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
    train_ds = de.zip((train_image_ds, train_mask_ds))
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
    train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)

    valid_image_ds = ds_valid_images.map(input_columns="image", operations=c_trans_normalize_img)
    valid_mask_ds = ds_valid_masks.map(input_columns="mask", operations=c_trans_normalize_mask)
    valid_ds = de.zip((valid_image_ds, valid_mask_ds))
    valid_ds = valid_ds.project(columns=["image", "mask"])
    if do_crop != "None":
        valid_ds = valid_ds.map(input_columns="mask", operations=c_center_crop)
    post_process = data_post_process
    valid_ds = valid_ds.map(input_columns=["image", "mask"], operations=post_process)
    valid_ds = valid_ds.batch(batch_size=batch_size, drop_remainder=True)

    if is_train:
        return train_ds
    else:
        return valid_ds
    

def get_unet_dataset(data_dir="./ischanllge", batch_size=16, is_train=True):
    repeat = 40
    augment = False
    cross_val_ind = 1

    is_plus = False
    if is_plus:
        do_crop = "None"
        img_size = [96, 96]
    else:
        do_crop = [388, 388]
        img_size = [572, 572]

    from network.cv.unet.mainplus import _load_multipage_tiff
    from network.cv.unet.mainplus import _get_val_train_indices
    from network.cv.unet.mainplus import data_post_process
    from network.cv.unet.mainplus import train_data_augmentation

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
    ds_train_images = de.NumpySlicesDataset(data=train_image_data, sampler=None, shuffle=False)
    ds_train_masks = de.NumpySlicesDataset(data=train_mask_data, sampler=None, shuffle=False)
    ds_valid_images = de.NumpySlicesDataset(data=valid_image_data, sampler=None, shuffle=False)
    ds_valid_masks = de.NumpySlicesDataset(data=valid_mask_data, sampler=None, shuffle=False)
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
    train_ds = de.zip((train_image_ds, train_mask_ds))
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
    train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)

    valid_image_ds = ds_valid_images.map(input_columns="image", operations=c_trans_normalize_img)
    valid_mask_ds = ds_valid_masks.map(input_columns="mask", operations=c_trans_normalize_mask)
    valid_ds = de.zip((valid_image_ds, valid_mask_ds))
    valid_ds = valid_ds.project(columns=["image", "mask"])
    if do_crop != "None":
        valid_ds = valid_ds.map(input_columns="mask", operations=c_center_crop)
    post_process = data_post_process
    valid_ds = valid_ds.map(input_columns=["image", "mask"], operations=post_process)
    valid_ds = valid_ds.batch(batch_size=batch_size, drop_remainder=True)

    if is_train:
        return train_ds
    else:
        return valid_ds
    

def get_dataset(dataset_name):
    datasets_dict = {
        "cifar10": get_cifar10,  # vgg16,resnet
        "ischanllgeplus": get_unetplus_dataset,  # unetplus
        "coco2017yolov4": get_coco2017yolov4_dataset,  # yolov4
        "ischanllge": get_unet_dataset,  # unet
        "rtpolarity": get_rtpolarity,  # textcnn
        }
    return datasets_dict[dataset_name]
