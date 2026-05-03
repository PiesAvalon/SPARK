import tensorflow as tf
import torch
import jittor

import glob

from src.dataset.torch_dataset import TorchDataset
from src.dataset.torch_dataset import TorchMemDataset
from src.dataset.jittor_dataset import JittorDataset
from src.dataset.jittor_dataset import JittorMemDataset


def get_dataset(pattern, method, framework):
    if framework == 'TensorFlow':
        # ds_list = tf.data.Dataset.list_files(file_pattern=pattern)
        # return ds_list.map(method)
        generator = TFNormalizedGenerator(method)
        dataset = tf.data.Dataset.from_generator(generator.tf_normalized_generator, args=[pattern],
                                                 output_types=(tf.float32, tf.float32))
        return dataset
    elif framework == 'PyTorch':
        return TorchDataset(pattern, method)
    elif framework == 'Jittor':
        return JittorDataset(pattern, method)
    else:
        raise Exception('No support DL framework.')


def get_dataset_from_mem(x, y, framework):
    if x.shape[0] != y.shape[0]:
        raise Exception('X sample num is not equal to Y sample num.')
    if framework == 'TensorFlow':
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        return dataset
    elif framework == 'PyTorch':
        return TorchMemDataset(x, y)
    elif framework == 'Jittor':
        return JittorMemDataset(x, y)
    else:
        raise Exception('No support DL framework.')


class TFNormalizedGenerator:
    def __init__(self, method):
        self.method = method

    def tf_normalized_generator(self, pattern):
        # 传进来的是bytes
        # 需要格式转换
        pattern = str(pattern)
        pattern = pattern[2:-1]
        for path in glob.glob(pattern):
            x, y = self.method(path)
            yield x, y
