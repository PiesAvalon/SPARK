import math
from typing import *

import mindspore.ops as ops
import numpy as np


def weighted_layer_indices(model, layer_names):
    indices = []
    for i in range(len(layer_names)):
        layer_name = layer_names[i]
        l = model.get_layers(layer_name)
        if l is None:
            continue

        params = l.get_parameters()
        weight_count = 0
        for param in params:
            weight_count += ops.size(param)
        if weight_count > 0:
            indices.append(i)
    return indices


def assert_indices(mutated_layer_indices: List[int], depth_layer: int):
    assert max(mutated_layer_indices) < depth_layer, "Max index should be less than layer depth"
    assert min(mutated_layer_indices) >= 0, "Min index should be greater than or equal to zero"


def _shuffle_conv2d(weights, mutate_ratio):
    new_weights = []
    for val in weights:
        # val is bias if len(val.shape) == 1
        if len(val.shape) > 1:
            val_shape = val.shape
            num_of_output_channels, num_of_input_channels, filter_height, filter_width = val_shape
            mutate_output_channels = generate_permutation(num_of_output_channels, mutate_ratio)
            for output_channel in mutate_output_channels:
                copy_list = val.copy()
                copy_list = np.reshape(copy_list,
                                       (filter_width * filter_height * num_of_input_channels, num_of_output_channels))
                selected_list = copy_list[:, output_channel]
                shuffle_selected_list = shuffle(selected_list)
                copy_list[:, output_channel] = shuffle_selected_list
                val = np.reshape(copy_list,
                                 (num_of_output_channels, num_of_input_channels, filter_height, filter_width))
        new_weights.append(val)
    return new_weights


def _shuffle_conv3d(weights, mutate_ratio):
    new_weights = []
    for val in weights:
        # val is bias if len(val.shape) == 1
        if len(val.shape) > 1:
            val_shape = val.shape
            num_of_input_channels, num_of_output_channels, filter_dimension, filter_height, filter_width = val_shape
            mutate_output_channels = generate_permutation(num_of_output_channels, mutate_ratio)

            for output_channel in mutate_output_channels:
                copy_list = val.copy()
                copy_list = np.reshape(copy_list, (filter_width * filter_dimension * filter_height *
                                                   num_of_input_channels, num_of_output_channels))
                selected_list = copy_list[:, output_channel]
                shuffle_selected_list = shuffle(selected_list)
                copy_list[:, output_channel] = shuffle_selected_list
                val = np.reshape(copy_list, (num_of_input_channels, num_of_output_channels, filter_dimension,
                                             filter_height, filter_width))
        new_weights.append(val)
    return new_weights


def _shuffle_dense(weights, mutate_ratio):
    new_weights = []
    for val in weights:
        # val is bias if len(val.shape) == 1
        if len(val.shape) > 1:
            val_shape = val.shape
            output_dim, input_dim = val_shape
            mutate_output_dims = generate_permutation(output_dim, mutate_ratio)
            copy_list = val.copy()
            for output_dim in mutate_output_dims:
                selected_list = copy_list[output_dim, :]
                shuffle_selected_list = shuffle(selected_list)
                copy_list[output_dim, :] = shuffle_selected_list
            val = copy_list
        new_weights.append(val)
    return new_weights


def generate_permutation(size_of_permutation, extract_portion):
    assert extract_portion <= 1
    num_of_extraction = math.floor(size_of_permutation * extract_portion)
    permutation = np.random.permutation(size_of_permutation)
    permutation = permutation[:num_of_extraction]
    return permutation


def shuffle(a):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    length = len(a)
    permutation = np.random.permutation(length)
    index_permutation = np.arange(length)
    shuffled_a[permutation] = a[index_permutation]
    return shuffled_a
