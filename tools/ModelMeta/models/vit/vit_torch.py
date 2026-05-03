import numpy as np
import torch
from torch import nn
import math
from easydict import EasyDict as edict
import torch
import torch.nn as nn
from importlib import import_module
from collections import namedtuple
torch.fx.wrap("randd")

MIN_NUM_PATCHES = 4


class VitConfig:
    """
    VitConfig
    """

    def __init__(self, configs):
        self.configs = configs

        # network init
        self.network_norm = nn.LayerNorm(configs.normalized_shape)
        self.network_init = torch.nn.init.normal_
        self.network_dropout_rate = 0.1
        self.network_pool = 'cls'
        self.network = ViT

        # stem
        self.stem_init = torch.nn.init.xavier_uniform_
        self.stem = VitStem

        # body
        self.body_norm = nn.LayerNorm
        self.body_drop_path_rate = 0.1
        self.body = Transformer

        # body attention
        self.attention_init = torch.nn.init.xavier_uniform_
        self.attention_activation = torch.nn.Softmax()
        self.attention_dropout_rate = 0.1
        self.attention = Attention

        # body feedforward
        self.feedforward_init = torch.nn.init.xavier_uniform_
        self.feedforward_activation = nn.GELU()
        self.feedforward_dropout_rate = 0.1
        self.feedforward = FeedForward

        # head
        self.head = origin_head
        self.head_init = torch.nn.init.xavier_uniform_
        self.head_dropout_rate = 0.1
        self.head_norm = nn.LayerNorm(configs.normalized_shape)
        self.head_activation =nn.GELU()

def randd(shape, dtype, device):
    return torch.rand(shape, dtype=dtype, device=device)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None,seed=0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob is not None and self.drop_prob > 0.:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + randd(shape, x.dtype, x.device)
            random_tensor.floor_()
            x.div_(keep_prob)
            x.mul_(random_tensor)
        return x


class BatchDense(nn.Module):
    """BatchDense module."""

    def __init__(self, in_features, out_features, has_bias=True):
        super().__init__()
        self.out_features = out_features
        self.dense = nn.Linear(in_features, out_features)#, bias=has_bias

    def forward(self, x):
        bs, seq_len, d_model = x.shape
        out = x.view(bs * seq_len, d_model)
        out = self.dense(out)
        out = out.view(bs, seq_len, self.out_features)
        return out


class ResidualCell(nn.Module):
    """Cell which implements x + f(x) function."""

    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, x):
        return self.cell(x) + x


def pretrain_head(vit_config):
    """Head for ViT pretraining."""
    d_model = vit_config.configs.d_model
    mlp_dim = vit_config.configs.mlp_dim
    num_classes = vit_config.configs.num_classes

    dropout_rate = vit_config.head_dropout_rate
    initialization = vit_config.head_init
    normalization = vit_config.head_norm
    activation = vit_config.head_activation

    dense1 = nn.Linear(d_model, mlp_dim)
    dense2 = nn.Linear(mlp_dim, num_classes)

    return nn.Sequential(
        normalization,
        dense1,
        activation,
        nn.Dropout(dropout_rate),
        dense2)


def origin_head(vit_config):
    """Head for ViT pretraining."""
    d_model = vit_config.configs.d_model
    num_classes = vit_config.configs.num_classes
    initialization = vit_config.head_init
    dense = nn.Linear(d_model, num_classes)
    return dense
    #return nn.Sequential(dense)


class VitStem(nn.Module):
    """Stem layer for ViT."""

    def __init__(self, vit_config):
        super().__init__()
        d_model = vit_config.configs.d_model
        patch_size = vit_config.configs.patch_size
        image_size = vit_config.configs.image_size
        initialization = vit_config.stem_init
        channels = 3

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches {num_patches} is too small'
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.patch_to_embedding = BatchDense(patch_dim, d_model, has_bias=True)

    def forward(self, img):
        p = self.patch_size
        bs, channels, h, w = img.shape
        x = img.view(bs, channels, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(bs, (h // p) * (w // p), channels * p * p)
        x = self.patch_to_embedding(x)
        return x


class ViT(nn.Module):
    """Vision Transformer implementation."""

    def __init__(self, vit_config):
        super().__init__()

        d_model = vit_config.configs.d_model
        patch_size = vit_config.configs.patch_size
        image_size = vit_config.configs.image_size

        initialization = vit_config.network_init
        pool = vit_config.network_pool
        pool = "mean"
        dropout_rate = vit_config.network_dropout_rate
        norm = vit_config.network_norm

        stem = vit_config.stem(vit_config)
        body = vit_config.body(vit_config)
        head = vit_config.head(vit_config)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'
        num_patches = (image_size // patch_size) ** 2

        self.pool = pool

        self.dropout = nn.Dropout(dropout_rate)
        self.stem = stem
        self.body = body
        self.head = head
        self.norm = norm

        self.add_Cascade_OPs = []

        self.in_shapes = {
            'INPUT': [1, 3, 224, 224],
            'dropout': [1, 49, 768],
             'stem.patch_to_embedding.dense': [49, 3072],
             'body.layers.0.0.cell.0': [1, 49, 768],
             'body.layers.0.0.cell.1.to_q': [49, 768],
             'body.layers.0.0.cell.1.to_k': [49, 768],
             'body.layers.0.0.cell.1.to_v': [49, 768],
             'body.layers.0.0.cell.1.to_out': [49, 768],
             'body.layers.0.0.cell.1.dropout': [1, 49, 768],
             'body.layers.0.0.cell.1.activation': [1, 12, 49, 49],
             'body.layers.0.0.cell.2': [1, 49, 768],
             'body.layers.0.1.cell.0': [1, 49, 768],
             'body.layers.0.1.cell.1.ff1.dense': [49, 768],
             'body.layers.0.1.cell.1.activation': [1, 49, 3072],
             'body.layers.0.1.cell.1.dropout': [1, 49, 768],
             'body.layers.0.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.0.1.cell.2': [1, 49, 768],
             'body.layers.1.0.cell.0': [1, 49, 768],
             'body.layers.1.0.cell.1.to_q': [49, 768],
             'body.layers.1.0.cell.1.to_k': [49, 768],
             'body.layers.1.0.cell.1.to_v': [49, 768],
             'body.layers.1.0.cell.1.to_out': [49, 768],
             'body.layers.1.0.cell.1.dropout': [1, 49, 768],
             'body.layers.1.0.cell.2': [1, 49, 768],
             'body.layers.1.1.cell.0': [1, 49, 768],
             'body.layers.1.1.cell.1.ff1.dense': [49, 768],
             'body.layers.1.1.cell.1.dropout': [1, 49, 768],
             'body.layers.1.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.1.1.cell.2': [1, 49, 768],
             'body.layers.2.0.cell.0': [1, 49, 768],
             'body.layers.2.0.cell.1.to_q': [49, 768],
             'body.layers.2.0.cell.1.to_k': [49, 768],
             'body.layers.2.0.cell.1.to_v': [49, 768],
             'body.layers.2.0.cell.1.to_out': [49, 768],
             'body.layers.2.0.cell.1.dropout': [1, 49, 768],
             'body.layers.2.0.cell.2': [1, 49, 768],
             'body.layers.2.1.cell.0': [1, 49, 768],
             'body.layers.2.1.cell.1.ff1.dense': [49, 768],
             'body.layers.2.1.cell.1.dropout': [1, 49, 768],
             'body.layers.2.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.2.1.cell.2': [1, 49, 768],
             'body.layers.3.0.cell.0': [1, 49, 768],
             'body.layers.3.0.cell.1.to_q': [49, 768],
             'body.layers.3.0.cell.1.to_k': [49, 768],
             'body.layers.3.0.cell.1.to_v': [49, 768],
             'body.layers.3.0.cell.1.to_out': [49, 768],
             'body.layers.3.0.cell.1.dropout': [1, 49, 768],
             'body.layers.3.0.cell.2': [1, 49, 768],
             'body.layers.3.1.cell.0': [1, 49, 768],
             'body.layers.3.1.cell.1.ff1.dense': [49, 768],
             'body.layers.3.1.cell.1.dropout': [1, 49, 768],
             'body.layers.3.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.3.1.cell.2': [1, 49, 768],
             'body.layers.4.0.cell.0': [1, 49, 768],
             'body.layers.4.0.cell.1.to_q': [49, 768],
             'body.layers.4.0.cell.1.to_k': [49, 768],
             'body.layers.4.0.cell.1.to_v': [49, 768],
             'body.layers.4.0.cell.1.to_out': [49, 768],
             'body.layers.4.0.cell.1.dropout': [1, 49, 768],
             'body.layers.4.0.cell.2': [1, 49, 768],
             'body.layers.4.1.cell.0': [1, 49, 768],
             'body.layers.4.1.cell.1.ff1.dense': [49, 768],
             'body.layers.4.1.cell.1.dropout': [1, 49, 768],
             'body.layers.4.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.4.1.cell.2': [1, 49, 768],
             'body.layers.5.0.cell.0': [1, 49, 768],
             'body.layers.5.0.cell.1.to_q': [49, 768],
             'body.layers.5.0.cell.1.to_k': [49, 768],
             'body.layers.5.0.cell.1.to_v': [49, 768],
             'body.layers.5.0.cell.1.to_out': [49, 768],
             'body.layers.5.0.cell.1.dropout': [1, 49, 768],
             'body.layers.5.0.cell.2': [1, 49, 768],
             'body.layers.5.1.cell.0': [1, 49, 768],
             'body.layers.5.1.cell.1.ff1.dense': [49, 768],
             'body.layers.5.1.cell.1.dropout': [1, 49, 768],
             'body.layers.5.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.5.1.cell.2': [1, 49, 768],
             'body.layers.6.0.cell.0': [1, 49, 768],
             'body.layers.6.0.cell.1.to_q': [49, 768],
             'body.layers.6.0.cell.1.to_k': [49, 768],
             'body.layers.6.0.cell.1.to_v': [49, 768],
             'body.layers.6.0.cell.1.to_out': [49, 768],
             'body.layers.6.0.cell.1.dropout': [1, 49, 768],
             'body.layers.6.0.cell.2': [1, 49, 768],
             'body.layers.6.1.cell.0': [1, 49, 768],
             'body.layers.6.1.cell.1.ff1.dense': [49, 768],
             'body.layers.6.1.cell.1.dropout': [1, 49, 768],
             'body.layers.6.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.6.1.cell.2': [1, 49, 768],
             'body.layers.7.0.cell.0': [1, 49, 768],
             'body.layers.7.0.cell.1.to_q': [49, 768],
             'body.layers.7.0.cell.1.to_k': [49, 768],
             'body.layers.7.0.cell.1.to_v': [49, 768],
             'body.layers.7.0.cell.1.to_out': [49, 768],
             'body.layers.7.0.cell.1.dropout': [1, 49, 768],
             'body.layers.7.0.cell.2': [1, 49, 768],
             'body.layers.7.1.cell.0': [1, 49, 768],
             'body.layers.7.1.cell.1.ff1.dense': [49, 768],
             'body.layers.7.1.cell.1.dropout': [1, 49, 768],
             'body.layers.7.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.7.1.cell.2': [1, 49, 768],
             'body.layers.8.0.cell.0': [1, 49, 768],
             'body.layers.8.0.cell.1.to_q': [49, 768],
             'body.layers.8.0.cell.1.to_k': [49, 768],
             'body.layers.8.0.cell.1.to_v': [49, 768],
             'body.layers.8.0.cell.1.to_out': [49, 768],
             'body.layers.8.0.cell.1.dropout': [1, 49, 768],
             'body.layers.8.0.cell.2': [1, 49, 768],
             'body.layers.8.1.cell.0': [1, 49, 768],
             'body.layers.8.1.cell.1.ff1.dense': [49, 768],
             'body.layers.8.1.cell.1.dropout': [1, 49, 768],
             'body.layers.8.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.8.1.cell.2': [1, 49, 768],
             'body.layers.9.0.cell.0': [1, 49, 768],
             'body.layers.9.0.cell.1.to_q': [49, 768],
             'body.layers.9.0.cell.1.to_k': [49, 768],
             'body.layers.9.0.cell.1.to_v': [49, 768],
             'body.layers.9.0.cell.1.to_out': [49, 768],
             'body.layers.9.0.cell.1.dropout': [1, 49, 768],
             'body.layers.9.0.cell.2': [1, 49, 768],
             'body.layers.9.1.cell.0': [1, 49, 768],
             'body.layers.9.1.cell.1.ff1.dense': [49, 768],
             'body.layers.9.1.cell.1.dropout': [1, 49, 768],
             'body.layers.9.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.9.1.cell.2': [1, 49, 768],
             'body.layers.10.0.cell.0': [1, 49, 768],
             'body.layers.10.0.cell.1.to_q': [49, 768],
             'body.layers.10.0.cell.1.to_k': [49, 768],
             'body.layers.10.0.cell.1.to_v': [49, 768],
             'body.layers.10.0.cell.1.to_out': [49, 768],
             'body.layers.10.0.cell.1.dropout': [1, 49, 768],
             'body.layers.10.0.cell.2': [1, 49, 768],
             'body.layers.10.1.cell.0': [1, 49, 768],
             'body.layers.10.1.cell.1.ff1.dense': [49, 768],
             'body.layers.10.1.cell.1.dropout': [1, 49, 768],
             'body.layers.10.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.10.1.cell.2': [1, 49, 768],
             'body.layers.11.0.cell.0': [1, 49, 768],
             'body.layers.11.0.cell.1.to_q': [49, 768],
             'body.layers.11.0.cell.1.to_k': [49, 768],
             'body.layers.11.0.cell.1.to_v': [49, 768],
             'body.layers.11.0.cell.1.to_out': [49, 768],
             'body.layers.11.0.cell.1.dropout': [1, 49, 768],
             'body.layers.11.0.cell.2': [1, 49, 768],
             'body.layers.11.1.cell.0': [1, 49, 768],
             'body.layers.11.1.cell.1.ff1.dense': [49, 768],
             'body.layers.11.1.cell.1.dropout': [1, 49, 768],
             'body.layers.11.1.cell.1.ff2.dense': [49, 3072],
             'body.layers.11.1.cell.2': [1, 49, 768],
             'head': [1, 768],
             'norm': [1, 49, 768],
             'OUTPUT': [1, 10]
        }

        self.out_shapes = {
            'INPUT': [1, 3, 224, 224],
            'dropout': [1, 49, 768],
             'stem.patch_to_embedding.dense': [49, 768],
             'body.layers.0.0.cell.0': [1, 49, 768],
             'body.layers.0.0.cell.1.to_q': [49, 768],
             'body.layers.0.0.cell.1.to_k': [49, 768],
             'body.layers.0.0.cell.1.to_v': [49, 768],
             'body.layers.0.0.cell.1.to_out': [49, 768],
             'body.layers.0.0.cell.1.dropout': [1, 49, 768],
             'body.layers.0.0.cell.1.activation': [1, 12, 49, 49],
             'body.layers.0.0.cell.2': [1, 49, 768],
             'body.layers.0.1.cell.0': [1, 49, 768],
             'body.layers.0.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.0.1.cell.1.activation': [1, 49, 3072],
             'body.layers.0.1.cell.1.dropout': [1, 49, 768],
             'body.layers.0.1.cell.1.ff2.dense': [49, 768],
             'body.layers.0.1.cell.2': [1, 49, 768],
             'body.layers.1.0.cell.0': [1, 49, 768],
             'body.layers.1.0.cell.1.to_q': [49, 768],
             'body.layers.1.0.cell.1.to_k': [49, 768],
             'body.layers.1.0.cell.1.to_v': [49, 768],
             'body.layers.1.0.cell.1.to_out': [49, 768],
             'body.layers.1.0.cell.1.dropout': [1, 49, 768],
             'body.layers.1.0.cell.2': [1, 49, 768],
             'body.layers.1.1.cell.0': [1, 49, 768],
             'body.layers.1.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.1.1.cell.1.dropout': [1, 49, 768],
             'body.layers.1.1.cell.1.ff2.dense': [49, 768],
             'body.layers.1.1.cell.2': [1, 49, 768],
             'body.layers.2.0.cell.0': [1, 49, 768],
             'body.layers.2.0.cell.1.to_q': [49, 768],
             'body.layers.2.0.cell.1.to_k': [49, 768],
             'body.layers.2.0.cell.1.to_v': [49, 768],
             'body.layers.2.0.cell.1.to_out': [49, 768],
             'body.layers.2.0.cell.1.dropout': [1, 49, 768],
             'body.layers.2.0.cell.2': [1, 49, 768],
             'body.layers.2.1.cell.0': [1, 49, 768],
             'body.layers.2.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.2.1.cell.1.dropout': [1, 49, 768],
             'body.layers.2.1.cell.1.ff2.dense': [49, 768],
             'body.layers.2.1.cell.2': [1, 49, 768],
             'body.layers.3.0.cell.0': [1, 49, 768],
             'body.layers.3.0.cell.1.to_q': [49, 768],
             'body.layers.3.0.cell.1.to_k': [49, 768],
             'body.layers.3.0.cell.1.to_v': [49, 768],
             'body.layers.3.0.cell.1.to_out': [49, 768],
             'body.layers.3.0.cell.1.dropout': [1, 49, 768],
             'body.layers.3.0.cell.2': [1, 49, 768],
             'body.layers.3.1.cell.0': [1, 49, 768],
             'body.layers.3.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.3.1.cell.1.dropout': [1, 49, 768],
             'body.layers.3.1.cell.1.ff2.dense': [49, 768],
             'body.layers.3.1.cell.2': [1, 49, 768],
             'body.layers.4.0.cell.0': [1, 49, 768],
             'body.layers.4.0.cell.1.to_q': [49, 768],
             'body.layers.4.0.cell.1.to_k': [49, 768],
             'body.layers.4.0.cell.1.to_v': [49, 768],
             'body.layers.4.0.cell.1.to_out': [49, 768],
             'body.layers.4.0.cell.1.dropout': [1, 49, 768],
             'body.layers.4.0.cell.2': [1, 49, 768],
             'body.layers.4.1.cell.0': [1, 49, 768],
             'body.layers.4.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.4.1.cell.1.dropout': [1, 49, 768],
             'body.layers.4.1.cell.1.ff2.dense': [49, 768],
             'body.layers.4.1.cell.2': [1, 49, 768],
             'body.layers.5.0.cell.0': [1, 49, 768],
             'body.layers.5.0.cell.1.to_q': [49, 768],
             'body.layers.5.0.cell.1.to_k': [49, 768],
             'body.layers.5.0.cell.1.to_v': [49, 768],
             'body.layers.5.0.cell.1.to_out': [49, 768],
             'body.layers.5.0.cell.1.dropout': [1, 49, 768],
             'body.layers.5.0.cell.2': [1, 49, 768],
             'body.layers.5.1.cell.0': [1, 49, 768],
             'body.layers.5.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.5.1.cell.1.dropout': [1, 49, 768],
             'body.layers.5.1.cell.1.ff2.dense': [49, 768],
             'body.layers.5.1.cell.2': [1, 49, 768],
             'body.layers.6.0.cell.0': [1, 49, 768],
             'body.layers.6.0.cell.1.to_q': [49, 768],
             'body.layers.6.0.cell.1.to_k': [49, 768],
             'body.layers.6.0.cell.1.to_v': [49, 768],
             'body.layers.6.0.cell.1.to_out': [49, 768],
             'body.layers.6.0.cell.1.dropout': [1, 49, 768],
             'body.layers.6.0.cell.2': [1, 49, 768],
             'body.layers.6.1.cell.0': [1, 49, 768],
             'body.layers.6.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.6.1.cell.1.dropout': [1, 49, 768],
             'body.layers.6.1.cell.1.ff2.dense': [49, 768],
             'body.layers.6.1.cell.2': [1, 49, 768],
             'body.layers.7.0.cell.0': [1, 49, 768],
             'body.layers.7.0.cell.1.to_q': [49, 768],
             'body.layers.7.0.cell.1.to_k': [49, 768],
             'body.layers.7.0.cell.1.to_v': [49, 768],
             'body.layers.7.0.cell.1.to_out': [49, 768],
             'body.layers.7.0.cell.1.dropout': [1, 49, 768],
             'body.layers.7.0.cell.2': [1, 49, 768],
             'body.layers.7.1.cell.0': [1, 49, 768],
             'body.layers.7.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.7.1.cell.1.dropout': [1, 49, 768],
             'body.layers.7.1.cell.1.ff2.dense': [49, 768],
             'body.layers.7.1.cell.2': [1, 49, 768],
             'body.layers.8.0.cell.0': [1, 49, 768],
             'body.layers.8.0.cell.1.to_q': [49, 768],
             'body.layers.8.0.cell.1.to_k': [49, 768],
             'body.layers.8.0.cell.1.to_v': [49, 768],
             'body.layers.8.0.cell.1.to_out': [49, 768],
             'body.layers.8.0.cell.1.dropout': [1, 49, 768],
             'body.layers.8.0.cell.2': [1, 49, 768],
             'body.layers.8.1.cell.0': [1, 49, 768],
             'body.layers.8.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.8.1.cell.1.dropout': [1, 49, 768],
             'body.layers.8.1.cell.1.ff2.dense': [49, 768],
             'body.layers.8.1.cell.2': [1, 49, 768],
             'body.layers.9.0.cell.0': [1, 49, 768],
             'body.layers.9.0.cell.1.to_q': [49, 768],
             'body.layers.9.0.cell.1.to_k': [49, 768],
             'body.layers.9.0.cell.1.to_v': [49, 768],
             'body.layers.9.0.cell.1.to_out': [49, 768],
             'body.layers.9.0.cell.1.dropout': [1, 49, 768],
             'body.layers.9.0.cell.2': [1, 49, 768],
             'body.layers.9.1.cell.0': [1, 49, 768],
             'body.layers.9.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.9.1.cell.1.dropout': [1, 49, 768],
             'body.layers.9.1.cell.1.ff2.dense': [49, 768],
             'body.layers.9.1.cell.2': [1, 49, 768],
             'body.layers.10.0.cell.0': [1, 49, 768],
             'body.layers.10.0.cell.1.to_q': [49, 768],
             'body.layers.10.0.cell.1.to_k': [49, 768],
             'body.layers.10.0.cell.1.to_v': [49, 768],
             'body.layers.10.0.cell.1.to_out': [49, 768],
             'body.layers.10.0.cell.1.dropout': [1, 49, 768],
             'body.layers.10.0.cell.2': [1, 49, 768],
             'body.layers.10.1.cell.0': [1, 49, 768],
             'body.layers.10.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.10.1.cell.1.dropout': [1, 49, 768],
             'body.layers.10.1.cell.1.ff2.dense': [49, 768],
             'body.layers.10.1.cell.2': [1, 49, 768],
             'body.layers.11.0.cell.0': [1, 49, 768],
             'body.layers.11.0.cell.1.to_q': [49, 768],
             'body.layers.11.0.cell.1.to_k': [49, 768],
             'body.layers.11.0.cell.1.to_v': [49, 768],
             'body.layers.11.0.cell.1.to_out': [49, 768],
             'body.layers.11.0.cell.1.dropout': [1, 49, 768],
             'body.layers.11.0.cell.2': [1, 49, 768],
             'body.layers.11.1.cell.0': [1, 49, 768],
             'body.layers.11.1.cell.1.ff1.dense': [49, 3072],
             'body.layers.11.1.cell.1.dropout': [1, 49, 768],
             'body.layers.11.1.cell.1.ff2.dense': [49, 768],
             'body.layers.11.1.cell.2': [1, 49, 768],
             'head': [1, 10],
             'norm': [1, 49, 768],
             'OUTPUT': [1, 10]}


        self.Cascade_OPs = None
        self.Basic_OPS = None

        self.layer_names = {
            "dropout": self.dropout,
            "stem": self.stem,
            "stem.patch_to_embedding": self.stem.patch_to_embedding,
            "stem.patch_to_embedding.dense": self.stem.patch_to_embedding.dense,
            "body": self.body,
            "body.layers": self.body.layers,
            "body.layers.0": self.body.layers[0],
            "body.layers.0.0": self.body.layers[0][0],
            "body.layers.0.0.cell": self.body.layers[0][0].cell,
            "body.layers.0.0.cell.0": self.body.layers[0][0].cell[0],
            "body.layers.0.0.cell.1": self.body.layers[0][0].cell[1],
            "body.layers.0.0.cell.1.to_q": self.body.layers[0][0].cell[1].to_q,
            "body.layers.0.0.cell.1.to_k": self.body.layers[0][0].cell[1].to_k,
            "body.layers.0.0.cell.1.to_v": self.body.layers[0][0].cell[1].to_v,
            "body.layers.0.0.cell.1.to_out": self.body.layers[0][0].cell[1].to_out,
            "body.layers.0.0.cell.1.dropout": self.body.layers[0][0].cell[1].dropout,
            "body.layers.0.0.cell.1.activation": self.body.layers[0][0].cell[1].activation,
            "body.layers.0.0.cell.2": self.body.layers[0][0].cell[2],
            "body.layers.0.1": self.body.layers[0][1],
            "body.layers.0.1.cell": self.body.layers[0][1].cell,
            "body.layers.0.1.cell.0": self.body.layers[0][1].cell[0],
            "body.layers.0.1.cell.1": self.body.layers[0][1].cell[1],
            "body.layers.0.1.cell.1.ff1": self.body.layers[0][1].cell[1].ff1,
            "body.layers.0.1.cell.1.ff1.dense": self.body.layers[0][1].cell[1].ff1.dense,
            "body.layers.0.1.cell.1.activation": self.body.layers[0][1].cell[1].activation,
            "body.layers.0.1.cell.1.dropout": self.body.layers[0][1].cell[1].dropout,
            "body.layers.0.1.cell.1.ff2": self.body.layers[0][1].cell[1].ff2,
            "body.layers.0.1.cell.1.ff2.dense": self.body.layers[0][1].cell[1].ff2.dense,
            "body.layers.0.1.cell.2": self.body.layers[0][1].cell[2],
            "body.layers.1": self.body.layers[1],
            "body.layers.1.0": self.body.layers[1][0],
            "body.layers.1.0.cell": self.body.layers[1][0].cell,
            "body.layers.1.0.cell.0": self.body.layers[1][0].cell[0],
            "body.layers.1.0.cell.1": self.body.layers[1][0].cell[1],
            "body.layers.1.0.cell.1.to_q": self.body.layers[1][0].cell[1].to_q,
            "body.layers.1.0.cell.1.to_k": self.body.layers[1][0].cell[1].to_k,
            "body.layers.1.0.cell.1.to_v": self.body.layers[1][0].cell[1].to_v,
            "body.layers.1.0.cell.1.to_out": self.body.layers[1][0].cell[1].to_out,
            "body.layers.1.0.cell.1.dropout": self.body.layers[1][0].cell[1].dropout,
            "body.layers.1.0.cell.2": self.body.layers[1][0].cell[2],
            "body.layers.1.1": self.body.layers[1][1],
            "body.layers.1.1.cell": self.body.layers[1][1].cell,
            "body.layers.1.1.cell.0": self.body.layers[1][1].cell[0],
            "body.layers.1.1.cell.1": self.body.layers[1][1].cell[1],
            "body.layers.1.1.cell.1.ff1": self.body.layers[1][1].cell[1].ff1,
            "body.layers.1.1.cell.1.ff1.dense": self.body.layers[1][1].cell[1].ff1.dense,
            "body.layers.1.1.cell.1.dropout": self.body.layers[1][1].cell[1].dropout,
            "body.layers.1.1.cell.1.ff2": self.body.layers[1][1].cell[1].ff2,
            "body.layers.1.1.cell.1.ff2.dense": self.body.layers[1][1].cell[1].ff2.dense,
            "body.layers.1.1.cell.2": self.body.layers[1][1].cell[2],
            "body.layers.2": self.body.layers[2],
            "body.layers.2.0": self.body.layers[2][0],
            "body.layers.2.0.cell": self.body.layers[2][0].cell,
            "body.layers.2.0.cell.0": self.body.layers[2][0].cell[0],
            "body.layers.2.0.cell.1": self.body.layers[2][0].cell[1],
            "body.layers.2.0.cell.1.to_q": self.body.layers[2][0].cell[1].to_q,
            "body.layers.2.0.cell.1.to_k": self.body.layers[2][0].cell[1].to_k,
            "body.layers.2.0.cell.1.to_v": self.body.layers[2][0].cell[1].to_v,
            "body.layers.2.0.cell.1.to_out": self.body.layers[2][0].cell[1].to_out,
            "body.layers.2.0.cell.1.dropout": self.body.layers[2][0].cell[1].dropout,
            "body.layers.2.0.cell.2": self.body.layers[2][0].cell[2],
            "body.layers.2.1": self.body.layers[2][1],
            "body.layers.2.1.cell": self.body.layers[2][1].cell,
            "body.layers.2.1.cell.0": self.body.layers[2][1].cell[0],
            "body.layers.2.1.cell.1": self.body.layers[2][1].cell[1],
            "body.layers.2.1.cell.1.ff1": self.body.layers[2][1].cell[1].ff1,
            "body.layers.2.1.cell.1.ff1.dense": self.body.layers[2][1].cell[1].ff1.dense,
            "body.layers.2.1.cell.1.dropout": self.body.layers[2][1].cell[1].dropout,
            "body.layers.2.1.cell.1.ff2": self.body.layers[2][1].cell[1].ff2,
            "body.layers.2.1.cell.1.ff2.dense": self.body.layers[2][1].cell[1].ff2.dense,
            "body.layers.2.1.cell.2": self.body.layers[2][1].cell[2],
            "body.layers.3": self.body.layers[3],
            "body.layers.3.0": self.body.layers[3][0],
            "body.layers.3.0.cell": self.body.layers[3][0].cell,
            "body.layers.3.0.cell.0": self.body.layers[3][0].cell[0],
            "body.layers.3.0.cell.1": self.body.layers[3][0].cell[1],
            "body.layers.3.0.cell.1.to_q": self.body.layers[3][0].cell[1].to_q,
            "body.layers.3.0.cell.1.to_k": self.body.layers[3][0].cell[1].to_k,
            "body.layers.3.0.cell.1.to_v": self.body.layers[3][0].cell[1].to_v,
            "body.layers.3.0.cell.1.to_out": self.body.layers[3][0].cell[1].to_out,
            "body.layers.3.0.cell.1.dropout": self.body.layers[3][0].cell[1].dropout,
            "body.layers.3.0.cell.2": self.body.layers[3][0].cell[2],
            "body.layers.3.1": self.body.layers[3][1],
            "body.layers.3.1.cell": self.body.layers[3][1].cell,
            "body.layers.3.1.cell.0": self.body.layers[3][1].cell[0],
            "body.layers.3.1.cell.1": self.body.layers[3][1].cell[1],
            "body.layers.3.1.cell.1.ff1": self.body.layers[3][1].cell[1].ff1,
            "body.layers.3.1.cell.1.ff1.dense": self.body.layers[3][1].cell[1].ff1.dense,
            "body.layers.3.1.cell.1.dropout": self.body.layers[3][1].cell[1].dropout,
            "body.layers.3.1.cell.1.ff2": self.body.layers[3][1].cell[1].ff2,
            "body.layers.3.1.cell.1.ff2.dense": self.body.layers[3][1].cell[1].ff2.dense,
            "body.layers.3.1.cell.2": self.body.layers[3][1].cell[2],
            "body.layers.4": self.body.layers[4],
            "body.layers.4.0": self.body.layers[4][0],
            "body.layers.4.0.cell": self.body.layers[4][0].cell,
            "body.layers.4.0.cell.0": self.body.layers[4][0].cell[0],
            "body.layers.4.0.cell.1": self.body.layers[4][0].cell[1],
            "body.layers.4.0.cell.1.to_q": self.body.layers[4][0].cell[1].to_q,
            "body.layers.4.0.cell.1.to_k": self.body.layers[4][0].cell[1].to_k,
            "body.layers.4.0.cell.1.to_v": self.body.layers[4][0].cell[1].to_v,
            "body.layers.4.0.cell.1.to_out": self.body.layers[4][0].cell[1].to_out,
            "body.layers.4.0.cell.1.dropout": self.body.layers[4][0].cell[1].dropout,
            "body.layers.4.0.cell.2": self.body.layers[4][0].cell[2],
            "body.layers.4.1": self.body.layers[4][1],
            "body.layers.4.1.cell": self.body.layers[4][1].cell,
            "body.layers.4.1.cell.0": self.body.layers[4][1].cell[0],
            "body.layers.4.1.cell.1": self.body.layers[4][1].cell[1],
            "body.layers.4.1.cell.1.ff1": self.body.layers[4][1].cell[1].ff1,
            "body.layers.4.1.cell.1.ff1.dense": self.body.layers[4][1].cell[1].ff1.dense,
            "body.layers.4.1.cell.1.dropout": self.body.layers[4][1].cell[1].dropout,
            "body.layers.4.1.cell.1.ff2": self.body.layers[4][1].cell[1].ff2,
            "body.layers.4.1.cell.1.ff2.dense": self.body.layers[4][1].cell[1].ff2.dense,
            "body.layers.4.1.cell.2": self.body.layers[4][1].cell[2],
            "body.layers.5": self.body.layers[5],
            "body.layers.5.0": self.body.layers[5][0],
            "body.layers.5.0.cell": self.body.layers[5][0].cell,
            "body.layers.5.0.cell.0": self.body.layers[5][0].cell[0],
            "body.layers.5.0.cell.1": self.body.layers[5][0].cell[1],
            "body.layers.5.0.cell.1.to_q": self.body.layers[5][0].cell[1].to_q,
            "body.layers.5.0.cell.1.to_k": self.body.layers[5][0].cell[1].to_k,
            "body.layers.5.0.cell.1.to_v": self.body.layers[5][0].cell[1].to_v,
            "body.layers.5.0.cell.1.to_out": self.body.layers[5][0].cell[1].to_out,
            "body.layers.5.0.cell.1.dropout": self.body.layers[5][0].cell[1].dropout,
            "body.layers.5.0.cell.2": self.body.layers[5][0].cell[2],
            "body.layers.5.1": self.body.layers[5][1],
            "body.layers.5.1.cell": self.body.layers[5][1].cell,
            "body.layers.5.1.cell.0": self.body.layers[5][1].cell[0],
            "body.layers.5.1.cell.1": self.body.layers[5][1].cell[1],
            "body.layers.5.1.cell.1.ff1": self.body.layers[5][1].cell[1].ff1,
            "body.layers.5.1.cell.1.ff1.dense": self.body.layers[5][1].cell[1].ff1.dense,
            "body.layers.5.1.cell.1.dropout": self.body.layers[5][1].cell[1].dropout,
            "body.layers.5.1.cell.1.ff2": self.body.layers[5][1].cell[1].ff2,
            "body.layers.5.1.cell.1.ff2.dense": self.body.layers[5][1].cell[1].ff2.dense,
            "body.layers.5.1.cell.2": self.body.layers[5][1].cell[2],
            "body.layers.6": self.body.layers[6],
            "body.layers.6.0": self.body.layers[6][0],
            "body.layers.6.0.cell": self.body.layers[6][0].cell,
            "body.layers.6.0.cell.0": self.body.layers[6][0].cell[0],
            "body.layers.6.0.cell.1": self.body.layers[6][0].cell[1],
            "body.layers.6.0.cell.1.to_q": self.body.layers[6][0].cell[1].to_q,
            "body.layers.6.0.cell.1.to_k": self.body.layers[6][0].cell[1].to_k,
            "body.layers.6.0.cell.1.to_v": self.body.layers[6][0].cell[1].to_v,
            "body.layers.6.0.cell.1.to_out": self.body.layers[6][0].cell[1].to_out,
            "body.layers.6.0.cell.1.dropout": self.body.layers[6][0].cell[1].dropout,
            "body.layers.6.0.cell.2": self.body.layers[6][0].cell[2],
            "body.layers.6.1": self.body.layers[6][1],
            "body.layers.6.1.cell": self.body.layers[6][1].cell,
            "body.layers.6.1.cell.0": self.body.layers[6][1].cell[0],
            "body.layers.6.1.cell.1": self.body.layers[6][1].cell[1],
            "body.layers.6.1.cell.1.ff1": self.body.layers[6][1].cell[1].ff1,
            "body.layers.6.1.cell.1.ff1.dense": self.body.layers[6][1].cell[1].ff1.dense,
            "body.layers.6.1.cell.1.dropout": self.body.layers[6][1].cell[1].dropout,
            "body.layers.6.1.cell.1.ff2": self.body.layers[6][1].cell[1].ff2,
            "body.layers.6.1.cell.1.ff2.dense": self.body.layers[6][1].cell[1].ff2.dense,
            "body.layers.6.1.cell.2": self.body.layers[6][1].cell[2],
            "body.layers.7": self.body.layers[7],
            "body.layers.7.0": self.body.layers[7][0],
            "body.layers.7.0.cell": self.body.layers[7][0].cell,
            "body.layers.7.0.cell.0": self.body.layers[7][0].cell[0],
            "body.layers.7.0.cell.1": self.body.layers[7][0].cell[1],
            "body.layers.7.0.cell.1.to_q": self.body.layers[7][0].cell[1].to_q,
            "body.layers.7.0.cell.1.to_k": self.body.layers[7][0].cell[1].to_k,
            "body.layers.7.0.cell.1.to_v": self.body.layers[7][0].cell[1].to_v,
            "body.layers.7.0.cell.1.to_out": self.body.layers[7][0].cell[1].to_out,
            "body.layers.7.0.cell.1.dropout": self.body.layers[7][0].cell[1].dropout,
            "body.layers.7.0.cell.2": self.body.layers[7][0].cell[2],
            "body.layers.7.1": self.body.layers[7][1],
            "body.layers.7.1.cell": self.body.layers[7][1].cell,
            "body.layers.7.1.cell.0": self.body.layers[7][1].cell[0],
            "body.layers.7.1.cell.1": self.body.layers[7][1].cell[1],
            "body.layers.7.1.cell.1.ff1": self.body.layers[7][1].cell[1].ff1,
            "body.layers.7.1.cell.1.ff1.dense": self.body.layers[7][1].cell[1].ff1.dense,
            "body.layers.7.1.cell.1.dropout": self.body.layers[7][1].cell[1].dropout,
            "body.layers.7.1.cell.1.ff2": self.body.layers[7][1].cell[1].ff2,
            "body.layers.7.1.cell.1.ff2.dense": self.body.layers[7][1].cell[1].ff2.dense,
            "body.layers.7.1.cell.2": self.body.layers[7][1].cell[2],
            "body.layers.8": self.body.layers[8],
            "body.layers.8.0": self.body.layers[8][0],
            "body.layers.8.0.cell": self.body.layers[8][0].cell,
            "body.layers.8.0.cell.0": self.body.layers[8][0].cell[0],
            "body.layers.8.0.cell.1": self.body.layers[8][0].cell[1],
            "body.layers.8.0.cell.1.to_q": self.body.layers[8][0].cell[1].to_q,
            "body.layers.8.0.cell.1.to_k": self.body.layers[8][0].cell[1].to_k,
            "body.layers.8.0.cell.1.to_v": self.body.layers[8][0].cell[1].to_v,
            "body.layers.8.0.cell.1.to_out": self.body.layers[8][0].cell[1].to_out,
            "body.layers.8.0.cell.1.dropout": self.body.layers[8][0].cell[1].dropout,
            "body.layers.8.0.cell.2": self.body.layers[8][0].cell[2],
            "body.layers.8.1": self.body.layers[8][1],
            "body.layers.8.1.cell": self.body.layers[8][1].cell,
            "body.layers.8.1.cell.0": self.body.layers[8][1].cell[0],
            "body.layers.8.1.cell.1": self.body.layers[8][1].cell[1],
            "body.layers.8.1.cell.1.ff1": self.body.layers[8][1].cell[1].ff1,
            "body.layers.8.1.cell.1.ff1.dense": self.body.layers[8][1].cell[1].ff1.dense,
            "body.layers.8.1.cell.1.dropout": self.body.layers[8][1].cell[1].dropout,
            "body.layers.8.1.cell.1.ff2": self.body.layers[8][1].cell[1].ff2,
            "body.layers.8.1.cell.1.ff2.dense": self.body.layers[8][1].cell[1].ff2.dense,
            "body.layers.8.1.cell.2": self.body.layers[8][1].cell[2],
            "body.layers.9": self.body.layers[9],
            "body.layers.9.0": self.body.layers[9][0],
            "body.layers.9.0.cell": self.body.layers[9][0].cell,
            "body.layers.9.0.cell.0": self.body.layers[9][0].cell[0],
            "body.layers.9.0.cell.1": self.body.layers[9][0].cell[1],
            "body.layers.9.0.cell.1.to_q": self.body.layers[9][0].cell[1].to_q,
            "body.layers.9.0.cell.1.to_k": self.body.layers[9][0].cell[1].to_k,
            "body.layers.9.0.cell.1.to_v": self.body.layers[9][0].cell[1].to_v,
            "body.layers.9.0.cell.1.to_out": self.body.layers[9][0].cell[1].to_out,
            "body.layers.9.0.cell.1.dropout": self.body.layers[9][0].cell[1].dropout,
            "body.layers.9.0.cell.2": self.body.layers[9][0].cell[2],
            "body.layers.9.1": self.body.layers[9][1],
            "body.layers.9.1.cell": self.body.layers[9][1].cell,
            "body.layers.9.1.cell.0": self.body.layers[9][1].cell[0],
            "body.layers.9.1.cell.1": self.body.layers[9][1].cell[1],
            "body.layers.9.1.cell.1.ff1": self.body.layers[9][1].cell[1].ff1,
            "body.layers.9.1.cell.1.ff1.dense": self.body.layers[9][1].cell[1].ff1.dense,
            "body.layers.9.1.cell.1.dropout": self.body.layers[9][1].cell[1].dropout,
            "body.layers.9.1.cell.1.ff2": self.body.layers[9][1].cell[1].ff2,
            "body.layers.9.1.cell.1.ff2.dense": self.body.layers[9][1].cell[1].ff2.dense,
            "body.layers.9.1.cell.2": self.body.layers[9][1].cell[2],
            "body.layers.10": self.body.layers[10],
            "body.layers.10.0": self.body.layers[10][0],
            "body.layers.10.0.cell": self.body.layers[10][0].cell,
            "body.layers.10.0.cell.0": self.body.layers[10][0].cell[0],
            "body.layers.10.0.cell.1": self.body.layers[10][0].cell[1],
            "body.layers.10.0.cell.1.to_q": self.body.layers[10][0].cell[1].to_q,
            "body.layers.10.0.cell.1.to_k": self.body.layers[10][0].cell[1].to_k,
            "body.layers.10.0.cell.1.to_v": self.body.layers[10][0].cell[1].to_v,
            "body.layers.10.0.cell.1.to_out": self.body.layers[10][0].cell[1].to_out,
            "body.layers.10.0.cell.1.dropout": self.body.layers[10][0].cell[1].dropout,
            "body.layers.10.0.cell.2": self.body.layers[10][0].cell[2],
            "body.layers.10.1": self.body.layers[10][1],
            "body.layers.10.1.cell": self.body.layers[10][1].cell,
            "body.layers.10.1.cell.0": self.body.layers[10][1].cell[0],
            "body.layers.10.1.cell.1": self.body.layers[10][1].cell[1],
            "body.layers.10.1.cell.1.ff1": self.body.layers[10][1].cell[1].ff1,
            "body.layers.10.1.cell.1.ff1.dense": self.body.layers[10][1].cell[1].ff1.dense,
            "body.layers.10.1.cell.1.dropout": self.body.layers[10][1].cell[1].dropout,
            "body.layers.10.1.cell.1.ff2": self.body.layers[10][1].cell[1].ff2,
            "body.layers.10.1.cell.1.ff2.dense": self.body.layers[10][1].cell[1].ff2.dense,
            "body.layers.10.1.cell.2": self.body.layers[10][1].cell[2],
            "body.layers.11": self.body.layers[11],
            "body.layers.11.0": self.body.layers[11][0],
            "body.layers.11.0.cell": self.body.layers[11][0].cell,
            "body.layers.11.0.cell.0": self.body.layers[11][0].cell[0],
            "body.layers.11.0.cell.1": self.body.layers[11][0].cell[1],
            "body.layers.11.0.cell.1.to_q": self.body.layers[11][0].cell[1].to_q,
            "body.layers.11.0.cell.1.to_k": self.body.layers[11][0].cell[1].to_k,
            "body.layers.11.0.cell.1.to_v": self.body.layers[11][0].cell[1].to_v,
            "body.layers.11.0.cell.1.to_out": self.body.layers[11][0].cell[1].to_out,
            "body.layers.11.0.cell.1.dropout": self.body.layers[11][0].cell[1].dropout,
            "body.layers.11.0.cell.2": self.body.layers[11][0].cell[2],
            "body.layers.11.1": self.body.layers[11][1],
            "body.layers.11.1.cell": self.body.layers[11][1].cell,
            "body.layers.11.1.cell.0": self.body.layers[11][1].cell[0],
            "body.layers.11.1.cell.1": self.body.layers[11][1].cell[1],
            "body.layers.11.1.cell.1.ff1": self.body.layers[11][1].cell[1].ff1,
            "body.layers.11.1.cell.1.ff1.dense": self.body.layers[11][1].cell[1].ff1.dense,
            "body.layers.11.1.cell.1.dropout": self.body.layers[11][1].cell[1].dropout,
            "body.layers.11.1.cell.1.ff2": self.body.layers[11][1].cell[1].ff2,
            "body.layers.11.1.cell.1.ff2.dense": self.body.layers[11][1].cell[1].ff2.dense,
            "body.layers.11.1.cell.2": self.body.layers[11][1].cell[2],
            "head": self.head,
            "norm": self.norm,
        }

        self.orders = {
            'stem.patch_to_embedding.dense': ["INPUT", "dropout"],
            'dropout': ["stem.patch_to_embedding.dense", "body.layers.0.0.cell.0"],
            'body.layers.0.0.cell.0': ["dropout", ["body.layers.0.0.cell.1.to_q", "body.layers.0.0.cell.1.to_k",
                                                   "body.layers.0.0.cell.1.to_v"]],
            'body.layers.0.0.cell.1.to_q': ["body.layers.0.0.cell.0", "body.layers.0.0.cell.1.activation"],
            'body.layers.0.0.cell.1.to_k': ["body.layers.0.0.cell.0", "body.layers.0.0.cell.1.activation"],
            'body.layers.0.0.cell.1.to_v': ["body.layers.0.0.cell.0", "body.layers.0.0.cell.1.to_out"],
            'body.layers.0.0.cell.1.activation': [["body.layers.0.0.cell.1.to_k", "body.layers.0.0.cell.1.to_q"],
                                                  "body.layers.0.0.cell.1.to_out"],
            'body.layers.0.0.cell.1.to_out': [["body.layers.0.0.cell.1.activation", "body.layers.0.0.cell.1.to_v"],
                                              "body.layers.0.0.cell.1.dropout"],
            'body.layers.0.0.cell.1.dropout': ["body.layers.0.0.cell.1.to_out", "body.layers.0.0.cell.2"],
            'body.layers.0.0.cell.2': ["body.layers.0.0.cell.1.dropout", "body.layers.0.1.cell.0"],
            'body.layers.0.1.cell.0': ["body.layers.0.0.cell.2", "body.layers.0.1.cell.1.ff1.dense"],
            'body.layers.0.1.cell.1.ff1.dense': ["body.layers.0.1.cell.0", "body.layers.0.1.cell.1.activation"],
            'body.layers.0.1.cell.1.activation': ["body.layers.0.1.cell.1.ff1.dense", "body.layers.0.1.cell.1.dropout"],
            'body.layers.0.1.cell.1.dropout': ["body.layers.0.1.cell.1.activation", "body.layers.0.1.cell.1.ff2.dense"],
            'body.layers.0.1.cell.1.ff2.dense': ["body.layers.0.1.cell.1.dropout", "body.layers.0.1.cell.2"],
            'body.layers.0.1.cell.2': ["body.layers.0.1.cell.1.ff2.dense", "body.layers.1.0.cell.0"],
            'body.layers.1.0.cell.0': ["body.layers.0.1.cell.2",
                                       ["body.layers.1.0.cell.1.to_q", "body.layers.1.0.cell.1.to_k",
                                        "body.layers.1.0.cell.1.to_v"]],
            'body.layers.1.0.cell.1.to_q': ["body.layers.1.0.cell.0", "body.layers.1.0.cell.1.activation"],
            'body.layers.1.0.cell.1.to_k': ["body.layers.1.0.cell.0", "body.layers.1.0.cell.1.activation"],
            'body.layers.1.0.cell.1.to_v': ["body.layers.1.0.cell.0", "body.layers.1.0.cell.1.to_out"],
            'body.layers.1.0.cell.1.activation': [["body.layers.1.0.cell.1.to_k", "body.layers.1.0.cell.1.to_q"],
                                                  "body.layers.1.0.cell.1.to_out"],
            'body.layers.1.0.cell.1.to_out': [["body.layers.1.0.cell.1.activation", "body.layers.1.0.cell.1.to_v"],
                                              "body.layers.1.0.cell.1.dropout"],
            'body.layers.1.0.cell.1.dropout': ["body.layers.1.0.cell.1.to_out", "body.layers.1.0.cell.2"],
            'body.layers.1.0.cell.2': ["body.layers.1.0.cell.1.dropout", "body.layers.1.1.cell.0"],
            'body.layers.1.1.cell.0': ["body.layers.1.0.cell.2", "body.layers.1.1.cell.1.ff1.dense"],
            'body.layers.1.1.cell.1.ff1.dense': ["body.layers.1.1.cell.0", "body.layers.1.1.cell.1.activation"],
            'body.layers.1.1.cell.1.activation': ["body.layers.1.1.cell.1.ff1.dense", "body.layers.1.1.cell.1.dropout"],
            'body.layers.1.1.cell.1.dropout': ["body.layers.1.1.cell.1.activation", "body.layers.1.1.cell.1.ff2.dense"],
            'body.layers.1.1.cell.1.ff2.dense': ["body.layers.1.1.cell.1.dropout", "body.layers.1.1.cell.2"],
            'body.layers.1.1.cell.2': ["body.layers.1.1.cell.1.ff2.dense", "body.layers.2.0.cell.0"],
            'body.layers.2.0.cell.0': ["body.layers.1.1.cell.2",
                                       ["body.layers.2.0.cell.1.to_q", "body.layers.2.0.cell.1.to_k",
                                        "body.layers.2.0.cell.1.to_v"]],
            'body.layers.2.0.cell.1.to_q': ["body.layers.2.0.cell.0", "body.layers.2.0.cell.1.activation"],
            'body.layers.2.0.cell.1.to_k': ["body.layers.2.0.cell.0", "body.layers.2.0.cell.1.activation"],
            'body.layers.2.0.cell.1.to_v': ["body.layers.2.0.cell.0", "body.layers.2.0.cell.1.to_out"],
            'body.layers.2.0.cell.1.activation': [["body.layers.2.0.cell.1.to_k", "body.layers.2.0.cell.1.to_q"],
                                                  "body.layers.2.0.cell.1.to_out"],
            'body.layers.2.0.cell.1.to_out': [["body.layers.2.0.cell.1.activation", "body.layers.2.0.cell.1.to_v"],
                                              "body.layers.2.0.cell.1.dropout"],
            'body.layers.2.0.cell.1.dropout': ["body.layers.2.0.cell.1.to_out", "body.layers.2.0.cell.2"],
            'body.layers.2.0.cell.2': ["body.layers.2.0.cell.1.dropout", "body.layers.2.1.cell.0"],
            'body.layers.2.1.cell.0': ["body.layers.2.0.cell.2", "body.layers.2.1.cell.1.ff1.dense"],
            'body.layers.2.1.cell.1.ff1.dense': ["body.layers.2.1.cell.0", "body.layers.2.1.cell.1.activation"],
            'body.layers.2.1.cell.1.activation': ["body.layers.2.1.cell.1.ff1.dense", "body.layers.2.1.cell.1.dropout"],
            'body.layers.2.1.cell.1.dropout': ["body.layers.2.1.cell.1.activation", "body.layers.2.1.cell.1.ff2.dense"],
            'body.layers.2.1.cell.1.ff2.dense': ["body.layers.2.1.cell.1.dropout", "body.layers.2.1.cell.2"],
            'body.layers.2.1.cell.2': ["body.layers.2.1.cell.1.ff2.dense", "body.layers.3.0.cell.0"],
            'body.layers.3.0.cell.0': ["body.layers.2.1.cell.2",
                                       ["body.layers.3.0.cell.1.to_q", "body.layers.3.0.cell.1.to_k",
                                        "body.layers.3.0.cell.1.to_v"]],
            'body.layers.3.0.cell.1.to_q': ["body.layers.3.0.cell.0", "body.layers.3.0.cell.1.activation"],
            'body.layers.3.0.cell.1.to_k': ["body.layers.3.0.cell.0", "body.layers.3.0.cell.1.activation"],
            'body.layers.3.0.cell.1.to_v': ["body.layers.3.0.cell.0", "body.layers.3.0.cell.1.to_out"],
            'body.layers.3.0.cell.1.activation': [["body.layers.3.0.cell.1.to_k", "body.layers.3.0.cell.1.to_q"],
                                                  "body.layers.3.0.cell.1.to_out"],
            'body.layers.3.0.cell.1.to_out': [["body.layers.3.0.cell.1.activation", "body.layers.3.0.cell.1.to_v"],
                                              "body.layers.3.0.cell.1.dropout"],
            'body.layers.3.0.cell.1.dropout': ["body.layers.3.0.cell.1.to_out", "body.layers.3.0.cell.2"],
            'body.layers.3.0.cell.2': ["body.layers.3.0.cell.1.dropout", "body.layers.3.1.cell.0"],
            'body.layers.3.1.cell.0': ["body.layers.3.0.cell.2", "body.layers.3.1.cell.1.ff1.dense"],
            'body.layers.3.1.cell.1.ff1.dense': ["body.layers.3.1.cell.0", "body.layers.3.1.cell.1.activation"],
            'body.layers.3.1.cell.1.activation': ["body.layers.3.1.cell.1.ff1.dense", "body.layers.3.1.cell.1.dropout"],
            'body.layers.3.1.cell.1.dropout': ["body.layers.3.1.cell.1.activation", "body.layers.3.1.cell.1.ff2.dense"],
            'body.layers.3.1.cell.1.ff2.dense': ["body.layers.3.1.cell.1.dropout", "body.layers.3.1.cell.2"],
            'body.layers.3.1.cell.2': ["body.layers.3.1.cell.1.ff2.dense", "body.layers.4.0.cell.0"],
            'body.layers.4.0.cell.0': ["body.layers.3.1.cell.2",
                                       ["body.layers.4.0.cell.1.to_q", "body.layers.4.0.cell.1.to_k",
                                        "body.layers.4.0.cell.1.to_v"]],
            'body.layers.4.0.cell.1.to_q': ["body.layers.4.0.cell.0", "body.layers.4.0.cell.1.activation"],
            'body.layers.4.0.cell.1.to_k': ["body.layers.4.0.cell.0", "body.layers.4.0.cell.1.activation"],
            'body.layers.4.0.cell.1.to_v': ["body.layers.4.0.cell.0", "body.layers.4.0.cell.1.to_out"],
            'body.layers.4.0.cell.1.activation': [["body.layers.4.0.cell.1.to_k", "body.layers.4.0.cell.1.to_q"],
                                                  "body.layers.4.0.cell.1.to_out"],
            'body.layers.4.0.cell.1.to_out': [["body.layers.4.0.cell.1.activation", "body.layers.4.0.cell.1.to_v"],
                                              "body.layers.4.0.cell.1.dropout"],
            'body.layers.4.0.cell.1.dropout': ["body.layers.4.0.cell.1.to_out", "body.layers.4.0.cell.2"],
            'body.layers.4.0.cell.2': ["body.layers.4.0.cell.1.dropout", "body.layers.4.1.cell.0"],
            'body.layers.4.1.cell.0': ["body.layers.4.0.cell.2", "body.layers.4.1.cell.1.ff1.dense"],
            'body.layers.4.1.cell.1.ff1.dense': ["body.layers.4.1.cell.0", "body.layers.4.1.cell.1.activation"],
            'body.layers.4.1.cell.1.activation': ["body.layers.4.1.cell.1.ff1.dense", "body.layers.4.1.cell.1.dropout"],
            'body.layers.4.1.cell.1.dropout': ["body.layers.4.1.cell.1.activation", "body.layers.4.1.cell.1.ff2.dense"],
            'body.layers.4.1.cell.1.ff2.dense': ["body.layers.4.1.cell.1.dropout", "body.layers.4.1.cell.2"],
            'body.layers.4.1.cell.2': ["body.layers.4.1.cell.1.ff2.dense", "body.layers.5.0.cell.0"],
            'body.layers.5.0.cell.0': ["body.layers.4.1.cell.2",
                                       ["body.layers.5.0.cell.1.to_q", "body.layers.5.0.cell.1.to_k",
                                        "body.layers.5.0.cell.1.to_v"]],
            'body.layers.5.0.cell.1.to_q': ["body.layers.5.0.cell.0", "body.layers.5.0.cell.1.activation"],
            'body.layers.5.0.cell.1.to_k': ["body.layers.5.0.cell.0", "body.layers.5.0.cell.1.activation"],
            'body.layers.5.0.cell.1.to_v': ["body.layers.5.0.cell.0", "body.layers.5.0.cell.1.to_out"],
            'body.layers.5.0.cell.1.activation': [["body.layers.5.0.cell.1.to_k", "body.layers.5.0.cell.1.to_q"],
                                                  "body.layers.5.0.cell.1.to_out"],
            'body.layers.5.0.cell.1.to_out': [["body.layers.5.0.cell.1.activation", "body.layers.5.0.cell.1.to_v"],
                                              "body.layers.5.0.cell.1.dropout"],
            'body.layers.5.0.cell.1.dropout': ["body.layers.5.0.cell.1.to_out", "body.layers.5.0.cell.2"],
            'body.layers.5.0.cell.2': ["body.layers.5.0.cell.1.dropout", "body.layers.5.1.cell.0"],
            'body.layers.5.1.cell.0': ["body.layers.5.0.cell.2", "body.layers.5.1.cell.1.ff1.dense"],
            'body.layers.5.1.cell.1.ff1.dense': ["body.layers.5.1.cell.0", "body.layers.5.1.cell.1.activation"],
            'body.layers.5.1.cell.1.activation': ["body.layers.5.1.cell.1.ff1.dense", "body.layers.5.1.cell.1.dropout"],
            'body.layers.5.1.cell.1.dropout': ["body.layers.5.1.cell.1.activation", "body.layers.5.1.cell.1.ff2.dense"],
            'body.layers.5.1.cell.1.ff2.dense': ["body.layers.5.1.cell.1.dropout", "body.layers.5.1.cell.2"],
            'body.layers.5.1.cell.2': ["body.layers.5.1.cell.1.ff2.dense", "body.layers.6.0.cell.0"],
            'body.layers.6.0.cell.0': ["body.layers.5.1.cell.2",
                                       ["body.layers.6.0.cell.1.to_q", "body.layers.6.0.cell.1.to_k",
                                        "body.layers.6.0.cell.1.to_v"]],
            'body.layers.6.0.cell.1.to_q': ["body.layers.6.0.cell.0", "body.layers.6.0.cell.1.activation"],
            'body.layers.6.0.cell.1.to_k': ["body.layers.6.0.cell.0", "body.layers.6.0.cell.1.activation"],
            'body.layers.6.0.cell.1.to_v': ["body.layers.6.0.cell.0", "body.layers.6.0.cell.1.to_out"],
            'body.layers.6.0.cell.1.activation': [["body.layers.6.0.cell.1.to_k", "body.layers.6.0.cell.1.to_q"],
                                                  "body.layers.6.0.cell.1.to_out"],
            'body.layers.6.0.cell.1.to_out': [["body.layers.6.0.cell.1.activation", "body.layers.6.0.cell.1.to_v"],
                                              "body.layers.6.0.cell.1.dropout"],
            'body.layers.6.0.cell.1.dropout': ["body.layers.6.0.cell.1.to_out", "body.layers.6.0.cell.2"],
            'body.layers.6.0.cell.2': ["body.layers.6.0.cell.1.dropout", "body.layers.6.1.cell.0"],
            'body.layers.6.1.cell.0': ["body.layers.6.0.cell.2", "body.layers.6.1.cell.1.ff1.dense"],
            'body.layers.6.1.cell.1.ff1.dense': ["body.layers.6.1.cell.0", "body.layers.6.1.cell.1.activation"],
            'body.layers.6.1.cell.1.activation': ["body.layers.6.1.cell.1.ff1.dense", "body.layers.6.1.cell.1.dropout"],
            'body.layers.6.1.cell.1.dropout': ["body.layers.6.1.cell.1.activation", "body.layers.6.1.cell.1.ff2.dense"],
            'body.layers.6.1.cell.1.ff2.dense': ["body.layers.6.1.cell.1.dropout", "body.layers.6.1.cell.2"],
            'body.layers.6.1.cell.2': ["body.layers.6.1.cell.1.ff2.dense", "body.layers.7.0.cell.0"],
            'body.layers.7.0.cell.0': ["body.layers.6.1.cell.2",
                                       ["body.layers.7.0.cell.1.to_q", "body.layers.7.0.cell.1.to_k",
                                        "body.layers.7.0.cell.1.to_v"]],
            'body.layers.7.0.cell.1.to_q': ["body.layers.7.0.cell.0", "body.layers.7.0.cell.1.activation"],
            'body.layers.7.0.cell.1.to_k': ["body.layers.7.0.cell.0", "body.layers.7.0.cell.1.activation"],
            'body.layers.7.0.cell.1.to_v': ["body.layers.7.0.cell.0", "body.layers.7.0.cell.1.to_out"],
            'body.layers.7.0.cell.1.activation': [["body.layers.7.0.cell.1.to_k", "body.layers.7.0.cell.1.to_q"],
                                                  "body.layers.7.0.cell.1.to_out"],
            'body.layers.7.0.cell.1.to_out': [["body.layers.7.0.cell.1.activation", "body.layers.7.0.cell.1.to_v"],
                                              "body.layers.7.0.cell.1.dropout"],
            'body.layers.7.0.cell.1.dropout': ["body.layers.7.0.cell.1.to_out", "body.layers.7.0.cell.2"],
            'body.layers.7.0.cell.2': ["body.layers.7.0.cell.1.dropout", "body.layers.7.1.cell.0"],
            'body.layers.7.1.cell.0': ["body.layers.7.0.cell.2", "body.layers.7.1.cell.1.ff1.dense"],
            'body.layers.7.1.cell.1.ff1.dense': ["body.layers.7.1.cell.0", "body.layers.7.1.cell.1.activation"],
            'body.layers.7.1.cell.1.activation': ["body.layers.7.1.cell.1.ff1.dense", "body.layers.7.1.cell.1.dropout"],
            'body.layers.7.1.cell.1.dropout': ["body.layers.7.1.cell.1.activation", "body.layers.7.1.cell.1.ff2.dense"],
            'body.layers.7.1.cell.1.ff2.dense': ["body.layers.7.1.cell.1.dropout", "body.layers.7.1.cell.2"],
            'body.layers.7.1.cell.2': ["body.layers.7.1.cell.1.ff2.dense", "body.layers.8.0.cell.0"],
            'body.layers.8.0.cell.0': ["body.layers.7.1.cell.2",
                                       ["body.layers.8.0.cell.1.to_q", "body.layers.8.0.cell.1.to_k",
                                        "body.layers.8.0.cell.1.to_v"]],
            'body.layers.8.0.cell.1.to_q': ["body.layers.8.0.cell.0", "body.layers.8.0.cell.1.activation"],
            'body.layers.8.0.cell.1.to_k': ["body.layers.8.0.cell.0", "body.layers.8.0.cell.1.activation"],
            'body.layers.8.0.cell.1.to_v': ["body.layers.8.0.cell.0", "body.layers.8.0.cell.1.to_out"],
            'body.layers.8.0.cell.1.activation': [["body.layers.8.0.cell.1.to_k", "body.layers.8.0.cell.1.to_q"],
                                                  "body.layers.8.0.cell.1.to_out"],
            'body.layers.8.0.cell.1.to_out': [["body.layers.8.0.cell.1.activation", "body.layers.8.0.cell.1.to_v"],
                                              "body.layers.8.0.cell.1.dropout"],
            'body.layers.8.0.cell.1.dropout': ["body.layers.8.0.cell.1.to_out", "body.layers.8.0.cell.2"],
            'body.layers.8.0.cell.2': ["body.layers.8.0.cell.1.dropout", "body.layers.8.1.cell.0"],
            'body.layers.8.1.cell.0': ["body.layers.8.0.cell.2", "body.layers.8.1.cell.1.ff1.dense"],
            'body.layers.8.1.cell.1.ff1.dense': ["body.layers.8.1.cell.0", "body.layers.8.1.cell.1.activation"],
            'body.layers.8.1.cell.1.activation': ["body.layers.8.1.cell.1.ff1.dense", "body.layers.8.1.cell.1.dropout"],
            'body.layers.8.1.cell.1.dropout': ["body.layers.8.1.cell.1.activation", "body.layers.8.1.cell.1.ff2.dense"],
            'body.layers.8.1.cell.1.ff2.dense': ["body.layers.8.1.cell.1.dropout", "body.layers.8.1.cell.2"],
            'body.layers.8.1.cell.2': ["body.layers.8.1.cell.1.ff2.dense", "body.layers.9.0.cell.0"],
            'body.layers.9.0.cell.0': ["body.layers.8.1.cell.2",
                                       ["body.layers.9.0.cell.1.to_q", "body.layers.9.0.cell.1.to_k",
                                        "body.layers.9.0.cell.1.to_v"]],
            'body.layers.9.0.cell.1.to_q': ["body.layers.9.0.cell.0", "body.layers.9.0.cell.1.activation"],
            'body.layers.9.0.cell.1.to_k': ["body.layers.9.0.cell.0", "body.layers.9.0.cell.1.activation"],
            'body.layers.9.0.cell.1.to_v': ["body.layers.9.0.cell.0", "body.layers.9.0.cell.1.to_out"],
            'body.layers.9.0.cell.1.activation': [["body.layers.9.0.cell.1.to_k", "body.layers.9.0.cell.1.to_q"],
                                                  "body.layers.9.0.cell.1.to_out"],
            'body.layers.9.0.cell.1.to_out': [["body.layers.9.0.cell.1.activation", "body.layers.9.0.cell.1.to_v"],
                                              "body.layers.9.0.cell.1.dropout"],
            'body.layers.9.0.cell.1.dropout': ["body.layers.9.0.cell.1.to_out", "body.layers.9.0.cell.2"],
            'body.layers.9.0.cell.2': ["body.layers.9.0.cell.1.dropout", "body.layers.9.1.cell.0"],
            'body.layers.9.1.cell.0': ["body.layers.9.0.cell.2", "body.layers.9.1.cell.1.ff1.dense"],
            'body.layers.9.1.cell.1.ff1.dense': ["body.layers.9.1.cell.0", "body.layers.9.1.cell.1.activation"],
            'body.layers.9.1.cell.1.activation': ["body.layers.9.1.cell.1.ff1.dense", "body.layers.9.1.cell.1.dropout"],
            'body.layers.9.1.cell.1.dropout': ["body.layers.9.1.cell.1.activation", "body.layers.9.1.cell.1.ff2.dense"],
            'body.layers.9.1.cell.1.ff2.dense': ["body.layers.9.1.cell.1.dropout", "body.layers.9.1.cell.2"],
            'body.layers.9.1.cell.2': ["body.layers.9.1.cell.1.ff2.dense", "body.layers.10.0.cell.0"],
            'body.layers.10.0.cell.0': ["body.layers.9.1.cell.2",
                                        ["body.layers.10.0.cell.1.to_q", "body.layers.10.0.cell.1.to_k",
                                         "body.layers.10.0.cell.1.to_v"]],
            'body.layers.10.0.cell.1.to_q': ["body.layers.10.0.cell.0", "body.layers.10.0.cell.1.activation"],
            'body.layers.10.0.cell.1.to_k': ["body.layers.10.0.cell.0", "body.layers.10.0.cell.1.activation"],
            'body.layers.10.0.cell.1.to_v': ["body.layers.10.0.cell.0", "body.layers.10.0.cell.1.to_out"],
            'body.layers.10.0.cell.1.activation': [["body.layers.10.0.cell.1.to_k", "body.layers.10.0.cell.1.to_q"],
                                                   "body.layers.10.0.cell.1.to_out"],
            'body.layers.10.0.cell.1.to_out': [["body.layers.10.0.cell.1.activation", "body.layers.10.0.cell.1.to_v"],
                                               "body.layers.10.0.cell.1.dropout"],
            'body.layers.10.0.cell.1.dropout': ["body.layers.10.0.cell.1.to_out", "body.layers.10.0.cell.2"],
            'body.layers.10.0.cell.2': ["body.layers.10.0.cell.1.dropout", "body.layers.10.1.cell.0"],
            'body.layers.10.1.cell.0': ["body.layers.10.0.cell.2", "body.layers.10.1.cell.1.ff1.dense"],
            'body.layers.10.1.cell.1.ff1.dense': ["body.layers.10.1.cell.0", "body.layers.10.1.cell.1.activation"],
            'body.layers.10.1.cell.1.activation': ["body.layers.10.1.cell.1.ff1.dense",
                                                   "body.layers.10.1.cell.1.dropout"],
            'body.layers.10.1.cell.1.dropout': ["body.layers.10.1.cell.1.activation",
                                                "body.layers.10.1.cell.1.ff2.dense"],
            'body.layers.10.1.cell.1.ff2.dense': ["body.layers.10.1.cell.1.dropout", "body.layers.10.1.cell.2"],
            'body.layers.10.1.cell.2': ["body.layers.10.1.cell.1.ff2.dense", "body.layers.11.0.cell.0"],
            'body.layers.11.0.cell.0': ["body.layers.10.1.cell.2",
                                        ["body.layers.11.0.cell.1.to_q", "body.layers.11.0.cell.1.to_k",
                                         "body.layers.11.0.cell.1.to_v"]],
            'body.layers.11.0.cell.1.to_q': ["body.layers.11.0.cell.0", "body.layers.11.0.cell.1.activation"],
            'body.layers.11.0.cell.1.to_k': ["body.layers.11.0.cell.0", "body.layers.11.0.cell.1.activation"],
            'body.layers.11.0.cell.1.to_v': ["body.layers.11.0.cell.0", "body.layers.11.0.cell.1.to_out"],
            'body.layers.11.0.cell.1.activation': [["body.layers.11.0.cell.1.to_k", "body.layers.11.0.cell.1.to_q"],
                                                   "body.layers.11.0.cell.1.to_out"],
            'body.layers.11.0.cell.1.to_out': [["body.layers.11.0.cell.1.activation", "body.layers.11.0.cell.1.to_v"],
                                               "body.layers.11.0.cell.1.dropout"],
            'body.layers.11.0.cell.1.dropout': ["body.layers.11.0.cell.1.to_out", "body.layers.11.0.cell.2"],
            'body.layers.11.0.cell.2': ["body.layers.11.0.cell.1.dropout", "body.layers.11.1.cell.0"],
            'body.layers.11.1.cell.0': ["body.layers.11.0.cell.2", "body.layers.11.1.cell.1.ff1.dense"],
            'body.layers.11.1.cell.1.ff1.dense': ["body.layers.11.1.cell.0", "body.layers.11.1.cell.1.activation"],
            'body.layers.11.1.cell.1.activation': ["body.layers.11.1.cell.1.ff1.dense",
                                                   "body.layers.11.1.cell.1.dropout"],
            'body.layers.11.1.cell.1.dropout': ["body.layers.11.1.cell.1.activation",
                                                "body.layers.11.1.cell.1.ff2.dense"],
            'body.layers.11.1.cell.1.ff2.dense': ["body.layers.11.1.cell.1.dropout", "body.layers.11.1.cell.2"],
            'body.layers.11.1.cell.2': ["body.layers.11.1.cell.1.ff2.dense", "norm"],
            'norm': ["body.layers.11.1.cell.2", "head"],
            'head': ["norm", "OUTPUT"]

        }






    def forward(self, img):
        x = self.stem(img)
        bs, seq_len, _ = x.shape

        x = self.dropout(x)

        x = self.body(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = x.mean(dim=-2)

        return self.head(x)

    def set_layers(self, layer_name, new_layer):
        if 'dropout' == layer_name:
            self.dropout = new_layer
            self.layer_names["dropout"] = new_layer

        elif 'stem' == layer_name:
            self.stem = new_layer
            self.layer_names["stem"] = new_layer

        elif 'stem.patch_to_embedding' == layer_name:
            self.stem.patch_to_embedding = new_layer
            self.layer_names["stem.patch_to_embedding"] = new_layer

        elif 'stem.patch_to_embedding.dense' == layer_name:
            self.stem.patch_to_embedding.dense = new_layer
            self.layer_names["stem.patch_to_embedding.dense"] = new_layer

        elif 'body' == layer_name:
            self.body = new_layer
            self.layer_names["body"] = new_layer

        elif 'body.layers' == layer_name:
            self.body.layers = new_layer
            self.layer_names["body.layers"] = new_layer

        elif 'body.layers.0' == layer_name:
            self.body.layers[0] = new_layer
            self.layer_names["body.layers.0"] = new_layer

        elif 'body.layers.0.0' == layer_name:
            self.body.layers[0][0] = new_layer
            self.layer_names["body.layers.0.0"] = new_layer

        elif 'body.layers.0.0.cell' == layer_name:
            self.body.layers[0][0].cell = new_layer
            self.layer_names["body.layers.0.0.cell"] = new_layer

        elif 'body.layers.0.0.cell.0' == layer_name:
            self.body.layers[0][0].cell[0] = new_layer
            self.layer_names["body.layers.0.0.cell.0"] = new_layer

        elif 'body.layers.0.0.cell.1' == layer_name:
            self.body.layers[0][0].cell[1] = new_layer
            self.layer_names["body.layers.0.0.cell.1"] = new_layer

        elif 'body.layers.0.0.cell.1.to_q' == layer_name:
            self.body.layers[0][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.0.0.cell.1.to_q"] = new_layer

        elif 'body.layers.0.0.cell.1.to_k' == layer_name:
            self.body.layers[0][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.0.0.cell.1.to_k"] = new_layer

        elif 'body.layers.0.0.cell.1.to_v' == layer_name:
            self.body.layers[0][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.0.0.cell.1.to_v"] = new_layer

        elif 'body.layers.0.0.cell.1.to_out' == layer_name:
            self.body.layers[0][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.0.0.cell.1.to_out"] = new_layer

        elif 'body.layers.0.0.cell.1.dropout' == layer_name:
            self.body.layers[0][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.0.0.cell.1.dropout"] = new_layer

        elif 'body.layers.0.0.cell.1.activation' == layer_name:
            self.body.layers[0][0].cell[1].activation = new_layer
            self.layer_names["body.layers.0.0.cell.1.activation"] = new_layer

        elif 'body.layers.0.0.cell.2' == layer_name:
            self.body.layers[0][0].cell[2] = new_layer
            self.layer_names["body.layers.0.0.cell.2"] = new_layer

        elif 'body.layers.0.1' == layer_name:
            self.body.layers[0][1] = new_layer
            self.layer_names["body.layers.0.1"] = new_layer

        elif 'body.layers.0.1.cell' == layer_name:
            self.body.layers[0][1].cell = new_layer
            self.layer_names["body.layers.0.1.cell"] = new_layer

        elif 'body.layers.0.1.cell.0' == layer_name:
            self.body.layers[0][1].cell[0] = new_layer
            self.layer_names["body.layers.0.1.cell.0"] = new_layer

        elif 'body.layers.0.1.cell.1' == layer_name:
            self.body.layers[0][1].cell[1] = new_layer
            self.layer_names["body.layers.0.1.cell.1"] = new_layer

        elif 'body.layers.0.1.cell.1.ff1' == layer_name:
            self.body.layers[0][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.0.1.cell.1.ff1"] = new_layer

        elif 'body.layers.0.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[0][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.0.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.0.1.cell.1.activation' == layer_name:
            self.body.layers[0][1].cell[1].activation = new_layer
            self.layer_names["body.layers.0.1.cell.1.activation"] = new_layer

        elif 'body.layers.0.1.cell.1.dropout' == layer_name:
            self.body.layers[0][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.0.1.cell.1.dropout"] = new_layer

        elif 'body.layers.0.1.cell.1.ff2' == layer_name:
            self.body.layers[0][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.0.1.cell.1.ff2"] = new_layer

        elif 'body.layers.0.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[0][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.0.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.0.1.cell.2' == layer_name:
            self.body.layers[0][1].cell[2] = new_layer
            self.layer_names["body.layers.0.1.cell.2"] = new_layer

        elif 'body.layers.1' == layer_name:
            self.body.layers[1] = new_layer
            self.layer_names["body.layers.1"] = new_layer

        elif 'body.layers.1.0' == layer_name:
            self.body.layers[1][0] = new_layer
            self.layer_names["body.layers.1.0"] = new_layer

        elif 'body.layers.1.0.cell' == layer_name:
            self.body.layers[1][0].cell = new_layer
            self.layer_names["body.layers.1.0.cell"] = new_layer

        elif 'body.layers.1.0.cell.0' == layer_name:
            self.body.layers[1][0].cell[0] = new_layer
            self.layer_names["body.layers.1.0.cell.0"] = new_layer

        elif 'body.layers.1.0.cell.1' == layer_name:
            self.body.layers[1][0].cell[1] = new_layer
            self.layer_names["body.layers.1.0.cell.1"] = new_layer

        elif 'body.layers.1.0.cell.1.to_q' == layer_name:
            self.body.layers[1][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.1.0.cell.1.to_q"] = new_layer

        elif 'body.layers.1.0.cell.1.to_k' == layer_name:
            self.body.layers[1][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.1.0.cell.1.to_k"] = new_layer

        elif 'body.layers.1.0.cell.1.to_v' == layer_name:
            self.body.layers[1][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.1.0.cell.1.to_v"] = new_layer

        elif 'body.layers.1.0.cell.1.to_out' == layer_name:
            self.body.layers[1][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.1.0.cell.1.to_out"] = new_layer

        elif 'body.layers.1.0.cell.1.dropout' == layer_name:
            self.body.layers[1][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.1.0.cell.1.dropout"] = new_layer

        elif 'body.layers.1.0.cell.2' == layer_name:
            self.body.layers[1][0].cell[2] = new_layer
            self.layer_names["body.layers.1.0.cell.2"] = new_layer

        elif 'body.layers.1.1' == layer_name:
            self.body.layers[1][1] = new_layer
            self.layer_names["body.layers.1.1"] = new_layer

        elif 'body.layers.1.1.cell' == layer_name:
            self.body.layers[1][1].cell = new_layer
            self.layer_names["body.layers.1.1.cell"] = new_layer

        elif 'body.layers.1.1.cell.0' == layer_name:
            self.body.layers[1][1].cell[0] = new_layer
            self.layer_names["body.layers.1.1.cell.0"] = new_layer

        elif 'body.layers.1.1.cell.1' == layer_name:
            self.body.layers[1][1].cell[1] = new_layer
            self.layer_names["body.layers.1.1.cell.1"] = new_layer

        elif 'body.layers.1.1.cell.1.ff1' == layer_name:
            self.body.layers[1][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.1.1.cell.1.ff1"] = new_layer

        elif 'body.layers.1.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[1][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.1.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.1.1.cell.1.dropout' == layer_name:
            self.body.layers[1][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.1.1.cell.1.dropout"] = new_layer

        elif 'body.layers.1.1.cell.1.ff2' == layer_name:
            self.body.layers[1][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.1.1.cell.1.ff2"] = new_layer

        elif 'body.layers.1.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[1][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.1.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.1.1.cell.2' == layer_name:
            self.body.layers[1][1].cell[2] = new_layer
            self.layer_names["body.layers.1.1.cell.2"] = new_layer

        elif 'body.layers.2' == layer_name:
            self.body.layers[2] = new_layer
            self.layer_names["body.layers.2"] = new_layer

        elif 'body.layers.2.0' == layer_name:
            self.body.layers[2][0] = new_layer
            self.layer_names["body.layers.2.0"] = new_layer

        elif 'body.layers.2.0.cell' == layer_name:
            self.body.layers[2][0].cell = new_layer
            self.layer_names["body.layers.2.0.cell"] = new_layer

        elif 'body.layers.2.0.cell.0' == layer_name:
            self.body.layers[2][0].cell[0] = new_layer
            self.layer_names["body.layers.2.0.cell.0"] = new_layer

        elif 'body.layers.2.0.cell.1' == layer_name:
            self.body.layers[2][0].cell[1] = new_layer
            self.layer_names["body.layers.2.0.cell.1"] = new_layer

        elif 'body.layers.2.0.cell.1.to_q' == layer_name:
            self.body.layers[2][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.2.0.cell.1.to_q"] = new_layer

        elif 'body.layers.2.0.cell.1.to_k' == layer_name:
            self.body.layers[2][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.2.0.cell.1.to_k"] = new_layer

        elif 'body.layers.2.0.cell.1.to_v' == layer_name:
            self.body.layers[2][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.2.0.cell.1.to_v"] = new_layer

        elif 'body.layers.2.0.cell.1.to_out' == layer_name:
            self.body.layers[2][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.2.0.cell.1.to_out"] = new_layer

        elif 'body.layers.2.0.cell.1.dropout' == layer_name:
            self.body.layers[2][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.2.0.cell.1.dropout"] = new_layer

        elif 'body.layers.2.0.cell.2' == layer_name:
            self.body.layers[2][0].cell[2] = new_layer
            self.layer_names["body.layers.2.0.cell.2"] = new_layer

        elif 'body.layers.2.1' == layer_name:
            self.body.layers[2][1] = new_layer
            self.layer_names["body.layers.2.1"] = new_layer

        elif 'body.layers.2.1.cell' == layer_name:
            self.body.layers[2][1].cell = new_layer
            self.layer_names["body.layers.2.1.cell"] = new_layer

        elif 'body.layers.2.1.cell.0' == layer_name:
            self.body.layers[2][1].cell[0] = new_layer
            self.layer_names["body.layers.2.1.cell.0"] = new_layer

        elif 'body.layers.2.1.cell.1' == layer_name:
            self.body.layers[2][1].cell[1] = new_layer
            self.layer_names["body.layers.2.1.cell.1"] = new_layer

        elif 'body.layers.2.1.cell.1.ff1' == layer_name:
            self.body.layers[2][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.2.1.cell.1.ff1"] = new_layer

        elif 'body.layers.2.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[2][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.2.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.2.1.cell.1.dropout' == layer_name:
            self.body.layers[2][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.2.1.cell.1.dropout"] = new_layer

        elif 'body.layers.2.1.cell.1.ff2' == layer_name:
            self.body.layers[2][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.2.1.cell.1.ff2"] = new_layer

        elif 'body.layers.2.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[2][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.2.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.2.1.cell.2' == layer_name:
            self.body.layers[2][1].cell[2] = new_layer
            self.layer_names["body.layers.2.1.cell.2"] = new_layer

        elif 'body.layers.3' == layer_name:
            self.body.layers[3] = new_layer
            self.layer_names["body.layers.3"] = new_layer

        elif 'body.layers.3.0' == layer_name:
            self.body.layers[3][0] = new_layer
            self.layer_names["body.layers.3.0"] = new_layer

        elif 'body.layers.3.0.cell' == layer_name:
            self.body.layers[3][0].cell = new_layer
            self.layer_names["body.layers.3.0.cell"] = new_layer

        elif 'body.layers.3.0.cell.0' == layer_name:
            self.body.layers[3][0].cell[0] = new_layer
            self.layer_names["body.layers.3.0.cell.0"] = new_layer

        elif 'body.layers.3.0.cell.1' == layer_name:
            self.body.layers[3][0].cell[1] = new_layer
            self.layer_names["body.layers.3.0.cell.1"] = new_layer

        elif 'body.layers.3.0.cell.1.to_q' == layer_name:
            self.body.layers[3][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.3.0.cell.1.to_q"] = new_layer

        elif 'body.layers.3.0.cell.1.to_k' == layer_name:
            self.body.layers[3][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.3.0.cell.1.to_k"] = new_layer

        elif 'body.layers.3.0.cell.1.to_v' == layer_name:
            self.body.layers[3][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.3.0.cell.1.to_v"] = new_layer

        elif 'body.layers.3.0.cell.1.to_out' == layer_name:
            self.body.layers[3][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.3.0.cell.1.to_out"] = new_layer

        elif 'body.layers.3.0.cell.1.dropout' == layer_name:
            self.body.layers[3][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.3.0.cell.1.dropout"] = new_layer

        elif 'body.layers.3.0.cell.2' == layer_name:
            self.body.layers[3][0].cell[2] = new_layer
            self.layer_names["body.layers.3.0.cell.2"] = new_layer

        elif 'body.layers.3.1' == layer_name:
            self.body.layers[3][1] = new_layer
            self.layer_names["body.layers.3.1"] = new_layer

        elif 'body.layers.3.1.cell' == layer_name:
            self.body.layers[3][1].cell = new_layer
            self.layer_names["body.layers.3.1.cell"] = new_layer

        elif 'body.layers.3.1.cell.0' == layer_name:
            self.body.layers[3][1].cell[0] = new_layer
            self.layer_names["body.layers.3.1.cell.0"] = new_layer

        elif 'body.layers.3.1.cell.1' == layer_name:
            self.body.layers[3][1].cell[1] = new_layer
            self.layer_names["body.layers.3.1.cell.1"] = new_layer

        elif 'body.layers.3.1.cell.1.ff1' == layer_name:
            self.body.layers[3][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.3.1.cell.1.ff1"] = new_layer

        elif 'body.layers.3.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[3][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.3.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.3.1.cell.1.dropout' == layer_name:
            self.body.layers[3][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.3.1.cell.1.dropout"] = new_layer

        elif 'body.layers.3.1.cell.1.ff2' == layer_name:
            self.body.layers[3][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.3.1.cell.1.ff2"] = new_layer

        elif 'body.layers.3.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[3][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.3.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.3.1.cell.2' == layer_name:
            self.body.layers[3][1].cell[2] = new_layer
            self.layer_names["body.layers.3.1.cell.2"] = new_layer

        elif 'body.layers.4' == layer_name:
            self.body.layers[4] = new_layer
            self.layer_names["body.layers.4"] = new_layer

        elif 'body.layers.4.0' == layer_name:
            self.body.layers[4][0] = new_layer
            self.layer_names["body.layers.4.0"] = new_layer

        elif 'body.layers.4.0.cell' == layer_name:
            self.body.layers[4][0].cell = new_layer
            self.layer_names["body.layers.4.0.cell"] = new_layer

        elif 'body.layers.4.0.cell.0' == layer_name:
            self.body.layers[4][0].cell[0] = new_layer
            self.layer_names["body.layers.4.0.cell.0"] = new_layer

        elif 'body.layers.4.0.cell.1' == layer_name:
            self.body.layers[4][0].cell[1] = new_layer
            self.layer_names["body.layers.4.0.cell.1"] = new_layer

        elif 'body.layers.4.0.cell.1.to_q' == layer_name:
            self.body.layers[4][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.4.0.cell.1.to_q"] = new_layer

        elif 'body.layers.4.0.cell.1.to_k' == layer_name:
            self.body.layers[4][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.4.0.cell.1.to_k"] = new_layer

        elif 'body.layers.4.0.cell.1.to_v' == layer_name:
            self.body.layers[4][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.4.0.cell.1.to_v"] = new_layer

        elif 'body.layers.4.0.cell.1.to_out' == layer_name:
            self.body.layers[4][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.4.0.cell.1.to_out"] = new_layer

        elif 'body.layers.4.0.cell.1.dropout' == layer_name:
            self.body.layers[4][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.4.0.cell.1.dropout"] = new_layer

        elif 'body.layers.4.0.cell.2' == layer_name:
            self.body.layers[4][0].cell[2] = new_layer
            self.layer_names["body.layers.4.0.cell.2"] = new_layer

        elif 'body.layers.4.1' == layer_name:
            self.body.layers[4][1] = new_layer
            self.layer_names["body.layers.4.1"] = new_layer

        elif 'body.layers.4.1.cell' == layer_name:
            self.body.layers[4][1].cell = new_layer
            self.layer_names["body.layers.4.1.cell"] = new_layer

        elif 'body.layers.4.1.cell.0' == layer_name:
            self.body.layers[4][1].cell[0] = new_layer
            self.layer_names["body.layers.4.1.cell.0"] = new_layer

        elif 'body.layers.4.1.cell.1' == layer_name:
            self.body.layers[4][1].cell[1] = new_layer
            self.layer_names["body.layers.4.1.cell.1"] = new_layer

        elif 'body.layers.4.1.cell.1.ff1' == layer_name:
            self.body.layers[4][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.4.1.cell.1.ff1"] = new_layer

        elif 'body.layers.4.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[4][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.4.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.4.1.cell.1.dropout' == layer_name:
            self.body.layers[4][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.4.1.cell.1.dropout"] = new_layer

        elif 'body.layers.4.1.cell.1.ff2' == layer_name:
            self.body.layers[4][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.4.1.cell.1.ff2"] = new_layer

        elif 'body.layers.4.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[4][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.4.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.4.1.cell.2' == layer_name:
            self.body.layers[4][1].cell[2] = new_layer
            self.layer_names["body.layers.4.1.cell.2"] = new_layer

        elif 'body.layers.5' == layer_name:
            self.body.layers[5] = new_layer
            self.layer_names["body.layers.5"] = new_layer

        elif 'body.layers.5.0' == layer_name:
            self.body.layers[5][0] = new_layer
            self.layer_names["body.layers.5.0"] = new_layer

        elif 'body.layers.5.0.cell' == layer_name:
            self.body.layers[5][0].cell = new_layer
            self.layer_names["body.layers.5.0.cell"] = new_layer

        elif 'body.layers.5.0.cell.0' == layer_name:
            self.body.layers[5][0].cell[0] = new_layer
            self.layer_names["body.layers.5.0.cell.0"] = new_layer

        elif 'body.layers.5.0.cell.1' == layer_name:
            self.body.layers[5][0].cell[1] = new_layer
            self.layer_names["body.layers.5.0.cell.1"] = new_layer

        elif 'body.layers.5.0.cell.1.to_q' == layer_name:
            self.body.layers[5][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.5.0.cell.1.to_q"] = new_layer

        elif 'body.layers.5.0.cell.1.to_k' == layer_name:
            self.body.layers[5][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.5.0.cell.1.to_k"] = new_layer

        elif 'body.layers.5.0.cell.1.to_v' == layer_name:
            self.body.layers[5][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.5.0.cell.1.to_v"] = new_layer

        elif 'body.layers.5.0.cell.1.to_out' == layer_name:
            self.body.layers[5][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.5.0.cell.1.to_out"] = new_layer

        elif 'body.layers.5.0.cell.1.dropout' == layer_name:
            self.body.layers[5][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.5.0.cell.1.dropout"] = new_layer

        elif 'body.layers.5.0.cell.2' == layer_name:
            self.body.layers[5][0].cell[2] = new_layer
            self.layer_names["body.layers.5.0.cell.2"] = new_layer

        elif 'body.layers.5.1' == layer_name:
            self.body.layers[5][1] = new_layer
            self.layer_names["body.layers.5.1"] = new_layer

        elif 'body.layers.5.1.cell' == layer_name:
            self.body.layers[5][1].cell = new_layer
            self.layer_names["body.layers.5.1.cell"] = new_layer

        elif 'body.layers.5.1.cell.0' == layer_name:
            self.body.layers[5][1].cell[0] = new_layer
            self.layer_names["body.layers.5.1.cell.0"] = new_layer

        elif 'body.layers.5.1.cell.1' == layer_name:
            self.body.layers[5][1].cell[1] = new_layer
            self.layer_names["body.layers.5.1.cell.1"] = new_layer

        elif 'body.layers.5.1.cell.1.ff1' == layer_name:
            self.body.layers[5][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.5.1.cell.1.ff1"] = new_layer

        elif 'body.layers.5.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[5][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.5.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.5.1.cell.1.dropout' == layer_name:
            self.body.layers[5][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.5.1.cell.1.dropout"] = new_layer

        elif 'body.layers.5.1.cell.1.ff2' == layer_name:
            self.body.layers[5][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.5.1.cell.1.ff2"] = new_layer

        elif 'body.layers.5.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[5][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.5.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.5.1.cell.2' == layer_name:
            self.body.layers[5][1].cell[2] = new_layer
            self.layer_names["body.layers.5.1.cell.2"] = new_layer

        elif 'body.layers.6' == layer_name:
            self.body.layers[6] = new_layer
            self.layer_names["body.layers.6"] = new_layer

        elif 'body.layers.6.0' == layer_name:
            self.body.layers[6][0] = new_layer
            self.layer_names["body.layers.6.0"] = new_layer

        elif 'body.layers.6.0.cell' == layer_name:
            self.body.layers[6][0].cell = new_layer
            self.layer_names["body.layers.6.0.cell"] = new_layer

        elif 'body.layers.6.0.cell.0' == layer_name:
            self.body.layers[6][0].cell[0] = new_layer
            self.layer_names["body.layers.6.0.cell.0"] = new_layer

        elif 'body.layers.6.0.cell.1' == layer_name:
            self.body.layers[6][0].cell[1] = new_layer
            self.layer_names["body.layers.6.0.cell.1"] = new_layer

        elif 'body.layers.6.0.cell.1.to_q' == layer_name:
            self.body.layers[6][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.6.0.cell.1.to_q"] = new_layer

        elif 'body.layers.6.0.cell.1.to_k' == layer_name:
            self.body.layers[6][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.6.0.cell.1.to_k"] = new_layer

        elif 'body.layers.6.0.cell.1.to_v' == layer_name:
            self.body.layers[6][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.6.0.cell.1.to_v"] = new_layer

        elif 'body.layers.6.0.cell.1.to_out' == layer_name:
            self.body.layers[6][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.6.0.cell.1.to_out"] = new_layer

        elif 'body.layers.6.0.cell.1.dropout' == layer_name:
            self.body.layers[6][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.6.0.cell.1.dropout"] = new_layer

        elif 'body.layers.6.0.cell.2' == layer_name:
            self.body.layers[6][0].cell[2] = new_layer
            self.layer_names["body.layers.6.0.cell.2"] = new_layer

        elif 'body.layers.6.1' == layer_name:
            self.body.layers[6][1] = new_layer
            self.layer_names["body.layers.6.1"] = new_layer

        elif 'body.layers.6.1.cell' == layer_name:
            self.body.layers[6][1].cell = new_layer
            self.layer_names["body.layers.6.1.cell"] = new_layer

        elif 'body.layers.6.1.cell.0' == layer_name:
            self.body.layers[6][1].cell[0] = new_layer
            self.layer_names["body.layers.6.1.cell.0"] = new_layer

        elif 'body.layers.6.1.cell.1' == layer_name:
            self.body.layers[6][1].cell[1] = new_layer
            self.layer_names["body.layers.6.1.cell.1"] = new_layer

        elif 'body.layers.6.1.cell.1.ff1' == layer_name:
            self.body.layers[6][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.6.1.cell.1.ff1"] = new_layer

        elif 'body.layers.6.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[6][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.6.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.6.1.cell.1.dropout' == layer_name:
            self.body.layers[6][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.6.1.cell.1.dropout"] = new_layer

        elif 'body.layers.6.1.cell.1.ff2' == layer_name:
            self.body.layers[6][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.6.1.cell.1.ff2"] = new_layer

        elif 'body.layers.6.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[6][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.6.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.6.1.cell.2' == layer_name:
            self.body.layers[6][1].cell[2] = new_layer
            self.layer_names["body.layers.6.1.cell.2"] = new_layer

        elif 'body.layers.7' == layer_name:
            self.body.layers[7] = new_layer
            self.layer_names["body.layers.7"] = new_layer

        elif 'body.layers.7.0' == layer_name:
            self.body.layers[7][0] = new_layer
            self.layer_names["body.layers.7.0"] = new_layer

        elif 'body.layers.7.0.cell' == layer_name:
            self.body.layers[7][0].cell = new_layer
            self.layer_names["body.layers.7.0.cell"] = new_layer

        elif 'body.layers.7.0.cell.0' == layer_name:
            self.body.layers[7][0].cell[0] = new_layer
            self.layer_names["body.layers.7.0.cell.0"] = new_layer

        elif 'body.layers.7.0.cell.1' == layer_name:
            self.body.layers[7][0].cell[1] = new_layer
            self.layer_names["body.layers.7.0.cell.1"] = new_layer

        elif 'body.layers.7.0.cell.1.to_q' == layer_name:
            self.body.layers[7][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.7.0.cell.1.to_q"] = new_layer

        elif 'body.layers.7.0.cell.1.to_k' == layer_name:
            self.body.layers[7][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.7.0.cell.1.to_k"] = new_layer

        elif 'body.layers.7.0.cell.1.to_v' == layer_name:
            self.body.layers[7][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.7.0.cell.1.to_v"] = new_layer

        elif 'body.layers.7.0.cell.1.to_out' == layer_name:
            self.body.layers[7][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.7.0.cell.1.to_out"] = new_layer

        elif 'body.layers.7.0.cell.1.dropout' == layer_name:
            self.body.layers[7][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.7.0.cell.1.dropout"] = new_layer

        elif 'body.layers.7.0.cell.2' == layer_name:
            self.body.layers[7][0].cell[2] = new_layer
            self.layer_names["body.layers.7.0.cell.2"] = new_layer

        elif 'body.layers.7.1' == layer_name:
            self.body.layers[7][1] = new_layer
            self.layer_names["body.layers.7.1"] = new_layer

        elif 'body.layers.7.1.cell' == layer_name:
            self.body.layers[7][1].cell = new_layer
            self.layer_names["body.layers.7.1.cell"] = new_layer

        elif 'body.layers.7.1.cell.0' == layer_name:
            self.body.layers[7][1].cell[0] = new_layer
            self.layer_names["body.layers.7.1.cell.0"] = new_layer

        elif 'body.layers.7.1.cell.1' == layer_name:
            self.body.layers[7][1].cell[1] = new_layer
            self.layer_names["body.layers.7.1.cell.1"] = new_layer

        elif 'body.layers.7.1.cell.1.ff1' == layer_name:
            self.body.layers[7][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.7.1.cell.1.ff1"] = new_layer

        elif 'body.layers.7.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[7][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.7.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.7.1.cell.1.dropout' == layer_name:
            self.body.layers[7][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.7.1.cell.1.dropout"] = new_layer

        elif 'body.layers.7.1.cell.1.ff2' == layer_name:
            self.body.layers[7][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.7.1.cell.1.ff2"] = new_layer

        elif 'body.layers.7.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[7][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.7.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.7.1.cell.2' == layer_name:
            self.body.layers[7][1].cell[2] = new_layer
            self.layer_names["body.layers.7.1.cell.2"] = new_layer

        elif 'body.layers.8' == layer_name:
            self.body.layers[8] = new_layer
            self.layer_names["body.layers.8"] = new_layer

        elif 'body.layers.8.0' == layer_name:
            self.body.layers[8][0] = new_layer
            self.layer_names["body.layers.8.0"] = new_layer

        elif 'body.layers.8.0.cell' == layer_name:
            self.body.layers[8][0].cell = new_layer
            self.layer_names["body.layers.8.0.cell"] = new_layer

        elif 'body.layers.8.0.cell.0' == layer_name:
            self.body.layers[8][0].cell[0] = new_layer
            self.layer_names["body.layers.8.0.cell.0"] = new_layer

        elif 'body.layers.8.0.cell.1' == layer_name:
            self.body.layers[8][0].cell[1] = new_layer
            self.layer_names["body.layers.8.0.cell.1"] = new_layer

        elif 'body.layers.8.0.cell.1.to_q' == layer_name:
            self.body.layers[8][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.8.0.cell.1.to_q"] = new_layer

        elif 'body.layers.8.0.cell.1.to_k' == layer_name:
            self.body.layers[8][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.8.0.cell.1.to_k"] = new_layer

        elif 'body.layers.8.0.cell.1.to_v' == layer_name:
            self.body.layers[8][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.8.0.cell.1.to_v"] = new_layer

        elif 'body.layers.8.0.cell.1.to_out' == layer_name:
            self.body.layers[8][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.8.0.cell.1.to_out"] = new_layer

        elif 'body.layers.8.0.cell.1.dropout' == layer_name:
            self.body.layers[8][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.8.0.cell.1.dropout"] = new_layer

        elif 'body.layers.8.0.cell.2' == layer_name:
            self.body.layers[8][0].cell[2] = new_layer
            self.layer_names["body.layers.8.0.cell.2"] = new_layer

        elif 'body.layers.8.1' == layer_name:
            self.body.layers[8][1] = new_layer
            self.layer_names["body.layers.8.1"] = new_layer

        elif 'body.layers.8.1.cell' == layer_name:
            self.body.layers[8][1].cell = new_layer
            self.layer_names["body.layers.8.1.cell"] = new_layer

        elif 'body.layers.8.1.cell.0' == layer_name:
            self.body.layers[8][1].cell[0] = new_layer
            self.layer_names["body.layers.8.1.cell.0"] = new_layer

        elif 'body.layers.8.1.cell.1' == layer_name:
            self.body.layers[8][1].cell[1] = new_layer
            self.layer_names["body.layers.8.1.cell.1"] = new_layer

        elif 'body.layers.8.1.cell.1.ff1' == layer_name:
            self.body.layers[8][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.8.1.cell.1.ff1"] = new_layer

        elif 'body.layers.8.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[8][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.8.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.8.1.cell.1.dropout' == layer_name:
            self.body.layers[8][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.8.1.cell.1.dropout"] = new_layer

        elif 'body.layers.8.1.cell.1.ff2' == layer_name:
            self.body.layers[8][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.8.1.cell.1.ff2"] = new_layer

        elif 'body.layers.8.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[8][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.8.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.8.1.cell.2' == layer_name:
            self.body.layers[8][1].cell[2] = new_layer
            self.layer_names["body.layers.8.1.cell.2"] = new_layer

        elif 'body.layers.9' == layer_name:
            self.body.layers[9] = new_layer
            self.layer_names["body.layers.9"] = new_layer

        elif 'body.layers.9.0' == layer_name:
            self.body.layers[9][0] = new_layer
            self.layer_names["body.layers.9.0"] = new_layer

        elif 'body.layers.9.0.cell' == layer_name:
            self.body.layers[9][0].cell = new_layer
            self.layer_names["body.layers.9.0.cell"] = new_layer

        elif 'body.layers.9.0.cell.0' == layer_name:
            self.body.layers[9][0].cell[0] = new_layer
            self.layer_names["body.layers.9.0.cell.0"] = new_layer

        elif 'body.layers.9.0.cell.1' == layer_name:
            self.body.layers[9][0].cell[1] = new_layer
            self.layer_names["body.layers.9.0.cell.1"] = new_layer

        elif 'body.layers.9.0.cell.1.to_q' == layer_name:
            self.body.layers[9][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.9.0.cell.1.to_q"] = new_layer

        elif 'body.layers.9.0.cell.1.to_k' == layer_name:
            self.body.layers[9][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.9.0.cell.1.to_k"] = new_layer

        elif 'body.layers.9.0.cell.1.to_v' == layer_name:
            self.body.layers[9][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.9.0.cell.1.to_v"] = new_layer

        elif 'body.layers.9.0.cell.1.to_out' == layer_name:
            self.body.layers[9][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.9.0.cell.1.to_out"] = new_layer

        elif 'body.layers.9.0.cell.1.dropout' == layer_name:
            self.body.layers[9][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.9.0.cell.1.dropout"] = new_layer

        elif 'body.layers.9.0.cell.2' == layer_name:
            self.body.layers[9][0].cell[2] = new_layer
            self.layer_names["body.layers.9.0.cell.2"] = new_layer

        elif 'body.layers.9.1' == layer_name:
            self.body.layers[9][1] = new_layer
            self.layer_names["body.layers.9.1"] = new_layer

        elif 'body.layers.9.1.cell' == layer_name:
            self.body.layers[9][1].cell = new_layer
            self.layer_names["body.layers.9.1.cell"] = new_layer

        elif 'body.layers.9.1.cell.0' == layer_name:
            self.body.layers[9][1].cell[0] = new_layer
            self.layer_names["body.layers.9.1.cell.0"] = new_layer

        elif 'body.layers.9.1.cell.1' == layer_name:
            self.body.layers[9][1].cell[1] = new_layer
            self.layer_names["body.layers.9.1.cell.1"] = new_layer

        elif 'body.layers.9.1.cell.1.ff1' == layer_name:
            self.body.layers[9][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.9.1.cell.1.ff1"] = new_layer

        elif 'body.layers.9.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[9][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.9.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.9.1.cell.1.dropout' == layer_name:
            self.body.layers[9][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.9.1.cell.1.dropout"] = new_layer

        elif 'body.layers.9.1.cell.1.ff2' == layer_name:
            self.body.layers[9][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.9.1.cell.1.ff2"] = new_layer

        elif 'body.layers.9.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[9][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.9.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.9.1.cell.2' == layer_name:
            self.body.layers[9][1].cell[2] = new_layer
            self.layer_names["body.layers.9.1.cell.2"] = new_layer

        elif 'body.layers.10' == layer_name:
            self.body.layers[10] = new_layer
            self.layer_names["body.layers.10"] = new_layer

        elif 'body.layers.10.0' == layer_name:
            self.body.layers[10][0] = new_layer
            self.layer_names["body.layers.10.0"] = new_layer

        elif 'body.layers.10.0.cell' == layer_name:
            self.body.layers[10][0].cell = new_layer
            self.layer_names["body.layers.10.0.cell"] = new_layer

        elif 'body.layers.10.0.cell.0' == layer_name:
            self.body.layers[10][0].cell[0] = new_layer
            self.layer_names["body.layers.10.0.cell.0"] = new_layer

        elif 'body.layers.10.0.cell.1' == layer_name:
            self.body.layers[10][0].cell[1] = new_layer
            self.layer_names["body.layers.10.0.cell.1"] = new_layer

        elif 'body.layers.10.0.cell.1.to_q' == layer_name:
            self.body.layers[10][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.10.0.cell.1.to_q"] = new_layer

        elif 'body.layers.10.0.cell.1.to_k' == layer_name:
            self.body.layers[10][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.10.0.cell.1.to_k"] = new_layer

        elif 'body.layers.10.0.cell.1.to_v' == layer_name:
            self.body.layers[10][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.10.0.cell.1.to_v"] = new_layer

        elif 'body.layers.10.0.cell.1.to_out' == layer_name:
            self.body.layers[10][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.10.0.cell.1.to_out"] = new_layer

        elif 'body.layers.10.0.cell.1.dropout' == layer_name:
            self.body.layers[10][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.10.0.cell.1.dropout"] = new_layer

        elif 'body.layers.10.0.cell.2' == layer_name:
            self.body.layers[10][0].cell[2] = new_layer
            self.layer_names["body.layers.10.0.cell.2"] = new_layer

        elif 'body.layers.10.1' == layer_name:
            self.body.layers[10][1] = new_layer
            self.layer_names["body.layers.10.1"] = new_layer

        elif 'body.layers.10.1.cell' == layer_name:
            self.body.layers[10][1].cell = new_layer
            self.layer_names["body.layers.10.1.cell"] = new_layer

        elif 'body.layers.10.1.cell.0' == layer_name:
            self.body.layers[10][1].cell[0] = new_layer
            self.layer_names["body.layers.10.1.cell.0"] = new_layer

        elif 'body.layers.10.1.cell.1' == layer_name:
            self.body.layers[10][1].cell[1] = new_layer
            self.layer_names["body.layers.10.1.cell.1"] = new_layer

        elif 'body.layers.10.1.cell.1.ff1' == layer_name:
            self.body.layers[10][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.10.1.cell.1.ff1"] = new_layer

        elif 'body.layers.10.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[10][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.10.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.10.1.cell.1.dropout' == layer_name:
            self.body.layers[10][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.10.1.cell.1.dropout"] = new_layer

        elif 'body.layers.10.1.cell.1.ff2' == layer_name:
            self.body.layers[10][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.10.1.cell.1.ff2"] = new_layer

        elif 'body.layers.10.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[10][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.10.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.10.1.cell.2' == layer_name:
            self.body.layers[10][1].cell[2] = new_layer
            self.layer_names["body.layers.10.1.cell.2"] = new_layer

        elif 'body.layers.11' == layer_name:
            self.body.layers[11] = new_layer
            self.layer_names["body.layers.11"] = new_layer

        elif 'body.layers.11.0' == layer_name:
            self.body.layers[11][0] = new_layer
            self.layer_names["body.layers.11.0"] = new_layer

        elif 'body.layers.11.0.cell' == layer_name:
            self.body.layers[11][0].cell = new_layer
            self.layer_names["body.layers.11.0.cell"] = new_layer

        elif 'body.layers.11.0.cell.0' == layer_name:
            self.body.layers[11][0].cell[0] = new_layer
            self.layer_names["body.layers.11.0.cell.0"] = new_layer

        elif 'body.layers.11.0.cell.1' == layer_name:
            self.body.layers[11][0].cell[1] = new_layer
            self.layer_names["body.layers.11.0.cell.1"] = new_layer

        elif 'body.layers.11.0.cell.1.to_q' == layer_name:
            self.body.layers[11][0].cell[1].to_q = new_layer
            self.layer_names["body.layers.11.0.cell.1.to_q"] = new_layer

        elif 'body.layers.11.0.cell.1.to_k' == layer_name:
            self.body.layers[11][0].cell[1].to_k = new_layer
            self.layer_names["body.layers.11.0.cell.1.to_k"] = new_layer

        elif 'body.layers.11.0.cell.1.to_v' == layer_name:
            self.body.layers[11][0].cell[1].to_v = new_layer
            self.layer_names["body.layers.11.0.cell.1.to_v"] = new_layer

        elif 'body.layers.11.0.cell.1.to_out' == layer_name:
            self.body.layers[11][0].cell[1].to_out = new_layer
            self.layer_names["body.layers.11.0.cell.1.to_out"] = new_layer

        elif 'body.layers.11.0.cell.1.dropout' == layer_name:
            self.body.layers[11][0].cell[1].dropout = new_layer
            self.layer_names["body.layers.11.0.cell.1.dropout"] = new_layer

        elif 'body.layers.11.0.cell.2' == layer_name:
            self.body.layers[11][0].cell[2] = new_layer
            self.layer_names["body.layers.11.0.cell.2"] = new_layer

        elif 'body.layers.11.1' == layer_name:
            self.body.layers[11][1] = new_layer
            self.layer_names["body.layers.11.1"] = new_layer

        elif 'body.layers.11.1.cell' == layer_name:
            self.body.layers[11][1].cell = new_layer
            self.layer_names["body.layers.11.1.cell"] = new_layer

        elif 'body.layers.11.1.cell.0' == layer_name:
            self.body.layers[11][1].cell[0] = new_layer
            self.layer_names["body.layers.11.1.cell.0"] = new_layer

        elif 'body.layers.11.1.cell.1' == layer_name:
            self.body.layers[11][1].cell[1] = new_layer
            self.layer_names["body.layers.11.1.cell.1"] = new_layer

        elif 'body.layers.11.1.cell.1.ff1' == layer_name:
            self.body.layers[11][1].cell[1].ff1 = new_layer
            self.layer_names["body.layers.11.1.cell.1.ff1"] = new_layer

        elif 'body.layers.11.1.cell.1.ff1.dense' == layer_name:
            self.body.layers[11][1].cell[1].ff1.dense = new_layer
            self.layer_names["body.layers.11.1.cell.1.ff1.dense"] = new_layer

        elif 'body.layers.11.1.cell.1.dropout' == layer_name:
            self.body.layers[11][1].cell[1].dropout = new_layer
            self.layer_names["body.layers.11.1.cell.1.dropout"] = new_layer

        elif 'body.layers.11.1.cell.1.ff2' == layer_name:
            self.body.layers[11][1].cell[1].ff2 = new_layer
            self.layer_names["body.layers.11.1.cell.1.ff2"] = new_layer

        elif 'body.layers.11.1.cell.1.ff2.dense' == layer_name:
            self.body.layers[11][1].cell[1].ff2.dense = new_layer
            self.layer_names["body.layers.11.1.cell.1.ff2.dense"] = new_layer

        elif 'body.layers.11.1.cell.2' == layer_name:
            self.body.layers[11][1].cell[2] = new_layer
            self.layer_names["body.layers.11.1.cell.2"] = new_layer

        elif 'head' == layer_name:
            self.head = new_layer
            self.layer_names["head"] = new_layer

        elif 'norm' == layer_name:
            self.norm = new_layer
            self.layer_names["norm"] = new_layer

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



class Attention(nn.Module):
    """Attention layer implementation."""

    def __init__(self, vit_config):
        super().__init__()
        d_model = vit_config.configs.d_model
        dim_head = vit_config.configs.dim_head
        heads = vit_config.configs.heads

        initialization = vit_config.attention_init
        activation = vit_config.attention_activation
        dropout_rate = vit_config.attention_dropout_rate

        inner_dim = heads * dim_head
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(d_model, inner_dim, bias=True)
        #nn.init.xavier_uniform_(self.to_q.weight)
        self.to_k = nn.Linear(d_model, inner_dim, bias=True)
        #nn.init.xavier_uniform_(self.to_k.weight)
        self.to_v = nn.Linear(d_model, inner_dim, bias=True)
        #nn.init.xavier_uniform_(self.to_v.weight)

        self.to_out = nn.Linear(inner_dim, d_model, bias=True)
        #nn.init.xavier_uniform_(self.to_out.weight)
        self.dropout = nn.Dropout(dropout_rate)

        self.activation = activation

    def forward(self, x):
        '''x size - BxNxd_model'''
        bs, seq_len, d_model = x.shape[0], x.shape[1], x.shape[2]
        h, d = self.heads, self.dim_head

        x_2d = x.reshape(-1, d_model)
        q, k, v = self.to_q(x_2d), self.to_k(x_2d), self.to_v(x_2d)

        q = q.view(bs, seq_len, h, d).permute(0, 2, 1, 3)
        k = k.view(bs, seq_len, h, d).permute(0, 2, 1, 3)
        v = v.view(bs, seq_len, h, d).permute(0, 2, 1, 3)

        attn_scores = q @ k.transpose(-2, -1) * self.scale  # bs x h x seq_len x seq_len
        attn_scores = self.activation(attn_scores)

        out = attn_scores @ v  # bs x h x seq_len x dim_head
        out = out.permute(0, 2, 1, 3).contiguous()

        out = out.view(bs * seq_len, h * d)
        out = self.to_out(out)
        out = out.view(bs, seq_len, d_model)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """FeedForward layer implementation."""

    def __init__(self, vit_config):
        super().__init__()

        d_model = vit_config.configs.d_model
        hidden_dim = vit_config.configs.mlp_dim

        initialization = vit_config.feedforward_init
        activation = vit_config.feedforward_activation
        dropout_rate = vit_config.feedforward_dropout_rate

        self.ff1 = BatchDense(d_model, hidden_dim, initialization)
        #nn.init.xavier_uniform_(self.ff1.weight)
        self.activation = activation
        self.dropout = nn.Dropout(1.-dropout_rate)
        self.ff2 = BatchDense(hidden_dim, d_model, initialization)
        #nn.init.xavier_uniform_(self.ff2.weight)

    def forward(self, x):
        y = self.ff1(x)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.ff2(y)
        y = self.dropout(y)

        return y


class Transformer(nn.Module):
    """Transformer implementation."""

    def __init__(self, vit_config):
        super().__init__()

        depth = vit_config.configs.depth
        drop_path_rate = vit_config.body_drop_path_rate

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        att_seeds = [np.random.randint(1024) for _ in range(depth)]
        mlp_seeds = [np.random.randint(1024) for _ in range(depth)]

        layers = []
        for i in range(depth):
            normalization = vit_config.body_norm((vit_config.configs.normalized_shape,))
            normalization2 = vit_config.body_norm((vit_config.configs.normalized_shape,))
            attention = vit_config.attention(vit_config)
            feedforward = vit_config.feedforward(vit_config)

            if drop_path_rate > 0:
                layers.append(
                    nn.Sequential(
                        ResidualCell(nn.Sequential(normalization,
                                                     attention,
                                                     DropPath(dpr[i], att_seeds[i]))),
                        ResidualCell(nn.Sequential(normalization2,
                                                     feedforward,
                                                     DropPath(dpr[i], mlp_seeds[i])))
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        ResidualCell(nn.Sequential(normalization,
                                                     attention)),
                        ResidualCell(nn.Sequential(normalization2,
                                                     feedforward))
                    )
                )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)





def load_function(func_name):
    """Load function using its name."""
    modules = func_name.split(".")
    if len(modules) > 1:
        module_path = ".".join(modules[:-1])
        name = modules[-1]
        module = import_module(module_path)
        return getattr(module, name)
    return func_name


vit_cfg = namedtuple('VitCfg', [
    'd_model',
    'depth',
    'heads',
    'mlp_dim',
    'dim_head',
    'patch_size',
    'normalized_shape',
    'image_size',
    'num_classes'
])


def vit_base_patch16(args):
    """vit_base_patch16"""
    config = vit_cfg(
        d_model=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dim_head=768 // 12,
        patch_size=16,
        normalized_shape=768,
        image_size=args.train_image_size,
        num_classes=args.class_num
    )

    if args.vit_config_path != '':
        print("get vit_config_path")
        vit_config = load_function(args.vit_config_path)(config)
    else:
        print("get default_vit_cfg")
        vit_config = VitConfig(config)

    model = vit_config.network(vit_config)

    return model


def vit_base_patch32(args):
    """vit_base_patch32"""
    config = vit_cfg(
        d_model=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dim_head=768 // 12,
        patch_size=32,
        normalized_shape=768,
        image_size=args.train_image_size,
        num_classes=args.class_num
    )

    if args.vit_config_path != '':
        print("get vit_config_path")
        vit_config = load_function(args.vit_config_path)(config)
    else:
        print("get default_vit_cfg")
        vit_config = VitConfig(config)

    model = vit_config.network(vit_config)

    return model


def get_network(backbone_name, args):
    """get_network"""
    if backbone_name == 'vit_base_patch32':
        backbone = vit_base_patch32(args=args)
    elif backbone_name == 'vit_base_patch16':
        backbone = vit_base_patch16(args=args)
    else:
        raise NotImplementedError
    return backbone


def get_vit_torch():
    parameters = edict({'auto_tune': 0,
                        'autoaugment': 1,
                        'aux_factor': 0.4,
                        'backbone': 'vit_base_patch32',
                        'batch_size': 32,
                        'beta1': 0.9,
                        'beta2': 0.999,
                        'checkpoint_url': '',
                        'class_num': 1001,
                        'config_path': '/root/MSTest/models/vit/config/vit_patch32_imagenet2012_config.yml',
                        'crop_min': 0.05,
                        'data_url': '',
                        'dataset_name': 'imagenet',
                        'dataset_path': '/root/MSTest/data/ImageNet2012_val',
                        'device_id': 0,
                        'device_num': 1,
                        'enable_modelarts': 0,
                        'eval_batch_size': 256,
                        'eval_engine': 'imagenet',
                        'eval_image_size': 224,
                        'eval_interval': 1,
                        'eval_num_workers': 12,
                        'eval_offset': 0,
                        'eval_path': '/opt/npu/datasets/imagenet/val',
                        'gc_flag': 0,
                        'interpolation': 'BILINEAR',
                        'keep_checkpoint_max': 3,
                        'label_smooth_factor': 0.1,
                        'local_rank': 0,
                        'loss_name': 'ce_smooth_mixup',
                        'loss_scale': 1024,
                        'lr_decay_mode': 'cosine',
                        'lr_init': 0.0,
                        'lr_max': 0.00355,
                        'lr_min': 0.0,
                        'max_epoch': 32,
                        'mixup': 0.2,
                        'no_weight_decay_filter': 'beta,bias',
                        'open_profiler': 0,
                        'opt': 'adamw',
                        'output_path': '/cache/train',
                        'poly_power': 2,
                        'pretrained': '',
                        'save_checkpoint': 1,
                        'save_checkpoint_epochs': 8,
                        'save_checkpoint_path': './outputs',
                        'seed': 1,
                        'split_point': 0.4,
                        'train_image_size': 224,
                        'train_num_workers': 14,
                        'train_url': '',
                        'use_label_smooth': 1,
                        'vit_config_path': '',
                        'warmup_epochs': 40,
                        'weight_decay': 0.05
                        })

    config = vit_cfg(
        d_model=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dim_head=768 // 12,
        patch_size=32,
        normalized_shape=768,
        image_size=224,
        num_classes=10
    )

    vit_config = VitConfig(config)
    model = vit_config.network(vit_config)
    return model


if __name__ == '__main__':
    model = get_vit_torch()
    data = torch.randn(1, 3, 224, 224)
    print(model(data))
