
from importlib import import_module
from easydict import EasyDict as edict
import numpy as np

import mindspore
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell, Dense, Dropout, SequentialCell
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore import Tensor

MIN_NUM_PATCHES = 4

class VitConfig:
    """
    VitConfig
    """
    def __init__(self, configs):
        self.configs = configs

        # network init
        self.network_norm = mindspore.nn.LayerNorm((configs.normalized_shape,))
        self.network_init = mindspore.common.initializer.Normal(sigma=1.0)
        self.network_dropout_rate = 0.1
        self.network_pool = 'cls'
        self.network = ViT

        # stem
        self.stem_init = mindspore.common.initializer.XavierUniform()
        self.stem = VitStem

        # body
        self.body_norm = mindspore.nn.LayerNorm
        self.body_drop_path_rate = 0.1
        self.body = Transformer

        # body attention
        self.attention_init = mindspore.common.initializer.XavierUniform()
        self.attention_activation = mindspore.nn.Softmax()
        self.attention_dropout_rate = 0.1
        self.attention = Attention

        # body feedforward
        self.feedforward_init = mindspore.common.initializer.XavierUniform()
        self.feedforward_activation = mindspore.nn.GELU()
        self.feedforward_dropout_rate = 0.1
        self.feedforward = FeedForward

        # head
        self.head = origin_head
        self.head_init = mindspore.common.initializer.XavierUniform()
        self.head_dropout_rate = 0.1
        self.head_norm = mindspore.nn.LayerNorm((configs.normalized_shape,))
        self.head_activation = mindspore.nn.GELU()


class DropPath(Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0) # always be 0
        self.rand = P.UniformReal(seed=seed) # seed must be 0, if set to other value, it's not rand for multiple call
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x) # B N C
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x


class BatchDense(Cell):
    """BatchDense module."""

    def __init__(self, in_features, out_features, initialization, has_bias=True):
        super().__init__()
        self.out_features = out_features
        self.dense = Dense(in_features, out_features, has_bias=has_bias)
        # self.dense.weight.set_data(initializer(initialization, [out_features, in_features]))
        self.reshape = P.Reshape()

    def construct(self, x):
        bs, seq_len, d_model = x.shape
        out = self.reshape(x, (bs * seq_len, d_model))
        out = self.dense(out)
        out = self.reshape(out, (bs, seq_len, self.out_features))
        return out


class ResidualCell(Cell):
    """Cell which implements x + f(x) function."""
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def construct(self, x, **kwargs):
        return self.cell(x, **kwargs) + x


def pretrain_head(vit_config):
    """Head for ViT pretraining."""
    d_model = vit_config.configs.d_model
    mlp_dim = vit_config.configs.mlp_dim
    num_classes = vit_config.configs.num_classes

    dropout_rate = vit_config.head_dropout_rate
    initialization = vit_config.head_init
    normalization = vit_config.head_norm
    activation = vit_config.head_activation

    dense1 = Dense(d_model, mlp_dim)
    # dense1.weight.set_data(initializer(initialization, [mlp_dim, d_model]))
    dense2 = Dense(mlp_dim, num_classes)
    # dense2.weight.set_data(initializer(initialization, [num_classes, mlp_dim]))

    return SequentialCell([
        normalization,
        dense1,
        activation,
        Dropout(p=(1. - dropout_rate)),
        dense2])


def origin_head(vit_config):
    """Head for ViT pretraining."""
    d_model = vit_config.configs.d_model
    num_classes = vit_config.configs.num_classes
    initialization = vit_config.head_init
    dense = Dense(d_model, num_classes)
    # dense.weight.set_data(initializer(initialization, [num_classes, d_model]))
    return dense
    # return SequentialCell([dense])


class VitStem(Cell):
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
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.patch_to_embedding = BatchDense(patch_dim, d_model, initialization, has_bias=True)

    def construct(self, img):
        p = self.patch_size
        bs, channels, h, w = img.shape
        x = self.reshape(img, (bs, channels, h//p, p, w//p, p))
        x = self.transpose(x, (0, 2, 4, 1, 3, 5))
        x = self.reshape(x, (bs, (h//p )*(w//p), channels*p*p))
        x = self.patch_to_embedding(x)
        return x


class ViT(Cell):
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

        if pool == "cls":
            # self.cls_token = Parameter(initializer(initialization, (1, 1, d_model)),
            #                            name='cls', requires_grad=True)
            # self.pos_embedding = Parameter(initializer(initialization, (1, num_patches + 1, d_model)),
            #                                name='pos_embedding', requires_grad=True)
            self.tile = P.Tile()
            self.cat_1 = P.Concat(axis=1)
        else:
            # self.pos_embedding = Parameter(initializer(initialization, (1, num_patches, d_model)),
            #                                name='pos_embedding', requires_grad=True)
            self.mean = P.ReduceMean(keep_dims=False)
        self.pool = pool

        self.cast = P.Cast()
        self.dropout = Dropout(p=(1. - dropout_rate))
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






    def construct(self, img):
        x = self.stem(img)
        bs, seq_len, _ = x.shape

        # if self.pool == "cls":
        #     cls_tokens = self.tile(self.cls_token, (bs, 1, 1))
        #     x = self.cat_1((cls_tokens, q_matmul_kx)) # now x has shape = (bs, seq_len+1, d)
        #     x += self.pos_embedding[:, :(seq_len + 1)]
        # else:
        #     x += self.pos_embedding[:, :seq_len]

        y = self.cast(x, mstype.float32)
        y = self.dropout(y)
        # x = self.cast(y, x.dtype)
        x = self.cast(y, mstype.float32)
        x = self.body(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = self.mean(x, (-2,))

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




class Attention(Cell):
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
        self.scale = Tensor([dim_head ** -0.5])

        self.to_q = Dense(d_model, inner_dim, has_bias=True)
        # self.to_q.weight.set_data(initializer(initialization, [inner_dim, d_model]))
        self.to_k = Dense(d_model, inner_dim, has_bias=True)
        # self.to_k.weight.set_data(initializer(initialization, [inner_dim, d_model]))
        self.to_v = Dense(d_model, inner_dim, has_bias=True)
        # self.to_v.weight.set_data(initializer(initialization, [inner_dim, d_model]))

        self.to_out = Dense(inner_dim, d_model, has_bias=True)
        # self.to_out.weight.set_data(initializer(initialization, [inner_dim, d_model]))
        self.dropout = Dropout(1 - dropout_rate)

        self.activation = activation

        # auxiliary functions
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.q_matmul_k = P.BatchMatMul(transpose_b=True)
        self.attn_matmul_v = P.BatchMatMul()
        self.softmax_nz = True

    def construct(self, x):
        '''x size - BxNxd_model'''
        bs, seq_len, d_model, h, d = x.shape[0], x.shape[1], x.shape[2], self.heads, self.dim_head

        x_2d = self.reshape(x, (-1, d_model))
        q, k, v = self.to_q(x_2d), self.to_k(x_2d), self.to_v(x_2d)

        if self.softmax_nz:
            q = self.reshape(q, (bs, seq_len, h, d))
            q = self.transpose(q, (0, 2, 1, 3))
            q = self.cast(q, mstype.float32)
            q = self.mul(q, self.scale)

            k = self.reshape(k, (bs, seq_len, h, d))
            k = self.transpose(k, (0, 2, 1, 3))
            v = self.reshape(v, (bs, seq_len, h, d))
            v = self.transpose(v, (0, 2, 1, 3))

            # q = self.cast(q, k.dtype)
            q = self.cast(q, mstype.float32)
            attn_scores = self.q_matmul_k(q, k)  # bs x h x seq_len x seq_len
            # attn_scores = self.cast(attn_scores, x.dtype)
            attn_scores = self.cast(attn_scores, mstype.float32)
            attn_scores = self.activation(attn_scores)
        else:
            q = self.reshape(q, (bs, seq_len, h, d))
            q = self.transpose(q, (0, 2, 1, 3))
            k = self.reshape(k, (bs, seq_len, h, d))
            k = self.transpose(k, (0, 2, 1, 3))
            v = self.reshape(v, (bs, seq_len, h, d))
            v = self.transpose(v, (0, 2, 1, 3))

            attn_scores = self.q_matmul_k(q, k)  # bs x h x seq_len x seq_len
            attn_scores = self.cast(attn_scores, mstype.float32)
            attn_scores = self.mul(attn_scores, self.scale)
            # attn_scores = self.cast(attn_scores, x.dtype)
            attn_scores = self.cast(attn_scores,  mstype.float32)
            attn_scores = self.activation(attn_scores)

        out = self.attn_matmul_v(attn_scores, v)  # bs x h x seq_len x dim_head
        out = self.transpose(out, (0, 2, 1, 3))

        out = self.reshape(out, (bs*seq_len, h*d))
        out = self.to_out(out)
        out = self.reshape(out, (bs, seq_len, d_model))
        # out = self.dropout(out)
        y = self.cast(out, mstype.float32)
        y = self.dropout(y)
        # out = self.cast(y, out.dtype)
        out = self.cast(y, mstype.float32)
        # out = self.reshape(out, (bs, seq_len, d_model))
        return out


class FeedForward(Cell):
    """FeedForward layer implementation."""

    def __init__(self, vit_config):
        super().__init__()

        d_model = vit_config.configs.d_model
        hidden_dim = vit_config.configs.mlp_dim

        initialization = vit_config.feedforward_init
        activation = vit_config.feedforward_activation
        dropout_rate = vit_config.feedforward_dropout_rate

        self.ff1 = BatchDense(d_model, hidden_dim, initialization)
        self.activation = activation
        self.dropout = Dropout(p=1. - dropout_rate)
        self.ff2 = BatchDense(hidden_dim, d_model, initialization)
        self.cast = P.Cast()

    def construct(self, x):
        y = self.ff1(x)
        y = self.activation(y)
        y = self.dropout(y)

        y = self.ff2(y)

        y = self.dropout(y)

        return y


class Transformer(Cell):
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
                    SequentialCell([
                        ResidualCell(SequentialCell([normalization,
                                                     attention,
                                                     DropPath(dpr[i], att_seeds[i])])),
                        ResidualCell(SequentialCell([normalization2,
                                                     feedforward,
                                                     DropPath(dpr[i], mlp_seeds[i])]))
                    ])
                )
            else:
                layers.append(
                    SequentialCell([
                        ResidualCell(SequentialCell([normalization,
                                                     attention])),
                        ResidualCell(SequentialCell([normalization2,
                                                     feedforward]))
                    ])
                )

        self.layers = SequentialCell(layers)

    def construct(self, x):
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


vit_cfg = edict({
    'd_model': 768,
    'depth': 12,
    'heads': 12,
    'mlp_dim': 3072,
    'dim_head': 64,
    'patch_size': 32,
    'normalized_shape': 768,
    'image_size': 224,
    'num_classes': 10  # 1001,
})


def vit_base_patch16(args):
    """vit_base_patch16"""
    vit_cfg.d_model = 768
    vit_cfg.depth = 12
    vit_cfg.heads = 12
    vit_cfg.mlp_dim = 3072
    vit_cfg.dim_head = vit_cfg.d_model // vit_cfg.heads
    vit_cfg.patch_size = 16
    vit_cfg.normalized_shape = vit_cfg.d_model
    vit_cfg.image_size = args.train_image_size
    vit_cfg.num_classes = args.class_num

    if args.vit_config_path != '':
        print("get vit_config_path")
        vit_config = load_function(args.vit_config_path)(vit_cfg)
    else:
        print("get default_vit_cfg")
        vit_config = VitConfig(vit_cfg)

    model = vit_config.network(vit_config)
    return model


def vit_base_patch32(args):
    """vit_base_patch32"""
    vit_cfg.d_model = 768
    vit_cfg.depth = 12
    vit_cfg.heads = 12
    vit_cfg.mlp_dim = 3072
    vit_cfg.dim_head = vit_cfg.d_model // vit_cfg.heads
    vit_cfg.patch_size = 32
    vit_cfg.normalized_shape = vit_cfg.d_model
    vit_cfg.image_size = args.train_image_size
    vit_cfg.num_classes = args.class_num

    if args.vit_config_path != '':
        print("get vit_config_path")
        vit_config = load_function(args.vit_config_path)(vit_cfg)
    else:
        print("get default_vit_cfg")
        vit_config = VitConfig(vit_cfg)

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


def vit_ms():
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

    vit_cfg.d_model = 768
    vit_cfg.depth = 12
    vit_cfg.heads = 12
    vit_cfg.mlp_dim = 3072
    vit_cfg.dim_head = vit_cfg.d_model // vit_cfg.heads
    vit_cfg.patch_size = 32
    vit_cfg.normalized_shape = vit_cfg.d_model
    vit_cfg.image_size = 224
    vit_cfg.num_classes = 10

    vit_config = VitConfig(vit_cfg)
    model = vit_config.network(vit_config)
    return model
