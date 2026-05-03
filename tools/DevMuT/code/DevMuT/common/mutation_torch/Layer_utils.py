import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.mutation_torch.basic_op_new import new_basic_ops
from common.mutation_torch.Cascade_op_new import new_cascade_ops
import os


if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET']=='GPU':
    final_device = 'cuda:0'
else:
    final_device = 'cpu'


class Layer_helpUtils:
    def __init__(self):
        # these layers take effect both for training and testing
        self.available_model_level_layers = {}
        # these layers only take effect for training
        self.available_source_level_layers = {}
        self.is_input_legal = {}

        self.available_model_level_layers['dense'] = Layer_helpUtils.dense
        self.is_input_legal['dense'] = Layer_helpUtils.dense_input_legal
        self.available_model_level_layers['conv_1d'] = Layer_helpUtils.conv1d
        self.is_input_legal['conv_1d'] = Layer_helpUtils.conv1d_input_legal
        self.available_model_level_layers['conv_2d'] = Layer_helpUtils.conv2d
        self.is_input_legal['conv_2d'] = Layer_helpUtils.conv2d_input_legal
        self.available_model_level_layers['conv_2d_transpose'] = Layer_helpUtils.conv_2d_transpose
        self.is_input_legal['conv_2d_transpose'] = Layer_helpUtils.conv_2d_transpose_input_legal
        self.available_model_level_layers['conv_3d'] = Layer_helpUtils.conv_3d
        self.is_input_legal['conv_3d'] = Layer_helpUtils.conv_3d_input_legal
        self.available_model_level_layers['conv_3d_transpose'] = Layer_helpUtils.conv_3d_transpose
        self.is_input_legal['conv_3d_transpose'] = Layer_helpUtils.conv_3d_transpose_input_legal
        self.available_model_level_layers['max_pooling_1d'] = Layer_helpUtils.max_pooling_1d
        self.is_input_legal['max_pooling_1d'] = Layer_helpUtils.max_pooling_1d_input_legal
        self.available_model_level_layers['max_pooling_2d'] = Layer_helpUtils.max_pooling_2d
        self.is_input_legal['max_pooling_2d'] = Layer_helpUtils.max_pooling_2d_input_legal

        self.available_model_level_layers['average_pooling_1d'] = Layer_helpUtils.average_pooling_1d
        self.is_input_legal['average_pooling_1d'] = Layer_helpUtils.average_pooling_1d_input_legal
        self.available_model_level_layers['average_pooling_2d'] = Layer_helpUtils.average_pooling_2d
        self.is_input_legal['average_pooling_2d'] = Layer_helpUtils.average_pooling_2d_input_legal
        self.available_model_level_layers['batch_normalization1d'] = Layer_helpUtils.batch_normalization1d
        self.available_model_level_layers['batch_normalization2d'] = Layer_helpUtils.batch_normalization2d
        self.available_model_level_layers['batch_normalization3d'] = Layer_helpUtils.batch_normalization3d
        self.is_input_legal['batch_normalization'] = Layer_helpUtils.batch_normalization_input_legal

        self.available_model_level_layers['leaky_relu_layer'] = Layer_helpUtils.leaky_relu_layer
        self.is_input_legal['leaky_relu_layer'] = Layer_helpUtils.leaky_relu_layer_input_legal
        self.available_model_level_layers['prelu_layer'] = Layer_helpUtils.prelu_layer
        self.is_input_legal['prelu_layer'] = Layer_helpUtils.prelu_layer_input_legal
        self.available_model_level_layers['elu_layer'] = Layer_helpUtils.elu_layer
        self.is_input_legal['elu_layer'] = Layer_helpUtils.elu_layer_input_legal
        self.available_model_level_layers['thresholded_relu_layer'] = Layer_helpUtils.threshold_layer
        self.is_input_legal['thresholded_relu_layer'] = Layer_helpUtils.threshold_layer_input_legal
        self.available_model_level_layers['softmax_layer'] = Layer_helpUtils.softmax_layer
        self.is_input_legal['softmax_layer'] = Layer_helpUtils.softmax_layer_input_legal
        self.available_model_level_layers['relu_layer'] = Layer_helpUtils.relu_layer
        self.is_input_legal['relu_layer'] = Layer_helpUtils.relu_layer_input_legal

    def is_layer_in_weight_change_white_list(self, layer):
        white_list = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                      nn.ConvTranspose2d, nn.ConvTranspose3d,
                      nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                      nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                      nn.LeakyReLU, nn.ELU, nn.Threshold,
                      nn.Softmax, nn.ReLU
                      ]
        for l in white_list:
            if isinstance(layer, l):
                return True
        return False

    @staticmethod
    def clone(layer):
        pass

    @staticmethod
    def dense():
        layer = nn.Linear(in_features=3, out_features=3)
        return layer

    @staticmethod
    def dense_input_legal():
        pass

    @staticmethod
    def conv1d():
        layer = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=1)
        return layer

    @staticmethod
    def conv1d_input_legal():
        pass

    @staticmethod
    def conv2d():
        layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 1))
        return layer

    @staticmethod
    def conv2d_input_legal():
        pass

    @staticmethod
    def conv_2d_transpose():
        layer = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        return layer

    @staticmethod
    def conv_2d_transpose_input_legal():
        pass

    @staticmethod
    def conv_3d():
        layer = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1))
        return layer

    @staticmethod
    def conv_3d_input_legal():
        pass

    @staticmethod
    def conv_3d_transpose():
        layer = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1))
        return layer

    @staticmethod
    def conv_3d_transpose_input_legal():
        pass

    @staticmethod
    def max_pooling_1d():
        layer = nn.MaxPool1d(kernel_size=1)
        return layer

    @staticmethod
    def max_pooling_1d_input_legal():
        pass

    @staticmethod
    def max_pooling_2d():
        layer = nn.MaxPool2d(kernel_size=1)
        return layer

    @staticmethod
    def max_pooling_2d_input_legal():
        pass

    @staticmethod
    def average_pooling_1d():
        layer = nn.AvgPool1d(kernel_size=1)
        return layer

    @staticmethod
    def average_pooling_1d_input_legal():
        pass

    @staticmethod
    def average_pooling_2d():
        layer = nn.AvgPool2d(kernel_size=1)
        return layer

    @staticmethod
    def average_pooling_2d_input_legal():
        pass

    @staticmethod
    def batch_normalization1d():
        layer = nn.BatchNorm1d(num_features=10)
        return layer

    @staticmethod
    def batch_normalization2d():
        layer = nn.BatchNorm2d(num_features=10)
        return layer

    @staticmethod
    def batch_normalization3d():
        layer = nn.BatchNorm3d(num_features=10)
        return layer

    @staticmethod
    def batch_normalization_input_legal():
        pass

    @staticmethod
    def leaky_relu_layer():
        layer = nn.LeakyReLU()
        return layer

    @staticmethod
    def leaky_relu_layer_input_legal():
        pass

    @staticmethod
    def prelu_layer():
        layer = nn.PReLU()
        return layer

    @staticmethod
    def prelu_layer_input_legal():
        pass

    @staticmethod
    def elu_layer():
        layer = nn.ELU()
        return layer

    @staticmethod
    def elu_layer_input_legal():
        pass

    @staticmethod
    def threshold_layer():
        layer = nn.Threshold(threshold=0.1, value=20)
        return layer

    @staticmethod
    def threshold_layer_input_legal():
        pass

    @staticmethod
    def softmax_layer():
        layer = nn.Softmax()
        return layer

    @staticmethod
    def softmax_layer_input_legal():
        pass

    @staticmethod
    def relu_layer():
        layer = nn.ReLU()
        return layer

    @staticmethod
    def relu_layer_input_legal():
        pass


class dwpw_basic(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, depthwise, activation='ReLU6'):
        super(dwpw_basic, self).__init__()
        self.dwpw_conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride,
                                   groups=1 if not depthwise else in_channel)
        self.dwpw_bn = nn.BatchNorm2d(out_channel)
        if activation:
            self.dwpw_activation = self.get_activation(activation.lower())

    def get_activation(self, name):
        return BasicOPUtils.available_activations()[name]

    def forward(self, x):
        x = self.dwpw_conv(x)
        x = self.dwpw_bn(x)
        x = self.dwpw_activation(x)
        return x


class dwpw_group(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, activation='relu6'):
        super(dwpw_group, self).__init__()

        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.activation = in_channel, \
            out_channel, kernel_size, stride, activation
        self.depthwise = dwpw_basic(in_channel, out_channel, kernel_size, stride, True, activation)  # Conv2_depthwise
        self.pointwise = dwpw_basic(out_channel, out_channel, kernel_size, stride, False, activation)  # Conv2_pointwise

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class convbnrelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(convbnrelu, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride = in_channels, out_channels, \
            kernel_size, stride
        self.conbnrelu_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conbnrelu_bn = nn.BatchNorm2d(out_channels)
        self.conbnrelu_relu = nn.ReLU()

    def forward(self, x):
        x = self.conbnrelu_conv(x)
        x = self.conbnrelu_bn(x)
        x = self.conbnrelu_relu(x)
        return x


class downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(downsample, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride = in_channels, out_channels, \
            kernel_size, stride
        self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.downsample_conv(x)
        x = self.downsample_bn(x)
        return x


def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    """_conv3x3"""
    if res_base:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            stride=stride,
            padding=1,
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=1,
        stride=stride,
        padding=0,
    )


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    """_conv1x1"""
    if res_base:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            stride=stride,
            padding=0,
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=1,
        stride=stride,
        padding=0,
    )


def _conv7x7(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    """_conv7x7"""
    if res_base:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=7,
            stride=stride,
            padding=3,
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=7,
        stride=stride,
        padding=0,
    )


def _bn(channel, res_base=False):
    """_bn"""
    if res_base:
        return nn.BatchNorm2d(
            channel,
            eps=1e-5,
            momentum=0.1,
        )
    return nn.BatchNorm2d(
        channel,
        eps=1e-4,
        momentum=0.9,
    )


def _bn_last(channel):
    """_bn_last"""
    return nn.BatchNorm2d(
        channel,
        eps=1e-4,
        momentum=0.9,
    )


def _fc(in_channel, out_channel, use_se=False):
    """_fc"""
    return nn.Linear(
        in_channel, out_channel, bias=True,  # weight_init=weight,
    )


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.5):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # inplace floor operation
        x.div_(keep_prob)
        x.mul_(random_tensor)
        return x


class BasicOPUtils:
    def __init__(self):
        self.extension_ops = BasicOPUtils.available_activations()
        self.activation_names = self.available_activations().keys()
        self.extension_ops.update(new_basic_ops)
        self.extension_ops['conv'] = BasicOPUtils.copy_convs
        self.extension_ops['pool'] = BasicOPUtils.copy_pool
        self.extension_ops['batchnorm'] = BasicOPUtils.copy_BN
        self.extension_ops['linear'] = BasicOPUtils.available_Dense
        self.extension_ops['dropout'] = BasicOPUtils.available_Dropout
        self.extension_ops['embedding'] = BasicOPUtils.available_embedding

    @staticmethod
    def available_activations():
        activations = {}
        activations['relu'] = nn.ReLU().to(final_device)
        activations['relu6'] = nn.ReLU6().to(final_device)
        activations['tanh'] = nn.Tanh().to(final_device)
        activations['sigmoid'] = nn.Sigmoid().to(final_device)
        activations['leakyrelu'] = nn.LeakyReLU().to(final_device)
        activations['elu'] = nn.ELU().to(final_device)
        activations['gelu'] = nn.GELU().to(final_device)
        activations['mish'] = nn.Mish().to(final_device)
        activations['softmax'] = nn.Softmax().to(final_device)
        return activations

    @staticmethod
    def copy_convs(**kwargs):

        in_channel, out_channel, kernel_size, stride, groups, bias, name = kwargs['param1'], kwargs['param2'], kwargs['param3'], kwargs['param4'], kwargs['param5'], kwargs['param6'], kwargs['param7']
        name = name.lower()
        if "1d" in name:
            if "transpose" in name:
                nn.ConvTranspose1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                                   groups=groups).to(final_device)
            else:
                nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                          groups=groups).to(final_device)
        if "2d" in name:
            if isinstance(kernel_size, tuple):
                padding_value = int((kernel_size[0] - 1) // 2)
            else:
                padding_value = int((kernel_size - 1) // 2)
            if "transpose" in name:
                return nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                                          groups=groups, padding=padding_value).to(final_device)
            else:
                return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                                 groups=groups, padding=padding_value).to(final_device)
        if "3d" in name:
            if "transpose" in name:
                return nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                                 groups=groups).to(final_device)
            else:
                return nn.ConvTranspose3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                                          groups=groups).to(final_device)

    @staticmethod
    def available_convs(in_channel, out_channel, kernel_size, stride, name):
        name = name.lower()
        if "1d" in name:
            convs = [nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride).to(final_device),
                     nn.ConvTranspose1d(in_channel, out_channel, kernel_size=kernel_size,
                                        stride=stride).to(final_device)]
            return convs
        if "2d" in name:
            padding_value = int((kernel_size - 1) // 2)
            convs = [
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding_value).to(
                    final_device),
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size,
                                   stride=stride, padding=padding_value).to(final_device)]
            return convs
        if "3d" in name:
            convs = [nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride).to(final_device),
                     nn.ConvTranspose3d(in_channel, out_channel, kernel_size=kernel_size,
                                        stride=stride).to(final_device)]
            return convs

    @staticmethod
    def available_embedding(**kwargs):
        vocab_size, embedding_size, name = kwargs['param1'], kwargs['param2'], kwargs['param3']
        name = name.lower()
        if "embedding" == name:
            return nn.Embedding(vocab_size, embedding_size).to(final_device)
        elif "embeddinglookup" == name:
            return nn.EmbeddingLookup(vocab_size, embedding_size).to(final_device)

    @staticmethod
    def available_Dense(**kwargs):
        in_feature, out_feature, bias = kwargs['param1'], kwargs['param2'], kwargs['param3']
        return nn.Linear(in_feature, out_feature, bias=bias).to(final_device)

    @staticmethod
    def available_Dropout(**kwargs):
        return nn.Dropout(kwargs['param1']).to(final_device)

    @staticmethod
    def available_Droppath(p=0.5):
        return DropPath(drop_prob=p)

    @staticmethod
    def available_BN(**kwargs):
        num_features, name = kwargs['param1'], kwargs['param2']
        name = name.lower()
        if "1d" in name:
            return nn.BatchNorm1d(num_features).to(final_device)
        if "2d" in name:
            return nn.BatchNorm2d(num_features).to(final_device)
        if "3d" in name:
            return nn.BatchNorm3d(num_features).to(final_device)

    @staticmethod
    def copy_BN(**kwargs):
        num_features, eps, momentum, name = kwargs['param1'], kwargs['param2'], kwargs['param3'], kwargs['param4']
        name = name.lower()
        if "1d" in name:
            return nn.BatchNorm1d(num_features, eps, momentum).to(final_device)
        if "2d" in name:
            return nn.BatchNorm2d(num_features, eps, momentum).to(final_device)
        if "3d" in name:
            return nn.BatchNorm3d(num_features, eps, momentum).to(final_device)

    @staticmethod
    def available_LN(shape_list):
        shape_list_nobatch = shape_list[1:]
        return nn.LayerNorm(shape_list_nobatch).to(final_device)

    @staticmethod
    def copy_pool(**kwargs):
        output_size, stride, name = kwargs['param1'], kwargs['param2'], kwargs['param3']
        name = name.lower()
        if "1d" in name:
            if "avg" in name:
                if "adaptive" in name:
                    return nn.AdaptiveAvgPool1d(output_size).to(final_device)
                else:
                    return nn.AvgPool1d(output_size, stride).to(final_device)

            elif "max" in name:
                if "adaptive" in name:
                    nn.AdaptiveMaxPool1d(output_size).to(final_device)
                else:
                    nn.MaxPool1d(output_size, stride).to(final_device)

        if "2d" in name:
            if "adaptive" in name and "avg" in name:
                return nn.AdaptiveAvgPool2d(output_size).to(final_device)
            elif "adaptive" not in name and "avg" in name:
                return nn.AvgPool2d(output_size, stride).to(final_device)
            if "adaptive" in name and "max" in name:
                return nn.AdaptiveMaxPool2d(output_size).to(final_device)
            elif "adaptive" not in name and "max" in name:
                return nn.MaxPool2d(output_size, stride).to(final_device)

    @staticmethod
    def available_pool(output_size, stride, name):
        name = name.lower()
        if "1d" in name:
            pools = [nn.AvgPool1d(output_size, stride).to(final_device), nn.MaxPool1d(output_size, stride).to(
                final_device)]  # nn.AdaptiveAvgPool1d(output_size),nn.AdaptiveMaxPool1d(output_size),

            return pools
        if "2d" in name:
            pools = [nn.AvgPool2d(output_size, stride).to(final_device), nn.MaxPool2d(output_size,
                                                                                      stride).to(final_device)]  # nn.AdaptiveAvgPool2d(output_size),nn.AdaptiveMaxPool2d(output_size),

            return pools

    @staticmethod
    def available_flatten():
        return nn.Flatten().to(final_device)

    @staticmethod
    def copy_basicop(original_layer):
        layer_type = str(original_layer.__class__.__name__)
        activations = BasicOPUtils.available_activations()

        if "conv" in layer_type.lower():
            return BasicOPUtils.copy_convs(param1=original_layer.out_channels, param2=original_layer.out_channels,
                                           param3=original_layer.kernel_size, param4=original_layer.stride, param5=original_layer.groups,
                                           param6=original_layer.bias, param7=layer_type)
        elif "pool" in layer_type.lower():
            return BasicOPUtils.copy_pool(param1=original_layer.kernel_size, param2=original_layer.stride, param3=layer_type)

        elif "batchnorm" in layer_type.lower():
            return BasicOPUtils.copy_BN(param1=original_layer.num_features, param2=original_layer.eps, param3=original_layer.momentum,
                                        param4=layer_type)

        elif "dense" in layer_type.lower():
            return BasicOPUtils.available_Dense(param1=original_layer.in_features, param2=original_layer.out_features,
                                                param3=original_layer.bias)
        elif "dropout" in layer_type.lower():
            return BasicOPUtils.available_Dropout(param1=original_layer.p)

        elif layer_type.lower() in list(activations.keys()):
            return activations[layer_type.lower()]


class CascadeOPUtils:
    def __init__(self):
        self.extension_ops = {
            "convbnrelu": CascadeOPUtils.avaiable_convbnrelu,
            "downsample": CascadeOPUtils.avaiable_downSample,
            "dwpw_group": CascadeOPUtils.avaiable_dwpw,
            "se": CascadeOPUtils.avaiable_se,
            "denselayer": CascadeOPUtils.avaiable_de,
            "residualblock": CascadeOPUtils.avaiable_resnetbasicBlock,
            "pwdwpw_residualblock": CascadeOPUtils.avaiable_PWDWPW_ResidualBlock,
            "inception": CascadeOPUtils.avaiable_Inception
        }
        self.extension_ops.update(new_cascade_ops)

    @staticmethod
    def get_cascadeops_names():
        return ["convbnrelu", "downsample", "dwpw_group", "se", "denselayer", "residualblock", "pwdwpw_residualblock",
                "inception"]

    @staticmethod
    def avaiable_convbnrelu(in_channels, out_channels, kernel_size, stride):
        op = convbnrelu(in_channels, out_channels, kernel_size, stride).to(final_device)
        return op

    @staticmethod
    def avaiable_downSample(in_channels, out_channels, kernel_size, stride):
        op = downsample(in_channels, out_channels, kernel_size, stride).to(final_device)
        return op

    @staticmethod
    def avaiable_dwpw(in_channel, out_channel, kernel_size, stride, activation='relu6'):
        op = dwpw_group(in_channel, out_channel, kernel_size, stride, activation).to(final_device)
        return op

    @staticmethod
    def avaiable_resnetbasicBlock(in_channel, out_channel, kernel_size, stride, activation='relu6'):
        return ResidualBlock(in_channel, out_channel, stride).to(final_device)

    @staticmethod
    def avaiable_invertedResidualBlock(in_channel, out_channel, kernel_size, stride, activation='relu6'):
        return InvertedResidual(in_channel, out_channel, kernel_size, stride).to(final_device)

    @staticmethod
    def avaiable_PWDWPW_ResidualBlock(in_channel, out_channel, kernel_size, stride, activation):
        return PWDWPW_ResidualBlock(in_channel, out_channel, kernel_size, stride, activation).to(final_device)

    @staticmethod
    def avaiable_ResidualBlock(in_channel, out_channel, kernel_size, stride, activation='relu6'):
        residual = CascadeOPUtils.avaiable_resnetbasicBlock(in_channel, out_channel, kernel_size, stride).to(
            final_device)
        pdp_residual = CascadeOPUtils.avaiable_PWDWPW_ResidualBlock(in_channel, out_channel, kernel_size, stride,
                                                                    activation).to(final_device)

        residual_values = [residual, pdp_residual]
        residual_values = np.random.permutation(residual_values)
        return residual_values

    @staticmethod
    def avaiable_se():
        return SE()

    @staticmethod
    def avaiable_de(in_c, out_c):
        return DenseLayer(in_c, out_c)

    @staticmethod
    def avaiable_Inception():
        inception = {}
        inception['inceptionA'] = Inception_A()
        return inception['inceptionA']


class PWDWPW_ResidualBlock(nn.Module):
    """
    Pointwise - -Depthwise - -Pointwise - -Add
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride, activation):
        super(PWDWPW_ResidualBlock, self).__init__()

        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.activation = \
            in_channel, out_channel, kernel_size, stride, activation
        self.PDP_ResidualBlock_1 = dwpw_basic(in_channel, out_channel, kernel_size, stride, False, activation)
        self.PDP_ResidualBlock_2 = dwpw_basic(out_channel, out_channel, kernel_size, stride, True, activation)
        self.PDP_ResidualBlock_3 = dwpw_basic(out_channel, in_channel, kernel_size, stride, False, activation)
        self.add = torch.add

    def forward(self, x):
        identity = x
        out1 = self.PDP_ResidualBlock_1(x)
        out2 = self.PDP_ResidualBlock_2(out1)
        out2 = self.PDP_ResidualBlock_3(out2)
        out = self.add(out2, identity)
        return out


class ResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()

        self.in_channels, self.out_channels = in_channel, out_channel
        self.stride = stride
        self.use_se = False
        self.se_block = False
        channel = out_channel // self.expansion
        self.residual_conv1 = _conv1x1(in_channel, channel, stride=1, use_se=self.use_se)

        self.kernel_size = self.residual_conv1.kernel_size

        self.residual_bn1 = _bn(channel)
        if self.use_se and self.stride != 1:
            self.residual_e2 = nn.Sequential(_conv3x3(channel, channel, stride=1, use_se=True), _bn(channel),
                                             nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.residual_conv2 = _conv3x3(channel, channel, stride=stride, use_se=self.use_se)
            self.residual_bn2 = _bn(channel)

        self.residual_conv3 = _conv1x1(channel, out_channel, stride=1, use_se=self.use_se)
        self.residual_bn3 = _bn(out_channel)

        if self.se_block:
            self.residual_se_global_pool = torch.mean
            self.residual_se_dense_0 = _fc(out_channel, int(out_channel / 4), use_se=self.use_se)
            self.residual_se_dense_1 = _fc(int(out_channel / 4), out_channel, use_se=self.use_se)
            self.residual_se_sigmoid = nn.Sigmoid()
            self.residual_se_mul = torch.mul
        self.residual_relu1 = nn.ReLU()
        self.residual_relu2 = nn.ReLU()
        self.residual_relu3 = nn.ReLU()
        self.residual_down_sample = True

        if stride != 1 or in_channel != out_channel:
            self.residual_down_sample = True
        self.residual_down_sample_layer = None

        if self.residual_down_sample:
            if self.use_se:
                if stride == 1:
                    self.residual_down_sample_layer = nn.Sequential(_conv1x1(in_channel, out_channel,
                                                                             stride, use_se=self.use_se),
                                                                    _bn(out_channel))
                else:
                    self.residual_down_sample_layer = nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        _conv1x1(in_channel, out_channel, 1,
                                 use_se=self.use_se), _bn(out_channel))
            else:
                self.residual_down_sample_layer = nn.Sequential(_conv1x1(in_channel, out_channel, stride,
                                                                         use_se=self.use_se), _bn(out_channel))

    def forward(self, x):
        identity = x
        out = self.residual_conv1(x)
        out = self.residual_bn1(out)
        out = self.residual_relu1(out)

        out = self.residual_conv2(out)
        out = self.residual_bn2(out)
        out = self.residual_relu2(out)

        out = self.residual_conv3(out)
        out = self.residual_bn3(out)

        identity = self.residual_down_sample_layer(identity)
        out = out + identity
        out = self.residual_relu3(out)
        return out


class InvertedResidual(nn.Module):
    """
    Mobilenetv2 residual block definition.
    """

    def __init__(self, inp, oup, stride, expand_ratio=2):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        self.InvertedResidual_ConvBNReLU = InvertedResidual_ConvBNReLU(inp, hidden_dim, kernel_size=1)
        self.InvertedResidual_conv = nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=1)
        self.InvertedResidual_bn = nn.BatchNorm2d(oup)
        self.add = torch.add

    def forward(self, x):
        identity = x
        x = self.InvertedResidual_ConvBNReLU(x)
        x = self.InvertedResidual_conv(x)
        x = self.InvertedResidual_bn(x)
        out = self.add(identity, x)
        return out


class InvertedResidual_ConvBNReLU(nn.Module):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groupss=1):
        super(InvertedResidual_ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        in_channels = in_planes
        out_channels = out_planes
        if groupss == 1:
            self.InvertedResidual_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                                   padding=padding)
        else:
            out_channels = in_planes
            self.InvertedResidual_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                                   padding=padding, groups=in_channels)

        self.InvertedResidual_bn = nn.BatchNorm2d(out_planes)
        self.InvertedResidual_relu = nn.ReLU6()

    def forward(self, x):
        x = self.InvertedResidual_conv(x)
        x = self.InvertedResidual_bn(x)
        output = self.InvertedResidual_relu(x)
        return output


class ResUnit(nn.Module):
    """
    ResUnit warpper definition.
    """

    def __init__(self, num_in, num_mid, num_out, kernel_size, stride=1, act_type='relu', use_se=False):
        super(ResUnit, self).__init__()
        self.use_se = use_se
        self.first_conv = (num_out != num_mid)
        self.use_short_cut_conv = True

        if self.first_conv:
            self.expand = Unit(num_in, num_mid, kernel_size=1,
                               stride=1, padding=0, act_type=act_type)
        else:
            self.expand = None
        self.conv1 = Unit(num_mid, num_mid, kernel_size=kernel_size, stride=stride,
                          padding=self._get_pad(kernel_size), act_type=act_type, num_groups=num_mid)
        if use_se:
            self.se = SE(num_mid)
        self.conv2 = Unit(num_mid, num_out, kernel_size=1, stride=1,
                          padding=0, act_type=act_type, use_act=False)
        if num_in != num_out or stride != 1:
            self.use_short_cut_conv = False
        self.add = torch.add if self.use_short_cut_conv else None

    def forward(self, x):
        """forward"""
        if self.first_conv:
            out = self.expand(x)
        else:
            out = x
        out = self.conv1(out)
        if self.use_se:
            out = self.se(out)
        out = self.conv2(out)
        if self.use_short_cut_conv:
            out = self.add(x, out)
        return out

    def _get_pad(self, kernel_size):
        """set the padding number"""
        pad = 0
        if kernel_size == 1:
            pad = 0
        elif kernel_size == 3:
            pad = 1
        elif kernel_size == 5:
            pad = 2
        elif kernel_size == 7:
            pad = 3
        else:
            raise NotImplementedError
        return pad


class Unit(nn.Module):
    """
    Unit warpper definition.
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, num_groups=1,
                 use_act=True, act_type='relu'):
        super(Unit, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_in,
                              out_channels=num_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=num_groups,
                              )
        self.bn = nn.BatchNorm2d(num_out)
        self.use_act = use_act
        self.act = self.Activation(act_type) if use_act else None

    def Activation(self, act_func):
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'relu6':
            self.act = nn.ReLU6()
        elif act_func in ('hsigmoid', 'hard_sigmoid'):
            self.act = nn.HSigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.HSwish()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class GlobalAvgPooling(nn.Module):
    """
    Global avg pooling definition.
    """

    def __init__(self, keep_dims=False):
        super(GlobalAvgPooling, self).__init__()
        self.mean = torch.mean
        self.keep_dims = keep_dims

    def forward(self, x):
        x = self.mean(x, (2, 3), keepdim=self.keep_dims)
        return x


class SE(nn.Module):
    """
    SE warpper definition.
    """

    def __init__(self):
        super(SE, self).__init__()
        self.SE_pool = GlobalAvgPooling(keep_dims=True).to(final_device)
        self.SE_act1 = self.Activation('relu').to(final_device)
        self.SE_act2 = self.Activation('hsigmoid').to(final_device)
        self.SE_mul = torch.mul

    def _make_divisible(self, x, divisor=8):
        return int(np.ceil(x * 1. / divisor) * divisor)

    def Activation(self, act_func):
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'relu6':
            self.act = nn.ReLU6()
        elif act_func in ('hsigmoid', 'hard_sigmoid'):
            self.act = nn.Hardsigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.Hardswish()
        else:
            raise NotImplementedError
        return self.act

    def forward(self, x):
        out = self.SE_pool(x)
        conv2out = torch.tensor(np.random.randn(out.shape[0], out.shape[1], 1, 1).astype(np.float32),
                                dtype=torch.float32).to(final_device)
        out = F.conv2d(out, weight=conv2out)

        out = self.SE_act1(out)
        conv2out_1 = torch.tensor(np.random.randn(out.shape[0], out.shape[1], 1, 1).astype(np.float32),
                                  dtype=torch.float32).to(final_device)
        out = F.conv2d(out, weight=conv2out_1)
        out = self.SE_act2(out)
        return out


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, out_channels, drop_rate=0.5):
        super(DenseLayer, self).__init__()

        self.in_channels, self.out_channels = num_input_features, out_channels
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(nn.BatchNorm2d(num_input_features),
                                         nn.ReLU(),
                                         nn.Conv2d(num_input_features, out_channels, kernel_size=1,
                                                   stride=1, padding_mode="zeros", bias=True, groups=1),
                                         nn.BatchNorm2d(out_channels),  # pw
                                         nn.ReLU(),
                                         nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1,
                                                   padding_mode="zeros", bias=False,
                                                   groups=out_channels)).to(final_device)  # dw

    def forward(self, x):
        new_features = self.dense_layer(x)
        if self.drop_rate > 0:
            new_features = nn.Dropout(p=self.drop_rate).to(final_device)(new_features)
        return torch.cat([x, new_features], 1)


class BasicConv2d(nn.Module):
    def __init__(self):
        super(BasicConv2d, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        feature_weight = torch.tensor(
            np.random.randn(x.shape[0], x.shape[1], 1, 1).astype(np.float32)).to(final_device)
        x = F.conv2d(x, feature_weight, stride=1, padding="same")
        in_shape_1 = x.shape[1]
        x = nn.BatchNorm2d(in_shape_1).to(final_device)(x)
        x = self.relu(x)
        return x


class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d().to(final_device)
        self.branch1 = nn.Sequential(
            BasicConv2d(),
            BasicConv2d()).to(final_device)
        self.branch2 = nn.Sequential(
            BasicConv2d(),
            BasicConv2d(),
            BasicConv2d()).to(final_device)
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=1),
            BasicConv2d()).to(final_device)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = torch.cat((x0, x1, x2, branch_pool), dim=1)
        return out


class ops2Cell(nn.Module):
    def __init__(self, op2convert):
        super(ops2Cell, self).__init__()
        self.opcell = op2convert

    def forward(self, x):
        return self.opcell()(x)


class EmptyCell(nn.Module):
    def __init__(self):
        super(EmptyCell, self).__init__()
        self.empty_cell = nn.Module()

    def forward(self, x):
        return x


class ops_reshape(nn.Module):
    def __init__(self, out_shape):
        super(ops_reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return torch.reshape(x, shape=self.out_shape)


class ops_concat(nn.Module):
    def __init__(self, axis):
        super(ops_concat, self).__init__()
        self.axis = axis

    def forward(self, x1, x2):
        x1, x2 = x1.to(final_device), x2.to(final_device)
        return torch.concat((x1, x2), axis=self.axis)


class CM_branchCell(nn.Module):
    def __init__(self, origin_op, insert_op, input_shape, out_shape):
        super(CM_branchCell, self).__init__()

        self.branch1CM = origin_op
        self.branch2CM = insert_op
        self.branchflattenCM = nn.Flatten()
        self.branchconcatCM = ops_concat(1)
        self.out_shape = tuple(out_shape)
        self.input_shape = tuple(input_shape)
        self.reshapeCM = ops_reshape(self.out_shape)
        self.output_fix = CM_replacehelper()

        pro = 1
        for val in out_shape:
            pro *= val
        self.product = pro

    def forward(self, x):
        x1 = self.branch1CM(x)
        x2 = self.branch2CM(x)
        x1 = self.branchflattenCM(x1)
        x2 = self.branchflattenCM(x2)
        out = self.branchconcatCM(x1, x2)
        out = self.output_fix(out, self.out_shape)
        return out


class CM_replacehelper(nn.Module):
    def __init__(self):
        super(CM_replacehelper, self).__init__()

    def forward(self, input_data, target_shape):
        flatten = torch.nn.Flatten()
        elements = get_element_nums(input_data)
        b1 = input_data.shape[0]
        s1 = elements // b1
        target_elements = 1
        for shape in target_shape:
            target_elements *= shape
        b2 = target_shape[0]
        s2 = target_elements // b2
        det = elements - target_elements
        if det == 0:
            return torch.reshape(input_data, target_shape)

        x1 = s2 - s1
        x2 = b2 - b1
        out = flatten(input_data)
        if x1 > 0:
            out = torch.nn.functional.pad(out, (0, x1))
        elif x1 < 0:
            out = torch.index_select(out, dim=-1,
                                     index=torch.tensor([i for i in range(s2)], dtype=torch.int64).to(final_device))

        if x2 > 0:
            out = torch.transpose(out, dim0=1, dim1=0)
            out = torch.nn.functional.pad(out, (0, x2))
            out = torch.transpose(out, dim1=1, dim0=0)
        elif x2 < 0:
            out = torch.index_select(out, dim=0,
                                     index=torch.tensor([i for i in range(b2)], dtype=torch.int64).to(final_device))
        return torch.reshape(out, target_shape)


class dtypecast(torch.nn.Module):
    def __init__(self, dtype):
        super(dtypecast, self).__init__()
        self.dtype = dtype

    def forward(self, x):

        return x.to(self.dtype)


class Replace_torch(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Replace_torch, self).__init__()
        self.in_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input_data):
        flatten = torch.nn.Flatten()
        elements = get_element_nums(input_data)
        b1 = input_data.shape[0]
        s1 = elements // b1
        target_elements = 1
        for shape in self.output_shape:
            target_elements *= shape
        b2 = self.output_shape[0]
        s2 = target_elements // b2
        det = elements - target_elements
        if det == 0:
            result = torch.reshape(input_data, self.output_shape)
            return torch.tensor(result).to(final_device)

        x1 = s2 - s1
        x2 = b2 - b1
        out = flatten(input_data)
        if x1 > 0:
            out = torch.nn.functional.pad(out, (0, x1))
        elif x1 < 0:
            out = torch.index_select(out, dim=-1, index=torch.tensor([i for i in range(s2)], dtype=torch.int64).to(final_device))

        if x2 > 0:
            out = torch.transpose(out, dim0=1, dim1=0)
            out = torch.nn.functional.pad(out, (0, x2))
            out = torch.transpose(out, dim1=1, dim0=0)
        elif x2 < 0:
            out = torch.index_select(out, dim=0,index=torch.tensor([i for i in range(b2)], dtype=torch.int64).to(final_device))
        result = torch.reshape(out, self.output_shape)
        return torch.as_tensor(result).to(final_device)


def get_element_nums(data):
    if isinstance(data, tuple):
        data = list(tuple)
    shape = data.shape
    res = 1
    for i in shape:
        res *= i
    return res
