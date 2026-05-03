import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
import mindspore.ops as ops
import mindspore


class SigmoidFocalClassificationLoss(nn.Cell):
    """"
    Sigmoid focal-loss for classification.

    Args:
        gamma (float): Hyper-parameter to balance the easy and hard examples. Default: 2.0
        alpha (float): Hyper-parameter to balance the positive and negative example. Default: 0.25

    Returns:
        Tensor, the focal loss.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.sigmiod_cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.sigmoid = ops.Sigmoid()
        self.pow = ops.Pow()
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.gamma = gamma
        self.alpha = alpha

    def construct(self, logits, label):
        label = self.onehot(label, ops.shape(logits)[-1], self.on_value, self.off_value)
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        sigmoid = self.sigmoid(logits)
        label = ops.cast(label, ms.float32)
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = self.pow(1 - p_t, self.gamma)
        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss


def _make_divisible(v, divisor, min_value=None):
    """nsures that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='pad', padding=0):
    padding = (kernel_size - 1) // 2 if padding == 0 else padding
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                     padding=padding, pad_mode=pad_mod, has_bias=True)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.97,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _last_conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same', pad=0):
    in_channels = in_channel
    out_channels = in_channel
    depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same',
                               padding=pad, group=in_channels)
    conv = _conv2d(in_channel, out_channel, kernel_size=1)
    return nn.SequentialCell([depthwise_conv, _bn(in_channel), nn.ReLU6(), conv])


class ConvBNReLU(nn.Cell):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.
        shared_conv(Cell): Use the weight shared conv, default: None.

    Returns:
        Tensor, output tensor.

    Examples:
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, shared_conv=None):
        super(ConvBNReLU, self).__init__()
        padding = kernel_size // 2
        in_channels = in_planes
        out_channels = out_planes
        if shared_conv is None:
            if groups == 1:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad', padding=padding)
            else:
                out_channels = in_planes
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad',
                                 padding=padding, group=in_channels)
            layers = [conv, _bn(out_planes), nn.ReLU6()]
        else:
            layers = [shared_conv, _bn(out_planes), nn.ReLU6()]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class FlattenConcat(nn.Cell):
    """
    Concatenate predictions into a single tensor.

    Args:
        config (dict): The default config of SSD.

    Returns:
        Tensor, flatten predictions.
    """

    def __init__(self, config):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = config.num_ssd_boxes
        self.concat = ops.Concat(axis=1)
        self.transpose = ops.Transpose()

    def construct(self, inputs):
        output = ()
        batch_size = ops.shape(inputs[0])[0]
        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (ops.reshape(x, (batch_size, -1)),)
        res = self.concat(output)
        return ops.reshape(res, (batch_size, self.num_ssd_boxes, -1))


class WeightSharedMultiBox(nn.Cell):
    """
    Weight shared Multi-box conv layers. Each multi-box layer contains class conf scores and localization predictions.
    All box predictors shares the same conv weight in different features.

    Args:
        config (dict): The default config of SSD.
        loc_cls_shared_addition(bool): Whether the location predictor and classifier prediction share the
                                       same addition layer.
    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self, num_classes, out_channels, num_default, num_features, num_addition_layers, num_ssd_boxes,
                 loc_cls_shared_addition=False):
        super(WeightSharedMultiBox, self).__init__()
        self.loc_cls_shared_addition = loc_cls_shared_addition

        if not loc_cls_shared_addition:
            loc_convs = [
                _conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)
            ]
            cls_convs = [
                _conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)
            ]
            addition_loc_layer_list = []
            addition_cls_layer_list = []
            for _ in range(num_features):
                addition_loc_layer = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, loc_convs[x]) for x in range(num_addition_layers)
                ]
                addition_cls_layer = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, cls_convs[x]) for x in range(num_addition_layers)
                ]
                addition_loc_layer_list.append(nn.SequentialCell(addition_loc_layer))
                addition_cls_layer_list.append(nn.SequentialCell(addition_cls_layer))
            self.addition_layer_loc = nn.CellList(addition_loc_layer_list)
            self.addition_layer_cls = nn.CellList(addition_cls_layer_list)
        else:
            convs = [
                _conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)
            ]
            addition_layer_list = []
            for _ in range(num_features):
                addition_layers = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, convs[x]) for x in range(num_addition_layers)
                ]
                addition_layer_list.append(nn.SequentialCell(addition_layers))
            self.addition_layer = nn.SequentialCell(addition_layer_list)

        loc_layers = [_conv2d(out_channels, 4 * num_default,
                              kernel_size=3, stride=1, pad_mod='pad')]
        cls_layers = [_conv2d(out_channels, num_classes * num_default,
                              kernel_size=3, stride=1, pad_mod='pad')]

        self.loc_layers = nn.SequentialCell(loc_layers)
        self.cls_layers = nn.SequentialCell(cls_layers)
        self.flatten_concat = FlattenConcatMulitx(num_ssd_boxes)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        num_heads = len(inputs)
        for i in range(num_heads):
            if self.loc_cls_shared_addition:
                features = self.addition_layer[i](inputs[i])
                loc_outputs += (self.loc_layers(features),)
                cls_outputs += (self.cls_layers(features),)
            else:
                features = self.addition_layer_loc[i](inputs[i])
                loc_outputs += (self.loc_layers(features),)
                features = self.addition_layer_cls[i](inputs[i])
                cls_outputs += (self.cls_layers(features),)

        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


class FeatureSelector(nn.Cell):
    """
    Select specific layers from an entire feature list
    """

    def __init__(self, feature_idxes):
        super(FeatureSelector, self).__init__()
        self.feature_idxes = feature_idxes

    def construct(self, feature_list):
        selected = ()
        for i in self.feature_idxes:
            selected = selected + (feature_list[i],)
        return selected


class FlattenConcatMulitx(nn.Cell):
    """FlattenConcat module."""

    def __init__(self, num_ssd_boxes):
        super(FlattenConcatMulitx, self).__init__()
        self.num_ssd_boxes = num_ssd_boxes

    def construct(self, inputs):
        output = ()
        batch_size = ops.shape(inputs[0])[0]
        for x in inputs:
            x = ops.transpose(x, (0, 2, 3, 1))
            output += (ops.reshape(x, (batch_size, -1)),)

        res = ops.concat(output, axis=1)

        return ops.reshape(res, (batch_size, self.num_ssd_boxes, -1))


def class_loss(logits, label):
    """Calculate category losses."""
    label = ops.one_hot(label, ops.shape(logits)[-1], mindspore.Tensor(1.0, mindspore.float32),
                        mindspore.Tensor(0.0, mindspore.float32))
    weight = ops.ones_like(logits)
    pos_weight = ops.ones_like(logits)
    sigmiod_cross_entropy = ops.binary_cross_entropy_with_logits(logits, label, weight.astype(mindspore.float32),
                                                                 pos_weight.astype(mindspore.float32))
    sigmoid = ops.sigmoid(logits)
    label = label.astype(mindspore.float32)
    p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
    modulating_factor = ops.pow(1 - p_t, 2.0)
    alpha_weight_factor = label * 0.75 + (1 - label) * (1 - 0.75)
    focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
    return focal_loss


class MultiBox(nn.Cell):
    """
    Multibox conv layers. Each multibox layer contains class conf scores and localization predictions.
    """

    def __init__(self, num_classes, out_channels, num_default, num_ssd_boxes):
        super(MultiBox, self).__init__()

        loc_layers = []
        cls_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers += [_last_conv2d(out_channel, 4 * num_default[k],
                                        kernel_size=3, stride=1, pad_mod='same', pad=0)]
            cls_layers += [_last_conv2d(out_channel, num_classes * num_default[k],
                                        kernel_size=3, stride=1, pad_mod='same', pad=0)]

        self.multi_loc_layers = nn.CellList(loc_layers)
        self.multi_cls_layers = nn.CellList(cls_layers)
        self.flatten_concat = FlattenConcatMulitx(num_ssd_boxes)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)
        # return loc_outputs, cls_outputs
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


def conv_bn_relu(in_channel, out_channel, kernel_size, stride, depthwise, activation='relu6'):
    padding = (kernel_size - 1) // 2
    output = [nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode="pad",
                        group=1 if not depthwise else in_channel, padding=padding), nn.BatchNorm2d(out_channel),
              nn.get_activation(activation)]
    return nn.SequentialCell(output)


class FpnTopDown(nn.Cell):
    """
    Fpn to extract features
    """

    def __init__(self, in_channel_list, out_channels):
        super(FpnTopDown, self).__init__()
        self.lateral_convs_list_ = []
        self.fpn_convs_ = []
        for channel in in_channel_list:
            l_conv = nn.Conv2d(channel, out_channels, kernel_size=1, stride=1,
                               has_bias=True, padding=0, pad_mode='same')
            fpn_conv = conv_bn_relu(out_channels, out_channels, kernel_size=3, stride=1, depthwise=False)
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.layer.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.layer.CellList(self.fpn_convs_)
        self.num_layers = len(in_channel_list)

    def construct(self, inputs):
        image_features = ()
        for i, feature in enumerate(inputs):
            image_features = image_features + (self.lateral_convs_list[i](feature),)

        features = (image_features[-1],)
        for i in range(len(inputs) - 1):
            top = len(inputs) - i - 1
            down = top - 1
            size = ops.shape(inputs[down])
            top_down = ops.ResizeBilinearV2()(features[-1],(size[2], size[3]))
            top_down = top_down + image_features[down]
            features = features + (top_down,)

        extract_features = ()
        num_features = len(features)
        for i in range(num_features):
            extract_features = extract_features + (self.fpn_convs_list[i](features[num_features - i - 1]),)

        return extract_features


class BottomUp(nn.Cell):
    """
    Bottom Up feature extractor
    """

    def __init__(self, levels, channels, kernel_size, stride):
        super(BottomUp, self).__init__()
        self.levels = levels
        bottom_up_cells = [
            conv_bn_relu(channels, channels, kernel_size, stride, False) for x in range(self.levels)
        ]
        self.blocks = nn.CellList(bottom_up_cells)

    def construct(self, features):
        for block in self.blocks:
            features = features + (block(features[-1]),)
        return features
