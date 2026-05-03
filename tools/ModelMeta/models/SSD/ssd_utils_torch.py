import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalClassificationLoss(nn.Module):
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
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, label):
        label = F.one_hot(label, num_classes=logits.size(-1)).float()
        sigmoid = torch.sigmoid(logits)
        sigmiod_cross_entropy = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = (1 - p_t).pow(self.gamma)
        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss.mean()


def _make_divisible(v, divisor, min_value=None):
    """Ensures that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v = new_v + divisor
    return new_v


def _conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=0):
    padding = (kernel_size - 1) // 2 if padding == 0 else padding
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=True)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.1)


def _last_conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=0):
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    if isinstance(padding, tuple):
        padding = padding[0]
    if isinstance(stride, tuple):
        stride = stride[0]
    depthwise_conv = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding=padding, groups=in_channel, bias=False)
    conv = _conv2d(in_channel, out_channel, kernel_size=1)
    return nn.Sequential(depthwise_conv, _bn(in_channel), nn.ReLU6(inplace=False), conv)


# class ConvBNReLU(nn.Module):
#     """
#     Convolution/Depthwise fused with Batchnorm and ReLU block definition.
#
#     Args:
#         in_planes (int): Input channel.
#         out_planes (int): Output channel.
#         kernel_size (int): Input kernel size.
#         stride (int): Stride size for the first convolutional layer. Default: 1.
#         groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.
#
#     Returns:
#         Tensor, output tensor.
#
#     Examples:
#         >>> ConvBNReLU(16, 256, kernel_size=1, stride=1, groups=1)
#     """
#
#     def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
#         super(ConvBNReLU, self).__init__()
#         padding = kernel_size // 2  # 'same' padding
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding, groups=groups)
#         self.bn = _bn(out_planes)
#         self.relu = nn.ReLU6(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return self.relu(x)

class ConvBNReLU(nn.Module):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, shared_conv=None):
        super(ConvBNReLU, self).__init__()
        padding = kernel_size // 2  # 'same' padding
        in_channels = in_planes
        out_channels = out_planes
        if shared_conv is None:
            if groups == 1:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, groups=groups,
                                 bias=False)
            else:
                out_channels = in_planes
                conv = nn.Conv2d(in_planes, out_channels, kernel_size, stride, padding=padding, groups=in_channels,
                                 bias=False)
            layers = [conv, _bn(out_planes), nn.ReLU6()]
        else:
            layers = [shared_conv, _bn(out_planes), nn.ReLU6()]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return output


class FlattenConcat(nn.Module):
    """
    Concatenate predictions into a single tensor.

    Args:
        num_ssd_boxes (int): The number of SSD boxes.

    Returns:
        Tensor, flatten predictions.
    """

    def __init__(self, num_ssd_boxes):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = num_ssd_boxes

    def forward(self, inputs):
        output = ()
        batch_size = inputs[0].size(0)
        for x in inputs:
            x = x.permute(0, 2, 3, 1)
            output = output + (x.reshape(batch_size, -1),)

        res = torch.cat(output, dim=1)
        return res.reshape(batch_size, self.num_ssd_boxes, -1)


class WeightSharedMultiBox(nn.Module):
    """
    Weight shared Multi-box conv layers. Each multi-box layer contains class conf scores and localization predictions.
    All box predictors shares the same conv weight in different features.

    Args:
        num_classes: Number of classes
        out_channels: Number of output channels
        num_default: Number of default boxes
        num_features: Number of feature maps
        num_addition_layers: Number of additional layers
        num_ssd_boxes: Number of SSD boxes
        loc_cls_shared_addition: Boolean indicating whether the location predictor and classifier prediction share the
                                 same additional layer.

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
                addition_loc_layer_list.append(nn.Sequential(*addition_loc_layer))
                addition_cls_layer_list.append(nn.Sequential(*addition_cls_layer))
            self.addition_layer_loc = nn.ModuleList(addition_loc_layer_list)
            self.addition_layer_cls = nn.ModuleList(addition_cls_layer_list)
        else:
            convs = [
                _conv2d(out_channels, out_channels, 3, 1) for _ in range(num_addition_layers)
            ]
            addition_layer_list = []
            for _ in range(num_features):
                addition_layers = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, convs[x]) for x in range(num_addition_layers)
                ]
                addition_layer_list.append(nn.Sequential(*addition_layers))
            self.addition_layer = nn.ModuleList(addition_layer_list)

        self.loc_layers = nn.Sequential(_conv2d(out_channels, 4 * num_default, 3, 1))
        self.cls_layers = nn.Sequential(_conv2d(out_channels, num_classes * num_default, 3, 1))
        self.flatten_concat = FlattenConcatMulitx(num_ssd_boxes)

    def forward(self, inputs):
        loc_outputs = []
        cls_outputs = []
        num_heads = len(inputs)
        for i in range(num_heads):
            if self.loc_cls_shared_addition:
                features = self.addition_layer[i](inputs[i])
                loc_outputs.append(self.loc_layers(features))
                cls_outputs.append(self.cls_layers(features))
            else:
                features = self.addition_layer_loc[i](inputs[i])
                loc_outputs.append(self.loc_layers(features))
                features = self.addition_layer_cls[i](inputs[i])
                cls_outputs.append(self.cls_layers(features))

        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


class FeatureSelector(nn.Module):
    """
    Select specific layers from an entire feature list
    """

    def __init__(self, feature_idxes):
        super(FeatureSelector, self).__init__()
        self.feature_idxes = feature_idxes

    def forward(self, feature_list):
        selected = []
        for i in self.feature_idxes:
            selected.append(feature_list[i])
        return tuple(selected)


class FlattenConcatMulitx(nn.Module):
    """FlattenConcat module."""

    def __init__(self, num_ssd_boxes):
        super(FlattenConcatMulitx, self).__init__()
        self.num_ssd_boxes = num_ssd_boxes

    def forward(self, inputs):
        batch_size = inputs[0].shape[0]
        output = [x.permute(0, 2, 3, 1).reshape(batch_size, -1) for x in inputs]
        res = torch.cat(output, dim=1)
        return res.view(batch_size, self.num_ssd_boxes, -1)


import torch
import torch.nn as nn
import torch.nn.functional as F


def class_loss(logits, label):
    """Calculate category losses."""
    label = F.one_hot(label, num_classes=logits.shape[-1])
    weight = torch.ones_like(logits)
    pos_weight = torch.ones_like(logits)
    sigmiod_cross_entropy = F.binary_cross_entropy_with_logits(logits, label.float(), weight=weight.float(),
                                                               pos_weight=pos_weight.float())
    sigmoid = torch.sigmoid(logits)
    p_t = label.float() * sigmoid + (1 - label.float()) * (1 - sigmoid)
    modulating_factor = torch.pow(1.0 - p_t, 2.0)
    alpha_weight_factor = label.float() * 0.75 + (1 - label.float()) * (1 - 0.75)
    focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
    return focal_loss


class MultiBox(nn.Module):
    """
    Multibox conv layers. Each multibox layer contains class conf scores and localization predictions.
    """

    def __init__(self, num_classes, out_channels, num_default, num_ssd_boxes):
        super(MultiBox, self).__init__()

        loc_layers = []
        cls_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers = loc_layers + [_last_conv2d(out_channel, 4 * num_default[k], kernel_size=3, stride=1, padding=1)]
            cls_layers = cls_layers + [_last_conv2d(out_channel, num_classes * num_default[k], kernel_size=3, stride=1, padding=1)]

        self.multi_loc_layers = nn.ModuleList(loc_layers)
        self.multi_cls_layers = nn.ModuleList(cls_layers)
        self.flatten_concat = FlattenConcatMulitx(num_ssd_boxes)

    def forward(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs = loc_outputs + (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs = cls_outputs + (self.multi_cls_layers[i](inputs[i]),)

        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, depthwise, activation='relu'):
    output = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2,
                        groups=1 if not depthwise else in_channels), nn.BatchNorm2d(out_channels)]
    if activation:
        if activation == 'relu':
            output.append(nn.ReLU())
        elif activation == 'relu6':
            output.append(nn.ReLU6())
        # add more activations if needed
    return nn.Sequential(*output)


class FpnTopDown(nn.Module):
    """
    Fpn to extract features
    """

    def __init__(self, in_channel_list, out_channels):
        super(FpnTopDown, self).__init__()
        self.lateral_convs_list = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) for in_channels in
             in_channel_list])
        self.fpn_convs_list = nn.ModuleList(
            [conv_bn_relu(out_channels, out_channels, kernel_size=3, stride=1, depthwise=False) for _ in
             in_channel_list])

    def forward(self, inputs):
        lateral_outs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs_list)]
        features = [lateral_outs[-1]]
        for i in range(len(inputs) - 1, 0, -1):
            top_down = F.interpolate(features[-1], size=lateral_outs[i - 1].shape[2:], mode='bilinear',
                                     align_corners=False)
            top_down = top_down + lateral_outs[i - 1]
            features.append(top_down)
        extract_features = [self.fpn_convs_list[i](features[-(i + 1)]) for i in range(len(features))]
        return extract_features


class BottomUp(nn.Module):
    """
    Bottom Up feature extractor
    """

    def __init__(self, levels, channels, kernel_size, stride):
        super(BottomUp, self).__init__()
        self.blocks = nn.ModuleList(
            [conv_bn_relu(channels, channels, kernel_size, stride, False) for _ in range(levels)])

    def forward(self, features):
        for block in self.blocks:
            features.append(block(features[-1]))
        return features
