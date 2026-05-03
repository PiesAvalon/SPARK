import mindspore
import numpy as np
from mindspore import nn, ops
import mindspore as ms
from functools import partial


def _make_divisible(x, divisor=8):
    return int(np.ceil(x * 1. / divisor) * divisor)


class Activation(nn.Cell):
    """
    Activation definition.

    Args:
        act_func(string): activation name.

    Returns:
         Tensor, output tensor.
    """

    def __init__(self, act_func):
        super(Activation, self).__init__()
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

    def construct(self, x):
        return self.act(x)


class GlobalAvgPooling(nn.Cell):
    """
    Global avg pooling definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
    """

    def __init__(self, keep_dims=False):
        super(GlobalAvgPooling, self).__init__()
        self.mean = ops.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class Unit(nn.Cell):
    """
    Unit warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        padding (int): Padding number.
        num_groups (int): Output num group.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    Examples:
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, num_groups=1,
                 use_act=True, act_type='relu'):
        super(Unit, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_in,
                              out_channels=num_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              group=num_groups,
                              has_bias=False,
                              pad_mode='pad')
        self.bn = nn.BatchNorm2d(num_out)
        self.use_act = use_act
        self.act = Activation(act_type) if use_act else None

    def construct(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class SE(nn.Cell):
    """
    SE warpper definition.

    Args:
        num_out (int): Numbers of output channels.
        ratio (int): middle output ratio.

    Returns:
        Tensor, output tensor.


    """

    def __init__(self, num_out, ratio=4):
        super(SE, self).__init__()
        num_mid = _make_divisible(num_out // ratio)
        self.pool = GlobalAvgPooling(keep_dims=True)
        self.conv1 = nn.Conv2d(in_channels=num_out, out_channels=num_mid,
                               kernel_size=1, has_bias=True, pad_mode='pad')
        self.act1 = Activation('relu')
        self.conv2 = nn.Conv2d(in_channels=num_mid, out_channels=num_out,
                               kernel_size=1, has_bias=True, pad_mode='pad')
        self.act2 = Activation('hsigmoid')
        self.mul = ops.Mul()

    def construct(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.mul(x, out)
        return out


class ResUnit(nn.Cell):
    """
    ResUnit warpper definition.

    Args:
        num_in (int): Input channel.
        num_mid (int): Middle channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        act_type (str): Activation type.
        use_se (bool): Use SE warpper or not.

    Returns:
        Tensor, output tensor.

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
        self.add = ops.Add() if self.use_short_cut_conv else None

    def construct(self, x):
        """construct"""
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


class MobileNetV3(nn.Cell):
    def __init__(self, model_cfgs, num_classes=1000, multiplier=1., final_drop=0.,
                 round_nearest=8, include_top=True, activation="None"):
        super(MobileNetV3, self).__init__()
        self.cfgs = model_cfgs['cfg']
        self.inplanes = 16
        self.features = []
        first_conv_in_channel = 3
        first_conv_out_channel = _make_divisible(multiplier * self.inplanes)

        self.features.append(nn.Conv2d(in_channels=first_conv_in_channel,
                                       out_channels=first_conv_out_channel,
                                       kernel_size=3, padding=1, stride=2,
                                       has_bias=False, pad_mode='pad'))
        self.features.append(nn.BatchNorm2d(first_conv_out_channel))
        self.features.append(Activation('hswish'))
        for layer_cfg in self.cfgs:
            self.features.append(self._make_layer(kernel_size=layer_cfg[0],
                                                  exp_ch=_make_divisible(multiplier * layer_cfg[1]),
                                                  out_channel=_make_divisible(multiplier * layer_cfg[2]),
                                                  use_se=layer_cfg[3],
                                                  act_func=layer_cfg[4],
                                                  stride=layer_cfg[5]))
        output_channel = _make_divisible(multiplier * model_cfgs["cls_ch_squeeze"])
        self.features.append(nn.Conv2d(in_channels=_make_divisible(multiplier * self.cfgs[-1][2]),
                                       out_channels=output_channel,
                                       kernel_size=1, padding=0, stride=1,
                                       has_bias=False, pad_mode='pad'))
        self.features.append(nn.BatchNorm2d(output_channel))
        self.features.append(Activation('hswish'))
        self.features.append(GlobalAvgPooling(keep_dims=True))
        self.features.append(nn.Conv2d(in_channels=output_channel,
                                       out_channels=model_cfgs['cls_ch_expand'],
                                       kernel_size=1, padding=0, stride=1,
                                       has_bias=False, pad_mode='pad'))
        self.features.append(Activation('hswish'))
        if final_drop > 0:
            self.features.append((nn.Dropout(final_drop)))

        # make it nn.CellList
        self.features = nn.SequentialCell(self.features)
        self.include_top = include_top
        self.need_activation = False
        if self.include_top:
            self.output = nn.Conv2d(in_channels=model_cfgs['cls_ch_expand'],
                                    out_channels=num_classes,
                                    kernel_size=1, has_bias=True, pad_mode='pad')
            self.squeeze = ops.Squeeze(axis=(2, 3))
            if activation != "None":
                self.need_activation = True
                if activation == "Sigmoid":
                    self.activation = ops.Sigmoid()
                elif activation == "Softmax":
                    self.activation = ops.Softmax()
                else:
                    raise NotImplementedError(f"The activation {activation} not in [Sigmoid, Softmax].")

        self._initialize_weights()

        self.layer_names = {
            "features": self.features,
            "features.0": self.features[0],
            "features.1": self.features[1],
            "features.2": self.features[2],
            "features.2.act": self.features[2].act,
            "features.3": self.features[3],
            "features.3.conv1": self.features[3].conv1,
            "features.3.conv1.conv": self.features[3].conv1.conv,
            "features.3.conv1.bn": self.features[3].conv1.bn,
            "features.3.conv1.act": self.features[3].conv1.act,
            "features.3.conv1.act.act": self.features[3].conv1.act.act,
            "features.3.conv2": self.features[3].conv2,
            "features.3.conv2.conv": self.features[3].conv2.conv,
            "features.3.conv2.bn": self.features[3].conv2.bn,
            "features.4": self.features[4],
            "features.4.expand": self.features[4].expand,
            "features.4.expand.conv": self.features[4].expand.conv,
            "features.4.expand.bn": self.features[4].expand.bn,
            "features.4.expand.act": self.features[4].expand.act,
            "features.4.expand.act.act": self.features[4].expand.act.act,
            "features.4.conv1": self.features[4].conv1,
            "features.4.conv1.conv": self.features[4].conv1.conv,
            "features.4.conv1.bn": self.features[4].conv1.bn,
            "features.4.conv1.act": self.features[4].conv1.act,
            "features.4.conv1.act.act": self.features[4].conv1.act.act,
            "features.4.conv2": self.features[4].conv2,
            "features.4.conv2.conv": self.features[4].conv2.conv,
            "features.4.conv2.bn": self.features[4].conv2.bn,
            "features.5": self.features[5],
            "features.5.expand": self.features[5].expand,
            "features.5.expand.conv": self.features[5].expand.conv,
            "features.5.expand.bn": self.features[5].expand.bn,
            "features.5.expand.act": self.features[5].expand.act,
            "features.5.expand.act.act": self.features[5].expand.act.act,
            "features.5.conv1": self.features[5].conv1,
            "features.5.conv1.conv": self.features[5].conv1.conv,
            "features.5.conv1.bn": self.features[5].conv1.bn,
            "features.5.conv1.act": self.features[5].conv1.act,
            "features.5.conv1.act.act": self.features[5].conv1.act.act,
            "features.5.conv2": self.features[5].conv2,
            "features.5.conv2.conv": self.features[5].conv2.conv,
            "features.5.conv2.bn": self.features[5].conv2.bn,
            "features.6": self.features[6],
            "features.6.expand": self.features[6].expand,
            "features.6.expand.conv": self.features[6].expand.conv,
            "features.6.expand.bn": self.features[6].expand.bn,
            "features.6.expand.act": self.features[6].expand.act,
            "features.6.expand.act.act": self.features[6].expand.act.act,
            "features.6.conv1": self.features[6].conv1,
            "features.6.conv1.conv": self.features[6].conv1.conv,
            "features.6.conv1.bn": self.features[6].conv1.bn,
            "features.6.conv1.act": self.features[6].conv1.act,
            "features.6.conv1.act.act": self.features[6].conv1.act.act,
            "features.6.se": self.features[6].se,
            "features.6.se.pool": self.features[6].se.pool,
            "features.6.se.conv1": self.features[6].se.conv1,
            "features.6.se.act1": self.features[6].se.act1,
            "features.6.se.act1.act": self.features[6].se.act1.act,
            "features.6.se.conv2": self.features[6].se.conv2,
            "features.6.se.act2": self.features[6].se.act2,
            "features.6.se.act2.act": self.features[6].se.act2.act,
            "features.6.conv2": self.features[6].conv2,
            "features.6.conv2.conv": self.features[6].conv2.conv,
            "features.6.conv2.bn": self.features[6].conv2.bn,
            "features.7": self.features[7],
            "features.7.expand": self.features[7].expand,
            "features.7.expand.conv": self.features[7].expand.conv,
            "features.7.expand.bn": self.features[7].expand.bn,
            "features.7.expand.act": self.features[7].expand.act,
            "features.7.expand.act.act": self.features[7].expand.act.act,
            "features.7.conv1": self.features[7].conv1,
            "features.7.conv1.conv": self.features[7].conv1.conv,
            "features.7.conv1.bn": self.features[7].conv1.bn,
            "features.7.conv1.act": self.features[7].conv1.act,
            "features.7.conv1.act.act": self.features[7].conv1.act.act,
            "features.7.se": self.features[7].se,
            "features.7.se.pool": self.features[7].se.pool,
            "features.7.se.conv1": self.features[7].se.conv1,
            "features.7.se.act1": self.features[7].se.act1,
            "features.7.se.act1.act": self.features[7].se.act1.act,
            "features.7.se.conv2": self.features[7].se.conv2,
            "features.7.se.act2": self.features[7].se.act2,
            "features.7.se.act2.act": self.features[7].se.act2.act,
            "features.7.conv2": self.features[7].conv2,
            "features.7.conv2.conv": self.features[7].conv2.conv,
            "features.7.conv2.bn": self.features[7].conv2.bn,
            "features.8": self.features[8],
            "features.8.expand": self.features[8].expand,
            "features.8.expand.conv": self.features[8].expand.conv,
            "features.8.expand.bn": self.features[8].expand.bn,
            "features.8.expand.act": self.features[8].expand.act,
            "features.8.expand.act.act": self.features[8].expand.act.act,
            "features.8.conv1": self.features[8].conv1,
            "features.8.conv1.conv": self.features[8].conv1.conv,
            "features.8.conv1.bn": self.features[8].conv1.bn,
            "features.8.conv1.act": self.features[8].conv1.act,
            "features.8.conv1.act.act": self.features[8].conv1.act.act,
            "features.8.se": self.features[8].se,
            "features.8.se.pool": self.features[8].se.pool,
            "features.8.se.conv1": self.features[8].se.conv1,
            "features.8.se.act1": self.features[8].se.act1,
            "features.8.se.act1.act": self.features[8].se.act1.act,
            "features.8.se.conv2": self.features[8].se.conv2,
            "features.8.se.act2": self.features[8].se.act2,
            "features.8.se.act2.act": self.features[8].se.act2.act,
            "features.8.conv2": self.features[8].conv2,
            "features.8.conv2.conv": self.features[8].conv2.conv,
            "features.8.conv2.bn": self.features[8].conv2.bn,
            "features.9": self.features[9],
            "features.9.expand": self.features[9].expand,
            "features.9.expand.conv": self.features[9].expand.conv,
            "features.9.expand.bn": self.features[9].expand.bn,
            "features.9.expand.act": self.features[9].expand.act,
            "features.9.expand.act.act": self.features[9].expand.act.act,
            "features.9.conv1": self.features[9].conv1,
            "features.9.conv1.conv": self.features[9].conv1.conv,
            "features.9.conv1.bn": self.features[9].conv1.bn,
            "features.9.conv1.act": self.features[9].conv1.act,
            "features.9.conv1.act.act": self.features[9].conv1.act.act,
            "features.9.conv2": self.features[9].conv2,
            "features.9.conv2.conv": self.features[9].conv2.conv,
            "features.9.conv2.bn": self.features[9].conv2.bn,
            "features.10": self.features[10],
            "features.10.expand": self.features[10].expand,
            "features.10.expand.conv": self.features[10].expand.conv,
            "features.10.expand.bn": self.features[10].expand.bn,
            "features.10.expand.act": self.features[10].expand.act,
            "features.10.expand.act.act": self.features[10].expand.act.act,
            "features.10.conv1": self.features[10].conv1,
            "features.10.conv1.conv": self.features[10].conv1.conv,
            "features.10.conv1.bn": self.features[10].conv1.bn,
            "features.10.conv1.act": self.features[10].conv1.act,
            "features.10.conv1.act.act": self.features[10].conv1.act.act,
            "features.10.conv2": self.features[10].conv2,
            "features.10.conv2.conv": self.features[10].conv2.conv,
            "features.10.conv2.bn": self.features[10].conv2.bn,
            "features.11": self.features[11],
            "features.11.expand": self.features[11].expand,
            "features.11.expand.conv": self.features[11].expand.conv,
            "features.11.expand.bn": self.features[11].expand.bn,
            "features.11.expand.act": self.features[11].expand.act,
            "features.11.expand.act.act": self.features[11].expand.act.act,
            "features.11.conv1": self.features[11].conv1,
            "features.11.conv1.conv": self.features[11].conv1.conv,
            "features.11.conv1.bn": self.features[11].conv1.bn,
            "features.11.conv1.act": self.features[11].conv1.act,
            "features.11.conv1.act.act": self.features[11].conv1.act.act,
            "features.11.conv2": self.features[11].conv2,
            "features.11.conv2.conv": self.features[11].conv2.conv,
            "features.11.conv2.bn": self.features[11].conv2.bn,
            "features.12": self.features[12],
            "features.12.expand": self.features[12].expand,
            "features.12.expand.conv": self.features[12].expand.conv,
            "features.12.expand.bn": self.features[12].expand.bn,
            "features.12.expand.act": self.features[12].expand.act,
            "features.12.expand.act.act": self.features[12].expand.act.act,
            "features.12.conv1": self.features[12].conv1,
            "features.12.conv1.conv": self.features[12].conv1.conv,
            "features.12.conv1.bn": self.features[12].conv1.bn,
            "features.12.conv1.act": self.features[12].conv1.act,
            "features.12.conv1.act.act": self.features[12].conv1.act.act,
            "features.12.conv2": self.features[12].conv2,
            "features.12.conv2.conv": self.features[12].conv2.conv,
            "features.12.conv2.bn": self.features[12].conv2.bn,
            "features.13": self.features[13],
            "features.13.expand": self.features[13].expand,
            "features.13.expand.conv": self.features[13].expand.conv,
            "features.13.expand.bn": self.features[13].expand.bn,
            "features.13.expand.act": self.features[13].expand.act,
            "features.13.expand.act.act": self.features[13].expand.act.act,
            "features.13.conv1": self.features[13].conv1,
            "features.13.conv1.conv": self.features[13].conv1.conv,
            "features.13.conv1.bn": self.features[13].conv1.bn,
            "features.13.conv1.act": self.features[13].conv1.act,
            "features.13.conv1.act.act": self.features[13].conv1.act.act,
            "features.13.se": self.features[13].se,
            "features.13.se.pool": self.features[13].se.pool,
            "features.13.se.conv1": self.features[13].se.conv1,
            "features.13.se.act1": self.features[13].se.act1,
            "features.13.se.act1.act": self.features[13].se.act1.act,
            "features.13.se.conv2": self.features[13].se.conv2,
            "features.13.se.act2": self.features[13].se.act2,
            "features.13.se.act2.act": self.features[13].se.act2.act,
            "features.13.conv2": self.features[13].conv2,
            "features.13.conv2.conv": self.features[13].conv2.conv,
            "features.13.conv2.bn": self.features[13].conv2.bn,
            "features.14": self.features[14],
            "features.14.expand": self.features[14].expand,
            "features.14.expand.conv": self.features[14].expand.conv,
            "features.14.expand.bn": self.features[14].expand.bn,
            "features.14.expand.act": self.features[14].expand.act,
            "features.14.expand.act.act": self.features[14].expand.act.act,
            "features.14.conv1": self.features[14].conv1,
            "features.14.conv1.conv": self.features[14].conv1.conv,
            "features.14.conv1.bn": self.features[14].conv1.bn,
            "features.14.conv1.act": self.features[14].conv1.act,
            "features.14.conv1.act.act": self.features[14].conv1.act.act,
            "features.14.se": self.features[14].se,
            "features.14.se.pool": self.features[14].se.pool,
            "features.14.se.conv1": self.features[14].se.conv1,
            "features.14.se.act1": self.features[14].se.act1,
            "features.14.se.act1.act": self.features[14].se.act1.act,
            "features.14.se.conv2": self.features[14].se.conv2,
            "features.14.se.act2": self.features[14].se.act2,
            "features.14.se.act2.act": self.features[14].se.act2.act,
            "features.14.conv2": self.features[14].conv2,
            "features.14.conv2.conv": self.features[14].conv2.conv,
            "features.14.conv2.bn": self.features[14].conv2.bn,
            "features.15": self.features[15],
            "features.15.expand": self.features[15].expand,
            "features.15.expand.conv": self.features[15].expand.conv,
            "features.15.expand.bn": self.features[15].expand.bn,
            "features.15.expand.act": self.features[15].expand.act,
            "features.15.expand.act.act": self.features[15].expand.act.act,
            "features.15.conv1": self.features[15].conv1,
            "features.15.conv1.conv": self.features[15].conv1.conv,
            "features.15.conv1.bn": self.features[15].conv1.bn,
            "features.15.conv1.act": self.features[15].conv1.act,
            "features.15.conv1.act.act": self.features[15].conv1.act.act,
            "features.15.se": self.features[15].se,
            "features.15.se.pool": self.features[15].se.pool,
            "features.15.se.conv1": self.features[15].se.conv1,
            "features.15.se.act1": self.features[15].se.act1,
            "features.15.se.act1.act": self.features[15].se.act1.act,
            "features.15.se.conv2": self.features[15].se.conv2,
            "features.15.se.act2": self.features[15].se.act2,
            "features.15.se.act2.act": self.features[15].se.act2.act,
            "features.15.conv2": self.features[15].conv2,
            "features.15.conv2.conv": self.features[15].conv2.conv,
            "features.15.conv2.bn": self.features[15].conv2.bn,
            "features.16": self.features[16],
            "features.16.expand": self.features[16].expand,
            "features.16.expand.conv": self.features[16].expand.conv,
            "features.16.expand.bn": self.features[16].expand.bn,
            "features.16.expand.act": self.features[16].expand.act,
            "features.16.expand.act.act": self.features[16].expand.act.act,
            "features.16.conv1": self.features[16].conv1,
            "features.16.conv1.conv": self.features[16].conv1.conv,
            "features.16.conv1.bn": self.features[16].conv1.bn,
            "features.16.conv1.act": self.features[16].conv1.act,
            "features.16.conv1.act.act": self.features[16].conv1.act.act,
            "features.16.se": self.features[16].se,
            "features.16.se.pool": self.features[16].se.pool,
            "features.16.se.conv1": self.features[16].se.conv1,
            "features.16.se.act1": self.features[16].se.act1,
            "features.16.se.act1.act": self.features[16].se.act1.act,
            "features.16.se.conv2": self.features[16].se.conv2,
            "features.16.se.act2": self.features[16].se.act2,
            "features.16.se.act2.act": self.features[16].se.act2.act,
            "features.16.conv2": self.features[16].conv2,
            "features.16.conv2.conv": self.features[16].conv2.conv,
            "features.16.conv2.bn": self.features[16].conv2.bn,
            "features.17": self.features[17],
            "features.17.expand": self.features[17].expand,
            "features.17.expand.conv": self.features[17].expand.conv,
            "features.17.expand.bn": self.features[17].expand.bn,
            "features.17.expand.act": self.features[17].expand.act,
            "features.17.expand.act.act": self.features[17].expand.act.act,
            "features.17.conv1": self.features[17].conv1,
            "features.17.conv1.conv": self.features[17].conv1.conv,
            "features.17.conv1.bn": self.features[17].conv1.bn,
            "features.17.conv1.act": self.features[17].conv1.act,
            "features.17.conv1.act.act": self.features[17].conv1.act.act,
            "features.17.se": self.features[17].se,
            "features.17.se.pool": self.features[17].se.pool,
            "features.17.se.conv1": self.features[17].se.conv1,
            "features.17.se.act1": self.features[17].se.act1,
            "features.17.se.act1.act": self.features[17].se.act1.act,
            "features.17.se.conv2": self.features[17].se.conv2,
            "features.17.se.act2": self.features[17].se.act2,
            "features.17.se.act2.act": self.features[17].se.act2.act,
            "features.17.conv2": self.features[17].conv2,
            "features.17.conv2.conv": self.features[17].conv2.conv,
            "features.17.conv2.bn": self.features[17].conv2.bn,
            "features.18": self.features[18],
            "features.19": self.features[19],
            "features.20": self.features[20],
            "features.20.act": self.features[20].act,
            "features.21": self.features[21],
            "features.22": self.features[22],
            "features.23": self.features[23],
            "features.23.act": self.features[23].act,
            "output": self.output,

        }
        self.original_names = {
            "features": self.features,
            "features.0": self.features[0],
            "features.1": self.features[1],
            "features.2": self.features[2],
            "features.2.act": self.features[2].act,
            "features.3": self.features[3],
            "features.3.conv1": self.features[3].conv1,
            "features.3.conv1.conv": self.features[3].conv1.conv,
            "features.3.conv1.bn": self.features[3].conv1.bn,
            "features.3.conv1.act": self.features[3].conv1.act,
            "features.3.conv1.act.act": self.features[3].conv1.act.act,
            "features.3.conv2": self.features[3].conv2,
            "features.3.conv2.conv": self.features[3].conv2.conv,
            "features.3.conv2.bn": self.features[3].conv2.bn,
            "features.4": self.features[4],
            "features.4.expand": self.features[4].expand,
            "features.4.expand.conv": self.features[4].expand.conv,
            "features.4.expand.bn": self.features[4].expand.bn,
            "features.4.expand.act": self.features[4].expand.act,
            "features.4.expand.act.act": self.features[4].expand.act.act,
            "features.4.conv1": self.features[4].conv1,
            "features.4.conv1.conv": self.features[4].conv1.conv,
            "features.4.conv1.bn": self.features[4].conv1.bn,
            "features.4.conv1.act": self.features[4].conv1.act,
            "features.4.conv1.act.act": self.features[4].conv1.act.act,
            "features.4.conv2": self.features[4].conv2,
            "features.4.conv2.conv": self.features[4].conv2.conv,
            "features.4.conv2.bn": self.features[4].conv2.bn,
            "features.5": self.features[5],
            "features.5.expand": self.features[5].expand,
            "features.5.expand.conv": self.features[5].expand.conv,
            "features.5.expand.bn": self.features[5].expand.bn,
            "features.5.expand.act": self.features[5].expand.act,
            "features.5.expand.act.act": self.features[5].expand.act.act,
            "features.5.conv1": self.features[5].conv1,
            "features.5.conv1.conv": self.features[5].conv1.conv,
            "features.5.conv1.bn": self.features[5].conv1.bn,
            "features.5.conv1.act": self.features[5].conv1.act,
            "features.5.conv1.act.act": self.features[5].conv1.act.act,
            "features.5.conv2": self.features[5].conv2,
            "features.5.conv2.conv": self.features[5].conv2.conv,
            "features.5.conv2.bn": self.features[5].conv2.bn,
            "features.6": self.features[6],
            "features.6.expand": self.features[6].expand,
            "features.6.expand.conv": self.features[6].expand.conv,
            "features.6.expand.bn": self.features[6].expand.bn,
            "features.6.expand.act": self.features[6].expand.act,
            "features.6.expand.act.act": self.features[6].expand.act.act,
            "features.6.conv1": self.features[6].conv1,
            "features.6.conv1.conv": self.features[6].conv1.conv,
            "features.6.conv1.bn": self.features[6].conv1.bn,
            "features.6.conv1.act": self.features[6].conv1.act,
            "features.6.conv1.act.act": self.features[6].conv1.act.act,
            "features.6.se": self.features[6].se,
            "features.6.se.pool": self.features[6].se.pool,
            "features.6.se.conv1": self.features[6].se.conv1,
            "features.6.se.act1": self.features[6].se.act1,
            "features.6.se.act1.act": self.features[6].se.act1.act,
            "features.6.se.conv2": self.features[6].se.conv2,
            "features.6.se.act2": self.features[6].se.act2,
            "features.6.se.act2.act": self.features[6].se.act2.act,
            "features.6.conv2": self.features[6].conv2,
            "features.6.conv2.conv": self.features[6].conv2.conv,
            "features.6.conv2.bn": self.features[6].conv2.bn,
            "features.7": self.features[7],
            "features.7.expand": self.features[7].expand,
            "features.7.expand.conv": self.features[7].expand.conv,
            "features.7.expand.bn": self.features[7].expand.bn,
            "features.7.expand.act": self.features[7].expand.act,
            "features.7.expand.act.act": self.features[7].expand.act.act,
            "features.7.conv1": self.features[7].conv1,
            "features.7.conv1.conv": self.features[7].conv1.conv,
            "features.7.conv1.bn": self.features[7].conv1.bn,
            "features.7.conv1.act": self.features[7].conv1.act,
            "features.7.conv1.act.act": self.features[7].conv1.act.act,
            "features.7.se": self.features[7].se,
            "features.7.se.pool": self.features[7].se.pool,
            "features.7.se.conv1": self.features[7].se.conv1,
            "features.7.se.act1": self.features[7].se.act1,
            "features.7.se.act1.act": self.features[7].se.act1.act,
            "features.7.se.conv2": self.features[7].se.conv2,
            "features.7.se.act2": self.features[7].se.act2,
            "features.7.se.act2.act": self.features[7].se.act2.act,
            "features.7.conv2": self.features[7].conv2,
            "features.7.conv2.conv": self.features[7].conv2.conv,
            "features.7.conv2.bn": self.features[7].conv2.bn,
            "features.8": self.features[8],
            "features.8.expand": self.features[8].expand,
            "features.8.expand.conv": self.features[8].expand.conv,
            "features.8.expand.bn": self.features[8].expand.bn,
            "features.8.expand.act": self.features[8].expand.act,
            "features.8.expand.act.act": self.features[8].expand.act.act,
            "features.8.conv1": self.features[8].conv1,
            "features.8.conv1.conv": self.features[8].conv1.conv,
            "features.8.conv1.bn": self.features[8].conv1.bn,
            "features.8.conv1.act": self.features[8].conv1.act,
            "features.8.conv1.act.act": self.features[8].conv1.act.act,
            "features.8.se": self.features[8].se,
            "features.8.se.pool": self.features[8].se.pool,
            "features.8.se.conv1": self.features[8].se.conv1,
            "features.8.se.act1": self.features[8].se.act1,
            "features.8.se.act1.act": self.features[8].se.act1.act,
            "features.8.se.conv2": self.features[8].se.conv2,
            "features.8.se.act2": self.features[8].se.act2,
            "features.8.se.act2.act": self.features[8].se.act2.act,
            "features.8.conv2": self.features[8].conv2,
            "features.8.conv2.conv": self.features[8].conv2.conv,
            "features.8.conv2.bn": self.features[8].conv2.bn,
            "features.9": self.features[9],
            "features.9.expand": self.features[9].expand,
            "features.9.expand.conv": self.features[9].expand.conv,
            "features.9.expand.bn": self.features[9].expand.bn,
            "features.9.expand.act": self.features[9].expand.act,
            "features.9.expand.act.act": self.features[9].expand.act.act,
            "features.9.conv1": self.features[9].conv1,
            "features.9.conv1.conv": self.features[9].conv1.conv,
            "features.9.conv1.bn": self.features[9].conv1.bn,
            "features.9.conv1.act": self.features[9].conv1.act,
            "features.9.conv1.act.act": self.features[9].conv1.act.act,
            "features.9.conv2": self.features[9].conv2,
            "features.9.conv2.conv": self.features[9].conv2.conv,
            "features.9.conv2.bn": self.features[9].conv2.bn,
            "features.10": self.features[10],
            "features.10.expand": self.features[10].expand,
            "features.10.expand.conv": self.features[10].expand.conv,
            "features.10.expand.bn": self.features[10].expand.bn,
            "features.10.expand.act": self.features[10].expand.act,
            "features.10.expand.act.act": self.features[10].expand.act.act,
            "features.10.conv1": self.features[10].conv1,
            "features.10.conv1.conv": self.features[10].conv1.conv,
            "features.10.conv1.bn": self.features[10].conv1.bn,
            "features.10.conv1.act": self.features[10].conv1.act,
            "features.10.conv1.act.act": self.features[10].conv1.act.act,
            "features.10.conv2": self.features[10].conv2,
            "features.10.conv2.conv": self.features[10].conv2.conv,
            "features.10.conv2.bn": self.features[10].conv2.bn,
            "features.11": self.features[11],
            "features.11.expand": self.features[11].expand,
            "features.11.expand.conv": self.features[11].expand.conv,
            "features.11.expand.bn": self.features[11].expand.bn,
            "features.11.expand.act": self.features[11].expand.act,
            "features.11.expand.act.act": self.features[11].expand.act.act,
            "features.11.conv1": self.features[11].conv1,
            "features.11.conv1.conv": self.features[11].conv1.conv,
            "features.11.conv1.bn": self.features[11].conv1.bn,
            "features.11.conv1.act": self.features[11].conv1.act,
            "features.11.conv1.act.act": self.features[11].conv1.act.act,
            "features.11.conv2": self.features[11].conv2,
            "features.11.conv2.conv": self.features[11].conv2.conv,
            "features.11.conv2.bn": self.features[11].conv2.bn,
            "features.12": self.features[12],
            "features.12.expand": self.features[12].expand,
            "features.12.expand.conv": self.features[12].expand.conv,
            "features.12.expand.bn": self.features[12].expand.bn,
            "features.12.expand.act": self.features[12].expand.act,
            "features.12.expand.act.act": self.features[12].expand.act.act,
            "features.12.conv1": self.features[12].conv1,
            "features.12.conv1.conv": self.features[12].conv1.conv,
            "features.12.conv1.bn": self.features[12].conv1.bn,
            "features.12.conv1.act": self.features[12].conv1.act,
            "features.12.conv1.act.act": self.features[12].conv1.act.act,
            "features.12.conv2": self.features[12].conv2,
            "features.12.conv2.conv": self.features[12].conv2.conv,
            "features.12.conv2.bn": self.features[12].conv2.bn,
            "features.13": self.features[13],
            "features.13.expand": self.features[13].expand,
            "features.13.expand.conv": self.features[13].expand.conv,
            "features.13.expand.bn": self.features[13].expand.bn,
            "features.13.expand.act": self.features[13].expand.act,
            "features.13.expand.act.act": self.features[13].expand.act.act,
            "features.13.conv1": self.features[13].conv1,
            "features.13.conv1.conv": self.features[13].conv1.conv,
            "features.13.conv1.bn": self.features[13].conv1.bn,
            "features.13.conv1.act": self.features[13].conv1.act,
            "features.13.conv1.act.act": self.features[13].conv1.act.act,
            "features.13.se": self.features[13].se,
            "features.13.se.pool": self.features[13].se.pool,
            "features.13.se.conv1": self.features[13].se.conv1,
            "features.13.se.act1": self.features[13].se.act1,
            "features.13.se.act1.act": self.features[13].se.act1.act,
            "features.13.se.conv2": self.features[13].se.conv2,
            "features.13.se.act2": self.features[13].se.act2,
            "features.13.se.act2.act": self.features[13].se.act2.act,
            "features.13.conv2": self.features[13].conv2,
            "features.13.conv2.conv": self.features[13].conv2.conv,
            "features.13.conv2.bn": self.features[13].conv2.bn,
            "features.14": self.features[14],
            "features.14.expand": self.features[14].expand,
            "features.14.expand.conv": self.features[14].expand.conv,
            "features.14.expand.bn": self.features[14].expand.bn,
            "features.14.expand.act": self.features[14].expand.act,
            "features.14.expand.act.act": self.features[14].expand.act.act,
            "features.14.conv1": self.features[14].conv1,
            "features.14.conv1.conv": self.features[14].conv1.conv,
            "features.14.conv1.bn": self.features[14].conv1.bn,
            "features.14.conv1.act": self.features[14].conv1.act,
            "features.14.conv1.act.act": self.features[14].conv1.act.act,
            "features.14.se": self.features[14].se,
            "features.14.se.pool": self.features[14].se.pool,
            "features.14.se.conv1": self.features[14].se.conv1,
            "features.14.se.act1": self.features[14].se.act1,
            "features.14.se.act1.act": self.features[14].se.act1.act,
            "features.14.se.conv2": self.features[14].se.conv2,
            "features.14.se.act2": self.features[14].se.act2,
            "features.14.se.act2.act": self.features[14].se.act2.act,
            "features.14.conv2": self.features[14].conv2,
            "features.14.conv2.conv": self.features[14].conv2.conv,
            "features.14.conv2.bn": self.features[14].conv2.bn,
            "features.15": self.features[15],
            "features.15.expand": self.features[15].expand,
            "features.15.expand.conv": self.features[15].expand.conv,
            "features.15.expand.bn": self.features[15].expand.bn,
            "features.15.expand.act": self.features[15].expand.act,
            "features.15.expand.act.act": self.features[15].expand.act.act,
            "features.15.conv1": self.features[15].conv1,
            "features.15.conv1.conv": self.features[15].conv1.conv,
            "features.15.conv1.bn": self.features[15].conv1.bn,
            "features.15.conv1.act": self.features[15].conv1.act,
            "features.15.conv1.act.act": self.features[15].conv1.act.act,
            "features.15.se": self.features[15].se,
            "features.15.se.pool": self.features[15].se.pool,
            "features.15.se.conv1": self.features[15].se.conv1,
            "features.15.se.act1": self.features[15].se.act1,
            "features.15.se.act1.act": self.features[15].se.act1.act,
            "features.15.se.conv2": self.features[15].se.conv2,
            "features.15.se.act2": self.features[15].se.act2,
            "features.15.se.act2.act": self.features[15].se.act2.act,
            "features.15.conv2": self.features[15].conv2,
            "features.15.conv2.conv": self.features[15].conv2.conv,
            "features.15.conv2.bn": self.features[15].conv2.bn,
            "features.16": self.features[16],
            "features.16.expand": self.features[16].expand,
            "features.16.expand.conv": self.features[16].expand.conv,
            "features.16.expand.bn": self.features[16].expand.bn,
            "features.16.expand.act": self.features[16].expand.act,
            "features.16.expand.act.act": self.features[16].expand.act.act,
            "features.16.conv1": self.features[16].conv1,
            "features.16.conv1.conv": self.features[16].conv1.conv,
            "features.16.conv1.bn": self.features[16].conv1.bn,
            "features.16.conv1.act": self.features[16].conv1.act,
            "features.16.conv1.act.act": self.features[16].conv1.act.act,
            "features.16.se": self.features[16].se,
            "features.16.se.pool": self.features[16].se.pool,
            "features.16.se.conv1": self.features[16].se.conv1,
            "features.16.se.act1": self.features[16].se.act1,
            "features.16.se.act1.act": self.features[16].se.act1.act,
            "features.16.se.conv2": self.features[16].se.conv2,
            "features.16.se.act2": self.features[16].se.act2,
            "features.16.se.act2.act": self.features[16].se.act2.act,
            "features.16.conv2": self.features[16].conv2,
            "features.16.conv2.conv": self.features[16].conv2.conv,
            "features.16.conv2.bn": self.features[16].conv2.bn,
            "features.17": self.features[17],
            "features.17.expand": self.features[17].expand,
            "features.17.expand.conv": self.features[17].expand.conv,
            "features.17.expand.bn": self.features[17].expand.bn,
            "features.17.expand.act": self.features[17].expand.act,
            "features.17.expand.act.act": self.features[17].expand.act.act,
            "features.17.conv1": self.features[17].conv1,
            "features.17.conv1.conv": self.features[17].conv1.conv,
            "features.17.conv1.bn": self.features[17].conv1.bn,
            "features.17.conv1.act": self.features[17].conv1.act,
            "features.17.conv1.act.act": self.features[17].conv1.act.act,
            "features.17.se": self.features[17].se,
            "features.17.se.pool": self.features[17].se.pool,
            "features.17.se.conv1": self.features[17].se.conv1,
            "features.17.se.act1": self.features[17].se.act1,
            "features.17.se.act1.act": self.features[17].se.act1.act,
            "features.17.se.conv2": self.features[17].se.conv2,
            "features.17.se.act2": self.features[17].se.act2,
            "features.17.se.act2.act": self.features[17].se.act2.act,
            "features.17.conv2": self.features[17].conv2,
            "features.17.conv2.conv": self.features[17].conv2.conv,
            "features.17.conv2.bn": self.features[17].conv2.bn,
            "features.18": self.features[18],
            "features.19": self.features[19],
            "features.20": self.features[20],
            "features.20.act": self.features[20].act,
            "features.21": self.features[21],
            "features.22": self.features[22],
            "features.23": self.features[23],
            "features.23.act": self.features[23].act,
            "output": self.output,

        }
        self.in_shapes = {
            'features.0': [-1, 3, 640, 640],
            'features.1': [-1, 16, 320, 320],
            'features.2.act': [-1, 16, 320, 320],
            'features.3.conv1.conv': [-1, 16, 320, 320],
            'features.3.conv1.bn': [-1, 16, 320, 320],
            'features.3.conv1.act.act': [-1, 16, 320, 320],
            'features.3.conv2.conv': [-1, 16, 320, 320],
            'features.3.conv2.bn': [-1, 16, 320, 320],
            'features.4.expand.conv': [-1, 16, 320, 320],
            'features.4.expand.bn': [-1, 64, 320, 320],
            'features.4.expand.act.act': [-1, 64, 320, 320],
            'features.4.conv1.conv': [-1, 64, 320, 320],
            'features.4.conv1.bn': [-1, 64, 160, 160],
            'features.4.conv1.act.act': [-1, 64, 160, 160],
            'features.4.conv2.conv': [-1, 64, 160, 160],
            'features.4.conv2.bn': [-1, 24, 160, 160],
            'features.5.expand.conv': [-1, 24, 160, 160],
            'features.5.expand.bn': [-1, 72, 160, 160],
            'features.5.expand.act.act': [-1, 72, 160, 160],
            'features.5.conv1.conv': [-1, 72, 160, 160],
            'features.5.conv1.bn': [-1, 72, 160, 160],
            'features.5.conv1.act.act': [-1, 72, 160, 160],
            'features.5.conv2.conv': [-1, 72, 160, 160],
            'features.5.conv2.bn': [-1, 24, 160, 16],
            'features.6.expand.conv': [-1, 24, 160, 16],
            'features.6.expand.bn': [-1, 72, 160, 160],
            'features.6.expand.act.act': [-1, 72, 160, 160],
            'features.6.conv1.conv': [-1, 72, 160, 160],
            'features.6.conv1.bn': [-1, 72, 80, 80],
            'features.6.conv1.act.act': [-1, 72, 80, 80],
            'features.6.se.pool': [-1, 72, 80, 80],
            'features.6.se.conv1': [-1, 72, 1, 1],
            'features.6.se.act1.act': [-1, 24, 1, 1],
            'features.6.se.conv2': [-1, 24, 1, 1],
            'features.6.se.act2.act': [-1, 72, 1, 1],
            'features.6.conv2.conv': [-1, 72, 1, 1],
            'features.6.conv2.bn': [-1, 40, 80, 80],
            'features.7.expand.conv': [-1, 40, 80, 80],
            'features.7.expand.bn': [-1, 120, 80, 80],
            'features.7.expand.act.act': [-1, 120, 80, 80],
            'features.7.conv1.conv': [-1, 120, 80, 80],
            'features.7.conv1.bn': [-1, 120, 80, 80],
            'features.7.conv1.act.act': [-1, 120, 80, 80],
            'features.7.se.pool': [-1, 120, 80, 80],
            'features.7.se.conv1': [-1, 120, 1, 1],
            'features.7.se.act1.act': [-1, 32, 1, 1],
            'features.7.se.conv2': [-1, 32, 1, 1],
            'features.7.se.act2.act': [-1, 120, 1, 1],
            'features.7.conv2.conv': [-1, 120, 1, 1],
            'features.7.conv2.bn': [-1, 40, 80, 80],
            'features.8.expand.conv': [-1, 40, 80, 80],
            'features.8.expand.bn': [-1, 120, 80, 80],
            'features.8.expand.act.act': [-1, 120, 80, 80],
            'features.8.conv1.conv': [-1, 120, 80, 80],
            'features.8.conv1.bn': [-1, 120, 80, 80],
            'features.8.conv1.act.act': [-1, 120, 80, 80],
            'features.8.se.pool': [-1, 120, 80, 80],
            'features.8.se.conv1': [-1, 120, 1, 1],
            'features.8.se.act1.act': [-1, 32, 1, 1],
            'features.8.se.conv2': [-1, 32, 1, 1],
            'features.8.se.act2.act': [-1, 120, 1, 1],
            'features.8.conv2.conv': [-1, 120, 1, 1],
            'features.8.conv2.bn': [-1, 40, 80, 80],
            'features.9.expand.conv': [-1, 40, 80, 80],
            'features.9.expand.bn': [-1, 240, 80, 80],
            'features.9.expand.act.act': [-1, 240, 80, 80],
            'features.9.conv1.conv': [-1, 240, 80, 80],
            'features.9.conv1.bn': [-1, 240, 40, 40],
            'features.9.conv1.act.act': [-1, 240, 40, 40],
            'features.9.conv2.conv': [-1, 240, 40, 40],
            'features.9.conv2.bn': [-1, 80, 40, 40],
            'features.10.expand.conv': [-1, 80, 40, 40],
            'features.10.expand.bn': [-1, 200, 40, 40],
            'features.10.expand.act.act': [-1, 200, 40, 40],
            'features.10.conv1.conv': [-1, 200, 40, 40],
            'features.10.conv1.bn': [-1, 200, 40, 40],
            'features.10.conv1.act.act': [-1, 200, 40, 40],
            'features.10.conv2.conv': [-1, 200, 40, 40],
            'features.10.conv2.bn': [-1, 80, 40, 40],
            'features.11.expand.conv': [-1, 80, 40, 40],
            'features.11.expand.bn': [-1, 184, 40, 40],
            'features.11.expand.act.act': [-1, 184, 40, 40],
            'features.11.conv1.conv': [-1, 184, 40, 40],
            'features.11.conv1.bn': [-1, 184, 40, 40],
            'features.11.conv1.act.act': [-1, 184, 40, 40],
            'features.11.conv2.conv': [-1, 184, 40, 40],
            'features.11.conv2.bn': [-1, 80, 40, 40],
            'features.12.expand.conv': [-1, 80, 40, 40],
            'features.12.expand.bn': [-1, 184, 40, 40],
            'features.12.expand.act.act': [-1, 184, 40, 40],
            'features.12.conv1.conv': [-1, 184, 40, 40],
            'features.12.conv1.bn': [-1, 184, 40, 40],
            'features.12.conv1.act.act': [-1, 184, 40, 40],
            'features.12.conv2.conv': [-1, 184, 40, 40],
            'features.12.conv2.bn': [-1, 80, 40, 40],
            'features.13.expand.conv': [-1, 80, 40, 40],
            'features.13.expand.bn': [-1, 480, 40, 40],
            'features.13.expand.act.act': [-1, 480, 40, 40],
            'features.13.conv1.conv': [-1, 480, 40, 40],
            'features.13.conv1.bn': [-1, 480, 40, 40],
            'features.13.conv1.act.act': [-1, 480, 40, 40],
            'features.13.se.pool': [-1, 480, 40, 40],
            'features.13.se.conv1': [-1, 480, 1, 1],
            'features.13.se.act1.act': [-1, 120, 1, 1],
            'features.13.se.conv2': [-1, 120, 1, 1],
            'features.13.se.act2.act': [-1, 120, 1, 1],
            'features.13.conv2.conv': [-1, 120, 1, 1],
            'features.13.conv2.bn': [-1, 112, 40, 40],
            'features.14.expand.conv': [-1, 112, 40, 40],
            'features.14.expand.bn': [-1, 672, 40, 40],
            'features.14.expand.act.act': [-1, 672, 40, 40],
            'features.14.conv1.conv': [-1, 672, 40, 40],
            'features.14.conv1.bn': [-1, 672, 40, 40],
            'features.14.conv1.act.act': [-1, 672, 40, 40],
            'features.14.se.pool': [-1, 672, 40, 40],
            'features.14.se.conv1': [-1, 672, 1, 1],
            'features.14.se.act1.act': [-1, 168, 1, 1],
            'features.14.se.conv2': [-1, 168, 1, 1],
            'features.14.se.act2.act': [-1, 168, 1, 1],
            'features.14.conv2.conv': [-1, 168, 1, 1],
            'features.14.conv2.bn': [-1, 112, 40, 40],
            'features.15.expand.conv': [-1, 112, 40, 40],
            'features.15.expand.bn': [-1, 672, 40, 40],
            'features.15.expand.act.act': [-1, 672, 40, 40],
            'features.15.conv1.conv': [-1, 672, 40, 40],
            'features.15.conv1.bn': [-1, 672, 20, 20],
            'features.15.conv1.act.act': [-1, 672, 20, 20],
            'features.15.se.pool': [-1, 672, 20, 20],
            'features.15.se.conv1': [-1, 672, 1, 1],
            'features.15.se.act1.act': [-1, 168, 1, 1],
            'features.15.se.conv2': [-1, 168, 1, 1],
            'features.15.se.act2.act': [-1, 672, 1, 1],
            'features.15.conv2.conv': [-1, 672, 1, 1],
            'features.15.conv2.bn': [-1, 160, 20, 20],
            'features.16.expand.conv': [-1, 160, 20, 20],
            'features.16.expand.bn': [-1, 960, 20, 20],
            'features.16.expand.act.act': [-1, 960, 20, 20],
            'features.16.conv1.conv': [-1, 960, 20, 20],
            'features.16.conv1.bn': [-1, 960, 20, 20],
            'features.16.conv1.act.act': [-1, 960, 20, 20],
            'features.16.se.pool': [-1, 960, 20, 20],
            'features.16.se.conv1': [-1, 960, 20, 20],
            'features.16.se.act1.act': [-1, 240, 1, 1],
            'features.16.se.conv2': [-1, 240, 1, 1],
            'features.16.se.act2.act': [-1, 960, 1, 1],
            'features.16.conv2.conv': [-1, 960, 1, 1],
            'features.16.conv2.bn': [-1, 160, 20, 20],
            'features.17.expand.conv': [-1, 160, 20, 20],
            'features.17.expand.bn': [-1, 960, 20, 20],
            'features.17.expand.act.act': [-1, 960, 20, 20],
            'features.17.conv1.conv': [-1, 960, 20, 20],
            'features.17.conv1.bn': [-1, 960, 20, 20],
            'features.17.conv1.act.act': [-1, 960, 20, 20],
            'features.17.se.pool': [-1, 960, 20, 20],
            'features.17.se.conv1': [-1, 960, 1, 1],
            'features.17.se.act1.act': [-1, 240, 1, 1],
            'features.17.se.conv2': [-1, 240, 1, 1],
            'features.17.se.act2.act': [-1, 960, 1, 1],
            'features.17.conv2.conv': [-1, 960, 1, 1],
            'features.17.conv2.bn': [-1, 160, 20, 20],
            'features.18': [-1, 160, 20, 20],
            'features.19': [-1, 960, 20, 20],
            'features.20.act': [-1, 960, 20, 20],
            'features.21': [-1, 960, 20, 20],
            'features.22': [-1, 960, 1, 1],
            'features.23.act': [-1, 1280, 1, 1],
            'output': [-1, 1280, 1, 1],

        }
        self.out_shapes = {'INPUT': [-1, 32, 640, 640],
                           'features.0': [-1, 16, 320, 320],
                           'features.1': [-1, 16, 320, 320],
                           'features.2.act': [-1, 16, 320, 320],
                           'features.3.conv1.conv': [-1, 16, 320, 320],
                           'features.3.conv1.bn': [-1, 16, 320, 320],
                           'features.3.conv1.act.act': [-1, 16, 320, 320],
                           'features.3.conv2.conv': [-1, 16, 320, 320],
                           'features.3.conv2.bn': [-1, 16, 320, 320],
                           'features.4.expand.conv': [-1, 64, 320, 320],
                           'features.4.expand.bn': [-1, 64, 320, 320],
                           'features.4.expand.act.act': [-1, 64, 320, 320],
                           'features.4.conv1.conv': [-1, 64, 160, 160],
                           'features.4.conv1.bn': [-1, 64, 160, 160],
                           'features.4.conv1.act.act': [-1, 64, 160, 160],
                           'features.4.conv2.conv': [-1, 24, 160, 160],
                           'features.4.conv2.bn': [-1, 24, 160, 160],
                           'features.5.expand.conv': [-1, 72, 160, 160],
                           'features.5.expand.bn': [-1, 72, 160, 160],
                           'features.5.expand.act.act': [-1, 72, 160, 160],
                           'features.5.conv1.conv': [-1, 72, 160, 160],
                           'features.5.conv1.bn': [-1, 72, 160, 160],
                           'features.5.conv1.act.act': [-1, 72, 160, 160],
                           'features.5.conv2.conv': [-1, 24, 160, 16],
                           'features.5.conv2.bn': [-1, 24, 160, 16],
                           'features.6.expand.conv': [-1, 72, 160, 160],
                           'features.6.expand.bn': [-1, 72, 160, 160],
                           'features.6.expand.act.act': [-1, 72, 160, 160],
                           'features.6.conv1.conv': [-1, 72, 80, 80],
                           'features.6.conv1.bn': [-1, 72, 80, 80],
                           'features.6.conv1.act.act': [-1, 72, 80, 80],
                           'features.6.se.pool': [-1, 72, 1, 1],
                           'features.6.se.conv1': [-1, 24, 1, 1],
                           'features.6.se.act1.act': [-1, 24, 1, 1],
                           'features.6.se.conv2': [-1, 72, 1, 1],
                           'features.6.se.act2.act': [-1, 72, 1, 1],
                           'features.6.conv2.conv': [-1, 40, 80, 80],
                           'features.6.conv2.bn': [-1, 40, 80, 80],
                           'features.7.expand.conv': [-1, 120, 80, 80],
                           'features.7.expand.bn': [-1, 120, 80, 80],
                           'features.7.expand.act.act': [-1, 120, 80, 80],
                           'features.7.conv1.conv': [-1, 120, 80, 80],
                           'features.7.conv1.bn': [-1, 120, 80, 80],
                           'features.7.conv1.act.act': [-1, 120, 80, 80],
                           'features.7.se.pool': [-1, 120, 1, 1],
                           'features.7.se.conv1': [-1, 32, 1, 1],
                           'features.7.se.act1.act': [-1, 32, 1, 1],
                           'features.7.se.conv2': [-1, 120, 1, 1],
                           'features.7.se.act2.act': [-1, 120, 1, 1],
                           'features.7.conv2.conv': [-1, 40, 80, 80],
                           'features.7.conv2.bn': [-1, 40, 80, 80],
                           'features.8.expand.conv': [-1, 120, 80, 80],
                           'features.8.expand.bn': [-1, 120, 80, 80],
                           'features.8.expand.act.act': [-1, 120, 80, 80],
                           'features.8.conv1.conv': [-1, 120, 80, 80],
                           'features.8.conv1.bn': [-1, 120, 80, 80],
                           'features.8.conv1.act.act': [-1, 120, 80, 80],
                           'features.8.se.pool': [-1, 120, 1, 1],
                           'features.8.se.conv1': [-1, 32, 1, 1],
                           'features.8.se.act1.act': [-1, 32, 1, 1],
                           'features.8.se.conv2': [-1, 120, 1, 1],
                           'features.8.se.act2.act': [-1, 120, 1, 1],
                           'features.8.conv2.conv': [-1, 40, 80, 80],
                           'features.8.conv2.bn': [-1, 40, 80, 80],
                           'features.9.expand.conv': [-1, 240, 80, 80],
                           'features.9.expand.bn': [-1, 240, 80, 80],
                           'features.9.expand.act.act': [-1, 240, 80, 80],
                           'features.9.conv1.conv': [-1, 240, 40, 40],
                           'features.9.conv1.bn': [-1, 240, 40, 40],
                           'features.9.conv1.act.act': [-1, 240, 40, 40],
                           'features.9.conv2.conv': [-1, 80, 40, 40],
                           'features.9.conv2.bn': [-1, 80, 40, 40],
                           'features.10.expand.conv': [-1, 200, 40, 40],
                           'features.10.expand.bn': [-1, 200, 40, 40],
                           'features.10.expand.act.act': [-1, 200, 40, 40],
                           'features.10.conv1.conv': [-1, 200, 40, 40],
                           'features.10.conv1.bn': [-1, 200, 40, 40],
                           'features.10.conv1.act.act': [-1, 200, 40, 40],
                           'features.10.conv2.conv': [-1, 80, 40, 40],
                           'features.10.conv2.bn': [-1, 80, 40, 40],
                           'features.11.expand.conv': [-1, 184, 40, 40],
                           'features.11.expand.bn': [-1, 184, 40, 40],
                           'features.11.expand.act.act': [-1, 184, 40, 40],
                           'features.11.conv1.conv': [-1, 184, 40, 40],
                           'features.11.conv1.bn': [-1, 184, 40, 40],
                           'features.11.conv1.act.act': [-1, 184, 40, 40],
                           'features.11.conv2.conv': [-1, 80, 40, 40],
                           'features.11.conv2.bn': [-1, 80, 40, 40],
                           'features.12.expand.conv': [-1, 184, 40, 40],
                           'features.12.expand.bn': [-1, 184, 40, 40],
                           'features.12.expand.act.act': [-1, 184, 40, 40],
                           'features.12.conv1.conv': [-1, 184, 40, 40],
                           'features.12.conv1.bn': [-1, 184, 40, 40],
                           'features.12.conv1.act.act': [-1, 184, 40, 40],
                           'features.12.conv2.conv': [-1, 80, 40, 40],
                           'features.12.conv2.bn': [-1, 80, 40, 40],
                           'features.13.expand.conv': [-1, 480, 40, 40],
                           'features.13.expand.bn': [-1, 480, 40, 40],
                           'features.13.expand.act.act': [-1, 480, 40, 40],
                           'features.13.conv1.conv': [-1, 480, 40, 40],
                           'features.13.conv1.bn': [-1, 480, 40, 40],
                           'features.13.conv1.act.act': [-1, 480, 40, 40],
                           'features.13.se.pool': [-1, 480, 1, 1],
                           'features.13.se.conv1': [-1, 120, 1, 1],
                           'features.13.se.act1.act': [-1, 120, 1, 1],
                           'features.13.se.conv2': [-1, 120, 1, 1],
                           'features.13.se.act2.act': [-1, 120, 1, 1],
                           'features.13.conv2.conv': [-1, 112, 40, 40],
                           'features.13.conv2.bn': [-1, 112, 40, 40],
                           'features.14.expand.conv': [-1, 672, 40, 40],
                           'features.14.expand.bn': [-1, 672, 40, 40],
                           'features.14.expand.act.act': [-1, 672, 40, 40],
                           'features.14.conv1.conv': [-1, 672, 40, 40],
                           'features.14.conv1.bn': [-1, 672, 40, 40],
                           'features.14.conv1.act.act': [-1, 672, 40, 40],
                           'features.14.se.pool': [-1, 672, 1, 1],
                           'features.14.se.conv1': [-1, 168, 1, 1],
                           'features.14.se.act1.act': [-1, 168, 1, 1],
                           'features.14.se.conv2': [-1, 168, 1, 1],
                           'features.14.se.act2.act': [-1, 168, 1, 1],
                           'features.14.conv2.conv': [-1, 112, 40, 40],
                           'features.14.conv2.bn': [-1, 112, 40, 40],
                           'features.15.expand.conv': [-1, 672, 40, 40],
                           'features.15.expand.bn': [-1, 672, 40, 40],
                           'features.15.expand.act.act': [-1, 672, 40, 40],
                           'features.15.conv1.conv': [-1, 672, 20, 20],
                           'features.15.conv1.bn': [-1, 672, 20, 20],
                           'features.15.conv1.act.act': [-1, 672, 20, 20],
                           'features.15.se.pool': [-1, 672, 1, 1],
                           'features.15.se.conv1': [-1, 168, 1, 1],
                           'features.15.se.act1.act': [-1, 168, 1, 1],
                           'features.15.se.conv2': [-1, 672, 1, 1],
                           'features.15.se.act2.act': [-1, 672, 1, 1],
                           'features.15.conv2.conv': [-1, 160, 20, 20],
                           'features.15.conv2.bn': [-1, 160, 20, 20],
                           'features.16.expand.conv': [-1, 960, 20, 20],
                           'features.16.expand.bn': [-1, 960, 20, 20],
                           'features.16.expand.act.act': [-1, 960, 20, 20],
                           'features.16.conv1.conv': [-1, 960, 20, 20],
                           'features.16.conv1.bn': [-1, 960, 20, 20],
                           'features.16.conv1.act.act': [-1, 960, 20, 20],
                           'features.16.se.pool': [-1, 960, 20, 20],
                           'features.16.se.conv1': [-1, 240, 1, 1],
                           'features.16.se.act1.act': [-1, 240, 1, 1],
                           'features.16.se.conv2': [-1, 960, 1, 1],
                           'features.16.se.act2.act': [-1, 960, 1, 1],
                           'features.16.conv2.conv': [-1, 160, 20, 20],
                           'features.16.conv2.bn': [-1, 160, 20, 20],
                           'features.17.expand.conv': [-1, 960, 20, 20],
                           'features.17.expand.bn': [-1, 960, 20, 20],
                           'features.17.expand.act.act': [-1, 960, 20, 20],
                           'features.17.conv1.conv': [-1, 960, 20, 20],
                           'features.17.conv1.bn': [-1, 960, 20, 20],
                           'features.17.conv1.act.act': [-1, 960, 20, 20],
                           'features.17.se.pool': [-1, 960, 1, 1],
                           'features.17.se.conv1': [-1, 240, 1, 1],
                           'features.17.se.act1.act': [-1, 240, 1, 1],
                           'features.17.se.conv2': [-1, 960, 1, 1],
                           'features.17.se.act2.act': [-1, 960, 1, 1],
                           'features.17.conv2.conv': [-1, 160, 20, 20],
                           'features.17.conv2.bn': [-1, 160, 20, 20],
                           'features.18': [-1, 960, 20, 20],
                           'features.19': [-1, 960, 20, 20],
                           'features.20.act': [-1, 960, 20, 20],
                           'features.21': [-1, 960, 1, 1],
                           'features.22': [-1, 1280, 1, 1],
                           'features.23.act': [-1, 1280, 1, 1],
                           'output': [-1, 1000, 1, 1],
                           }
        self.orders = {
            'features.0': ["INPUT", "features.1"],
            'features.1': ["features.0", "features.2.act"],
            'features.2.act': ["features.1", "features.3.conv1.conv"],
            'features.3.conv1.conv': ["features.2.act", "features.3.conv1.bn"],
            'features.3.conv1.bn': ["features.3.conv1.conv", "features.3.conv1.act.act"],
            'features.3.conv1.act.act': ["features.3.conv1.bn", "features.3.conv2.conv"],
            'features.3.conv2.conv': ["features.3.conv1.act.act", "features.3.conv2.bn"],
            'features.3.conv2.bn': ["features.3.conv2.conv", "features.4.expand.conv"],
            'features.4.expand.conv': ["features.3.conv2.bn", "features.4.expand.bn"],
            'features.4.expand.bn': ["features.4.expand.conv", "features.4.expand.act.act"],
            'features.4.expand.act.act': ["features.4.expand.bn", "features.4.conv1.conv"],
            'features.4.conv1.conv': ["features.4.expand.act.act", "features.4.conv1.bn"],
            'features.4.conv1.bn': ["features.4.conv1.conv", "features.4.conv1.act.act"],
            'features.4.conv1.act.act': ["features.4.conv1.bn", "features.4.conv2.conv"],
            'features.4.conv2.conv': ["features.4.conv1.act.act", "features.4.conv2.bn"],
            'features.4.conv2.bn': ["features.4.conv2.conv", "features.5.expand.conv"],
            'features.5.expand.conv': ["features.4.conv2.bn", "features.5.expand.bn"],
            'features.5.expand.bn': ["features.5.expand.conv", "features.5.expand.act.act"],
            'features.5.expand.act.act': ["features.5.expand.bn", "features.5.conv1.conv"],
            'features.5.conv1.conv': ["features.5.expand.act.act", "features.5.conv1.bn"],
            'features.5.conv1.bn': ["features.5.conv1.conv", "features.5.conv1.act.act"],
            'features.5.conv1.act.act': ["features.5.conv1.bn", "features.5.conv2.conv"],
            'features.5.conv2.conv': ["features.5.conv1.act.act", "features.5.conv2.bn"],
            'features.5.conv2.bn': ["features.5.conv2.conv", "features.6.expand.conv"],
            'features.6.expand.conv': ["features.5.conv2.bn", "features.6.expand.bn"],
            'features.6.expand.bn': ["features.6.expand.conv", "features.6.expand.act.act"],
            'features.6.expand.act.act': ["features.6.expand.bn", "features.6.conv1.conv"],
            'features.6.conv1.conv': ["features.6.expand.act.act", "features.6.conv1.bn"],
            'features.6.conv1.bn': ["features.6.conv1.conv", "features.6.conv1.act.act"],
            'features.6.conv1.act.act': ["features.6.conv1.bn", "features.6.se.pool"],
            'features.6.se.pool': ["features.6.conv1.act.act", "features.6.se.conv1"],
            'features.6.se.conv1': ["features.6.se.pool", "features.6.se.act1.act"],
            'features.6.se.act1.act': ["features.6.se.conv1", "features.6.se.conv2"],
            'features.6.se.conv2': ["features.6.se.act1.act", "features.6.se.act2.act"],
            'features.6.se.act2.act': ["features.6.se.conv2", "features.6.conv2.conv"],
            'features.6.conv2.conv': ["features.6.se.act2.act", "features.6.conv2.bn"],
            'features.6.conv2.bn': ["features.6.conv2.conv", "features.7.expand.conv"],
            'features.7.expand.conv': ["features.6.conv2.bn", "features.7.expand.bn"],
            'features.7.expand.bn': ["features.7.expand.conv", "features.7.expand.act.act"],
            'features.7.expand.act.act': ["features.7.expand.bn", "features.7.conv1.conv"],
            'features.7.conv1.conv': ["features.7.expand.act.act", "features.7.conv1.bn"],
            'features.7.conv1.bn': ["features.7.conv1.conv", "features.7.conv1.act.act"],
            'features.7.conv1.act.act': ["features.7.conv1.bn", "features.7.se.pool"],
            'features.7.se.pool': ["features.7.conv1.act.act", "features.7.se.conv1"],
            'features.7.se.conv1': ["features.7.se.pool", "features.7.se.act1.act"],
            'features.7.se.act1.act': ["features.7.se.conv1", "features.7.se.conv2"],
            'features.7.se.conv2': ["features.7.se.act1.act", "features.7.se.act2.act"],
            'features.7.se.act2.act': ["features.7.se.conv2", "features.7.conv2.conv"],
            'features.7.conv2.conv': ["features.7.se.act2.act", "features.7.conv2.bn"],
            'features.7.conv2.bn': ["features.7.conv2.conv", "features.8.expand.conv"],
            'features.8.expand.conv': ["features.7.conv2.bn", "features.8.expand.bn"],
            'features.8.expand.bn': ["features.8.expand.conv", "features.8.expand.act.act"],
            'features.8.expand.act.act': ["features.8.expand.bn", "features.8.conv1.conv"],
            'features.8.conv1.conv': ["features.8.expand.act.act", "features.8.conv1.bn"],
            'features.8.conv1.bn': ["features.8.conv1.conv", "features.8.conv1.act.act"],
            'features.8.conv1.act.act': ["features.8.conv1.bn", "features.8.se.pool"],
            'features.8.se.pool': ["features.8.conv1.act.act", "features.8.se.conv1"],
            'features.8.se.conv1': ["features.8.se.pool", "features.8.se.act1.act"],
            'features.8.se.act1.act': ["features.8.se.conv1", "features.8.se.conv2"],
            'features.8.se.conv2': ["features.8.se.act1.act", "features.8.se.act2.act"],
            'features.8.se.act2.act': ["features.8.se.conv2", "features.8.conv2.conv"],
            'features.8.conv2.conv': ["features.8.se.act2.act", "features.8.conv2.bn"],
            'features.8.conv2.bn': ["features.8.conv2.conv", "features.9.expand.conv"],
            'features.9.expand.conv': ["features.8.conv2.bn", "features.9.expand.bn"],
            'features.9.expand.bn': ["features.9.expand.conv", "features.9.expand.act.act"],
            'features.9.expand.act.act': ["features.9.expand.bn", "features.9.conv1.conv"],
            'features.9.conv1.conv': ["features.9.expand.act.act", "features.9.conv1.bn"],
            'features.9.conv1.bn': ["features.9.conv1.conv", "features.9.conv1.act.act"],
            'features.9.conv1.act.act': ["features.9.conv1.bn", "features.9.conv2.conv"],
            'features.9.conv2.conv': ["features.9.conv1.act.act", "features.9.conv2.bn"],
            'features.9.conv2.bn': ["features.9.conv2.conv", "features.10.expand.conv"],
            'features.10.expand.conv': ["features.9.conv2.bn", "features.10.expand.bn"],
            'features.10.expand.bn': ["features.10.expand.conv", "features.10.expand.act.act"],
            'features.10.expand.act.act': ["features.10.expand.bn", "features.10.conv1.conv"],
            'features.10.conv1.conv': ["features.10.expand.act.act", "features.10.conv1.bn"],
            'features.10.conv1.bn': ["features.10.conv1.conv", "features.10.conv1.act.act"],
            'features.10.conv1.act.act': ["features.10.conv1.bn", "features.10.conv2.conv"],
            'features.10.conv2.conv': ["features.10.conv1.act.act", "features.10.conv2.bn"],
            'features.10.conv2.bn': ["features.10.conv2.conv", "features.11.expand.conv"],
            'features.11.expand.conv': ["features.10.conv2.bn", "features.11.expand.bn"],
            'features.11.expand.bn': ["features.11.expand.conv", "features.11.expand.act.act"],
            'features.11.expand.act.act': ["features.11.expand.bn", "features.11.conv1.conv"],
            'features.11.conv1.conv': ["features.11.expand.act.act", "features.11.conv1.bn"],
            'features.11.conv1.bn': ["features.11.conv1.conv", "features.11.conv1.act.act"],
            'features.11.conv1.act.act': ["features.11.conv1.bn", "features.11.conv2.conv"],
            'features.11.conv2.conv': ["features.11.conv1.act.act", "features.11.conv2.bn"],
            'features.11.conv2.bn': ["features.11.conv2.conv", "features.12.expand.conv"],
            'features.12.expand.conv': ["features.11.conv2.bn", "features.12.expand.bn"],
            'features.12.expand.bn': ["features.12.expand.conv", "features.12.expand.act.act"],
            'features.12.expand.act.act': ["features.12.expand.bn", "features.12.conv1.conv"],
            'features.12.conv1.conv': ["features.12.expand.act.act", "features.12.conv1.bn"],
            'features.12.conv1.bn': ["features.12.conv1.conv", "features.12.conv1.act.act"],
            'features.12.conv1.act.act': ["features.12.conv1.bn", "features.12.conv2.conv"],
            'features.12.conv2.conv': ["features.12.conv1.act.act", "features.12.conv2.bn"],
            'features.12.conv2.bn': ["features.12.conv2.conv", "features.13.expand.conv"],
            'features.13.expand.conv': ["features.12.conv2.bn", "features.13.expand.bn"],
            'features.13.expand.bn': ["features.13.expand.conv", "features.13.expand.act.act"],
            'features.13.expand.act.act': ["features.13.expand.bn", "features.13.conv1.conv"],
            'features.13.conv1.conv': ["features.13.expand.act.act", "features.13.conv1.bn"],
            'features.13.conv1.bn': ["features.13.conv1.conv", "features.13.conv1.act.act"],
            'features.13.conv1.act.act': ["features.13.conv1.bn", "features.13.se.pool"],
            'features.13.se.pool': ["features.13.conv1.act.act", "features.13.se.conv1"],
            'features.13.se.conv1': ["features.13.se.pool", "features.13.se.act1.act"],
            'features.13.se.act1.act': ["features.13.se.conv1", "features.13.se.conv2"],
            'features.13.se.conv2': ["features.13.se.act1.act", "features.13.se.act2.act"],
            'features.13.se.act2.act': ["features.13.se.conv2", "features.13.conv2.conv"],
            'features.13.conv2.conv': ["features.13.se.act2.act", "features.13.conv2.bn"],
            'features.13.conv2.bn': ["features.13.conv2.conv", "features.14.expand.conv"],
            'features.14.expand.conv': ["features.13.conv2.bn", "features.14.expand.bn"],
            'features.14.expand.bn': ["features.14.expand.conv", "features.14.expand.act.act"],
            'features.14.expand.act.act': ["features.14.expand.bn", "features.14.conv1.conv"],
            'features.14.conv1.conv': ["features.14.expand.act.act", "features.14.conv1.bn"],
            'features.14.conv1.bn': ["features.14.conv1.conv", "features.14.conv1.act.act"],
            'features.14.conv1.act.act': ["features.14.conv1.bn", "features.14.se.pool"],
            'features.14.se.pool': ["features.14.conv1.act.act", "features.14.se.conv1"],
            'features.14.se.conv1': ["features.14.se.pool", "features.14.se.act1.act"],
            'features.14.se.act1.act': ["features.14.se.conv1", "features.14.se.conv2"],
            'features.14.se.conv2': ["features.14.se.act1.act", "features.14.se.act2.act"],
            'features.14.se.act2.act': ["features.14.se.conv2", "features.14.conv2.conv"],
            'features.14.conv2.conv': ["features.14.se.act2.act", "features.14.conv2.bn"],
            'features.14.conv2.bn': ["features.14.conv2.conv", "features.15.expand.conv"],
            'features.15.expand.conv': ["features.14.conv2.bn", "features.15.expand.bn"],
            'features.15.expand.bn': ["features.15.expand.conv", "features.15.expand.act.act"],
            'features.15.expand.act.act': ["features.15.expand.bn", "features.15.conv1.conv"],
            'features.15.conv1.conv': ["features.15.expand.act.act", "features.15.conv1.bn"],
            'features.15.conv1.bn': ["features.15.conv1.conv", "features.15.conv1.act.act"],
            'features.15.conv1.act.act': ["features.15.conv1.bn", "features.15.se.pool"],
            'features.15.se.pool': ["features.15.conv1.act.act", "features.15.se.conv1"],
            'features.15.se.conv1': ["features.15.se.pool", "features.15.se.act1.act"],
            'features.15.se.act1.act': ["features.15.se.conv1", "features.15.se.conv2"],
            'features.15.se.conv2': ["features.15.se.act1.act", "features.15.se.act2.act"],
            'features.15.se.act2.act': ["features.15.se.conv2", "features.15.conv2.conv"],
            'features.15.conv2.conv': ["features.15.se.act2.act", "features.15.conv2.bn"],
            'features.15.conv2.bn': ["features.15.conv2.conv", "features.16.expand.conv"],
            'features.16.expand.conv': ["features.15.conv2.bn", "features.16.expand.bn"],
            'features.16.expand.bn': ["features.16.expand.conv", "features.16.expand.act.act"],
            'features.16.expand.act.act': ["features.16.expand.bn", "features.16.conv1.conv"],
            'features.16.conv1.conv': ["features.16.expand.act.act", "features.16.conv1.bn"],
            'features.16.conv1.bn': ["features.16.conv1.conv", "features.16.conv1.act.act"],
            'features.16.conv1.act.act': ["features.16.conv1.bn", "features.16.se.pool"],
            'features.16.se.pool': ["features.16.conv1.act.act", "features.16.se.conv1"],
            'features.16.se.conv1': ["features.16.se.pool", "features.16.se.act1.act"],
            'features.16.se.act1.act': ["features.16.se.conv1", "features.16.se.conv2"],
            'features.16.se.conv2': ["features.16.se.act1.act", "features.16.se.act2.act"],
            'features.16.se.act2.act': ["features.16.se.conv2", "features.16.conv2.conv"],
            'features.16.conv2.conv': ["features.16.se.act2.act", "features.16.conv2.bn"],
            'features.16.conv2.bn': ["features.16.conv2.conv", "features.17.expand.conv"],
            'features.17.expand.conv': ["features.16.conv2.bn", "features.17.expand.bn"],
            'features.17.expand.bn': ["features.17.expand.conv", "features.17.expand.act.act"],
            'features.17.expand.act.act': ["features.17.expand.bn", "features.17.conv1.conv"],
            'features.17.conv1.conv': ["features.17.expand.act.act", "features.17.conv1.bn"],
            'features.17.conv1.bn': ["features.17.conv1.conv", "features.17.conv1.act.act"],
            'features.17.conv1.act.act': ["features.17.conv1.bn", "features.17.se.pool"],
            'features.17.se.pool': ["features.17.conv1.act.act", "features.17.se.conv1"],
            'features.17.se.conv1': ["features.17.se.pool", "features.17.se.act1.act"],
            'features.17.se.act1.act': ["features.17.se.conv1", "features.17.se.conv2"],
            'features.17.se.conv2': ["features.17.se.act1.act", "features.17.se.act2.act"],
            'features.17.se.act2.act': ["features.17.se.conv2", "features.17.conv2.conv"],
            'features.17.conv2.conv': ["features.17.se.act2.act", "features.17.conv2.bn"],
            'features.17.conv2.bn': ["features.17.conv2.conv", "features.18"],
            'features.18': ["features.17.conv2.bn", "features.19"],
            'features.19': ["features.18", "features.20.act"],
            'features.20.act': ["features.19", "features.21"],
            'features.21': ["features.20.act", "features.22"],
            'features.22': ["features.21", "features.23.act"],
            'features.23.act': ["features.22", "output"],
            'output': ["features.23.act", "OUTPUT"],

        }
        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

    def construct(self, x):
        x = self.features(x)
        if self.include_top:
            x = self.output(x)
            x = self.squeeze(x)
            if self.need_activation:
                x = self.activation(x)
        return x

    def _make_layer(self, kernel_size, exp_ch, out_channel, use_se, act_func, stride=1):
        mid_planes = exp_ch
        out_planes = out_channel

        layer = ResUnit(self.inplanes, mid_planes, out_planes,
                        kernel_size, stride=stride, act_type=act_func, use_se=use_se)
        self.inplanes = out_planes
        return layer

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, (nn.Conv2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(ms.Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                             m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        ms.numpy.zeros(m.bias.data.shape, dtype="float32"))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    ms.Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    ms.numpy.zeros(m.beta.data.shape, dtype="float32"))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(ms.Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(ms.numpy.zeros(m.bias.data.shape, dtype="float32"))

    def set_layers(self,layer_name,new_layer):
        if 'features' == layer_name:
            self.features= new_layer
            self.layer_names["features"]=new_layer
            self.origin_layer_names["features"]=new_layer
        elif 'features.0' == layer_name:
            self.features[0]= new_layer
            self.layer_names["features.0"]=new_layer
            self.origin_layer_names["features.0"]=new_layer
        elif 'features.1' == layer_name:
            self.features[1]= new_layer
            self.layer_names["features.1"]=new_layer
            self.origin_layer_names["features.1"]=new_layer
        elif 'features.2' == layer_name:
            self.features[2]= new_layer
            self.layer_names["features.2"]=new_layer
            self.origin_layer_names["features.2"]=new_layer
        elif 'features.2.act' == layer_name:
            self.features[2].act= new_layer
            self.layer_names["features.2.act"]=new_layer
            self.origin_layer_names["features.2.act"]=new_layer
        elif 'features.3' == layer_name:
            self.features[3]= new_layer
            self.layer_names["features.3"]=new_layer
            self.origin_layer_names["features.3"]=new_layer
        elif 'features.3.conv1' == layer_name:
            self.features[3].conv1= new_layer
            self.layer_names["features.3.conv1"]=new_layer
            self.origin_layer_names["features.3.conv1"]=new_layer
        elif 'features.3.conv1.conv' == layer_name:
            self.features[3].conv1.conv= new_layer
            self.layer_names["features.3.conv1.conv"]=new_layer
            self.origin_layer_names["features.3.conv1.conv"]=new_layer
        elif 'features.3.conv1.bn' == layer_name:
            self.features[3].conv1.bn= new_layer
            self.layer_names["features.3.conv1.bn"]=new_layer
            self.origin_layer_names["features.3.conv1.bn"]=new_layer
        elif 'features.3.conv1.act' == layer_name:
            self.features[3].conv1.act= new_layer
            self.layer_names["features.3.conv1.act"]=new_layer
            self.origin_layer_names["features.3.conv1.act"]=new_layer
        elif 'features.3.conv1.act.act' == layer_name:
            self.features[3].conv1.act.act= new_layer
            self.layer_names["features.3.conv1.act.act"]=new_layer
            self.origin_layer_names["features.3.conv1.act.act"]=new_layer
        elif 'features.3.conv2' == layer_name:
            self.features[3].conv2= new_layer
            self.layer_names["features.3.conv2"]=new_layer
            self.origin_layer_names["features.3.conv2"]=new_layer
        elif 'features.3.conv2.conv' == layer_name:
            self.features[3].conv2.conv= new_layer
            self.layer_names["features.3.conv2.conv"]=new_layer
            self.origin_layer_names["features.3.conv2.conv"]=new_layer
        elif 'features.3.conv2.bn' == layer_name:
            self.features[3].conv2.bn= new_layer
            self.layer_names["features.3.conv2.bn"]=new_layer
            self.origin_layer_names["features.3.conv2.bn"]=new_layer
        elif 'features.4' == layer_name:
            self.features[4]= new_layer
            self.layer_names["features.4"]=new_layer
            self.origin_layer_names["features.4"]=new_layer
        elif 'features.4.expand' == layer_name:
            self.features[4].expand= new_layer
            self.layer_names["features.4.expand"]=new_layer
            self.origin_layer_names["features.4.expand"]=new_layer
        elif 'features.4.expand.conv' == layer_name:
            self.features[4].expand.conv= new_layer
            self.layer_names["features.4.expand.conv"]=new_layer
            self.origin_layer_names["features.4.expand.conv"]=new_layer
        elif 'features.4.expand.bn' == layer_name:
            self.features[4].expand.bn= new_layer
            self.layer_names["features.4.expand.bn"]=new_layer
            self.origin_layer_names["features.4.expand.bn"]=new_layer
        elif 'features.4.expand.act' == layer_name:
            self.features[4].expand.act= new_layer
            self.layer_names["features.4.expand.act"]=new_layer
            self.origin_layer_names["features.4.expand.act"]=new_layer
        elif 'features.4.expand.act.act' == layer_name:
            self.features[4].expand.act.act= new_layer
            self.layer_names["features.4.expand.act.act"]=new_layer
            self.origin_layer_names["features.4.expand.act.act"]=new_layer
        elif 'features.4.conv1' == layer_name:
            self.features[4].conv1= new_layer
            self.layer_names["features.4.conv1"]=new_layer
            self.origin_layer_names["features.4.conv1"]=new_layer
        elif 'features.4.conv1.conv' == layer_name:
            self.features[4].conv1.conv= new_layer
            self.layer_names["features.4.conv1.conv"]=new_layer
            self.origin_layer_names["features.4.conv1.conv"]=new_layer
        elif 'features.4.conv1.bn' == layer_name:
            self.features[4].conv1.bn= new_layer
            self.layer_names["features.4.conv1.bn"]=new_layer
            self.origin_layer_names["features.4.conv1.bn"]=new_layer
        elif 'features.4.conv1.act' == layer_name:
            self.features[4].conv1.act= new_layer
            self.layer_names["features.4.conv1.act"]=new_layer
            self.origin_layer_names["features.4.conv1.act"]=new_layer
        elif 'features.4.conv1.act.act' == layer_name:
            self.features[4].conv1.act.act= new_layer
            self.layer_names["features.4.conv1.act.act"]=new_layer
            self.origin_layer_names["features.4.conv1.act.act"]=new_layer
        elif 'features.4.conv2' == layer_name:
            self.features[4].conv2= new_layer
            self.layer_names["features.4.conv2"]=new_layer
            self.origin_layer_names["features.4.conv2"]=new_layer
        elif 'features.4.conv2.conv' == layer_name:
            self.features[4].conv2.conv= new_layer
            self.layer_names["features.4.conv2.conv"]=new_layer
            self.origin_layer_names["features.4.conv2.conv"]=new_layer
        elif 'features.4.conv2.bn' == layer_name:
            self.features[4].conv2.bn= new_layer
            self.layer_names["features.4.conv2.bn"]=new_layer
            self.origin_layer_names["features.4.conv2.bn"]=new_layer
        elif 'features.5' == layer_name:
            self.features[5]= new_layer
            self.layer_names["features.5"]=new_layer
            self.origin_layer_names["features.5"]=new_layer
        elif 'features.5.expand' == layer_name:
            self.features[5].expand= new_layer
            self.layer_names["features.5.expand"]=new_layer
            self.origin_layer_names["features.5.expand"]=new_layer
        elif 'features.5.expand.conv' == layer_name:
            self.features[5].expand.conv= new_layer
            self.layer_names["features.5.expand.conv"]=new_layer
            self.origin_layer_names["features.5.expand.conv"]=new_layer
        elif 'features.5.expand.bn' == layer_name:
            self.features[5].expand.bn= new_layer
            self.layer_names["features.5.expand.bn"]=new_layer
            self.origin_layer_names["features.5.expand.bn"]=new_layer
        elif 'features.5.expand.act' == layer_name:
            self.features[5].expand.act= new_layer
            self.layer_names["features.5.expand.act"]=new_layer
            self.origin_layer_names["features.5.expand.act"]=new_layer
        elif 'features.5.expand.act.act' == layer_name:
            self.features[5].expand.act.act= new_layer
            self.layer_names["features.5.expand.act.act"]=new_layer
            self.origin_layer_names["features.5.expand.act.act"]=new_layer
        elif 'features.5.conv1' == layer_name:
            self.features[5].conv1= new_layer
            self.layer_names["features.5.conv1"]=new_layer
            self.origin_layer_names["features.5.conv1"]=new_layer
        elif 'features.5.conv1.conv' == layer_name:
            self.features[5].conv1.conv= new_layer
            self.layer_names["features.5.conv1.conv"]=new_layer
            self.origin_layer_names["features.5.conv1.conv"]=new_layer
        elif 'features.5.conv1.bn' == layer_name:
            self.features[5].conv1.bn= new_layer
            self.layer_names["features.5.conv1.bn"]=new_layer
            self.origin_layer_names["features.5.conv1.bn"]=new_layer
        elif 'features.5.conv1.act' == layer_name:
            self.features[5].conv1.act= new_layer
            self.layer_names["features.5.conv1.act"]=new_layer
            self.origin_layer_names["features.5.conv1.act"]=new_layer
        elif 'features.5.conv1.act.act' == layer_name:
            self.features[5].conv1.act.act= new_layer
            self.layer_names["features.5.conv1.act.act"]=new_layer
            self.origin_layer_names["features.5.conv1.act.act"]=new_layer
        elif 'features.5.conv2' == layer_name:
            self.features[5].conv2= new_layer
            self.layer_names["features.5.conv2"]=new_layer
            self.origin_layer_names["features.5.conv2"]=new_layer
        elif 'features.5.conv2.conv' == layer_name:
            self.features[5].conv2.conv= new_layer
            self.layer_names["features.5.conv2.conv"]=new_layer
            self.origin_layer_names["features.5.conv2.conv"]=new_layer
        elif 'features.5.conv2.bn' == layer_name:
            self.features[5].conv2.bn= new_layer
            self.layer_names["features.5.conv2.bn"]=new_layer
            self.origin_layer_names["features.5.conv2.bn"]=new_layer
        elif 'features.6' == layer_name:
            self.features[6]= new_layer
            self.layer_names["features.6"]=new_layer
            self.origin_layer_names["features.6"]=new_layer
        elif 'features.6.expand' == layer_name:
            self.features[6].expand= new_layer
            self.layer_names["features.6.expand"]=new_layer
            self.origin_layer_names["features.6.expand"]=new_layer
        elif 'features.6.expand.conv' == layer_name:
            self.features[6].expand.conv= new_layer
            self.layer_names["features.6.expand.conv"]=new_layer
            self.origin_layer_names["features.6.expand.conv"]=new_layer
        elif 'features.6.expand.bn' == layer_name:
            self.features[6].expand.bn= new_layer
            self.layer_names["features.6.expand.bn"]=new_layer
            self.origin_layer_names["features.6.expand.bn"]=new_layer
        elif 'features.6.expand.act' == layer_name:
            self.features[6].expand.act= new_layer
            self.layer_names["features.6.expand.act"]=new_layer
            self.origin_layer_names["features.6.expand.act"]=new_layer
        elif 'features.6.expand.act.act' == layer_name:
            self.features[6].expand.act.act= new_layer
            self.layer_names["features.6.expand.act.act"]=new_layer
            self.origin_layer_names["features.6.expand.act.act"]=new_layer
        elif 'features.6.conv1' == layer_name:
            self.features[6].conv1= new_layer
            self.layer_names["features.6.conv1"]=new_layer
            self.origin_layer_names["features.6.conv1"]=new_layer
        elif 'features.6.conv1.conv' == layer_name:
            self.features[6].conv1.conv= new_layer
            self.layer_names["features.6.conv1.conv"]=new_layer
            self.origin_layer_names["features.6.conv1.conv"]=new_layer
        elif 'features.6.conv1.bn' == layer_name:
            self.features[6].conv1.bn= new_layer
            self.layer_names["features.6.conv1.bn"]=new_layer
            self.origin_layer_names["features.6.conv1.bn"]=new_layer
        elif 'features.6.conv1.act' == layer_name:
            self.features[6].conv1.act= new_layer
            self.layer_names["features.6.conv1.act"]=new_layer
            self.origin_layer_names["features.6.conv1.act"]=new_layer
        elif 'features.6.conv1.act.act' == layer_name:
            self.features[6].conv1.act.act= new_layer
            self.layer_names["features.6.conv1.act.act"]=new_layer
            self.origin_layer_names["features.6.conv1.act.act"]=new_layer
        elif 'features.6.se' == layer_name:
            self.features[6].se= new_layer
            self.layer_names["features.6.se"]=new_layer
            self.origin_layer_names["features.6.se"]=new_layer
        elif 'features.6.se.pool' == layer_name:
            self.features[6].se.pool= new_layer
            self.layer_names["features.6.se.pool"]=new_layer
            self.origin_layer_names["features.6.se.pool"]=new_layer
        elif 'features.6.se.conv1' == layer_name:
            self.features[6].se.conv1= new_layer
            self.layer_names["features.6.se.conv1"]=new_layer
            self.origin_layer_names["features.6.se.conv1"]=new_layer
        elif 'features.6.se.act1' == layer_name:
            self.features[6].se.act1= new_layer
            self.layer_names["features.6.se.act1"]=new_layer
            self.origin_layer_names["features.6.se.act1"]=new_layer
        elif 'features.6.se.act1.act' == layer_name:
            self.features[6].se.act1.act= new_layer
            self.layer_names["features.6.se.act1.act"]=new_layer
            self.origin_layer_names["features.6.se.act1.act"]=new_layer
        elif 'features.6.se.conv2' == layer_name:
            self.features[6].se.conv2= new_layer
            self.layer_names["features.6.se.conv2"]=new_layer
            self.origin_layer_names["features.6.se.conv2"]=new_layer
        elif 'features.6.se.act2' == layer_name:
            self.features[6].se.act2= new_layer
            self.layer_names["features.6.se.act2"]=new_layer
            self.origin_layer_names["features.6.se.act2"]=new_layer
        elif 'features.6.se.act2.act' == layer_name:
            self.features[6].se.act2.act= new_layer
            self.layer_names["features.6.se.act2.act"]=new_layer
            self.origin_layer_names["features.6.se.act2.act"]=new_layer
        elif 'features.6.conv2' == layer_name:
            self.features[6].conv2= new_layer
            self.layer_names["features.6.conv2"]=new_layer
            self.origin_layer_names["features.6.conv2"]=new_layer
        elif 'features.6.conv2.conv' == layer_name:
            self.features[6].conv2.conv= new_layer
            self.layer_names["features.6.conv2.conv"]=new_layer
            self.origin_layer_names["features.6.conv2.conv"]=new_layer
        elif 'features.6.conv2.bn' == layer_name:
            self.features[6].conv2.bn= new_layer
            self.layer_names["features.6.conv2.bn"]=new_layer
            self.origin_layer_names["features.6.conv2.bn"]=new_layer
        elif 'features.7' == layer_name:
            self.features[7]= new_layer
            self.layer_names["features.7"]=new_layer
            self.origin_layer_names["features.7"]=new_layer
        elif 'features.7.expand' == layer_name:
            self.features[7].expand= new_layer
            self.layer_names["features.7.expand"]=new_layer
            self.origin_layer_names["features.7.expand"]=new_layer
        elif 'features.7.expand.conv' == layer_name:
            self.features[7].expand.conv= new_layer
            self.layer_names["features.7.expand.conv"]=new_layer
            self.origin_layer_names["features.7.expand.conv"]=new_layer
        elif 'features.7.expand.bn' == layer_name:
            self.features[7].expand.bn= new_layer
            self.layer_names["features.7.expand.bn"]=new_layer
            self.origin_layer_names["features.7.expand.bn"]=new_layer
        elif 'features.7.expand.act' == layer_name:
            self.features[7].expand.act= new_layer
            self.layer_names["features.7.expand.act"]=new_layer
            self.origin_layer_names["features.7.expand.act"]=new_layer
        elif 'features.7.expand.act.act' == layer_name:
            self.features[7].expand.act.act= new_layer
            self.layer_names["features.7.expand.act.act"]=new_layer
            self.origin_layer_names["features.7.expand.act.act"]=new_layer
        elif 'features.7.conv1' == layer_name:
            self.features[7].conv1= new_layer
            self.layer_names["features.7.conv1"]=new_layer
            self.origin_layer_names["features.7.conv1"]=new_layer
        elif 'features.7.conv1.conv' == layer_name:
            self.features[7].conv1.conv= new_layer
            self.layer_names["features.7.conv1.conv"]=new_layer
            self.origin_layer_names["features.7.conv1.conv"]=new_layer
        elif 'features.7.conv1.bn' == layer_name:
            self.features[7].conv1.bn= new_layer
            self.layer_names["features.7.conv1.bn"]=new_layer
            self.origin_layer_names["features.7.conv1.bn"]=new_layer
        elif 'features.7.conv1.act' == layer_name:
            self.features[7].conv1.act= new_layer
            self.layer_names["features.7.conv1.act"]=new_layer
            self.origin_layer_names["features.7.conv1.act"]=new_layer
        elif 'features.7.conv1.act.act' == layer_name:
            self.features[7].conv1.act.act= new_layer
            self.layer_names["features.7.conv1.act.act"]=new_layer
            self.origin_layer_names["features.7.conv1.act.act"]=new_layer
        elif 'features.7.se' == layer_name:
            self.features[7].se= new_layer
            self.layer_names["features.7.se"]=new_layer
            self.origin_layer_names["features.7.se"]=new_layer
        elif 'features.7.se.pool' == layer_name:
            self.features[7].se.pool= new_layer
            self.layer_names["features.7.se.pool"]=new_layer
            self.origin_layer_names["features.7.se.pool"]=new_layer
        elif 'features.7.se.conv1' == layer_name:
            self.features[7].se.conv1= new_layer
            self.layer_names["features.7.se.conv1"]=new_layer
            self.origin_layer_names["features.7.se.conv1"]=new_layer
        elif 'features.7.se.act1' == layer_name:
            self.features[7].se.act1= new_layer
            self.layer_names["features.7.se.act1"]=new_layer
            self.origin_layer_names["features.7.se.act1"]=new_layer
        elif 'features.7.se.act1.act' == layer_name:
            self.features[7].se.act1.act= new_layer
            self.layer_names["features.7.se.act1.act"]=new_layer
            self.origin_layer_names["features.7.se.act1.act"]=new_layer
        elif 'features.7.se.conv2' == layer_name:
            self.features[7].se.conv2= new_layer
            self.layer_names["features.7.se.conv2"]=new_layer
            self.origin_layer_names["features.7.se.conv2"]=new_layer
        elif 'features.7.se.act2' == layer_name:
            self.features[7].se.act2= new_layer
            self.layer_names["features.7.se.act2"]=new_layer
            self.origin_layer_names["features.7.se.act2"]=new_layer
        elif 'features.7.se.act2.act' == layer_name:
            self.features[7].se.act2.act= new_layer
            self.layer_names["features.7.se.act2.act"]=new_layer
            self.origin_layer_names["features.7.se.act2.act"]=new_layer
        elif 'features.7.conv2' == layer_name:
            self.features[7].conv2= new_layer
            self.layer_names["features.7.conv2"]=new_layer
            self.origin_layer_names["features.7.conv2"]=new_layer
        elif 'features.7.conv2.conv' == layer_name:
            self.features[7].conv2.conv= new_layer
            self.layer_names["features.7.conv2.conv"]=new_layer
            self.origin_layer_names["features.7.conv2.conv"]=new_layer
        elif 'features.7.conv2.bn' == layer_name:
            self.features[7].conv2.bn= new_layer
            self.layer_names["features.7.conv2.bn"]=new_layer
            self.origin_layer_names["features.7.conv2.bn"]=new_layer
        elif 'features.8' == layer_name:
            self.features[8]= new_layer
            self.layer_names["features.8"]=new_layer
            self.origin_layer_names["features.8"]=new_layer
        elif 'features.8.expand' == layer_name:
            self.features[8].expand= new_layer
            self.layer_names["features.8.expand"]=new_layer
            self.origin_layer_names["features.8.expand"]=new_layer
        elif 'features.8.expand.conv' == layer_name:
            self.features[8].expand.conv= new_layer
            self.layer_names["features.8.expand.conv"]=new_layer
            self.origin_layer_names["features.8.expand.conv"]=new_layer
        elif 'features.8.expand.bn' == layer_name:
            self.features[8].expand.bn= new_layer
            self.layer_names["features.8.expand.bn"]=new_layer
            self.origin_layer_names["features.8.expand.bn"]=new_layer
        elif 'features.8.expand.act' == layer_name:
            self.features[8].expand.act= new_layer
            self.layer_names["features.8.expand.act"]=new_layer
            self.origin_layer_names["features.8.expand.act"]=new_layer
        elif 'features.8.expand.act.act' == layer_name:
            self.features[8].expand.act.act= new_layer
            self.layer_names["features.8.expand.act.act"]=new_layer
            self.origin_layer_names["features.8.expand.act.act"]=new_layer
        elif 'features.8.conv1' == layer_name:
            self.features[8].conv1= new_layer
            self.layer_names["features.8.conv1"]=new_layer
            self.origin_layer_names["features.8.conv1"]=new_layer
        elif 'features.8.conv1.conv' == layer_name:
            self.features[8].conv1.conv= new_layer
            self.layer_names["features.8.conv1.conv"]=new_layer
            self.origin_layer_names["features.8.conv1.conv"]=new_layer
        elif 'features.8.conv1.bn' == layer_name:
            self.features[8].conv1.bn= new_layer
            self.layer_names["features.8.conv1.bn"]=new_layer
            self.origin_layer_names["features.8.conv1.bn"]=new_layer
        elif 'features.8.conv1.act' == layer_name:
            self.features[8].conv1.act= new_layer
            self.layer_names["features.8.conv1.act"]=new_layer
            self.origin_layer_names["features.8.conv1.act"]=new_layer
        elif 'features.8.conv1.act.act' == layer_name:
            self.features[8].conv1.act.act= new_layer
            self.layer_names["features.8.conv1.act.act"]=new_layer
            self.origin_layer_names["features.8.conv1.act.act"]=new_layer
        elif 'features.8.se' == layer_name:
            self.features[8].se= new_layer
            self.layer_names["features.8.se"]=new_layer
            self.origin_layer_names["features.8.se"]=new_layer
        elif 'features.8.se.pool' == layer_name:
            self.features[8].se.pool= new_layer
            self.layer_names["features.8.se.pool"]=new_layer
            self.origin_layer_names["features.8.se.pool"]=new_layer
        elif 'features.8.se.conv1' == layer_name:
            self.features[8].se.conv1= new_layer
            self.layer_names["features.8.se.conv1"]=new_layer
            self.origin_layer_names["features.8.se.conv1"]=new_layer
        elif 'features.8.se.act1' == layer_name:
            self.features[8].se.act1= new_layer
            self.layer_names["features.8.se.act1"]=new_layer
            self.origin_layer_names["features.8.se.act1"]=new_layer
        elif 'features.8.se.act1.act' == layer_name:
            self.features[8].se.act1.act= new_layer
            self.layer_names["features.8.se.act1.act"]=new_layer
            self.origin_layer_names["features.8.se.act1.act"]=new_layer
        elif 'features.8.se.conv2' == layer_name:
            self.features[8].se.conv2= new_layer
            self.layer_names["features.8.se.conv2"]=new_layer
            self.origin_layer_names["features.8.se.conv2"]=new_layer
        elif 'features.8.se.act2' == layer_name:
            self.features[8].se.act2= new_layer
            self.layer_names["features.8.se.act2"]=new_layer
            self.origin_layer_names["features.8.se.act2"]=new_layer
        elif 'features.8.se.act2.act' == layer_name:
            self.features[8].se.act2.act= new_layer
            self.layer_names["features.8.se.act2.act"]=new_layer
            self.origin_layer_names["features.8.se.act2.act"]=new_layer
        elif 'features.8.conv2' == layer_name:
            self.features[8].conv2= new_layer
            self.layer_names["features.8.conv2"]=new_layer
            self.origin_layer_names["features.8.conv2"]=new_layer
        elif 'features.8.conv2.conv' == layer_name:
            self.features[8].conv2.conv= new_layer
            self.layer_names["features.8.conv2.conv"]=new_layer
            self.origin_layer_names["features.8.conv2.conv"]=new_layer
        elif 'features.8.conv2.bn' == layer_name:
            self.features[8].conv2.bn= new_layer
            self.layer_names["features.8.conv2.bn"]=new_layer
            self.origin_layer_names["features.8.conv2.bn"]=new_layer
        elif 'features.9' == layer_name:
            self.features[9]= new_layer
            self.layer_names["features.9"]=new_layer
            self.origin_layer_names["features.9"]=new_layer
        elif 'features.9.expand' == layer_name:
            self.features[9].expand= new_layer
            self.layer_names["features.9.expand"]=new_layer
            self.origin_layer_names["features.9.expand"]=new_layer
        elif 'features.9.expand.conv' == layer_name:
            self.features[9].expand.conv= new_layer
            self.layer_names["features.9.expand.conv"]=new_layer
            self.origin_layer_names["features.9.expand.conv"]=new_layer
        elif 'features.9.expand.bn' == layer_name:
            self.features[9].expand.bn= new_layer
            self.layer_names["features.9.expand.bn"]=new_layer
            self.origin_layer_names["features.9.expand.bn"]=new_layer
        elif 'features.9.expand.act' == layer_name:
            self.features[9].expand.act= new_layer
            self.layer_names["features.9.expand.act"]=new_layer
            self.origin_layer_names["features.9.expand.act"]=new_layer
        elif 'features.9.expand.act.act' == layer_name:
            self.features[9].expand.act.act= new_layer
            self.layer_names["features.9.expand.act.act"]=new_layer
            self.origin_layer_names["features.9.expand.act.act"]=new_layer
        elif 'features.9.conv1' == layer_name:
            self.features[9].conv1= new_layer
            self.layer_names["features.9.conv1"]=new_layer
            self.origin_layer_names["features.9.conv1"]=new_layer
        elif 'features.9.conv1.conv' == layer_name:
            self.features[9].conv1.conv= new_layer
            self.layer_names["features.9.conv1.conv"]=new_layer
            self.origin_layer_names["features.9.conv1.conv"]=new_layer
        elif 'features.9.conv1.bn' == layer_name:
            self.features[9].conv1.bn= new_layer
            self.layer_names["features.9.conv1.bn"]=new_layer
            self.origin_layer_names["features.9.conv1.bn"]=new_layer
        elif 'features.9.conv1.act' == layer_name:
            self.features[9].conv1.act= new_layer
            self.layer_names["features.9.conv1.act"]=new_layer
            self.origin_layer_names["features.9.conv1.act"]=new_layer
        elif 'features.9.conv1.act.act' == layer_name:
            self.features[9].conv1.act.act= new_layer
            self.layer_names["features.9.conv1.act.act"]=new_layer
            self.origin_layer_names["features.9.conv1.act.act"]=new_layer
        elif 'features.9.conv2' == layer_name:
            self.features[9].conv2= new_layer
            self.layer_names["features.9.conv2"]=new_layer
            self.origin_layer_names["features.9.conv2"]=new_layer
        elif 'features.9.conv2.conv' == layer_name:
            self.features[9].conv2.conv= new_layer
            self.layer_names["features.9.conv2.conv"]=new_layer
            self.origin_layer_names["features.9.conv2.conv"]=new_layer
        elif 'features.9.conv2.bn' == layer_name:
            self.features[9].conv2.bn= new_layer
            self.layer_names["features.9.conv2.bn"]=new_layer
            self.origin_layer_names["features.9.conv2.bn"]=new_layer
        elif 'features.10' == layer_name:
            self.features[10]= new_layer
            self.layer_names["features.10"]=new_layer
            self.origin_layer_names["features.10"]=new_layer
        elif 'features.10.expand' == layer_name:
            self.features[10].expand= new_layer
            self.layer_names["features.10.expand"]=new_layer
            self.origin_layer_names["features.10.expand"]=new_layer
        elif 'features.10.expand.conv' == layer_name:
            self.features[10].expand.conv= new_layer
            self.layer_names["features.10.expand.conv"]=new_layer
            self.origin_layer_names["features.10.expand.conv"]=new_layer
        elif 'features.10.expand.bn' == layer_name:
            self.features[10].expand.bn= new_layer
            self.layer_names["features.10.expand.bn"]=new_layer
            self.origin_layer_names["features.10.expand.bn"]=new_layer
        elif 'features.10.expand.act' == layer_name:
            self.features[10].expand.act= new_layer
            self.layer_names["features.10.expand.act"]=new_layer
            self.origin_layer_names["features.10.expand.act"]=new_layer
        elif 'features.10.expand.act.act' == layer_name:
            self.features[10].expand.act.act= new_layer
            self.layer_names["features.10.expand.act.act"]=new_layer
            self.origin_layer_names["features.10.expand.act.act"]=new_layer
        elif 'features.10.conv1' == layer_name:
            self.features[10].conv1= new_layer
            self.layer_names["features.10.conv1"]=new_layer
            self.origin_layer_names["features.10.conv1"]=new_layer
        elif 'features.10.conv1.conv' == layer_name:
            self.features[10].conv1.conv= new_layer
            self.layer_names["features.10.conv1.conv"]=new_layer
            self.origin_layer_names["features.10.conv1.conv"]=new_layer
        elif 'features.10.conv1.bn' == layer_name:
            self.features[10].conv1.bn= new_layer
            self.layer_names["features.10.conv1.bn"]=new_layer
            self.origin_layer_names["features.10.conv1.bn"]=new_layer
        elif 'features.10.conv1.act' == layer_name:
            self.features[10].conv1.act= new_layer
            self.layer_names["features.10.conv1.act"]=new_layer
            self.origin_layer_names["features.10.conv1.act"]=new_layer
        elif 'features.10.conv1.act.act' == layer_name:
            self.features[10].conv1.act.act= new_layer
            self.layer_names["features.10.conv1.act.act"]=new_layer
            self.origin_layer_names["features.10.conv1.act.act"]=new_layer
        elif 'features.10.conv2' == layer_name:
            self.features[10].conv2= new_layer
            self.layer_names["features.10.conv2"]=new_layer
            self.origin_layer_names["features.10.conv2"]=new_layer
        elif 'features.10.conv2.conv' == layer_name:
            self.features[10].conv2.conv= new_layer
            self.layer_names["features.10.conv2.conv"]=new_layer
            self.origin_layer_names["features.10.conv2.conv"]=new_layer
        elif 'features.10.conv2.bn' == layer_name:
            self.features[10].conv2.bn= new_layer
            self.layer_names["features.10.conv2.bn"]=new_layer
            self.origin_layer_names["features.10.conv2.bn"]=new_layer
        elif 'features.11' == layer_name:
            self.features[11]= new_layer
            self.layer_names["features.11"]=new_layer
            self.origin_layer_names["features.11"]=new_layer
        elif 'features.11.expand' == layer_name:
            self.features[11].expand= new_layer
            self.layer_names["features.11.expand"]=new_layer
            self.origin_layer_names["features.11.expand"]=new_layer
        elif 'features.11.expand.conv' == layer_name:
            self.features[11].expand.conv= new_layer
            self.layer_names["features.11.expand.conv"]=new_layer
            self.origin_layer_names["features.11.expand.conv"]=new_layer
        elif 'features.11.expand.bn' == layer_name:
            self.features[11].expand.bn= new_layer
            self.layer_names["features.11.expand.bn"]=new_layer
            self.origin_layer_names["features.11.expand.bn"]=new_layer
        elif 'features.11.expand.act' == layer_name:
            self.features[11].expand.act= new_layer
            self.layer_names["features.11.expand.act"]=new_layer
            self.origin_layer_names["features.11.expand.act"]=new_layer
        elif 'features.11.expand.act.act' == layer_name:
            self.features[11].expand.act.act= new_layer
            self.layer_names["features.11.expand.act.act"]=new_layer
            self.origin_layer_names["features.11.expand.act.act"]=new_layer
        elif 'features.11.conv1' == layer_name:
            self.features[11].conv1= new_layer
            self.layer_names["features.11.conv1"]=new_layer
            self.origin_layer_names["features.11.conv1"]=new_layer
        elif 'features.11.conv1.conv' == layer_name:
            self.features[11].conv1.conv= new_layer
            self.layer_names["features.11.conv1.conv"]=new_layer
            self.origin_layer_names["features.11.conv1.conv"]=new_layer
        elif 'features.11.conv1.bn' == layer_name:
            self.features[11].conv1.bn= new_layer
            self.layer_names["features.11.conv1.bn"]=new_layer
            self.origin_layer_names["features.11.conv1.bn"]=new_layer
        elif 'features.11.conv1.act' == layer_name:
            self.features[11].conv1.act= new_layer
            self.layer_names["features.11.conv1.act"]=new_layer
            self.origin_layer_names["features.11.conv1.act"]=new_layer
        elif 'features.11.conv1.act.act' == layer_name:
            self.features[11].conv1.act.act= new_layer
            self.layer_names["features.11.conv1.act.act"]=new_layer
            self.origin_layer_names["features.11.conv1.act.act"]=new_layer
        elif 'features.11.conv2' == layer_name:
            self.features[11].conv2= new_layer
            self.layer_names["features.11.conv2"]=new_layer
            self.origin_layer_names["features.11.conv2"]=new_layer
        elif 'features.11.conv2.conv' == layer_name:
            self.features[11].conv2.conv= new_layer
            self.layer_names["features.11.conv2.conv"]=new_layer
            self.origin_layer_names["features.11.conv2.conv"]=new_layer
        elif 'features.11.conv2.bn' == layer_name:
            self.features[11].conv2.bn= new_layer
            self.layer_names["features.11.conv2.bn"]=new_layer
            self.origin_layer_names["features.11.conv2.bn"]=new_layer
        elif 'features.12' == layer_name:
            self.features[12]= new_layer
            self.layer_names["features.12"]=new_layer
            self.origin_layer_names["features.12"]=new_layer
        elif 'features.12.expand' == layer_name:
            self.features[12].expand= new_layer
            self.layer_names["features.12.expand"]=new_layer
            self.origin_layer_names["features.12.expand"]=new_layer
        elif 'features.12.expand.conv' == layer_name:
            self.features[12].expand.conv= new_layer
            self.layer_names["features.12.expand.conv"]=new_layer
            self.origin_layer_names["features.12.expand.conv"]=new_layer
        elif 'features.12.expand.bn' == layer_name:
            self.features[12].expand.bn= new_layer
            self.layer_names["features.12.expand.bn"]=new_layer
            self.origin_layer_names["features.12.expand.bn"]=new_layer
        elif 'features.12.expand.act' == layer_name:
            self.features[12].expand.act= new_layer
            self.layer_names["features.12.expand.act"]=new_layer
            self.origin_layer_names["features.12.expand.act"]=new_layer
        elif 'features.12.expand.act.act' == layer_name:
            self.features[12].expand.act.act= new_layer
            self.layer_names["features.12.expand.act.act"]=new_layer
            self.origin_layer_names["features.12.expand.act.act"]=new_layer
        elif 'features.12.conv1' == layer_name:
            self.features[12].conv1= new_layer
            self.layer_names["features.12.conv1"]=new_layer
            self.origin_layer_names["features.12.conv1"]=new_layer
        elif 'features.12.conv1.conv' == layer_name:
            self.features[12].conv1.conv= new_layer
            self.layer_names["features.12.conv1.conv"]=new_layer
            self.origin_layer_names["features.12.conv1.conv"]=new_layer
        elif 'features.12.conv1.bn' == layer_name:
            self.features[12].conv1.bn= new_layer
            self.layer_names["features.12.conv1.bn"]=new_layer
            self.origin_layer_names["features.12.conv1.bn"]=new_layer
        elif 'features.12.conv1.act' == layer_name:
            self.features[12].conv1.act= new_layer
            self.layer_names["features.12.conv1.act"]=new_layer
            self.origin_layer_names["features.12.conv1.act"]=new_layer
        elif 'features.12.conv1.act.act' == layer_name:
            self.features[12].conv1.act.act= new_layer
            self.layer_names["features.12.conv1.act.act"]=new_layer
            self.origin_layer_names["features.12.conv1.act.act"]=new_layer
        elif 'features.12.conv2' == layer_name:
            self.features[12].conv2= new_layer
            self.layer_names["features.12.conv2"]=new_layer
            self.origin_layer_names["features.12.conv2"]=new_layer
        elif 'features.12.conv2.conv' == layer_name:
            self.features[12].conv2.conv= new_layer
            self.layer_names["features.12.conv2.conv"]=new_layer
            self.origin_layer_names["features.12.conv2.conv"]=new_layer
        elif 'features.12.conv2.bn' == layer_name:
            self.features[12].conv2.bn= new_layer
            self.layer_names["features.12.conv2.bn"]=new_layer
            self.origin_layer_names["features.12.conv2.bn"]=new_layer
        elif 'features.13' == layer_name:
            self.features[13]= new_layer
            self.layer_names["features.13"]=new_layer
            self.origin_layer_names["features.13"]=new_layer
        elif 'features.13.expand' == layer_name:
            self.features[13].expand= new_layer
            self.layer_names["features.13.expand"]=new_layer
            self.origin_layer_names["features.13.expand"]=new_layer
        elif 'features.13.expand.conv' == layer_name:
            self.features[13].expand.conv= new_layer
            self.layer_names["features.13.expand.conv"]=new_layer
            self.origin_layer_names["features.13.expand.conv"]=new_layer
        elif 'features.13.expand.bn' == layer_name:
            self.features[13].expand.bn= new_layer
            self.layer_names["features.13.expand.bn"]=new_layer
            self.origin_layer_names["features.13.expand.bn"]=new_layer
        elif 'features.13.expand.act' == layer_name:
            self.features[13].expand.act= new_layer
            self.layer_names["features.13.expand.act"]=new_layer
            self.origin_layer_names["features.13.expand.act"]=new_layer
        elif 'features.13.expand.act.act' == layer_name:
            self.features[13].expand.act.act= new_layer
            self.layer_names["features.13.expand.act.act"]=new_layer
            self.origin_layer_names["features.13.expand.act.act"]=new_layer
        elif 'features.13.conv1' == layer_name:
            self.features[13].conv1= new_layer
            self.layer_names["features.13.conv1"]=new_layer
            self.origin_layer_names["features.13.conv1"]=new_layer
        elif 'features.13.conv1.conv' == layer_name:
            self.features[13].conv1.conv= new_layer
            self.layer_names["features.13.conv1.conv"]=new_layer
            self.origin_layer_names["features.13.conv1.conv"]=new_layer
        elif 'features.13.conv1.bn' == layer_name:
            self.features[13].conv1.bn= new_layer
            self.layer_names["features.13.conv1.bn"]=new_layer
            self.origin_layer_names["features.13.conv1.bn"]=new_layer
        elif 'features.13.conv1.act' == layer_name:
            self.features[13].conv1.act= new_layer
            self.layer_names["features.13.conv1.act"]=new_layer
            self.origin_layer_names["features.13.conv1.act"]=new_layer
        elif 'features.13.conv1.act.act' == layer_name:
            self.features[13].conv1.act.act= new_layer
            self.layer_names["features.13.conv1.act.act"]=new_layer
            self.origin_layer_names["features.13.conv1.act.act"]=new_layer
        elif 'features.13.se' == layer_name:
            self.features[13].se= new_layer
            self.layer_names["features.13.se"]=new_layer
            self.origin_layer_names["features.13.se"]=new_layer
        elif 'features.13.se.pool' == layer_name:
            self.features[13].se.pool= new_layer
            self.layer_names["features.13.se.pool"]=new_layer
            self.origin_layer_names["features.13.se.pool"]=new_layer
        elif 'features.13.se.conv1' == layer_name:
            self.features[13].se.conv1= new_layer
            self.layer_names["features.13.se.conv1"]=new_layer
            self.origin_layer_names["features.13.se.conv1"]=new_layer
        elif 'features.13.se.act1' == layer_name:
            self.features[13].se.act1= new_layer
            self.layer_names["features.13.se.act1"]=new_layer
            self.origin_layer_names["features.13.se.act1"]=new_layer
        elif 'features.13.se.act1.act' == layer_name:
            self.features[13].se.act1.act= new_layer
            self.layer_names["features.13.se.act1.act"]=new_layer
            self.origin_layer_names["features.13.se.act1.act"]=new_layer
        elif 'features.13.se.conv2' == layer_name:
            self.features[13].se.conv2= new_layer
            self.layer_names["features.13.se.conv2"]=new_layer
            self.origin_layer_names["features.13.se.conv2"]=new_layer
        elif 'features.13.se.act2' == layer_name:
            self.features[13].se.act2= new_layer
            self.layer_names["features.13.se.act2"]=new_layer
            self.origin_layer_names["features.13.se.act2"]=new_layer
        elif 'features.13.se.act2.act' == layer_name:
            self.features[13].se.act2.act= new_layer
            self.layer_names["features.13.se.act2.act"]=new_layer
            self.origin_layer_names["features.13.se.act2.act"]=new_layer
        elif 'features.13.conv2' == layer_name:
            self.features[13].conv2= new_layer
            self.layer_names["features.13.conv2"]=new_layer
            self.origin_layer_names["features.13.conv2"]=new_layer
        elif 'features.13.conv2.conv' == layer_name:
            self.features[13].conv2.conv= new_layer
            self.layer_names["features.13.conv2.conv"]=new_layer
            self.origin_layer_names["features.13.conv2.conv"]=new_layer
        elif 'features.13.conv2.bn' == layer_name:
            self.features[13].conv2.bn= new_layer
            self.layer_names["features.13.conv2.bn"]=new_layer
            self.origin_layer_names["features.13.conv2.bn"]=new_layer
        elif 'features.14' == layer_name:
            self.features[14]= new_layer
            self.layer_names["features.14"]=new_layer
            self.origin_layer_names["features.14"]=new_layer
        elif 'features.14.expand' == layer_name:
            self.features[14].expand= new_layer
            self.layer_names["features.14.expand"]=new_layer
            self.origin_layer_names["features.14.expand"]=new_layer
        elif 'features.14.expand.conv' == layer_name:
            self.features[14].expand.conv= new_layer
            self.layer_names["features.14.expand.conv"]=new_layer
            self.origin_layer_names["features.14.expand.conv"]=new_layer
        elif 'features.14.expand.bn' == layer_name:
            self.features[14].expand.bn= new_layer
            self.layer_names["features.14.expand.bn"]=new_layer
            self.origin_layer_names["features.14.expand.bn"]=new_layer
        elif 'features.14.expand.act' == layer_name:
            self.features[14].expand.act= new_layer
            self.layer_names["features.14.expand.act"]=new_layer
            self.origin_layer_names["features.14.expand.act"]=new_layer
        elif 'features.14.expand.act.act' == layer_name:
            self.features[14].expand.act.act= new_layer
            self.layer_names["features.14.expand.act.act"]=new_layer
            self.origin_layer_names["features.14.expand.act.act"]=new_layer
        elif 'features.14.conv1' == layer_name:
            self.features[14].conv1= new_layer
            self.layer_names["features.14.conv1"]=new_layer
            self.origin_layer_names["features.14.conv1"]=new_layer
        elif 'features.14.conv1.conv' == layer_name:
            self.features[14].conv1.conv= new_layer
            self.layer_names["features.14.conv1.conv"]=new_layer
            self.origin_layer_names["features.14.conv1.conv"]=new_layer
        elif 'features.14.conv1.bn' == layer_name:
            self.features[14].conv1.bn= new_layer
            self.layer_names["features.14.conv1.bn"]=new_layer
            self.origin_layer_names["features.14.conv1.bn"]=new_layer
        elif 'features.14.conv1.act' == layer_name:
            self.features[14].conv1.act= new_layer
            self.layer_names["features.14.conv1.act"]=new_layer
            self.origin_layer_names["features.14.conv1.act"]=new_layer
        elif 'features.14.conv1.act.act' == layer_name:
            self.features[14].conv1.act.act= new_layer
            self.layer_names["features.14.conv1.act.act"]=new_layer
            self.origin_layer_names["features.14.conv1.act.act"]=new_layer
        elif 'features.14.se' == layer_name:
            self.features[14].se= new_layer
            self.layer_names["features.14.se"]=new_layer
            self.origin_layer_names["features.14.se"]=new_layer
        elif 'features.14.se.pool' == layer_name:
            self.features[14].se.pool= new_layer
            self.layer_names["features.14.se.pool"]=new_layer
            self.origin_layer_names["features.14.se.pool"]=new_layer
        elif 'features.14.se.conv1' == layer_name:
            self.features[14].se.conv1= new_layer
            self.layer_names["features.14.se.conv1"]=new_layer
            self.origin_layer_names["features.14.se.conv1"]=new_layer
        elif 'features.14.se.act1' == layer_name:
            self.features[14].se.act1= new_layer
            self.layer_names["features.14.se.act1"]=new_layer
            self.origin_layer_names["features.14.se.act1"]=new_layer
        elif 'features.14.se.act1.act' == layer_name:
            self.features[14].se.act1.act= new_layer
            self.layer_names["features.14.se.act1.act"]=new_layer
            self.origin_layer_names["features.14.se.act1.act"]=new_layer
        elif 'features.14.se.conv2' == layer_name:
            self.features[14].se.conv2= new_layer
            self.layer_names["features.14.se.conv2"]=new_layer
            self.origin_layer_names["features.14.se.conv2"]=new_layer
        elif 'features.14.se.act2' == layer_name:
            self.features[14].se.act2= new_layer
            self.layer_names["features.14.se.act2"]=new_layer
            self.origin_layer_names["features.14.se.act2"]=new_layer
        elif 'features.14.se.act2.act' == layer_name:
            self.features[14].se.act2.act= new_layer
            self.layer_names["features.14.se.act2.act"]=new_layer
            self.origin_layer_names["features.14.se.act2.act"]=new_layer
        elif 'features.14.conv2' == layer_name:
            self.features[14].conv2= new_layer
            self.layer_names["features.14.conv2"]=new_layer
            self.origin_layer_names["features.14.conv2"]=new_layer
        elif 'features.14.conv2.conv' == layer_name:
            self.features[14].conv2.conv= new_layer
            self.layer_names["features.14.conv2.conv"]=new_layer
            self.origin_layer_names["features.14.conv2.conv"]=new_layer
        elif 'features.14.conv2.bn' == layer_name:
            self.features[14].conv2.bn= new_layer
            self.layer_names["features.14.conv2.bn"]=new_layer
            self.origin_layer_names["features.14.conv2.bn"]=new_layer
        elif 'features.15' == layer_name:
            self.features[15]= new_layer
            self.layer_names["features.15"]=new_layer
            self.origin_layer_names["features.15"]=new_layer
        elif 'features.15.expand' == layer_name:
            self.features[15].expand= new_layer
            self.layer_names["features.15.expand"]=new_layer
            self.origin_layer_names["features.15.expand"]=new_layer
        elif 'features.15.expand.conv' == layer_name:
            self.features[15].expand.conv= new_layer
            self.layer_names["features.15.expand.conv"]=new_layer
            self.origin_layer_names["features.15.expand.conv"]=new_layer
        elif 'features.15.expand.bn' == layer_name:
            self.features[15].expand.bn= new_layer
            self.layer_names["features.15.expand.bn"]=new_layer
            self.origin_layer_names["features.15.expand.bn"]=new_layer
        elif 'features.15.expand.act' == layer_name:
            self.features[15].expand.act= new_layer
            self.layer_names["features.15.expand.act"]=new_layer
            self.origin_layer_names["features.15.expand.act"]=new_layer
        elif 'features.15.expand.act.act' == layer_name:
            self.features[15].expand.act.act= new_layer
            self.layer_names["features.15.expand.act.act"]=new_layer
            self.origin_layer_names["features.15.expand.act.act"]=new_layer
        elif 'features.15.conv1' == layer_name:
            self.features[15].conv1= new_layer
            self.layer_names["features.15.conv1"]=new_layer
            self.origin_layer_names["features.15.conv1"]=new_layer
        elif 'features.15.conv1.conv' == layer_name:
            self.features[15].conv1.conv= new_layer
            self.layer_names["features.15.conv1.conv"]=new_layer
            self.origin_layer_names["features.15.conv1.conv"]=new_layer
        elif 'features.15.conv1.bn' == layer_name:
            self.features[15].conv1.bn= new_layer
            self.layer_names["features.15.conv1.bn"]=new_layer
            self.origin_layer_names["features.15.conv1.bn"]=new_layer
        elif 'features.15.conv1.act' == layer_name:
            self.features[15].conv1.act= new_layer
            self.layer_names["features.15.conv1.act"]=new_layer
            self.origin_layer_names["features.15.conv1.act"]=new_layer
        elif 'features.15.conv1.act.act' == layer_name:
            self.features[15].conv1.act.act= new_layer
            self.layer_names["features.15.conv1.act.act"]=new_layer
            self.origin_layer_names["features.15.conv1.act.act"]=new_layer
        elif 'features.15.se' == layer_name:
            self.features[15].se= new_layer
            self.layer_names["features.15.se"]=new_layer
            self.origin_layer_names["features.15.se"]=new_layer
        elif 'features.15.se.pool' == layer_name:
            self.features[15].se.pool= new_layer
            self.layer_names["features.15.se.pool"]=new_layer
            self.origin_layer_names["features.15.se.pool"]=new_layer
        elif 'features.15.se.conv1' == layer_name:
            self.features[15].se.conv1= new_layer
            self.layer_names["features.15.se.conv1"]=new_layer
            self.origin_layer_names["features.15.se.conv1"]=new_layer
        elif 'features.15.se.act1' == layer_name:
            self.features[15].se.act1= new_layer
            self.layer_names["features.15.se.act1"]=new_layer
            self.origin_layer_names["features.15.se.act1"]=new_layer
        elif 'features.15.se.act1.act' == layer_name:
            self.features[15].se.act1.act= new_layer
            self.layer_names["features.15.se.act1.act"]=new_layer
            self.origin_layer_names["features.15.se.act1.act"]=new_layer
        elif 'features.15.se.conv2' == layer_name:
            self.features[15].se.conv2= new_layer
            self.layer_names["features.15.se.conv2"]=new_layer
            self.origin_layer_names["features.15.se.conv2"]=new_layer
        elif 'features.15.se.act2' == layer_name:
            self.features[15].se.act2= new_layer
            self.layer_names["features.15.se.act2"]=new_layer
            self.origin_layer_names["features.15.se.act2"]=new_layer
        elif 'features.15.se.act2.act' == layer_name:
            self.features[15].se.act2.act= new_layer
            self.layer_names["features.15.se.act2.act"]=new_layer
            self.origin_layer_names["features.15.se.act2.act"]=new_layer
        elif 'features.15.conv2' == layer_name:
            self.features[15].conv2= new_layer
            self.layer_names["features.15.conv2"]=new_layer
            self.origin_layer_names["features.15.conv2"]=new_layer
        elif 'features.15.conv2.conv' == layer_name:
            self.features[15].conv2.conv= new_layer
            self.layer_names["features.15.conv2.conv"]=new_layer
            self.origin_layer_names["features.15.conv2.conv"]=new_layer
        elif 'features.15.conv2.bn' == layer_name:
            self.features[15].conv2.bn= new_layer
            self.layer_names["features.15.conv2.bn"]=new_layer
            self.origin_layer_names["features.15.conv2.bn"]=new_layer
        elif 'features.16' == layer_name:
            self.features[16]= new_layer
            self.layer_names["features.16"]=new_layer
            self.origin_layer_names["features.16"]=new_layer
        elif 'features.16.expand' == layer_name:
            self.features[16].expand= new_layer
            self.layer_names["features.16.expand"]=new_layer
            self.origin_layer_names["features.16.expand"]=new_layer
        elif 'features.16.expand.conv' == layer_name:
            self.features[16].expand.conv= new_layer
            self.layer_names["features.16.expand.conv"]=new_layer
            self.origin_layer_names["features.16.expand.conv"]=new_layer
        elif 'features.16.expand.bn' == layer_name:
            self.features[16].expand.bn= new_layer
            self.layer_names["features.16.expand.bn"]=new_layer
            self.origin_layer_names["features.16.expand.bn"]=new_layer
        elif 'features.16.expand.act' == layer_name:
            self.features[16].expand.act= new_layer
            self.layer_names["features.16.expand.act"]=new_layer
            self.origin_layer_names["features.16.expand.act"]=new_layer
        elif 'features.16.expand.act.act' == layer_name:
            self.features[16].expand.act.act= new_layer
            self.layer_names["features.16.expand.act.act"]=new_layer
            self.origin_layer_names["features.16.expand.act.act"]=new_layer
        elif 'features.16.conv1' == layer_name:
            self.features[16].conv1= new_layer
            self.layer_names["features.16.conv1"]=new_layer
            self.origin_layer_names["features.16.conv1"]=new_layer
        elif 'features.16.conv1.conv' == layer_name:
            self.features[16].conv1.conv= new_layer
            self.layer_names["features.16.conv1.conv"]=new_layer
            self.origin_layer_names["features.16.conv1.conv"]=new_layer
        elif 'features.16.conv1.bn' == layer_name:
            self.features[16].conv1.bn= new_layer
            self.layer_names["features.16.conv1.bn"]=new_layer
            self.origin_layer_names["features.16.conv1.bn"]=new_layer
        elif 'features.16.conv1.act' == layer_name:
            self.features[16].conv1.act= new_layer
            self.layer_names["features.16.conv1.act"]=new_layer
            self.origin_layer_names["features.16.conv1.act"]=new_layer
        elif 'features.16.conv1.act.act' == layer_name:
            self.features[16].conv1.act.act= new_layer
            self.layer_names["features.16.conv1.act.act"]=new_layer
            self.origin_layer_names["features.16.conv1.act.act"]=new_layer
        elif 'features.16.se' == layer_name:
            self.features[16].se= new_layer
            self.layer_names["features.16.se"]=new_layer
            self.origin_layer_names["features.16.se"]=new_layer
        elif 'features.16.se.pool' == layer_name:
            self.features[16].se.pool= new_layer
            self.layer_names["features.16.se.pool"]=new_layer
            self.origin_layer_names["features.16.se.pool"]=new_layer
        elif 'features.16.se.conv1' == layer_name:
            self.features[16].se.conv1= new_layer
            self.layer_names["features.16.se.conv1"]=new_layer
            self.origin_layer_names["features.16.se.conv1"]=new_layer
        elif 'features.16.se.act1' == layer_name:
            self.features[16].se.act1= new_layer
            self.layer_names["features.16.se.act1"]=new_layer
            self.origin_layer_names["features.16.se.act1"]=new_layer
        elif 'features.16.se.act1.act' == layer_name:
            self.features[16].se.act1.act= new_layer
            self.layer_names["features.16.se.act1.act"]=new_layer
            self.origin_layer_names["features.16.se.act1.act"]=new_layer
        elif 'features.16.se.conv2' == layer_name:
            self.features[16].se.conv2= new_layer
            self.layer_names["features.16.se.conv2"]=new_layer
            self.origin_layer_names["features.16.se.conv2"]=new_layer
        elif 'features.16.se.act2' == layer_name:
            self.features[16].se.act2= new_layer
            self.layer_names["features.16.se.act2"]=new_layer
            self.origin_layer_names["features.16.se.act2"]=new_layer
        elif 'features.16.se.act2.act' == layer_name:
            self.features[16].se.act2.act= new_layer
            self.layer_names["features.16.se.act2.act"]=new_layer
            self.origin_layer_names["features.16.se.act2.act"]=new_layer
        elif 'features.16.conv2' == layer_name:
            self.features[16].conv2= new_layer
            self.layer_names["features.16.conv2"]=new_layer
            self.origin_layer_names["features.16.conv2"]=new_layer
        elif 'features.16.conv2.conv' == layer_name:
            self.features[16].conv2.conv= new_layer
            self.layer_names["features.16.conv2.conv"]=new_layer
            self.origin_layer_names["features.16.conv2.conv"]=new_layer
        elif 'features.16.conv2.bn' == layer_name:
            self.features[16].conv2.bn= new_layer
            self.layer_names["features.16.conv2.bn"]=new_layer
            self.origin_layer_names["features.16.conv2.bn"]=new_layer
        elif 'features.17' == layer_name:
            self.features[17]= new_layer
            self.layer_names["features.17"]=new_layer
            self.origin_layer_names["features.17"]=new_layer
        elif 'features.17.expand' == layer_name:
            self.features[17].expand= new_layer
            self.layer_names["features.17.expand"]=new_layer
            self.origin_layer_names["features.17.expand"]=new_layer
        elif 'features.17.expand.conv' == layer_name:
            self.features[17].expand.conv= new_layer
            self.layer_names["features.17.expand.conv"]=new_layer
            self.origin_layer_names["features.17.expand.conv"]=new_layer
        elif 'features.17.expand.bn' == layer_name:
            self.features[17].expand.bn= new_layer
            self.layer_names["features.17.expand.bn"]=new_layer
            self.origin_layer_names["features.17.expand.bn"]=new_layer
        elif 'features.17.expand.act' == layer_name:
            self.features[17].expand.act= new_layer
            self.layer_names["features.17.expand.act"]=new_layer
            self.origin_layer_names["features.17.expand.act"]=new_layer
        elif 'features.17.expand.act.act' == layer_name:
            self.features[17].expand.act.act= new_layer
            self.layer_names["features.17.expand.act.act"]=new_layer
            self.origin_layer_names["features.17.expand.act.act"]=new_layer
        elif 'features.17.conv1' == layer_name:
            self.features[17].conv1= new_layer
            self.layer_names["features.17.conv1"]=new_layer
            self.origin_layer_names["features.17.conv1"]=new_layer
        elif 'features.17.conv1.conv' == layer_name:
            self.features[17].conv1.conv= new_layer
            self.layer_names["features.17.conv1.conv"]=new_layer
            self.origin_layer_names["features.17.conv1.conv"]=new_layer
        elif 'features.17.conv1.bn' == layer_name:
            self.features[17].conv1.bn= new_layer
            self.layer_names["features.17.conv1.bn"]=new_layer
            self.origin_layer_names["features.17.conv1.bn"]=new_layer
        elif 'features.17.conv1.act' == layer_name:
            self.features[17].conv1.act= new_layer
            self.layer_names["features.17.conv1.act"]=new_layer
            self.origin_layer_names["features.17.conv1.act"]=new_layer
        elif 'features.17.conv1.act.act' == layer_name:
            self.features[17].conv1.act.act= new_layer
            self.layer_names["features.17.conv1.act.act"]=new_layer
            self.origin_layer_names["features.17.conv1.act.act"]=new_layer
        elif 'features.17.se' == layer_name:
            self.features[17].se= new_layer
            self.layer_names["features.17.se"]=new_layer
            self.origin_layer_names["features.17.se"]=new_layer
        elif 'features.17.se.pool' == layer_name:
            self.features[17].se.pool= new_layer
            self.layer_names["features.17.se.pool"]=new_layer
            self.origin_layer_names["features.17.se.pool"]=new_layer
        elif 'features.17.se.conv1' == layer_name:
            self.features[17].se.conv1= new_layer
            self.layer_names["features.17.se.conv1"]=new_layer
            self.origin_layer_names["features.17.se.conv1"]=new_layer
        elif 'features.17.se.act1' == layer_name:
            self.features[17].se.act1= new_layer
            self.layer_names["features.17.se.act1"]=new_layer
            self.origin_layer_names["features.17.se.act1"]=new_layer
        elif 'features.17.se.act1.act' == layer_name:
            self.features[17].se.act1.act= new_layer
            self.layer_names["features.17.se.act1.act"]=new_layer
            self.origin_layer_names["features.17.se.act1.act"]=new_layer
        elif 'features.17.se.conv2' == layer_name:
            self.features[17].se.conv2= new_layer
            self.layer_names["features.17.se.conv2"]=new_layer
            self.origin_layer_names["features.17.se.conv2"]=new_layer
        elif 'features.17.se.act2' == layer_name:
            self.features[17].se.act2= new_layer
            self.layer_names["features.17.se.act2"]=new_layer
            self.origin_layer_names["features.17.se.act2"]=new_layer
        elif 'features.17.se.act2.act' == layer_name:
            self.features[17].se.act2.act= new_layer
            self.layer_names["features.17.se.act2.act"]=new_layer
            self.origin_layer_names["features.17.se.act2.act"]=new_layer
        elif 'features.17.conv2' == layer_name:
            self.features[17].conv2= new_layer
            self.layer_names["features.17.conv2"]=new_layer
            self.origin_layer_names["features.17.conv2"]=new_layer
        elif 'features.17.conv2.conv' == layer_name:
            self.features[17].conv2.conv= new_layer
            self.layer_names["features.17.conv2.conv"]=new_layer
            self.origin_layer_names["features.17.conv2.conv"]=new_layer
        elif 'features.17.conv2.bn' == layer_name:
            self.features[17].conv2.bn= new_layer
            self.layer_names["features.17.conv2.bn"]=new_layer
            self.origin_layer_names["features.17.conv2.bn"]=new_layer
        elif 'features.18' == layer_name:
            self.features[18]= new_layer
            self.layer_names["features.18"]=new_layer
            self.origin_layer_names["features.18"]=new_layer
        elif 'features.19' == layer_name:
            self.features[19]= new_layer
            self.layer_names["features.19"]=new_layer
            self.origin_layer_names["features.19"]=new_layer
        elif 'features.20' == layer_name:
            self.features[20]= new_layer
            self.layer_names["features.20"]=new_layer
            self.origin_layer_names["features.20"]=new_layer
        elif 'features.20.act' == layer_name:
            self.features[20].act= new_layer
            self.layer_names["features.20.act"]=new_layer
            self.origin_layer_names["features.20.act"]=new_layer
        elif 'features.21' == layer_name:
            self.features[21]= new_layer
            self.layer_names["features.21"]=new_layer
            self.origin_layer_names["features.21"]=new_layer
        elif 'features.22' == layer_name:
            self.features[22]= new_layer
            self.layer_names["features.22"]=new_layer
            self.origin_layer_names["features.22"]=new_layer
        elif 'features.23' == layer_name:
            self.features[23]= new_layer
            self.layer_names["features.23"]=new_layer
            self.origin_layer_names["features.23"]=new_layer
        elif 'features.23.act' == layer_name:
            self.features[23].act= new_layer
            self.layer_names["features.23.act"]=new_layer
            self.origin_layer_names["features.23.act"]=new_layer
        elif 'output' == layer_name:
            self.output= new_layer
            self.layer_names["output"]=new_layer
            self.origin_layer_names["output"]=new_layer

    def set_origin_layers(self,layer_name,new_layer):
        if 'features' == layer_name:
            self.features= new_layer
            self.origin_layer_names["features"]=new_layer
        elif 'features.0' == layer_name:
            self.features[0]= new_layer
            self.origin_layer_names["features.0"]=new_layer
        elif 'features.1' == layer_name:
            self.features[1]= new_layer
            self.origin_layer_names["features.1"]=new_layer
        elif 'features.2' == layer_name:
            self.features[2]= new_layer
            self.origin_layer_names["features.2"]=new_layer
        elif 'features.2.act' == layer_name:
            self.features[2].act= new_layer
            self.origin_layer_names["features.2.act"]=new_layer
        elif 'features.3' == layer_name:
            self.features[3]= new_layer
            self.origin_layer_names["features.3"]=new_layer
        elif 'features.3.conv1' == layer_name:
            self.features[3].conv1= new_layer
            self.origin_layer_names["features.3.conv1"]=new_layer
        elif 'features.3.conv1.conv' == layer_name:
            self.features[3].conv1.conv= new_layer
            self.origin_layer_names["features.3.conv1.conv"]=new_layer
        elif 'features.3.conv1.bn' == layer_name:
            self.features[3].conv1.bn= new_layer
            self.origin_layer_names["features.3.conv1.bn"]=new_layer
        elif 'features.3.conv1.act' == layer_name:
            self.features[3].conv1.act= new_layer
            self.origin_layer_names["features.3.conv1.act"]=new_layer
        elif 'features.3.conv1.act.act' == layer_name:
            self.features[3].conv1.act.act= new_layer
            self.origin_layer_names["features.3.conv1.act.act"]=new_layer
        elif 'features.3.conv2' == layer_name:
            self.features[3].conv2= new_layer
            self.origin_layer_names["features.3.conv2"]=new_layer
        elif 'features.3.conv2.conv' == layer_name:
            self.features[3].conv2.conv= new_layer
            self.origin_layer_names["features.3.conv2.conv"]=new_layer
        elif 'features.3.conv2.bn' == layer_name:
            self.features[3].conv2.bn= new_layer
            self.origin_layer_names["features.3.conv2.bn"]=new_layer
        elif 'features.4' == layer_name:
            self.features[4]= new_layer
            self.origin_layer_names["features.4"]=new_layer
        elif 'features.4.expand' == layer_name:
            self.features[4].expand= new_layer
            self.origin_layer_names["features.4.expand"]=new_layer
        elif 'features.4.expand.conv' == layer_name:
            self.features[4].expand.conv= new_layer
            self.origin_layer_names["features.4.expand.conv"]=new_layer
        elif 'features.4.expand.bn' == layer_name:
            self.features[4].expand.bn= new_layer
            self.origin_layer_names["features.4.expand.bn"]=new_layer
        elif 'features.4.expand.act' == layer_name:
            self.features[4].expand.act= new_layer
            self.origin_layer_names["features.4.expand.act"]=new_layer
        elif 'features.4.expand.act.act' == layer_name:
            self.features[4].expand.act.act= new_layer
            self.origin_layer_names["features.4.expand.act.act"]=new_layer
        elif 'features.4.conv1' == layer_name:
            self.features[4].conv1= new_layer
            self.origin_layer_names["features.4.conv1"]=new_layer
        elif 'features.4.conv1.conv' == layer_name:
            self.features[4].conv1.conv= new_layer
            self.origin_layer_names["features.4.conv1.conv"]=new_layer
        elif 'features.4.conv1.bn' == layer_name:
            self.features[4].conv1.bn= new_layer
            self.origin_layer_names["features.4.conv1.bn"]=new_layer
        elif 'features.4.conv1.act' == layer_name:
            self.features[4].conv1.act= new_layer
            self.origin_layer_names["features.4.conv1.act"]=new_layer
        elif 'features.4.conv1.act.act' == layer_name:
            self.features[4].conv1.act.act= new_layer
            self.origin_layer_names["features.4.conv1.act.act"]=new_layer
        elif 'features.4.conv2' == layer_name:
            self.features[4].conv2= new_layer
            self.origin_layer_names["features.4.conv2"]=new_layer
        elif 'features.4.conv2.conv' == layer_name:
            self.features[4].conv2.conv= new_layer
            self.origin_layer_names["features.4.conv2.conv"]=new_layer
        elif 'features.4.conv2.bn' == layer_name:
            self.features[4].conv2.bn= new_layer
            self.origin_layer_names["features.4.conv2.bn"]=new_layer
        elif 'features.5' == layer_name:
            self.features[5]= new_layer
            self.origin_layer_names["features.5"]=new_layer
        elif 'features.5.expand' == layer_name:
            self.features[5].expand= new_layer
            self.origin_layer_names["features.5.expand"]=new_layer
        elif 'features.5.expand.conv' == layer_name:
            self.features[5].expand.conv= new_layer
            self.origin_layer_names["features.5.expand.conv"]=new_layer
        elif 'features.5.expand.bn' == layer_name:
            self.features[5].expand.bn= new_layer
            self.origin_layer_names["features.5.expand.bn"]=new_layer
        elif 'features.5.expand.act' == layer_name:
            self.features[5].expand.act= new_layer
            self.origin_layer_names["features.5.expand.act"]=new_layer
        elif 'features.5.expand.act.act' == layer_name:
            self.features[5].expand.act.act= new_layer
            self.origin_layer_names["features.5.expand.act.act"]=new_layer
        elif 'features.5.conv1' == layer_name:
            self.features[5].conv1= new_layer
            self.origin_layer_names["features.5.conv1"]=new_layer
        elif 'features.5.conv1.conv' == layer_name:
            self.features[5].conv1.conv= new_layer
            self.origin_layer_names["features.5.conv1.conv"]=new_layer
        elif 'features.5.conv1.bn' == layer_name:
            self.features[5].conv1.bn= new_layer
            self.origin_layer_names["features.5.conv1.bn"]=new_layer
        elif 'features.5.conv1.act' == layer_name:
            self.features[5].conv1.act= new_layer
            self.origin_layer_names["features.5.conv1.act"]=new_layer
        elif 'features.5.conv1.act.act' == layer_name:
            self.features[5].conv1.act.act= new_layer
            self.origin_layer_names["features.5.conv1.act.act"]=new_layer
        elif 'features.5.conv2' == layer_name:
            self.features[5].conv2= new_layer
            self.origin_layer_names["features.5.conv2"]=new_layer
        elif 'features.5.conv2.conv' == layer_name:
            self.features[5].conv2.conv= new_layer
            self.origin_layer_names["features.5.conv2.conv"]=new_layer
        elif 'features.5.conv2.bn' == layer_name:
            self.features[5].conv2.bn= new_layer
            self.origin_layer_names["features.5.conv2.bn"]=new_layer
        elif 'features.6' == layer_name:
            self.features[6]= new_layer
            self.origin_layer_names["features.6"]=new_layer
        elif 'features.6.expand' == layer_name:
            self.features[6].expand= new_layer
            self.origin_layer_names["features.6.expand"]=new_layer
        elif 'features.6.expand.conv' == layer_name:
            self.features[6].expand.conv= new_layer
            self.origin_layer_names["features.6.expand.conv"]=new_layer
        elif 'features.6.expand.bn' == layer_name:
            self.features[6].expand.bn= new_layer
            self.origin_layer_names["features.6.expand.bn"]=new_layer
        elif 'features.6.expand.act' == layer_name:
            self.features[6].expand.act= new_layer
            self.origin_layer_names["features.6.expand.act"]=new_layer
        elif 'features.6.expand.act.act' == layer_name:
            self.features[6].expand.act.act= new_layer
            self.origin_layer_names["features.6.expand.act.act"]=new_layer
        elif 'features.6.conv1' == layer_name:
            self.features[6].conv1= new_layer
            self.origin_layer_names["features.6.conv1"]=new_layer
        elif 'features.6.conv1.conv' == layer_name:
            self.features[6].conv1.conv= new_layer
            self.origin_layer_names["features.6.conv1.conv"]=new_layer
        elif 'features.6.conv1.bn' == layer_name:
            self.features[6].conv1.bn= new_layer
            self.origin_layer_names["features.6.conv1.bn"]=new_layer
        elif 'features.6.conv1.act' == layer_name:
            self.features[6].conv1.act= new_layer
            self.origin_layer_names["features.6.conv1.act"]=new_layer
        elif 'features.6.conv1.act.act' == layer_name:
            self.features[6].conv1.act.act= new_layer
            self.origin_layer_names["features.6.conv1.act.act"]=new_layer
        elif 'features.6.se' == layer_name:
            self.features[6].se= new_layer
            self.origin_layer_names["features.6.se"]=new_layer
        elif 'features.6.se.pool' == layer_name:
            self.features[6].se.pool= new_layer
            self.origin_layer_names["features.6.se.pool"]=new_layer
        elif 'features.6.se.conv1' == layer_name:
            self.features[6].se.conv1= new_layer
            self.origin_layer_names["features.6.se.conv1"]=new_layer
        elif 'features.6.se.act1' == layer_name:
            self.features[6].se.act1= new_layer
            self.origin_layer_names["features.6.se.act1"]=new_layer
        elif 'features.6.se.act1.act' == layer_name:
            self.features[6].se.act1.act= new_layer
            self.origin_layer_names["features.6.se.act1.act"]=new_layer
        elif 'features.6.se.conv2' == layer_name:
            self.features[6].se.conv2= new_layer
            self.origin_layer_names["features.6.se.conv2"]=new_layer
        elif 'features.6.se.act2' == layer_name:
            self.features[6].se.act2= new_layer
            self.origin_layer_names["features.6.se.act2"]=new_layer
        elif 'features.6.se.act2.act' == layer_name:
            self.features[6].se.act2.act= new_layer
            self.origin_layer_names["features.6.se.act2.act"]=new_layer
        elif 'features.6.conv2' == layer_name:
            self.features[6].conv2= new_layer
            self.origin_layer_names["features.6.conv2"]=new_layer
        elif 'features.6.conv2.conv' == layer_name:
            self.features[6].conv2.conv= new_layer
            self.origin_layer_names["features.6.conv2.conv"]=new_layer
        elif 'features.6.conv2.bn' == layer_name:
            self.features[6].conv2.bn= new_layer
            self.origin_layer_names["features.6.conv2.bn"]=new_layer
        elif 'features.7' == layer_name:
            self.features[7]= new_layer
            self.origin_layer_names["features.7"]=new_layer
        elif 'features.7.expand' == layer_name:
            self.features[7].expand= new_layer
            self.origin_layer_names["features.7.expand"]=new_layer
        elif 'features.7.expand.conv' == layer_name:
            self.features[7].expand.conv= new_layer
            self.origin_layer_names["features.7.expand.conv"]=new_layer
        elif 'features.7.expand.bn' == layer_name:
            self.features[7].expand.bn= new_layer
            self.origin_layer_names["features.7.expand.bn"]=new_layer
        elif 'features.7.expand.act' == layer_name:
            self.features[7].expand.act= new_layer
            self.origin_layer_names["features.7.expand.act"]=new_layer
        elif 'features.7.expand.act.act' == layer_name:
            self.features[7].expand.act.act= new_layer
            self.origin_layer_names["features.7.expand.act.act"]=new_layer
        elif 'features.7.conv1' == layer_name:
            self.features[7].conv1= new_layer
            self.origin_layer_names["features.7.conv1"]=new_layer
        elif 'features.7.conv1.conv' == layer_name:
            self.features[7].conv1.conv= new_layer
            self.origin_layer_names["features.7.conv1.conv"]=new_layer
        elif 'features.7.conv1.bn' == layer_name:
            self.features[7].conv1.bn= new_layer
            self.origin_layer_names["features.7.conv1.bn"]=new_layer
        elif 'features.7.conv1.act' == layer_name:
            self.features[7].conv1.act= new_layer
            self.origin_layer_names["features.7.conv1.act"]=new_layer
        elif 'features.7.conv1.act.act' == layer_name:
            self.features[7].conv1.act.act= new_layer
            self.origin_layer_names["features.7.conv1.act.act"]=new_layer
        elif 'features.7.se' == layer_name:
            self.features[7].se= new_layer
            self.origin_layer_names["features.7.se"]=new_layer
        elif 'features.7.se.pool' == layer_name:
            self.features[7].se.pool= new_layer
            self.origin_layer_names["features.7.se.pool"]=new_layer
        elif 'features.7.se.conv1' == layer_name:
            self.features[7].se.conv1= new_layer
            self.origin_layer_names["features.7.se.conv1"]=new_layer
        elif 'features.7.se.act1' == layer_name:
            self.features[7].se.act1= new_layer
            self.origin_layer_names["features.7.se.act1"]=new_layer
        elif 'features.7.se.act1.act' == layer_name:
            self.features[7].se.act1.act= new_layer
            self.origin_layer_names["features.7.se.act1.act"]=new_layer
        elif 'features.7.se.conv2' == layer_name:
            self.features[7].se.conv2= new_layer
            self.origin_layer_names["features.7.se.conv2"]=new_layer
        elif 'features.7.se.act2' == layer_name:
            self.features[7].se.act2= new_layer
            self.origin_layer_names["features.7.se.act2"]=new_layer
        elif 'features.7.se.act2.act' == layer_name:
            self.features[7].se.act2.act= new_layer
            self.origin_layer_names["features.7.se.act2.act"]=new_layer
        elif 'features.7.conv2' == layer_name:
            self.features[7].conv2= new_layer
            self.origin_layer_names["features.7.conv2"]=new_layer
        elif 'features.7.conv2.conv' == layer_name:
            self.features[7].conv2.conv= new_layer
            self.origin_layer_names["features.7.conv2.conv"]=new_layer
        elif 'features.7.conv2.bn' == layer_name:
            self.features[7].conv2.bn= new_layer
            self.origin_layer_names["features.7.conv2.bn"]=new_layer
        elif 'features.8' == layer_name:
            self.features[8]= new_layer
            self.origin_layer_names["features.8"]=new_layer
        elif 'features.8.expand' == layer_name:
            self.features[8].expand= new_layer
            self.origin_layer_names["features.8.expand"]=new_layer
        elif 'features.8.expand.conv' == layer_name:
            self.features[8].expand.conv= new_layer
            self.origin_layer_names["features.8.expand.conv"]=new_layer
        elif 'features.8.expand.bn' == layer_name:
            self.features[8].expand.bn= new_layer
            self.origin_layer_names["features.8.expand.bn"]=new_layer
        elif 'features.8.expand.act' == layer_name:
            self.features[8].expand.act= new_layer
            self.origin_layer_names["features.8.expand.act"]=new_layer
        elif 'features.8.expand.act.act' == layer_name:
            self.features[8].expand.act.act= new_layer
            self.origin_layer_names["features.8.expand.act.act"]=new_layer
        elif 'features.8.conv1' == layer_name:
            self.features[8].conv1= new_layer
            self.origin_layer_names["features.8.conv1"]=new_layer
        elif 'features.8.conv1.conv' == layer_name:
            self.features[8].conv1.conv= new_layer
            self.origin_layer_names["features.8.conv1.conv"]=new_layer
        elif 'features.8.conv1.bn' == layer_name:
            self.features[8].conv1.bn= new_layer
            self.origin_layer_names["features.8.conv1.bn"]=new_layer
        elif 'features.8.conv1.act' == layer_name:
            self.features[8].conv1.act= new_layer
            self.origin_layer_names["features.8.conv1.act"]=new_layer
        elif 'features.8.conv1.act.act' == layer_name:
            self.features[8].conv1.act.act= new_layer
            self.origin_layer_names["features.8.conv1.act.act"]=new_layer
        elif 'features.8.se' == layer_name:
            self.features[8].se= new_layer
            self.origin_layer_names["features.8.se"]=new_layer
        elif 'features.8.se.pool' == layer_name:
            self.features[8].se.pool= new_layer
            self.origin_layer_names["features.8.se.pool"]=new_layer
        elif 'features.8.se.conv1' == layer_name:
            self.features[8].se.conv1= new_layer
            self.origin_layer_names["features.8.se.conv1"]=new_layer
        elif 'features.8.se.act1' == layer_name:
            self.features[8].se.act1= new_layer
            self.origin_layer_names["features.8.se.act1"]=new_layer
        elif 'features.8.se.act1.act' == layer_name:
            self.features[8].se.act1.act= new_layer
            self.origin_layer_names["features.8.se.act1.act"]=new_layer
        elif 'features.8.se.conv2' == layer_name:
            self.features[8].se.conv2= new_layer
            self.origin_layer_names["features.8.se.conv2"]=new_layer
        elif 'features.8.se.act2' == layer_name:
            self.features[8].se.act2= new_layer
            self.origin_layer_names["features.8.se.act2"]=new_layer
        elif 'features.8.se.act2.act' == layer_name:
            self.features[8].se.act2.act= new_layer
            self.origin_layer_names["features.8.se.act2.act"]=new_layer
        elif 'features.8.conv2' == layer_name:
            self.features[8].conv2= new_layer
            self.origin_layer_names["features.8.conv2"]=new_layer
        elif 'features.8.conv2.conv' == layer_name:
            self.features[8].conv2.conv= new_layer
            self.origin_layer_names["features.8.conv2.conv"]=new_layer
        elif 'features.8.conv2.bn' == layer_name:
            self.features[8].conv2.bn= new_layer
            self.origin_layer_names["features.8.conv2.bn"]=new_layer
        elif 'features.9' == layer_name:
            self.features[9]= new_layer
            self.origin_layer_names["features.9"]=new_layer
        elif 'features.9.expand' == layer_name:
            self.features[9].expand= new_layer
            self.origin_layer_names["features.9.expand"]=new_layer
        elif 'features.9.expand.conv' == layer_name:
            self.features[9].expand.conv= new_layer
            self.origin_layer_names["features.9.expand.conv"]=new_layer
        elif 'features.9.expand.bn' == layer_name:
            self.features[9].expand.bn= new_layer
            self.origin_layer_names["features.9.expand.bn"]=new_layer
        elif 'features.9.expand.act' == layer_name:
            self.features[9].expand.act= new_layer
            self.origin_layer_names["features.9.expand.act"]=new_layer
        elif 'features.9.expand.act.act' == layer_name:
            self.features[9].expand.act.act= new_layer
            self.origin_layer_names["features.9.expand.act.act"]=new_layer
        elif 'features.9.conv1' == layer_name:
            self.features[9].conv1= new_layer
            self.origin_layer_names["features.9.conv1"]=new_layer
        elif 'features.9.conv1.conv' == layer_name:
            self.features[9].conv1.conv= new_layer
            self.origin_layer_names["features.9.conv1.conv"]=new_layer
        elif 'features.9.conv1.bn' == layer_name:
            self.features[9].conv1.bn= new_layer
            self.origin_layer_names["features.9.conv1.bn"]=new_layer
        elif 'features.9.conv1.act' == layer_name:
            self.features[9].conv1.act= new_layer
            self.origin_layer_names["features.9.conv1.act"]=new_layer
        elif 'features.9.conv1.act.act' == layer_name:
            self.features[9].conv1.act.act= new_layer
            self.origin_layer_names["features.9.conv1.act.act"]=new_layer
        elif 'features.9.conv2' == layer_name:
            self.features[9].conv2= new_layer
            self.origin_layer_names["features.9.conv2"]=new_layer
        elif 'features.9.conv2.conv' == layer_name:
            self.features[9].conv2.conv= new_layer
            self.origin_layer_names["features.9.conv2.conv"]=new_layer
        elif 'features.9.conv2.bn' == layer_name:
            self.features[9].conv2.bn= new_layer
            self.origin_layer_names["features.9.conv2.bn"]=new_layer
        elif 'features.10' == layer_name:
            self.features[10]= new_layer
            self.origin_layer_names["features.10"]=new_layer
        elif 'features.10.expand' == layer_name:
            self.features[10].expand= new_layer
            self.origin_layer_names["features.10.expand"]=new_layer
        elif 'features.10.expand.conv' == layer_name:
            self.features[10].expand.conv= new_layer
            self.origin_layer_names["features.10.expand.conv"]=new_layer
        elif 'features.10.expand.bn' == layer_name:
            self.features[10].expand.bn= new_layer
            self.origin_layer_names["features.10.expand.bn"]=new_layer
        elif 'features.10.expand.act' == layer_name:
            self.features[10].expand.act= new_layer
            self.origin_layer_names["features.10.expand.act"]=new_layer
        elif 'features.10.expand.act.act' == layer_name:
            self.features[10].expand.act.act= new_layer
            self.origin_layer_names["features.10.expand.act.act"]=new_layer
        elif 'features.10.conv1' == layer_name:
            self.features[10].conv1= new_layer
            self.origin_layer_names["features.10.conv1"]=new_layer
        elif 'features.10.conv1.conv' == layer_name:
            self.features[10].conv1.conv= new_layer
            self.origin_layer_names["features.10.conv1.conv"]=new_layer
        elif 'features.10.conv1.bn' == layer_name:
            self.features[10].conv1.bn= new_layer
            self.origin_layer_names["features.10.conv1.bn"]=new_layer
        elif 'features.10.conv1.act' == layer_name:
            self.features[10].conv1.act= new_layer
            self.origin_layer_names["features.10.conv1.act"]=new_layer
        elif 'features.10.conv1.act.act' == layer_name:
            self.features[10].conv1.act.act= new_layer
            self.origin_layer_names["features.10.conv1.act.act"]=new_layer
        elif 'features.10.conv2' == layer_name:
            self.features[10].conv2= new_layer
            self.origin_layer_names["features.10.conv2"]=new_layer
        elif 'features.10.conv2.conv' == layer_name:
            self.features[10].conv2.conv= new_layer
            self.origin_layer_names["features.10.conv2.conv"]=new_layer
        elif 'features.10.conv2.bn' == layer_name:
            self.features[10].conv2.bn= new_layer
            self.origin_layer_names["features.10.conv2.bn"]=new_layer
        elif 'features.11' == layer_name:
            self.features[11]= new_layer
            self.origin_layer_names["features.11"]=new_layer
        elif 'features.11.expand' == layer_name:
            self.features[11].expand= new_layer
            self.origin_layer_names["features.11.expand"]=new_layer
        elif 'features.11.expand.conv' == layer_name:
            self.features[11].expand.conv= new_layer
            self.origin_layer_names["features.11.expand.conv"]=new_layer
        elif 'features.11.expand.bn' == layer_name:
            self.features[11].expand.bn= new_layer
            self.origin_layer_names["features.11.expand.bn"]=new_layer
        elif 'features.11.expand.act' == layer_name:
            self.features[11].expand.act= new_layer
            self.origin_layer_names["features.11.expand.act"]=new_layer
        elif 'features.11.expand.act.act' == layer_name:
            self.features[11].expand.act.act= new_layer
            self.origin_layer_names["features.11.expand.act.act"]=new_layer
        elif 'features.11.conv1' == layer_name:
            self.features[11].conv1= new_layer
            self.origin_layer_names["features.11.conv1"]=new_layer
        elif 'features.11.conv1.conv' == layer_name:
            self.features[11].conv1.conv= new_layer
            self.origin_layer_names["features.11.conv1.conv"]=new_layer
        elif 'features.11.conv1.bn' == layer_name:
            self.features[11].conv1.bn= new_layer
            self.origin_layer_names["features.11.conv1.bn"]=new_layer
        elif 'features.11.conv1.act' == layer_name:
            self.features[11].conv1.act= new_layer
            self.origin_layer_names["features.11.conv1.act"]=new_layer
        elif 'features.11.conv1.act.act' == layer_name:
            self.features[11].conv1.act.act= new_layer
            self.origin_layer_names["features.11.conv1.act.act"]=new_layer
        elif 'features.11.conv2' == layer_name:
            self.features[11].conv2= new_layer
            self.origin_layer_names["features.11.conv2"]=new_layer
        elif 'features.11.conv2.conv' == layer_name:
            self.features[11].conv2.conv= new_layer
            self.origin_layer_names["features.11.conv2.conv"]=new_layer
        elif 'features.11.conv2.bn' == layer_name:
            self.features[11].conv2.bn= new_layer
            self.origin_layer_names["features.11.conv2.bn"]=new_layer
        elif 'features.12' == layer_name:
            self.features[12]= new_layer
            self.origin_layer_names["features.12"]=new_layer
        elif 'features.12.expand' == layer_name:
            self.features[12].expand= new_layer
            self.origin_layer_names["features.12.expand"]=new_layer
        elif 'features.12.expand.conv' == layer_name:
            self.features[12].expand.conv= new_layer
            self.origin_layer_names["features.12.expand.conv"]=new_layer
        elif 'features.12.expand.bn' == layer_name:
            self.features[12].expand.bn= new_layer
            self.origin_layer_names["features.12.expand.bn"]=new_layer
        elif 'features.12.expand.act' == layer_name:
            self.features[12].expand.act= new_layer
            self.origin_layer_names["features.12.expand.act"]=new_layer
        elif 'features.12.expand.act.act' == layer_name:
            self.features[12].expand.act.act= new_layer
            self.origin_layer_names["features.12.expand.act.act"]=new_layer
        elif 'features.12.conv1' == layer_name:
            self.features[12].conv1= new_layer
            self.origin_layer_names["features.12.conv1"]=new_layer
        elif 'features.12.conv1.conv' == layer_name:
            self.features[12].conv1.conv= new_layer
            self.origin_layer_names["features.12.conv1.conv"]=new_layer
        elif 'features.12.conv1.bn' == layer_name:
            self.features[12].conv1.bn= new_layer
            self.origin_layer_names["features.12.conv1.bn"]=new_layer
        elif 'features.12.conv1.act' == layer_name:
            self.features[12].conv1.act= new_layer
            self.origin_layer_names["features.12.conv1.act"]=new_layer
        elif 'features.12.conv1.act.act' == layer_name:
            self.features[12].conv1.act.act= new_layer
            self.origin_layer_names["features.12.conv1.act.act"]=new_layer
        elif 'features.12.conv2' == layer_name:
            self.features[12].conv2= new_layer
            self.origin_layer_names["features.12.conv2"]=new_layer
        elif 'features.12.conv2.conv' == layer_name:
            self.features[12].conv2.conv= new_layer
            self.origin_layer_names["features.12.conv2.conv"]=new_layer
        elif 'features.12.conv2.bn' == layer_name:
            self.features[12].conv2.bn= new_layer
            self.origin_layer_names["features.12.conv2.bn"]=new_layer
        elif 'features.13' == layer_name:
            self.features[13]= new_layer
            self.origin_layer_names["features.13"]=new_layer
        elif 'features.13.expand' == layer_name:
            self.features[13].expand= new_layer
            self.origin_layer_names["features.13.expand"]=new_layer
        elif 'features.13.expand.conv' == layer_name:
            self.features[13].expand.conv= new_layer
            self.origin_layer_names["features.13.expand.conv"]=new_layer
        elif 'features.13.expand.bn' == layer_name:
            self.features[13].expand.bn= new_layer
            self.origin_layer_names["features.13.expand.bn"]=new_layer
        elif 'features.13.expand.act' == layer_name:
            self.features[13].expand.act= new_layer
            self.origin_layer_names["features.13.expand.act"]=new_layer
        elif 'features.13.expand.act.act' == layer_name:
            self.features[13].expand.act.act= new_layer
            self.origin_layer_names["features.13.expand.act.act"]=new_layer
        elif 'features.13.conv1' == layer_name:
            self.features[13].conv1= new_layer
            self.origin_layer_names["features.13.conv1"]=new_layer
        elif 'features.13.conv1.conv' == layer_name:
            self.features[13].conv1.conv= new_layer
            self.origin_layer_names["features.13.conv1.conv"]=new_layer
        elif 'features.13.conv1.bn' == layer_name:
            self.features[13].conv1.bn= new_layer
            self.origin_layer_names["features.13.conv1.bn"]=new_layer
        elif 'features.13.conv1.act' == layer_name:
            self.features[13].conv1.act= new_layer
            self.origin_layer_names["features.13.conv1.act"]=new_layer
        elif 'features.13.conv1.act.act' == layer_name:
            self.features[13].conv1.act.act= new_layer
            self.origin_layer_names["features.13.conv1.act.act"]=new_layer
        elif 'features.13.se' == layer_name:
            self.features[13].se= new_layer
            self.origin_layer_names["features.13.se"]=new_layer
        elif 'features.13.se.pool' == layer_name:
            self.features[13].se.pool= new_layer
            self.origin_layer_names["features.13.se.pool"]=new_layer
        elif 'features.13.se.conv1' == layer_name:
            self.features[13].se.conv1= new_layer
            self.origin_layer_names["features.13.se.conv1"]=new_layer
        elif 'features.13.se.act1' == layer_name:
            self.features[13].se.act1= new_layer
            self.origin_layer_names["features.13.se.act1"]=new_layer
        elif 'features.13.se.act1.act' == layer_name:
            self.features[13].se.act1.act= new_layer
            self.origin_layer_names["features.13.se.act1.act"]=new_layer
        elif 'features.13.se.conv2' == layer_name:
            self.features[13].se.conv2= new_layer
            self.origin_layer_names["features.13.se.conv2"]=new_layer
        elif 'features.13.se.act2' == layer_name:
            self.features[13].se.act2= new_layer
            self.origin_layer_names["features.13.se.act2"]=new_layer
        elif 'features.13.se.act2.act' == layer_name:
            self.features[13].se.act2.act= new_layer
            self.origin_layer_names["features.13.se.act2.act"]=new_layer
        elif 'features.13.conv2' == layer_name:
            self.features[13].conv2= new_layer
            self.origin_layer_names["features.13.conv2"]=new_layer
        elif 'features.13.conv2.conv' == layer_name:
            self.features[13].conv2.conv= new_layer
            self.origin_layer_names["features.13.conv2.conv"]=new_layer
        elif 'features.13.conv2.bn' == layer_name:
            self.features[13].conv2.bn= new_layer
            self.origin_layer_names["features.13.conv2.bn"]=new_layer
        elif 'features.14' == layer_name:
            self.features[14]= new_layer
            self.origin_layer_names["features.14"]=new_layer
        elif 'features.14.expand' == layer_name:
            self.features[14].expand= new_layer
            self.origin_layer_names["features.14.expand"]=new_layer
        elif 'features.14.expand.conv' == layer_name:
            self.features[14].expand.conv= new_layer
            self.origin_layer_names["features.14.expand.conv"]=new_layer
        elif 'features.14.expand.bn' == layer_name:
            self.features[14].expand.bn= new_layer
            self.origin_layer_names["features.14.expand.bn"]=new_layer
        elif 'features.14.expand.act' == layer_name:
            self.features[14].expand.act= new_layer
            self.origin_layer_names["features.14.expand.act"]=new_layer
        elif 'features.14.expand.act.act' == layer_name:
            self.features[14].expand.act.act= new_layer
            self.origin_layer_names["features.14.expand.act.act"]=new_layer
        elif 'features.14.conv1' == layer_name:
            self.features[14].conv1= new_layer
            self.origin_layer_names["features.14.conv1"]=new_layer
        elif 'features.14.conv1.conv' == layer_name:
            self.features[14].conv1.conv= new_layer
            self.origin_layer_names["features.14.conv1.conv"]=new_layer
        elif 'features.14.conv1.bn' == layer_name:
            self.features[14].conv1.bn= new_layer
            self.origin_layer_names["features.14.conv1.bn"]=new_layer
        elif 'features.14.conv1.act' == layer_name:
            self.features[14].conv1.act= new_layer
            self.origin_layer_names["features.14.conv1.act"]=new_layer
        elif 'features.14.conv1.act.act' == layer_name:
            self.features[14].conv1.act.act= new_layer
            self.origin_layer_names["features.14.conv1.act.act"]=new_layer
        elif 'features.14.se' == layer_name:
            self.features[14].se= new_layer
            self.origin_layer_names["features.14.se"]=new_layer
        elif 'features.14.se.pool' == layer_name:
            self.features[14].se.pool= new_layer
            self.origin_layer_names["features.14.se.pool"]=new_layer
        elif 'features.14.se.conv1' == layer_name:
            self.features[14].se.conv1= new_layer
            self.origin_layer_names["features.14.se.conv1"]=new_layer
        elif 'features.14.se.act1' == layer_name:
            self.features[14].se.act1= new_layer
            self.origin_layer_names["features.14.se.act1"]=new_layer
        elif 'features.14.se.act1.act' == layer_name:
            self.features[14].se.act1.act= new_layer
            self.origin_layer_names["features.14.se.act1.act"]=new_layer
        elif 'features.14.se.conv2' == layer_name:
            self.features[14].se.conv2= new_layer
            self.origin_layer_names["features.14.se.conv2"]=new_layer
        elif 'features.14.se.act2' == layer_name:
            self.features[14].se.act2= new_layer
            self.origin_layer_names["features.14.se.act2"]=new_layer
        elif 'features.14.se.act2.act' == layer_name:
            self.features[14].se.act2.act= new_layer
            self.origin_layer_names["features.14.se.act2.act"]=new_layer
        elif 'features.14.conv2' == layer_name:
            self.features[14].conv2= new_layer
            self.origin_layer_names["features.14.conv2"]=new_layer
        elif 'features.14.conv2.conv' == layer_name:
            self.features[14].conv2.conv= new_layer
            self.origin_layer_names["features.14.conv2.conv"]=new_layer
        elif 'features.14.conv2.bn' == layer_name:
            self.features[14].conv2.bn= new_layer
            self.origin_layer_names["features.14.conv2.bn"]=new_layer
        elif 'features.15' == layer_name:
            self.features[15]= new_layer
            self.origin_layer_names["features.15"]=new_layer
        elif 'features.15.expand' == layer_name:
            self.features[15].expand= new_layer
            self.origin_layer_names["features.15.expand"]=new_layer
        elif 'features.15.expand.conv' == layer_name:
            self.features[15].expand.conv= new_layer
            self.origin_layer_names["features.15.expand.conv"]=new_layer
        elif 'features.15.expand.bn' == layer_name:
            self.features[15].expand.bn= new_layer
            self.origin_layer_names["features.15.expand.bn"]=new_layer
        elif 'features.15.expand.act' == layer_name:
            self.features[15].expand.act= new_layer
            self.origin_layer_names["features.15.expand.act"]=new_layer
        elif 'features.15.expand.act.act' == layer_name:
            self.features[15].expand.act.act= new_layer
            self.origin_layer_names["features.15.expand.act.act"]=new_layer
        elif 'features.15.conv1' == layer_name:
            self.features[15].conv1= new_layer
            self.origin_layer_names["features.15.conv1"]=new_layer
        elif 'features.15.conv1.conv' == layer_name:
            self.features[15].conv1.conv= new_layer
            self.origin_layer_names["features.15.conv1.conv"]=new_layer
        elif 'features.15.conv1.bn' == layer_name:
            self.features[15].conv1.bn= new_layer
            self.origin_layer_names["features.15.conv1.bn"]=new_layer
        elif 'features.15.conv1.act' == layer_name:
            self.features[15].conv1.act= new_layer
            self.origin_layer_names["features.15.conv1.act"]=new_layer
        elif 'features.15.conv1.act.act' == layer_name:
            self.features[15].conv1.act.act= new_layer
            self.origin_layer_names["features.15.conv1.act.act"]=new_layer
        elif 'features.15.se' == layer_name:
            self.features[15].se= new_layer
            self.origin_layer_names["features.15.se"]=new_layer
        elif 'features.15.se.pool' == layer_name:
            self.features[15].se.pool= new_layer
            self.origin_layer_names["features.15.se.pool"]=new_layer
        elif 'features.15.se.conv1' == layer_name:
            self.features[15].se.conv1= new_layer
            self.origin_layer_names["features.15.se.conv1"]=new_layer
        elif 'features.15.se.act1' == layer_name:
            self.features[15].se.act1= new_layer
            self.origin_layer_names["features.15.se.act1"]=new_layer
        elif 'features.15.se.act1.act' == layer_name:
            self.features[15].se.act1.act= new_layer
            self.origin_layer_names["features.15.se.act1.act"]=new_layer
        elif 'features.15.se.conv2' == layer_name:
            self.features[15].se.conv2= new_layer
            self.origin_layer_names["features.15.se.conv2"]=new_layer
        elif 'features.15.se.act2' == layer_name:
            self.features[15].se.act2= new_layer
            self.origin_layer_names["features.15.se.act2"]=new_layer
        elif 'features.15.se.act2.act' == layer_name:
            self.features[15].se.act2.act= new_layer
            self.origin_layer_names["features.15.se.act2.act"]=new_layer
        elif 'features.15.conv2' == layer_name:
            self.features[15].conv2= new_layer
            self.origin_layer_names["features.15.conv2"]=new_layer
        elif 'features.15.conv2.conv' == layer_name:
            self.features[15].conv2.conv= new_layer
            self.origin_layer_names["features.15.conv2.conv"]=new_layer
        elif 'features.15.conv2.bn' == layer_name:
            self.features[15].conv2.bn= new_layer
            self.origin_layer_names["features.15.conv2.bn"]=new_layer
        elif 'features.16' == layer_name:
            self.features[16]= new_layer
            self.origin_layer_names["features.16"]=new_layer
        elif 'features.16.expand' == layer_name:
            self.features[16].expand= new_layer
            self.origin_layer_names["features.16.expand"]=new_layer
        elif 'features.16.expand.conv' == layer_name:
            self.features[16].expand.conv= new_layer
            self.origin_layer_names["features.16.expand.conv"]=new_layer
        elif 'features.16.expand.bn' == layer_name:
            self.features[16].expand.bn= new_layer
            self.origin_layer_names["features.16.expand.bn"]=new_layer
        elif 'features.16.expand.act' == layer_name:
            self.features[16].expand.act= new_layer
            self.origin_layer_names["features.16.expand.act"]=new_layer
        elif 'features.16.expand.act.act' == layer_name:
            self.features[16].expand.act.act= new_layer
            self.origin_layer_names["features.16.expand.act.act"]=new_layer
        elif 'features.16.conv1' == layer_name:
            self.features[16].conv1= new_layer
            self.origin_layer_names["features.16.conv1"]=new_layer
        elif 'features.16.conv1.conv' == layer_name:
            self.features[16].conv1.conv= new_layer
            self.origin_layer_names["features.16.conv1.conv"]=new_layer
        elif 'features.16.conv1.bn' == layer_name:
            self.features[16].conv1.bn= new_layer
            self.origin_layer_names["features.16.conv1.bn"]=new_layer
        elif 'features.16.conv1.act' == layer_name:
            self.features[16].conv1.act= new_layer
            self.origin_layer_names["features.16.conv1.act"]=new_layer
        elif 'features.16.conv1.act.act' == layer_name:
            self.features[16].conv1.act.act= new_layer
            self.origin_layer_names["features.16.conv1.act.act"]=new_layer
        elif 'features.16.se' == layer_name:
            self.features[16].se= new_layer
            self.origin_layer_names["features.16.se"]=new_layer
        elif 'features.16.se.pool' == layer_name:
            self.features[16].se.pool= new_layer
            self.origin_layer_names["features.16.se.pool"]=new_layer
        elif 'features.16.se.conv1' == layer_name:
            self.features[16].se.conv1= new_layer
            self.origin_layer_names["features.16.se.conv1"]=new_layer
        elif 'features.16.se.act1' == layer_name:
            self.features[16].se.act1= new_layer
            self.origin_layer_names["features.16.se.act1"]=new_layer
        elif 'features.16.se.act1.act' == layer_name:
            self.features[16].se.act1.act= new_layer
            self.origin_layer_names["features.16.se.act1.act"]=new_layer
        elif 'features.16.se.conv2' == layer_name:
            self.features[16].se.conv2= new_layer
            self.origin_layer_names["features.16.se.conv2"]=new_layer
        elif 'features.16.se.act2' == layer_name:
            self.features[16].se.act2= new_layer
            self.origin_layer_names["features.16.se.act2"]=new_layer
        elif 'features.16.se.act2.act' == layer_name:
            self.features[16].se.act2.act= new_layer
            self.origin_layer_names["features.16.se.act2.act"]=new_layer
        elif 'features.16.conv2' == layer_name:
            self.features[16].conv2= new_layer
            self.origin_layer_names["features.16.conv2"]=new_layer
        elif 'features.16.conv2.conv' == layer_name:
            self.features[16].conv2.conv= new_layer
            self.origin_layer_names["features.16.conv2.conv"]=new_layer
        elif 'features.16.conv2.bn' == layer_name:
            self.features[16].conv2.bn= new_layer
            self.origin_layer_names["features.16.conv2.bn"]=new_layer
        elif 'features.17' == layer_name:
            self.features[17]= new_layer
            self.origin_layer_names["features.17"]=new_layer
        elif 'features.17.expand' == layer_name:
            self.features[17].expand= new_layer
            self.origin_layer_names["features.17.expand"]=new_layer
        elif 'features.17.expand.conv' == layer_name:
            self.features[17].expand.conv= new_layer
            self.origin_layer_names["features.17.expand.conv"]=new_layer
        elif 'features.17.expand.bn' == layer_name:
            self.features[17].expand.bn= new_layer
            self.origin_layer_names["features.17.expand.bn"]=new_layer
        elif 'features.17.expand.act' == layer_name:
            self.features[17].expand.act= new_layer
            self.origin_layer_names["features.17.expand.act"]=new_layer
        elif 'features.17.expand.act.act' == layer_name:
            self.features[17].expand.act.act= new_layer
            self.origin_layer_names["features.17.expand.act.act"]=new_layer
        elif 'features.17.conv1' == layer_name:
            self.features[17].conv1= new_layer
            self.origin_layer_names["features.17.conv1"]=new_layer
        elif 'features.17.conv1.conv' == layer_name:
            self.features[17].conv1.conv= new_layer
            self.origin_layer_names["features.17.conv1.conv"]=new_layer
        elif 'features.17.conv1.bn' == layer_name:
            self.features[17].conv1.bn= new_layer
            self.origin_layer_names["features.17.conv1.bn"]=new_layer
        elif 'features.17.conv1.act' == layer_name:
            self.features[17].conv1.act= new_layer
            self.origin_layer_names["features.17.conv1.act"]=new_layer
        elif 'features.17.conv1.act.act' == layer_name:
            self.features[17].conv1.act.act= new_layer
            self.origin_layer_names["features.17.conv1.act.act"]=new_layer
        elif 'features.17.se' == layer_name:
            self.features[17].se= new_layer
            self.origin_layer_names["features.17.se"]=new_layer
        elif 'features.17.se.pool' == layer_name:
            self.features[17].se.pool= new_layer
            self.origin_layer_names["features.17.se.pool"]=new_layer
        elif 'features.17.se.conv1' == layer_name:
            self.features[17].se.conv1= new_layer
            self.origin_layer_names["features.17.se.conv1"]=new_layer
        elif 'features.17.se.act1' == layer_name:
            self.features[17].se.act1= new_layer
            self.origin_layer_names["features.17.se.act1"]=new_layer
        elif 'features.17.se.act1.act' == layer_name:
            self.features[17].se.act1.act= new_layer
            self.origin_layer_names["features.17.se.act1.act"]=new_layer
        elif 'features.17.se.conv2' == layer_name:
            self.features[17].se.conv2= new_layer
            self.origin_layer_names["features.17.se.conv2"]=new_layer
        elif 'features.17.se.act2' == layer_name:
            self.features[17].se.act2= new_layer
            self.origin_layer_names["features.17.se.act2"]=new_layer
        elif 'features.17.se.act2.act' == layer_name:
            self.features[17].se.act2.act= new_layer
            self.origin_layer_names["features.17.se.act2.act"]=new_layer
        elif 'features.17.conv2' == layer_name:
            self.features[17].conv2= new_layer
            self.origin_layer_names["features.17.conv2"]=new_layer
        elif 'features.17.conv2.conv' == layer_name:
            self.features[17].conv2.conv= new_layer
            self.origin_layer_names["features.17.conv2.conv"]=new_layer
        elif 'features.17.conv2.bn' == layer_name:
            self.features[17].conv2.bn= new_layer
            self.origin_layer_names["features.17.conv2.bn"]=new_layer
        elif 'features.18' == layer_name:
            self.features[18]= new_layer
            self.origin_layer_names["features.18"]=new_layer
        elif 'features.19' == layer_name:
            self.features[19]= new_layer
            self.origin_layer_names["features.19"]=new_layer
        elif 'features.20' == layer_name:
            self.features[20]= new_layer
            self.origin_layer_names["features.20"]=new_layer
        elif 'features.20.act' == layer_name:
            self.features[20].act= new_layer
            self.origin_layer_names["features.20.act"]=new_layer
        elif 'features.21' == layer_name:
            self.features[21]= new_layer
            self.origin_layer_names["features.21"]=new_layer
        elif 'features.22' == layer_name:
            self.features[22]= new_layer
            self.origin_layer_names["features.22"]=new_layer
        elif 'features.23' == layer_name:
            self.features[23]= new_layer
            self.origin_layer_names["features.23"]=new_layer
        elif 'features.23.act' == layer_name:
            self.features[23].act= new_layer
            self.origin_layer_names["features.23.act"]=new_layer
        elif 'output' == layer_name:
            self.output= new_layer
            self.origin_layer_names["output"]=new_layer

    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name, out):
        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name] = out

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

    def set_Basic_OPS(self, b):
        self.Basic_OPS = b

    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self, c):
        self.Cascade_OPs = c


def mobilenet_v3(model_name, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model_cfgs = {
        "large": {
            "cfg": [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hswish', 2],
                [3, 200, 80, False, 'hswish', 1],
                [3, 184, 80, False, 'hswish', 1],
                [3, 184, 80, False, 'hswish', 1],
                [3, 480, 112, True, 'hswish', 1],
                [3, 672, 112, True, 'hswish', 1],
                [5, 672, 160, True, 'hswish', 2],
                [5, 960, 160, True, 'hswish', 1],
                [5, 960, 160, True, 'hswish', 1]],
            "cls_ch_squeeze": 960,
            "cls_ch_expand": 1280,
        },
        "small": {
            "cfg": [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hswish', 2],
                [5, 240, 40, True, 'hswish', 1],
                [5, 240, 40, True, 'hswish', 1],
                [5, 120, 48, True, 'hswish', 1],
                [5, 144, 48, True, 'hswish', 1],
                [5, 288, 96, True, 'hswish', 2],
                [5, 576, 96, True, 'hswish', 1],
                [5, 576, 96, True, 'hswish', 1]],
            "cls_ch_squeeze": 576,
            "cls_ch_expand": 1280,
        }
    }
    return MobileNetV3(model_cfgs[model_name], **kwargs)


if __name__ == '__main__':
    mobilenet_v3_large = partial(mobilenet_v3, model_name="large")
    mobilenet_v3_small = partial(mobilenet_v3, model_name="small")
    model = mobilenet_v3_large()
    # # t = np.random.randn(1, 3, 224, 224)
    # # t = mindspore.Tensor(t, dtype=mindspore.float32)
    # # print(len(model(t)))
    #
    # layers = model.cells_and_names()
    # for layer in layers:
    #     print(layer[0])
    # from models.summary import summary
    #
    net = mobilenet_v3_large()
    # summary(net, (3, 640, 640))
    #
    # from models.get_result import get_son, get_outshape, get_order
    # get_order("m3son.txt", "layer_info/mobilenetv3_order.txt")

    from models.get_result import get_order, get_outshape, get_input, write_layernames, write_setmethod, \
        write_setoriginalmethod
    #
    # write_layernames(net, "mobilenetv3")
    write_setmethod(net, "mobilenetv3")
    write_setoriginalmethod(net, "mobilenetv3")
