import mindspore.ops

from common.mutation_ms.cascade_op_extension.endecoder import avaiable_encoder
import mindspore.nn as nn


class InvertedResidual_ConvBNReLU(nn.Cell):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(InvertedResidual_ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        in_channels = in_planes
        out_channels = out_planes
        if groups == 1:
            self.InvertedResidual_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad',
                                                   padding=padding)
        else:
            out_channels = in_planes
            self.InvertedResidual_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad',
                                                   padding=padding, group=in_channels)

        self.InvertedResidual_bn = nn.BatchNorm2d(out_planes)
        self.InvertedResidual_relu = nn.ReLU6()

    def construct(self, x):
        x = self.InvertedResidual_conv(x)
        x = self.InvertedResidual_bn(x)
        output = self.InvertedResidual_relu(x)
        return output



class InvertedResidual(nn.Cell):
    """
    Mobilenetv2 residual block definition.
    """

    def __init__(self, inp, oup, stride, expand_ratio=2):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        self.InvertedResidual_ConvBNReLU = InvertedResidual_ConvBNReLU(inp, hidden_dim, kernel_size=1)
        self.InvertedResidual_conv = nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False)
        self.InvertedResidual_bn = nn.BatchNorm2d(oup)
        self.add = mindspore.ops.Add()

    def construct(self, x):
        identity = x
        x = self.InvertedResidual_ConvBNReLU(x)
        x = self.InvertedResidual_conv(x)
        x = self.InvertedResidual_bn(x)
        out = self.add(identity, x)
        return out

def avaiable_InvertedResidual(**kwargs):
    layer = InvertedResidual(inp=kwargs['param1'], oup=kwargs['param2'],stride=kwargs['param3'])
    return layer

new_cascade_ops={
    'encoder': avaiable_encoder,
    "invertedResidual":avaiable_InvertedResidual
}