## rule3 适用于Conv2d AvgPool2d MaxPool2d 2d->3d
import torch
import torch.nn as nn
import random
import numpy as np
import traceback

class TransLayer_rule3_Conv2d(nn.Module):
    def __init__(self, layer_2d):
        super(TransLayer_rule3_Conv2d, self).__init__()
        self.layer_2d = layer_2d
        self.layer_3d = None

        # 将参数转换为3D
        self.convert_to_3d()

    def convert_to_3d(self):
        layer_class = self.layer_2d.__class__.__name__
        argument = {} #权重 偏置
        # 获取原始层的参数
        for name, param in self.layer_2d.named_parameters(recurse=False):
            argument[name] = param

        if layer_class == "Conv2d":
            # 将 Conv2d 转换为 Conv3d
            new_kernel_size = (1, self.layer_2d.kernel_size[0], self.layer_2d.kernel_size[1])
            new_stride = (1, self.layer_2d.stride[0], self.layer_2d.stride[1])
            new_padding = (0, self.layer_2d.padding[0], self.layer_2d.padding[1])
            new_dilation = (1, self.layer_2d.dilation[0], self.layer_2d.dilation[1])

            self.layer_3d = nn.Conv3d(
                in_channels=self.layer_2d.in_channels,
                out_channels=self.layer_2d.out_channels,
                kernel_size=new_kernel_size,
                stride=new_stride,
                padding=new_padding,
                dilation=new_dilation,
                bias=(self.layer_2d.bias is not None)
            )

            # 扩展权重为3D
            with torch.no_grad():
                self.layer_3d.weight.copy_(self.layer_2d.weight.unsqueeze(2))  # 在第2维度增加一个长度为1的维度
                if self.layer_2d.bias is not None:
                    self.layer_3d.bias.copy_(self.layer_2d.bias)

        elif layer_class == "MaxPool2d":
            # 将 MaxPool2d 转换为 MaxPool3d
            new_kernel_size = (1, self.layer_2d.kernel_size, self.layer_2d.kernel_size) if isinstance(self.layer_2d.kernel_size, int) else (1, *self.layer_2d.kernel_size)
            new_stride = (1, self.layer_2d.stride, self.layer_2d.stride) if isinstance(self.layer_2d.stride, int) else (1, *self.layer_2d.stride)
            new_padding = (0, self.layer_2d.padding, self.layer_2d.padding) if isinstance(self.layer_2d.padding, int) else (0, *self.layer_2d.padding)

            self.layer_3d = nn.MaxPool3d(
                kernel_size=new_kernel_size,
                stride=new_stride,
                padding=new_padding,
                dilation=1,
                ceil_mode=self.layer_2d.ceil_mode
            )

        elif layer_class == "AvgPool2d":
            # 将 AvgPool2d 转换为 AvgPool3d
            new_kernel_size = (1, self.layer_2d.kernel_size, self.layer_2d.kernel_size) if isinstance(self.layer_2d.kernel_size, int) else (1, *self.layer_2d.kernel_size)
            new_stride = (1, self.layer_2d.stride, self.layer_2d.stride) if isinstance(self.layer_2d.stride, int) else (1, *self.layer_2d.stride)
            new_padding = (0, self.layer_2d.padding, self.layer_2d.padding) if isinstance(self.layer_2d.padding, int) else (0, *self.layer_2d.padding)

            self.layer_3d = nn.AvgPool3d(
                kernel_size=new_kernel_size,
                stride=new_stride,
                padding=new_padding
            )

        else:
            raise ValueError(f"Unsupported layer type: {layer_class}")

    def forward(self, x):
        x = x.unsqueeze(2)  # 在第二维度增加一个长度为1的维度
        x = self.layer_3d(x)
        x = x.squeeze(2)  # 删除第二维度的维度
        return x

class TransLayer_rule3_AvgPool2d(nn.Module):
    def __init__(self, layer_2d):
        super(TransLayer_rule3_AvgPool2d, self).__init__()
        self.layer_2d = layer_2d
        self.layer_3d = None

        # 将参数转换为3D
        self.convert_to_3d()

    def convert_to_3d(self):
        layer_class = self.layer_2d.__class__.__name__
        argument = {} #权重 偏置
        # 获取原始层的参数
        for name, param in self.layer_2d.named_parameters(recurse=False):
            argument[name] = param

        if layer_class == "Conv2d":
            # 将 Conv2d 转换为 Conv3d
            new_kernel_size = (1, self.layer_2d.kernel_size[0], self.layer_2d.kernel_size[1])
            new_stride = (1, self.layer_2d.stride[0], self.layer_2d.stride[1])
            new_padding = (0, self.layer_2d.padding[0], self.layer_2d.padding[1])
            new_dilation = (1, self.layer_2d.dilation[0], self.layer_2d.dilation[1])

            self.layer_3d = nn.Conv3d(
                in_channels=self.layer_2d.in_channels,
                out_channels=self.layer_2d.out_channels,
                kernel_size=new_kernel_size,
                stride=new_stride,
                padding=new_padding,
                dilation=new_dilation,
                bias=(self.layer_2d.bias is not None)
            )

            # 扩展权重为3D
            with torch.no_grad():
                self.layer_3d.weight.copy_(self.layer_2d.weight.unsqueeze(2))  # 在第2维度增加一个长度为1的维度
                if self.layer_2d.bias is not None:
                    self.layer_3d.bias.copy_(self.layer_2d.bias)

        elif layer_class == "MaxPool2d":
            # 将 MaxPool2d 转换为 MaxPool3d
            new_kernel_size = (1, self.layer_2d.kernel_size, self.layer_2d.kernel_size) if isinstance(self.layer_2d.kernel_size, int) else (1, *self.layer_2d.kernel_size)
            new_stride = (1, self.layer_2d.stride, self.layer_2d.stride) if isinstance(self.layer_2d.stride, int) else (1, *self.layer_2d.stride)
            new_padding = (0, self.layer_2d.padding, self.layer_2d.padding) if isinstance(self.layer_2d.padding, int) else (0, *self.layer_2d.padding)

            self.layer_3d = nn.MaxPool3d(
                kernel_size=new_kernel_size,
                stride=new_stride,
                padding=new_padding,
                dilation=1,
                ceil_mode=self.layer_2d.ceil_mode
            )

        elif layer_class == "AvgPool2d":
            # 将 AvgPool2d 转换为 AvgPool3d
            new_kernel_size = (1, self.layer_2d.kernel_size, self.layer_2d.kernel_size) if isinstance(self.layer_2d.kernel_size, int) else (1, *self.layer_2d.kernel_size)
            new_stride = (1, self.layer_2d.stride, self.layer_2d.stride) if isinstance(self.layer_2d.stride, int) else (1, *self.layer_2d.stride)
            new_padding = (0, self.layer_2d.padding, self.layer_2d.padding) if isinstance(self.layer_2d.padding, int) else (0, *self.layer_2d.padding)

            self.layer_3d = nn.AvgPool3d(
                kernel_size=new_kernel_size,
                stride=new_stride,
                padding=new_padding
            )

        else:
            raise ValueError(f"Unsupported layer type: {layer_class}")

    def forward(self, x):
        x = x.unsqueeze(2)  # 在第二维度增加一个长度为1的维度
        x = self.layer_3d(x)
        x = x.squeeze(2)  # 删除第二维度的维度
        return x

class TransLayer_rule3_MaxPool2d(nn.Module):
    def __init__(self, layer_2d):
        super(TransLayer_rule3_MaxPool2d, self).__init__()
        self.layer_2d = layer_2d
        self.layer_3d = None

        # 将参数转换为3D
        self.convert_to_3d()

    def convert_to_3d(self):
        layer_class = self.layer_2d.__class__.__name__
        argument = {} #权重 偏置
        # 获取原始层的参数
        for name, param in self.layer_2d.named_parameters(recurse=False):
            argument[name] = param

        if layer_class == "Conv2d":
            # 将 Conv2d 转换为 Conv3d
            new_kernel_size = (1, self.layer_2d.kernel_size[0], self.layer_2d.kernel_size[1])
            new_stride = (1, self.layer_2d.stride[0], self.layer_2d.stride[1])
            new_padding = (0, self.layer_2d.padding[0], self.layer_2d.padding[1])
            new_dilation = (1, self.layer_2d.dilation[0], self.layer_2d.dilation[1])

            self.layer_3d = nn.Conv3d(
                in_channels=self.layer_2d.in_channels,
                out_channels=self.layer_2d.out_channels,
                kernel_size=new_kernel_size,
                stride=new_stride,
                padding=new_padding,
                dilation=new_dilation,
                bias=(self.layer_2d.bias is not None)
            )

            # 扩展权重为3D
            with torch.no_grad():
                self.layer_3d.weight.copy_(self.layer_2d.weight.unsqueeze(2))  # 在第2维度增加一个长度为1的维度
                if self.layer_2d.bias is not None:
                    self.layer_3d.bias.copy_(self.layer_2d.bias)

        elif layer_class == "MaxPool2d":
            # 将 MaxPool2d 转换为 MaxPool3d
            new_kernel_size = (1, self.layer_2d.kernel_size, self.layer_2d.kernel_size) if isinstance(self.layer_2d.kernel_size, int) else (1, *self.layer_2d.kernel_size)
            new_stride = (1, self.layer_2d.stride, self.layer_2d.stride) if isinstance(self.layer_2d.stride, int) else (1, *self.layer_2d.stride)
            new_padding = (0, self.layer_2d.padding, self.layer_2d.padding) if isinstance(self.layer_2d.padding, int) else (0, *self.layer_2d.padding)

            self.layer_3d = nn.MaxPool3d(
                kernel_size=new_kernel_size,
                stride=new_stride,
                padding=new_padding,
                dilation=1,
                ceil_mode=self.layer_2d.ceil_mode
            )

        elif layer_class == "AvgPool2d":
            # 将 AvgPool2d 转换为 AvgPool3d
            new_kernel_size = (1, self.layer_2d.kernel_size, self.layer_2d.kernel_size) if isinstance(self.layer_2d.kernel_size, int) else (1, *self.layer_2d.kernel_size)
            new_stride = (1, self.layer_2d.stride, self.layer_2d.stride) if isinstance(self.layer_2d.stride, int) else (1, *self.layer_2d.stride)
            new_padding = (0, self.layer_2d.padding, self.layer_2d.padding) if isinstance(self.layer_2d.padding, int) else (0, *self.layer_2d.padding)

            self.layer_3d = nn.AvgPool3d(
                kernel_size=new_kernel_size,
                stride=new_stride,
                padding=new_padding
            )

        else:
            raise ValueError(f"Unsupported layer type: {layer_class}")

    def forward(self, x):
        x = x.unsqueeze(2)  # 在第二维度增加一个长度为1的维度
        x = self.layer_3d(x)
        x = x.squeeze(2)  # 删除第二维度的维度
        return x










if __name__ == "__main__" and False:

    # 示例：使用 Conv2dTo3dWrapper 包装一个 Conv2d
    conv2d = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    conv_wrapper = TransLayer_rule3_Conv2d(conv2d)
    print(conv_wrapper)

    # 示例：使用 Conv2dTo3dWrapper 包装一个 MaxPool2d
    maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    maxpool_wrapper = TransLayer_rule3_MaxPool2d(maxpool2d)
    print(maxpool_wrapper)

    # 示例：使用 Conv2dTo3dWrapper 包装一个 AvgPool2d
    avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    avgpool_wrapper = TransLayer_rule3_AvgPool2d(avgpool2d)
    print(avgpool_wrapper)


    # 示例输入
    input_2d = torch.randn(1, 3, 224, 224)  # 输入为一个2D图像

    # 使用包装后的层进行前向传播
    output = conv_wrapper(input_2d)
    print("Output shape after Conv2dTo3dWrapper (Conv2d):", output.shape)

    output1 = conv2d(input_2d)
    print("Original Conv2d output shape:", output1.shape)
    print(torch.allclose(output, output1, atol=1e-6))  # 允许一定的误差范围来比较结果

    output = maxpool_wrapper(input_2d)
    print("Output shape after Conv2dTo3dWrapper (MaxPool2d):", output.shape)

    output1 = maxpool2d(input_2d)
    print("Original MaxPool2d output shape:", output1.shape)
    print(torch.allclose(output, output1, atol=1e-6))  # 允许一定的误差范围来比较结果

    output = avgpool_wrapper(input_2d)
    print("Output shape after Conv2dTo3dWrapper (AvgPool2d):", output.shape)

    output1 = avgpool2d(input_2d)
    print("Original AvgPool2d output shape:", output1.shape)
    print(torch.allclose(output, output1, atol=1e-6))  # 允许一定的误差范围来比较结果

