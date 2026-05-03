## rule2 适用于padding='same'的Conv2d层
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransLayer_rule2(nn.Module):
    def __init__(self, layer_2d):
        super(TransLayer_rule2, self).__init__()
        if not isinstance(layer_2d, nn.Conv2d):
            raise ValueError("This wrapper only supports Conv2d layers")

        # 修改传入的 Conv2d 层，将其 padding 参数设为 0
        self.layer_2d = nn.Conv2d(
            in_channels=layer_2d.in_channels,
            out_channels=layer_2d.out_channels,
            kernel_size=layer_2d.kernel_size,
            stride=layer_2d.stride,
            padding=0,  # 强制设置 padding 为 0
            dilation=layer_2d.dilation,
            bias=(layer_2d.bias is not None)
        )

        # 复制原始层的权重和偏置
        with torch.no_grad():
            self.layer_2d.weight.copy_(layer_2d.weight)
            if layer_2d.bias is not None:
                self.layer_2d.bias.copy_(layer_2d.bias)

    def forward(self, x):
        # 获取卷积核大小、步幅和膨胀系数
        kernel_size = self.layer_2d.kernel_size
        stride = self.layer_2d.stride
        dilation = self.layer_2d.dilation

        # 计算所需的填充大小
        padding_h = self._calculate_padding(x.shape[2], stride[0], kernel_size[0], dilation[0])
        padding_w = self._calculate_padding(x.shape[3], stride[1], kernel_size[1], dilation[1])

        # 使用F.pad对输入进行填充
        x = F.pad(x, (padding_w // 2, padding_w - padding_w // 2, padding_h // 2, padding_h - padding_h // 2))

        # 通过Conv2d层
        x = self.layer_2d(x)
        return x

    def _calculate_padding(self, input_size, stride, kernel_size, dilation):
        output_size = (input_size + stride - 1) // stride  # 向上取整
        total_padding = max((output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size, 0)
        return total_padding

if __name__ == "__main__" and False:
    # 示例：使用 SamePaddingConv2dWrapper 包装一个 Conv2d
    conv2d = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding='same')  # 使用 padding=0 的 Conv2d
    same_padding_conv_wrapper = TransLayer_rule2(conv2d)


    input_2d = torch.randn(1, 3, 224, 224)  # 输入为一个2D图像

    #变异前的结果
    ans=conv2d(input_2d)
    # print(ans)
    print(ans.shape)

    #变异后的结果
    # 使用包装后的层进行前向传播
    ans2=same_padding_conv_wrapper(input_2d)
    # print(ans2)
    print(ans.shape)

    # 计算不相等元素的布尔掩码
    inequality_mask = ans != ans2
    # 统计不相等元素的数量
    inequality_count = torch.sum(inequality_mask).item()
    print(inequality_count) ## 已验证，完全相等！！！！！！！！！！