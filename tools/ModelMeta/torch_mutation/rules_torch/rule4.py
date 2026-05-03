## rule4 nn.BatchNorm2d(in_features)
import torch
import torch.nn as nn

class TransLayer_rule4(nn.Module):
    def __init__(self, layer_1):
        super(TransLayer_rule4, self).__init__()
        # 使用现有 BatchNorm 层的属性来初始化
        self.num_features = layer_1.num_features
        self.eps = layer_1.eps
        self.momentum = layer_1.momentum
        self.affine = layer_1.affine
        self.track_running_stats = layer_1.track_running_stats

        if self.affine:
            self.weight = layer_1.weight
            self.bias = layer_1.bias

        if self.track_running_stats:
            self.register_buffer('running_mean', layer_1.running_mean)
            self.register_buffer('running_var', layer_1.running_var)

    def forward(self, input):
        reduce_dim = [0]  # Batch dimension
        i = 2
        while i < len(list(input.size())):  # Collect spatial dimensions for reduction
            reduce_dim.append(i)
            i += 1

        if self.training or not self.track_running_stats:
            mean = torch.mean(input, dim=reduce_dim)
            variance = torch.var(input, dim=reduce_dim, unbiased=False)
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance
        else:
            mean = self.running_mean
            variance = self.running_var

        shape = [1, -1] + [1] * (input.dim() - 2)  # Shape for broadcasting
        mean = mean.view(*shape)
        variance = variance.view(*shape)
        if self.affine:
            weight = self.weight.view(*shape)
            bias = self.bias.view(*shape)
        else:
            weight = 1.0
            bias = 0.0

        output = (input - mean) / torch.sqrt(variance + self.eps) * weight + bias
        return output

if __name__ == "__main__" and False:
    # 初始化 PyTorch 的 BatchNorm2d 层
    bn2 = nn.BatchNorm2d(3)
    bn2.eval()  # 设置为评估模式

    # 使用自定义的 BatchNorm2d 类
    custom_bn2 = TransLayer_rule4(bn2)
    custom_bn2.eval()  # 设置为评估模式

    # 创建示例输入张量
    input_tensor = torch.randn(8, 3, 32, 32)

    # 通过原始的 BatchNorm2d 层计算输出
    output_pytorch = bn2(input_tensor)

    # 通过自定义的 BatchNorm2d 层计算输出
    output_custom = custom_bn2(input_tensor)

    # 比较两个输出张量是否足够接近
    are_outputs_close = torch.allclose(output_pytorch, output_custom, atol=1e-6)
    print("Are the outputs close? ", are_outputs_close)

    print(bn2)
    print(custom_bn2)
