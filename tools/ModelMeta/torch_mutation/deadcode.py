import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.fx

"""
deadcode1:SELayer —— ReLU() Hardsigmoid()
deadcode2:DenseLayer —— ReLU() 
deadcode3:Inception_A —— ReLU() AvgPool2d()
deadcode4:PWDWPW_ResidualBlock —— ReLU6() 
deadcode5:ResidualBlock —— ReLU() 
deadcode6:DropPath —— 无
deadcode7:Dense —— 无
"""

## deadcode1
@torch.fx.wrap
def foo(out):
    return torch.randn(out.shape[0], out.shape[1], 1, 1).float().to(out.device)

class GlobalAvgPooling(nn.Module):
    """
    Global avg pooling definition.
    """

    def __init__(self, keep_dims=False):
        super(GlobalAvgPooling, self).__init__()
        self.keep_dims = keep_dims

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        if self.keep_dims:
            out = F.adaptive_avg_pool2d(x, (1, 1))
        else:
            out = F.adaptive_avg_pool2d(x, (1, 1)).view(x.shape[0], -1)
        out = out.to(dtype)
        return out

class SELayer(nn.Module):
    """
    SE wrapper definition.

    Args:
        ratio (int): middle output ratio.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, ratio=1):
        super(SELayer, self).__init__()
        self.SE_pool = GlobalAvgPooling(keep_dims=True)
        self.SE_act1 = self.Activation('relu')
        self.SE_act2 = self.Activation('hsigmoid')

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
        dtype = x.dtype
        x = x.to(torch.float32)
        out = self.SE_pool(x)
        out = F.conv2d(out, weight=foo(out))
        out = self.SE_act1(out)
        out = F.conv2d(out, weight=foo(out))
        out = self.SE_act2(out)
        out = out.to(dtype)
        return out

## deadcode2

# 包装函数，使其不会被 TorchScript 追踪
@torch.fx.wrap
def create_batch_norm_params(size, device):
    return (torch.zeros(size, device=device), torch.zeros(size, device=device))


@torch.fx.wrap
def create_feature_weight(shape, device):
    return torch.randn(shape[1], shape[1], 1, 1).float().to(device)


class DenseLayer(nn.Module):
    def __init__(self):
        super(DenseLayer, self).__init__()
        self.drop_rate = 0.5
        self.relu = nn.ReLU()
        self.relu_1 = nn.ReLU()

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)

        # 第一次 BatchNorm 和 ReLU
        running_mean, running_var = create_batch_norm_params(x.shape[1], x.device)
        new_features = F.batch_norm(x, running_mean=running_mean,
                                    running_var=running_var,
                                    weight=None, bias=None, training=False)
        new_features = self.relu(new_features)

        # 第一次卷积
        feature_weight = create_feature_weight(new_features.shape, new_features.device)
        new_features = F.conv2d(new_features, weight=feature_weight, stride=1, padding=0)

        # 第二次 BatchNorm 和 ReLU
        running_mean, running_var = create_batch_norm_params(new_features.shape[1], new_features.device)
        new_features = F.batch_norm(new_features, running_mean=running_mean,
                                    running_var=running_var,
                                    weight=None, bias=None, training=False)
        new_features = self.relu_1(new_features)

        # 第二次卷积
        feature_weight_1 = create_feature_weight(new_features.shape, new_features.device)
        new_features = F.conv2d(new_features, weight=feature_weight_1, stride=1, padding=0)

        # Dropout
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=False)

        # 恢复原始数据类型，并将输入和新特征拼接在一起
        x = x.to(dtype)
        new_features = new_features.to(dtype)
        return torch.cat([x, new_features], dim=1)


## deadcode3

# 包装函数，使其不会被 TorchScript 追踪
@torch.fx.wrap
def create_feature_weight(shape, device):
    return torch.randn(shape[1], shape[1], 1, 1).float().to(device)


@torch.fx.wrap
def create_batch_norm_params(size, device):
    return torch.zeros(size, device=device), torch.zeros(size, device=device)


class BasicConv2d(nn.Module):
    def __init__(self):
        super(BasicConv2d, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)

        # 生成卷积权重
        feature_weight = create_feature_weight(x.shape, x.device)
        x = F.conv2d(x, weight=feature_weight, stride=1, padding=0)

        # 生成 BatchNorm 参数
        running_mean, running_var = create_batch_norm_params(x.shape[1], x.device)
        x = F.batch_norm(x, running_mean=running_mean,
                         running_var=running_var, training=False)

        x = self.relu(x)
        x = x.to(dtype)
        return x


class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d()
        self.branch1 = nn.Sequential(
            BasicConv2d(),
            BasicConv2d())
        self.branch2 = nn.Sequential(
            BasicConv2d(),
            BasicConv2d(),
            BasicConv2d())
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=1),
            BasicConv2d())

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)

        out = torch.cat((x0, x1, x2, branch_pool), dim=1)
        out = out.to(dtype)
        return out


## deadcode4
# 包装函数，使其不会被 TorchScript 追踪
@torch.fx.wrap
def create_weight_DWPWBasic(shape, device):
    return torch.randn(shape[1], shape[1], 1, 1).float().to(device)


@torch.fx.wrap
def create_batch_norm_params(size, device):
    return torch.ones(size, device=device), torch.ones(size, device=device)


class DWPWBasic(nn.Module):
    def __init__(self, activation='relu6'):
        super(DWPWBasic, self).__init__()
        self.dwpw_activation = nn.ReLU6()

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)

        # 生成卷积权重
        weight = create_weight_DWPWBasic(x.shape, x.device)
        x = F.conv2d(x, weight, stride=1, padding=0)

        # 生成 BatchNorm 参数
        running_mean, running_var = create_batch_norm_params(x.shape[1], x.device)
        x = F.batch_norm(x, running_mean=running_mean,
                         running_var=running_var, training=False)

        x = x.to(dtype)
        x = self.dwpw_activation(x)
        return x


class PWDWPW_ResidualBlock(nn.Module):
    """
    Pointwise - Depthwise - Pointwise - Add
    """

    def __init__(self):
        super(PWDWPW_ResidualBlock, self).__init__()
        self.PDP_ResidualBlock = DWPWBasic('relu6')

    def forward(self, x):
        identity = x
        out1 = self.PDP_ResidualBlock(x)
        out2 = self.PDP_ResidualBlock(out1)
        out2 = self.PDP_ResidualBlock(out2)
        out = torch.add(out2, identity)
        return out

## deadcode5

# 包装函数，使其不会被 TorchScript 追踪
@torch.fx.wrap
def create_weight(shape, kernel_size, device):
    return torch.randn(shape[1], shape[1], kernel_size, kernel_size).float().to(device)


@torch.fx.wrap
def create_batch_norm_params(size, device):
    return torch.zeros(size, device=device), torch.zeros(size, device=device)


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.residual_relu1 = nn.ReLU()
        self.residual_relu2 = nn.ReLU()
        self.residual_relu3 = nn.ReLU()

        self.residual_down_sample_layer = None

    def forward(self, x):
        identity = x
        dtype = x.dtype
        x = x.to(torch.float32)
        identity = identity.to(torch.float32)

        # 第一次卷积和 BatchNorm
        feature_weight = create_weight(x.shape, 1, x.device)
        out = F.conv2d(x, feature_weight, stride=1, padding=0)
        running_mean, running_var = create_batch_norm_params(out.shape[1], x.device)
        out = F.batch_norm(out, eps=1e-5, momentum=0.1, running_mean=running_mean,
                           running_var=running_var, training=False)
        out = self.residual_relu1(out)

        # 第二次卷积和 BatchNorm
        feature_weight = create_weight(out.shape, 3, out.device)
        out = F.conv2d(out, feature_weight, stride=1, padding=1)
        running_mean, running_var = create_batch_norm_params(out.shape[1], out.device)
        out = F.batch_norm(out, eps=1e-5, momentum=0.1, running_mean=running_mean,
                           running_var=running_var, training=False)
        out = self.residual_relu2(out)

        # 第三次卷积和 BatchNorm
        feature_weight = create_weight(out.shape, 1, out.device)
        out = F.conv2d(out, feature_weight, stride=1, padding=0)
        running_mean, running_var = create_batch_norm_params(out.shape[1], out.device)
        out = F.batch_norm(out, eps=1e-5, momentum=0.1, running_mean=running_mean,
                           running_var=running_var, training=False)

        # 对 identity 进行卷积和 BatchNorm
        feature_weight = create_weight(identity.shape, 1, identity.device)
        identity = F.conv2d(identity, feature_weight, stride=1, padding=0)
        running_mean, running_var = create_batch_norm_params(identity.shape[1], identity.device)
        identity = F.batch_norm(identity, eps=1e-5, momentum=0.1, running_mean=running_mean,
                                running_var=running_var, training=False)

        # 将 out 和 identity 相加，并应用 ReLU
        out = out + identity
        out = self.residual_relu3(out)
        out = out.to(dtype)
        return out

## deadcode6

# 包装函数，使其不会被 TorchScript 追踪
@torch.fx.wrap
def create_random_tensor_(shape, keep_prob, device):
    random_tensor = keep_prob + torch.rand(shape, dtype=torch.float32, device=device)
    random_tensor.floor_()  # binarize
    return random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.5):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        dtype = x.dtype
        if self.training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with data in any dimension

            # 生成随机张量并进行二值化
            random_tensor = create_random_tensor_(shape, keep_prob, x.device)

            # 计算输出
            output = x.div(keep_prob) * random_tensor
            output = output.to(dtype)
            return output
        return x


## deadcode7
# 包装函数，使其不会被 TorchScript 追踪
@torch.fx.wrap
def create_random_tensor(shape, device, dtype):
    return torch.randn(shape, dtype=dtype, device=device)


class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()

    def forward(self, deada):
        dtype = deada.dtype
        feature_a = deada.shape[-2]
        feature_b = deada.shape[-1]

        # 使用包装的函数创建随机张量
        for_matmul_edge = create_random_tensor((feature_a, feature_b), deada.device, torch.float32)
        matmul_edge = torch.mul(deada, for_matmul_edge)

        for_add_edge = create_random_tensor((feature_a, feature_b), deada.device, torch.float32)
        add_edge = torch.add(matmul_edge, for_add_edge)

        add_edge = add_edge.to(dtype)
        return add_edge

