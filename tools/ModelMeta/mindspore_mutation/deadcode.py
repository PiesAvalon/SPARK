"""
deadcode1:SELayer —— ReLU() Hardsigmoid()
deadcode2:DenseLayer —— ReLU() 
deadcode3:Inception_A —— ReLU() AvgPool2d()
deadcode4:PWDWPW_ResidualBlock —— ReLU6() 
deadcode5:ResidualBlock —— ReLU() 
deadcode6:DropPath —— 无
deadcode7:Dense —— 无
"""
import collections
import mindspore
import numpy as np
from mindspore import nn, ops as ops
from mindspore.ops import operations as P
import mindspore.context as context
import mindspore as ms
from mindspore.rewrite import SymbolTree
from mindspore.rewrite import ScopedValue, NodeType
from mindspore.rewrite.node import Node, NodeManager

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")

# deadcode1
class GlobalAvgPooling(nn.Cell):
    """
    Global avg pooling definition.
    """

    def __init__(self, keep_dims=False):
        super(GlobalAvgPooling, self).__init__()
        self.keep_dims = keep_dims
        self.mean = ops.mean

    def construct(self, x):
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        x = self.mean(x, (2, 3), self.keep_dims)
        x = ops.cast(x, dtype)
        return x
# 1
class SELayer(nn.Cell):
    """
    SE warpper definition.

    Args:
        num_out (int): Numbers of output channels.
        ratio (int): middle output ratio.

    Returns:
        Tensor, output tensor.

    """

    def __init__(self, ratio=1):
        super(SELayer, self).__init__()
        self.ratio = ratio
        self.SE_pool = GlobalAvgPooling(keep_dims=True)
        self.SE_act1 = self.Activation('relu')
        self.SE_act2 = self.Activation('hsigmoid')
        self.SE_mul = ops.Mul()

    @staticmethod
    def _make_divisible(x, divisor=8):
        return int(np.ceil(x * 1. / divisor) * divisor)

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
        return self.act

    def construct(self, x):
        out = self.SE_pool(x)
        # print("out shape:", out.shape)
        dtype = out.dtype
        conv2out = mindspore.Tensor(np.random.randn(out.shape[0], out.shape[1], 1, 1).astype(np.float32),
                                    mindspore.float32)
        out = ops.cast(out, mindspore.float32)
        out = ops.conv2d(out, weight=conv2out,
                         pad_mode='pad')
        out = ops.cast(out, dtype)
        # print("new out shape:", out.shape)
        out = self.SE_act1(out)
        dtype = out.dtype
        out = ops.cast(out, mindspore.float32)
        conv2out_1 = mindspore.Tensor(np.random.randn(out.shape[0], out.shape[1], 1, 1).astype(np.float32),
                                      mindspore.float32)
        out = ops.conv2d(out, weight=conv2out_1,
                         pad_mode='pad')
        out = ops.cast(out, dtype)
        out = self.SE_act2(out)
        # out = self.SE_mul(x, out)
        return out


# 2
class DenseLayer(nn.Cell):
    def __init__(self):
        super(DenseLayer, self).__init__()
        self.drop_rate = 0.5
        self.relu = nn.ReLU()
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        in_shape = x.shape[1]
        new_features = nn.BatchNorm2d(in_shape)(x)
        new_features = self.relu(new_features)
        feature_weight = mindspore.Tensor(
            np.random.randn(new_features.shape[0], new_features.shape[1], 1, 1).astype(np.float32))
        new_features = ops.conv2d(new_features, feature_weight, stride=1, pad_mode="same")
        in_shape_1 = new_features.shape[1]
        new_features = nn.BatchNorm2d(in_shape_1)(new_features)
        new_features = self.relu_1(new_features)
        feature_weight_1 = mindspore.Tensor(
            np.random.randn(new_features.shape[0], new_features.shape[1], 1, 1).astype(np.float32))
        new_features = ops.conv2d(new_features, feature_weight_1, stride=1, pad_mode="same")
        # print("new_features shape:", new_features.shape)
        if self.drop_rate > 0:
            new_features = nn.Dropout(p=self.drop_rate)(new_features)
        new_features = ops.cast(new_features, dtype)
        x = ops.cast(x, dtype)
        return ops.Concat(1)([x, new_features])

# x = mindspore.Tensor(np.random.randn(3, 3, 5, 6).astype(np.float32))
# gap = DenseLayer()
# print(gap(x).shape)

# 3
class BasicConv2d(nn.Cell):
    def __init__(self):
        super(BasicConv2d, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        feature_weight = mindspore.Tensor(
            np.random.randn(x.shape[0], x.shape[1], 1, 1).astype(np.float32))
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        x = ops.conv2d(x, feature_weight, stride=1, pad_mode="same")
        in_shape_1 = x.shape[1]
        x = nn.BatchNorm2d(in_shape_1)(x)
        x = self.relu(x)
        x = ops.cast(x, dtype)
        return x

# 3
class Inception_A(nn.Cell):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = BasicConv2d()
        self.branch1 = nn.SequentialCell([
            BasicConv2d(),
            BasicConv2d()])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(),
            BasicConv2d(),
            BasicConv2d()])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=1),
            BasicConv2d()])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        # print("x.shape", x.shape)
        branch_pool = self.branch_pool(x)
        branch_pool = ops.cast(branch_pool, dtype)
        # print("branch_pool.shape", branch_pool.shape)
        out = self.concat((x0, x1, x2, branch_pool))
        return out

# x = mindspore.Tensor(np.random.randn(3, 3, 32, 32).astype(np.float32))
# inp = Inception_A()
# print(inp(x).shape)

# 4

class dwpw_basic(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size, stride, depthwise, activation='relu6'):
        super(dwpw_basic, self).__init__()
        self.dwpw_conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode="same",
                                   group=1 if not depthwise else in_channel)
        self.dwpw_bn = nn.BatchNorm2d(out_channel)
        if activation:
            # self.dwpw_activation = nn.get_activation(activation)
            self.dwpw_activation = nn.ReLU6() # zgb 为对齐

    def construct(self, x):
        # print("iamhere")
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        x = self.dwpw_conv(x)
        # print("x before", np.isnan(x.asnumpy()).any())

        # x = self.dwpw_bn(x)
        # np.save("x_ms.npy", x.asnumpy())
        x = ops.batch_norm(x, running_mean=mindspore.numpy.randn(x.shape[1]),
                           running_var=mindspore.numpy.randn(x.shape[1]), momentum=0.9, eps=1e-5,
                           weight=mindspore.numpy.randn(x.shape[1]),
                           bias=mindspore.numpy.randn(x.shape[1]))
        # print("x after", np.isnan(x.asnumpy()).any())

        x = self.dwpw_activation(x)
        x = ops.cast(x, dtype)
        # print("x final", np.isnan(x.asnumpy()).any())
        return x
# 4
class PWDWPW_ResidualBlock(nn.Cell):
    """
    Pointwise - -Depthwise - -Pointwise - -Add
    """

    def __init__(self):
        super(PWDWPW_ResidualBlock, self).__init__()

        self.PDP_ResidualBlock_3 = None
        self.PDP_ResidualBlock_2 = None
        self.PDP_ResidualBlock_1 = None
        self.add = P.Add()

    def construct(self, x):
        identity = x
        in_channel = x.shape[1]
        self.PDP_ResidualBlock_1 = dwpw_basic(in_channel, in_channel, 1, 1, False, 'relu6')
        out1 = self.PDP_ResidualBlock_1(x)
        in_channel = out1.shape[1]

        self.PDP_ResidualBlock_2 = dwpw_basic(in_channel, in_channel, 1, 1, True, 'relu6')

        out2 = self.PDP_ResidualBlock_2(out1)
        in_channel = out2.shape[1]

        self.PDP_ResidualBlock_3 = dwpw_basic(in_channel, in_channel, 1, 1, False, 'relu6')

        out2 = self.PDP_ResidualBlock_3(out2)
        out = self.add(out2, identity)
        return out

# a = mindspore.Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
# res_block = PWDWPW_ResidualBlock()
# print(res_block(a).shape)

# 5

def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    """_conv3x3"""
    if res_base:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode="pad",
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=0,
        pad_mode="same",
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
            pad_mode="pad",
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=1,
        stride=stride,
        padding=0,
        pad_mode="same",
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
            pad_mode="pad",
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=7,
        stride=stride,
        padding=0,
        pad_mode="same",
    )


def _bn(channel, res_base=False):
    """_bn"""
    if res_base:
        return nn.BatchNorm2d(
            channel,
            eps=1e-5,
            momentum=0.1,
            gamma_init=1,
            beta_init=0,
            moving_mean_init=0,
            moving_var_init=1,
        )
    return nn.BatchNorm2d(
        channel,
        eps=1e-4,
        momentum=0.9,
        gamma_init=1,
        beta_init=0,
        moving_mean_init=0,
        moving_var_init=1,
    )


def _bn_last(channel):
    """_bn_last"""
    return nn.BatchNorm2d(
        channel,
        eps=1e-4,
        momentum=0.9,
        gamma_init=0,
        beta_init=0,
        moving_mean_init=0,
        moving_var_init=1,
    )


def _fc(in_channel, out_channel, use_se=False):
    """_fc"""
    return nn.Dense(
        in_channel, out_channel, has_bias=True, bias_init=0,  # weight_init=weight,
    )
# 5
class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.use_se = False
        self.se_block = False

        self.residual_relu1 = nn.ReLU()
        self.residual_relu2 = nn.ReLU()
        self.residual_relu3 = nn.ReLU()

        self.residual_down_sample_layer = None

    def construct(self, x):
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        identity = x
        in_channel = x.shape[1]
        self.residual_conv1 = _conv1x1(in_channel, in_channel, stride=1, use_se=self.use_se)
        out = self.residual_conv1(x)
        in_channel = out.shape[1]
        self.residual_bn1 = _bn(in_channel)

        out = self.residual_bn1(out)
        out = self.residual_relu1(out)
        in_channel = out.shape[1]
        self.residual_conv2 = _conv3x3(in_channel, in_channel, stride=1, use_se=self.use_se)
        out = self.residual_conv2(out)
        in_channel = out.shape[1]

        self.residual_bn2 = _bn(in_channel)

        out = self.residual_bn2(out)
        out = self.residual_relu2(out)
        in_channel = out.shape[1]

        self.residual_conv3 = _conv1x1(in_channel, in_channel, stride=1, use_se=self.use_se)

        out = self.residual_conv3(out)
        out_channel = out.shape[1]
        self.residual_bn3 = _bn(out_channel)
        out = self.residual_bn3(out)
        in_channel = out.shape[1]
        self.residual_down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, in_channel, 1,
                                                                      use_se=self.use_se), _bn(in_channel)])
        identity = self.residual_down_sample_layer(identity)
        out = out + identity
        out = self.residual_relu3(out)
        out = ops.cast(out, dtype)
        return out

# a = mindspore.Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
# res_block = ResidualBlock()
# print(res_block(a).shape)

# 6
class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.5, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0)  # always be 0
        self.rand = P.UniformReal(seed=seed)  # seed must be 0, if set to other value, it's not rand for multiple call
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)  # B N C
            random_tensor = self.rand((x_shape[0], 1, 1)) if len(x_shape) == 3 else self.rand((x_shape[0], 1, 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x

# a = mindspore.Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
# drop_path = DropPath()
# print(drop_path(a).shape)
# (1, 3, 224, 224)

class op_mul(nn.Cell):
    def __init__(self):
        super(op_mul, self).__init__()
        ops_mul = ops.Mul()

    def construct(self, deada, for_matmul_edge):
        return ops_mul(deada, for_matmul_edge)
# 7
class Dense(nn.Cell):
    def __init__(self):
        super(Dense, self).__init__()

    def construct(self, deada):
        feature_a = deada.shape[-2]
        feature_b = deada.shape[-1]
        for_matmul_edge = mindspore.numpy.randn(feature_a, feature_b)
        matmul_edge = ops.Mul()(deada, for_matmul_edge)
        for_add_edge = mindspore.numpy.randn(feature_a, feature_b)
        add_edge = ops.Add()(matmul_edge, for_add_edge)
        return add_edge

# a = mindspore.Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
# dense = Dense()
# print(dense(a).shape)
# (1, 3, 224, 224)

banned_ops = [mindspore.ops.operations.array_ops.Shape,
              mindspore.ops.operations.array_ops.Concat,
              type(None)
              ]
banned_cell = [mindspore.nn.layer.CentralCrop, ]
banned_trees = [mindspore.ops.ResizeBilinearV2, # 版本不一致？
                mindspore.ops.operations.Shape,
                type(None)
                ]
# hash_table：一个用于记录节点访问次数的字典，防止重复处理节点。 变化
# nodedict：用于存储符合特定条件的节点（即 CallCell、CallPrimitive 等）节点的标识符作为键，节点所属的符号树作为值
# depth：记录当前递归的深度
# def scan_node(stree, hash_table, nodedict=None, depth=0):
#     # global hash_table
#     # for node in stree.nodes(all_nodes=False):
#     if type(stree) == mindspore.rewrite.api.symbol_tree.SymbolTree:
#         stree = stree._symbol_tree
#     for node in stree.all_nodes():
#         if isinstance(node, NodeManager):
#             for sub_node in node.get_tree_nodes():
#                 subtree = sub_node.symbol_tree
#                 scan_node(subtree, hash_table, nodedict=nodedict, depth=depth + 1)
#         if (node.get_node_type() == NodeType.CallCell and node.get_instance_type() not in banned_cell) or (
#                 node.get_node_type() == NodeType.CallPrimitive and node.get_instance_type() not in banned_ops) \
#                 or (node.get_node_type() == NodeType.Tree and node.get_instance_type() not in banned_trees) \
#                 or node.get_node_type() == NodeType.CellContainer:
#             if hash_table[mindspore.rewrite.api.node.Node(node).get_handler()] == 1:
#                 continue
#             hash_table[mindspore.rewrite.api.node.Node(node).get_handler()] += 1
#             if node.get_node_type() not in [NodeType.CellContainer, NodeType.Tree]:
#                 nodedict[mindspore.rewrite.api.node.Node(node).get_handler()] = node.get_belong_symbol_tree()
#     return True,nodedict




class MyConvPoolLayerSameShape(nn.Cell):
    """
    使用 Conv1d/2d/3d + Pool，但最后输出形状与输入相同 [N, C, W] 的示例。
    关键：1D、2D、3D 池化时 kernel_size=1, stride=1 不进行下采样；卷积 stride=1, pad_mode='same'；
    在使用 2D/3D 卷积前后，需要用 ops.ExpandDims / ops.Squeeze 进行升/降维。
    """
    def __init__(self, 
                 channels=1, 
                 kernel_size=3, 
                 stride=1, 
                 pad_mode='same'):
        super(MyConvPoolLayerSameShape, self).__init__()

        # ------------ 1D 卷积与池化 ------------
        # 让 in_channels = out_channels = channels，不改变通道数
        self.conv1d = nn.Conv1d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                pad_mode=pad_mode)
        # 若想保持 W 不变，就让 kernel_size=1, stride=1
        self.avgpool1d = nn.AvgPool1d(kernel_size=1, stride=1)
        self.relu1d = nn.ReLU()

        # ------------ 2D 卷积与池化 ------------
        self.conv2d = nn.Conv2d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                pad_mode=pad_mode)
        # 同理，为避免高度宽度缩小，kernel_size=(1,1), stride=(1,1)
        self.avgpool2d = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.relu2d = nn.ReLU()

        # ------------ 3D 卷积与池化 ------------
        self.conv3d = nn.Conv3d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                pad_mode=pad_mode)
        # 不在 D、H、W 上下采样 => (1,1,1)
        self.avgpool3d = nn.AvgPool3d(kernel_size=(1,1,1), stride=(1,1,1))
        self.relu3d = nn.ReLU()

        # 这里可选加一个 dropout (并不会改变 shape)
        self.dropout = nn.Dropout(0.5)

    def construct(self, x):
        """
        x.shape = [N, C, W]，其中 N=batch_size, C=通道, W=宽度(时序长度等)。
        整个网络经过 1D->2D->3D->再 squeeze 回来，最后保持与 x 相同的形状。
        """

        # ========== 阶段1: 1D 卷积 + ReLU + Pool1D =============
        # shape 仍然是 [N, C, W]
        x = ops.Squeeze(0)(x)
        out = self.conv1d(x)
        out = self.relu1d(out)
        out = self.avgpool1d(out)  # kernel_size=1, stride=1 => 不变
        # out shape: [N, C, W]

        # ========== 阶段2: 升维 => 2D 卷积 + ReLU + Pool2D => 降维 =============
        # 2D 卷积需要 [N, C, H, W]，这里把 H=1
        out = ops.ExpandDims()(out, 2)  # [N, C, 1, W]

        out = self.conv2d(out)
        out = self.relu2d(out)
        out = self.avgpool2d(out)  # kernel_size=(1,1), stride=(1,1) => 不变
        # out shape: [N, C, 1, W]

        # 再 squeeze 掉维度 2 => 回到 [N, C, W]
        out = ops.Squeeze(2)(out)

        # ========== 阶段3: 升维 => 3D 卷积 + ReLU + Pool3D => 降维 =============
        # 3D 卷积需要 [N, C, D, H, W]，这里让 D=1, H=1
        out = ops.ExpandDims()(out, 2)  # [N, C, 1, W] => [N, C, 1, 1, W]
        out = ops.ExpandDims()(out, 3)  # => [N, C, 1, 1, W]

        out = self.conv3d(out)
        out = self.relu3d(out)
        out = self.avgpool3d(out)  # kernel_size=(1,1,1), stride=(1,1,1) => 不变
        out = self.dropout(out)
        # out shape: [N, C, 1, 1, W]

        # squeeze 掉 D=1 和 H=1 => [N, C, W]
        out = ops.Squeeze(2)(out)  # squeeze第2个维度 => [N, C, 1, W]
        out = ops.Squeeze(2)(out)  # 再次 squeeze => [N, C, W]

        # 现在 out 形状与 x 相同
        return out

# net = MyConvPoolLayerSameShape()

# # 构造一个测试输入, 形状 [N, C, W] = [2, 1, 16]
# x = Tensor(np.random.randn(2, 1, 16).astype(np.float32))
# y = net(x)
# print("输出形状:", y.shape)


class MyAdaptiveMaxPoolLayer(nn.Cell):
    """
    演示将 AvgPoolXd 改成 AdaptiveMaxPoolXd 并保持输入输出形状相同的示例。
    假设输入固定为 [N, C, 16]，则中间会升维变成 [N, C, 1, 16] (2D) 和 [N, C, 1, 1, 16] (3D)。
    我们在每个阶段使用 AdaptiveMaxPoolXd(output_size=...)，让输出形状恢复到对应大小。
    """
    def __init__(self, 
                 channels=1,
                 kernel_size=3, 
                 stride=1, 
                 pad_mode='same', 
                 # 假设固定输入宽度=16，若实际业务需要不同宽度，请修改该值
                 fixed_width=16
                 ):
        super(MyAdaptiveMaxPoolLayer, self).__init__()

        # ------------- 1D 卷积 + AdaptiveMaxPool1D -------------
        # 卷积：保持 in_channels=out_channels=channels，不改变通道数
        self.conv1d = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            pad_mode=pad_mode
        )
        self.relu1d = nn.ReLU()
        # 假设输入宽度=16，希望输出也保持 16
        self.adapool1d = nn.AdaptiveMaxPool1d(output_size=fixed_width)

        # ------------- 2D 卷积 + AdaptiveMaxPool2D -------------
        self.conv2d = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            pad_mode=pad_mode
        )
        self.relu2d = nn.ReLU()
        # 2D 输入是 [N, C, 1, W] => 希望输出仍是 [N, C, 1, W]
        # 因此 (H_out, W_out) = (1, fixed_width)
        self.adapool2d = nn.AdaptiveMaxPool2d(output_size=(1, fixed_width))

        # ------------- 3D 卷积 + AdaptiveMaxPool3D -------------
        self.conv3d = nn.Conv3d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            pad_mode=pad_mode
        )
        self.relu3d = nn.ReLU()
        # 3D 输入是 [N, C, 1, 1, W] => 希望输出仍是 [N, C, 1, 1, W]
        # 因此 (D_out, H_out, W_out) = (1, 1, fixed_width)
        self.adapool3d = nn.AdaptiveMaxPool3d(output_size=(1, 1, fixed_width))

        self.dropout = nn.Dropout(0.5)

    def construct(self, x):
        """
        x: 假设固定 shape=[N, C, 16]。
        最终希望输出 shape=[N, C, 16]，与输入保持相同尺寸。
        """
        # ========== 阶段1: Conv1d + ReLU + AdaptiveMaxPool1d ==========
        x = ops.Squeeze(0)(x)
        out = self.conv1d(x)         # [N, C, 16] -> [N, C, 16] (pad_mode='same', stride=1)
        out = self.relu1d(out)       # 不变
        # Pool1d => [N, C, 16], 因为 output_size=16
        out = self.adapool1d(out)

        # ========== 阶段2: 升维 -> Conv2d + ReLU + AdaptiveMaxPool2d -> 降维 ==========
        # [N, C, 16] -> [N, C, 1, 16]
        out = ops.ExpandDims()(out, 2)

        out = self.conv2d(out)       # [N, C, 1, 16] -> [N, C, 1, 16] (same padding)
        out = self.relu2d(out)
        # AdaptiveMaxPool2d(output_size=(1, 16)) => 输出仍 [N, C, 1, 16]
        out = self.adapool2d(out)

        # squeeze -> [N, C, 16]
        out = ops.Squeeze(2)(out)

        # ========== 阶段3: 再升维 -> Conv3d + ReLU + AdaptiveMaxPool3d -> 降维 ==========
        # [N, C, 16] -> [N, C, 1, 16] -> [N, C, 1, 1, 16]
        out = ops.ExpandDims()(out, 2)  # [N, C, 1, 16]
        out = ops.ExpandDims()(out, 3)  # [N, C, 1, 1, 16]

        out = self.conv3d(out)         # => [N, C, 1, 1, 16]
        out = self.relu3d(out)
        # AdaptiveMaxPool3d(output_size=(1,1,16)) => [N, C, 1, 1, 16]
        out = self.adapool3d(out)

        out = self.dropout(out)

        # squeeze 回到 [N, C, 16]
        out = ops.Squeeze(2)(out)  # [N, C, 1, 16]
        out = ops.Squeeze(2)(out)  # [N, C, 16]

        return out

class MyTransposeConvLayer(nn.Cell):
    """
    使用 Conv1dTranspose, Conv2dTranspose, Conv3dTranspose，
    并保证最终输出形状与输入相同的示例。

    思路：
    1. 在 1D 阶段：输入 [N, C, W] -> 1D 转置卷积 -> [N, C, W]。
    2. 在 2D 阶段：先扩维 -> [N, C, 1, W] -> 2D 转置卷积 -> [N, C, 1, W] -> squeeze -> [N, C, W]。
    3. 在 3D 阶段：先扩两维 -> [N, C, 1, 1, W] -> 3D 转置卷积 -> [N, C, 1, 1, W] -> squeeze -> [N, C, W]。
    只要卷积配置 (kernel_size, stride, padding) 不导致尺寸变化，输入输出的 W 就能保持一致；若想通道也不变，in_channels=out_channels。
    """
    def __init__(self,
                 channels=1,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(MyTransposeConvLayer, self).__init__()
        
        # --- 1D 转置卷积 ---
        self.conv1dT = nn.Conv1dTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding
        )
        
        # --- 2D 转置卷积 ---
        self.conv2dT = nn.Conv2dTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding
        )
        
        # --- 3D 转置卷积 ---
        self.conv3dT = nn.Conv3dTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding
        )
        
        self.relu = nn.ReLU()
    
    def construct(self, x):
        """
        x 形状: [N, C, W]
        最终希望输出形状也为 [N, C, W].
        """
        # 1) 1D 转置卷积
        x = ops.Squeeze(0)(x)
        out = self.conv1dT(x)  # 若 kernel_size=3, stride=1, padding=1 => W 不变
        out = self.relu(out)   # 形状依然 [N, C, W]
        
        # 2) 升维 -> 2D 转置卷积 -> squeeze
        out = ops.ExpandDims()(out, 2)  # [N, C, 1, W]
        out = self.conv2dT(out)        # [N, C, 1, W] (形状不变, stride=1)
        out = self.relu(out)
        out = ops.Squeeze(2)(out)      # -> [N, C, W]
        
        # 3) 再升维 -> 3D 转置卷积 -> squeeze
        out = ops.ExpandDims()(out, 2)  # [N, C, 1, W] -> 变成 2D
        out = ops.ExpandDims()(out, 3)  # -> [N, C, 1, 1, W] (3D)
        
        out = self.conv3dT(out)        # 若 stride=1, kernel_size=3, padding=1 => [N, C, 1, 1, W]
        out = self.relu(out)
        
        # squeeze 掉那两个额外的维度 => 回到 [N, C, W]
        out = ops.Squeeze(2)(out)  # => [N, C, 1, W]
        out = ops.Squeeze(2)(out)  # => [N, C, W]
        
        # 此时 out 形状和 x 一样
        return out




def has_child_node(net, node_name):
    layers = net.cells_and_names()
    parent_node = None
    for name, _ in layers:
        if name == node_name:
            parent_node = name
            continue
        if parent_node is not None and name.startswith(parent_node + '.'):
            return True
    return False

if __name__ == '__main__':
    nodedict = collections.OrderedDict()  # 特殊字典，保持插入的顺序
    hash_table = collections.defaultdict(int)  # 每次访问一个不存在的键时，该键会自动被赋值为整数 0
    stree= SymbolTree.create(Dense())
    
    # for node in stree.nodes():
    #     print(node.get_name(),type(node))
        # for node1 in node.cells_and_names():
        #     print(node1.get_name())
    # flag, nodedict = scan_node(stree, hash_table, nodedict)

    # node_list = list(nodedict.keys())
    # all_instance_types = [node_info['instance_type'] for node_info in nodedict.values()]
    # print(set(all_instance_types))
    # for i in node_list:
    #     print(i.get_node_type())
    # in_channel = 64  # 假设输入通道数为 64
    # input_tensor = mindspore.Tensor(np.random.randn(1, in_channel, 224, 224), mindspore.float32)

    # # op_layer = PWDWPW_ResidualBlock()
    # # 创建 PWDWPW_ResidualBlock 实例
    op_layer = op_mul()

    # # 运行 construct 方法实例化所有子算子
    # op_layer.construct(input_tensor)

    # # 遍历并获取所有子算子
    # all_layers = []

    # # 遍历 op_layer 的属性，筛选出所有 nn.Cell 子模块
    # for name, layer in op_layer.__dict__.items():
    #     if isinstance(layer, nn.Cell):
    #         all_layers.append((name, layer))

    # # 打印出所有子层
    for name, layer in op_layer._cells.items():
        print(f"Layer Name: {name}, Layer: {layer}")
    # 使用 get_children() 方法遍历所有子层
    # for name, child in op_layer.cells_and_names():
    #     if not has_child_node(op_layer,name) and not name == '' and not 'deadcode' in str(type(child)):
    #         print(f"Layer Name: {name}, Layer: {child}")

"""
deadcode1:SELayer —— ReLU() Hardsigmoid() /
{<class 'mindspore.ops.operations.manually_defined.ops_def.Cast'>, <class 'mindspore.ops.operations.nn_ops.Conv2D'>, <class 'mindspore.nn.layer.activation.HSigmoid'>, <class 'mindspore.nn.layer.activation.ReLU'>, <class 'mindspore.ops.auto_generate.gen_ops_prim.ReduceMean'>, <class 'mindspore.ops.auto_generate.gen_ops_prim.BiasAdd'>
deadcode2:DenseLayer —— ReLU() /
{<class 'mindspore.ops.operations.manually_defined.ops_def.Cast'>, <class 'mindspore.ops.auto_generate.gen_ops_prim.BiasAdd'>, <class 'mindspore.nn.layer.activation.ReLU'>, <class 'mindspore.ops.operations.nn_ops.Conv2D'>}
deadcode3:Inception_A —— ReLU() AvgPool2d() /
{<class 'mindspore.ops.auto_generate.gen_ops_prim.BiasAdd'>, <class 'mindspore.nn.layer.activation.ReLU'>, <class 'mindspore.nn.layer.pooling.AvgPool2d'>, <class 'mindspore.ops.operations.nn_ops.Conv2D'>, <class 'mindspore.ops.operations.manually_defined.ops_def.Cast'>}
deadcode4:PWDWPW_ResidualBlock —— ReLU6() /
{<class 'mindspore.ops.auto_generate.gen_ops_prim.Add'>}，没有解决
deadcode5:ResidualBlock —— ReLU() /
{<class 'mindspore.ops.operations.manually_defined.ops_def.Cast'>, <class 'mindspore.nn.layer.activation.ReLU'>}
deadcode6:DropPath —— 无 / <class 'mindspore.ops.auto_generate.gen_ops_prim.Floor'>
deadcode7:Dense —— 无 / 无
"""