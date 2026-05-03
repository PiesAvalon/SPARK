import collections
import math
import os
import mindspore
import mindspore.nn as nn
import numpy as np
from infoplus.MindSporeInfoPlus import mindsporeinfoplus

np.random.seed(6)
mindspore.set_seed(6)
import torch

torch.manual_seed(6)
from mindspore import Tensor, ops, dataset as de, SymbolTree, Node, NodeType
from mindspore.dataset import vision, transforms as C
from scipy.stats import truncnorm
import mindspore.common.dtype as mstype


def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            neg_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            neg_slope = param
        else:
            raise ValueError("neg_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + neg_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Unsupported mode {}, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


# def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
#     fan = _calculate_correct_fan(inputs_shape, mode)
#     gain = calculate_gain(nonlinearity, a)
#     std = gain / math.sqrt(fan)
#     return np.random.normal(0, std, size=inputs_shape).astype(np.float32)
#
#
# def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
#     fan = _calculate_correct_fan(inputs_shape, mode)
#     gain = calculate_gain(nonlinearity, a)
#     std = gain / math.sqrt(fan)
#     bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
#     return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    # if use_se:
    #     weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=3)
    # else:
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = Tensor(np.zeros(weight_shape, dtype=np.float32))

    # if res_base:
    #     return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
    #                      padding=1, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    # if use_se:
    #     weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
    # else:
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = Tensor(np.zeros(weight_shape, dtype=np.float32))

    # if res_base:
    #     return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
    #                      padding=0, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    # if use_se:
    #     weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=7)
    # else:
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = Tensor(np.zeros(weight_shape, dtype=np.float32))

    # if res_base:
    #     return nn.Conv2d(in_channel, out_channel,
    #                      kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)
    # return nn.Conv2d(in_channel, out_channel,
    #                  kernel_size=7, stride=stride, padding=0, pad_mode='same')


def _bn(channel, res_base=False):
    # if res_base:
    #     return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.1)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9)


def _bn_last(channel):
    # return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
    #                       gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9)


def _fc(in_channel, out_channel, use_se=False):
    # if use_se:
    #     weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
    #     # weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32)
    # else:
    weight_shape = (out_channel, in_channel)
    weight = Tensor(np.zeros(weight_shape, dtype=np.float32))

    return nn.Dense(in_channel, out_channel, weight_init=weight)


class ResNet(nn.Cell):
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes,
                 use_se=False,
                 res_base=False):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
        self.use_se = use_se
        self.res_base = res_base
        self.se_block = False
        # if self.use_se:
        #     self.se_block = True

        # if self.use_se:
        #     self.conv1_0 = _conv3x3(3, 32, stride=2, use_se=self.use_se)
        #     self.bn1_0 = _bn(32)
        #     self.conv1_1 = _conv3x3(32, 32, stride=1, use_se=self.use_se)
        #     self.bn1_1 = _bn(32)
        #     self.conv1_2 = _conv3x3(32, 64, stride=1, use_se=self.use_se)
        # else:
        # self.conv1 = nn.Conv2d(3, 64,
        #                        kernel_size=7, stride=2, padding=0, pad_mode='same')
        # self.conv1 = nn.Conv2d(3, 64,
        #                        kernel_size=7, stride=2, padding=3, pad_mode='pad')
        weight_shape = (64, 3, 7, 7)
        weight = Tensor(np.zeros(weight_shape, dtype=np.float32))
        self.conv1 = nn.Conv2d(3, 64,
                               kernel_size=7, stride=2, padding=3, pad_mode='pad', weight_init=weight)
        self.bn1 = _bn(64, self.res_base)
        self.relu = nn.ReLU()

        # if self.res_base:
        #     self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        # else:
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       use_se=False)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       use_se=False)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2],
                                       use_se=False)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3],
                                       use_se=False)

        self.mean = ops.mean
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes, use_se=False)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, use_se=False):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            se_block(bool): Use se block in SE-ResNet50 net. Default: False.
        Returns:
            SequentialCell, the output layer.
        """
        layers = []
        resnet_block = block(in_channel, out_channel, stride=stride, use_se=use_se)
        layers.append(resnet_block)
        # if se_block:
        #     for _ in range(1, layer_num - 1):
        #         resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
        #         layers.append(resnet_block)
        #     resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
        #     layers.append(resnet_block)
        # else:
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        # if self.use_se:
        #     x = self.conv1_0(x)
        #     x = self.bn1_0(x)
        #     x = self.relu(x)
        #     x = self.conv1_1(x)
        #     x = self.bn1_1(x)
        #     x = self.relu(x)
        #     x = self.conv1_2(x)
        # else:
        # print("fucking here!!!!!!!!!!!!")
        x = self.conv1(x)
        # print("before bn1 shape: ", x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        # if self.res_base:
        #     x = self.pad(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3), keep_dims=True)
        out = self.flatten(out)
        out = self.end_point(out)
        return out
        # return c2


class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 use_se=False, se_block=False):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.se_block = se_block
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1, use_se=self.use_se)
        # print("channel: ", channel)
        self.bn1 = _bn(channel)
        if self.use_se and self.stride != 1:
            self.e2 = nn.SequentialCell([_conv3x3(channel, channel, stride=1, use_se=True), _bn(channel),
                                         nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')])
        else:
            self.conv2 = _conv3x3(channel, channel, stride=stride, use_se=self.use_se)
            self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1, use_se=self.use_se)
        # print("in_channel: ", in_channel, "out_channel: ", out_channel)
        self.bn3 = _bn(out_channel)

        if self.se_block:
            self.se_global_pool = ops.ReduceMean(keep_dims=False)
            self.se_dense_0 = _fc(out_channel, int(out_channel / 4), use_se=self.use_se)
            self.se_dense_1 = _fc(int(out_channel / 4), out_channel, use_se=self.use_se)
            self.se_sigmoid = nn.Sigmoid()
            self.se_mul = ops.Mul()
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        # if self.down_sample:
            # if self.use_se:
            #     if stride == 1:
            #         self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel,
            #                                                              stride, use_se=self.use_se), _bn(out_channel)])
            #     else:
            #         self.down_sample_layer = nn.SequentialCell([nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same'),
            #                                                     _conv1x1(in_channel, out_channel, 1,
            #                                                              use_se=self.use_se), _bn(out_channel)])
            # else:
            # self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
            #                                                          use_se=self.use_se), _bn(out_channel)])
        weight_shape = (out_channel, in_channel, 1, 1)
        weight = Tensor(np.zeros(weight_shape, dtype=np.float32))
        self._conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                                  padding=0, pad_mode='same', weight_init=weight)
        self._bn1 = nn.BatchNorm2d(out_channel, eps=1e-4, momentum=0.9)

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # if self.use_se and self.stride != 1:
        #     out = self.e2(out)
        # else:
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # if self.down_sample:
        identity1 = self._conv1x1(identity)
        identity2 = self._bn1(identity1)

        out = out + identity2
        out = self.relu(out)

        return out


def resnet50(class_num=10):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


def create_cifar10_dataset(data_home, image_size, batch_size, training=True):
    """Data operations."""
    data_dir = os.path.join(data_home, "cifar-10-batches-bin")
    if not training:
        data_dir = os.path.join(data_home, "cifar-10-verify-bin")
    sampler = de.SequentialSampler(num_samples=100)
    data_set = de.Cifar10Dataset(data_dir, sampler=sampler)

    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize(image_size)  # interpolation default BILINEAR
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set


def ChebyshevDistance(x, y):
    if isinstance(x, mindspore.Tensor):
        x = x.asnumpy()
    elif isinstance(x, torch.Tensor):
        if torch.get_device(x) != "CPU":
            x = x.cpu()
        x = x.detach().numpy()
    if isinstance(y, mindspore.Tensor):
        y = y.asnumpy()
    elif isinstance(y, torch.Tensor):
        if torch.get_device(y) != "CPU":
            y = y.cpu()
        y = y.detach().numpy()
    # x = x.asnumpy()
    # y = y.asnumpy()
    # try:
    out = np.max(np.abs(x - y))
    # except ValueError as e:
    # print(e)
    # out = e
    return out


def distance(x1, x2):
    distance_real = ChebyshevDistance
    dis = distance_real(x1, x2)
    return dis


def update_params(old_op, ans_dict):
    type_list = [bool, str, int, tuple, list, float, np.ndarray, Tensor]
    attrs_list = list(old_op.__dict__.items())
    edit_flag = False
    ans = {}
    for i in range(len(attrs_list)):
        if "grad_ops_label" in attrs_list[i][0]:
            edit_flag = True
            continue
        if edit_flag and "grad_ops_label" not in attrs_list[i][0]:
            if "Prim" in str(attrs_list[i][1]) and "<" in str(attrs_list[i][1]):
                edit_flag = False
                continue
            # print(type(getattr(old_op, attrs_list[i][0])))
            ans[attrs_list[i][0]] = getattr(old_op, attrs_list[i][0]) \
                if type(getattr(old_op, attrs_list[i][0])) in type_list else None
    if old_op.__class__.__name__ not in ans_dict.keys():
        ans_dict[old_op.__class__.__name__] = set()
    ans_dict[old_op.__class__.__name__].add(str(ans))


def calculate_layer_shape(model: nn.Cell, np_data: list, model_dtypes_ms: list):
    input_data = mindsporeinfoplus.np_2_tensor(np_data, model_dtypes_ms)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=model,
        input_data=input_data,
        dtypes=model_dtypes_ms,
        col_names=['input_size', 'output_size', 'name'],
        mode="train",
        verbose=0,
        depth=10
    )
    current_layer_shape_dict = mindsporeinfoplus.get_input_size(global_layer_info)
    shape_fenzi = 0
    for key in current_layer_shape_dict.keys():
        shape_fenzi += len(current_layer_shape_dict[key])
    return shape_fenzi


def calculate_layer_dtype(model: nn.Cell, np_data: list, model_dtypes_ms: list):
    input_data = mindsporeinfoplus.np_2_tensor(np_data, model_dtypes_ms)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=model,
        input_data=input_data,
        dtypes=model_dtypes_ms,
        col_names=['input_size', 'output_size', 'name'],
        mode="train",
        verbose=0,
        depth=10
    )
    current_layer_dtype_dict = mindsporeinfoplus.get_dtypes(global_layer_info)
    dtype_fenzi = 0
    for key in current_layer_dtype_dict.keys():
        dtype_fenzi += len(current_layer_dtype_dict[key])
    return dtype_fenzi


def calculate_layer_sequence(model: nn.Cell):
    current_layer_sequence_set = set()
    stree = SymbolTree.create(model)
    if mindspore.__version__ == "2.2.0":
        head_node = stree._symbol_tree.get_head()
    else:
        head_node = stree._symbol_tree.get_head_node()
    if head_node is None:
        print("head_node None, return")
        return 0
    node: Node = head_node.get_next()
    prev_layer = None
    while node is not None:
        if node.get_instance() is not None:
            if prev_layer is not None:
                current_layer_sequence_set.add((prev_layer, node.get_instance().__class__.__name__))
            prev_layer = node.get_instance().__class__.__name__
        node = node.get_next()
    return len(current_layer_sequence_set)


def calculate_op_num(model: nn.Cell):
    current_op_list = []
    for _, cell in model.cells_and_names():
        current_op_list.append(type(cell))
    return len(current_op_list)


def calculate_op_type(model: nn.Cell):
    current_op_set = set()
    stree = SymbolTree.create(model)
    if mindspore.__version__ == "2.2.0":
        head_node = stree._symbol_tree.get_head()
    else:
        head_node = stree._symbol_tree.get_head_node()
    if head_node is None:
        print("head_node None, return")
        return 0
    node: Node = head_node.get_next()
    while node is not None:
        if node.get_instance() is not None:
            current_op_set.add(node.get_instance().__class__.__name__)
        node = node.get_next()
    # print(current_op_set)
    return len(current_op_set)


def calculate_edge_num(model: nn.Cell):
    current_edge_list = []
    stree = SymbolTree.create(model)
    if mindspore.__version__ == "2.2.0":
        for in_node in stree.nodes(all_nodes=True):
            for out_node in in_node.get_users():
                current_edge_list.append(out_node.get_name())
    else:
        for in_node in stree.nodes():
            for out_node in in_node.get_users():
                current_edge_list.append(out_node.get_instance().__class__.__name__)
    return len(current_edge_list)


def calculate_all_coverage(model: nn.Cell, np_data: list, model_dtypes_ms: list):
    return calculate_layer_shape(model, np_data, model_dtypes_ms), calculate_layer_dtype \
        (model, np_data, model_dtypes_ms), calculate_layer_sequence(model), \
        calculate_op_num(model), calculate_op_type(model), calculate_edge_num(model)


if __name__ == '__main__':
    mindspore.set_context(device_target="GPU", device_id=7)
    net_ms = resnet50(10)
    inpu_np = np.ones([1, 3, 224, 224])
    np_data = [inpu_np]
    model_dtypes_ms = [mindspore.float32]
    inpu = mindspore.Tensor(inpu_np, mindspore.float32)
    print(net_ms(inpu).shape)
    from mindspore import _checkparam as Validator
    from mindspore.rewrite.node import NodeManager


    class MyTree(SymbolTree):
        def nodes(self, all_nodes: bool = False):
            Validator.check_value_type("all_nodes", all_nodes, [bool], "nodes")
            nodes = self.all_nodes_pzy() if all_nodes else self._symbol_tree.nodes()
            for node in nodes:
                yield Node(node)

        def all_nodes_pzy(self):
            """
            Get all nodes including nodes in CallFunction node, CellContainer node and sub symbol tree.

            Returns:
                A list of nodes.
            """
            nodes = []
            node_managers = [self._symbol_tree]
            for tree_node in self._symbol_tree.get_tree_nodes():
                stree = tree_node.symbol_tree
                nodes.extend(stree.all_nodes())
            while node_managers:
                node_manager = node_managers.pop()
                nodes.extend(node_manager.nodes())
                for node in node_manager.nodes():

                    if isinstance(node, NodeManager):
                        node_managers.append(node)

            return nodes


    stree = SymbolTree.create(net_ms)
    # print(type(stree))
    # print(type(stree._symbol_tree))
    # for tree_node in stree._symbol_tree.all_nodes():
    #     if isinstance(tree_node, NodeManager):
    #         print(tree_node.get_name())
    #         print([node.get_node_type() for node in tree_node.get_tree_nodes()])
    # print("len", len(stree._symbol_tree.all_nodes()))

    banned_ops = [mindspore.ops.operations.array_ops.Shape,
                  # mindspore.ops.operations.array_ops.Concat,
                  type(None)
                  ]
    banned_cell = [mindspore.nn.layer.CentralCrop, ]
    banned_trees = [mindspore.ops.ResizeBilinearV2,
                    mindspore.ops.operations.Shape,
                    type(None)
                    ]


    def scan_node(stree, hash_table, nodedict=None, depth=0):
        # global hash_table
        # for node in stree.nodes(all_nodes=False):
        if type(stree) == mindspore.rewrite.api.symbol_tree.SymbolTree:
            stree = stree._symbol_tree
        for node in stree.all_nodes():
            if isinstance(node, NodeManager):
                for sub_node in node.get_tree_nodes():
                    print("depth", depth)
                    print("node_to_sub", sub_node.get_name())
                    print("node_to_sub get_node_type", sub_node.get_node_type())
                    print("stree name", stree._ori_cls_name)
                    subtree = sub_node.symbol_tree
                    scan_node(subtree, hash_table, nodedict=nodedict, depth=depth + 1)
            if (node.get_node_type() == NodeType.CallCell and node.get_instance_type() not in banned_cell) or (
                    node.get_node_type() == NodeType.CallPrimitive and node.get_instance_type() not in banned_ops) \
                    or (node.get_node_type() == NodeType.Tree and node.get_instance_type() not in banned_trees) \
                    or node.get_node_type() == NodeType.CellContainer:
                # if (node.get_node_type() == NodeType.Tree) \
                #         and node.get_instance() is not None:
                #     print("node_to_sub", node.get_name())
                #     print("node_to_sub get_node_type", node.get_node_type())
                #     # subtree = node.symbol_tree
                #     subtree = TreeNodeHelper.get_sub_tree(mindspore.rewrite.api.node.Node(node))
                #     # subtree = TreeNodeHelper.get_sub_tree(node)
                #     # print("node_to_sub", node.get_name())
                #     # print("node_to_sub type", node.get_instance())
                #     # scan_node(subtree, nodelist, hash_table, nodedict=nodedict)
                #     scan_node(subtree, hash_table, nodedict=nodedict)
                if hash_table[mindspore.rewrite.api.node.Node(node).get_handler()] == 1:
                    # print("scanned node:", mindspore.rewrite.api.node.Node(node).get_name())
                    continue
                hash_table[mindspore.rewrite.api.node.Node(node).get_handler()] += 1
                if node.get_node_type() not in [NodeType.CellContainer, NodeType.Tree]:
                    # nodelist.append(node)
                    # if nodedict is not None:
                    nodedict[mindspore.rewrite.api.node.Node(node).get_handler()] = node.get_belong_symbol_tree()
        return True


    nodedict = collections.OrderedDict()
    hash_table = collections.defaultdict(int)
    scan_node(stree, hash_table, nodedict)
    print("nodedict:", [node.get_name() for node in nodedict.keys()])
    print("length:", len(nodedict))
    print("hash_table:", hash_table)
