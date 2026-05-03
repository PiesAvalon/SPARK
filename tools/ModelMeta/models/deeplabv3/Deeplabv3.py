import collections

import cv2
import mindspore
import numpy as np
import numpy.random
from mindspore import context, ops
from mindspore.rewrite import SymbolTree, NodeType
from configs.DeeplabConfig import config
from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.dataset as de


class SoftmaxCrossEntropyLoss(nn.Cell):
    def __init__(self, num_cls=21, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss


class SegDataset:
    def __init__(self,
                 image_mean,
                 image_std,
                 data_file='',
                 batch_size=32,
                 crop_size=512,
                 max_scale=2.0,
                 min_scale=0.5,
                 ignore_label=255,
                 num_classes=21,
                 num_readers=2,
                 num_parallel_calls=4,
                 shard_id=None,
                 shard_num=None):

        self.data_file = data_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.num_readers = num_readers
        self.num_parallel_calls = num_parallel_calls
        self.shard_id = shard_id
        self.shard_num = shard_num
        assert max_scale > min_scale

    def preprocess_(self, image, label):
        # bgr image
        image_out = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        label_out = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image_out.shape[0]), int(sc * image_out.shape[1])
        image_out = cv2.resize(image_out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label_out, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        image_out = (image_out - self.image_mean) / self.image_std
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size]

        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]

        image_out = image_out.transpose((2, 0, 1))
        image_out = image_out.copy()
        label_out = label_out.copy()
        return image_out, label_out

    def get_dataset(self, repeat=1):
        data_set = de.MindDataset(self.data_file, columns_list=["data", "label"],
                                  shuffle=True, num_parallel_workers=self.num_readers,
                                  num_shards=self.shard_num, shard_id=self.shard_id)
        transforms_list = self.preprocess_
        data_set = data_set.map(operations=transforms_list, input_columns=["data", "label"],
                                output_columns=["data", "label"],
                                num_parallel_workers=self.num_parallel_calls)
        data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
        data_set = data_set.batch(self.batch_size, drop_remainder=True)
        data_set = data_set.repeat(repeat)
        return data_set


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     weight_init='HeUniform', has_bias=False)


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    # print("in_planes:", in_planes, "out_planes:", out_planes, "stride:", stride, "dilation:", dilation, "padding:",
    #       padding)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=padding,
                     dilation=dilation, weight_init='HeUniform', has_bias=False)


class Resnet(nn.Cell):
    def __init__(self, block, block_num, output_stride, use_batch_statistics=False):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, pad_mode='pad', padding=3,
                               weight_init='HeUniform', has_bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, block_num[0], use_batch_statistics=use_batch_statistics)
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2, use_batch_statistics=use_batch_statistics)
        if output_stride == 16:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=2, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)
        elif output_stride == 8:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=1, base_dilation=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=4, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)

    def _make_layer(self, block, planes, blocks, stride=1, base_dilation=1, grids=None, use_batch_statistics=False):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, use_batch_statistics=use_batch_statistics)
            ])
        if grids is None:
            grids = [1] * blocks
        layers = [
            block(self.inplanes, planes, stride, downsample, dilation=base_dilation * grids[0],
                  use_batch_statistics=use_batch_statistics)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=base_dilation * grids[i],
                      use_batch_statistics=use_batch_statistics))
        return nn.SequentialCell(layers)

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # print("layer1", out.shape)
        out = self.layer1(out)
        # print("layer2", out.shape)
        out = self.layer2(out)
        # print("layer3", out.shape)
        out = self.layer3(out)
        # print("layer4", out.shape)
        out = self.layer4(out)
        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_batch_statistics=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        self.conv2 = conv3x3(planes, planes, stride, dilation, dilation)
        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics, momentum=0.9)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, use_batch_statistics=use_batch_statistics)

        self.relu = nn.ReLU()
        self.downsample = downsample

        self.add = mindspore.ops.add

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)
        return out


class ASPP(nn.Cell):
    def __init__(self, atrous_rates, phase='train', in_channels=2048, num_classes=21,
                 use_batch_statistics=False):
        super(ASPP, self).__init__()
        self.phase = phase
        out_channels = 256
        self.aspp1 = ASPPConv(in_channels, out_channels, atrous_rates[0], use_batch_statistics=use_batch_statistics)
        self.aspp2 = ASPPConv(in_channels, out_channels, atrous_rates[1], use_batch_statistics=use_batch_statistics)
        self.aspp3 = ASPPConv(in_channels, out_channels, atrous_rates[2], use_batch_statistics=use_batch_statistics)
        self.aspp4 = ASPPConv(in_channels, out_channels, atrous_rates[3], use_batch_statistics=use_batch_statistics)
        self.aspp_pooling = ASPPPooling(in_channels, out_channels, use_batch_statistics=use_batch_statistics)
        self.conv1 = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1,
                               weight_init='HeUniform', has_bias=False, )
        self.bn1 = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, num_classes, kernel_size=1, weight_init='HeUniform', has_bias=False, )
        self.concat = ops.concat
        self.drop = nn.Dropout(0.3)

    def construct(self, x):
        # print("aspp input", x.shape)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp_pooling(x)

        x = self.concat((x1, x2), axis=1)
        x = self.concat((x, x3), axis=1)
        x = self.concat((x, x4), axis=1)
        x = self.concat((x, x5), axis=1)
        # print("before conv1", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        # print("after bn1 maximum", ops.max(x))
        x = self.relu(x)
        # print("after relu maximum", ops.max(x))
        if self.training:
            x = self.drop(x)
        x = self.conv2(x)
        return x


class ASPPPooling(nn.Cell):
    def __init__(self, in_channels, out_channels, use_batch_statistics=False):
        super(ASPPPooling, self).__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, weight_init='HeUniform',
                      has_bias=False, ),
            nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics),
            nn.ReLU()
        ])
        self.shape = P.Shape()

    def construct(self, x):
        size = self.shape(x)
        out = nn.AvgPool2d(size[2])(x)
        out = self.conv(out)
        out = P.ResizeNearestNeighbor((size[2], size[3]), True)(out)
        return out


class ASPPConv(nn.Cell):
    def __init__(self, in_channels, out_channels, atrous_rate=1, use_batch_statistics=False):
        super(ASPPConv, self).__init__()
        if atrous_rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init='HeUniform'
                             , )
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', padding=atrous_rate,
                             dilation=atrous_rate, weight_init='HeUniform', has_bias=False, )
        bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        relu = nn.ReLU()
        self.aspp_conv = nn.SequentialCell([conv, bn, relu])

    def construct(self, x):
        # print("before aspp_conv shape", x.shape)
        out = self.aspp_conv(x)
        return out


def cosine_lr(base_lr, decay_steps, total_steps):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))


class DeepLabV3(nn.Cell):
    def __init__(self, phase='train', num_classes=21, output_stride=16, freeze_bn=False):
        super(DeepLabV3, self).__init__()
        self.resnet = Resnet(Bottleneck, [3, 4, 23, 3], output_stride=output_stride,
                             use_batch_statistics=False)
        self.aspp = ASPP([1, 6, 12, 18], phase, 2048, num_classes,
                         use_batch_statistics=False)
        self.shape = P.Shape()

    def construct(self, x):
        size = self.shape(x)
        # print("x.shape:", x.shape)
        out = self.resnet(x)
        # print("before aspp.shape:", out.shape)
        out = self.aspp(out)
        # print("after aspp.shape:", out.shape)
        out = ops.interpolate(out, (size[2], size[3]), mode='bilinear', align_corners=True)
        # out = P.ResizeBilinearV2((size[2], size[3]), True)(out)
        # print("out.shape:", out.shape)
        return out


def scan_node(stree, nodelist, hash_table, nodedict=None):
    # global hash_table
    for node in stree.nodes(all_nodes=True):
        if node.get_node_type() == NodeType.Tree and node.get_instance() is not None:
            # subtree = TreeNodeHelper.get_sub_tree(mindspore.rewrite.api.node.Node(node))
            subtree = TreeNodeHelper.get_sub_tree(node)
            # print("node_to_sub", node.get_name())
            # print("node_to_sub type", node.get_instance())
            scan_node(subtree, nodelist, hash_table, nodedict)
        if hash_table[node.get_handler()] == 1:
            continue
        hash_table[node.get_handler()] += 1
        nodelist.append(node)
        nodedict[node.get_handler()] = node._node.get_belong_symbol_tree()
    return True


if __name__ == '__main__':
    args = config
    args.batch_size = 4
    device_id = 1
    args.device_target = "CPU"
    if args.device_target == "CPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU")
    else:
        context.set_context(device_target="GPU", device_id=device_id)
    # dataset
    net = DeepLabV3('train', args.num_classes, 8, args.freeze_bn)
    # print("args.ckpt_pre_trained:", args.ckpt_pre_trained)
    # if args.ckpt_pre_trained:
    #     param_dict = load_checkpoint(args.ckpt_pre_trained)
    #     if args.filter_weight:
    #         print("here")
    #         filter_list = ["network.aspp.conv2.weight", "network.aspp.conv2.bias"]
    #         for key in list(param_dict.keys()):
    #             for filter_key in filter_list:
    #                 if filter_key not in key:
    #                     continue
    #                 print('filter {}'.format(key))
    #                 del param_dict[key]
    #         load_param_into_net(net, param_dict)
    #         print('load_model {} success'.format(args.ckpt_pre_trained))
    #     else:
    #         print("there")
    #         trans_param_dict = {}
    #         for key, val in param_dict.items():
    #             key = key.replace("down_sample_layer", "downsample")
    #             trans_param_dict[f"network.resnet.{key}"] = val
    #         load_param_into_net(net, trans_param_dict)
    #         print('load_model {} success'.format(args.ckpt_pre_trained))
    stree = SymbolTree.create(net)
    # for i in range(4):
    nodelist = []
    nodedict = collections.OrderedDict()
    hash_table = collections.defaultdict(int)
    a = numpy.random.randn(1, 3, 513, 513)
    a = mindspore.Tensor(a, dtype=mindspore.float32)
    output = net(a)
    print(output.shape)
    scan_node(stree, nodelist, hash_table, nodedict)
    # print(nodedict)
    for key in nodedict.keys():
        print(key.get_name(), nodedict[key])
