import argparse
import ast
import cv2
import mindspore
import numpy as np
from mindspore import context
import mindspore.dataset as de
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, weight_init='xavier_uniform')


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=padding,
                     dilation=dilation, weight_init='xavier_uniform')


class Resnet(nn.Cell):
    """Resnet"""

    def __init__(self, block, block_num, output_stride, use_batch_statistics=True):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, pad_mode='pad', padding=3,
                               weight_init='xavier_uniform')
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

    def _make_layer(self, block, planes, blocks, stride=1, base_dilation=1, grids=None, use_batch_statistics=True):
        """Resnet._make_layer"""
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
        """Resnet.construct"""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        low_level_feat = out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out, low_level_feat


class Bottleneck(nn.Cell):
    """Bottleneck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_batch_statistics=True):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        self.conv2 = conv3x3(planes, planes, stride, dilation, dilation)
        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, use_batch_statistics=use_batch_statistics)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.add = P.Add()

    def construct(self, x):
        """Bottleneck.construct"""
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


class ASPPConv(nn.Cell):
    """ASPPConv"""

    def __init__(self, in_channels, out_channels, atrous_rate=1, use_batch_statistics=True):
        super(ASPPConv, self).__init__()
        if atrous_rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init='xavier_uniform')
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', padding=atrous_rate,
                             dilation=atrous_rate, weight_init='xavier_uniform')
        bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        relu = nn.ReLU()
        self.aspp_conv = nn.SequentialCell([conv, bn, relu])

    def construct(self, x):
        out = self.aspp_conv(x)
        return out


class ASPPPooling(nn.Cell):
    """ASPPPooling"""

    def __init__(self, in_channels, out_channels, use_batch_statistics=True):
        super(ASPPPooling, self).__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, weight_init='xavier_uniform'),
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


class ASPP(nn.Cell):
    """ASPP"""

    def __init__(self, atrous_rates, phase='train', in_channels=2048, num_classes=21,
                 use_batch_statistics=True):
        super(ASPP, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        out_channels = 256
        self.aspp1 = ASPPConv(in_channels, out_channels, atrous_rates[0], use_batch_statistics=use_batch_statistics)
        self.aspp2 = ASPPConv(in_channels, out_channels, atrous_rates[1], use_batch_statistics=use_batch_statistics)
        self.aspp3 = ASPPConv(in_channels, out_channels, atrous_rates[2], use_batch_statistics=use_batch_statistics)
        self.aspp4 = ASPPConv(in_channels, out_channels, atrous_rates[3], use_batch_statistics=use_batch_statistics)
        self.aspp_pooling = ASPPPooling(in_channels, out_channels, use_batch_statistics=use_batch_statistics)
        self.conv1 = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1,
                               weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.concat = P.Concat(axis=1)
        self.drop = nn.Dropout(0.3)

    def construct(self, x):
        """ASPP.construct"""
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp_pooling(x)
        x = self.concat((x1, x2))
        x = self.concat((x, x3))
        x = self.concat((x, x4))
        x = self.concat((x, x5))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.phase == 'train':
            x = self.drop(x)
        return x


class DeepLabV3Plus(nn.Cell):
    """DeepLabV3Plus"""

    def __init__(self, phase='train', num_classes=21, output_stride=16, freeze_bn=False):
        super(DeepLabV3Plus, self).__init__()
        use_batch_statistics = not freeze_bn
        self.resnet = Resnet(Bottleneck, [3, 4, 23, 3], output_stride=output_stride,
                             use_batch_statistics=use_batch_statistics)
        self.aspp = ASPP([1, 6, 12, 18], phase, 2048, num_classes,
                         use_batch_statistics=use_batch_statistics)
        self.shape = P.Shape()
        self.conv2 = nn.Conv2d(256, 48, kernel_size=1, weight_init='xavier_uniform')
        self.bn2 = nn.BatchNorm2d(48, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.concat = P.Concat(axis=1)
        self.last_conv = nn.SequentialCell([
            conv3x3(304, 256, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(256, use_batch_statistics=use_batch_statistics),
            nn.ReLU(),
            conv3x3(256, 256, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(256, use_batch_statistics=use_batch_statistics),
            nn.ReLU(),
            conv1x1(256, num_classes, stride=1)
        ])

    def construct(self, x):
        # print("================================================================")
        # print(x.shape)
        # print("================================================================")
        """DeepLabV3Plus.construct"""
        size = self.shape(x)
        out, low_level_features = self.resnet(x)
        size2 = self.shape(low_level_features)
        out = self.aspp(out)
        out = P.ResizeNearestNeighbor((size2[2], size2[3]), True)(out)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        out = self.concat((out, low_level_features))
        out = self.last_conv(out)
        out = P.ResizeBilinear((size[2], size[3]), True)(out)
        return out


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser('MindSpore DeepLabV3+ training')
    # Ascend or CPU
    parser.add_argument('--train_dir', type=str, default='', help='where training log and CKPTs saved')

    # dataset
    parser.add_argument('--data_file', type=str, default='', help='path and Name of one MindRecord file')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--crop_size', type=int, default=513, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[103.53, 116.28, 123.675], help='image mean')
    parser.add_argument('--image_std', type=list, default=[57.375, 57.120, 58.395], help='image std')
    parser.add_argument('--min_scale', type=float, default=0.5, help='minimum scale of data argumentation')
    parser.add_argument('--max_scale', type=float, default=2.0, help='maximum scale of data argumentation')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=21, help='number of classes')

    # optimizer
    parser.add_argument('--train_epochs', type=int, default=300, help='epoch')
    parser.add_argument('--lr_type', type=str, default='cos', help='type of learning rate')
    parser.add_argument('--base_lr', type=float, default=0.08, help='base learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=40000, help='learning rate decay step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--loss_scale', type=float, default=3072.0, help='loss scale')

    # model
    parser.add_argument('--model', type=str, default='DeepLabV3plus_s16', help='select model')
    parser.add_argument('--freeze_bn', action='store_true', help='freeze bn')
    parser.add_argument('--ckpt_pre_trained', type=str, default='', help='PreTrained model')

    # train
    parser.add_argument('--device_target', type=str, default='CPU', choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--is_distributed', action='store_true', help='distributed training')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    parser.add_argument('--save_steps', type=int, default=1, help='steps interval for saving')
    parser.add_argument('--keep_checkpoint_max', type=int, default=200, help='max checkpoint for saving')

    # ModelArts
    parser.add_argument('--modelArts_mode', type=ast.literal_eval, default=False,
                        help='train on modelarts or not, default is False')
    parser.add_argument('--train_url', type=str, default='', help='where training log and CKPTs saved')
    parser.add_argument('--data_url', type=str, default='', help='the directory path of saved file')
    parser.add_argument('--dataset_filename', type=str, default='', help='Name of the MindRecord file')
    parser.add_argument('--pretrainedmodel_filename', type=str, default='', help='Name of the pretraining model file')

    args, _ = parser.parse_known_args()
    return args


class SegDataset:
    """SegDataset"""

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
        """SegDataset.preprocess_"""
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
        """SegDataset.get_dataset"""
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


class SoftmaxCrossEntropyLoss(nn.Cell):
    """SoftmaxCrossEntropyLoss"""

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
        """SoftmaxCrossEntropyLoss.construct"""
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





def deeplabv3_mindspore(logits, labels):
    num_cls = 21
    ignore_label = 255
    one_hot = P.OneHot(axis=-1)
    on_value = Tensor(1.0, mstype.float32)
    off_value = Tensor(0.0, mstype.float32)
    cast = P.Cast()
    ce = nn.SoftmaxCrossEntropyWithLogits()
    not_equal = P.NotEqual()
    mul = P.Mul()
    sum = P.ReduceSum(False)
    div = P.RealDiv()
    transpose = P.Transpose()
    reshape = P.Reshape()

    labels_int = cast(labels, mstype.int32)
    labels_int = reshape(labels_int, (-1,))
    logits_ = transpose(logits, (0, 2, 3, 1))
    logits_ = reshape(logits_, (-1, num_cls))
    weights = not_equal(labels_int, ignore_label)
    weights = cast(weights, mstype.float32)
    one_hot_labels = one_hot(labels_int, num_cls, on_value, off_value)
    loss = ce(logits_, one_hot_labels)
    loss = mul(weights, loss)
    loss = div(sum(loss), sum(weights))
    return loss



class Losser(nn.Cell):
    def __init__(self, network, criterion):
        super(Losser, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss


if __name__ == '__main__':
    args = parse_args()
    device_id = 0
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=args.device_target)
    if args.device_target != "CPU":
        context.set_context(device_id=device_id)
    data_file = args.data_file
    ckpt_file = args.ckpt_pre_trained
    train_dir = args.train_dir
    # dataset = SegDataset(image_mean=args.image_mean,
    #                      image_std=args.image_std,
    #                      data_file=data_file,
    #                      batch_size=args.batch_size,
    #                      crop_size=args.crop_size,
    #                      max_scale=args.max_scale,
    #                      min_scale=args.min_scale,
    #                      ignore_label=args.ignore_label,
    #                      num_classes=args.num_classes,
    #                      num_readers=2,
    #                      num_parallel_calls=4,
    #                      shard_id=args.rank,
    #                      shard_num=args.group_size)
    # dataset = dataset.get_dataset(repeat=1)
    network = DeepLabV3Plus('train', args.num_classes, 8, args.freeze_bn)
    loser = SoftmaxCrossEntropyLoss(args.num_classes, args.ignore_label)
    loser.add_flags_recursive(fp32=True)
    net = Losser(network, loser)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.0005, momentum=0.9, weight_decay=0.0001,
                      loss_scale=args.loss_scale)


    def forward_fn(data, label):
        loss = net(data, label)
        return loss


    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)


    def train_step(data, label):
        (loss), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, opt(grads))
        return loss


    # for data in dataset:
    a = np.random.randn(4, 3, 513, 513)
    a = mindspore.Tensor(a, dtype=mindspore.float32)
    b = np.random.randn(4, 513, 513)
    b = mindspore.Tensor(b, dtype=mindspore.float32)
    print("================================")
    loss_ms = train_step(a, b)
    print(loss_ms)
    print("================================")
