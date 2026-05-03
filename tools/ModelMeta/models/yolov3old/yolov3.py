import os
from PIL import Image
from pycocotools.coco import COCO
from mindspore import nn, ops, Tensor, SymbolTree
from configs.yolov3config import config as default_config, get_config
import mindspore as ms
import numpy as np
group_size = 1
per_batch_size = 16
steps_per_epoch = 1
min_keypoints_per_image = 10
log_interval = 1


class ResidualBlock(nn.Cell):
    """
    DarkNet V1 residual block definition.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.

    Returns:
        Tensor, output tensor.
    Examples:
        ResidualBlock(3, 208)
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResidualBlock, self).__init__()
        out_chls = out_channels // 2
        self.conv1 = conv_block(in_channels, out_chls, kernel_size=1, stride=1)
        self.conv2 = conv_block(out_chls, out_channels, kernel_size=3, stride=1)
        self.add = ops.add

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.add(out, identity)

        return out


def conv_block(in_channels,
               out_channels,
               kernel_size,
               stride,
               dilation=1):
    pad_mode = 'same'
    padding = 0

    return nn.SequentialCell(
        [nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channels, momentum=0.1),
         nn.ReLU()]
    )


class DarkNet(nn.Cell):
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 detect=False):
        super(DarkNet, self).__init__()

        self.outchannel = out_channels[-1]
        self.detect = detect

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 5:
            raise ValueError("the length of layer_num, inchannel, outchannel list must be 5!")
        self.conv0 = conv_block(3,
                                in_channels[0],
                                kernel_size=3,
                                stride=1)
        self.conv1 = conv_block(in_channels[0],
                                out_channels[0],
                                kernel_size=3,
                                stride=2)
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=out_channels[0],
                                       out_channel=out_channels[0])
        self.conv2 = conv_block(in_channels[1],
                                out_channels[1],
                                kernel_size=3,
                                stride=2)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=out_channels[1],
                                       out_channel=out_channels[1])
        self.conv3 = conv_block(in_channels[2],
                                out_channels[2],
                                kernel_size=3,
                                stride=2)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=out_channels[2],
                                       out_channel=out_channels[2])
        self.conv4 = conv_block(in_channels[3],
                                out_channels[3],
                                kernel_size=3,
                                stride=2)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=out_channels[3],
                                       out_channel=out_channels[3])
        self.conv5 = conv_block(in_channels[4],
                                out_channels[4],
                                kernel_size=3,
                                stride=2)
        self.layer5 = self._make_layer(block,
                                       layer_nums[4],
                                       in_channel=out_channels[4],
                                       out_channel=out_channels[4])

    def _make_layer(self, block, layer_num, in_channel, out_channel):
        layers = []
        darkblk = block(in_channel, out_channel)
        layers.append(darkblk)

        for _ in range(1, layer_num):
            darkblk = block(out_channel, out_channel)
            layers.append(darkblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)
        c3 = self.layer1(c2)
        c4 = self.conv2(c3)
        c5 = self.layer2(c4)
        c6 = self.conv3(c5)
        c7 = self.layer3(c6)
        c8 = self.conv4(c7)
        c9 = self.layer4(c8)
        c10 = self.conv5(c9)
        c11 = self.layer5(c10)
        if self.detect:
            return c7, c9, c11

        return c11

    def get_out_channels(self):
        return self.outchannel


def _conv_bn_relu(in_channel,
                  out_channel,
                  ksize,
                  stride=1,
                  padding=0,
                  dilation=1,
                  alpha=0.1,
                  momentum=0.9,
                  eps=1e-5,
                  pad_mode="same"):
    """Get a conv2d batchnorm and relu layer"""
    return nn.SequentialCell(
        [nn.Conv2d(in_channel,
                   out_channel,
                   kernel_size=ksize,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channel, momentum=momentum, eps=eps),
         nn.LeakyReLU(alpha)]
    )


# class YOLOv3(nn.Cell):
#     def __init__(self, backbone_shape, backbone, out_channel):
#         super(YOLOv3, self).__init__()
#         self.out_channel = out_channel
#         self.backbone = backbone
#         self.backblock0 = YoloBlock(backbone_shape[-1], out_chls=backbone_shape[-2], out_channels=out_channel)
#
#         self.conv1 = _conv_bn_relu(in_channel=backbone_shape[-2], out_channel=backbone_shape[-2] // 2, ksize=1)
#         self.backblock1 = YoloBlock(in_channels=backbone_shape[-2] + backbone_shape[-3],
#                                     out_chls=backbone_shape[-3],
#                                     out_channels=out_channel)
#
#         self.conv2 = _conv_bn_relu(in_channel=backbone_shape[-3], out_channel=backbone_shape[-3] // 2, ksize=1)
#         self.backblock2 = YoloBlock(in_channels=backbone_shape[-3] + backbone_shape[-4],
#                                     out_chls=backbone_shape[-4],
#                                     out_channels=out_channel)
#         self.concat = ops.Concat(axis=1)
#
#     def construct(self, x):
#         img_hight = ops.Shape()(x)[2]
#         img_width = ops.Shape()(x)[3]
#         feature_map1, feature_map2, feature_map3 = self.backbone(x)
#         con1, big_object_output = self.backblock0(feature_map3)
#
#         con1 = self.conv1(con1)
#         ups1 = ops.ResizeNearestNeighbor((img_hight // 16, img_width // 16))(con1)
#         con1 = self.concat((ups1, feature_map2))
#         con2, medium_object_output = self.backblock1(con1)
#
#         con2 = self.conv2(con2)
#         ups2 = ops.ResizeNearestNeighbor((img_hight // 8, img_width // 8))(con2)
#         con3 = self.concat((ups2, feature_map1))
#         _, small_object_output = self.backblock2(con3)
#
#         return big_object_output, medium_object_output, small_object_output


class YOLOv3(nn.Cell):
    def __init__(self, backbone_shape, backbone, out_channel):
        super(YOLOv3, self).__init__()
        self.out_channel = out_channel
        self.backbone = backbone
        self.backblock0 = YoloBlock(backbone_shape[-1], out_chls=backbone_shape[-2], out_channels=out_channel)

        # self.conv1 = _conv_bn_relu(in_channel=backbone_shape[-2], out_channel=backbone_shape[-2] // 2, ksize=1)
        self.conv1 = nn.Conv2d(backbone_shape[-2],
                               backbone_shape[-2] // 2,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               pad_mode="same")
        self.bn1 = nn.BatchNorm2d(backbone_shape[-2] // 2, momentum=0.9, eps=1e-5)
        self.relu1 = nn.LeakyReLU(alpha=0.1)
        self.backblock1 = YoloBlock(in_channels=backbone_shape[-2] + backbone_shape[-3],
                                    out_chls=backbone_shape[-3],
                                    out_channels=out_channel)

        # self.conv2 = _conv_bn_relu(in_channel=backbone_shape[-3], out_channel=backbone_shape[-3] // 2, ksize=1)
        self.conv2 = nn.Conv2d(backbone_shape[-3],
                               backbone_shape[-3] // 2,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               pad_mode="same")
        self.bn2 = nn.BatchNorm2d(backbone_shape[-3] // 2, momentum=0.9, eps=1e-5)
        self.relu2 = nn.LeakyReLU(alpha=0.1)
        self.backblock2 = YoloBlock(in_channels=backbone_shape[-3] + backbone_shape[-4],
                                    out_chls=backbone_shape[-4],
                                    out_channels=out_channel)
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        # input_shape of x is (batch_size, 3, h, w)
        # feature_map1 is (batch_size, backbone_shape[2], h/8, w/8)
        # feature_map2 is (batch_size, backbone_shape[3], h/16, w/16)
        # feature_map3 is (batch_size, backbone_shape[4], h/32, w/32)
        img_hight = ops.Shape()(x)[2]
        img_width = ops.Shape()(x)[3]
        feature_map1, feature_map2, feature_map3 = self.backbone(x)
        con1, big_object_output = self.backblock0(feature_map3)

        con1 = self.conv1(con1)
        con1 = self.bn1(con1)
        con1 = self.relu1(con1)
        ups1 = ops.ResizeNearestNeighbor((img_hight // 16, img_width // 16))(con1)
        con1 = self.concat((ups1, feature_map2))
        con2, medium_object_output = self.backblock1(con1)

        con2 = self.conv2(con2)
        con2 = self.bn2(con2)
        con2 = self.relu2(con2)
        ups2 = ops.ResizeNearestNeighbor((img_hight // 8, img_width // 8))(con2)
        con3 = self.concat((ups2, feature_map1))
        _, small_object_output = self.backblock2(con3)

        return big_object_output, medium_object_output, small_object_output


class YoloBlock(nn.Cell):
    """
    YoloBlock for YOLOv3.

    Args:
        in_channels: Integer. Input channel.
        out_chls: Integer. Middle channel.
        out_channels: Integer. Output channel.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3).

    Examples:
        YoloBlock(1024, 512, 255)

    """

    def __init__(self, in_channels, out_chls, out_channels):
        super(YoloBlock, self).__init__()
        out_chls_2 = out_chls * 2

        self.conv0 = _conv_bn_relu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv2 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv4 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        out = self.conv6(c6)
        return c5, out


class DetectionBlock(nn.Cell):
    """
     YOLOv3 detection Network. It will finally output the detection result.

     Args:
         scale: Character.
         config: Configuration.
         is_training: Bool, Whether train or not, default True.

     Returns:
         Tuple, tuple of output tensor,(f1,f2,f3).

     Examples:
         DetectionBlock(scale='l',stride=32,config=config)
     """

    def __init__(self, scale, config=None, is_training=True):
        super(DetectionBlock, self).__init__()
        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = ms.Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4 + 1 + self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=-1)
        self.conf_training = is_training

    def construct(self, x, input_shape):
        num_batch = ops.Shape()(x)[0]
        grid_size = ops.Shape()(x)[2:4]

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = ops.Reshape()(x, (num_batch, self.num_anchors_per_scale, self.num_attrib,
                                       grid_size[0], grid_size[1]))
        prediction = ops.Transpose()(prediction, (0, 3, 4, 1, 2))

        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        grid_x = ops.Cast()(ops.tuple_to_array(range_x), ms.float32)
        grid_y = ops.Cast()(ops.tuple_to_array(range_y), ms.float32)
        # Tensor of shape [grid_size[0], grid_size[1], 1, 1] representing the coordinate of x/y axis for each grid
        # [batch, gridx, gridy, 1, 1]
        grid_x = self.tile(self.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(self.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = self.concat((grid_x, grid_y))

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]

        # gridsize1 is x
        # gridsize0 is y
        box_xy = (self.sigmoid(box_xy) + grid) / ops.Cast()(ops.tuple_to_array((grid_size[1],
                                                                                grid_size[0])), ms.float32)
        # box_wh is w->h
        box_wh = ops.Exp()(box_wh) * self.anchors / input_shape

        # if self.conf_training:
        # print("fucker here!!!!!!!!!!!!!")
        return grid, prediction, box_xy, box_wh
        # box_confidence = prediction[:, :, :, :, 4:5]
        # box_probs = prediction[:, :, :, :, 5:]
        # box_confidence = self.sigmoid(box_confidence)
        # box_probs = self.sigmoid(box_probs)
        # return self.concat((box_xy, box_wh, box_confidence, box_probs))


class yolov3(nn.Cell):
    def __init__(self, is_training, config=default_config):
        super(yolov3, self).__init__()
        self.config = config
        self.keep_detect = self.config.keep_detect
        self.tenser_to_array = ops.TupleToArray()

        # YOLOv3 network
        self.feature_map = YOLOv3(backbone=DarkNet(ResidualBlock, self.config.backbone_layers,
                                                   self.config.backbone_input_shape,
                                                   self.config.backbone_shape,
                                                   detect=True),
                                  backbone_shape=self.config.backbone_shape,
                                  out_channel=self.config.out_channel)

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock('l', is_training=is_training, config=self.config)
        self.detect_2 = DetectionBlock('m', is_training=is_training, config=self.config)
        self.detect_3 = DetectionBlock('s', is_training=is_training, config=self.config)

    def construct(self, x):
        input_shape = ops.shape(x)[2:4]
        input_shape = ops.cast(self.tenser_to_array(input_shape), ms.float32)
        # print("x shape is ", ops.shape(x))
        big_object_output, medium_object_output, small_object_output = self.feature_map(x)
        if not self.keep_detect:
            print("not keep detect")
            return big_object_output, medium_object_output, small_object_output
        output_big = self.detect_1(big_object_output, input_shape)
        # print("type of output_big is ", type(output_big))
        output_me = self.detect_2(medium_object_output, input_shape)
        output_small = self.detect_3(small_object_output, input_shape)
        # print("length of output is ", len(output_big), len(output_me), len(output_small))
        # print("output shape is ", ops.shape(output_big[0][0]), ops.shape(output_big[1][0]), ops.shape(output_big[2][0]), ops.shape(output_big[3][0]))
        # (18, 18, 1, 2)(18, 18, 3, 85)(18, 18, 3, 2)(18, 18, 3, 2)
        # big is the final output which has smallest feature map
        return output_big, output_me, output_small


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def has_valid_annotation(anno):
    """Check annotation file."""
    # if it's empty, there is no annotation
    if not anno:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different criteria for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCOYoloDataset:
    """YOLOV3 Dataset for COCO."""

    def __init__(self, root, ann_file, remove_images_without_annotations=True,
                 filter_crowd_anno=True, is_training=True):
        self.coco = COCO(ann_file)
        self.root = root
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.filter_crowd_anno = filter_crowd_anno
        self.is_training = is_training

        # filter images without any annotations
        if remove_images_without_annotations:
            img_ids = []
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    img_ids.append(img_id)
            self.img_ids = img_ids

        self.categories = {cat["id"]: cat["name"] for cat in self.coco.cats.values()}

        self.cat_ids_to_continuous_ids = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.continuous_ids_cat_ids = {
            v: k for k, v in self.cat_ids_to_continuous_ids.items()
        }

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            (img, target) (tuple): target is a dictionary contains "bbox", "segmentation" or "keypoints",
                generated by the image's annotation. img is a PIL image.
        """
        coco = self.coco
        img_id = self.img_ids[index]
        img_path = coco.loadImgs(img_id)[0]["file_name"]
        if not self.is_training:
            img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
            return img, img_id
        img = np.fromfile(os.path.join(self.root, img_path), dtype="int8")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        # filter crowd annotations
        if self.filter_crowd_anno:
            annos = [anno for anno in target if anno["iscrowd"] == 0]
        else:
            annos = [anno for anno in target]

        target = {}
        boxes = [anno["bbox"] for anno in annos]
        target["bboxes"] = boxes

        classes = [anno["category_id"] for anno in annos]
        classes = [self.cat_ids_to_continuous_ids[cl] for cl in classes]
        target["labels"] = classes

        bboxes = target['bboxes']
        labels = target['labels']
        out_target = []
        for bbox, label in zip(bboxes, labels):
            tmp = []
            # convert to [x_min y_min x_max y_max]
            bbox = self._convetTopDown(bbox)
            tmp.extend(bbox)
            tmp.append(int(label))
            # tmp [x_min y_min x_max y_max, label]
            out_target.append(tmp)
        return img, out_target, [], [], [], [], [], []

    def __len__(self):
        return len(self.img_ids)

    def _convetTopDown(self, bbox):
        x_min = bbox[0]
        y_min = bbox[1]
        w = bbox[2]
        h = bbox[3]
        return [x_min, y_min, x_min + w, y_min + h]


class Iou(nn.Cell):
    def __init__(self):
        super(Iou, self).__init__()
        self.min = ops.Minimum()
        self.max = ops.Maximum()

    def construct(self, box1, box2):
        # box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
        # box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
        # convert to topLeft and rightDown
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        box1_mins = box1_xy - box1_wh / ops.scalar_to_tensor(2.0)  # topLeft
        box1_maxs = box1_xy + box1_wh / ops.scalar_to_tensor(2.0)  # rightDown

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        box2_mins = box2_xy - box2_wh / ops.scalar_to_tensor(2.0)
        box2_maxs = box2_xy + box2_wh / ops.scalar_to_tensor(2.0)

        intersect_mins = self.max(box1_mins, box2_mins)
        intersect_maxs = self.min(box1_maxs, box2_maxs)
        intersect_wh = self.max(intersect_maxs - intersect_mins, ops.scalar_to_tensor(0.0))
        # ops.squeeze: for effiecient slice
        intersect_area = ops.Squeeze(-1)(intersect_wh[:, :, :, :, :, 0:1]) * \
                         ops.Squeeze(-1)(intersect_wh[:, :, :, :, :, 1:2])
        box1_area = ops.Squeeze(-1)(box1_wh[:, :, :, :, :, 0:1]) * ops.Squeeze(-1)(box1_wh[:, :, :, :, :, 1:2])
        box2_area = ops.Squeeze(-1)(box2_wh[:, :, :, :, :, 0:1]) * ops.Squeeze(-1)(box2_wh[:, :, :, :, :, 1:2])
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        # iou : [batch, gx, gy, anchors, maxboxes]
        return iou


class XYLoss(nn.Cell):
    """Loss for x and y."""

    def __init__(self):
        super(XYLoss, self).__init__()
        self.cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, object_mask, box_loss_scale, predict_xy, true_xy):
        xy_loss = object_mask * box_loss_scale * self.cross_entropy(predict_xy, true_xy)
        xy_loss = self.reduce_sum(xy_loss, ())
        return xy_loss


class WHLoss(nn.Cell):
    """Loss for w and h."""

    def __init__(self):
        super(WHLoss, self).__init__()
        self.square = ops.Square()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, object_mask, box_loss_scale, predict_wh, true_wh):
        wh_loss = object_mask * box_loss_scale * 0.5 * ops.Square()(true_wh - predict_wh)
        wh_loss = self.reduce_sum(wh_loss, ())
        return wh_loss


class ConfidenceLoss(nn.Cell):
    """Loss for confidence."""

    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, object_mask, predict_confidence, ignore_mask):
        confidence_loss = self.cross_entropy(predict_confidence, object_mask)
        confidence_loss = object_mask * confidence_loss + (1 - object_mask) * confidence_loss * ignore_mask
        confidence_loss = self.reduce_sum(confidence_loss, ())
        return confidence_loss


class ClassLoss(nn.Cell):
    """Loss for classification."""

    def __init__(self):
        super(ClassLoss, self).__init__()
        self.cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, object_mask, predict_class, class_probs):
        class_loss = object_mask * self.cross_entropy(predict_class, class_probs)
        class_loss = self.reduce_sum(class_loss, ())
        return class_loss


def yololossblock(grid, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape, scale):
    object_mask = y_true[:, :, :, :, 4:5]
    class_probs = y_true[:, :, :, :, 5:]
    grid_shape = ops.Shape()(prediction)[1:3]
    grid_shape = ops.Cast()(ops.tuple_to_array(grid_shape[::-1]), ms.float32)
    pred_boxes = ops.Concat(axis=-1)((pred_xy, pred_wh))
    true_xy = y_true[:, :, :, :, :2] * grid_shape - grid
    true_wh = y_true[:, :, :, :, 2:4]
    true_wh = ops.Select()(ops.Equal()(true_wh, 0.0), ops.Fill()(ops.DType()(true_wh),
                                                                 ops.Shape()(true_wh), 1.0), true_wh)
    if scale == 's':
        # anchor mask
        idx = (0, 1, 2)
    elif scale == 'm':
        idx = (3, 4, 5)
    else:
        idx = (6, 7, 8)
    true_wh = ops.Log()(true_wh / ms.Tensor([config.anchor_scales[i] for i in idx], ms.float32) * input_shape)
    # 2-w*h for large picture, use small scale, since small obj need more precise
    box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]
    gt_shape = ops.Shape()(gt_box)
    gt_box = ops.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))
    # add one more dimension for broadcast
    iou = Iou()(ops.ExpandDims()(pred_boxes, -2), gt_box)
    # gt_box is x,y,h,w after normalize
    # [batch, grid[0], grid[1], num_anchor, num_gt]
    best_iou = ops.ReduceMax(keep_dims=False)(iou, -1)
    # [batch, grid[0], grid[1], num_anchor]
    # ignore_mask IOU too small
    ignore_mask = best_iou < ms.Tensor(config.ignore_threshold, ms.float32)
    ignore_mask = ops.Cast()(ignore_mask, ms.float32)
    ignore_mask = ops.ExpandDims()(ignore_mask, -1)
    # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
    # so we turn off its gradient
    ignore_mask = ops.stop_gradient(ignore_mask)
    xy_loss = XYLoss()(object_mask, box_loss_scale, prediction[:, :, :, :, :2], true_xy)
    wh_loss = WHLoss()(object_mask, box_loss_scale, prediction[:, :, :, :, 2:4], true_wh)
    confidence_loss = ConfidenceLoss()(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
    class_loss = ClassLoss()(object_mask, prediction[:, :, :, :, 5:], class_probs)
    loss = xy_loss + wh_loss + confidence_loss + class_loss
    batch_size = ops.Shape()(prediction)[0]
    return loss / batch_size


class YoloLossBlock(nn.Cell):
    def __init__(self, scale, config=None):
        super(YoloLossBlock, self).__init__()
        self.config = config
        if scale == 's':
            # anchor mask
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = ms.Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = ms.Tensor(self.config.ignore_threshold, ms.float32)
        self.concat = ops.Concat(axis=-1)
        self.iou = Iou()
        self.reduce_max = ops.ReduceMax(keep_dims=False)
        self.xy_loss = XYLoss()
        self.wh_loss = WHLoss()
        self.confidenceLoss = ConfidenceLoss()
        self.classLoss = ClassLoss()

    def construct(self, grid, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape):
        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]

        grid_shape = ops.Shape()(prediction)[1:3]
        grid_shape = ops.Cast()(ops.tuple_to_array(grid_shape[::-1]), ms.float32)

        pred_boxes = self.concat((pred_xy, pred_wh))
        true_xy = y_true[:, :, :, :, :2] * grid_shape - grid
        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = ops.Select()(ops.Equal()(true_wh, 0.0), ops.Fill()(ops.DType()(true_wh),
                                                                     ops.Shape()(true_wh), 1.0), true_wh)

        true_wh = ops.Log()(true_wh / self.anchors * input_shape)
        # 2-w*h for large picture, use small scale, since small obj need more precise
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = ops.Shape()(gt_box)
        gt_box = ops.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        # add one more dimension for broadcast
        iou = self.iou(ops.ExpandDims()(pred_boxes, -2), gt_box)
        # gt_box is x,y,h,w after normalize
        # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = self.reduce_max(iou, -1)
        # [batch, grid[0], grid[1], num_anchor]

        # ignore_mask IOU too small
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = ops.Cast()(ignore_mask, ms.float32)
        ignore_mask = ops.ExpandDims()(ignore_mask, -1)
        # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
        # so we turn off its gradient
        ignore_mask = ops.stop_gradient(ignore_mask)
        xy_loss = self.xy_loss(object_mask, box_loss_scale, prediction[:, :, :, :, :2], true_xy)
        wh_loss = self.wh_loss(object_mask, box_loss_scale, prediction[:, :, :, :, 2:4], true_wh)
        confidence_loss = self.confidenceLoss(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
        class_loss = self.classLoss(object_mask, prediction[:, :, :, :, 5:], class_probs)
        loss = xy_loss + wh_loss + confidence_loss + class_loss
        batch_size = ops.Shape()(prediction)[0]
        return loss / batch_size


config = get_config()


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


def loser(x, yolo_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
    input_shape = ops.shape(x)[2:4]
    input_shape = ops.cast(ops.TupleToArray()(input_shape), ms.float32)
    loss_l = YoloLossBlock('l', config)(*yolo_out[0], y_true_0, gt_0, input_shape)
    loss_m = YoloLossBlock('m', config)(*yolo_out[1], y_true_1, gt_1, input_shape)
    loss_s = YoloLossBlock('s', config)(*yolo_out[2], y_true_2, gt_2, input_shape)
    return loss_l + loss_m + loss_s


def conver_testing_shape(args):
    """Convert testing shape to list."""
    testing_shape = [int(args.testing_shape), int(args.testing_shape)]
    return testing_shape


def load_parameters(network, file_name):
    config.logger.info("yolov3 pretrained network model: %s", file_name)
    param_dict = ms.load_checkpoint(file_name)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    ms.load_param_into_net(network, param_dict_new)
    config.logger.info('load_model %s success', file_name)


def set_eval(network):
    network.yolo_network.detect_1.conf_training = False
    network.yolo_network.detect_2.conf_training = False
    network.yolo_network.detect_3.conf_training = False
    return network


def settrain(network):
    network.yolo_network.detect_1.conf_training = True
    network.yolo_network.detect_2.conf_training = True
    network.yolo_network.detect_3.conf_training = True
    return network


class losser(nn.Cell):
    def __init__(self, network, config=default_config):
        super(losser, self).__init__()
        self.yolo_network = network
        self.config = config
        self.tenser_to_array = ops.TupleToArray()
        self.loss_big = YoloLossBlock('l', self.config)
        self.loss_me = YoloLossBlock('m', self.config)
        self.loss_small = YoloLossBlock('s', self.config)

    def construct(self, x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        # print("===================")
        # print(x.shape)
        # print(y_true_0.shape)
        # print(y_true_1.shape)
        # print(y_true_2.shape)
        # print(gt_0.shape)
        # print(gt_1.shape)
        # print(gt_2.shape)
        input_shape = ops.shape(x)[2:4]
        input_shape = ops.cast(self.tenser_to_array(input_shape), ms.float32)
        yolo_out = self.yolo_network(x)
        loss_l = self.loss_big(*yolo_out[0], y_true_0, gt_0, input_shape)
        loss_m = self.loss_me(*yolo_out[1], y_true_1, gt_1, input_shape)
        loss_s = self.loss_small(*yolo_out[2], y_true_2, gt_2, input_shape)
        return loss_l + loss_m + loss_s


def set_graph_kernel_context():
    if ms.get_context("device_target") == "CPU":
        ms.set_context(enable_graph_kernel=True)
        ms.set_context(graph_kernel_flags="--enable_parallel_fusion "
                                          "--enable_trans_op_optimize "
                                          "--disable_cluster_ops=ReduceMax,Reshape "
                                          "--enable_expand_ops=Conv2D")


def keep_loss_fp32(network):
    """Keep loss of network with float32"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, (YoloLossBlock,)):
            cell.to_float(ms.float32)


def initialize(device):
    device_id = int(os.getenv('DEVICE_ID', '0'))
    ms.set_context(device_target=device, save_graphs=False, device_id=device_id)
    set_graph_kernel_context()


if __name__ == '__main__':
    eval_flag = False
    device = "CPU"
    initialize(device)
    network = yolov3(is_training=True)
    stree = SymbolTree.create(network)
    stree.print_node_tabulate()
    input = Tensor(np.ones([1, 3, 576, 576]), ms.float32)
    network(input)

