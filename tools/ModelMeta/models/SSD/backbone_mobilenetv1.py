import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from models.SSD.ssd_utils import ConvBNReLU, FeatureSelector, MultiBox


# class SSDWithMobileNetV1(nn.Cell):
#     def __init__(self, ):
#         super(SSDWithMobileNetV1, self).__init__()
#         cnn = [
#             conv_bn_relu(3, 32, 3, 2, False),  # Conv0
#             conv_bn_relu(32, 32, 3, 1, True),  # Conv1_depthwise
#             conv_bn_relu(32, 64, 1, 1, False),  # Conv1_pointwise
#             conv_bn_relu(64, 64, 3, 2, True),  # Conv2_depthwise
#             conv_bn_relu(64, 128, 1, 1, False),  # Conv2_pointwise
#             conv_bn_relu(128, 128, 3, 1, True),  # Conv3_depthwise
#             conv_bn_relu(128, 128, 1, 1, False),  # Conv3_pointwise
#             conv_bn_relu(128, 128, 3, 2, True),  # Conv4_depthwise
#             conv_bn_relu(128, 256, 1, 1, False),  # Conv4_pointwise
#             conv_bn_relu(256, 256, 3, 1, True),  # Conv5_depthwise
#             conv_bn_relu(256, 256, 1, 1, False),  # Conv5_pointwise
#             conv_bn_relu(256, 256, 3, 2, True),  # Conv6_depthwise
#             conv_bn_relu(256, 512, 1, 1, False),  # Conv6_pointwise
#             conv_bn_relu(512, 512, 3, 1, True),  # Conv7_depthwise
#             conv_bn_relu(512, 512, 1, 1, False),  # Conv7_pointwise
#             conv_bn_relu(512, 512, 3, 1, True),  # Conv8_depthwise
#             conv_bn_relu(512, 512, 1, 1, False),  # Conv8_pointwise
#             conv_bn_relu(512, 512, 3, 1, True),  # Conv9_depthwise
#             conv_bn_relu(512, 512, 1, 1, False),  # Conv9_pointwise
#             conv_bn_relu(512, 512, 3, 1, True),  # Conv10_depthwise
#             conv_bn_relu(512, 512, 1, 1, False),  # Conv10_pointwise
#             conv_bn_relu(512, 512, 3, 1, True),  # Conv11_depthwise
#             conv_bn_relu(512, 512, 1, 1, False),  # Conv11_pointwise
#             conv_bn_relu(512, 512, 3, 2, True),  # Conv12_depthwise
#             conv_bn_relu(512, 1024, 1, 1, False),  # Conv12_pointwise
#             conv_bn_relu(1024, 1024, 3, 1, True),  # Conv13_depthwise
#             conv_bn_relu(1024, 1024, 1, 1, False),  # Conv13_pointwise
#         ]
#         self.network = nn.CellList(cnn)
#         layer_indexs = [14, 26]
#         self.selector = FeatureSelector(layer_indexs)
#         in_channels = [256, 512, 1024, 512, 256, 256]
#         out_channels = [512, 1024, 512, 256, 256, 128]
#         strides = [1, 1, 2, 2, 2, 2]
#         residual_list = []
#         for i in range(2, len(in_channels)):
#             residual = ConvBNReLU(in_channels[i], out_channels[i], stride=strides[i], )
#             residual_list.append(residual)
#         self.multi_residual = nn.layer.CellList(residual_list)
#         self.multi_box = MultiBox(81, [512, 1024, 512, 256, 256, 128], [3, 6, 6, 6, 6, 6], 1917)
#
#     def construct(self, x):
#         output = x
#         features = ()
#         for block in self.network:
#             output = block(output)
#             features = features + (output,)
#         feature, output = self.selector(features)
#         print("feature shape: ", feature.shape)
#         print("output1 shape: ", output.shape)
#         multi_feature = (feature, output)
#         feature = output
#         print("feature", feature.shape)
#         for residual in self.multi_residual:
#             feature = residual(feature)
#             multi_feature += (feature,)
#
#         pred_loc, pred_label = self.multi_box(multi_feature)
#         if not self.training:
#             pred_label = ops.Sigmoid()(pred_label)
#
#         pred_loc = ops.cast(pred_loc, mindspore.float32)
#         pred_label = ops.cast(pred_label, mindspore.float32)
#         return pred_loc, pred_label


# class SSDWithMobileNetV1(nn.Cell):
#     def __init__(self):
#         super(SSDWithMobileNetV1, self).__init__()
#         # Define convolutional blocks explicitly
#         self.conv0 = conv_bn_relu(3, 32, 3, 2, False)
#         self.conv1_depthwise = conv_bn_relu(32, 32, 3, 1, True)
#         self.conv1_pointwise = conv_bn_relu(32, 64, 1, 1, False)
#         self.conv2_depthwise = conv_bn_relu(64, 64, 3, 2, True)
#         self.conv2_pointwise = conv_bn_relu(64, 128, 1, 1, False)
#         self.conv3_depthwise = conv_bn_relu(128, 128, 3, 1, True)
#         self.conv3_pointwise = conv_bn_relu(128, 128, 1, 1, False)
#         self.conv4_depthwise = conv_bn_relu(128, 128, 3, 2, True)
#         self.conv4_pointwise = conv_bn_relu(128, 256, 1, 1, False)
#         self.conv5_depthwise = conv_bn_relu(256, 256, 3, 1, True)
#         self.conv5_pointwise = conv_bn_relu(256, 256, 1, 1, False)
#         self.conv6_depthwise = conv_bn_relu(256, 256, 3, 2, True)
#         self.conv6_pointwise = conv_bn_relu(256, 512, 1, 1, False)
#         # Repetitive pattern noticed, continuing explicitly for all layers...
#         self.conv7_depthwise = conv_bn_relu(512, 512, 3, 1, True)
#         self.conv7_pointwise = conv_bn_relu(512, 512, 1, 1, False)
#         self.conv8_depthwise = conv_bn_relu(512, 512, 3, 1, True)
#         self.conv8_pointwise = conv_bn_relu(512, 512, 1, 1, False)
#         self.conv9_depthwise = conv_bn_relu(512, 512, 3, 1, True)
#         self.conv9_pointwise = conv_bn_relu(512, 512, 1, 1, False)
#         self.conv10_depthwise = conv_bn_relu(512, 512, 3, 1, True)
#         self.conv10_pointwise = conv_bn_relu(512, 512, 1, 1, False)
#         self.conv11_depthwise = conv_bn_relu(512, 512, 3, 1, True)
#         self.conv11_pointwise = conv_bn_relu(512, 512, 1, 1, False)
#         self.conv12_depthwise = conv_bn_relu(512, 512, 3, 2, True)
#         self.conv12_pointwise = conv_bn_relu(512, 1024, 1, 1, False)
#         self.conv13_depthwise = conv_bn_relu(1024, 1024, 3, 1, True)
#         self.conv13_pointwise = conv_bn_relu(1024, 1024, 1, 1, False)
#
#         # Feature selector
#         self.selector = FeatureSelector([14, 26])
#
#         # Define additional convolutional blocks for multi_residual
#         self.residual1 = ConvBNReLU(1024, 512, stride=2)
#         self.residual2 = ConvBNReLU(512, 256, stride=2)
#         self.residual3 = ConvBNReLU(256, 256, stride=2)
#         self.residual4 = ConvBNReLU(256, 128, stride=2)
#
#         # MultiBox
#         self.multi_box = MultiBox(81, [512, 1024, 512, 256, 256, 128], [3, 6, 6, 6, 6, 6], 1917)
#
#     def construct(self, x):
#         features = ()
#         x = self.conv0(x)
#         features = features + (x,)
#         x = self.conv1_depthwise(x)
#         features = features + (x,)
#         x = self.conv1_pointwise(x)
#         features = features + (x,)
#         x = self.conv2_depthwise(x)
#         features = features + (x,)
#         x = self.conv2_pointwise(x)
#         features = features + (x,)
#         x = self.conv3_depthwise(x)
#         features = features + (x,)
#         x = self.conv3_pointwise(x)
#         features = features + (x,)
#         x = self.conv4_depthwise(x)
#         features = features + (x,)
#         x = self.conv4_pointwise(x)
#         features = features + (x,)
#         x = self.conv5_depthwise(x)
#         features = features + (x,)
#         x = self.conv5_pointwise(x)
#         features = features + (x,)
#         x = self.conv6_depthwise(x)
#         features = features + (x,)
#         x = self.conv6_pointwise(x)
#         features = features + (x,)
#         x = self.conv7_depthwise(x)
#         features = features + (x,)
#         x = self.conv7_pointwise(x)
#         features = features + (x,)
#         x = self.conv8_depthwise(x)
#         features = features + (x,)
#         x = self.conv8_pointwise(x)
#         features = features + (x,)
#         x = self.conv9_depthwise(x)
#         features = features + (x,)
#         x = self.conv9_pointwise(x)
#         features = features + (x,)
#         x = self.conv10_depthwise(x)
#         features = features + (x,)
#         x = self.conv10_pointwise(x)
#         features = features + (x,)
#         x = self.conv11_depthwise(x)
#         features = features + (x,)
#         x = self.conv11_pointwise(x)
#         features = features + (x,)
#         x = self.conv12_depthwise(x)
#         features = features + (x,)
#         x = self.conv12_pointwise(x)
#         features = features + (x,)
#         x = self.conv13_depthwise(x)
#         features = features + (x,)
#         x = self.conv13_pointwise(x)
#         features = features + (x,)
#         # print("features: ", len(features))
#         feature, output = self.selector(features)
#         # print("feature shape: ", feature.shape)
#         # print("output1 shape: ", output.shape)
#         multi_feature = (feature, output)
#         # output = feature
#         # print("output shape: ", output.shape)
#         output = self.residual1(output)
#         multi_feature += (output,)
#         output = self.residual2(output)
#         multi_feature += (output,)
#         output = self.residual3(output)
#         multi_feature += (output,)
#         output = self.residual4(output)
#         multi_feature += (output,)
#
#         pred_loc, pred_label = self.multi_box(multi_feature)
#
#         if not self.training:
#             pred_label = ops.Sigmoid()(pred_label)
#
#         pred_loc = ops.cast(pred_loc, mindspore.float32)
#         pred_label = ops.cast(pred_label, mindspore.float32)
#
#         return pred_loc, pred_label


class SSDWithMobileNetV1(nn.Cell):
    def __init__(self):
        super(SSDWithMobileNetV1, self).__init__()
        # Convolution Block 0
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, pad_mode="pad", padding=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.relu0 = nn.ReLU6()

        # Block 1 - Depthwise
        self.conv1_depthwise = nn.Conv2d(32, 32, kernel_size=3, stride=1, pad_mode="pad", padding=1, group=32)
        self.bn1_depthwise = nn.BatchNorm2d(32)
        self.relu1_depthwise = nn.ReLU6()
        # Block 1 - Pointwise
        self.conv1_pointwise = nn.Conv2d(32, 64, kernel_size=1, stride=1, pad_mode="pad", padding=0)
        self.bn1_pointwise = nn.BatchNorm2d(64)
        self.relu1_pointwise = nn.ReLU6()

        # Block 2 - Depthwise
        self.conv2_depthwise = nn.Conv2d(64, 64, kernel_size=3, stride=2, pad_mode="pad", padding=1, group=64)
        self.bn2_depthwise = nn.BatchNorm2d(64)
        self.relu2_depthwise = nn.ReLU6()
        # Block 2 - Pointwise
        self.conv2_pointwise = nn.Conv2d(64, 128, kernel_size=1, stride=1, pad_mode="pad", padding=0)
        self.bn2_pointwise = nn.BatchNorm2d(128)
        self.relu2_pointwise = nn.ReLU6()

        # Block 3 - Depthwise
        self.conv3_depthwise = nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode="pad", padding=1, group=128)
        self.bn3_depthwise = nn.BatchNorm2d(128)
        self.relu3_depthwise = nn.ReLU6()
        # Block 3 - Pointwise
        self.conv3_pointwise = nn.Conv2d(128, 128, kernel_size=1, stride=1, pad_mode="pad", padding=0)
        self.bn3_pointwise = nn.BatchNorm2d(128)
        self.relu3_pointwise = nn.ReLU6()

        # Block 4 - Depthwise
        self.conv4_depthwise = nn.Conv2d(128, 128, kernel_size=3, stride=2, pad_mode="pad", padding=1, group=128)
        self.bn4_depthwise = nn.BatchNorm2d(128)
        self.relu4_depthwise = nn.ReLU6()
        # Block 4 - Pointwise
        self.conv4_pointwise = nn.Conv2d(128, 256, kernel_size=1, stride=1, pad_mode="pad", padding=0)
        self.bn4_pointwise = nn.BatchNorm2d(256)
        self.relu4_pointwise = nn.ReLU6()

        # Block 5 - Depthwise
        self.conv5_depthwise = nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1, group=256)
        self.bn5_depthwise = nn.BatchNorm2d(256)
        self.relu5_depthwise = nn.ReLU6()
        # Block 5 - Pointwise
        self.conv5_pointwise = nn.Conv2d(256, 256, kernel_size=1, stride=1, pad_mode="pad", padding=0)
        self.bn5_pointwise = nn.BatchNorm2d(256)
        self.relu5_pointwise = nn.ReLU6()

        # Block 6 - Depthwise
        self.conv6_depthwise = nn.Conv2d(256, 256, kernel_size=3, stride=2, pad_mode="pad", padding=1, group=256)
        self.bn6_depthwise = nn.BatchNorm2d(256)
        self.relu6_depthwise = nn.ReLU6()
        # Block 6 - Pointwise
        self.conv6_pointwise = nn.Conv2d(256, 512, kernel_size=1, stride=1, pad_mode="pad", padding=0)
        self.bn6_pointwise = nn.BatchNorm2d(512)
        self.relu6_pointwise = nn.ReLU6()

        # Blocks 7-11 are repetitive, maintaining the structure but not depthwise stride
        for i in range(7, 12):
            setattr(self, f"conv{i}_depthwise",
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode="pad", padding=1, group=512))
            setattr(self, f"bn{i}_depthwise", nn.BatchNorm2d(512))
            setattr(self, f"relu{i}_depthwise", nn.ReLU6())
            setattr(self, f"conv{i}_pointwise", nn.Conv2d(512, 512, kernel_size=1, stride=1, pad_mode="pad", padding=0))
            setattr(self, f"bn{i}_pointwise", nn.BatchNorm2d(512))
            setattr(self, f"relu{i}_pointwise", nn.ReLU6())

        # Block 12 - Depthwise
        self.conv12_depthwise = nn.Conv2d(512, 512, kernel_size=3, stride=2, pad_mode="pad", padding=1, group=512)
        self.bn12_depthwise = nn.BatchNorm2d(512)
        self.relu12_depthwise = nn.ReLU6()
        # Block 12 - Pointwise
        self.conv12_pointwise = nn.Conv2d(512, 1024, kernel_size=1, stride=1, pad_mode="pad", padding=0)
        self.bn12_pointwise = nn.BatchNorm2d(1024)
        self.relu12_pointwise = nn.ReLU6()

        # Block 13 - Depthwise
        self.conv13_depthwise = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, pad_mode="pad", padding=1, group=1024)
        self.bn13_depthwise = nn.BatchNorm2d(1024)
        self.relu13_depthwise = nn.ReLU6()
        # Block 13 - Pointwise
        self.conv13_pointwise = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, pad_mode="pad", padding=0)
        self.bn13_pointwise = nn.BatchNorm2d(1024)
        self.relu13_pointwise = nn.ReLU6()

        # Feature selector
        self.selector = FeatureSelector([14, 26])
        # Define additional convolutional blocks for multi_residual
        self.residual1 = ConvBNReLU(1024, 512, stride=2)
        self.residual2 = ConvBNReLU(512, 256, stride=2)
        self.residual3 = ConvBNReLU(256, 256, stride=2)
        self.residual4 = ConvBNReLU(256, 128, stride=2)
        # MultiBox
        self.multi_box = MultiBox(81, [512, 1024, 512, 256, 256, 128], [3, 6, 6, 6, 6, 6], 1917)

    def construct(self, x):
        features = ()
        # Conv Block 0
        x = self.relu0(self.bn0(self.conv0(x)))
        features = features + (x,)
        # Block 1
        x = self.relu1_depthwise(self.bn1_depthwise(self.conv1_depthwise(x)))
        features = features + (x,)
        x = self.relu1_pointwise(self.bn1_pointwise(self.conv1_pointwise(x)))
        features = features + (x,)

        # Block 2
        x = self.relu2_depthwise(self.bn2_depthwise(self.conv2_depthwise(x)))
        features = features + (x,)
        x = self.relu2_pointwise(self.bn2_pointwise(self.conv2_pointwise(x)))
        features = features + (x,)

        # Block 3
        x = self.relu3_depthwise(self.bn3_depthwise(self.conv3_depthwise(x)))
        features = features + (x,)
        x = self.relu3_pointwise(self.bn3_pointwise(self.conv3_pointwise(x)))
        features = features + (x,)

        # Block 4
        x = self.relu4_depthwise(self.bn4_depthwise(self.conv4_depthwise(x)))
        features = features + (x,)
        x = self.relu4_pointwise(self.bn4_pointwise(self.conv4_pointwise(x)))
        features = features + (x,)

        # Block 5
        x = self.relu5_depthwise(self.bn5_depthwise(self.conv5_depthwise(x)))
        features = features + (x,)
        x = self.relu5_pointwise(self.bn5_pointwise(self.conv5_pointwise(x)))
        features = features + (x,)

        # Block 6
        x = self.relu6_depthwise(self.bn6_depthwise(self.conv6_depthwise(x)))
        features = features + (x,)
        x = self.relu6_pointwise(self.bn6_pointwise(self.conv6_pointwise(x)))
        features = features + (x,)
        # Blocks 7 to 11
        for i in range(7, 12):
            x = getattr(self, f"relu{i}_depthwise")(
                getattr(self, f"bn{i}_depthwise")(getattr(self, f"conv{i}_depthwise")(x)))
            features = features + (x,)
            x = getattr(self, f"relu{i}_pointwise")(
                getattr(self, f"bn{i}_pointwise")(getattr(self, f"conv{i}_pointwise")(x)))
            features = features + (x,)
        # Block 12
        x = self.relu12_depthwise(self.bn12_depthwise(self.conv12_depthwise(x)))
        features = features + (x,)
        x = self.relu12_pointwise(self.bn12_pointwise(self.conv12_pointwise(x)))
        features = features + (x,)
        # Block 13
        x = self.relu13_depthwise(self.bn13_depthwise(self.conv13_depthwise(x)))
        features = features + (x,)
        x = self.relu13_pointwise(self.bn13_pointwise(self.conv13_pointwise(x)))
        features = features + (x,)
        # print("features: ", len(features))
        feature, output = self.selector(features)
        # print("feature shape: ", feature.shape)
        # print("output1 shape: ", output.shape)
        multi_feature = (feature, output)
        # print("output shape: ", output.shape)
        output = self.residual1(output)
        multi_feature += (output,)
        output = self.residual2(output)
        multi_feature += (output,)
        output = self.residual3(output)
        multi_feature += (output,)
        output = self.residual4(output)
        multi_feature += (output,)

        pred_loc, pred_label = self.multi_box(multi_feature)

        if not self.training:
            pred_label = ops.Sigmoid()(pred_label)

        pred_loc = ops.cast(pred_loc, mindspore.float32)
        pred_label = ops.cast(pred_label, mindspore.float32)

        return pred_loc, pred_label


class FlattenConcat(nn.Cell):
    """FlattenConcat module."""

    def __init__(self):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = 8732

    def construct(self, inputs):
        output = ()
        batch_size = ops.shape(inputs[0])[0]
        for x in inputs:
            x = ops.transpose(x, (0, 2, 3, 1))
            output = output + (ops.reshape(x, (batch_size, -1)),)
        res = ops.concat(output, axis=1)
        return ops.reshape(res, (batch_size, self.num_ssd_boxes, -1))


if __name__ == '__main__':
    image = mindspore.Tensor(np.random.rand(4, 3, 300, 300), mindspore.float32)
    num_matched_boxes = mindspore.Tensor([[33]], mindspore.int32)
    gt_label = mindspore.Tensor(np.random.randn(1, 1917), mindspore.int32)  # np.load("./official/gt_label.npy")
    get_loc = mindspore.Tensor(np.random.randn(1, 1917, 4),
                               mindspore.float32)  # mindspore.Tensor(np.load("./official/get_loc.npy"), mindspore.float32)

    network = SSDWithMobileNetV1()
    result1, result2 = network(image)
    print(result1.shape)
    print(result2.shape)
    # new_network = SSDWithMobileNetV1_new()
    # result1, result2 = new_network(image)
    # print(result1.shape)
    # print(result2.shape)

    # # Define the learning rate
    # lr = 1e-4
    #
    # # Define the optimizer
    # opt = nn.Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), lr,
    #                   0.9, 0.00015, float(1024))
    #
    #
    # # Define the forward procedure
    # def forward_fn(x, gt_loc, gt_label, num_matched_boxes):
    #
    #     pred_loc, pred_label = network(x)
    #
    #
    #
    #     mask = ops.less(0, gt_label).astype(mindspore.float32)
    #     num_matched_boxes = ops.sum(num_matched_boxes.astype(mindspore.float32))
    #
    #     # Positioning loss
    #     mask_loc = ops.tile(ops.expand_dims(mask, -1), (1, 1, 4))
    #     smooth_l1 = nn.SmoothL1Loss()(pred_loc, gt_loc) * mask_loc
    #     loss_loc = ops.sum(ops.sum(smooth_l1, -1), -1)
    #
    #     # Category loss
    #     loss_cls = class_loss(pred_label, gt_label)
    #     loss_cls = ops.sum(loss_cls, (1, 2))
    #
    #     return ops.sum((loss_cls + loss_loc) / num_matched_boxes)
    #
    #
    # grad_fn = mindspore.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)
    #
    #
    #
    # # Gradient updates
    # def train_step(x, gt_loc, gt_label, num_matched_boxes):
    #     loss, grads = grad_fn(x, gt_loc, gt_label, num_matched_boxes)
    #     opt(grads)
    #     return loss
    #
    #
    # for epoch in range(5):
    #     network.set_train(True)
    #     loss=train_step(image, get_loc, gt_label, num_matched_boxes)
    #     print("loss: "+str(loss))

    # #fpn topdown
    # layer_indexs = [10, 22, 26]
    # features = ()
    # for i in layer_indexs:
    #     features = features + (feature_list[i],)
    # fpn = FpnTopDown([256, 512, 1024], 256)
    # bottom_up = BottomUp(2, 256, 3, 2)
    # features = fpn(features)
    # features = bottom_up(features)

    # #multi box
    # from src.model_utils.config import config
    # from ssd_utils import WeightSharedMultiBox
    # multi_box = WeightSharedMultiBox(config)
    # pred_loc, pred_label = multi_box(features)
    #
    # op=FlattenConcat()
    # print(op(pred_loc).shape)
