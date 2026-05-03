import mindspore
import mindspore.ops as ops
import torch


def loss_SSDmultibox_ms():
    return SSDmultibox_ms_cal


def SSDmultibox_ms_cal(pred_loc, pred_label, gt_loc, gt_label, num_matched_boxes):
    mask = ops.less(0, gt_label).astype(mindspore.float32)
    num_matched_boxes = mindspore.numpy.sum(num_matched_boxes.astype(mindspore.float32))

    # Positioning loss
    mask_loc = ops.tile(ops.expand_dims(mask, -1), (1, 1, 4))
    smooth_l1 = mindspore.nn.SmoothL1Loss()(pred_loc, gt_loc) * mask_loc
    loss_loc = mindspore.numpy.sum(mindspore.numpy.sum(smooth_l1, -1), -1)

    # Category loss
    from network.cv.SSD.ssd_utils import class_loss
    loss_cls = class_loss(pred_label, gt_label)
    loss_cls = mindspore.numpy.sum(loss_cls, (1, 2))

    return mindspore.numpy.sum((loss_cls + loss_loc) / num_matched_boxes)


class loss_SSDmultibox_torch(torch.nn.Module):
    def __init__(self):
        super(loss_SSDmultibox_torch, self).__init__()

    def forward(self, pred_loc, pred_label, gt_loc, gt_label, num_matched_boxes):
        mask = (gt_label > 0).float()
        num_matched_boxes = num_matched_boxes.float().sum()

        # Positioning loss
        mask_loc = mask.unsqueeze(-1).repeat(1, 1, 4)
        smooth_l1 = torch.nn.SmoothL1Loss(reduction='none')(pred_loc, gt_loc) * mask_loc
        loss_loc = smooth_l1.sum(dim=-1).sum(dim=-1)

        # Category loss
        from network.cv.SSD.ssd_utils_torch import class_loss
        loss_cls = class_loss(pred_label, gt_label)
        loss_cls = loss_cls.sum(dim=(1, 2))

        return ((loss_cls + loss_loc) / num_matched_boxes).sum()


class loss_fasttext_torch(torch.nn.Module):
    def __init__(self):
        super(loss_fasttext_torch, self).__init__()

    def forward(self, predict_score, label_idx):
        label_idx = torch.squeeze(input=label_idx, dim=1)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')(predict_score, label_idx)

        return loss


def loss_fasttext_ms(predict_score, label_idx):
    loss_func = mindspore.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    squeeze = ops.Squeeze(axis=1)

    label_idx = squeeze(label_idx)
    loss = loss_func(predict_score, label_idx)
    return loss


def loss_yolo_ms():
    from network.cv.yolov4.main_new import yolov4loss_ms
    return yolov4loss_ms()


def loss_yolo_torch():
    from network.cv.yolov4.yolov4_pytorch import yolov4loss_torch
    return yolov4loss_torch()


def loss_unet_ms():
    from network.cv.unet.unet_model import CrossEntropyWithLogits
    return CrossEntropyWithLogits()


def loss_unet_torch():
    from network.cv.unet.main_torch import CrossEntropyWithLogits
    return CrossEntropyWithLogits()


def loss_textcnn_ms():
    from network.nlp.textcnn.run_train import loss_com_ms
    return loss_com_ms


def loss_textcnn_torch():
    from network.nlp.textcnn.run_train import loss_torch
    return loss_torch()


def loss_fasttext_ms():
    from network.nlp.fasttext.run_train import loss_ms
    return loss_ms


def loss_maskrcnn_torch():
    from network.cv.MaskRCNN.maskrcnn_resnet50.main_torch import mask_rcnn_loss as loss_torch
    return loss_torch()


def loss_maskrcnn_ms():
    from network.cv.MaskRCNN.maskrcnn_resnet50.main_new import mask_rcnn_loss as loss_ms
    return loss_ms()


def loss_deepv3plus_torch():
    from network.cv.deeplabv3plus.main_torch import deeplabv3_torch
    return deeplabv3_torch()


def loss_deepv3plus_ms():
    from network.cv.deeplabv3plus.main import deeplabv3_mindspore as loss_ms
    return loss_ms


def loss_unet3d_ms():
    from network.cv.Unet3d.main import unet3d_loss_ms as loss_ms
    return loss_ms()


def loss_unet3d_torch():
    from network.cv.Unet3d.main_torch import loss_unet3d_torch as loss_torch
    return loss_torch()


def loss_transformer_torch():
    from network.nlp.transformer.main_torch import get_loss
    return get_loss()


def loss_transformer_ms():
    from network.nlp.transformer.main import get_loss
    return get_loss()


def loss_bert_torch():
    from network.nlp.bert.main_squad_torch import loss_torch
    return loss_torch()


def loss_bert_ms():
    from network.nlp.bert.main_squad import loss_ms
    return loss_ms


def loss_retinaface_torch():
    from network.cv.retinaface.main_torch import loss_retinaface_torch
    loss_torch = loss_retinaface_torch()
    return loss_torch


def loss_retinaface_ms():
    from network.cv.retinaface.main_ms import retinafaceloss_ms
    loss_ms = retinafaceloss_ms()
    return loss_ms


def loss_pangu_ms():
    from network.nlp.S_Pangu_alpha.main_single import loss_ms
    loss_ms_fun = loss_ms()
    return loss_ms_fun


def loss_pangu_torch():
    from network.nlp.S_Pangu_alpha.main_torch_single import loss_torch
    loss_torch_fun = loss_torch()
    return loss_torch_fun


def get_loss(loss_name):
    loss = {}
    loss['CrossEntropy'] = [mindspore.nn.CrossEntropyLoss, torch.nn.CrossEntropyLoss]
    loss['ssdmultix'] = [loss_SSDmultibox_ms, loss_SSDmultibox_torch]
    loss['retinafacemultix'] = [loss_retinaface_ms, loss_retinaface_torch]
    loss['yololoss'] = [loss_yolo_ms, loss_yolo_torch]
    loss['unetloss'] = [loss_unet_ms, loss_unet_torch]
    loss['unet3dloss'] = [loss_unet3d_ms, loss_unet3d_torch]
    loss['fasttextloss'] = [loss_fasttext_ms, loss_fasttext_torch]
    loss['textcnnloss'] = [loss_textcnn_ms, loss_textcnn_torch]
    loss['deepv3plusloss'] = [loss_deepv3plus_ms, loss_deepv3plus_torch]
    loss['transformerloss'] = [loss_transformer_ms, loss_transformer_torch]
    loss['bertloss'] = [loss_bert_ms, loss_bert_torch]
    loss['rcnnloss'] = [loss_maskrcnn_ms, loss_maskrcnn_torch]
    loss['panguloss'] = [loss_pangu_ms, loss_pangu_torch]

    return loss[loss_name]
