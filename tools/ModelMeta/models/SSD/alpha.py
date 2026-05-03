from mindspore import nn, ops
import mindspore as ms
network = SSD300Vgg16()
init_net_param(network)
opt = nn.Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), lr,
                  0.9, 0.00015, float(1024))


# Define the forward procedure
def forward_fn(x, gt_loc, gt_label, num_matched_boxes):
    pred_loc, pred_label = network(x)
    mask = ops.less(0, gt_label).astype(ms.float32)
    num_matched_boxes = ops.sum(num_matched_boxes.astype(ms.float32))

    # Positioning loss
    mask_loc = ops.tile(ops.expand_dims(mask, -1), (1, 1, 4))
    smooth_l1 = nn.SmoothL1Loss()(pred_loc, gt_loc) * mask_loc
    loss_loc = ops.sum(ops.sum(smooth_l1, -1), -1)

    # Category loss
    loss_cls = class_loss(pred_label, gt_label)
    loss_cls = ops.sum(loss_cls, (1, 2))

    return ops.sum((loss_cls + loss_loc) / num_matched_boxes)


grad_fn = ms.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)
loss_scaler = DynamicLossScaler(1024, 2, 1000)


# Gradient updates
def train_step(x, gt_loc, gt_label, num_matched_boxes):
    loss, grads = grad_fn(x, gt_loc, gt_label, num_matched_boxes)
    grads = loss_scaler.scale(grads)
    opt(grads)
    return loss