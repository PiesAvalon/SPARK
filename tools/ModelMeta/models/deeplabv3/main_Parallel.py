import os
import platform
import mindspore.common.dtype as mstype
import cv2
import numpy as np
from configs.DeeplabConfig import config
import mindspore
from models.deeplabv3 import main
from mindspore import nn, Tensor
from models.deeplabv3.Deeplabv3 import DeepLabV3


def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


def eval_batch_scales_ms(args, eval_net, img_lst, scales,
                         base_crop_size=513, flip=True):
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = eval_batch_ms(args, eval_net, img_lst, crop_size=sizes_[0], flip=flip)
    # print(sizes_)
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_batch_ms(args, eval_net, img_lst, crop_size=crop_size_, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))
    return result_msk


def eval_batch_ms(args, eval_net, img_lst, crop_size=513, flip=True):
    result_lst = []
    batch_size = len(img_lst)
    batch_img = np.zeros((args.batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    for l in range(batch_size):
        img_ = img_lst[l]
        img_, resize_h, resize_w = pre_process(args, img_, crop_size)
        batch_img[l] = img_
        resize_hw.append([resize_h, resize_w])

    batch_img = np.ascontiguousarray(batch_img)
    net_out = eval_net(Tensor(batch_img, mstype.float32))
    net_out = net_out.asnumpy()

    if flip:
        batch_img = batch_img[:, :, :, ::-1].copy()
        net_out_flip = eval_net(Tensor(batch_img, mstype.float32))
        net_out += net_out_flip.asnumpy()[:, :, :, ::-1]

    for bs in range(batch_size):
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        result_lst.append(probs_)
    return result_lst


def pre_process(args, img_, crop_size=513):
    # resize
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(args.image_mean)
    image_std = np.array(args.image_std)
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_, resize_h, resize_w


def get_loss():
    loser2 = main.SoftmaxCrossEntropyLoss(num_cls=21, ignore_label=255)
    return loser2


def train_eval_deeplabv3(model1, model2, data_dir, batch_size, now_time):
    epoch_num = 36
    args = config
    path = data_dir.split("/")[:-1]
    path = "/".join(path)
    path = os.path.join(path, "voc_val_lst.txt")
    args.data_lst = path
    args.data_root = data_dir
    args.batch_size = batch_size
    try:
        # 此处不能用CIFAR10!!!!!
        dataset = main.SegDataset(image_mean=args.image_mean,
                                  image_std=args.image_std,
                                  data_file=data_dir,
                                  batch_size=args.batch_size,
                                  crop_size=args.crop_size,
                                  max_scale=args.max_scale,
                                  min_scale=args.min_scale,
                                  ignore_label=args.ignore_label,
                                  num_classes=args.num_classes,
                                  num_readers=2,
                                  num_parallel_calls=4,
                                  shard_id=args.rank,
                                  shard_num=args.group_size)
        dataset = dataset.get_dataset(repeat=1)
    except Exception as e:
        print("dataset error", e, "try cifar10")
        print(e)
        exit(666)

    loser1 = get_loss()
    loser2 = get_loss()
    model_name = str(model1.__class__.__name__)
    if os.path.exists("log") is False:
        os.mkdir("log")
    f = open(os.path.join("mutated_net/" + str(model1.__class__.__name__)+"/", str(now_time),
                          "/loss_" + str(platform.platform()) + str(mindspore.get_context('device_target')) + ".txt"),
             "w")
    model_old_trainable_params = model1.trainable_params()
    new_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in model_old_trainable_params:
        modelms_trainable_param.name = model_name + str(
            layer_nums) + "_" + modelms_trainable_param.name
        new_trainable_params.append(modelms_trainable_param)
        layer_nums += 1
    model_mutant_trainable_params = model2.trainable_params()
    mutant_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in model_mutant_trainable_params:
        modelms_trainable_param.name = model_name + str(
            layer_nums) + "_" + modelms_trainable_param.name
        mutant_trainable_params.append(modelms_trainable_param)
        layer_nums += 1
    opt1 = nn.SGD(params=filter(lambda x: x.requires_grad, model1.get_parameters()), learning_rate=0.0005, momentum=0.9,
                  weight_decay=0.0001)
    opt2 = nn.SGD(params=filter(lambda x: x.requires_grad, model2.get_parameters()), learning_rate=0.0005, momentum=0.9,
                  weight_decay=0.0001)
    ds = dataset.create_tuple_iterator(output_numpy=True)

    def forward_fn1(data, label):
        output = model1(data)
        loss = loser1(output, label)
        return loss

    def forward_fn2(data, label):
        output = model2(data)
        loss = loser2(output, label)
        return loss

    grad_fn1 = mindspore.ops.value_and_grad(forward_fn1, None, opt1.parameters, has_aux=False)
    grad_fn2 = mindspore.ops.value_and_grad(forward_fn2, None, opt2.parameters, has_aux=False)

    def train_step1(data, label):
        (loss), grads = grad_fn1(data, label)
        loss = mindspore.ops.depend(loss, opt1(grads))
        return loss

    def train_step2(data, label):
        (loss), grads = grad_fn2(data, label)
        loss = mindspore.ops.depend(loss, opt2(grads))
        return loss

    per_batch = 200
    losses_ms_avg1 = []
    losses_ms_avg2 = []
    try:
        with open(args.data_lst) as f1:
            img_lst = f1.readlines()
    except Exception as e:
        print("Please generate data list file first.", e)
        exit(666)
    for epoch in range(epoch_num):
        nums = 0
        losses_ms1 = []
        losses_ms2 = []
        for data in ds:
            nums += data[0].shape[0]
            data0 = mindspore.Tensor(data[0], dtype=mindspore.float32)
            data1 = mindspore.Tensor(data[1], dtype=mindspore.float32)
            loss_ms1 = train_step1(data0, data1)
            loss_ms2 = train_step2(data0, data1)
            # print(loss_ms)
            if nums % per_batch == 0:
                print("batch:" + str(nums) + " ms_loss1:" + str(
                    loss_ms1.asnumpy()) + " ms_loss2:" + str(
                    loss_ms2.asnumpy()))
                f.write("batch:" + str(nums) + " ms_loss1:" + str(
                    loss_ms1.asnumpy()) + " ms_loss2:" + str(
                    loss_ms2.asnumpy()) + "\n")
            losses_ms1.append(loss_ms1.asnumpy())
            losses_ms2.append(loss_ms2.asnumpy())
            break
        losses_ms_avg1.append(np.mean(losses_ms1))
        losses_ms_avg2.append(np.mean(losses_ms2))
        print("epoch {}: ".format(epoch), " ms_loss1: ",
              str(np.mean(losses_ms1)), " ms_loss2: ",
              str(np.mean(losses_ms2)) + "\n")
        f.write("epoch {}: ".format(epoch) + " ms_loss1: "
                + str(np.mean(losses_ms1)) + " ms_loss2: "
                + str(np.mean(losses_ms2)) + "\n")
        model1.set_train(False)
        model2.set_train(False)
        hist_ms1 = np.zeros((args.num_classes, args.num_classes))
        hist_ms2 = np.zeros((args.num_classes, args.num_classes))
        batch_img_lst = []
        batch_msk_lst = []
        bi = 0
        image_num = 0
        for i, line in enumerate(img_lst):
            img_path, msk_path = line.strip().split(' ')
            img_path = os.path.join(args.data_root, img_path)
            msk_path = os.path.join(args.data_root, msk_path)
            img_ = cv2.imread(img_path)
            msk_ = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            batch_img_lst.append(img_)
            batch_msk_lst.append(msk_)
            bi += 1
            if bi == args.batch_size:
                batch_res_ms1 = eval_batch_scales_ms(args, model1, batch_img_lst, scales=args.scales,
                                                     base_crop_size=args.crop_size, flip=args.flip)
                batch_res_ms2 = eval_batch_scales_ms(args, model2, batch_img_lst, scales=args.scales,
                                                     base_crop_size=args.crop_size, flip=args.flip)
                for mi in range(args.batch_size):
                    hist_ms1 += cal_hist(batch_msk_lst[mi].flatten(), batch_res_ms1[mi].flatten(), args.num_classes)
                    hist_ms2 += cal_hist(batch_msk_lst[mi].flatten(), batch_res_ms2[mi].flatten(), args.num_classes)
                bi = 0
                batch_img_lst = []
                batch_msk_lst = []
            image_num = i
            break

        if bi > 0:
            batch_res_ms1 = eval_batch_scales_ms(args, model1, batch_img_lst, scales=args.scales,
                                                 base_crop_size=args.crop_size, flip=args.flip)
            batch_res_ms2 = eval_batch_scales_ms(args, model2, batch_img_lst, scales=args.scales,
                                                 base_crop_size=args.crop_size, flip=args.flip)
            for mi in range(bi):
                hist_ms1 += cal_hist(batch_msk_lst[mi].flatten(), batch_res_ms1[mi].flatten(), args.num_classes)
                hist_ms2 += cal_hist(batch_msk_lst[mi].flatten(), batch_res_ms2[mi].flatten(), args.num_classes)

            print('processed {} images'.format(image_num + 1))

        iou_ms1 = np.diag(hist_ms1) / (hist_ms1.sum(1) + hist_ms1.sum(0) - np.diag(hist_ms1))
        iou_ms2 = np.diag(hist_ms2) / (hist_ms2.sum(1) + hist_ms2.sum(0) - np.diag(hist_ms2))
        print('mindspore_per-class IoU old', iou_ms1)
        print('mindspore_per-class IoU new', iou_ms2)
        f.write('mindspore_per-class IoU old' + str(iou_ms1))
        f.write('mindspore_per-class IoU new' + str(iou_ms2))
        print('mindspore_mean IoU old', np.nanmean(iou_ms1))
        print('mindspore_mean IoU new', np.nanmean(iou_ms2))
        f.write(' mindspore_mean IoU old' + str(np.nanmean(iou_ms1)))
        f.write(' mindspore_mean IoU new' + str(np.nanmean(iou_ms2)))


if __name__ == '__main__':
    model1 = DeepLabV3('train', 21, 8, False)
    model2 = DeepLabV3('train', 21, 8, False)
    train_eval_deeplabv3(model1, model2, "../../datasets/cifar10")
