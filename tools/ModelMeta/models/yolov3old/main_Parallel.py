import os
import platform
import time
import mindspore
import mindspore as ms
import numpy as np
from mindspore import nn
from configs.yolov3config import config
from models.yolov3.main_new import losser, create_yolo_dataset, conver_testing_shape, get_param_groups
from models.yolov3.util import DetectionEngine
import psutil


def train_eval_yolov3(model1, model2, data_dir, batch_size, now_time):
    per_batch_size = batch_size
    config.max_epoch = 6
    loser1 = losser(model1, config)
    loser2 = losser(model2, config)
    if not os.path.exists("log"):
        os.makedirs("log")
    f = open(os.path.join("mutated_net/" + str(model1.__class__.__name__)+"/", str(now_time),
                          "/loss_" + str(platform.platform()) + str(mindspore.get_context('device_target')) + ".txt"),
             "w")
    # config.data_root = os.path.join(config.data_dir, 'train2014')
    # config.annFile 为评估调用的
    config.data_dir = data_dir
    config.annFile = os.path.join(config.data_dir, 'annotations/instances_val2014.json')
    train_root = os.path.join(config.data_dir, 'train2014')
    # set dataset path temporarily to val set to reduce time cost
    test_root = os.path.join(config.data_dir, 'val2014')
    train_dataset = create_yolo_dataset(image_dir=train_root,
                                        anno_path=os.path.join(config.data_dir,
                                                               'annotations/instances_train2014.json'),
                                        is_training=True,
                                        batch_size=per_batch_size, device_num=1, shuffle=False,
                                        rank=0)
    mem = psutil.virtual_memory()
    print('当前可用内存：%.4f GB' % (mem.available / 1024 / 1024 / 1024))
    f.write('当前可用内存：%.4f GB' % (mem.available / 1024 / 1024 / 1024))
    dataiter = train_dataset.create_dict_iterator(output_numpy=True)
    # lr = main.get_lr(config)
    # print("lr:", lr)
    opt1 = nn.SGD(params=get_param_groups(loser1), momentum=config.momentum, learning_rate=1e-5,
                  weight_decay=config.weight_decay)
    opt2 = nn.SGD(params=get_param_groups(loser2), momentum=config.momentum, learning_rate=1e-5,
                  weight_decay=config.weight_decay)
    if config.testing_shape:
        config.test_img_shape = conver_testing_shape(config)
    testset = create_yolo_dataset(test_root, os.path.join(config.data_dir, 'annotations/instances_val2014.json'),
                                  is_training=False,
                                  batch_size=per_batch_size, device_num=1,
                                  rank=0, shuffle=False)
    testdata = testset.create_dict_iterator(num_epochs=1, output_numpy=True)
    steps_per_epoch = train_dataset.get_dataset_size()
    print("steps_per_epoch", steps_per_epoch)

    def forward_fn1(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        loss = loser1(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2)
        return loss

    def forward_fn2(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        loss = loser2(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2)
        return loss

    grad_fn1 = mindspore.ops.value_and_grad(forward_fn1, None, opt1.parameters, has_aux=False)
    grad_fn2 = mindspore.ops.value_and_grad(forward_fn2, None, opt2.parameters, has_aux=False)

    def train_step1(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        (loss), grads = grad_fn1(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2)
        loss = mindspore.ops.depend(loss, opt1(grads))
        return loss

    def train_step2(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        (loss), grads = grad_fn2(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2)
        loss = mindspore.ops.depend(loss, opt2(grads))
        return loss

    per_batch = 400
    losses_ms_avg1, losses_ms_avg2 = [], []
    for epoch_idx in range(config.max_epoch):
        model1.set_train()
        model2.set_train()
        # loser1.set_train()
        # loser2.set_train()
        nums = 0
        losses_ms1, losses_ms2 = [], []
        print("epoch_idx", epoch_idx, "started")
        for step_idx, data in enumerate(dataiter):
            if epoch_idx == 0 and step_idx == 0:
                print("entered the first step")
            # print("data['image'].shape[0]",data['image'].shape[0])
            nums += data['image'].shape[0]
            # print("nums", str(nums))
            # print('iter[{}], shape{}'.format(step_idx, input_shape[0]))
            images_ms = ms.Tensor.from_numpy(data["image"])
            batch_y_true_0_ms = ms.Tensor.from_numpy(data['bbox1'])
            batch_y_true_1_ms = ms.Tensor.from_numpy(data['bbox2'])
            batch_y_true_2_ms = ms.Tensor.from_numpy(data['bbox3'])
            batch_gt_box0_ms = ms.Tensor.from_numpy(data['gt_box1'])
            batch_gt_box1_ms = ms.Tensor.from_numpy(data['gt_box2'])
            batch_gt_box2_ms = ms.Tensor.from_numpy(data['gt_box3'])
            loss1 = train_step1(images_ms, batch_y_true_0_ms, batch_y_true_1_ms, batch_y_true_2_ms, batch_gt_box0_ms,
                                batch_gt_box1_ms,
                                batch_gt_box2_ms)
            loss2 = train_step2(images_ms, batch_y_true_0_ms, batch_y_true_1_ms, batch_y_true_2_ms, batch_gt_box0_ms,
                                batch_gt_box1_ms,
                                batch_gt_box2_ms)
            if nums % per_batch == 0:
                print("batch:" + str(nums) + " ms_loss1:" + str(loss1.asnumpy()) + " ms_loss2:" + str(
                    loss2.asnumpy()))
                f.write("batch:" + str(nums) + " ms_loss1:" + str(loss1.asnumpy()) + " ms_loss2:" + str(
                    loss2.asnumpy()) + "\n")
            losses_ms1.append(loss1.asnumpy())
            losses_ms2.append(loss2.asnumpy())
            # break
        losses_ms_avg1.append(np.mean(losses_ms1))
        losses_ms_avg2.append(np.mean(losses_ms2))
        print("epoch {}: ".format(epoch_idx), "losses_ms1: ", str(np.mean(losses_ms1)), " losses_ms2: ",
              str(np.mean(losses_ms2)) + "\n")
        f.write("epoch {}: ".format(epoch_idx) + "losses_ms1: " + str(np.mean(losses_ms1)) + " losses_ms2: "
                + str(np.mean(losses_ms2)) + "\n")
        start_time = time.time()

        def set_eval(model1):
            model1.detect_1.conf_training = False
            model1.detect_2.conf_training = False
            model1.detect_3.conf_training = False
            return model1

        def settrain(model1):
            model1.detect_1.conf_training = True
            model1.detect_2.conf_training = True
            model1.detect_3.conf_training = True
            return model1

        model1 = set_eval(model1)
        model2 = set_eval(model2)
        model1.set_train(False)
        model2.set_train(False)
        config.outputs_dir = os.path.join(config.log_path)
        detection_torch = DetectionEngine(config)
        detection_ms = DetectionEngine(config)
        print('Start inference....')
        for i, data in enumerate(testdata):
            image = data["image"]
            image_shape = data["image_shape"]
            image_id = data["img_id"]
            # print(type(image))
            image_ms = mindspore.Tensor(image)
            output_big_ms_1, output_me_ms_1, output_small_ms_1 = model1(image_ms)
            output_big_ms, output_me_ms, output_small_ms = model2(image_ms)
            output_big_ms_1, output_me_ms_1, output_small_ms_1 = output_big_ms_1.asnumpy(), output_me_ms_1.asnumpy(), output_small_ms_1.asnumpy()
            output_big_ms, output_me_ms, output_small_ms = output_big_ms.asnumpy(), output_me_ms.asnumpy(), output_small_ms.asnumpy()
            # image_id = image_id.numpy()
            # image_shape = image_shape.numpy()
            detection_torch.detect([output_big_ms_1, output_me_ms_1, output_small_ms_1], per_batch_size, image_shape,
                                   image_id)
            detection_ms.detect([output_big_ms, output_me_ms, output_small_ms], per_batch_size, image_shape, image_id)
            if i % 50 == 0:
                print('Processing... {:.2f}% '.format(i / testset.get_dataset_size() * 100))
                break
        print('Calculating mAP...')
        detection_torch.do_nms_for_results()
        detection_ms.do_nms_for_results()
        result_file_path_torch = detection_torch.write_result()
        result_file_path_ms = detection_ms.write_result()
        print('result file path_ms1: %s', result_file_path_torch)
        print('result file path_ms2: %s', result_file_path_ms)
        eval_result_torch = detection_torch.get_eval_result()
        eval_result_ms = detection_ms.get_eval_result()
        cost_time = time.time() - start_time
        eval_print_str_torch = '\n=============coco eval result_mindspore_old=========\n' + eval_result_torch
        eval_print_str_ms = '\n=============coco eval result_mindspore_new=========\n' + eval_result_ms
        print(eval_print_str_torch)
        f.write(eval_result_torch + "\n")
        print(eval_print_str_ms)
        f.write(eval_result_ms + "\n")
        print('testing cost time %.2f h', cost_time / 3600.)
        model1 = settrain(model1)
        model2 = settrain(model2)
        model1.set_train()
        model2.set_train()
