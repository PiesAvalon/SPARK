import os
import troubleshooter as ts
seed = 20230818
ts.widget.fix_random(seed)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from copy import deepcopy
from pprint import pprint
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from models.FastText.Fasttext import FastText
from models.FastText.fasttext_torch import FastText_torch
from models.FastText.main_Parallel import load_infer_dataset
from models.textcnn.dataset import MovieReview
from models.textcnn.textcnn import TextCNN
import platform
import cv2
import mindspore
import numpy as np
import torch
from mindspore import load_param_into_net, load_checkpoint, Model, nn, ops
from infoplus.MindSporeInfoPlus import mindsporeinfoplus
from infoplus.TorchInfoPlus import torchinfoplus
from models.UNet.main import UNetMedical
from models.UNet.main_torch import UNetMedical_torch, DiceCoeff, UnetEval_torch
from models.UNet.src.data_loader import create_dataset
from models.UNet.src.utils import UnetEval, TempLoss, dice_coeff
# from models.deeplabv3.Deeplabv3 import DeepLabV3
# from models.deeplabv3.main_Parallel import resize_long, cal_hist
from models.deeplabv3.main_torch import DeepLabV3_torch
# from models.resnet50.resnet50 import resnet50, distance
# from models.resnet50.resnet50_torch import resnet50 as resnet50_torch


def EuclideanDistance(x, y):
    # 欧式距离
    # print("x.type", type(x))
    # print("y.type", type(y))
    if isinstance(x, mindspore.Tensor):
        x = x.asnumpy()
        x = torch.tensor(x)
    if isinstance(y, mindspore.Tensor):
        y = y.asnumpy()
        y = torch.tensor(y)

    out = torch.sqrt(torch.sum(torch.square(torch.sub(x, y))))
    return out.detach().cpu().numpy()


# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        # print(name, parameter.numpy().shape)
        pt_params[name] = parameter.numpy()
    return pt_params


bn_ms2pt = {"gamma": "weight",
            "beta": "bias",
            "moving_mean": "running_mean",
            "moving_variance": "running_var",
            "embedding_table": "weight",
            }


# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        for bn_name in bn_ms2pt:
            if bn_name in name:
                name = name.replace(bn_name, bn_ms2pt[bn_name])
        value = param.data.asnumpy()
        value = torch.tensor(value, dtype=torch.float32)
        # print(name)
        ms_params[name] = value
    return ms_params


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


# def eval_yolov3(model_ms, model_torch, f: TextIO, config):
#     def set_eval(model1):
#         model1.detect_1.conf_training = False
#         model1.detect_2.conf_training = False
#         model1.detect_3.conf_training = False
#         return model1
#
#     # data_dir = "/data1/pzy/raw/coco2014"
#     data_dir = "/root/datasets/coco2014"
#     config.data_dir = data_dir
#     config.annFile = os.path.join(config.data_dir, 'annotations/instances_val2014.json')
#     start_time = time.time()
#     test_root = os.path.join(config.data_dir, 'val2014')
#     testset = create_yolo_dataset(test_root,
#                                   os.path.join(config.data_dir, 'annotations/instances_val2014.json'),
#                                   is_training=False,
#                                   batch_size=1, device_num=1,
#                                   rank=0, shuffle=False)
#     testdata = testset.create_dict_iterator(num_epochs=1, output_numpy=True)
#     model_ms = set_eval(model_ms)
#     model_ms.set_train(False)
#     model_torch = set_eval(model_torch)
#     model_torch.eval()
#     config.outputs_dir = os.path.join(config.log_path)
#     detection_ms = DetectionEngine(config)
#     detection_torch = DetectionEngine(config)
#     print('Start inference....')
#     for i, data in enumerate(testdata):
#         image = data["image"]
#         image_shape = data["image_shape"]
#         image_id = data["img_id"]
#         # print(type(image))
#         image_ms = mindspore.Tensor(image)
#         image_torch = torch.tensor(image).to(device)
#         output_big_ms, output_me_ms, output_small_ms = model_ms(image_ms)
#         output_big_torch, output_me_torch, output_small_torch = model_torch(image_torch)
#         # print("type", type(output_big_ms))
#         output_big_ms, output_me_ms, output_small_ms = output_big_ms.asnumpy(), output_me_ms.asnumpy(), output_small_ms.asnumpy()
#         output_big_torch, output_me_torch, output_small_torch = output_big_torch.detach().cpu().numpy(), output_me_torch.detach().cpu().numpy(), output_small_torch.detach().cpu().numpy()
#         # image_id = image_id.numpy()
#         # image_shape = image_shape.numpy()
#         detection_ms.detect([output_big_ms, output_me_ms, output_small_ms], 1, image_shape, image_id)
#         detection_torch.detect([output_big_torch, output_me_torch, output_small_torch], 1, image_shape, image_id)
#         if i % 50 == 0:
#             print('Processing... {:.2f}% '.format(i / testset.get_dataset_size() * 100))
#             # break
#     print('Calculating mAP...')
#     detection_ms.do_nms_for_results()
#     detection_torch.do_nms_for_results()
#
#     result_file_path_ms = detection_ms.write_result()
#     result_file_path_torch = detection_torch.write_result()
#
#     print('result file path_ms2: %s', result_file_path_ms)
#     print('result file path_torch: %s', result_file_path_torch)
#
#     eval_result_ms = detection_ms.get_eval_result()
#     eval_result_torch = detection_torch.get_eval_result()
#
#     cost_time = time.time() - start_time
#     eval_print_str_ms = '\n=============coco eval result_mindspore_old=========\n' + eval_result_ms
#     eval_print_str_torch = '\n=============coco eval result_torch=========\n' + eval_result_torch
#     print(eval_print_str_ms)
#     print(eval_print_str_torch)
#     f.write(eval_result_ms + "\n")
#     f.write(eval_result_torch + "\n")
#     print('testing cost time %.2f h', cost_time / 3600.)
#
#
# def eval_yolov3_torch(model, f: TextIO, config):
#     def set_eval(model1):
#         model1.detect_1.conf_training = False
#         model1.detect_2.conf_training = False
#         model1.detect_3.conf_training = False
#         return model1
#
#     model = set_eval(model)
#     model.train(False)
#     # data_dir = "/data1/pzy/raw/coco2014"
#     data_dir = "/root/datasets/coco2014"
#     config.data_dir = data_dir
#     config.annFile = os.path.join(config.data_dir, 'annotations/instances_val2014.json')
#     start_time = time.time()
#     test_root = os.path.join(config.data_dir, 'val2014')
#     testset = create_yolo_dataset(test_root,
#                                   os.path.join(config.data_dir, 'annotations/instances_val2014.json'),
#                                   is_training=False,
#                                   batch_size=1, device_num=1,
#                                   rank=0, shuffle=False)
#     testdata = testset.create_dict_iterator(num_epochs=1, output_numpy=True)
#     config.outputs_dir = os.path.join(config.log_path)
#
#     detection_torch = DetectionEngine(config)
#     print('Start inference....')
#     for i, data in enumerate(testdata):
#         image = data["image"]
#         image_shape = data["image_shape"]
#         image_id = data["img_id"]
#         # print(type(image))
#         image_t = torch.tensor(image).to(device)
#         output_big, output_me, output_small = model(image_t)
#         output_big = output_big.cpu().detach().numpy()
#         output_me = output_me.cpu().detach().numpy()
#         output_small = output_small.cpu().detach().numpy()
#         # image_id = image_id.numpy()
#         # image_shape = image_shape.numpy()
#         detection_torch.detect([output_small, output_me, output_big], 1, image_shape, image_id)
#
#         if i % 50 == 0:
#             print('Processing... {:.2f}% '.format(i / testset.get_dataset_size() * 100))
#             break
#     print('Calculating mAP...')
#     detection_torch.do_nms_for_results()
#     result_file_path_torch = detection_torch.write_result()
#     print('result file path_ms1: %s', result_file_path_torch)
#     eval_result_torch = detection_torch.get_eval_result()
#     cost_time = time.time() - start_time
#     eval_print_str_torch = '\n=============coco eval result_torch_old=========\n' + eval_result_torch
#     print(eval_print_str_torch)
#     f.write(eval_result_torch + "\n")
#     print('testing cost time %.2f h', cost_time / 3600.)
#
#
# def ckpt_convert_yolov3():
#     network = yolov3(is_training=True)  # yolov3 mindspore
#     ckpt_path = "/root/MR/ckpts/yolov3/yolov3darknet53shape416_ascend_v190_coco2014_official_cv_map31.8.ckpt"
#     if platform.system() == "Windows":
#         ckpt_path = r"D:\迅雷下载\yolov3darknet53shape416_ascend_v190_coco2014_official_cv_map31.8.ckpt"
#     load_param_into_net(network, load_checkpoint(ckpt_path))
#     print("=" * 20)
#     ms_param = mindspore_params(network)
#     torch.save(ms_param, 'models/yolov3/yolov3.pth')
#     input_np = np.random.randn(1, 3, 224, 224)
#     inputs = torch.from_numpy(input_np).float().to(device)
#     net = YOLOV3DarkNet53(True).to(device)  # yolov3 Pytorch
#     # for i in net.state_dict():
#     #     print(i)
#     print("device", device)
#     weights_dict = torch.load("models/yolov3/yolov3.pth", map_location="cpu")
#     # param_convert(weights_dict, net.state_dict())
#     load_weights_dict = {k: v for k, v in weights_dict.items()
#                          if k in net.state_dict()}
#     pprint(weights_dict.keys())
#     pprint(load_weights_dict.keys())
#     net.load_state_dict(load_weights_dict, strict=False)
#
#     print(net(inputs)[0][0].shape)
#     config = get_config()
#     eval_yolov3(network, net, open("eval_ms_result.txt", "w"), config)
#     # eval_yolov3_torch(net, open("eval_torch_result.txt", "w"), config)


def eval_Unet_ms(model_ms, data_dir,
                 cross_valid_ind=1):
    net = UnetEval(model_ms, eval_activate="Softmax".lower())
    _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False,
                                      do_crop=[388, 388], img_size=[572, 572])
    model = Model(net, loss_fn=TempLoss(), metrics={"dice_coeff": dice_coeff(show_eval=False, print_res=False)})

    print("============== Starting ms Evaluating ============")
    eval_score = model.eval(valid_dataset, dataset_sink_mode=False)["dice_coeff"]
    print("============== mindspore dice coeff is:", eval_score[0])
    print("============== mindspore IOU is:", eval_score[1])


def eval_Unet_torch(model_torch, data_dir, cross_valid_ind=1):
    from configs.Unetconfig import config
    config.use_deconv = True
    config.use_ds = False
    config.use_bn = False
    config.batch_size = 1
    _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False,
                                      do_crop=[388, 388], img_size=[572, 572])
    testnet = UnetEval_torch(model_torch, eval_activate="Softmax".lower())
    valid_ds = valid_dataset.create_tuple_iterator(output_numpy=True)
    metric = DiceCoeff()
    print("============== Starting torch Evaluating ============")
    metric.clear()
    for data in valid_ds:
        # inputs, labels = data[0].numpy(), data[1].numpy()
        inputs, labels = data[0], data[1]
        inputs, labels = torch.tensor(inputs).to(device), torch.tensor(labels).to(
            device)  # Send tensors to the appropriate device (CPU or GPU)
        logits = testnet(inputs)
        metric.update(logits, labels)
    dice_coeff_avg, iou_avg = metric.compute()
    print("torch Dice Coefficient:", dice_coeff_avg, "torch IOU:", iou_avg)


def eval_Unet(model_input, data_dir, cross_valid_ind=1):
    from configs.Unetconfig import config
    config.use_deconv = True
    config.use_ds = False
    config.use_bn = False
    config.batch_size = 1
    _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False,
                                      do_crop=[388, 388], img_size=[572, 572])

    valid_ds = valid_dataset.create_tuple_iterator(output_numpy=True)
    metric = dice_coeff()

    for data in valid_ds:
        metric.clear()
        # inputs, labels = data[0].numpy(), data[1].numpy()
        inputs, labels = data[0], data[1]
        inputs, labels = mindspore.Tensor(inputs, mindspore.float32), mindspore.Tensor(labels,
                                                                                       mindspore.int32)  # Send tensors to the appropriate device (CPU or GPU)
        inputs_torch, labels_torch = torch.tensor(data[0], dtype=torch.float32).to(device), \
            torch.tensor(data[1], dtype=torch.int64).to(device)
        if str(model_input.__class__) == "<class 'models.UNet.src.utils.UnetEval'>":
            logits = model_input(inputs)
        else:
            logits = model_input(inputs_torch)
            logits = mindspore.Tensor(logits.detach().cpu().numpy(), mindspore.float32)
        metric.update(logits, labels)

    # print("logits shape:", logit.shape, "labels shape:", label.shape)
    dice = metric.eval()
    print("accuracy", dice)

    return dice


def ckpt_convert_UNetMedical():
    network = UNetMedical(n_channels=1, n_classes=2)
    ckpt_path = "/root/MR/ckpts/Unet/unet2d_ascend_v190_isbichallenge_official_cv_iou90.00.ckpt"
    if platform.system() == "Windows":
        ckpt_path = r"G:\迅雷下载\unet2d_ascend_v190_isbichallenge_official_cv_iou90.00.ckpt"
    load_param_into_net(network, load_checkpoint(ckpt_path))
    inpu_np = np.ones([1, 1, 572, 572])
    network.set_train(False)
    ms_param = mindspore_params(network)
    eval_ms_net = UnetEval(network, eval_activate="Softmax".lower())
    np_data = [inpu_np for _ in range(1)]
    dtypes = [mindspore.float32]
    input_data = mindsporeinfoplus.np_2_tensor(np_data, dtypes)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=eval_ms_net,
        input_data=input_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'],
        verbose=0,
        depth=8)
    # print(res)
    # print("net_ms: ", [i.name for i in net_ms.get_parameters()])
    output_data = mindsporeinfoplus.get_output_datas(global_layer_info)
    print("=" * 20)
    torch.save(ms_param, 'models/UNet/Unet_medical.pth')
    net = UNetMedical_torch(n_channels=1, n_classes=2).to(device)
    # for i in net.state_dict():
    #     print(i)
    print("device", device)
    weights_dict = torch.load('models/UNet/Unet_medical.pth', map_location=device)
    # param_convert(weights_dict, net.state_dict())
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in net.state_dict()}
    pprint(weights_dict.keys())
    pprint(load_weights_dict.keys())
    net.load_state_dict(load_weights_dict, strict=False)
    eval_torch_net = UnetEval_torch(net, eval_activate="Softmax".lower())
    dtypes = [torch.float32]
    torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, "cpu")

    result, global_layer_info = torchinfoplus.summary(
        model=eval_torch_net,
        input_data=torch_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=0)
    # print(result)
    output_data1 = torchinfoplus.get_output_datas(global_layer_info)
    # print(input_datas)
    # pprint(input_datas)
    print("===========================================")
    # pprint(input_datas2)
    print("maximum", compare_layer(output_data, output_data1))
    # print(net(inputs).shape)
    # eval_Unet_ms2(network, data_dir="datasets/archive",
    #               cross_valid_ind=1)
    # eval_Unet_torch(net, "datasets/archive", cross_valid_ind=1)

    # print(str(eval_ms_net.__class__))
    # print(str(eval_torch_net.__class__))

    print(eval_Unet(eval_ms_net, "datasets/archive", cross_valid_ind=1))
    print(eval_Unet(eval_torch_net, "datasets/archive", cross_valid_ind=1))


class BuildEvalNetwork(nn.Cell):
    def __init__(self, network, input_format="NCHW"):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)
        self.transpose = ops.Transpose()
        self.format = input_format

    def construct(self, input_data):
        if self.format == "NHWC":
            input_data = self.transpose(input_data, (0, 3, 1, 2))
        output = self.network(input_data)
        output = self.softmax(output)
        return output


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


def eval_batch(args, eval_net, img_lst, crop_size=513, flip=True):
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
    net_out = eval_net(mindspore.Tensor(batch_img, mindspore.float32))
    net_out = net_out.asnumpy()

    if flip:
        batch_img = batch_img[:, :, :, ::-1]
        net_out_flip = eval_net(mindspore.Tensor(batch_img, mindspore.float32))
        net_out += net_out_flip.asnumpy()[:, :, :, ::-1]

    for bs in range(batch_size):
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        result_lst.append(probs_)

    return result_lst


def eval_batch_torch(args, eval_net, img_lst, crop_size=513, flip=True):
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
    net_out = eval_net(torch.tensor(batch_img, dtype=torch.float32).to(device))
    # print("netoput", type(net_out))
    net_out = net_out.detach().cpu().numpy()

    if flip:
        batch_img = batch_img[:, :, :, ::-1].copy()
        net_out_flip = eval_net(torch.tensor(batch_img, dtype=torch.float32).to(device))
        net_out += net_out_flip.numpy()[:, :, :, ::-1]

    for bs in range(batch_size):
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        result_lst.append(probs_)
    return result_lst


def eval_batch_scales(args, eval_net, img_lst, scales,
                      base_crop_size=513, flip=True):
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = eval_batch(args, eval_net, img_lst, crop_size=sizes_[0], flip=flip)
    # print(sizes_)
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_batch(args, eval_net, img_lst, crop_size=crop_size_, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))
    return result_msk


def eval_batch_scales_torch(args, eval_net, img_lst, scales,
                            base_crop_size=513, flip=True):
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = eval_batch_torch(args, eval_net, img_lst, crop_size=sizes_[0], flip=flip)
    # print(sizes_)
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_batch_torch(args, eval_net, img_lst, crop_size=crop_size_, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))
    return result_msk


def eval_DeeplabV3_ms(model_ms):
    from configs.DeeplabConfig import config as args
    eval_net = BuildEvalNetwork(model_ms, args.input_format)
    args.data_root = "/root/MR/datasets/VOC2012/VOC2012"
    args.data_lst = "/root/MR/datasets/VOC2012/voc_val_lst.txt"
    # load model
    eval_net.set_train(False)
    with open(args.data_lst) as f:
        img_lst = f.readlines()
    # evaluate
    hist = np.zeros((args.num_classes, args.num_classes))
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
            batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                          base_crop_size=args.crop_size, flip=args.flip)
            for mi in range(args.batch_size):
                hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)

            bi = 0
            batch_img_lst = []
            batch_msk_lst = []
            # print('processed {} images'.format(i + 1))
        image_num = i

    if bi > 0:
        batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                      base_crop_size=args.crop_size, flip=args.flip)
        for mi in range(bi):
            hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)
        # print('processed {} images'.format(image_num + 1))

    # print(hist)
    # iu_torch = np.diag(hist_torch) / (hist_torch.sum(1) + hist_torch.sum(0) - np.diag(hist_torch))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('per-class IoU', iu)
    print('mean IoU', np.nanmean(iu))


def eval_DeeplabV3_torch(model_torch):
    model_torch.train(False)
    from configs.DeeplabConfig import config as args
    args.batch_size = 1
    args.data_root = "/root/MR/datasets/VOC2012/VOC2012"
    args.data_lst = "/root/MR/datasets/VOC2012/voc_val_lst.txt"
    hist_torch = np.zeros((args.num_classes, args.num_classes))
    batch_img_lst = []
    batch_msk_lst = []
    bi = 0
    with open(args.data_lst) as f:
        img_lst = f.readlines()
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
            batch_res_torch = eval_batch_scales_torch(args, model_torch, batch_img_lst, scales=args.scales,
                                                      base_crop_size=args.crop_size, flip=args.flip)
            for mi in range(args.batch_size):
                hist_torch += cal_hist(batch_msk_lst[mi].flatten(), batch_res_torch[mi].flatten(), args.num_classes)
            bi = 0
            batch_img_lst = []
            batch_msk_lst = []
            # print('processed {} images'.format(i + 1))

    if bi > 0:
        batch_res_torch = eval_batch_scales_torch(args, model_torch, batch_img_lst, scales=args.scales,
                                                  base_crop_size=args.crop_size, flip=args.flip)
        for mi in range(bi):
            hist_torch += cal_hist(batch_msk_lst[mi].flatten(), batch_res_torch[mi].flatten(), args.num_classes)

        # print('processed {} images'.format(image_num + 1))

    # print(hist)
    iu_torch = np.diag(hist_torch) / (hist_torch.sum(1) + hist_torch.sum(0) - np.diag(hist_torch))
    print('torch_per-class IoU', iu_torch)


def ckpt_convert_DeeplabV3():
    network = DeepLabV3('eval', 21, 8, False)  # yolov3 mindspore
    # ckpt_path = "/data1/CKPTS/Deeplabv3/" \
    #             "deeplabv3s8r2_ascend_v190_voc2012_official_cv_s8acc78.51_ns8mul79.45_s8mulflip79.77.ckpt"
    network.set_train(False)
    ckpt_path = "/root/MR/ckpts/Deeplabv3/deeplabv3s8r2_ascend_v190_voc2012_official_cv_s8acc78.51_ns8mul79.45_s8mulflip79.77.ckpt"
    if platform.system() == "Windows":
        ckpt_path = r"G:\迅雷下载\deeplabv3s8r2_ascend_v190_voc2012_official_cv_s8acc78.51_ns8mul79.45_s8mulflip79.77.ckpt"
    load_param_into_net(network, load_checkpoint(ckpt_path))
    print("=" * 20)
    # ms_param = mindspore_params(network)
    # torch.save(ms_param, 'models/deeplabv3/deeplabv3.pth')
    # input_np = np.ones([2, 3, 513, 513])
    # np_data = [input_np for _ in range(1)]
    # dtypes = [mindspore.float32]
    #
    # input_data = mindsporeinfoplus.np_2_tensor(np_data, dtypes)
    #
    # res, global_layer_info = mindsporeinfoplus.summary_plus(
    #     model=network,
    #     input_data=input_data,
    #     #     =[(96, 16), (96, 16), (96, 16), (96, 16)],
    #     dtypes=dtypes,
    #     col_names=['input_size', 'output_size', 'name'],
    #     verbose=0,
    #     depth=8)
    # # print("res: ", res)
    # # print("net_ms: ", [i.name for i in net_ms.get_parameters()])
    # output_data = mindsporeinfoplus.get_output_datas(global_layer_info)
    net = DeepLabV3_torch().to(device)  # yolov3 Pytorch
    net.eval()
    # for i in net.state_dict():
    #     print(i)
    print("device", device)
    weights_dict = torch.load('models/deeplabv3/deeplabv3.pth', map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in net.state_dict()}
    pprint(weights_dict.keys())
    pprint(load_weights_dict.keys())
    net.load_state_dict(load_weights_dict, strict=False)
    # dtypes = [torch.float32]
    #
    # torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, device)
    #
    # result, global_layer_info = torchinfoplus.summary(
    #     model=net,
    #     input_data=torch_data,
    #     # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
    #     dtypes=dtypes,
    #     col_names=['input_size', 'output_size', 'name'], depth=8,
    #     verbose=0)
    # # print("result: ", result)
    # output_data1 = torchinfoplus.get_output_datas(global_layer_info)
    # # print(input_datas)
    # # pprint(input_datas)
    # print("===========================================")
    # # pprint(input_datas2)
    # print("maximum", compare_layer(output_data, output_data1))
    # print(net(inputs).shape)
    eval_DeeplabV3_ms(network)
    eval_DeeplabV3_torch(net)


def ckpt_convert_resnet():
    network = resnet50()  # yolov3 mindspore
    # ckpt_path = "/data1/CKPTS/resnet50/resnet50_ascend_v190_cifar10_official_cv_top1acc91.00.ckpt"
    ckpt_path = "/data/CKPTS/resnet50/resnet50_ascend_v190_cifar10_official_cv_top1acc91.00.ckpt"
    load_param_into_net(network, load_checkpoint(ckpt_path))
    print("=" * 20)
    inpu_np1 = np.ones([1, 3, 224, 224])
    network.set_train(False)
    np_data = [inpu_np1]
    dtypes = [mindspore.float32]
    input_data = mindsporeinfoplus.np_2_tensor(np_data, dtypes)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=network,
        input_data=input_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'],
        verbose=0,
        depth=8)
    # print("net_ms: ", [i.name for i in net_ms.get_parameters()])
    input_data = mindsporeinfoplus.get_input_datas(global_layer_info)
    output_data = mindsporeinfoplus.get_output_datas(global_layer_info)
    ms_param = mindspore_params(network)
    torch.save(ms_param, 'models/resnet50/resnet50.pth')
    net_torch = resnet50_torch()
    net_torch.eval()
    weights_dict = torch.load('models/resnet50/resnet50.pth', map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in net_torch.state_dict()}
    # print("weights_dict.keys()")
    # pprint(weights_dict.keys())
    # print("===========================================")
    # print("load_weights_dict.keys()")
    # pprint(load_weights_dict.keys())
    net_torch.load_state_dict(load_weights_dict, strict=False)
    dtypes = [torch.float32]
    torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, device)

    result, global_layer_info = torchinfoplus.summary(
        model=net_torch,
        input_data=torch_data,
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=0)
    input_data1 = torchinfoplus.get_input_datas(global_layer_info)
    output_data1 = torchinfoplus.get_output_datas(global_layer_info)
    print("===========================================")
    # pprint(input_datas2)
    print("maximum", compare_layer(output_data, output_data1))
    # print("maximum", compare_layer(input_data, input_data1))


def tmp():
    network = resnet50()  # yolov3 mindspore
    ckpt_path = "/data1/CKPTS/resnet50/resnet50_ascend_v190_cifar10_official_cv_top1acc91.00.ckpt"
    if platform.system() == "Windows":
        ckpt_path = r"G:\迅雷下载\resnet50_ascend_v190_cifar10_official_cv_top1acc91.00.ckpt"
    load_param_into_net(network, load_checkpoint(ckpt_path))
    ms_param = mindspore_params(network)
    torch.save(ms_param, 'models/resnet50/resnet50.pth')
    net_torch = resnet50_torch()
    weights_dict = torch.load('models/resnet50/resnet50.pth', map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in net_torch.state_dict()}

    net_torch.load_state_dict(load_weights_dict, strict=False)


def compare_layer(input_data_dict_new, input_data_dict_old):
    # pprint(input_data_dict_new)
    maximum = 0
    for layer in input_data_dict_new.keys():
        if input_data_dict_new[layer] is not None and input_data_dict_old[layer] is not None:
            layer_np_new = input_data_dict_new[layer][0]
            layer_up_old = input_data_dict_old[layer][0]
            print("layer: ", layer, "distance chess: ", distance(layer_np_new, layer_up_old)
                  # , "distance_euclidean: ",
                  # EuclideanDistance(layer_np_new, layer_up_old)
                  )
            # try:
            maximum = max(maximum, distance(layer_np_new, layer_up_old))
            # except TypeError as e:
            #     print(e)
            #     return 0
    return maximum


def ckpt_convert_FastText():
    network = FastText()  # Fasttext mindspore
    ckpt_path = "/root/MR/ckpts/FastText/fasttext_ascend_v190_dbpedia_official_nlp_acc98.6.ckpt"
    if platform.system() == "Windows":
        ckpt_path = r"D:\迅雷下载\yolov3darknet53shape416_ascend_v190_coco2014_official_cv_map31.8.ckpt"
    load_param_into_net(network, load_checkpoint(ckpt_path))
    print("=" * 20)
    ms_param = mindspore_params(network)
    torch.save(ms_param, 'models/FastText/FastText.pth')
    inpu_np1 = np.ones([4096, 64])
    inpu_np2 = np.ones([4096, 1])
    network.set_train(False)
    np_data = [inpu_np1, inpu_np2]
    dtypes = [mindspore.int32, mindspore.int32]
    input_data = mindsporeinfoplus.np_2_tensor(np_data, dtypes)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=network,
        input_data=input_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'],
        verbose=0,
        depth=8)
    # print("net_ms: ", [i.name for i in net_ms.get_parameters()])
    input_data = mindsporeinfoplus.get_input_datas(global_layer_info)
    output_data = mindsporeinfoplus.get_output_datas(global_layer_info)
    net = FastText_torch().to(device)  # yolov3 Pytorch
    # for i in net.state_dict():
    #     print(i)
    print("device", device)
    weights_dict = torch.load("models/FastText/FastText.pth", map_location="cpu")
    # param_convert(weights_dict, net.state_dict())
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in net.state_dict()}
    pprint(weights_dict.keys())
    pprint(net.state_dict().keys())
    pprint(load_weights_dict.keys())
    net.load_state_dict(load_weights_dict, strict=False)
    dtypes = [torch.int64, torch.int64]
    torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, "cpu")

    result, global_layer_info = torchinfoplus.summary(
        model=net,
        input_data=torch_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=0)
    input_data1 = torchinfoplus.get_input_datas(global_layer_info)
    output_data1 = torchinfoplus.get_output_datas(global_layer_info)
    # print(input_datas)
    # pprint(input_datas)
    print("===========================================")
    # pprint(input_datas2)
    print("maximum", compare_layer(input_data, input_data1))
    print("maximum", compare_layer(output_data, output_data1))
    eval_FastText(net, network)


def infer_ms(prediction):
    from mindspore.ops import operations as P
    argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
    log_softmax = mindspore.nn.LogSoftmax(axis=1)
    predicted_idx = log_softmax(prediction)
    predicted_idx, _ = argmax(predicted_idx)

    return predicted_idx


# def infer_torch(prediction):
#     predicted_idx = torch.nn.functional.log_softmax(input=prediction, dim=1)
#     predicteds, _ = torch.max(input=predicted_idx, dim=1, keepdim=True)
#     return predicteds

def infer_torch(prediction):
    log_softmax = torch.nn.LogSoftmax(dim=1)
    predicted_log_probabilities = log_softmax(prediction)
    predicted_idx = torch.argmax(predicted_log_probabilities, dim=1, keepdim=True)

    return predicted_idx


def eval_FastText(model_torch, model_ms):
    print("++++++++开始评估++++++++")
    target_sens1, target_sens2, predictions1, predictions2 = [], [], [], []
    test_data = load_infer_dataset(datafile="/root/MR/datasets/dbpedia", bucket=[64], batch_size=96)
    test_iter = test_data.create_dict_iterator(output_numpy=True, num_epochs=1)
    for batch_test in test_iter:
        src_tokens1 = torch.LongTensor(batch_test['src_tokens']).to(device)
        src_tokens_length1 = torch.LongTensor(batch_test['src_tokens_length']).to(device)
        outputs_torch = model_torch(src_tokens1, src_tokens_length1, )
        predicted_idx1 = infer_torch(outputs_torch)

        src_tokens2 = mindspore.Tensor(batch_test['src_tokens'], mindspore.int32)
        src_tokens_length2 = mindspore.Tensor(batch_test['src_tokens_length'], mindspore.int32)
        outputs_ms = model_ms(src_tokens2, src_tokens_length2)
        predicted_idx2 = infer_ms(outputs_ms)
        print("outputs", distance(outputs_torch.detach().cpu().numpy(), outputs_ms.asnumpy()))
        print("predict", distance(predicted_idx1.detach().cpu().numpy(), predicted_idx2.asnumpy()))
        target_sens1.append(deepcopy(batch_test['label_idx']))
        target_sens2.append(deepcopy(batch_test['label_idx']))
        predictions1.append(predicted_idx1.type(torch.int64).cpu().numpy())
        predictions2.append(predicted_idx2.asnumpy())

    # target_sens1 = np.array(target_sens1).flatten()
    merge_target_sens1, merge_predictions1, merge_target_sens2, merge_predictions2 = [], [], [], []

    for i in range(len(target_sens1)):
        merge_target_sens1.extend(target_sens1[i].flatten())
        merge_predictions1.extend(predictions1[i].flatten())

        merge_target_sens2.extend(target_sens2[i].flatten())
        merge_predictions2.extend(predictions2[i].flatten())

    # print("merge_target_sens1: ", len(merge_target_sens1))
    # print("merge_predictions1: ", len(merge_predictions1))
    # print("merge_target_sens2: ", len(merge_target_sens2))
    # print("merge_predictions2: ", len(merge_predictions2))

    acc1 = accuracy_score(merge_target_sens1, merge_predictions1)
    acc2 = accuracy_score(merge_target_sens2, merge_predictions2)

    print(f"Accuracy(torch): {acc1 * 100}")
    print(f"Accuracy(mindspore): {acc2 * 100}")


def ckpt_convert_TextCNN():
    network = TextCNN(vocab_len=20288, word_len=51, num_classes=2, vec_length=40)  # TextCNN mindspore
    ckpt_path = "/data/CKPTS/textcnn/textcnn_ascend_v190_moviereview_official_nlp_acc77.44.ckpt"
    load_param_into_net(network, load_checkpoint(ckpt_path))
    print("=" * 20)
    ms_param = mindspore_params(network)
    torch.save(ms_param, 'models/textcnn/textcnn.pth')
    inpu_np1 = np.ones([1, 51])
    network.set_train(False)
    np_data = [inpu_np1]
    dtypes = [mindspore.int32]
    input_data = mindsporeinfoplus.np_2_tensor(np_data, dtypes)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=network,
        input_data=input_data,
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'],
        verbose=0,
        depth=8)
    # print("net_ms: ", [i.name for i in net_ms.get_parameters()])
    input_data = mindsporeinfoplus.get_input_datas(global_layer_info)
    output_data = mindsporeinfoplus.get_output_datas(global_layer_info)
    from models.textcnn.textcnn_torch import TextCNN as TextCNN_torch
    net = TextCNN_torch(vocab_len=20288, word_len=51, num_classes=2, vec_length=40).to(device)  # TextCNN Pytorch
    # for i in net.state_dict():
    #     print(i)
    print("device", device)
    weights_dict = torch.load("models/textcnn/textcnn.pth", map_location="cpu")
    # param_convert(weights_dict, net.state_dict())
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in net.state_dict()}
    pprint(weights_dict.keys())
    pprint(net.state_dict().keys())
    pprint(load_weights_dict.keys())
    net.load_state_dict(load_weights_dict, strict=False)
    dtypes = [torch.int64, torch.int64]
    torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, "cpu")
    result, global_layer_info = torchinfoplus.summary(
        model=net,
        input_data=torch_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=0)
    input_data1 = torchinfoplus.get_input_datas(global_layer_info)
    output_data1 = torchinfoplus.get_output_datas(global_layer_info)
    # print(input_datas)
    # pprint(input_datas)
    print("===========================================")
    # pprint(input_datas2)
    print("maximum", compare_layer(input_data, input_data1))
    print("maximum", compare_layer(output_data, output_data1))
    input_size = (1, 51)
    diff_finder = ts.migrator.NetDifferenceFinder(pt_net=net, ms_net=network, fix_seed=seed, auto_conv_ckpt=0)  #
    diff_finder.compare(auto_inputs=((input_size, np.int32),))
    eval_TextCNN(net, network)


def eval_TextCNN(model_torch: torch.nn.Module, model_ms: nn.Cell):
    model_torch.eval()
    model_ms.set_train(False)

    test_data_size = 0
    correct_torch = 0
    correct_ms = 0
    instance = MovieReview(root_dir="datasets/rt-polaritydata", maxlen=51, split=0.9)
    test_dataset = instance.create_train_dataset(batch_size=1)
    dataset_size = test_dataset.get_dataset_size()
    test_iter = test_dataset.create_dict_iterator(output_numpy=False)

    for item in tqdm(test_iter, total=dataset_size):
        text, targets = item['data'], item['label']
        test_data_size += text.shape[0]
        text_array, targets_array = text.asnumpy(), targets.asnumpy()
        with torch.no_grad():
            text_tensor, targets_tensor = torch.LongTensor(text_array).to(device), torch.LongTensor(
                targets_array).to(device)

            output_torch = model_torch(text_tensor)
        output_ms = model_ms(text)
        indices_ms = np.argmax(output_ms.asnumpy(), axis=1)
        result_ms = (np.equal(indices_ms, targets.asnumpy()) * 1).reshape(-1)
        accuracy_ms = result_ms.sum()
        correct_ms = correct_ms + accuracy_ms
        with torch.no_grad():
            indices = torch.argmax(output_torch.to(device), dim=1)
            result = (np.equal(indices.detach().cpu().numpy(), targets_tensor.detach().cpu().numpy()) * 1).reshape(-1)
            accuracy = result.sum()
            correct_torch = correct_torch + accuracy

    print("Pytorch Test Accuracy: {}%".format(
        100 * correct_torch / test_data_size) + " " + "Mindpsore Test Accuacy: {}%".format(
        100 * correct_ms / test_data_size))


if __name__ == '__main__':
    device = "cpu"
    mindspore.set_context(pynative_synchronize=True)
    mindspore.set_context(device_target="GPU", device_id=0)
    # ckpt_convert_FastText()
    # ckpt_convert_UNetMedical()
    ckpt_convert_resnet()
    # ckpt_convert_DeeplabV3()
    # ckpt_convert_TextCNN()

    # network = resnet50()
    # ckpt_path = "/data/CKPTS/resnet50/resnet50_ascend_v190_cifar10_official_cv_top1acc91.00.ckpt"
    # #
    # load_param_into_net(network, load_checkpoint(ckpt_path))
    # print("=" * 20)
    # inpu_np1 = np.ones([1, 3, 224, 224])
    # network.set_train(False)
    #
    # ms_param = mindspore_params(network)
    # torch.save(ms_param, 'models/resnet50/resnet50.pth')
    # net_torch = resnet50_torch().to(device)
    # net_torch.eval()
    # weights_dict = torch.load('models/resnet50/resnet50.pth', map_location=device)
    # load_weights_dict = {k: v for k, v in weights_dict.items()
    #                      if k in net_torch.state_dict()}
    # net_torch.load_state_dict(load_weights_dict, strict=False)
    # input_size = (1, 3, 224, 224)
    # diff_finder = ts.migrator.NetDifferenceFinder(pt_net=net_torch, ms_net=network, fix_seed=seed, auto_conv_ckpt=0)  #
    # diff_finder.compare(auto_inputs=((input_size, np.float32),))

    # bn_ms = network.layer1[0].bn3
    # bn_torch = net_torch.layer1[0].bn3
    #
    # weights = []
    # params_generator = bn_ms.get_parameters()
    # params_dict_keys = []
    # params_dict = {}
    # for param in params_generator:
    #     params_dict_keys.append(param.name)
    #     weights.append(param.init_data().asnumpy())
    #     params_dict[param.name] = param.init_data().asnumpy()
    # params_torch = bn_torch.state_dict()
    #
    # print(bn_ms)
    # print(bn_torch)
    # print(params_dict_keys)
    # print(params_torch.keys())
    #
    # print(np.mean(np.abs(params_torch['weight'].detach().cpu().numpy() - params_dict['layer1.0.bn3.gamma'])))
    # print(np.mean(np.abs(params_torch['bias'].detach().cpu().numpy() - params_dict['layer1.0.bn3.beta'])))
    # print(
    #     np.mean(np.abs(params_torch['running_mean'].detach().cpu().numpy() - params_dict['layer1.0.bn3.moving_mean'])))
    # print(np.mean(
    #     np.abs(params_torch['running_var'].detach().cpu().numpy() - params_dict['layer1.0.bn3.moving_variance'])))
    #
    # x_ = np.random.randn(1, 3, 224, 224)
    # x = torch.tensor(x_, dtype=torch.float32).to(device)
    # result1 = net_torch(x)
    #
    # x = mindspore.Tensor(x_, mindspore.float32)
    # result2 = network(x)
    #
    # print(np.mean(np.abs(result1.detach().cpu().numpy() - result2.asnumpy())))
    #
    # # mindspore
    # x = result2
    # identity = result2
    # out = network.layer1[0].conv1(x)
    # out = network.layer1[0].bn1(out)
    # out = network.layer1[0].relu(out)
    # out = network.layer1[0].conv2(out)
    # out = network.layer1[0].bn2(out)
    # out = network.layer1[0].relu(out)
    # out1 = network.layer1[0].conv3(out)
    # out_ms = network.layer1[0].bn3(out1)
    #
    # # torch
    # x = result1
    # identity = result1
    # out = net_torch.layer1[0].conv1(x)
    # out = net_torch.layer1[0].bn1(out)
    # out = net_torch.layer1[0].relu(out)
    # out = net_torch.layer1[0].conv2(out)
    # out = net_torch.layer1[0].bn2(out)
    # out = net_torch.layer1[0].relu(out)
    # out2 = net_torch.layer1[0].conv3(out)
    # out_torch = net_torch.layer1[0].bn3(out2)
    #
    # print(np.max(np.abs(out_torch.detach().cpu().numpy() - out_ms.asnumpy())))
    # print(np.max(np.abs(out2.detach().cpu().numpy() - out1.asnumpy())))
