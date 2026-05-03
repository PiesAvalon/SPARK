
import mindspore
import mindspore.nn as nn_ms
import numpy as np
import torch
import torch.nn as nn_torch
from common.dataset_utils import get_dataset
from common.model_utils import get_model
#from common.help_utils import check_gpu_mem_usedRate
from copy import deepcopy
import psutil
import pynvml
from mindspore import load_param_into_net, load_checkpoint
from common.loss_utils import get_loss
from common.opt_utils import get_optimizer
from mindspore.common import dtype as mstype
from scripts.run_train_imageclassification import start_imageclassification_train
import numpy as np

def test_fasterrcnn():
    model_name = "fasterrcnn"
    device = "cuda:0"
    model1, model2 = get_model(model_name, "GPU", device_id=0, input_size=(1, 3, 768, 1280))

    data1_torch = torch.tensor(np.random.randn(1, 3, 768, 1280), dtype=torch.float32).to(device)
    data2_torch = torch.tensor(np.random.randn(1, 3), dtype=torch.float32).to(device)
    data3_torch = torch.tensor(np.random.randn(1, 128, 4), dtype=torch.float32).to(device)
    data4_torch = torch.tensor(np.random.randn(1, 128), dtype=torch.float32).to(device)
    data5_torch = torch.tensor(np.random.randn(1, 128), dtype=torch.float32).to(device)

    data1_ms = mindspore.Tensor(np.random.randn(1, 3, 768, 1280), mindspore.float32)
    data2_ms = mindspore.Tensor(np.random.randn(1, 3), mindspore.float32)
    data3_ms = mindspore.Tensor(np.random.randn(1, 128, 4), mindspore.float32)
    data4_ms = mindspore.Tensor(np.random.randn(1, 128), mindspore.float32)
    data5_ms = mindspore.Tensor(np.random.randn(1, 128), mindspore.float32)

    result1 = model1(data1_ms, data2_ms, data3_ms, data4_ms, data5_ms)
    result2 = model2(data1_torch, data2_torch, data3_torch, data4_torch, data5_torch)
    return result1, result2

def test_maskrcnn():
    # maskrcnn test case

    model_name = "maskrcnn"
    device = "cuda:0"
    model1, model2 = get_model(model_name, "GPU", device_id=0, input_size=(1, 3, 768, 1280))

    model_input_size = [(1, 3, 768, 1280), (1, 3), (1, 128, 4), (1, 128), (1, 128), (1, 128, 768, 1280)]
    model_dtypes_ms = [mindspore.float32, mindspore.float32, mindspore.float32, mindspore.int32, mindspore.bool_,
                       mindspore.bool_]
    model_dtypes_t = [torch.float32, torch.float32, torch.float32, torch.int32, torch.bool, torch.bool]

    data1_torch = torch.tensor(np.random.randn(1, 3, 768, 1280), dtype=torch.float32).to(device)
    data2_torch = torch.tensor(np.random.randn(1, 3), dtype=torch.float32).to(device)
    data3_torch = torch.tensor(np.random.randn(1, 128, 4), dtype=torch.float32).to(device)
    data4_torch = torch.tensor(np.random.randn(1, 128), dtype=torch.int32).to(device)
    data5_torch = torch.tensor(np.random.randn(1, 128), dtype=torch.bool).to(device)
    data6_torch = torch.tensor(np.random.randn(1, 128, 768, 1280), dtype=torch.bool).to(device)

    data1_ms = mindspore.Tensor(np.random.randn(1, 3, 768, 1280), mindspore.float32)
    data2_ms = mindspore.Tensor(np.random.randn(1, 3), mindspore.float32)
    data3_ms = mindspore.Tensor(np.random.randn(1, 128, 4), mindspore.float32)
    data4_ms = mindspore.Tensor(np.random.randn(1, 128), mindspore.int32)
    data5_ms = mindspore.Tensor(np.random.randn(1, 128), mindspore.bool_)
    data6_ms = mindspore.Tensor(np.random.randn(1, 128, 768, 1280), mindspore.bool_)

    result1 = model1(data1_ms, data2_ms, data3_ms, data4_ms, data5_ms, data6_ms)
    result2 = model2(data1_torch, data2_torch, data3_torch, data4_torch, data5_torch, data6_torch)

    return result1,result2


def run_fasttext(model_ms,model_torch,train_configs):
    from sklearn.metrics import accuracy_score, classification_report

    

    loss_name = train_configs['loss_name']
    device = train_configs['device']
    device_id = train_configs['device_id']
    learning_rate = train_configs['learning_rate']
    batch_size = train_configs['batch_size']
    dataset_name = train_configs['dataset_name']
    optimizer = train_configs['optimizer']
    epoch_num = train_configs['epochs']
    model_name = train_configs['model_name']
    ds_path = r"/data1/pzy/mindb/dbpedia"

    print("Enter Train Stage")
    loss_ms, loss_t = get_loss(loss_name)

    if device == "GPU":
        loss_t = loss_t.to("cuda:" + str(device_id))

    dataset = get_dataset(dataset_name)

    train_data = dataset(ds_path,batch_size=batch_size, is_train=True)
    train_iter = train_data.create_dict_iterator(output_numpy=True, num_epochs=epoch_num)
    test_data = dataset(ds_path,batch_size=batch_size, is_train=False)
    test_iter = test_data.create_dict_iterator(output_numpy=True, num_epochs=epoch_num)

    optimizer_ms,optimizer_torch=get_optimizer(optimizer)
    optimizer_torch = optimizer_torch(model_torch.parameters(), lr=learning_rate)
    optimizer_ms = optimizer_ms(model_ms.trainable_params(), learning_rate, beta1=0.9, beta2=0.999)

    def infer_ms(prediction):
        from mindspore.ops import operations as P
        argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
        log_softmax = mindspore.nn.LogSoftmax(axis=1)
        predicted_idx = log_softmax(prediction)
        predicted_idx, _ = argmax(predicted_idx)

        return predicted_idx

    def infer_torch(prediction):
        predicted_idx = torch.nn.functional.log_softmax(input=prediction, dim=1)
        predicteds, _ = torch.max(input=predicted_idx, dim=1, keepdim=True)
        return predicteds

    def forward_fn(data, length, label):
        predict_score = model_ms(data, length)
        loss = loss_ms(predict_score, label)
        return loss, predict_score

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer_ms.parameters, has_aux=True)

    def train_step(data, length, label):
        (loss, _), grads = grad_fn(data, length, label)
        loss = mindspore.ops.depend(loss, optimizer_ms(grads))
        return loss

    for epoch in range(epoch_num):
        print("epochs: " + str(epoch))

        for batch_train in train_iter:
            pro_token_text = batch_train['src_token_text']
            pro_tokens_text_length = batch_train['src_tokens_text_length']

            pro_token_text_tensor = torch.Tensor(pro_token_text).to("cuda:" + str(device_id))
            pro_tokens_text_length_tensor = torch.Tensor(pro_tokens_text_length).to("cuda:" + str(device_id))
            pro_label_idx_longtensor = torch.LongTensor(batch_train['label_idx_tag']).to("cuda:" + str(device_id))

            model1out = model_torch(pro_token_text_tensor.to("cuda:" + str(device_id)),
                                pro_tokens_text_length_tensor.to("cuda:" + str(device_id)))
            loss_torch = loss_t(model1out, pro_label_idx_longtensor)
            optimizer_torch.zero_grad()
            print("loss_torch:", loss_torch.item())
            loss_torch.backward()
            optimizer_torch.step()

            pro_token_text_mstensor = mindspore.Tensor(pro_token_text)
            pro_tokens_text_length_mstensor = mindspore.Tensor(pro_tokens_text_length)
            pro_label_idx_mstensor = mindspore.Tensor(batch_train['label_idx_tag'], mindspore.int32)
            model2out = model_ms(pro_token_text_mstensor, pro_tokens_text_length_mstensor)

            loss_m = train_step(mindspore.Tensor(pro_token_text), mindspore.Tensor(pro_tokens_text_length),
                                mindspore.Tensor(batch_train['label_idx_tag']))
            print("loss_ms:", loss_m)

            break

        print("++++++++开始评估mindspore++++++++")
        target_sens1, target_sens2, predictions1, predictions2 = [], [], [], []
        for batch_test in test_iter:
            src_tokens1 = torch.LongTensor(batch_test['src_tokens']).to("cuda:" + str(device_id))
            src_tokens_length1 = torch.LongTensor(batch_test['src_tokens_length']).to("cuda:" + str(device_id))
            outputs_torch = model_torch(src_tokens1, src_tokens_length1, )
            predicted_idx1 = infer_torch(outputs_torch)

            src_tokens2 = mindspore.Tensor(batch_test['src_tokens'], mstype.int32)
            src_tokens_length2 = mindspore.Tensor(batch_test['src_tokens_length'], mstype.int32)
            outputs_ms = model_ms(src_tokens2, src_tokens_length2)
            predicted_idx2 = infer_ms(outputs_ms)

            target_sens1.append(deepcopy(batch_test['label_idx']))
            target_sens2.append(deepcopy(batch_test['label_idx']))
            predictions1.append(predicted_idx1.type(torch.int32).cpu().numpy())
            predictions2.append(predicted_idx2.asnumpy())

        # target_sens1 = np.array(target_sens1).flatten()
        merge_target_sens1, merge_predictions1, merge_target_sens2, merge_predictions2 = [], [], [], []

        for i in range(len(target_sens1)):
            merge_target_sens1.extend(target_sens1[i].flatten())
            merge_predictions1.extend(predictions1[i].flatten())

            merge_target_sens2.extend(target_sens2[i].flatten())
            merge_predictions2.extend(predictions2[i].flatten())

        acc1 = accuracy_score(merge_target_sens1, merge_target_sens1)
        acc2 = accuracy_score(merge_target_sens2, merge_predictions2)

        print("Accuracy(torch): ", acc1)
        print("Accuracy(mindspore): ", acc2)

def run_textcnn():
    device = "GPU"
    device_id = 0
    final_device = "cuda:0"
    batch_size = 10
    model_name = "textcnn"

    model_ms, model_torch = get_model(model_name, device=device, device_id=device_id,input_size=(32,51))
    loss_ms, loss_t = get_loss("textcnnloss")

    epoch_num, batch_size = 50, 32
    num_classes = 2
    dataset = get_dataset("rtpolarity")

    train_dataset = dataset(data_dir="/data1/pzy/mindb/rt-polarity/rt-polaritydata", batch_size=batch_size,
                            is_train=True)
    test_dataset = dataset(data_dir="/data1/pzy/mindb/rt-polarity/rt-polaritydata", batch_size=batch_size,
                           is_train=False)

    train_iter = train_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)
    test_iter = test_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)

    if device == "GPU":
        loss_t = loss_t.to(final_device)

    learning_rate = float(1e-5)

    opt_t = torch.optim.Adam(filter(lambda x: x.requires_grad, model_torch.parameters()), lr=learning_rate,
                             weight_decay=float(3e-5))
    opt_ms = nn_ms.Adam(filter(lambda x: x.requires_grad, model_ms.get_parameters()), learning_rate=learning_rate,
                        weight_decay=float(3e-5))

    def forward_fn(data, label, num_classes):
        outputs = model_ms(data)
        loss = loss_ms(outputs, label, num_classes)
        return loss, outputs

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt_ms.parameters, has_aux=True)

    def train_step(data, label, num_classes):
        (loss, _), grads = grad_fn(data, label, num_classes)
        loss = mindspore.ops.depend(loss, opt_ms(grads))
        return loss

    for epoch in range(epoch_num):
        model_torch.train()
        model_ms.set_train(True)
        print("epoch", epoch, "/", epoch_num)
        batch = 0

        for item in train_iter:
            text_array, targets_array = item['data'].asnumpy(), item['label'].asnumpy()

            text_tensor, targets_tensor = mindspore.Tensor(text_array, dtype=mstype.int32), mindspore.Tensor(targets_array, dtype=mstype.int32)
            loss_ms = train_step(text_tensor, targets_tensor, num_classes)
            
            opt_t.zero_grad()
            output_torch = model_torch(torch.LongTensor(text_array).to(final_device))
            loss_t_result = loss_t(output_torch, torch.LongTensor(targets_array).to(final_device), num_classes)
            loss_t_result.backward()
            opt_t.step()




            if batch % 100 == 0:
                print("batch:" + str(batch) + " torch_loss:" + str(loss_t_result.item()) + " mindspre_loss:" + str(
                    loss_ms))
            batch += batch_size

        # 测试步骤开始
        model_torch.eval()
        model_ms.set_train(False)

        test_data_size = 0
        correct_torch = 0
        correct_ms = 0

        with torch.no_grad():
            for item in test_iter:
                text, targets = item['data'], item['label']
                test_data_size += text.shape[0]
                text_array, targets_array = text.asnumpy(), targets.asnumpy()

                text_tensor, targets_tensor = torch.LongTensor(text_array).to(final_device), torch.LongTensor(
                    targets_array).to(final_device)

                output_torch = model_torch(text_tensor)
                output_ms = model_ms(text)
                indices_ms = np.argmax(output_ms.asnumpy(), axis=1)
                result_ms = (np.equal(indices_ms, targets.asnumpy()) * 1).reshape(-1)
                accuracy_ms = result_ms.sum()
                correct_ms = correct_ms + accuracy_ms

                indices = torch.argmax(output_torch.to(final_device), dim=1)
                result = (np.equal(indices.detach().cpu().numpy(), targets_tensor.detach().cpu().numpy()) * 1).reshape(
                    -1)
                accuracy = result.sum()
                correct_torch = correct_torch + accuracy

        print("Pytorch Test Accuracy: {}%".format(
            100 * correct_torch / test_data_size) + " " + "Mindpsore Test Accuacy: {}".format(
            correct_ms / test_data_size))


def test_retinaface():
    a = np.load("./dataset/WIDERFACE/image2.npy", allow_pickle=False)
    b = np.load("./dataset/WIDERFACE/truths3.npy", allow_pickle=False)
    c = np.load("./dataset/WIDERFACE/conf20.npy", allow_pickle=False)
    d = np.load("./dataset/WIDERFACE/landm17.npy", allow_pickle=False)
    a_t, b_t, c_t, d_t = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32), torch.tensor(c,dtype=torch.int64), torch.tensor(d, dtype=torch.float32)

    print("a.shape: " + str(a.shape))
    print("b.shape: " + str(b.shape))
    print("c.shape: " + str(c.shape))
    print("d.shape: " + str(d.shape))

    a_ms, b_ms, c_ms, d_ms = mindspore.Tensor(a, dtype=mindspore.float32), \
                             mindspore.Tensor(b, dtype=mindspore.float32), \
                             mindspore.Tensor(c, dtype=mindspore.int64), \
                             mindspore.Tensor(d, dtype=mindspore.float32)

    model1, model2 = get_model("retinaface")
    loss_fun1, loss_fun2 = get_loss("retinafacemultix")

    pred_loc, pre_conf, pre_landm = model1(a_ms)
    loss_ms_result = loss_fun1(pred_loc, pre_conf, pre_landm, b_ms, c_ms, d_ms)

    pred_loc, pre_conf, pre_landm = model2(a_t)
    loss_t_result = loss_fun2(pred_loc, pre_conf, pre_landm, b_t, c_t, d_t)

def test_yolov4():
    model1, model2 = get_model("yolov4")
    y0 = np.random.randn(1, 13, 13, 3, 85)
    y1 = np.random.randn(1, 26, 26, 3, 85)
    y2 = np.random.randn(1, 52, 52, 3, 85)
    gt0 = np.random.randn(1, 90, 4)
    gt1 = np.random.randn(1, 90, 4)
    gt2 = np.random.randn(1, 90, 4)

    data = torch.randn(1, 3, 416, 416)
    y_true_0 = torch.tensor(y0, dtype=torch.float32)
    y_true_1 = torch.tensor(y1, dtype=torch.float32)
    y_true_2 = torch.tensor(y2, dtype=torch.float32)
    gt_0 = torch.tensor(gt0, dtype=torch.float32)
    gt_1 = torch.tensor(gt1, dtype=torch.float32)
    gt_2 = torch.tensor(gt2, dtype=torch.float32)
    input_shape = data.shape[2:4]
    input_shape = torch.tensor(tuple(input_shape[::-1]), dtype=torch.float32)

    # images = mindspore.Tensor(np.random.randn(1, 3, 416, 416), dtype=mindspore.float32)
    # y_true_0 = mindspore.Tensor(np.random.randn(1, 13, 13, 3, 85), dtype=mindspore.float32)
    # y_true_1 = mindspore.Tensor(np.random.randn(1, 26, 26, 3, 85), dtype=mindspore.float32)
    # y_true_2 = mindspore.Tensor(np.random.randn(1, 52, 52, 3, 85), dtype=mindspore.float32)
    # gt_0 = mindspore.Tensor(np.random.randn(1, 90, 4), dtype=mindspore.float32)
    # gt_1 = mindspore.Tensor(np.random.randn(1, 90, 4), dtype=mindspore.float32)
    # gt_2 = mindspore.Tensor(np.random.randn(1, 90, 4), dtype=mindspore.float32)
    # in_shape = images.shape[2:4]
    # in_shape = mindspore.Tensor(tuple(in_shape), dtype=mindspore.float32)

    loss_ms, loss_torch = get_loss("yololoss")
    # opt = nn_ms.Momentum(params=model1.trainable_params(),learning_rate=1e-4,momentum=0.9,weight_decay=0.0005,loss_scale= 1024)
    # yolo_network_out= model1(images)
    # loss = loss_ms(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2,  in_shape)
    #
    #
    # def forward_fn(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape):
    #     loss = loss_ms(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape)
    #
    #     return loss
    #
    #
    # grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)
    #
    #
    # def train_step(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape):
    #     (loss), grads = grad_fn(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape)
    #     loss = mindspore.ops.depend(loss, opt(grads))
    #     return loss
    #
    #
    # print("================================================================")
    # yolo_network_out = model1(images)
    # loss_ms = train_step(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, in_shape)
    # print("loss_ms", loss_ms)
    # print("================================================================")

    yolo_output = model2(data)
    optimizer = torch.optim.SGD(model2.parameters(),
                                lr=1e-4, momentum=0.9, weight_decay=0.0005)
    loss = loss_torch(yolo_output, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_yolov5():
    model1, model2 = get_model("yolov5")

    # data0 = mindspore.Tensor(np.random.randn(1, 640, 640, 3), dtype=mindspore.float32)
    # data1 = mindspore.Tensor(np.random.randn(1, 20, 20, 3, 85), dtype=mindspore.float32)
    # data2 = mindspore.Tensor(np.random.randn(1, 40, 40, 3, 85), dtype=mindspore.float32)
    # data3 = mindspore.Tensor(np.random.randn(1, 80, 80, 3, 85), dtype=mindspore.float32)
    # data4 = mindspore.Tensor(np.random.randn(1, 150, 4), dtype=mindspore.float32)
    # data5 = mindspore.Tensor(np.random.randn(1, 150, 4), dtype=mindspore.float32)
    # data6 = mindspore.Tensor(np.random.randn(1, 150, 4), dtype=mindspore.float32)
    # images = data0
    # input_shape = images.shape[1:3]
    # input_shape = mindspore.Tensor(input_shape, mindspore.float32)
    #
    # yolo_network_out=model1(data0)
    loss_ms, loss_torch = get_loss("yololoss")
    # loss_result_ms=loss_ms(yolo_network_out, data1, data2, data3, data4, data5, data6,input_shape)
    data = torch.randn(1, 640, 640, 3)
    data1 = torch.randn(1, 20, 20, 3, 85)
    data2 = torch.randn(1, 40, 40, 3, 85)
    data3 = torch.randn(1, 80, 80, 3, 85)
    data4 = torch.randn(1, 150, 4)
    data5 = torch.randn(1, 150, 4)
    data6 = torch.randn(1, 150, 4)
    input_shape = data.shape[2:4]
    input_shape = torch.tensor(tuple(input_shape[::-1]), dtype=torch.float32)
    yolo_network_out = model2(data)

    optimizer = torch.optim.SGD(model2.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0005)
    loss = loss_torch(yolo_network_out, data1, data2, data3, data4, data5, data6, input_shape)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test_ssd_multibox():
    model1, model2 = get_model("SSDvgg16")
    loss_ms, loss_torch = get_loss('ssdmultix')

    image = mindspore.Tensor(np.random.rand(1, 3, 640, 640), mindspore.float32)
    num_matched_boxes = mindspore.Tensor([[33]], mindspore.int32)
    gt_label = mindspore.Tensor(np.random.randn(1, 38600), mindspore.int32)  # vgg16/mobilenetv1/mobilenetv2 38600 resnet/mobienetv1-fpn 51150
    get_loc = mindspore.Tensor(np.random.randn(1, 38600, 4), mindspore.float32)

    # pred_loc, pred_label=model1(image)
    # loss_result=loss_ms(pred_loc, pred_label, get_loc, gt_label, num_matched_boxes)
    # print("loss_ms: " + str(loss_result))

    image = torch.tensor(np.random.randn(1, 3, 640, 640), dtype=torch.float32)
    num_matched_boxes = torch.tensor([[33]], dtype=torch.int32)
    gt_label = torch.tensor(np.random.rand(1, 38600), dtype=torch.int32)
    gt_loc = torch.tensor(np.random.randn(1, 38600, 4), dtype=torch.float32)
    optimizer = torch.optim.SGD(model2.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0005)

    pred_loc, pred_label = model2(image)
    loss = loss_torch(pred_loc, pred_label, gt_loc, gt_label, num_matched_boxes)

    optimizer.zero_grad()
    loss.backward()
    print("loss_torch: "+str(loss))
    optimizer.step()

def test_yolov3():
    model1, model2 = get_model("yolov3")
    y0 = np.random.randn(1, 13, 13, 3, 85)
    y1 = np.random.randn(1, 26, 26, 3, 85)
    y2 = np.random.randn(1, 52, 52, 3, 85)
    gt0 = np.random.randn(1, 50, 4)
    gt1 = np.random.randn(1, 50, 4)
    gt2 = np.random.randn(1, 50, 4)

    data = torch.randn(1, 3, 416, 416)
    y_true_0 = torch.tensor(y0, dtype=torch.float32)
    y_true_1 = torch.tensor(y1, dtype=torch.float32)
    y_true_2 = torch.tensor(y2, dtype=torch.float32)
    gt_0 = torch.tensor(gt0, dtype=torch.float32)
    gt_1 = torch.tensor(gt1, dtype=torch.float32)
    gt_2 = torch.tensor(gt2, dtype=torch.float32)
    input_shape = data.shape[2:4]
    input_shape = torch.tensor(tuple(input_shape[::-1]), dtype=torch.float32)

    images = mindspore.Tensor(np.random.randn(1, 3, 416, 416), dtype=mindspore.float32)
    y_true_0 = mindspore.Tensor(np.random.randn(1, 13, 13, 3, 85), dtype=mindspore.float32)
    y_true_1 = mindspore.Tensor(np.random.randn(1, 26, 26, 3, 85), dtype=mindspore.float32)
    y_true_2 = mindspore.Tensor(np.random.randn(1, 52, 52, 3, 85), dtype=mindspore.float32)
    gt_0 = mindspore.Tensor(np.random.randn(1, 90, 4), dtype=mindspore.float32)
    gt_1 = mindspore.Tensor(np.random.randn(1, 90, 4), dtype=mindspore.float32)
    gt_2 = mindspore.Tensor(np.random.randn(1, 90, 4), dtype=mindspore.float32)
    in_shape = images.shape[2:4]
    in_shape = mindspore.Tensor(tuple(in_shape), dtype=mindspore.float32)

    loss_ms, loss_torch = get_loss("yololoss")
    opt = nn_ms.Momentum(params=model1.trainable_params(), learning_rate=1e-4, momentum=0.9, weight_decay=0.0005,
                         loss_scale=1024)
    yolo_network_out = model1(images)
    loss = loss_ms(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, in_shape)

    def forward_fn(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape):
        loss = loss_ms(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape)

        return loss

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)

    def train_step(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape):
        (loss), grads = grad_fn(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape)
        loss = mindspore.ops.depend(loss, opt(grads))
        return loss

    print("================================================================")
    yolo_network_out = model1(images)
    loss_ms = train_step(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, in_shape)
    print("loss_ms", loss_ms)
    print("================================================================")

    # yolo_output = model2(data)
    # optimizer = torch.optim.SGD(model2.parameters(),
    #                             lr=1e-4, momentum=0.9, weight_decay=0.0005)
    # loss = loss_torch(yolo_output, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape)
    #
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

def test_unet():
    model1, model2 = get_model("unet")
    loss1, loss2 = get_loss("unetloss")

    # t = np.random.randn(1, 1, 572, 572)
    # a = np.random.randn(1, 2, 388, 388)
    # t = mindspore.Tensor(t, dtype=mindspore.float32)
    # a = mindspore.Tensor(a, dtype=mindspore.float32)
    #
    # optimizer = mindspore.nn.Adam(params=model1.trainable_params(), learning_rate=1e-2, weight_decay=0.0005, loss_scale=1024.0)
    #
    #
    # def forward_fn(data, label):
    #     logits = model1(data)
    #     loss = loss1(logits, label)
    #     return loss, logits
    #
    #
    # grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    #
    #
    # def train_step(data, label):
    #     (loss, _), grads = grad_fn(data, label)
    #     loss = mindspore.ops.depend(loss, optimizer(grads))
    #     return loss
    #
    #
    # print("================================================================")
    # loss_ms = train_step(t, a)
    # print("loss_ms", loss_ms)
    # print("================================================================")

    t = np.random.randn(1, 1, 572, 572)
    a = np.random.randn(1, 2, 388, 388)
    t = torch.tensor(t, dtype=torch.float32)
    a = torch.tensor(a, dtype=torch.float32)

    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-2, weight_decay=0.0005)

    # print(net(t))
    def train_step(data, label):
        data, label = data, label
        optimizer.zero_grad()
        logits = model2(data)
        loss = loss2(logits, label)
        loss.backward()
        optimizer.step()
        return loss.item()

    print("================================================================")
    loss_pytorch = train_step(t, a)
    print(loss_pytorch)

def test_unetplus():
    model1, model2 = get_model("unetplus")
    loss1, loss2 = get_loss("unetloss")

    # optimizer = mindspore.nn.Adam(params=model1.trainable_params(), learning_rate=1e-2, weight_decay=0.0005,loss_scale= 1024.0)
    #
    #
    # def forward_fn(data, label):
    #     logits = model1(data)
    #     loss = loss1(logits, label)
    #     return loss, logits
    #
    #
    # grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    #
    #
    # def train_step(data, label):
    #     (loss, _), grads = grad_fn(data, label)
    #     loss = mindspore.ops.depend(loss, optimizer(grads))
    #     return loss
    #
    #
    # print("================================================================")
    # nt = np.random.randn(1, 1, 96, 96)
    # na = np.random.randn(1, 2, 96, 96)
    # t = mindspore.Tensor(nt, dtype=mindspore.float32)
    # a = mindspore.Tensor(na, dtype=mindspore.float32)
    #
    # loss_result=train_step(t,a)
    # print(loss_result)
    # print("================================================================")

    nt = np.random.randn(1, 1, 96, 96)
    na = np.random.randn(1, 2, 96, 96)
    t = torch.tensor(nt, dtype=torch.float32)
    a = torch.tensor(na, dtype=torch.float32)

    optimizer = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=0.0005)

    def train_step(data, label):
        optimizer.zero_grad()
        data, label = data, label
        logits = model2(data)
        loss = loss2(logits, label)
        loss.backward()
        optimizer.step()
        return loss.item()

    print("================================================================")
    loss_pytorch = train_step(t, a)
    print(loss_pytorch)
    print("================================================================")

def test_fasttext():
    b_size = 16
    model1, model2 = get_model("fasttext", device="GPU", device_id=0, input_size=(b_size, 64))
    loss1, loss2 = get_loss("fasttextloss")

    data1 = np.load("./fasttextt_test_protoken_ext.npy")[:b_size]  # （32，64）
    data2 = np.load("./fasttextt_test_protokenstextlength.npy")[:b_size]  # （32，1）
    data3 = np.load("./fasttextt_test_label.npy")[:b_size]  # （32，1）

    print("data1.shape: " + str(data1.shape))
    print("data2.shape: " + str(data2.shape))
    print("data3.shape: " + str(data3.shape))

    print("data1.dtype: " + str(data1.dtype))
    print("data2.dtype: " + str(data2.dtype))
    print("data3.dtype: " + str(data3.dtype))

    data1_ms = mindspore.Tensor(data1, mindspore.int32)
    data2_ms = mindspore.Tensor(data2, mindspore.int32)
    data3_ms = mindspore.Tensor(data3, mindspore.int32)

    data1_torch = torch.LongTensor(data1).to("cuda:0")
    data2_torch = torch.LongTensor(data2).to("cuda:0")
    data3_torch = torch.LongTensor(data3).to("cuda:0")

    model1out = model1(data1_ms, data2_ms)
    loss_ms = loss1(model1out, data3_ms)

    model2out = model2(data1_torch, data2_torch)
    loss_torch = loss2(model2out, data3_torch)

    print(loss_ms, loss_torch)


def test_textcnn():
    model_name = "textcnn"
    device_target = "CPU"
    device_id = 0
    input_size = (32, 51)
    final_device = "cuda:" + str(device_id)
    final_device = "cpu"

    model1, model2 = get_model(model_name, device_target, device_id=device_id, input_size=tuple(input_size))
    loss1, loss2 = get_loss("textcnnloss")
    num_classes = 2

    data1 = np.load(r"./textcnn_testdata.npy")  # (32,51)
    data2 = np.load(r"./textcnn_label.npy")  # (32,)

    print("data1.shape: " + str(data1.shape))
    print("data2.shape: " + str(data2.shape))

    print("data1.dtype: " + str(data1.dtype))
    print("data2.dtype: " + str(data2.dtype))

    data1_ms = mindspore.Tensor(data1, mindspore.int32)
    data2_ms = mindspore.Tensor(data2, mindspore.int32)

    data1_torch = torch.LongTensor(data1).to(final_device)
    data2_torch = torch.LongTensor(data2).to(final_device)

    model1out = model1(data1_ms)
    loss_ms = loss1(model1out, data2_ms, num_classes)

    model2out = model2(data1_torch)
    loss_torch = loss2(model2out, data2_torch, num_classes)

    print(loss_ms, loss_torch)

def test_deeplabv3plus():
    model1, model2 = get_model("deeplabv3plus")
    loss1, loss2 = get_loss('deepv3plusloss')

    data1 = np.random.randn(4, 3, 513, 513)  # (4, 3, 513, 513)
    data2 = np.random.randn(4, 513, 513)  # (4, 513, 513)

    data1_ms = mindspore.Tensor(data1, mindspore.float32)
    data2_ms = mindspore.Tensor(data2, mindspore.float32)

    data1_torch = torch.tensor(data1, dtype=torch.float32)
    data2_torch = torch.tensor(data2, dtype=torch.float32)

    model1out = model1(data1_ms)
    loss_ms = loss1(model1out, data2_ms)

    model2out = model2(data1_torch)
    loss_torch = loss2(model2out, data2_torch)

    print(loss_ms, loss_torch)

def test_deeplabv3():
    model1, model2 = get_model("deeplabv3")
    loss1, loss2 = get_loss('deepv3plusloss')

    data1 = np.random.randn(4, 3, 513, 513)  # (4, 3, 513, 513)
    data2 = np.random.randn(4, 513, 513)  # (4, 513, 513)

    data1_ms = mindspore.Tensor(data1, mindspore.float32)
    data2_ms = mindspore.Tensor(data2, mindspore.float32)

    data1_torch = torch.tensor(data1, dtype=torch.float32)
    data2_torch = torch.tensor(data2, dtype=torch.float32)

    model1out = model1(data1_ms)
    loss_ms = loss1(model1out, data2_ms)

    model2out = model2(data1_torch)
    loss_torch = loss2(model2out, data2_torch)

    print(loss_ms, loss_torch)


def test_unet3d():
    input_size = (2, 1, 224, 224, 96)
    model1, model2 = get_model("unet3d")
    loss_ms, loss_torch = get_loss("unet3dloss")

    input_np = np.random.randn(2, 1, 224, 224, 96)
    label = np.random.randn(2, 4, 224, 224, 96)

    input_data_ms = mindspore.Tensor(input_np, dtype=mindspore.float32)
    label_data_ms = mindspore.Tensor(label, dtype=mindspore.float32)

    input_data_torch = torch.tensor(input_np, dtype=torch.float32)
    label_data_torch = torch.tensor(label, dtype=torch.float32)

    optimizer_ms = nn_ms.Adam(params=model1.trainable_params(), learning_rate=0.0005)
    optimizer_torch = torch.optim.Adam(params=model2.parameters(), lr=0.0005)

    logits_ms = model1(input_data_ms)
    logits_torch = model2(input_data_torch)

    def forward_fn(logits, label):
        loss = loss_ms(logits, label)
        return loss, logits

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer_ms.parameters, has_aux=False)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, optimizer_ms(grads))
        return loss

    loss_ms = train_step(logits_ms, label_data_ms)
    print(loss_ms)

    loss_torch = loss_torch(logits_torch, label_data_torch)
    print(loss_torch)

    loss_torch.backward()
    optimizer_torch.step()
    optimizer_torch.zero_grad()

def test_transformer():
    model1, model2 = get_model("transformer")
    loss_ms, loss_torch = get_loss("transformerloss")

    lr = 1e-4
    # optimizer = mindspore.nn.Adam(model1.trainable_params(), lr)
    optimizer = torch.optim.Adam(model2.parameters(), lr)
    a = np.load("transformer_testdata0.npy")
    b = np.load("transformer_testdata1.npy")
    c = np.load("transformer_testdata2.npy")
    d = np.load("transformer_testdata3.npy")
    e = np.load("transformer_testdata4.npy")
    f = np.load("transformer_testdata5.npy")
    a = torch.tensor(a, dtype=torch.int64)
    b = torch.tensor(b, dtype=torch.int64)
    c = torch.tensor(c, dtype=torch.int64)
    d = torch.tensor(d, dtype=torch.int64)
    e = torch.tensor(e, dtype=torch.int64)
    f = torch.tensor(f, dtype=torch.int64)

    # source_ids = mindspore.Tensor(np.load("transformer_testdata0.npy"), mindspore.int32)  # (96,16)
    # source_mask = mindspore.Tensor(np.load("transformer_testdata1.npy"), mindspore.int32)  # (96,16)
    # target_ids = mindspore.Tensor(np.load("transformer_testdata2.npy"), mindspore.int32)  # (96,16)
    # target_mask = mindspore.Tensor(np.load("transformer_testdata3.npy"), mindspore.int32)  # (96,16)
    # label_ids = mindspore.Tensor(np.load("transformer_testdata4.npy"), mindspore.int32)  # (96,16)
    # label_weights =mindspore.Tensor(np.load("transformer_testdata5.npy"), mindspore.int32)  # (96,16)
    #
    #
    # def forward_fn(data0, data1, data2, data3, data4, data5):
    #     prediction_scores = model1(data0, data1, data2, data3)
    #     seq_length = mindspore.ops.Shape()(data0)[1]
    #     loss = loss_ms(prediction_scores, seq_length, data4, data5)
    #     return loss
    #
    #
    # grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
    #
    #
    # def train_step(data0, data1, data2, data3, data4, data5):
    #     (loss), grads = grad_fn(data0, data1, data2, data3, data4, data5)
    #     loss = mindspore.ops.depend(loss, optimizer(grads))
    #     return loss
    #
    #
    # loss_result = train_step(source_ids, source_mask, target_ids, target_mask, label_ids, label_weights)
    # print(loss_result)

    prediction_scores = model2(a, b, c, d)
    loss_result = loss_torch(a, prediction_scores, e, f)

    print(loss_result)
    loss_result.backward()
    optimizer.step()
    optimizer.zero_grad()

def test_bert():
    model1, model2 = get_model("bert", "CPU", device_id=0, input_size=(10,384))
    loss1, loss2 = get_loss('bertloss')
    # data0 = torch.tensor(np.load("./bert_data0.npy"), dtype=torch.int64)
    # data1 = torch.tensor(np.load("./bert_data1.npy"), dtype=torch.int64)
    # data2 = torch.tensor(np.load("./bert_data2.npy"), dtype=torch.int64)
    # data3 = torch.tensor(np.load("./bert_data3.npy"), dtype=torch.int64)
    # data4 = torch.tensor(np.load("./bert_data4.npy"), dtype=torch.int64)
    # data5 = torch.tensor(np.load("./bert_data5.npy"), dtype=torch.int64)
    # data6 = torch.tensor(np.load("./bert_data6.npy"), dtype=torch.int64)
    # output = model2(data0, data1, data2)
    # loss = loss2(output, data3, data4)
    # loss.backward()
    # print("loss_torch: "+str(loss))

    # data0 = mindspore.Tensor(np.load("./bert_data0.npy"), dtype=mindspore.int64)
    # data1 = mindspore.Tensor(np.load("./bert_data1.npy"), dtype=mindspore.int64)
    # data2 = mindspore.Tensor(np.load("./bert_data2.npy"), dtype=mindspore.int64)
    # data3 = mindspore.Tensor(np.load("./bert_data3.npy"), dtype=mindspore.int64)
    # data4 = mindspore.Tensor(np.load("./bert_data4.npy"), dtype=mindspore.int64)
    # data5 = mindspore.Tensor(np.load("./bert_data5.npy"), dtype=mindspore.int64)
    # data6 = mindspore.Tensor(np.load("./bert_data6.npy"), dtype=mindspore.int64)

    data0 = mindspore.Tensor(np.random.randn(10,384), dtype=mindspore.int64)
    data1 = mindspore.Tensor(np.random.randn(10,384), dtype=mindspore.int64)
    data2 = mindspore.Tensor(np.random.randn(10,384), dtype=mindspore.int64)



    output = model1(data0, data1, data2)
    print(type(output))
    print(data0.shape)
    # loss = loss1(output, data3, data4)
    # print("loss_ms: " + str(loss))


def get_free_GraphiscCard():
    UNIT = 1024 * 1024

    pynvml.nvmlInit()  # 初始化
    gpuDeriveInfo = pynvml.nvmlSystemGetDriverVersion()
    print("Drive版本: ", str(gpuDeriveInfo))  # 显示驱动信息

    gpuDeviceCount = pynvml.nvmlDeviceGetCount()  # 获取Nvidia GPU块数
    print("GPU个数：", gpuDeviceCount)

    for i in range(gpuDeviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 获取GPU i的handle，后续通过handle来处理

        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通过handle获取GPU i的信息

        gpuName = str(pynvml.nvmlDeviceGetName(handle))

        gpuTemperature = pynvml.nvmlDeviceGetTemperature(handle, 0)

        gpuFanSpeed = pynvml.nvmlDeviceGetFanSpeed(handle)

        gpuPowerState = pynvml.nvmlDeviceGetPowerState(handle)

        gpuUtilRate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpuMemoryRate = pynvml.nvmlDeviceGetUtilizationRates(handle).memory

        # print("第 %d 张卡：" % i, "-" * 30)
        # print("显卡名：", gpuName)
        # print("内存总容量：", memoryInfo.total / UNIT, "MB")
        # print("使用容量：", memoryInfo.used / UNIT, "MB")
        # print("剩余容量：", memoryInfo.free / UNIT, "MB")
        # print("显存空闲率：", memoryInfo.free / memoryInfo.total)
        # print("温度：", gpuTemperature, "摄氏度")
        # print("风扇速率：", gpuFanSpeed)
        # print("供电水平：", gpuPowerState)
        # print("gpu计算核心满速使用率：", gpuUtilRate)
        # print("gpu内存读写满速使用率：", gpuMemoryRate)
        # print("内存占用率：", memoryInfo.used / memoryInfo.total)
        free=memoryInfo.free / memoryInfo.total
        if free>0.5:
            return i



# if __name__ == '__main__':
#     from common.mutation_ms.Layer_utils import CascadeOPUtils as msCascadeOP
#     from common.mutation_ms.Layer_utils import BasicOPUtils as msBasicOP
#
#     from common.mutation_torch.Layer_utils import CascadeOPUtils as torchCascadeOP
#     from common.mutation_torch.Layer_utils import BasicOPUtils as torchBasicOP
#     from mindspore import context
#     context.set_context(device_target="GPU")
#
#     in_channels, out_channels = 16, 16
#     kernel_size = 3
#     stride = 2
#
#     data = np.random.randn(2, in_channels, 16, 16)
#     data_ms = mindspore.Tensor(data, mindspore.float32)
#     data_torch = torch.tensor(data, dtype=torch.float32).cuda()
#
#     activations_alter_ms = msBasicOP.available_activations()
#     activations_alter_torch = torchBasicOP.available_activations()
#     acts_ms = list(activations_alter_ms.keys())
#     acts_torch = list(activations_alter_torch.keys())
#
#     # already test PASS
#     # net1 = msBasicOP.available_convs(in_channels, out_channels, kernel_size, stride, "conv2d")
#     # net2 = torchBasicOP.available_convs(in_channels, out_channels, kernel_size, stride, "conv2d")
#     # ms_result = net1[1](data_ms)
#     # torch_result = net2[1](data_torch)
#     # print(np.max(np.abs(ms_result.asnumpy() - torch_result.detach().cpu().numpy())))
#
#     # already test PASS
#     # net1 = msBasicOP.available_Dense(in_feature=16, out_feature=10, has_bias=True)
#     # net2 = torchBasicOP.available_Dense(in_feature=16, out_feature=10, has_bias=True).cuda()
#     # ms_result = net1(data_ms)
#     # torch_result = net2(data_torch)
#     # print(np.max(np.abs(ms_result.asnumpy() - torch_result.detach().cpu().numpy())))
#
#     #already test PASS
#     # net1s = msBasicOP.available_pool(4, stride, "avgpool2d")
#     # net2s = torchBasicOP.available_pool(4, stride, "avgpool2d")
#     #
#     # net1 = net1s[1]
#     # net2 = net2s[1].cuda()
#     #
#     # ms_result = net1(data_ms)
#     # torch_result = net2(data_torch)
#     # print(np.max(np.abs(ms_result.asnumpy() - torch_result.detach().cpu().numpy())))
#
#
#
#     #shape inconsistency
#     convbnrelu_ms = msCascadeOP.avaiable_convbnrelu(in_channels, out_channels, kernel_size, stride=stride)
#     convbnrelu_torch = torchCascadeOP.avaiable_convbnrelu(in_channels, out_channels, kernel_size, stride=stride)
#
#
#
#     weight_dict = torch.load('/data1/CKPTS/convbnrelu/convbnrelu.pth')
#     convbnrelu_torch.load_state_dict(weight_dict, strict=False)
#
#     mindspore.load_checkpoint("/data1/CKPTS/convbnrelu/convbnrelu.ckpt", convbnrelu_ms)
#
#
#
#
#
#
#     convbnrelu_ms_result = convbnrelu_ms(data_ms).asnumpy()
#     convbnrelu_torch_result = convbnrelu_torch(data_torch).detach().cpu().numpy()
#     print("convbnrelu result: ", np.max(np.abs(convbnrelu_torch_result-convbnrelu_ms_result)))
#
#     # shape inconsistency
#     # downSample_ms = msCascadeOP.avaiable_downSample(in_channels, out_channels, kernel_size, stride=stride)
#     # downSample_torch = torchCascadeOP.avaiable_downSample(in_channels, out_channels, kernel_size, stride=stride)
#     # downSample_ms_result = downSample_ms(data_ms).asnumpy()
#     # downSample_torch_result = downSample_torch(data_torch).detach().numpy()
#     # print("downSample result: ", np.max(np.abs(downSample_torch_result - downSample_ms_result)))
#
#     #output inconsistency
#     # se_ms = msCascadeOP.avaiable_se()
#     # se_torch = torchCascadeOP.avaiable_se()
#     # se_ms_result = se_ms(data_ms).asnumpy()
#     # se_torch_result = se_torch(data_torch).detach().numpy()
#     # print("se result: ", np.max(np.abs(se_torch_result - se_ms_result)))
#
#     # output inconsistency
#     # de_ms = msCascadeOP.avaiable_de(in_channels,out_channels)
#     # de_torch = torchCascadeOP.avaiable_de(in_channels, out_channels)
#     # de_ms_result = de_ms(data_ms).asnumpy()
#     # de_torch_result = de_torch(data_torch).detach().numpy()
#     # print("de result: ", np.max(np.abs(de_torch_result - de_ms_result)))
#
#     # output inconsistency
#     # inception_ms = msCascadeOP.avaiable_Inception()
#     # inception_torch = torchCascadeOP.avaiable_Inception()
#     # inception_ms_result = inception_ms(data_ms).asnumpy()
#     # inception_torch_result = inception_torch(data_torch).detach().numpy()
#     # print("inception result: ", np.max(np.abs(inception_torch_result - inception_ms_result)))
#
#     # shape inconsistency
#     # dwpw_ms = msCascadeOP.avaiable_dwpw(in_channels,out_channels, kernel_size, stride=stride,activation=acts_ms[0])
#     # dwpw_torch = torchCascadeOP.avaiable_dwpw(in_channels, out_channels, kernel_size, stride=stride, activation=acts_torch[0])
#     # dwpw_ms_result = dwpw_ms(data_ms).asnumpy()
#     # dwpw_torch_result = dwpw_torch(data_torch).detach().numpy()
#     # print("dwpw result: ", np.max(np.abs(dwpw_torch_result - dwpw_ms_result)))
#
#
#     # ResidualBlock_ms = msCascadeOP.avaiable_ResidualBlock(in_channels, out_channels, kernel_size, stride=stride, activation=acts_ms[0])
#     # ResidualBlock_torch = torchCascadeOP.avaiable_ResidualBlock(in_channels, out_channels, kernel_size, stride=stride, activation=acts_torch[0])
#     #
#     # #output inconsistency
#     # ResidualBlock_ms_result = ResidualBlock_ms[0](data_ms).asnumpy()
#     # ResidualBlock_torch_result = ResidualBlock_torch[0](data_torch).detach().numpy()
#     # print("ResidualBlock result: ", np.max(np.abs(ResidualBlock_torch_result - ResidualBlock_ms_result)))
#     #
#     #
#     # # output inconsistency
#     # ResidualBlock_ms_result = ResidualBlock_ms[1](data_ms).asnumpy()
#     # ResidualBlock_torch_result = ResidualBlock_torch[1](data_torch).detach().numpy()
#     # print("ResidualBlock result: ", np.max(np.abs(ResidualBlock_torch_result - ResidualBlock_ms_result)))

if __name__ == '__main__':
    import mindspore
    import numpy as np

    d = mindspore.Tensor(np.random.randn(*(1, 3, 224, 224)), mindspore.float32)
    op = mindspore.nn.Conv2d(3, 64, 1, 1).to_float(mindspore.float16)
    op(d)


    import mindspore
    import numpy as np

    d = mindspore.Tensor(np.random.randn(*(1, 3, 224, 224)), mindspore.float16)
    op = mindspore.nn.Conv2d(3, 64, 1, 1)
    op(d)

    # data_0 = np.random.randn(*(16000, 39))
    # data_1 = np.random.randn(*(16000, 39))
    #
    # data1 = mindspore.Tensor(data_0, mindspore.int32)
    # data2 = mindspore.Tensor(data_1, mindspore.float32)
    # 
    # out1 = model1(data1,data2)

    # data2 = torch.tensor(data, dtype=torch.float32).cuda()
    # out2 = model2(data2)

    # output_ms = YoloUtil.reformat_outputs_second_generation(out1)
    # output_torch = YoloUtil.reformat_outputs_second_generation(out2)
