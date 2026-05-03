import numpy as np
import pandas as pd
from torch_mutation.cargo import *
import torch
import copy
import time
import json
import torch.fx as fx
from torch_mutation.MR_structure import *
from torch_mutation.cargo import compute_gpu_cpu
from torch_mutation.api_mutation import api_mutation
from torch_mutation.metrics import MAEDistance
from torch_mutation.cargo import get_loss
import sys
import torch_mutation.config as config
from torch_mutation.handel_shape import handle_format
import metrics
MR_structure_name_list = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']
MR_structures_map = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B}
nlp_cargo = ["LSTM","FastText", "TextCNN", "SentimentNet", "GPT"]
device = config.device
deadcode_name_list=['Dense', 'SELayer', 'DenseLayer', 'Inception_A', 'PWDWPW_ResidualBlock', 'ResidualBlock', 'DropPath']
def run_log_torch(seed_model, mutate_times,log_path, num_samples,base_path,train_config,execution_config):
    localtime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    with open(log_path, "r") as json_file:
        log_dict = json.load(json_file)

    if isinstance(datasets_path_cargo[seed_model],list):
        data_0 = np.load(datasets_path_cargo[seed_model][0])
        data_1 = np.load(datasets_path_cargo[seed_model][1])
        samples_0 = np.random.choice(data_0.shape[0], num_samples, replace=False)
        samples_data_0 = data_0[samples_0] # 随机选择num_samples个数据 (num_samples, 3, 32, 32)
        samples_1 = np.random.choice(data_1.shape[0], num_samples, replace=False)
        samples_data_1 = data_1[samples_1] # 随机选择num_samples个数据 (num_samples, 3, 32, 32)   
        data_selected_0 = torch.tensor(samples_data,dtype=torch.float32 if seed_model in nlp_cargo else torch.int32)
        data_selected_1 = torch.tensor(samples_data,dtype=torch.float32 if seed_model in nlp_cargo else torch.int32)
        data_selected = (data_selected_0, data_selected_1)
        data_npy = [data_selected_0.cpu().numpy(), data_selected_1.cpu().numpy()]

        npy_path = os.path.join("results", seed_model, str(localtime), 'data0_npy.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, data_npy[0])
        npy_path = os.path.join("results", seed_model, str(localtime), 'data1_npy.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, data_npy[1])
    else:
        dataset_basic_dir = "/home/cvgroup/myz/czx/SemTest_master/test_data"
        labels_path_cargo = {
            "vgg16": dataset_basic_dir+"/cifar10_y.npy", #1
            "resnet": dataset_basic_dir+"/cifar10_y.npy", #4
            "mobilenetv2":dataset_basic_dir+"/cifar10_y.npy",
            "vit":dataset_basic_dir+"/cifar10_y.npy",
            "yolov3": [dataset_basic_dir+"/yolov3_y1.npy",
                    dataset_basic_dir+"/yolov3_y2.npy",
                    dataset_basic_dir+"/yolov3_y3.npy",
                    dataset_basic_dir + "/yolov3_y4.npy",
                    dataset_basic_dir + "/yolov3_y5.npy",
                    dataset_basic_dir + "/yolov3_y6.npy"],
            "yolov4": [dataset_basic_dir+"/yolov4_y1.npy",
                    dataset_basic_dir+"/yolov4_y2.npy",
                    dataset_basic_dir+"/yolov4_y3.npy",
                    dataset_basic_dir + "/yolov4_y4.npy",
                    dataset_basic_dir + "/yolov4_y5.npy",
                    dataset_basic_dir + "/yolov4_y6.npy"],
            "TextCNN": dataset_basic_dir+"/textcnn_y.npy",  # 12
            "SSDresnet50fpn": [dataset_basic_dir+"/SSDresnet50fpn_y1.npy",dataset_basic_dir+"/SSDresnet50fpn_y2.npy",dataset_basic_dir+"/SSDresnet50fpn_y3.npy"],
            "SSDmobilenetv1": [dataset_basic_dir+"/SSDmobilenetv1_y1.npy",dataset_basic_dir+"/SSDmobilenetv1_y2.npy",dataset_basic_dir+"/SSDmobilenetv1_y3.npy"],# 8
            "unet": dataset_basic_dir+"/unet_y.npy", #9
            "openpose": [dataset_basic_dir+"/openpose_y1.npy",dataset_basic_dir+"/openpose_y2.npy",dataset_basic_dir+"/openpose_y3.npy"],
            "crnn":dataset_basic_dir+"/CRNN_y.npy",
            "DeepLabV3": dataset_basic_dir+"/deeplabv3_y.npy", #10
        }

        data = np.load(datasets_path_cargo[seed_model])
        indexs = [i for i in range(data.shape[0])]
        samples = np.random.permutation(indexs)[:num_samples]

        data_x_path = base_path + 'data0_npy_x.npy'
        # samples_data = np.load(data_x_path)
        samples_data = data[samples]
        # indexs = [i for i in range(samples_data.shape[0])]
        # samples = np.random.permutation(indexs)[:num_samples]
        # samples_data = data[samples] # 随机选择num_samples个数据 (num_samples, 3, 32, 32)
        data_selected = torch.tensor(samples_data,dtype=torch.int32 if seed_model in nlp_cargo else torch.float32).to(device)
        data_npy = data_selected.cpu().numpy()

        npy_path = os.path.join("results", seed_model, str(localtime), 'data0_npy_x.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, data_npy)

        if seed_model == "openpose" or "SSD" in seed_model:
            labels_path1 = os.path.join("results", seed_model, str(localtime), 'data1_npy_y.npy')
            labels_path2 = os.path.join("results", seed_model, str(localtime), 'data2_npy_y.npy')
            labels_path3 = os.path.join("results", seed_model, str(localtime), 'data3_npy_y.npy')
            targets1, targets2, targets3 = labels_path_cargo[seed_model]
            samples_label1 = np.load(targets1)[samples]
            samples_label2 = np.load(targets2)[samples]
            samples_label3 = np.load(targets3)[samples]

            if seed_model == "openpose":
                dt = torch.float32
            elif "SSD" in seed_model:
                dt = torch.int32

            label_selected1 = torch.tensor(samples_label1,dtype=torch.float32).to(device).long()
            label_selected2 = torch.tensor(samples_label2, dtype=dt).to(device).long()
            label_selected3 = torch.tensor(samples_label3, dtype=dt).to(device).long()
            np.save(labels_path1, samples_label1)
            np.save(labels_path2, samples_label2)
            np.save(labels_path3, samples_label3)


        elif seed_model in [ "ssimae"]:
            # data_selected = Tensor(samples_data, dtype=mstype.float64)
            pass
        elif seed_model in ["crnn", "DeepLabV3","TextCNN", "resnet", 'vit']:
            targets = np.load(labels_path_cargo[seed_model])
            print("targets",targets.shape)
            samples_label = targets[samples]
            print(samples)
            print("samples_label",samples_label.shape)
            label_selected = torch.tensor(samples_label,dtype=torch.int32).to(device)
            print("label_selected",label_selected.shape)
            # np.save(labels_path, label_selected)

        elif seed_model in ["yolov3", "yolov4"]:

            labels_path1 = os.path.join("results", seed_model, str(localtime), 'data1_npy_y.npy')
            labels_path2 = os.path.join("results", seed_model, str(localtime), 'data2_npy_y.npy')
            labels_path3 = os.path.join("results", seed_model, str(localtime), 'data3_npy_y.npy')
            labels_path4 = os.path.join("results", seed_model, str(localtime), 'data4_npy_y.npy')
            labels_path5 = os.path.join("results", seed_model, str(localtime), 'data5_npy_y.npy')
            labels_path6 = os.path.join("results", seed_model, str(localtime), 'data6_npy_y.npy')

            targets1, targets2, targets3, targets4, targets5, targets6 = labels_path_cargo[seed_model]
            samples_label1 = np.load(targets1)
            samples_label2 = np.load(targets2)
            samples_label3 = np.load(targets3)
            samples_label4 = np.load(targets4)
            samples_label5 = np.load(targets5)
            samples_label6 = np.load(targets6)

            # if seed_model == "openpose":
            dt = torch.float32
            # elif "SSD" in seed_model:
            #     dt = mstype.int32

            label_selected1 = torch.tensor(samples_data,dtype=torch.float32)
            label_selected2 = torch.tensor(samples_label2, dtype=dt)
            label_selected3 = torch.tensor(samples_label3, dtype=dt)
            label_selected4 = torch.tensor(samples_data,dtype=torch.float32)
            label_selected5 = torch.tensor(samples_label5, dtype=dt)
            label_selected6 = torch.tensor(samples_label6, dtype=dt)
            np.save(labels_path1, samples_label1)
            np.save(labels_path2, samples_label2)
            np.save(labels_path3, samples_label3)
            np.save(labels_path4, samples_label4)
            np.save(labels_path5, samples_label5)
            np.save(labels_path6, samples_label6)
        else:
            # labels_path = os.path.join("results", seed_model, str(localtime), 'data1_npy_y.npy')
            print(labels_path_cargo[seed_model])
            targets = np.load(labels_path_cargo[seed_model])
            print("targets:",targets.shape)
            samples_label = targets[samples]
            label_selected = torch.tensor(samples_label,dtype=torch.float32).to(device)
            # np.save(labels_path, label_selected)


    seed_model_net=get_model(seed_model, device).to(device)
    seed_optimizer = train_config['opt_name']
    seed_optimizer_ms, seed_optimizer_pt = get_optimizer(seed_optimizer)
    if seed_model in ["ssimae"]:
        pass
    elif seed_model =="patchcore":
        from models.PatchCore.src.oneStep import OneStepCell
        seed_model_net = OneStepCell(seed_model_net)

    else:

        if "SSD" in seed_model:
            loss_name = "ssdmultix"

        elif seed_model in ["TextCNN"]:
            loss_name = "textcnnloss"

        elif seed_model in ["resnet","openpose","crnn","DeepLabV3","ssimae","vit"]:
            loss_name = "CrossEntropy"

        elif seed_model in ["yolov3", "yolov4"]:
            loss_name = "yolov4loss"

        elif seed_model in ["unet"]:
            loss_name = "unetloss"
        else:
            loss_name = "CrossEntropy"
        print("loss_name:",loss_name)
        loss_fun_ms, loss_fun_pt = get_loss(loss_name)
        loss_fun_pt = loss_fun_pt()

    # seed_optimizer = log_dict[str(n)]['opt_name']
    # seed_optimizer_ms, seed_optimizer_pt = get_optimizer(seed_optimizer)

    d=fx.symbolic_trace(seed_model_net)
    d.eval()
    with torch.no_grad():

        original_outputs = handle_format(d(data_selected))[0] 

    metrics_dict = dict()
    D = {seed_model: d} # 所有模型
    print(D)
    d = copy.deepcopy(d)

    # 循环变异：
    for n in range(mutate_times):
        print('-----------------------total_Mutate_time:%d start!-----------------------' % n)
        # with torch.no_grad():
        if "Success" in log_dict[str(n)]['state']: # 对于变异成功的模型：正常生成变异模型，正常检测
            start_time = time.time()

            # 选择死代码
            selected_deadcode_name =log_dict[str(n)]['select_deadcode']

            # 选择 MR_structure
            selected_MR_structure_name = log_dict[str(n)]['selected_MR_structure']

            # 命名新模型：种子模型、MR_structure、第几次用这个MR_structure
            d_new_name = log_dict[str(n)]['d_new_name']

            # 算子变异对死代码还是原模型？
            api_mutation_type = log_dict[str(n)]['api_mutation_type(seed_model or deadcode)']

            graph = d.graph
            nodelist = []
            for node in graph.nodes:
                if node.op in ['call_module', 'root'] or (node.op == "call_function" and any(substring in node.name for substring in ['uoc', 'pioc', 'absoc_a', 'absoc_b'])):
                    nodelist.append(node)
            print("nodelist:",len(nodelist))
            # 选择插入位置
            subs_place, dep_places = log_dict[str(n)]['subs_place'], log_dict[str(n)]['dep_places']
            if subs_place > len(nodelist) or any(x > len(nodelist) for x in dep_places):
                subs_place, dep_places = select_places(range(0, len(nodelist)), 5)
            else:
                subs_place, dep_places = log_dict[str(n)]['subs_place'], log_dict[str(n)]['dep_places']
            if subs_place is None or dep_places is None: # 没找到合适的插入位置
                sys.exit("mutate failed for Cannot find suitable places！")

            # print("~~~~~~~~~~~~~~~~~选择对%s中的算子进行api变异！~~~~~~~~~~~~~~~" % api_mutation_type)

            # try:
            add_module = MR_structures_map[selected_MR_structure_name](selected_deadcode_name, api_mutation_type, log_dict,n, LOG_FLAG=True)
            # except Exception as e:
            #     exit(e)
            print("subs_place, dep_places",subs_place, dep_places)
            dep_places.sort(reverse=True)
            aa = nodelist[dep_places[-1]]
            bb = nodelist[dep_places[-2]]
            cc = nodelist[dep_places[-3]]
            dd = nodelist[dep_places[-4]]

            if selected_MR_structure_name == "PIOC":
                if len(aa.users) == 0 or len(bb.users) == 0 or len(cc.users) == 0:
                    sys.exit("选择插入的节点位置不正确！")
                with cc.graph.inserting_after(cc):
                    new_hybrid_node = cc.graph.call_function(add_module, args=(cc, cc, cc))
                    cc.replace_all_uses_with(new_hybrid_node)
                    new_hybrid_node.update_arg(0, aa)
                    new_hybrid_node.update_arg(1, bb)
                    new_hybrid_node.update_arg(2, cc)
            else:  # selected_MR_structure_name != "PIOC"
                if len(aa.users) == 0 or len(bb.users) == 0 or len(cc.users) == 0 or len(dd.users) == 0:
                    sys.exit("选择插入的节点位置不正确！")
                with dd.graph.inserting_after(dd):
                    new_hybrid_node = dd.graph.call_function(add_module, args=(dd, dd, dd, dd))
                    dd.replace_all_uses_with(new_hybrid_node)
                    new_hybrid_node.update_arg(0, aa)
                    new_hybrid_node.update_arg(1, bb)
                    new_hybrid_node.update_arg(2, cc)
                    new_hybrid_node.update_arg(3, dd)
            graph.lint()  # 检查是否有图错误并重新编译图
            d.recompile()
            D[d_new_name] = d  # 加入到种子模型池

            if api_mutation_type == 'seed_model' and execution_config["ifapimut"]:
                # try:
                api_mutation(d, log_dict, n, LOG_FLAG=True)  # 进行API变异
                # except Exception as e:
                #     print(f"Error during api_mutation: {e}")
            end_time = time.time()
            elapsed_time = end_time - start_time  # 生成1个模型的时间

            # 推理
            d = d.to(device)
            new_outputs = handle_format(d(data_selected))[0]  # torch.Size([5, 10])  torch.Size([10])
            # if isinstance(new_outputs, torch.Tensor):
            distance = metrics.ChebyshevDistance(original_outputs, new_outputs)  # np.max(np.abs(x - y))
                # 损失函数
            # loss_fun_ms, loss_fun_torch = get_loss('CrossEntropy')
            # loss_fun_ms, loss_fun_torch = loss_fun_ms(), loss_fun_torch().to(device)
            # loss_torch = loss_fun_torch(new_outputs, y_outputs)
            if seed_model == "openpose":
                optimizer_torch = seed_optimizer_pt(d.parameters(), lr=0.005)
                from models.openpose.src.loss_torch import openpose_loss, BuildTrainNetwork
                criterion = openpose_loss()
                train_net = BuildTrainNetwork(d, criterion).to(device)
                print(d(data_selected)[0],d(data_selected)[1], label_selected1.shape, label_selected2.shape, label_selected3.shape)
                # print(d(data_selected).shape)
                loss = criterion(d(data_selected)[0],d(data_selected)[1], label_selected1, label_selected2, label_selected3)
                loss = train_net(data_selected, label_selected1, label_selected2, label_selected3)

                loss.backward()
                optimizer_torch.step()
                # loss,grads = train_step_openpose(train_net_ms, seed_optimizer_fun, data_selected, label_selected1, label_selected2, label_selected3)
            elif seed_model == "ssimae":
                optimizer_torch = seed_optimizer_pt(params=d.parameters(), lr=0.0005, weight_decay=1.0e-5)
                from models.ssimae.src.network_torch import SSIMLoss,NetWithLoss
                loss = SSIMLoss()
                train_net_ms = NetWithLoss(d, loss).to(device)
                loss = train_net_ms(data_selected)
                loss.backward()
                optimizer_torch.step()
                # loss,grads = train_step_ssimae(train_net_ms, seed_optimizer_fun, data_selected)

            elif seed_model == "crnn":
                from models.CRNN.crnn_torch import CTCLoss_torch
                from models.CRNN.src.model_utils.config import config as crnnconfig
                optimizer_torch = seed_optimizer_pt(d.parameters(), lr=0.02, momentum=0.95, nesterov=True)

                crnnloss = CTCLoss_torch(max_sequence_length=crnnconfig.num_step,
                                max_label_length=crnnconfig.max_text_length,
                                batch_size=num_samples)
                print(d(data_selected).shape, label_selected.shape)
                d.train() 
                loss_torch_result = crnnloss(d(data_selected), label_selected)
                print(loss_torch_result)
                loss_torch_result.backward()
                optimizer_torch.step()
                d.eval() 
                # loss,grads = train_step_crnn(train_net_ms, seed_optimizer_fun, data_selected, label_selected)
            elif "SSD" in seed_model:
                optimizer_torch = seed_optimizer_pt(d.parameters(), lr=0.02)
                loss_fun_pt = loss_fun_pt.to(device)
                pred_loc_torch, pred_label_torch = d(data_selected)
                print(pred_loc_torch.shape, pred_label_torch.shape, label_selected1.shape, label_selected2.shape, label_selected3.shape)
                loss = loss_fun_pt(pred_loc_torch, pred_label_torch, label_selected1, label_selected2, label_selected3)
                loss.backward()
                optimizer_torch.step()# loss, grads = train_step_ssd(seed_model_net, loss_fun_pt, seed_optimizer_fun, data_selected, label_selected1, label_selected2, label_selected3)
            elif "yolo" in seed_model:
                input_shape = data_selected.shape[2:4]
                input_shape_ms = torch.Tensor(tuple(input_shape[::-1]), torch.float32)

                yolo_output = d(data_selected)

                loss = loss_fun_pt(yolo_output,label_selected1,
                                                label_selected2,
                                                label_selected3,
                                                label_selected4, label_selected5, label_selected6,input_shape_ms)
            else:
                optimizer_torch = seed_optimizer_pt(d.parameters(), lr=0.02)
                loss_fun_pt = loss_fun_pt
                pred = d(data_selected)
                print("1111")
                print("data_selected",data_selected.shape)
                print("pred:",pred.shape)
                if len(pred) != 1:
                    pred = pred[:1]
                # print(data_selected.shape)
                
                label_selected = label_selected.long()
                print(pred.shape, label_selected.shape)
                loss = loss_fun_pt(pred, label_selected)
                loss.backward()
                optimizer_torch.step()
            gpu_memory1, cpu_memory1 = compute_gpu_cpu()
            d = copy.deepcopy(D[log_dict[str(n)]['select_d_name']]) # 下一个种子模型
            gpu_memory2, cpu_memory2 = compute_gpu_cpu()
            optimizer_torch.zero_grad()
            gpu_memory_used = gpu_memory2 - gpu_memory1 # 本次蜕变生成的模型占用的资源
            cpu_memory_used = cpu_memory1 - cpu_memory2
            loss = 0
            metrics_dict[d_new_name] = [distance,elapsed_time,gpu_memory_used,cpu_memory_used,loss]

        else: # 对于crash的新模型，都取None
            print("Failed")
            d_new_name=log_dict[str(n)]['d_name']
            d = copy.deepcopy(D[log_dict[str(n)]['select_d_name']])  # 下一个种子模型
            metrics_dict[d_new_name] = ['None']*5


        # 保存指标
        df = pd.DataFrame([(k, v[0], v[1], v[2], v[3],v[4]) for k, v in metrics_dict.items()],
            columns=['New_Model_Name', 'MAE_Distance', 'Time', 'Gpu_Memory_Used', 'Cpu_Memory_Used','Loss'])
        save_path = os.path.join("results", seed_model, str(localtime),"METRICS_RESULTS_" + str(device).replace(':', '_') + ".xlsx")
        df.to_excel(save_path, index=False)

        # 假设 metrics_dict、seed_model、localtime、device 都已定义
        df = pd.DataFrame(
            [(k, v[0], v[1], v[2], v[3], v[4]) for k, v in metrics_dict.items()],
            columns=['New_Model_Name', 'MAE_Distance', 'Time', 'Gpu_Memory_Used', 'Cpu_Memory_Used', 'Loss']
        )

        # 构造保存路径，后缀改为 .csv
        save_path = os.path.join(
            "results",
            seed_model,
            str(localtime),
            "METRICS_RESULTS_" + str(device).replace(':', '_') + ".csv"
        )
        print(save_path)
        # 保存为 CSV
        df.to_csv(save_path, index=False)

    # 保存json:
    dict_save_path = os.path.join("results", seed_model, str(localtime),
                                  "TORCH_LOG_DICT_" + str(device).replace(':', '_') + ".json")
    os.makedirs(os.path.dirname(dict_save_path), exist_ok=True)
    with open(dict_save_path, 'w', encoding='utf-8') as file:
        json.dump(log_dict, file, ensure_ascii=False, indent=4)







