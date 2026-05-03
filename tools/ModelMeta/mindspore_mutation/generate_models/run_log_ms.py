import pandas as pd
import mindspore
import mindspore as ms
import collections
import uuid
import copy
import datetime
import json
import os
import platform
import random
import sys
import time
from mindspore import JitConfig
import mindspore.context as context
from mindspore import export, load_checkpoint, load_param_into_net
from mindspore.rewrite import ScopedValue, NodeType
from mindspore.rewrite.node import Node, NodeManager
from numpy import ndarray
from openpyxl import Workbook
import mindspore.numpy as mnp
from mindspore import Tensor
from infoplus.MindSporeInfoPlus import mindsporeinfoplus
import torch
import torch.optim as optim
from mindspore import Tensor
from mindspore.rewrite import SymbolTree
# from mindspore_gl import GraphField
import pickle
from mindspore_mutation.cargo import *
import torch.distributions as dist
import copy
import time
import json
import torch.fx as fx
from mindspore_mutation.MR_structure import *
from mindspore_mutation.cargo import match_rule,reflect_name,MCMC,compute_gpu_cpu
from mindspore_mutation.api_mutation import api_mutation
from mindspore_mutation.calculate_coverage import model2cov,find_layer_type
from mindspore_mutation.cargo import select_places,max_seed_model_api_times
import psutil
import sys
from openpyxl import Workbook
import os
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
from mindspore import save_checkpoint
import torch_mutation.config as pt_config
import mindspore_mutation.config as ms_config
# from mindspore_mutation.calculate_coverage import model2cov
from mindspore_mutation.handel_shape import handle_format
from mindspore_mutation import metrics
import gc
# from memory_profiler import profile

ms_device=ms_config.ms_device
pt_device=ms_config.pt_device

MR_structure_name_list = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']
MR_structures_map = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B} # to do
nlp_cargo = ["LSTM","FastText", "TextCNN", "SentimentNet", "GPT"]
# MR_structure_name_list = ['PIOC']
# MR_structures_map = {"PIOC": PIOC} # to do

deadcode_name_list=['Dense', 'SELayer', 'DenseLayer', 'Inception_A', 'PWDWPW_ResidualBlock', 'ResidualBlock', 'DropPath']
# deadcode_name_list=['MyConvPoolLayerSameShape', 'MyAdaptiveMaxPoolLayer', 'MyTransposeConvLayer']


# @profile
def run_log_ms(seed_model, mutate_times, num_samples, mr_index, ifapimut, log_path,device,train_config):
    # for mr_index in range(0, 4):
    if device == -1:
        pt_device = "cpu"
        ms_device = "cpu"
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
    else:
        pt_device = "cuda:"+str(device)
        ms_device = "gpu"
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU", device_id=device)

    localtime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    log_dict = {}
    dtypes = [mindspore.float32] if seed_model not in nlp_cargo else [mindspore.int32]

    with open(log_path, "r") as json_file:
        log_dict = json.load(json_file)

    # 数值数据
    if isinstance(datasets_path_cargo[seed_model],list):
        data_0 = np.load(datasets_path_cargo[seed_model][0])
        data_1 = np.load(datasets_path_cargo[seed_model][1])
        samples_0 = np.random.choice(data_0.shape[0], num_samples, replace=False)
        samples_data_0 = data_0[samples_0] # 随机选择num_samples个数据 (num_samples, 3, 32, 32)   

        samples_1 = np.random.choice(data_1.shape[0], num_samples, replace=False)
        samples_data_1 = data_1[samples_1] # 随机选择num_samples个数据 (num_samples, 3, 32, 32)   
        data_selected_0 = Tensor(samples_data_0, dtype=mstype.float32 if seed_model in nlp_cargo else mstype.int32)
        data_selected_1 = Tensor(samples_data_1, dtype=mstype.float32 if seed_model in nlp_cargo else mstype.int32)
        data_selected = (data_selected_0, data_selected_1)
        data_npy = [data_selected_0.asnumpy(), data_selected_1.asnumpy()]

        npy_path = os.path.join("results", seed_model, str(localtime), 'data0_npy.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, data_npy[0])
        npy_path = os.path.join("results", seed_model, str(localtime), 'data1_npy.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, data_npy[1])
    else:
        data = np.load(datasets_path_cargo[seed_model])
        samples = np.random.choice(data.shape[0], num_samples, replace=False)
        samples_data = data[samples] # 随机选择num_samples个数据 (num_samples, 3, 32, 32)
        data_selected = Tensor(samples_data, dtype=mstype.int32 if seed_model in nlp_cargo else mstype.float32)
        data_npy = data_selected.asnumpy()


        npy_path = os.path.join("results", seed_model, str(localtime), 'data0_npy.npy')
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
                dt = mstype.float32
            elif "SSD" in seed_model:
                dt = mstype.int32

            label_selected1 = Tensor(samples_label1, dtype=mstype.float32)
            label_selected2 = Tensor(samples_label2, dtype=dt)
            label_selected3 = Tensor(samples_label3, dtype=dt)
            np.save(labels_path1, samples_label1)
            np.save(labels_path2, samples_label2)
            np.save(labels_path3, samples_label3)


        elif seed_model in ["patchcore", "ssimae"]:
            # data_selected = Tensor(samples_data, dtype=mstype.float64)
            pass
        elif seed_model in ["crnn", "DeepLabV3","TextCNN", "resnet"]:
            labels_path = os.path.join("results", seed_model, str(localtime), 'data1_npy_y.npy')
            # print(labels_path_cargo[seed_model])
            targets = np.load(labels_path_cargo[seed_model])
            samples_label = targets[samples]
            label_selected = Tensor(samples_label, dtype=mstype.int32)
            np.save(labels_path, label_selected)

        elif seed_model in ["yolov3", "yolov4"]:

            labels_path1 = os.path.join("results", seed_model, str(localtime), 'data1_npy_y.npy')
            labels_path2 = os.path.join("results", seed_model, str(localtime), 'data2_npy_y.npy')
            labels_path3 = os.path.join("results", seed_model, str(localtime), 'data3_npy_y.npy')
            labels_path4 = os.path.join("results", seed_model, str(localtime), 'data4_npy_y.npy')
            labels_path5 = os.path.join("results", seed_model, str(localtime), 'data5_npy_y.npy')
            labels_path6 = os.path.join("results", seed_model, str(localtime), 'data6_npy_y.npy')

            targets1, targets2, targets3, targets4, targets5, targets6 = labels_path_cargo[seed_model]
            samples_label1 = np.load(targets1)[samples]
            samples_label2 = np.load(targets2)[samples]
            samples_label3 = np.load(targets3)[samples]
            samples_label4 = np.load(targets4)[samples]
            samples_label5 = np.load(targets5)[samples]
            samples_label6 = np.load(targets6)[samples]

            # if seed_model == "openpose":
            dt = mstype.float32
            # elif "SSD" in seed_model:
            #     dt = mstype.int32

            label_selected1 = Tensor(samples_label1, dtype=mstype.float32)
            label_selected2 = Tensor(samples_label2, dtype=dt)
            label_selected3 = Tensor(samples_label3, dtype=dt)
            label_selected4 = Tensor(samples_label4, dtype=mstype.float32)
            label_selected5 = Tensor(samples_label5, dtype=dt)
            label_selected6 = Tensor(samples_label6, dtype=dt)
            np.save(labels_path1, samples_label1)
            np.save(labels_path2, samples_label2)
            np.save(labels_path3, samples_label3)
            np.save(labels_path4, samples_label4)
            np.save(labels_path5, samples_label5)
            np.save(labels_path6, samples_label6)
        else:
            labels_path = os.path.join("results", seed_model, str(localtime), 'data1_npy_y.npy')
            # print(labels_path_cargo[seed_model])
            targets = np.load(labels_path_cargo[seed_model])
            samples_label = targets[samples]
            label_selected = Tensor(samples_label, dtype=mstype.float32)
            np.save(labels_path, label_selected)


    seed_model_net = get_model(seed_model)

    if seed_model in ["ssimae"]:
        pass
    elif seed_model =="patchcore":
        from models.PatchCore.src.oneStep import OneStepCell
        seed_model_net = OneStepCell(seed_model_net)

    else:
        loss_fun_ms, _ = get_loss(train_config['loss_name'])
        loss_fun_ms = loss_fun_ms()

    seed_optimizer = train_config['opt_name']
    seed_optimizer_ms, _ = get_optimizer(seed_optimizer)

    new_net = copy.deepcopy(seed_model_net)
    stree = SymbolTree.create(new_net)
    metrics_dict = dict()
    option_layers = []
    for name, child in new_net.cells_and_names():
        if not has_child_node(new_net, name) and not name == '' and not 'deadcode' in str(type(child)):
            if name.split("_")[0] not in MR_structure_name_list:
                option_layers.append((name, child, name, type(child)))
    original_outputs = handle_format(seed_model_net(data_selected))
    new_outputs = original_outputs
    select_d_name = seed_model
    D = {seed_model: stree} # 所有模型
    O = {seed_model: original_outputs} # 所有模型
    N = {seed_model: seed_model_net} # 所有模型
    R = {0:[0.0001, seed_model]} # 所有变异成功的模型。{序号:[reward,model_name]}
    MR_structure_selected_nums = {k: 0 for k in MR_structure_name_list}  # 字典，每种MR_structure已被用几次。用于命名新模型
    seed_model_api_times=0 # api级别的变异几次是种子模型

    option_index = []
    # 以追加模式打开文件
    with open('/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/results/TextCNN/example.txt', 'a', encoding='utf-8') as file:
        file.write('text_to_append' + "\n")
    file.close()

    for n in range(mutate_times):
        print('-----------------------total_Mutate_time:%d start!-----------------------' % n)
        start_time=time.time()
        # try:
        if "Success" in log_dict[str(n)]['state']: # 对于变异成功的模型：正常生成变异模型，正常检测
            log_dict[n] = {}
            log_dict[n]['d_name'] = select_d_name
            old_d_name=select_d_name # 用于训练Q

            # 选择死代码
            selected_deadcode_name =log_dict[str(n)]['select_deadcode']

            # 选择 MR_structure
            selected_MR_structure_name = log_dict[str(n)]['selected_MR_structure']

            # 命名新模型：种子模型、MR_structure、第几次用这个MR_structure
            d_new_name = log_dict[str(n)]['d_new_name']

            # 算子变异对死代码还是原模型？
            api_mutation_type = log_dict[str(n)]['api_mutation_type(seed_model or deadcode)']
            

            # # 选择插入位置
            nodedict = collections.OrderedDict()  # 特殊字典，保持插入的顺序
            hash_table = collections.defaultdict(int)  # 每次访问一个不存在的键时，该键会自动被赋值为整数 0
            flag, nodedict = scan_node(stree, hash_table, nodedict)
            node_list=list(nodedict.values())
            node_list = []
            for k, v in nodedict.items():
                node_list.append(k)
            length=len(nodedict)
            print('length', length)
            print("mutate_type:", selected_MR_structure_name, ";  op_type:", selected_deadcode_name, ';  api_mutation_type:', api_mutation_type, flush=True)

            # 选择插入位置
            subs_place, dep_places = log_dict[str(n)]['subs_place'], log_dict[str(n)]['dep_places']
            a = node_list[dep_places[-1]]
            b = node_list[dep_places[-2]]
            c = node_list[dep_places[-3]]
            d = node_list[dep_places[-4]]
            aa = mindspore.rewrite.api.node.Node(a)
            bb = mindspore.rewrite.api.node.Node(b)
            cc = mindspore.rewrite.api.node.Node(c)
            dd = mindspore.rewrite.api.node.Node(d)
            add_module = MR_structures_map[selected_MR_structure_name](selected_deadcode_name, api_mutation_type, log_dict, n, LOG_FLAG=True)

            seat = 0
            if selected_MR_structure_name == "PIOC" and selected_deadcode_name in ["Dense", "Conv", "SELayer", "DenseLayer", "Inception_A",
                                                "PWDWPW_ResidualBlock", "ResidualBlock", "DropPath"]:

                tree = cc.get_symbol_tree()  # 获取节点 c 所属的符号树（SymbolTree）

                position = tree.after(cc)  # 节点 c 在符号树中的后继位置
                next_node = cc.get_users()[0]  # 下一个使用 c 输出的节点
                if len(next_node.get_args()) > 1:
                    for idx, arg in enumerate(next_node.get_args()):
                        if arg == cc.get_targets()[0]:
                            seat = idx  # 匹配位置
                            break
                        # print("arg != c.get_target()[0]", arg, c.get_targets()[0], flush=True)
                new_node = mindspore.rewrite.api.node.Node.create_call_cell(add_module,
                                                                            # targets=[str(uuid.uuid4())],
                                                                            targets=[stree.unique_name("x")],
                                                                            name="{}_{}".format(selected_MR_structure_name,MR_structure_selected_nums[selected_MR_structure_name]),  # 给新节点命名
                                                                            args=ScopedValue.create_name_values(["aa", "bb", "cc"]))
                new_node.set_arg_by_node(0, aa) # 设置新节点的输入参数
                new_node.set_arg_by_node(1, bb)
                new_node.set_arg_by_node(2, cc)
            else:  # selected_MR_structure_name != "PIOC"
                tree = dd.get_symbol_tree()
                position = tree.after(dd)
                next_node = dd.get_users()[0]
                if len(next_node.get_args()) > 1:
                    for idx, arg in enumerate(next_node.get_args()):
                        if arg == dd.get_targets()[0]:
                            seat = idx
                            break
                new_node = mindspore.rewrite.api.node.Node.create_call_cell(add_module,
                                                                            # targets=[str(uuid.uuid4())],
                                                                            targets=[stree.unique_name("x")],
                                                                            name="{}_{}".format(selected_MR_structure_name,MR_structure_selected_nums[selected_MR_structure_name]),
                                                                            args=ScopedValue.create_name_values(["aa", "bb", "cc","dd"]))
                new_node.set_arg_by_node(0, aa)
                new_node.set_arg_by_node(1, bb)
                if selected_MR_structure_name == "UOC":
                    new_node.set_arg_by_node(2, cc)
                    # print("c.get_symbol_tree", c.get_symbol_tree())
                    new_node.set_arg_by_node(3, dd)
                    # print("d.get_symbol_tree", d.get_symbol_tree())
                else:
                    new_node.set_arg_by_node(2, dd)
                    new_node.set_arg_by_node(3, cc)
            tree.insert(position, new_node)
            next_node.set_arg_by_node(seat, new_node)
            new_net = stree.get_network()
            new_outputs = new_net(data_selected) # 出错处
            D[d_new_name] = stree  # 加入到种子模型池
            # # print(log_dict[n]['state'])
            # new_net = stree.get_network()
            # new_outputs = new_net(data_selected) # 出错处
            # new_outputs = new_net(data_selected) # 出错处
            new_output = handle_format(new_outputs)
            N[d_new_name] = new_net  # 加入到种子模型池
            # new_outputs = new_net(data_selected) # 出错处
            new_output = handle_format(new_outputs)
            O[d_new_name] = copy.deepcopy(new_output)

            print('ChebyshevDistance:',metrics.ChebyshevDistance(original_outputs,new_output),';  MAEDistance:',metrics.MAEDistance(original_outputs,new_output))
            dist_chess = metrics.ChebyshevDistance(original_outputs,new_output)
            gpu_memory2, cpu_memory2 = compute_gpu_cpu()
            metrics_dict[d_new_name] = [dist_chess,gpu_memory2,cpu_memory2]
            if new_output.shape!=original_outputs.shape:
                print('new_output.shape!=original_outputs.shape!')


            if api_mutation_type == 'seed_model' and ifapimut and "Failed" not in log_dict[n]['state']:
                try:
                    new_net, stree, log_dict,option_index = api_mutation(new_net, option_layers, option_index, log_dict, n, LOG_FLAG=False) # 进行API变异 to do
                    print(f"Success during api_mutation")
                except Exception as e:
                    print(f"Error during api_mutation: {e}")
                    log_dict[n]['state'] = f"Failed: api_mutation failed: {str(e)}"

            if "Failed" not in log_dict[n]['state']:
                print('ChebyshevDistance:',metrics.ChebyshevDistance(original_outputs,new_output),';  MAEDistance:',metrics.MAEDistance(original_outputs,new_output))
                dist_chess = metrics.ChebyshevDistance(original_outputs,new_output)
                gpu_memory2, cpu_memory2 = compute_gpu_cpu()

                if True:
                    if seed_model == "openpose":
                        from models.openpose.src.loss import BuildTrainNetwork as BuildTrainNetwork_ms
                        from models.openpose.src.loss import openpose_loss as openpose_loss_ms
                        criterion = openpose_loss_ms()
                        train_net_ms = BuildTrainNetwork_ms(new_net, criterion)
                        loss = train_net_ms(data_selected, label_selected1, label_selected2, label_selected3)
                        # loss,grads = train_step_openpose(train_net_ms, seed_optimizer_fun, data_selected, label_selected1, label_selected2, label_selected3)
                    elif seed_model == "crnn":
                        from mindspore.nn.wrap import WithLossCell
                        from models.CRNN.src.loss import CTCLoss
                        from models.CRNN.src.model_utils.config import config as crnnconfig

                        crnnloss = CTCLoss(max_sequence_length=crnnconfig.num_step,
                                        max_label_length=crnnconfig.max_text_length,
                                        batch_size=num_samples)
                        train_net_ms = WithLossCell(new_net, crnnloss)
                        loss = train_net_ms(data_selected, label_selected)
                        # loss,grads = train_step_crnn(train_net_ms, seed_optimizer_fun, data_selected, label_selected)
                    elif "SSD" in seed_model:
                        print(loss_fun_ms)
                        pred_loc_ms, pred_label_ms = new_net(data_selected)
                        loss = loss_fun_ms(pred_loc_ms, pred_label_ms, label_selected1, label_selected2, label_selected3)
                        # loss, grads = train_step_ssd(seed_model_net, loss_fun_ms, seed_optimizer_fun, data_selected, label_selected1, label_selected2, label_selected3)

                    else:
                        pred = new_net(data_selected)
                        loss = loss_fun_ms(pred, label_selected)
                    
            end_time = time.time()
            elapsed_time = end_time - start_time  # 生成1个模型的时间\
            metrics_dict[d_new_name] = []
            if "Failed" not in log_dict[n]['state']:
                # metrics_dict[d_new_name].append(elapsed_time)
                metrics_dict[d_new_name].append(dist_chess)
                metrics_dict[d_new_name].append(gpu_memory2)
                metrics_dict[d_new_name].append(cpu_memory2)
            else:
                metrics_dict[d_new_name] = ["None"]*7

        df = pd.DataFrame([(index, v[0], v[1], v[2]) for index, (k, v) in enumerate(metrics_dict.items())],
            columns=['name', 'Distance', 'Gpu_Memory_Used', 'Cpu_Memory_Used'])
        save_path = os.path.join("results", seed_model, str(localtime),str(ms_device).replace(':', '_') + "_" + train_config['seed_model'] + ".csv")
        df.to_csv(save_path, index=False)

        # 保存json:
        dict_save_path = os.path.join("results", seed_model, str(localtime),"TORCH_LOG_DICT_" + str(ms_device).replace(':', '_') + ".json")
        os.makedirs(os.path.dirname(dict_save_path), exist_ok=True)
        with open(dict_save_path, 'w', encoding='utf-8') as file:
            json.dump(log_dict, file, ensure_ascii=False, indent=4)

    for index, (name, new_net) in enumerate(N.items()):
        print(index)
        if index == 0:
            continue
        try:
            new_net, stree, log_dict,option_index = api_mutation(new_net, option_layers, option_index, log_dict, n, LOG_FLAG=False) # 进行API变异 to do
            print(f"Success during api_mutation")
        except Exception as e:
            print(f"Error during api_mutation: {e}")
            log_dict[n]['state'] = f"Failed: api_mutation failed: {str(e)}"
            # continue
        option_layers = []
        
        for name, child in new_net.cells_and_names():
            if not has_child_node(new_net, name) and not name == '' and not 'deadcode' in str(type(child)):
                if name.split("_")[0] not in MR_structure_name_list:
                    option_layers.append((name, child, name, type(child)))


        

        

