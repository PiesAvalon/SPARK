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
from mindspore_mutation.calculate_coverage import model2cov,find_layer_type,json2cov
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
from mindspore_mutation.union import union_json
# from mindspore_mutation.calculate_coverage import model2cov
from mindspore_mutation.handel_shape import handle_format
from mindspore_mutation import metrics
import gc
# from memory_profiler import profile

ms_device=ms_config.ms_device
pt_device=ms_config.pt_device

# MR_structure_name_list = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']
# MR_structures_map = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B} # to do
# nlp_cargo = ["LSTM","FastText", "TextCNN", "SentimentNet", "GPT"]
MR_structure_name_list_ori = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']
MR_structures_map_ori = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B} # to do
nlp_cargo = ["LSTM","FastText", "TextCNN", "SentimentNet", "GPT"]

deadcode_name_list=['Dense', 'SELayer', 'DenseLayer', 'Inception_A', 'PWDWPW_ResidualBlock', 'ResidualBlock', 'DropPath']
# deadcode_name_list=['MyConvPoolLayerSameShape', 'MyAdaptiveMaxPoolLayer', 'MyTransposeConvLayer']


# @profile
def run_mcmc_ms(seed_model, mutate_times,num_samples,mr_index, ifapimut,ifTompson,device,train_config):
    
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
    
    
    
    MCMC_selector = MCMC()
    last_MR_structure_name = None
    last_reward = 0  # 用于更新MCMC
    log_dict = {}

    MR_structure_name_list = [MR_structure_name_list_ori[i] for i in mr_index]
    print(MR_structure_name_list)
    valid_keys = [key for key in MR_structure_name_list if key in MR_structure_name_list]
    MR_structures_map = {key: MR_structures_map_ori[key] for key in valid_keys}
    print(MR_structures_map)

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


        elif seed_model in [ "ssimae"]:
            # data_selected = Tensor(samples_data, dtype=mstype.float64)
            pass
        elif seed_model in ["crnn", "DeepLabV3","TextCNN", "resnet", 'vit']:
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

    option_layers = []
    for name, child in new_net.cells_and_names():
        if not has_child_node(new_net, name) and not name == '' and not 'deadcode' in str(type(child)):
            if name.split("_")[0] not in MR_structure_name_list:
                option_layers.append((name, child, name, type(child)))

    # input_data = mindsporeinfoplus.np_2_tensor([data_npy], dtypes)
    # print(data_selected.shape)
    original_outputs = handle_format(new_net(data_selected))
    new_outputs = original_outputs

    metrics_dict = dict()
    select_d_name = seed_model
    D = {seed_model: stree} # 所有模型
    O = {seed_model: original_outputs} # 所有模型
    N = {seed_model: seed_model_net} # 所有模型
    R = {0:[0.0001, seed_model]} # 所有变异成功的模型。{序号:[reward,model_name]}
    MR_structure_selected_nums = {k: 0 for k in MR_structure_name_list}  # 字典，每种MR_structure已被用几次。用于命名新模型
    seed_model_api_times=0 # api级别的变异几次是种子模型

    # 要写入的文本内容
    option_index = []
    # 循环变异：
    for n in range(mutate_times):
        print('-----------------------total_Mutate_time:%d start!-----------------------' % n)
        start_time=time.time()
        try:
            log_dict[n] = {}
            log_dict[n]['d_name'] = select_d_name
            log_dict[n]['state'] = "None"
            # 选择死代码
            selected_deadcode_name=random.choice(deadcode_name_list)

            # 选择 MR_structure
            selected_MR_structure_name = MCMC_selector.choose_mutator(last_MR_structure_name)
            selected_MR_structure = MCMC_selector.mutators[selected_MR_structure_name]
            selected_MR_structure.total += 1
            last_MR_structure_name = selected_MR_structure_name
            MR_structure_selected_nums[selected_MR_structure_name] += 1


            # 命名新模型：种子模型、MR_structure、第几次用这个MR_structure
            d_new_name = "{}-{}{}".format(seed_model, selected_MR_structure_name,MR_structure_selected_nums[selected_MR_structure_name])
            log_dict[n]['d_new_name'] = d_new_name

            # 算子变异对死代码还是原模型？ to do改列表
            if selected_deadcode_name in ('DropPath', 'Dense') and seed_model_api_times < max_seed_model_api_times(seed_model):  # 如果选择的死代码中无可变异算子，且原模型中有算子可被变异
                api_mutation_type = 'seed_model'
                seed_model_api_times+=1
            elif selected_deadcode_name not in ('DropPath', 'Dense') and seed_model_api_times < max_seed_model_api_times(seed_model): # 如果选择的死代码中有可变异算子，且原模型中有算子可被变异
                api_mutation_type = random.choice(['seed_model', 'deadcode'])  # 随机选择是对原模型还是死代码变异
                if api_mutation_type=='seed_model':
                    seed_model_api_times += 1
            elif selected_deadcode_name not in ('DropPath', 'Dense') and seed_model_api_times >= max_seed_model_api_times(seed_model):  # 如果选择的死代码中有可变异算子，且原模型中无算子可被变异
                api_mutation_type = 'deadcode'
            else: # 种子模型没有变异的算子了，且选择的死代码中也没有变异的算子
                api_mutation_type = 'None'
                log_dict[n]['state'] = "Success:But no APIs available for mutation, so no API-level mutation was performed."
            # api_mutation_type = 'None' #用于Q4消融实验



            # 选择插入位置
            nodedict = collections.OrderedDict()  # 特殊字典，保持插入的顺序
            hash_table = collections.defaultdict(int)  # 每次访问一个不存在的键时，该键会自动被赋值为整数 0
            flag, nodedict = scan_node(stree, hash_table, nodedict)
            
            length=len(nodedict)
            print('length', length)
            print("mutate_type:", selected_MR_structure_name, ";  op_type:", selected_deadcode_name, ';  api_mutation_type:', api_mutation_type, flush=True)
            sys.setrecursionlimit(4000)

            def select_node(nodedict, recurive_depth=0):
                if recurive_depth >= 3500:
                    return None, None, None, None, recurive_depth
                subs_place, dep_places = \
                    select_places(range(0, length - 1), 5)
                if dep_places is None:
                    return select_node(nodedict, recurive_depth + 1)
                dep_places.sort(reverse=True)
                node_list = []
                for k, v in nodedict.items():
                    node_list.append(k)
                a = node_list[dep_places[-1]]
                b = node_list[dep_places[-2]]
                c = node_list[dep_places[-3]]
                d = node_list[dep_places[-4]]
                a = mindspore.rewrite.api.node.Node(a)
                b = mindspore.rewrite.api.node.Node(b)
                c = mindspore.rewrite.api.node.Node(c)
                d = mindspore.rewrite.api.node.Node(d)
                if not a._node.get_belong_symbol_tree() == b._node.get_belong_symbol_tree() == c._node.get_belong_symbol_tree() == d._node.get_belong_symbol_tree():
                    return select_node(nodedict, recurive_depth + 1)
                elif not (check_node(d) and check_node(c) and check_node(b) and check_node(a)):
                    return select_node(nodedict, recurive_depth + 1)
                elif selected_MR_structure_name == "PIOC" and (c.get_users()[0].get_node_type() == NodeType.Output or c.get_users()[0].get_node_type() == NodeType.Tree):
                    return select_node(nodedict, recurive_depth + 1)
                elif d.get_users()[0].get_node_type() == NodeType.Output or d.get_users()[0].get_node_type() == NodeType.Tree :
                    return select_node(nodedict, recurive_depth + 1)
                else:
                    log_dict[n]['subs_place'], log_dict[n]['dep_places'] = subs_place, dep_places
                    return a, b, c, d, recurive_depth

            # start_time = time.time()
            aa, bb, cc, dd, recurive_depth = select_node(nodedict, 0)
            # end_time = time.time()
            # print("recurive depth:", recurive_depth, "select_node time:", end_time - start_time)
            if recurive_depth>=3500: # 没找到合适的插入位置
                log_dict[n]['state'] = f"Failed:Cannot find suitable places！"
            if aa is None:
                print("mutate Failed for Cannot find suitable places")
                # continue
            # print("~~~~~~~~~~~~~~~~~选择对%s中的算子进行api变异！~~~~~~~~~~~~~~~" % api_mutation_type)
            # add_module = MR_structures_map[selected_MR_structure_name](selected_deadcode_name, api_mutation_type, log_dict, n, LOG_FLAG=False)
            
            add_module = MR_structures_map[selected_MR_structure_name](selected_deadcode_name, api_mutation_type, log_dict, n, LOG_FLAG=False)

            seat = 0
            if "Failed" not in log_dict[n]['state']:
                if selected_MR_structure_name == "PIOC" and selected_deadcode_name in ["Dense", "Conv", "SELayer", "DenseLayer", "Inception_A",
                                                    "PWDWPW_ResidualBlock", "ResidualBlock", "DropPath"]:
                    
                    if not (check_node(cc) and check_node(bb) and check_node(aa)):
                        log_dict[n]['state'] = "Failed:选择插入的节点位置不正确！"
                        print("Failed:选择插入的节点位置不正确！")
                        # continue

                    tree = cc.get_symbol_tree()  # 获取节点 c 所属的符号树（SymbolTree）

                    position = tree.after(cc)  # 节点 c 在符号树中的后继位置
                    next_node = cc.get_users()[0]  # 下一个使用 c 输出的节点
                    # print("next_node.get_name()", next_node.get_name())
                    # print("c.get_name()", cc.get_name())
                    # if next_node.get_node_type() == NodeType.Output or next_node.get_node_type() == NodeType.Tree or next_node.get_node_type() == NodeType.CallCell:
                    #     print("mutate failed for next_node.get_node_type() ==", next_node.get_node_type(), "name",next_node.get_name(), flush=True)
                    #     continue
                        # log_dict[n]['state'] = "Failed:选择插入的节点位置不正确！"
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

                    if not (check_node(dd) and check_node(cc) and check_node(bb) and check_node(aa)):
                        log_dict[n]['state'] = "Failed:选择插入的节点位置不正确！"
                        print("Failed:选择插入的节点位置不正确！")
                        # continue
                    tree = dd.get_symbol_tree()
                    position = tree.after(dd)
                    next_node = dd.get_users()[0]
                    # print("next_node.get_name()", next_node.get_name())
                    # print("d.get_name()", dd.get_name())
                    # if  next_node.get_node_type() == NodeType.Output or next_node.get_node_type() == NodeType.Tree or next_node.get_node_type() == NodeType.CallCell:
                    #     print("mutate failed for next_node.get_node_type() ==", next_node.get_node_type(), "name",next_node.get_name(), flush=True)
                    #     continue
                        # log_dict[n]['state'] = "Failed:选择插入的节点位置不正确！"
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
            D[d_new_name] = stree  # 加入到种子模型池
            # print(log_dict[n]['state'])
            new_net = stree.get_network()

            if api_mutation_type == 'seed_model' and ifapimut and "Failed" not in log_dict[n]['state']:
                try:
                    new_net, stree, log_dict,option_index = api_mutation(new_net, option_layers, option_index, log_dict, n, LOG_FLAG=False) # 进行API变异 to do
                    print(f"Success during api_mutation")
                except Exception as e:
                    print(f"Error during api_mutation: {e}")
                    log_dict[n]['state'] = f"Failed: api_mutation failed: {str(e)}"

            N[d_new_name] = new_net  # 加入到种子模型池
            new_outputs = new_net(data_selected) # 出错处
            # print(len(new_outputs))
            # print(new_outputs[0].shape)
            
            new_output = handle_format(new_outputs)
            # print(new_output.shape)
            # print(type(new_output))
            # print(type(original_outputs))
            O[d_new_name] = copy.deepcopy(new_output)
            print('ChebyshevDistance:',metrics.ChebyshevDistance(original_outputs,new_output),';  MAEDistance:',metrics.MAEDistance(original_outputs,new_output))
            dist_chess = metrics.ChebyshevDistance(original_outputs,new_output)
            if new_output.shape!=original_outputs.shape:
                print('new_output.shape!=original_outputs.shape!')
                # sys.exit('new_output.shape!=original_outputs.shape!')

            if "Failed" not in log_dict[n]['state']:
                print('ChebyshevDistance:',metrics.ChebyshevDistance(original_outputs,new_output),';  MAEDistance:',metrics.MAEDistance(original_outputs,new_output))
                dist_chess = metrics.ChebyshevDistance(original_outputs,new_output)
                gpu_memory2, cpu_memory2 = compute_gpu_cpu()

                if True:
                    if seed_model == "ssimae":
                        from models.ssimae.src.network import SSIMLoss, AutoEncoder, NetWithLoss
                        loss = SSIMLoss()
                        train_net_ms = NetWithLoss(new_net, loss)
                        loss = train_net_ms(data_selected)

                    elif seed_model == "openpose":
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
                    elif "yolo" in seed_model:
                        input_shape = data_selected.shape[2:4]
                        input_shape_ms = Tensor(tuple(input_shape[::-1]), ms.float32)

                        yolo_output = new_net(data_selected)

                        loss = loss_fun_ms(yolo_output,label_selected1,
                                                        label_selected2,
                                                        label_selected3,
                                                        label_selected4, label_selected5, label_selected6,input_shape_ms)
                        # loss = train_step_yolo(seed_model_net,
                        #                                 loss_fun_ms,
                        #                                 seed_optimizer_fun,
                        #                                 data_selected,
                        #                                 label_selected1,
                        #                                 label_selected2,
                        #                                 label_selected3,
                        #                                 label_selected4, label_selected5, label_selected6,input_shape_ms)


                    else:
                        pred = new_net(data_selected)
                        loss = loss_fun_ms(pred, label_selected)

                if new_output.shape != original_outputs.shape:
                    print('new_output.shape!=original_outputs.shape!')



            end_time = time.time()
            elapsed_time = end_time - start_time  # 生成1个模型的时间\
            metrics_dict[d_new_name] = []
            if "Failed" not in log_dict[n]['state']:
                metrics_dict[d_new_name].append(elapsed_time)

                metrics_dict[d_new_name].append(gpu_memory2)
                metrics_dict[d_new_name].append(cpu_memory2)
            else:
                metrics_dict[d_new_name] = ["None"]*7

        except Exception as e:
            print(e)
            log_dict[n]['state'] = f"Failed: Error during mutation: {str(e)}"

        start_time = time.time()
        # 根据crash or not选择下一个种子模型
        with torch.no_grad():
            if ('state' in log_dict[n]) and ("Failed" in log_dict[n]['state']):  # 在上述过程中变异失败了
              
                reward,done=-1,True
                # Thompson sampling strategy选择1个模型作为下次变异的种子模型
                # d_probs = torch.distributions.Beta(torch.tensor([value[0] for value in R.values()]), torch.ones(len(R))).sample()
                # select_d_name = R[torch.argmax(d_probs).item()][1]
                if ifTompson:
                    # Thompson sampling strategy选择1个模型作为下次变异的种子模型
                    d_probs = torch.distributions.Beta(torch.tensor([value[0] for value in R.values()]), torch.ones(len(R))).sample()
                    select_d_name = R[torch.argmax(d_probs).item()][1]
                else:
                    select_d_name = random.choice(list(R.items()))[1][1]

                
                
                # last_net = D[select_d_name].get_network()
                # stree = SymbolTree.create(N[select_d_name])
                stree = D[select_d_name]
                # stree = pickle.loads(pickle.dumps(D[select_d_name]))
                metrics_dict[d_new_name]=["None"]*4
                # next_net = last_net
                # next_output = stree.get_network()(data_selected)
                next_output = O[select_d_name]
                # next_output = N[select_d_name](data_selected)
                # formatted_data = handle_format(next_output)
                # formatted_data = handle_format(next_output)[0].unsqueeze(0)
                # formatted_data = handle_format(next_output)
                formatted_data = O[select_d_name]
                #$formatted_data = handle_format(next_output).unsqueeze(0)
                
            else:
                done = False
                json_file_path= os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results", seed_model, str(localtime),"model_json" , str(n) + ".json")
                os.makedirs(os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results", seed_model, str(localtime),"model_json"), exist_ok=True)
                all_json_path="/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/ms_all_layer_info.json"
                api_config_pool_path = '/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/mindspore_api_config_pool.json'
                folder_path = '/home/cvgroup/myz/czx/semtest-gitee/'
                # input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov = 1,1,1,1,1,1 #to do
                input_data = mindsporeinfoplus.np_2_tensor([data_npy], dtypes)
                # next_net = copy.deepcopy(new_net)
                stree = stree # SymbolTree.create(new_net)  # 
                input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside, output_datas = model2cov(new_net,
                                                                                                        input_data,
                                                                                                        dtypes,
                                                                                                        json_file_path,
                                                                                                        all_json_path,
                                                                                                        api_config_pool_path,
                                                                                                        folder_path)
                union_json_path = os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results", seed_model, str(localtime),"union.json")
                json_folder_path = os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results", seed_model, str(localtime),"model_json")
                for root, dirs, files in os.walk(json_folder_path):
                    for file in files:
                        if file.endswith('.json'):
                            file_path = os.path.join(root, file)
                            # print(file_path)
                            union_json(file_path, union_json_path)

                input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_covs = json2cov(union_json_path,
                                                                                            all_json_path,
                                                                                        api_config_pool_path
                                                                                        )
                reward= (input_cov + config_cov + api_cov)/3
                metrics_dict[d_new_name] = [input_cov,config_cov, api_cov, reward]
                metrics_dict[d_new_name].append(elapsed_time)

                metrics_dict[d_new_name].append(gpu_memory2)
                metrics_dict[d_new_name].append(cpu_memory2)
                metrics_dict[d_new_name].append(loss)
                metrics_dict[d_new_name].append(dist_chess)
                
                R[len(R)]=[reward,d_new_name] # 新模型的reward,且只有合法模型在R中，但所有模型在D中
                select_d_name=d_new_name
                # next_output = output_datas
                # new_net = stree.get_network()
                next_output = O[d_new_name]# new_net(data_selected)
                # formatted_data = handle_format(next_output)[0].unsqueeze(0)
                formatted_data = O[d_new_name]
                # print(formatted_data.shape)
        ## with torch.no_grad()结束
        # 更新Q
        # Quantum_Q_value = Quantum_Q(handle_format(D[old_d_name](data_selected))[0].unsqueeze(0))[0, selected_MR_structure_idx].unsqueeze(0)
        

        selected_MR_structure.delta_bigger_than_zero = selected_MR_structure.delta_bigger_than_zero + 1 \
            if (reward - last_reward) > 0 else selected_MR_structure.delta_bigger_than_zero
        last_reward = reward

        end_time = time.time()
        findbug_time = end_time - start_time  
        if ('state' in log_dict[n]) and ("Failed" in log_dict[n]['state']):
            metrics_dict[d_new_name] = ["None"]*11
            
        else:
            metrics_dict[d_new_name].append(findbug_time)
        print(metrics_dict[d_new_name])

        del formatted_data
        gc.collect()
        torch.cuda.empty_cache()

        # 写入json
        if ('state' in log_dict[n]) and ("Success" not in log_dict[n]['state']):  # 变异失败了
            log_dict[n]['select_d_name'] = select_d_name
        else:
            log_dict[n]['state']='Success!'
            log_dict[n]['select_deadcode'] = selected_deadcode_name
            log_dict[n]['selected_MR_structure'] = selected_MR_structure_name
            log_dict[n]['api_mutation_type(seed_model or deadcode)'] = api_mutation_type
            log_dict[n]['select_d_name'] = select_d_name
        # 保存json:
        dict_save_path = os.path.join("results", seed_model, str(localtime),"TORCH_LOG_DICT_" + str(ms_device).replace(':', '_') + ".json")
        os.makedirs(os.path.dirname(dict_save_path), exist_ok=True)
        with open(dict_save_path, 'w', encoding='utf-8') as file:
            json.dump(log_dict, file, ensure_ascii=False, indent=4)

        # 保存指标
        df = pd.DataFrame([(k, v[0], v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9]) for k, v in metrics_dict.items()],
                        columns=['New_Model_Name', 'LIC','LPC','LSC','Avg_cov','Elapsed_time', 'Gpu_Memory_Used', 'Cpu_Memory_Used', "loss","distance", 'loss_time'])
                        # columns=['New_Model_Name', 'LIC','LPC','LSC','Avg_cov','Elapsed_time', 'Gpu_Memory_Used', 'Cpu_Memory_Used','loss_time'])
        save_path = os.path.join("results", seed_model, str(localtime),str(ms_device).replace(':', '_') + "_" + train_config['seed_model'] + ".csv")
        df.to_csv(save_path, index=False)


        print('state',log_dict[n]['state'])

        print('-----------------------total_Mutate_time:%d ended!-----------------------' % n)
        # new_net = copy.deepcopy(next_net)
        new_outputs = next_output

        

