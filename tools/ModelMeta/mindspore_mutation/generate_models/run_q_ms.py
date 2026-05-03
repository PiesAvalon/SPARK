import pandas as pd
from infoplus.MindSporeInfoPlus import mindsporeinfoplus
import torch.optim as optim
from mindspore_mutation.cargo import *
import copy
import time
import json
from mindspore_mutation.MR_structure import *
from mindspore_mutation.cargo import match_rule,reflect_name,MCMC,compute_gpu_cpu
from mindspore_mutation.api_mutation import api_mutation
from mindspore_mutation.calculate_coverage import model2cov,find_layer_type,json2cov
from mindspore_mutation.cargo import select_places,max_seed_model_api_times
import sys
import os
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore_mutation.handel_shape import handle_format
from mindspore_mutation import metrics
from mindspore_mutation.cargo import get_loss,get_optimizer
from mindspore_mutation.union import union_json
import gc
sys.setrecursionlimit(3000)
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")

MR_structure_name_list_ori = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']
MR_structures_map_ori = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B} # to do
nlp_cargo = ["LSTM","FastText", "TextCNN", "SentimentNet", "GPT"]
# MR_structure_name_list = ['PIOC']
# MR_structures_map = {"PIOC": PIOC} # to do

deadcode_name_list=['Dense', 'SELayer', 'DenseLayer', 'Inception_A', 'PWDWPW_ResidualBlock', 'ResidualBlock', 'DropPath']
# deadcode_name_list=['MyConvPoolLayerSameShape', 'MyAdaptiveMaxPoolLayer', 'MyTransposeConvLayer']


def train_step_3(model, loss_fn, optimizer, data, target):
    def forward_fn_3(x, y):
        pred = model(x)
        loss = loss_fn(pred, y)
        return loss

    # 计算梯度
    grad_fn = ops.value_and_grad(forward_fn_3, None, optimizer.parameters, has_aux=False)
    loss, grads = grad_fn(data, target)

    # 更新参数
    optimizer(grads)

    return loss, grads

def train_step_crnn(losser, optimizer_ms, data, label):
    def forward_fn(data, label):
        loss = losser(data, label)
        return loss

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer_ms.parameters, has_aux=False)

    (loss), grads = grad_fn(data, label)
    loss = mindspore.ops.depend(loss, optimizer_ms(grads))
    return loss, grads

def train_step_openpose(losser, opt_ms, data, label,l1,l2):
    def forward_fn(data, label, l1, l2):
        loss = losser(data, label, l1, l2)
        return loss

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt_ms.parameters, has_aux=False)

    (loss), grads = grad_fn(data, label, l1, l2)
    loss = mindspore.ops.depend(loss, opt_ms(grads))

    return loss, grads


def train_step_ssimae(losser, optimizer_ms, data):
    def forward_fn(data):
        loss = losser(data)
        return loss

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer_ms.parameters, has_aux=False)

    (loss), grads = grad_fn(data)
    loss = mindspore.ops.depend(loss, optimizer_ms(grads))
    return loss, grads

def train_step_ssd(model_ms, loss_fun, optimizer_ms, ssdx, y1, y2, y3):
    def forward_fn(x, y1,y2,y3):
        pred_loc_ms, pred_label_ms = model_ms(x)
        loss = loss_fun(pred_loc_ms, pred_label_ms, y1,y2,y3)
        return loss

    # Get gradient function
    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer_ms.parameters, has_aux=False)

    (loss), grads = grad_fn(ssdx,y1,y2,y3)
    loss = mindspore.ops.depend(loss, optimizer_ms(grads))
    return loss, grads

def train_step_yolo(model_ms, loss_fun, optimizer_ms, images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1, batch_gt_box2,input_shape):
    def forward_fn(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,
                   batch_gt_box2):
        yolo_output = model_ms(images)
        loss = loss_fun(yolo_output, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,batch_gt_box2,input_shape)
        return loss

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer_ms.parameters, has_aux=False)

    (loss), grads = grad_fn(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0,
                            batch_gt_box1,
                            batch_gt_box2)
    loss = mindspore.ops.depend(loss, optimizer_ms(grads))
    return loss


# @profile
def run_q_ms(seed_model, mutate_times, num_samples, mr_index, ifapimut, num_quantiles,ifeplison, ifTompson,device,train_config,csv_file_path):


    if device == -1:
        pt_device = "cpu"
        ms_device = "cpu"
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
    else:
        pt_device = "cuda:6"
        ms_device = "gpu"
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU", device_id=device)



    localtime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    log_dict = {}
    dtypes = [mindspore.float32] if seed_model not in nlp_cargo else [mindspore.int32]

    #筛选出本轮运行的指定MR
    if len(mr_index) == 1:
        MR_structure_name_list = [MR_structure_name_list_ori[mr_index[0]]]
    else:
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
        indexs = [i for i in range(data.shape[0])]
        samples = np.random.permutation(indexs)[:num_samples]
        samples_data = data[samples] # 随机选择num_samples个数据 (num_samples, 3, 32, 32)
        data_selected = Tensor(samples_data, dtype=mstype.int32 if seed_model in nlp_cargo else mstype.float32)
        print(data_selected.shape)
        data_npy = data_selected.asnumpy()
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
                dt = mstype.float32
            elif "SSD" in seed_model:
                dt = mstype.int32

            label_selected1 = Tensor(samples_label1, dtype=mstype.float32)
            label_selected2 = Tensor(samples_label2, dtype=dt)
            label_selected3 = Tensor(samples_label3, dtype=dt)
            np.save(labels_path1, samples_label1)
            np.save(labels_path2, samples_label2)
            np.save(labels_path3, samples_label3)
        elif seed_model in [ "srgan"]:
            batch, channels, h, w = 1, 3, 24, 24
            upscale = 4
            # 低分辨率 LR: 24×24
            lr_np = np.random.randn(batch, channels, h, w).astype(np.float32)
            # 高分辨率 HR: (h*upscale)×(w*upscale) = 96×96
            hr_np = np.random.randn(batch, channels, h*upscale, w*upscale).astype(np.float32)

            data_selected = Tensor(lr_np)
            label_selected = Tensor(hr_np)
            pass

        elif seed_model in [ "ssimae"]:
            # data_selected = Tensor(samples_data, dtype=mstype.float64)
            pass
        elif seed_model in ["crnn", "DeepLabV3","TextCNN", "resnet", 'vit']:
            labels_path = os.path.join("results", seed_model, str(localtime), 'data1_npy_y.npy')
            # print(labels_path_cargo[seed_model])
            targets = np.load(labels_path_cargo[seed_model])
            print("targets",targets.shape)
            samples_label = targets[samples]
            print(samples)
            print(samples_label.shape)
            label_selected = Tensor(samples_label, dtype=mstype.int32)
            print("label_selected",label_selected.shape)
            print("label_selected",label_selected)
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

        loss_fun_ms, _ = get_loss(loss_name)
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

    original_outputs = handle_format(new_net(data_selected))
    new_outputs = original_outputs
    print("new_outputs.shape:",new_outputs.shape)
    last_outputs = copy.deepcopy(original_outputs)
    # Q网络环境参数
    n_actions = len(mr_index)  # 动作数量 ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']
    state_dim = 1 # 状态维度（模型的数值输出）
    print(n_actions)
    for i in original_outputs.shape:
        state_dim *= i
    print(state_dim)

    gamma = 0.99  # 折扣因子
    target_update = 1 # update_state轮更新一次Target_Q网络的参数
    epsilon = ifeplison
    epsilon_end = 0.01
    epsilon_decay = 0.995
    # 初始化Q网络
    Quantum_Q = DQN(state_dim, n_actions,num_quantiles).to(pt_device)
    Target_Q = DQN(state_dim, n_actions,num_quantiles).to(pt_device)
    Target_Q.eval()  # 目标网络在推理过程中不需要梯度计算


    # 优化器和损失函数
    optimizer = optim.Adam(Quantum_Q.parameters(), lr=0.001)
    quantiles = (torch.linspace(0, 1, num_quantiles) + 0.5 / num_quantiles).to(pt_device)

    metrics_dict = dict()
    select_d_name = seed_model
    D = {seed_model: stree} # 所有模型
    INDEX_N = [] # 所有模型
    O = {seed_model: original_outputs} # 所有模型对应的输出
    N = {seed_model: seed_model_net} # 所有模型
    R = {0:[0.0001, seed_model]} # 所有变异成功的模型。{序号:[reward,model_name]}
    MR_structure_selected_nums = {k: 0 for k in MR_structure_name_list}  # 字典，每种MR_structure已被用几次。用于命名新模型
    seed_model_api_times = 0 #api级别的变异几次是种子模型

    states = -1
    next_states = -1
    actions = -1
    next_actions = -1
    next_quantiles = -2

    tar_set = set()
    tar_set_all = []

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
            selected_deadcode_name = random.choice(deadcode_name_list)

            # 选择 MR_structure,利用ε-greedy策略选择是否用QRDQN方法选择MR_structure
            if random.uniform(0, 1) <= epsilon and False:
                if len(mr_index) == 1:
                    selected_MR_structure_idx = random.choice([0])
                    print("random select SMR!")
                    print(selected_MR_structure_idx)
                    next_quantile_values = None
                elif len(mr_index) == 3:
                    selected_MR_structure_idx = random.choice([0, 1, 2])
                    print("random select SMR!")
                    print(selected_MR_structure_idx)
                    next_quantile_values = None 
                else:
                    selected_MR_structure_idx = random.choice(mr_index)
                    print("random select SMR!")
                    print(selected_MR_structure_idx)
                    next_quantile_values = None
            else:
                with torch.no_grad():
                    print("Q-network select SMR!")
                    # print(new_outputs.shape)
                    # print(handle_format(new_outputs).shape)
                    formatted_data = handle_format(new_outputs)
                    # np.save("./formatted_data.npy",formatted_data.asnumpy())
                    formatted_data = torch.from_numpy(formatted_data.asnumpy()).to(pt_device)
                    # print(formatted_data.shape)
                    next_quantile_values = Quantum_Q(formatted_data)
                    print("next_quantile_values",next_quantile_values.shape)
                    next_q_values = next_quantile_values.mean(dim=2)
                    print("next_q_values",next_q_values)
                    next_actions = torch.argmax(next_q_values, dim=1)
                    print("next_actions",next_actions)
                    next_quantiles = next_quantile_values[range(num_samples), next_actions]
                    selected_MR_structure_idx = next_actions.argmax().item()
                    print("selected_MR_structure_idx",selected_MR_structure_idx)
                exit()
                    # formatted_data = handle_format(new_outputs)[0]
                    # formatted_data = torch.from_numpy(formatted_data.asnumpy()).to(pt_device)
                    # selected_MR_structure_idx = Quantum_Q(formatted_data).argmax().item()


            # selected_MR_structure_idx = mr_index
            print(MR_structure_name_list)
            selected_MR_structure_name = MR_structure_name_list[selected_MR_structure_idx]
            MR_structure_selected_nums[selected_MR_structure_name] += 1
            # print(f'-----------------------{selected_MR_structure_name} start-----------------------')
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

            # # 选择插入位置
            nodedict = collections.OrderedDict()  # 特殊字典，保持插入的顺序
            hash_table = collections.defaultdict(int)  # 每次访问一个不存在的键时，该键会自动被赋值为整数 0
            flag, nodedict = scan_node(stree, hash_table, nodedict)
            node_list=list(nodedict.values())

            length = len(nodedict)
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
            if recurive_depth >= 3500: # 没找到合适的插入位置
                log_dict[n]['state'] = f"Failed:Cannot find suitable places！"
            if aa is None:

                print("mutate Failed for Cannot find suitable places")
            add_module = MR_structures_map[selected_MR_structure_name](selected_deadcode_name, api_mutation_type, log_dict, n, LOG_FLAG=False)
            # add_module = MR_structures_map_ori[selected_MR_structure_name](selected_deadcode_name, api_mutation_type, log_dict, n, LOG_FLAG=False)
            # MR_structures_map_ori
            seat = 0

            if "Failed" not in log_dict[n]['state']:
                if selected_MR_structure_name == "PIOC" and selected_deadcode_name in ["Dense", "Conv", "SELayer", "DenseLayer", "Inception_A",
                                                    "PWDWPW_ResidualBlock", "ResidualBlock", "DropPath", 'MyConvPoolLayerSameShape', 'MyAdaptiveMaxPoolLayer', 'MyTransposeConvLayer']:
                    if not (check_node(cc) and check_node(bb) and check_node(aa)):
                        log_dict[n]['state'] = "Failed:选择插入的节点位置不正确！"
                        print("Failed:选择插入的节点位置不正确！")
                        # continue

                    tree = cc.get_symbol_tree()  # 获取节点 c 所属的符号树（SymbolTree）
                    position = tree.after(cc)  # 节点 c 在符号树中的后继位置
                    next_node = cc.get_users()[0]  # 下一个使用 c 输出的节点

                    if len(next_node.get_args()) > 1:
                        for idx, arg in enumerate(next_node.get_args()):
                            if arg == cc.get_targets()[0]:
                                seat = idx  # 匹配位置
                                break


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
                # # print(log_dict[n]['state'])
                new_net = stree.get_network()
                print("data_selected: ", data_selected.shape)
                new_outputs = new_net(data_selected) # 出错处
                N[d_new_name] = new_net  # 加入到种子模型池
                new_output = handle_format(new_outputs)
                O[d_new_name] = copy.deepcopy(new_output)
                print("len(N)",len(N))
                INDEX_N.append(n)
                
            log_dict[n]["ifapimut"] = ifapimut
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
                    elif seed_model == "ssimae":
                        from models.ssimae.src.network import SSIMLoss, AutoEncoder, NetWithLoss
                        loss = SSIMLoss()
                        train_net_ms = NetWithLoss(new_net, loss)
                        loss = train_net_ms(data_selected)
                        # loss,grads = train_step_ssimae(train_net_ms, seed_optimizer_fun, data_selected)
                    elif seed_model == "srgan":
                        from models.SRGAN.src.loss.psnr_loss import PSNRLoss
                        batch, channels, h, w = 1, 3, 24, 24
                        upscale = 4
                        hr_np = np.random.randn(batch, channels, h*upscale, w*upscale).astype(np.float32)
                        hr = Tensor(hr_np)
                        loss = PSNRLoss(new_net)
                        loss_loss = loss(label_selected,data_selected)
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



                    else:
                        pred = new_net(data_selected)
                        loss = loss_fun_ms(pred, label_selected)
                    if False:
                        modelms_trainable_params = new_net.trainable_params()
                        new_trainable_params = []
                        layer_nums = 0
                        for modelms_trainable_param in modelms_trainable_params:
                            modelms_trainable_param.name = train_config['seed_model'] + str(
                                layer_nums) + "_" + modelms_trainable_param.name
                            new_trainable_params.append(modelms_trainable_param)
                            layer_nums += 1

                            seed_optimizer_fun = seed_optimizer_ms(params=new_trainable_params, learning_rate=1e-4)
                        
                        # seed_optimizer_fun = seed_optimizer_ms(params=new_net.trainable_params(), learning_rate=1e-4)
                        if seed_model == "patchcore":
                            pass
                        elif seed_model in ["ssimae", "crnn", "openpose"]:
                            if seed_model == "ssimae":
                                from models.ssimae.src.network import SSIMLoss, AutoEncoder, NetWithLoss
                                loss = SSIMLoss()
                                train_net_ms = NetWithLoss(new_net, loss)
                                loss,grads = train_step_ssimae(train_net_ms, seed_optimizer_fun, data_selected)


                            elif seed_model == "crnn":
                                from mindspore.nn.wrap import WithLossCell
                                from models.CRNN.src.loss import CTCLoss
                                from models.CRNN.src.model_utils.config import config as crnnconfig

                                crnnloss = CTCLoss(max_sequence_length=crnnconfig.num_step,
                                                max_label_length=crnnconfig.max_text_length,
                                                batch_size=num_samples)
                                train_net_ms = WithLossCell(new_net, crnnloss)
                                loss,grads = train_step_crnn(train_net_ms, seed_optimizer_fun, data_selected, label_selected)

                            elif seed_model == "openpose":
                                from models.openpose.src.loss import BuildTrainNetwork as BuildTrainNetwork_ms
                                from models.openpose.src.loss import openpose_loss as openpose_loss_ms
                                criterion = openpose_loss_ms()
                                train_net_ms = BuildTrainNetwork_ms(new_net, criterion)
                                loss,grads = train_step_openpose(train_net_ms, seed_optimizer_fun, data_selected, label_selected1, label_selected2, label_selected3)


                        elif "yolo" in seed_model:
                            input_shape = data_selected.shape[2:4]
                            input_shape_ms = Tensor(tuple(input_shape[::-1]), ms.float32)

                            yolo_output = seed_model_net(data_selected)

                            loss = train_step_yolo(seed_model_net,
                                                            loss_fun_ms,
                                                            seed_optimizer_fun,
                                                            data_selected,
                                                            label_selected1,
                                                            label_selected2,
                                                            label_selected3,
                                                            label_selected4, label_selected5, label_selected6,input_shape_ms)



                        elif "SSD" in seed_model:
                            print(loss_fun_ms)
                            loss, grads = train_step_ssd(seed_model_net, loss_fun_ms, seed_optimizer_fun, data_selected, label_selected1, label_selected2, label_selected3)

                        else:
                            # loss,grads = train_step_3(new_net, loss_fun_ms, seed_optimizer_fun, data_selected, label_selected)
                            #print("loss_ms", loss_ms)
                            # def forward_fn_3(x, y):
                            pred = new_net(data_selected)
                            loss = loss_fun_ms(pred, label_selected)
                                # return loss

                            # # 计算梯度
                            # grad_fn = ops.value_and_grad(forward_fn_3, None, seed_optimizer_fun.parameters, has_aux=False)
                            # loss, grads = grad_fn(data_selected, label_selected)

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
                
                reward, done = -1, True
                if ifTompson:
                    # Thompson sampling strategy选择1个模型作为下次变异的种子模型
                    d_probs = torch.distributions.Beta(torch.tensor([value[0] for value in R.values()]), torch.ones(len(R))).sample()
                    select_d_name = R[torch.argmax(d_probs).item()][1]
                else:
                    select_d_name = random.choice(list(R.items()))[1][1]

                stree = D[select_d_name]
                metrics_dict[d_new_name]=["None"]*4
                next_output = O[select_d_name]
                formatted_data = O[select_d_name]
                
            else:
                done = False
                json_file_path= os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results", seed_model, str(localtime),"model_json" , str(n) + ".json")
                os.makedirs(os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results", seed_model, str(localtime),"model_json"), exist_ok=True)
                all_json_path="/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/ms_all_layer_info.json"
                api_config_pool_path = '/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/mindspore_api_config_pool.json'
                folder_path = '/home/cvgroup/myz/czx/semtest-gitee/'
                input_data = mindsporeinfoplus.np_2_tensor([data_npy], dtypes)
                stree = stree # SymbolTree.create(new_net)  #


                input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside, output_datas = model2cov(new_net,
                                                                                                        input_data,
                                                                                                        dtypes,
                                                                                                        json_file_path,
                                                                                                        all_json_path,
                                                                                                        api_config_pool_path,
                                                                                                        folder_path)
                print("input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside: ", input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside)

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
                R[len(R)] = [reward, d_new_name] # 新模型的reward,且只有合法模型在R中，但所有模型在D中
                select_d_name=d_new_name
                next_output = O[d_new_name]# new_net(data_selected)
                formatted_data = O[d_new_name]
        ## with torch.no_grad()结束

        # 更新Q
        if next_actions > -1 :
            if actions==-1:
                # print(actions==-1)
                # metrics_dict[d_new_name].append(0)
                actions = next_actions

            formatted_data_torch = torch.from_numpy(formatted_data.asnumpy()).to(pt_device)
            # print("formatted_data2: ", formatted_data_torch.shape)
            quantile_values = Target_Q(formatted_data_torch)
            quantile_values = quantile_values[range(num_samples), actions]
            target_quantiles = reward + gamma * (1 - done) * next_quantiles
            td_error = target_quantiles.unsqueeze(1) - quantile_values.unsqueeze(2)
            huber_loss = torch.where(td_error.abs() <= 1.0, 0.5 * td_error ** 2,td_error.abs() - 0.5)  # 形状: (batch_size, num_quantiles, num_quantiles)
            quantile_loss = (quantiles - (td_error.detach() < 0).float()).abs() * huber_loss  # 形状: (batch_size, num_quantiles, num_quantiles)
            loss = quantile_loss.mean()  # 标量
            metrics_dict[d_new_name].append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            actions = next_actions
            del formatted_data, quantile_values, next_quantile_values
        else:
            metrics_dict[d_new_name].append(None)

        if n % target_update == 0:  # 更新Target_Q
            Target_Q.load_state_dict(Quantum_Q.state_dict())

        #epsilon 衰减
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        # print("epsilon:",epsilon)
        end_time = time.time()
        findbug_time = end_time - start_time  
        if ('state' in log_dict[n]) and ("Failed" in log_dict[n]['state']):
            metrics_dict[d_new_name] = ["None"]*11
            
        else:
            metrics_dict[d_new_name].append(findbug_time)
        # print(metrics_dict[d_new_name])
        # print(log_dict[n]['state'])
        gc.collect()
        torch.cuda.empty_cache()    
        # print(metrics_dict)
        # 写入json
        if ('state' in log_dict[n]) and ("Failed" in log_dict[n]['state']):  # 变异失败了
            log_dict[n]['select_d_name'] = select_d_name
        else:
            log_dict[n]['state']='Success!'
            log_dict[n]['select_deadcode'] = selected_deadcode_name
            log_dict[n]['selected_MR_structure'] = selected_MR_structure_name
            log_dict[n]['api_mutation_type(seed_model or deadcode)'] = api_mutation_type
            log_dict[n]['select_d_name'] = select_d_name
            print("yes")
        # 保存json:
        dict_save_path = os.path.join("results", seed_model, str(localtime),"Mindpore_LOG_DICT_" + str(ms_device).replace(':', '_') + ".json")
        os.makedirs(os.path.dirname(dict_save_path), exist_ok=True)
        with open(dict_save_path, 'w', encoding='utf-8') as file:
            json.dump(log_dict, file, ensure_ascii=False, indent=4)

        # 保存指标
        print(metrics_dict[d_new_name])
        df = pd.DataFrame([(k, v[0], v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10]) for k, v in metrics_dict.items()],
                        columns=['New_Model_Name', 'LIC','LPC','LSC','Avg_cov','Elapsed_time', 'Gpu_Memory_Used', 'Cpu_Memory_Used', "loss","distance","q_loss", 'loss_time'])
                        # columns=['New_Model_Name', 'LIC','LPC','LSC','Avg_cov','Elapsed_time', 'Gpu_Memory_Used', 'Cpu_Memory_Used','loss_time'])
        # save_path = os.path.join("results", seed_model, str(localtime),str(ms_device).replace(':', '_') + "_" + train_config['seed_model'] + ".csv")
        df.to_csv(csv_file_path, index=False)

        input_data = mindsporeinfoplus.np_2_tensor([data_npy], dtypes)
        json_file_path = os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results", seed_model,str(localtime), "model_json", str(n) + ".json")
        os.makedirs(os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results", seed_model, str(localtime),"model_json"), exist_ok=True)
        all_json_path = "/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/ms_all_layer_info.json"
        api_config_pool_path = '/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/mindspore_api_config_pool.json'
        folder_path = '/home/cvgroup/myz/czx/semtest-gitee/'
        input_data = mindsporeinfoplus.np_2_tensor([data_npy], dtypes)

        input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside, output_datas = model2cov(new_net,
                                                                                                            input_data,
                                                                                                            dtypes,
                                                                                                            json_file_path,
                                                                                                            all_json_path,
                                                                                                            api_config_pool_path,
                                                                                                            folder_path)

        print("input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside: ", input_cov, config_cov,
              api_cov, op_type_cov, op_num_cov, edge_cov, inside)

        new_outputs = next_output
        last_outputs = copy.deepcopy(next_output)
        # print('state',log_dict[n]['state'])

        print('-----------------------total_Mutate_time:%d ended!-----------------------' % n)

