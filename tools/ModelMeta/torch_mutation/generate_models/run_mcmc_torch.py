import numpy as np
import pandas as pd
from torch_mutation.cargo import *
import torch
import torch.optim as optim
import torch.distributions as dist
import copy
import time
import json
import torch.fx as fx
from torch_mutation.MR_structure import *
from torch_mutation.cargo import match_rule,reflect_name,MCMC,compute_gpu_cpu
from torch_mutation.api_mutation import api_mutation
from torch_mutation.cargo import select_places,max_seed_model_api_times
import psutil
import sys
import torch_mutation.config
from torch_mutation.calculate_coverage import model2cov
from torch_mutation.handel_shape import handle_format

MR_structure_name_list_ori = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']
MR_structures_map_ori = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B}
deadcode_name_list=['Dense', 'SELayer', 'DenseLayer', 'Inception_A', 'PWDWPW_ResidualBlock', 'ResidualBlock', 'DropPath']
nlp_cargo = ["LSTM","FastText", "TextCNN", "SentimentNet", "GPT"]
device = config.device

def run_mcmc_torch(seed_model, mutate_times,num_samples, mr_index, ifapimut):
    localtime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())


    MCMC_selector = MCMC()
    last_MR_structure_name = None
    last_reward = 0  # 用于更新MCMC
    log_dict = {}
    if len(mr_index) == 1:
        MR_structure_name_list = [MR_structure_name_list_ori[mr_index[0]]]
    else:
        MR_structure_name_list = [MR_structure_name_list_ori[i] for i in mr_index]
    print(MR_structure_name_list)
    valid_keys = [key for key in MR_structure_name_list if key in MR_structure_name_list]
    MR_structures_map = {key: MR_structures_map_ori[key] for key in valid_keys}
    print(MR_structures_map)
    # 数值数据
    # data = np.load(datasets_path_cargo[seed_model])
    # samples = np.random.choice(data.shape[0], num_samples, replace=False)
    # samples_data = data[samples] # 随机选择num_samples个数据 (num_samples, 3, 32, 32)
    # data_selected = torch.tensor(samples_data,dtype=torch.int32 if seed_model in ["LSTM", "textcnn", "FastText"] else torch.float32).to(device)
    # # 保存选中的数据
    # npy_path = os.path.join("results", seed_model, str(localtime), 'data0_npy.npy')
    # os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    # np.save(npy_path, [data_selected.cpu().numpy()])

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
        data_selected = torch.tensor(samples_data,dtype=torch.int32 if seed_model in nlp_cargo else torch.float32)
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
                dt = torch.float32
            elif "SSD" in seed_model:
                dt = torch.int32

            label_selected1 = torch.tensor(samples_data,dtype=torch.float32)
            label_selected2 = torch.tensor(samples_label2, dtype=dt)
            label_selected3 = torch.tensor(samples_label3, dtype=dt)
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
            label_selected = torch.tensor(samples_data,dtype=torch.int32)
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
            labels_path = os.path.join("results", seed_model, str(localtime), 'data1_npy_y.npy')
            # print(labels_path_cargo[seed_model])
            targets = np.load(labels_path_cargo[seed_model])
            samples_label = targets[samples]
            label_selected = torch.tensor(samples_data,dtype=torch.float32)
            np.save(labels_path, label_selected)

    seed_model_net=get_model(seed_model, device).to(device)
    d=fx.symbolic_trace(seed_model_net)
    original_outputs = handle_format(d(data_selected))[0]
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

        loss_fun_ms, loss_fun_pt = get_loss(loss_name)
        loss_fun_pt = loss_fun_pt()

        
    metrics_dict = dict()
    select_d_name = seed_model
    D = {seed_model: d} # 所有模型
    R = {0:[1, seed_model]} # 所有变异成功的模型。{序号:[reward,model_name]}
    MR_structure_selected_nums = {k: 0 for k in MR_structure_name_list}  # 字典，每种MR_structure已被用几次。用于命名新模型
    seed_model_api_times=0 # api级别的变异几次是种子模型
    d = copy.deepcopy(d)

    # 循环变异：
    for n in range(mutate_times):
        print('-----------------------total_Mutate_time:%d start!-----------------------' % n)
        log_dict[n] = {}
        log_dict[n]['d_name'] = select_d_name

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

        # 算子变异对死代码还是原模型？
        if selected_deadcode_name in ('DropPath', 'Dense') and seed_model_api_times<max_seed_model_api_times(seed_model):  # 如果选择的死代码中无可变异算子，且原模型中有算子可被变异
            api_mutation_type = 'seed_model'
            seed_model_api_times+=1
        elif selected_deadcode_name not in ('DropPath', 'Dense') and seed_model_api_times<max_seed_model_api_times(seed_model): # 如果选择的死代码中有可变异算子，且原模型中有算子可被变异
            api_mutation_type = random.choice(['seed_model', 'deadcode'])  # 随机选择是对原模型还是死代码变异
            if api_mutation_type=='seed_model':
                seed_model_api_times += 1
        elif selected_deadcode_name not in ('DropPath', 'Dense') and seed_model_api_times >= max_seed_model_api_times(seed_model):  # 如果选择的死代码中有可变异算子，且原模型中无算子可被变异
            api_mutation_type = 'deadcode'
        else: # 种子模型没有变异的算子了，且选择的死代码中也没有变异的算子
            api_mutation_type = 'None'
            log_dict[n]['state'] = "Success:But no APIs available for mutation, so no API-level mutation was performed."

        with torch.no_grad():
            graph = d.graph
            nodelist = []
            for node in graph.nodes:
                if node.op in ['call_module', 'root'] or \
                        (node.op == "call_function" and  any(substring in node.name for substring in ['uoc', 'pioc', 'absoc_a', 'absoc_b'])):
                    nodelist.append(node)

            # 选择插入位置
            subs_place, dep_places = select_places(range(0, len(nodelist)), 5)

            if subs_place is None or dep_places is None: # 没找到合适的插入位置
                print("mutate failed for Cannot find suitable places！")
                sys.exit("Terminating the program due to unsuitable places.")

            print("~~~~~~~~~~~~~~~~~选择对%s中的算子进行api变异！~~~~~~~~~~~~~~~" % api_mutation_type)
            try:
                add_module = MR_structures_map[selected_MR_structure_name](selected_deadcode_name, api_mutation_type, log_dict, n, LOG_FLAG=False)
            except Exception as e:
                print(e)
                exit(250)

            dep_places.sort(reverse=True)
            aa = nodelist[dep_places[-1]]
            bb = nodelist[dep_places[-2]]
            cc = nodelist[dep_places[-3]]
            dd = nodelist[dep_places[-4]]

            if selected_MR_structure_name == "PIOC":
                if len(aa.users) == 0 or len(bb.users) == 0 or len(cc.users) == 0:
                    log_dict[n]['state'] = "Failed:选择插入的节点位置不正确！"
                    print("选择插入的节点位置不正确！" )
                    continue
                with cc.graph.inserting_after(cc):
                    new_hybrid_node = cc.graph.call_function(add_module, args=(cc, cc, cc))
                    cc.replace_all_uses_with(new_hybrid_node)
                    new_hybrid_node.update_arg(0, aa)
                    new_hybrid_node.update_arg(1, bb)
                    new_hybrid_node.update_arg(2, cc)
            else:  # selected_MR_structure_name != "PIOC"
                if len(aa.users) == 0 or len(bb.users) == 0 or len(cc.users) == 0 or len(dd.users) == 0:
                    log_dict[n]['state'] = "Failed:选择插入的节点位置不正确！"
                    print("选择插入的节点位置不正确！")
                    continue
                with dd.graph.inserting_after(dd):
                    new_hybrid_node = dd.graph.call_function(add_module, args=(dd, dd, dd, dd))
                    dd.replace_all_uses_with(new_hybrid_node)
                    new_hybrid_node.update_arg(0, aa)
                    new_hybrid_node.update_arg(1, bb)
                    new_hybrid_node.update_arg(2, cc)
                    new_hybrid_node.update_arg(3, dd)
            graph.lint() # 检查是否有图错误并重新编译图
            d.recompile()
            D[d_new_name] = d  # 加入到种子模型池

            if api_mutation_type == 'seed_model':
                try:
                    api_mutation(d, log_dict, n,LOG_FLAG=False) # 进行API变异
                except Exception as e:
                    print(f"Error during api_mutation: {e}")
                    log_dict[n]['state'] = f"Failed: api_mutation failed: {str(e)}"

            # 推理
            d = d.to(device)
            new_outputs = handle_format(d(data_selected))[0]
            if new_outputs.shape!=original_outputs.shape:
                print('new_outputs.shape!=original_outputs.shape!')
                sys.exit('new_outputs.shape!=original_outputs.shape!')
            if True:
                if seed_model == "openpose":
                    from models.openpose.src.loss import BuildTrainNetwork as BuildTrainNetwork_ms
                    from models.openpose.src.loss import openpose_loss as openpose_loss_ms
                    criterion = openpose_loss_ms()
                    train_net_ms = BuildTrainNetwork_ms(d, criterion)
                    loss = train_net_ms(data_selected, label_selected1, label_selected2, label_selected3)
                    # loss,grads = train_step_openpose(train_net_ms, seed_optimizer_fun, data_selected, label_selected1, label_selected2, label_selected3)
                elif seed_model == "ssimae":
                    from models.ssimae.src.network import SSIMLoss, AutoEncoder, NetWithLoss
                    loss = SSIMLoss()
                    train_net_ms = NetWithLoss(d, loss)
                    loss = train_net_ms(data_selected)
                    # loss,grads = train_step_ssimae(train_net_ms, seed_optimizer_fun, data_selected)

                elif seed_model == "crnn":
                    from mindspore.nn.wrap import WithLossCell
                    from models.CRNN.src.loss import CTCLoss
                    from models.CRNN.src.model_utils.config import config as crnnconfig

                    crnnloss = CTCLoss(max_sequence_length=crnnconfig.num_step,
                                    max_label_length=crnnconfig.max_text_length,
                                    batch_size=num_samples)
                    train_net_ms = WithLossCell(d, crnnloss)
                    loss = train_net_ms(data_selected, label_selected)
                    # loss,grads = train_step_crnn(train_net_ms, seed_optimizer_fun, data_selected, label_selected)
                elif "SSD" in seed_model:
                    pred_loc_ms, pred_label_ms = d(data_selected)
                    loss = loss_fun_pt(pred_loc_ms, pred_label_ms, label_selected1, label_selected2, label_selected3)
                    # loss, grads = train_step_ssd(seed_model_net, loss_fun_pt, seed_optimizer_fun, data_selected, label_selected1, label_selected2, label_selected3)
                elif "yolo" in seed_model:
                    input_shape = data_selected.shape[2:4]
                    input_shape_ms = torch.Tensor(tuple(input_shape[::-1]), torch.float32)

                    yolo_output = d(data_selected)

                    loss = loss_fun_pt(yolo_output,label_selected1,
                                                    label_selected2,
                                                    label_selected3,
                                                    label_selected4, label_selected5, label_selected6,input_shape_ms)
                    # loss = train_step_yolo(seed_model_net,
                    #                                 loss_fun_pt,
                    #                                 seed_optimizer_fun,
                    #                                 data_selected,
                    #                                 label_selected1,
                    #                                 label_selected2,
                    #                                 label_selected3,
                    #                                 label_selected4, label_selected5, label_selected6,input_shape_ms)


                else:
                    pred = d(data_selected)
                    loss = loss_fun_pt(pred, label_selected)
            # for param in d.parameters(): #检验device:gpu or cpu
            #     print(f"参数在设备: {param.device}")
            # print(data_selected.device)

            # 选择下一个种子模型
            if ('state' in log_dict[n]) and ("Success" not in log_dict[n]['state']):  # 在上述过程中变异失败了
                reward=-1
                # Thompson sampling strategy选择1个模型作为下次变异的种子模型
                d_probs = torch.distributions.Beta(torch.tensor([value[0] for value in R.values()]), torch.ones(len(R))).sample()
                select_d_name = R[torch.argmax(d_probs).item()][1]
                d = copy.deepcopy(D[select_d_name])
                metrics_dict[d_new_name]=["None"]*7
            else:
                json_file_path=os.path.join("results", seed_model, str(localtime),"model_json" , d_new_name+ ".json")
                all_json_path = os.path.join("results", seed_model, str(localtime), "all_layer_info.json")
                api_config_pool_path=r"./torch_api_config_pool.json"
                folder_path=os.path.join("results", seed_model, str(localtime),"model_json")
                input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov = model2cov(d, data_selected, [torch.float32],
                                                                            json_file_path,all_json_path,api_config_pool_path,folder_path)
                reward= (input_cov+config_cov+api_cov)/3
                metrics_dict[d_new_name] = [input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, reward]

                R[len(R)]=[reward,d_new_name] # 新模型的reward,且只有合法模型在R中，但所有模型在D中
                d = copy.deepcopy(d)  # 以此轮变异生成的新模型作为下次变异的种子模型
                select_d_name=d_new_name
        ## with torch.no_grad()结束

        selected_MR_structure.delta_bigger_than_zero = selected_MR_structure.delta_bigger_than_zero + 1 \
            if (reward - last_reward) > 0 else selected_MR_structure.delta_bigger_than_zero
        last_reward = reward

        # 写入json
        log_dict[n]['select_deadcode'] = selected_deadcode_name
        log_dict[n]['selected_MR_structure'] = selected_MR_structure_name
        log_dict[n]['subs_place'], log_dict[n]['dep_places'] = subs_place, dep_places
        log_dict[n]['api_mutation_type(seed_model or deadcode)'] = api_mutation_type
        log_dict[n]['d_new_name'] = d_new_name
        log_dict[n]['select_d_name'] = select_d_name
        # 保存json:
        dict_save_path = os.path.join("results", seed_model, str(localtime),"TORCH_LOG_DICT_" + str(device).replace(':', '_') + ".json")
        os.makedirs(os.path.dirname(dict_save_path), exist_ok=True)
        with open(dict_save_path, 'w', encoding='utf-8') as file:
            json.dump(log_dict, file, ensure_ascii=False, indent=4)

        # 保存指标
        df = pd.DataFrame([(k, v[0], v[1],v[2],v[3],v[4],v[5],v[6]) for k, v in metrics_dict.items()],
                          columns=['New_Model_Name', 'Input_cov','Config_cov','Api_cov','op_type_cov', 'op_num_cov', 'edge_cov','Avg_cov'])
        save_path = os.path.join("results", seed_model, str(localtime),"METRICS_RESULTS_" + str(device).replace(':', '_') + ".xlsx")
        df.to_excel(save_path, index=False)

        torch.cuda.empty_cache()
        print('-----------------------total_Mutate_time:%d ended!-----------------------' % n)

