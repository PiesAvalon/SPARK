import os
import yaml

import time
from copy import deepcopy
import mindspore
from mindspore.ops import operations as P
import numpy as np

from common.log_recoder import Logger
from common.mutation_ms.model_mutation_operators_followlog import WS_mut_followlog,NS_mut_followlog,GF_mut_followlog,NAI_mut_followlog,NEB_mut_followlog,LS_mut_followlog,LC_mut_followlog,PM_mut_followlog,RA_mut_followlog,LA_mut_followlog,CM_mut_followlog,LD_mut_followlog
from common.mutation_torch.model_mutation_operators import WS_mut,NS_mut,GF_mut,NAI_mut,NEB_mut,LS_mut,LC_mut,PM_mut,RA_mut,LA_mut,CM_mut,LD_mut
from common.model_utils import get_model
from common.dataset_utils import get_dataset
from common.mutation_main import YoloUtil as YoloUtil

ms_mutation_fun={
    "WS":WS_mut_followlog,
    "NS":NS_mut_followlog,
    "GF":GF_mut_followlog,
    "NAI":NAI_mut_followlog,
    "NEB":NEB_mut_followlog,
    "LC":LC_mut_followlog,
    "LS":LS_mut_followlog,
    "RA":RA_mut_followlog,
    "LA":LA_mut_followlog,
    "PM":PM_mut_followlog,
    "CM":CM_mut_followlog,
    "LD":LD_mut_followlog,
}

torch_mutation_fun={
    "WS":WS_mut,
    "NS":NS_mut,
    "GF":GF_mut,
    "NAI":NAI_mut,
    "NEB":NEB_mut,
    "LC":LC_mut,
    "LS":LS_mut,
    "RA":RA_mut,
    "LA":LA_mut,
    "PM":PM_mut,
    "CM":CM_mut,
    "LD":LD_mut,
}


def init_logging(log_path):
    log = Logger(log_file=log_path + '/run.log')
    return log.logger


def cal_mutation_score(model, data_forcal, origin_outputs):

    if mutation_eval_metric == "origin_diff":
        if not isinstance(origin_outputs[0], tuple):
            if isinstance(data_forcal, list):  # mutiply inputs and single outputs
                out_diffs = []
                for idx in range(0, data_forcal[0].shape[0], test_size):
                    inputs = []
                    for val in data_forcal:
                        inputs.append(val[idx: (idx + test_size), :])
                    mutation_output = model(*inputs)
                    origin_output = mindspore.ops.flatten(origin_outputs[int(idx / test_size)])
                    mutation_output = mindspore.ops.flatten(mutation_output)
                    out_diff = (origin_output - mutation_output).asnumpy()
                    out_diffs.append(np.linalg.norm(out_diff))

            else:  # single inputs and single outputs
                out_diffs = []
                for idx in range(0, data_forcal.shape[0], test_size):
                    imgs_ms = data_forcal[idx:(idx + test_size), :]

                    outputs = model(imgs_ms)

                    origin_output = mindspore.ops.flatten(origin_outputs[int(idx / test_size)])
                    output = mindspore.ops.flatten(outputs)
                    out_diff = (origin_output - output).asnumpy()
                    out_diffs.append(np.linalg.norm(out_diff))



        else:
            if isinstance(data_forcal, list):  # mutiply inputs and mutiply outputs
                mutation_outputs = []
                for idx in range(0, data_forcal[0].shape[0], test_size):
                    inputs = []
                    for val in data_forcal:
                        inputs.append(val[idx: (idx + test_size), :])
                    mutation_output = model(*inputs)
                    mutation_outputs.append(mutation_output)

            else:  # single inputs and mutiply outputs
                mutation_outputs = []
                for idx in range(0, data_forcal.shape[0], test_size):
                    imgs_ms = data_forcal[idx:idx + test_size, :]

                    mutation_output = model(imgs_ms)  # mutation_outputs is a tuple whose element is tensor
                    mutation_outputs.append(mutation_output)

                if 'yolo' in model_name:
                    yolo_util = YoloUtil()
                    origin_outputs = yolo_util.reformat_outputs_first_generation(origin_outputs)
                    mutation_outputs = yolo_util.reformat_outputs_first_generation(mutation_outputs)

            out_diffs = []
            for idx in range(len(mutation_outputs)):
                origin_output = origin_outputs[idx]
                mutation_output = mutation_outputs[idx]

                for idj in range(len(origin_output)):
                    distance_vec = origin_output[idj] - mutation_output[idj]
                    distance_vec = mindspore.ops.flatten(distance_vec)

                    out_diff = np.linalg.norm(mindspore.Tensor(distance_vec).asnumpy())
                    out_diffs.append(out_diff)

        score = np.mean(out_diffs)
    return score

def get_filter_data(test_iter):
    if model_name == "fasttext":
        for item in test_iter:
            src_tokens = deepcopy(item[0])
            src_tokens_length = deepcopy(item[1])
            srctokens_ms_forcal = mindspore.Tensor(src_tokens, mindspore.int32)
            srctokenslength_ms_forcal = mindspore.Tensor(src_tokens_length, mindspore.int32)
            break
        imgs_ms_forcal = [srctokens_ms_forcal, srctokenslength_ms_forcal]


    elif model_name == "bert":
        for item in test_iter:
            data1 = deepcopy(item[0])
            data2 = deepcopy(item[1])
            data3 = deepcopy(item[2])

            data1_ms = mindspore.Tensor(data1, mindspore.int64)
            data2_ms = mindspore.Tensor(data2, mindspore.int64)
            data3_ms = mindspore.Tensor(data3, mindspore.int64)
            break
        imgs_ms_forcal = [data1_ms, data2_ms, data3_ms]

    elif model_name == "maskrcnn":
        for item in test_iter:
            data1 = deepcopy(item[0])
            data2 = deepcopy(item[1])
            data3 = deepcopy(item[2])
            data4 = deepcopy(item[3])
            data5 = deepcopy(item[4])
            data6 = deepcopy(item[5])

            data1_ms = mindspore.Tensor(data1, mindspore.float32)
            data2_ms = mindspore.Tensor(data2, mindspore.float32)
            data3_ms = mindspore.Tensor(data3, mindspore.float32)
            data4_ms = mindspore.Tensor(data4, mindspore.int32)
            data5_ms = mindspore.Tensor(data5, mindspore.bool_)
            data6_ms = mindspore.Tensor(data6, mindspore.bool_)
            break
        imgs_ms_forcal = [data1_ms, data2_ms, data3_ms, data4_ms, data5_ms, data6_ms]

    elif model_name == "fasterrcnn":
        for item in test_iter:
            data1 = deepcopy(item[0])
            data2 = deepcopy(item[1])
            data3 = deepcopy(item[2])
            data4 = deepcopy(item[3])
            data5 = deepcopy(item[4])

            data1_ms = mindspore.Tensor(data1, mindspore.float32)
            data2_ms = mindspore.Tensor(data2, mindspore.float32)
            data3_ms = mindspore.Tensor(data3, mindspore.float32)
            data4_ms = mindspore.Tensor(data4, mindspore.int32)
            data5_ms = mindspore.Tensor(data5, mindspore.float32)
            break
        imgs_ms_forcal = [data1_ms, data2_ms, data3_ms, data4_ms, data5_ms]

    elif model_name == "transformer":
        for item in test_iter:
            data1 = deepcopy(item[0])
            data2 = deepcopy(item[1])
            data3 = deepcopy(item[4])
            data4 = deepcopy(item[5])

            data1_ms = mindspore.Tensor(data1, mindspore.float32)
            data2_ms = mindspore.Tensor(data2, mindspore.float32)
            data3_ms = mindspore.Tensor(data3, mindspore.float32)
            data4_ms = mindspore.Tensor(data4, mindspore.int32)

            break
        imgs_ms_forcal = [data1_ms, data2_ms, data3_ms, data4_ms]

    elif model_name == "pangu":
        for item in test_iter:
            data1 = deepcopy(item[0])
            data2 = deepcopy(item[1])
            data3 = deepcopy(item[2])

            input_ids = mindspore.Tensor(data1, mindspore.int32)
            input_position = mindspore.Tensor(data2, mindspore.int32)
            attention_mask = mindspore.Tensor(data3, mindspore.float32)

            tokens = P.StridedSlice()(input_ids, (0, 0), (data_forcal_size, -1), (1, 1))
            input_position = P.StridedSlice()(input_position, (0, 0), (data_forcal_size, 1024), (1, 1))
            decoder_attention_masks = P.StridedSlice()(attention_mask, (0, 0, 0),
                                                       (data_forcal_size, 1024, 1024), (1, 1, 1))
            break
        imgs_ms_forcal = [tokens, input_position, decoder_attention_masks]


    else:
        train_config=config['train_config']
        for item in test_iter:
            imgs_array = deepcopy(item[0])
            if train_config['dtypes'][0] == "float":
                dtype = mindspore.float32
            elif train_config['dtypes'][0] == "int":
                dtype = mindspore.int32

            imgs_ms_forcal = mindspore.Tensor(imgs_array, dtype)
            break

    return imgs_ms_forcal



if __name__ == '__main__':

    model_name = "vit"

    f = open('./config.txt', 'w')
    config_path = f'./common/config/{model_name}.yaml'
    f.write(config_path.lower())
    f.close()

    dataset_path = r"/data1/pzy/raw/cifar10"
    case = "case1"


    config_path = os.getcwd() + '/common/config/' + model_name + '.yaml'
    with open(config_path.lower(), 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    mutation_eval_metric = config['mutation_config']['mutation_eval_metric']

    input_size_list = config['train_config']['input_size']
    input_size_lists = []
    for input_size_each in input_size_list:
        input_size_str = input_size_each[1:-1].split(",")
        input_size_list = []
        for val in input_size_str:
            if val == "":
                continue
            input_size_list.append(int(val))
        input_size_list = tuple(input_size_list)
        input_size_lists.append(input_size_list)

    input_size = input_size_lists[0]

    #create logger
    time_tuple = time.localtime(time.time())
    time_stamp = "{}.{}.{}.{}.{}.{}".format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],
                                            time_tuple[4], time_tuple[5])
    log_path = "/" + str(model_name) + "-" + time_stamp
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    run_log = init_logging(log_path)



    model_ms_origin, model_torch_origin = get_model(model_name,config['train_config']['device'],device_id=config['train_config']['device_id'],input_size=input_size)

    test_size = config['train_config']['test_size']
    data_forcal_size=int(config['train_config']['batch_size'])*2
    dataset = get_dataset(dataset_name=config['train_config']['dataset_name'])
    selected_model_num = int(config['mutation_config']['select_mutmodel_nums'])


    test_set = dataset(data_dir=dataset_path, batch_size=data_forcal_size, is_train=True)
    test_iter = test_set.create_tuple_iterator(output_numpy=True)
    imgs_ms_forcal = get_filter_data(test_iter)

    origin_outputs = []
    if isinstance(imgs_ms_forcal, list):  # mulptily model inputs
        for idx in range(0, data_forcal_size, test_size):
            inputs = []
            for val in imgs_ms_forcal:
                inputs.append(val[idx:idx + test_size, :])

            outputs = model_ms_origin(*inputs)
            origin_outputs.append(outputs)

    else:  # single model inputs
        for idx in range(0, imgs_ms_forcal.shape[0], test_size):
            imgs_ms = imgs_ms_forcal[idx:idx + test_size, :]

            outputs = model_ms_origin(imgs_ms)
            origin_outputs.append(outputs)
    


    with open(f"./common/mutation_case/{case}.yaml", 'r', encoding="utf-8") as f:
         traces = yaml.load(f, Loader=yaml.FullLoader)

    mutation_type=[]
    keys = traces.keys()
    for key in keys:
        mutation_type.append(traces[key]['mutation_type'])

    mutation_iterations = len(traces.keys())
    mutation_scores=[]
    mutation_pass_times=0
    muttype_count = dict(zip(mutation_type, [0] * len(mutation_type)))

    for i in range(1,mutation_iterations+1):
        trace_info=traces["trace"+str(i)]
        run_log.info("================== start {} generation!({}/{}) ==================".format(i, i,mutation_iterations))

        mutation_type=trace_info['mutation_type']
        select_layer_name=trace_info['select_layer_name']
        select_layer_isbasic=trace_info['select_layer_isbasic']
        new_layer_isbasic=trace_info['new_layer_isbasic']
        new_layer_type=trace_info['new_layer_type']
        activation_name=trace_info['activation_name']
        mutate_param_selname=trace_info['mutate_param_selname']
        param_value=trace_info['param_value']
        mutate_layer_indice=trace_info['mutate_layer_indice']

        if mutation_type in ['WS',"NS","GF","NAI","NEB"]:
            mutate_fun=ms_mutation_fun[mutation_type]
            run_log.info("{} mutation start!".format(mutate_fun.__name__.split("_")[0]))
            mut_result=mutate_fun(model_ms_origin, input_size,select_layer_name[0])
            if mut_result == 'True' or mut_result == True:
                muttype_count[mutation_type] += 1


        elif mutation_type =="LD":
            run_log.info("LD mutation start!")
            mut_result=LD_mut_followlog(model_ms_origin, input_size, del_layer_name=select_layer_name[0])
            if mut_result == 'True' or mut_result == True:
                muttype_count[mutation_type] += 1

        elif mutation_type=="CM":
            run_log.info("CM mutation start!")
            mut_result=CM_mut_followlog(model_ms_origin, input_size, select_layer_isbasic, select_layer_name[0], new_layer_type, activation_name)
            if mut_result == 'True' or mut_result == True:
                muttype_count[mutation_type] += 1

        elif mutation_type in ["RA","LA","LC"]:
            mutate_fun=ms_mutation_fun[mutation_type]
            run_log.info("{} mutation start!".format(mutate_fun.__name__.split("_")[0]))
            mut_result=mutate_fun(model_ms_origin, input_size, select_layer_isbasic, select_layer_name[0],new_layer_isbasic,new_layer_type, activation_name)
            if mut_result == 'True' or mut_result == True:
                muttype_count[mutation_type] += 1

        elif mutation_type == "PM":
            run_log.info("PM mutation start!")
            if isinstance(param_value,str):
                param_value=param_value[1:-1].split(",")
                param_value=tuple([int(val) for val in param_value])

            mut_result = PM_mut_followlog(model_ms_origin, input_size, select_layer_name[0], mutate_layer_indice, mutate_param_selname,param_value)
            if mut_result == 'True' or mut_result == True:
                muttype_count[mutation_type] += 1

        elif mutation_type == "LS":
            run_log.info("LS mutation start!")
            mut_result = LS_mut_followlog(model_ms_origin, input_size, select_layer_name[0], select_layer_name[1])
            if mut_result == 'True' or mut_result == True:
                muttype_count[mutation_type] += 1

        if mut_result == 'True' or mut_result == True:
            mutation_pass_times+=1
            mutate_score = cal_mutation_score(model_ms_origin, imgs_ms_forcal, origin_outputs)
            mutation_scores.append(mutate_score)
        else:
            mutation_scores.append(-100)

    run_log.info("mutation_scores: {}".format(mutation_scores))
    first_select_generations = np.argsort(np.array(mutation_scores))[(len(mutation_scores) - selected_model_num):]

    run_log.info("**** stage1 select generations: {} ****\n".format(first_select_generations + 1))


    run_log.info('mutation iteration:{}, mutation success:{}'.format(mutation_iterations, mutation_pass_times))
    run_log.info('mutation type:{}\n'.format(muttype_count))


    os.remove('./config.txt')










