import numpy as np
import pynvml
from copy import deepcopy
import torch.nn as nn_torch
import mindspore
import mindspore.nn as nn_ms
from mindspore.ops import operations as P
from common.mutation_ms.Other_utils import create_replacecell as create1
from common.mutation_torch.Other_utils import create_replacecell as create2


class YoloUtil:
    def __init__(self):
        super(YoloUtil, self).__init__()

    @staticmethod
    def reformat_outputs_first_generation(multi_outputs):
        reformat_outputs_list = []
        for tuple_output in multi_outputs:
            outputs_list = []
            for tensor_tuple in tuple_output:
                outputs_list.extend(tensor_tuple)
            reformat_outputs_list.append(tuple(outputs_list))
        return reformat_outputs_list

    @staticmethod
    def reformat_outputs_second_generation(multi_outputs):
        reformat_outputs_list = []
        for tuple_output in multi_outputs:
            for tensor_tuple in tuple_output:
                reformat_outputs_list.append(tensor_tuple)
        return tuple(reformat_outputs_list)


def get_filter_data(model_name, test_iter, data_forcal_size, dtypes):
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

    elif model_name == "wide_and_deep":
        for item in test_iter:
            data1 = deepcopy(item[0])
            data2 = deepcopy(item[1])

            data1_ms = mindspore.Tensor(data1, mindspore.int32)
            data2_ms = mindspore.Tensor(data2, mindspore.float32)
            break
        imgs_ms_forcal = [data1_ms, data2_ms]

    elif model_name == "deepspeech2":
        for item in test_iter:
            data1 = deepcopy(item[0])
            data2 = deepcopy(item[1])

            data1_ms = mindspore.Tensor(data1, mindspore.float32)
            data2_ms = mindspore.Tensor(data2, mindspore.float32)
            data2_ms = data2_ms.reshape((10, 1))
            break
        imgs_ms_forcal = [data1_ms, data2_ms]

    else:
        for item in test_iter:
            imgs_array = deepcopy(item[0])
            if dtypes[0] == "float":
                dtype = mindspore.float32
            elif dtypes[0] == "int":
                dtype = mindspore.int32

            imgs_ms_forcal = mindspore.Tensor(imgs_array, dtype)
            break

    return imgs_ms_forcal


def nvidia_info():
    nvidia_dict = {
        "state": True,
        "nvidia_version": "",
        "nvidia_count": 0,
        "gpus": []
    }
    try:
        pynvml.nvmlInit()
        nvidia_dict["nvidia_version"] = pynvml.nvmlSystemGetDriverVersion()
        nvidia_dict["nvidia_count"] = pynvml.nvmlDeviceGetCount()
        for i in range(nvidia_dict["nvidia_count"]):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "gpu_name": pynvml.nvmlDeviceGetName(handle),
                "total": memory_info.total,
                "free": memory_info.free,
                "used": memory_info.used,
                "temperature": f"{pynvml.nvmlDeviceGetTemperature(handle, 0)}℃",
                "powerStatus": pynvml.nvmlDeviceGetPowerState(handle)
            }
            nvidia_dict['gpus'].append(gpu)
    except pynvml.NVMLError as _:
        nvidia_dict["state"] = False
    except Exception as _:
        nvidia_dict["state"] = False

    return nvidia_dict


def check_gpu_mem_usedRate():
    info = nvidia_info()
    used = info['gpus'][0]['used']
    tot = info['gpus'][0]['total']
    return used, tot


class GetGPUInfo:
    def __init__(self, use_index=(0,)):
        self.use_index = use_index

    @staticmethod
    def get_gpu_info(use_index=(0,)) -> str:
        """
        :param use_index: 使用的GPU的物理编号
        :return: 显存使用的信息str
        """

        def func(number):
            # number单位是MB
            if number // 1024 > 0:  # 如果number对1024取整是大于0的说明单位是GB
                return f"{number / 1024.0:.3f}GB"  # 返回值的单位是GB
            else:
                return f"{number:.3f}MB"

        # 初始化管理工具
        pynvml.nvmlInit()
        # device = torch.cuda.current_device()  # int
        gpu_count = pynvml.nvmlDeviceGetCount()  # int
        information = []
        for index in range(gpu_count):
            # 不是使用的gpu，就剔除
            if index not in use_index:
                continue
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = meminfo.total / 1024 ** 2  # 总的显存大小,单位是MB
            used = meminfo.used / 1024 ** 2  # 已用显存大小
            free = meminfo.free / 1024 ** 2  # 剩余显存大小
            information.append(f"\nMemory Total:{func(total)}; Memory Used:{func(used)}; Memory Free:{func(free)}")
        # 关闭管理工具
        pynvml.nvmlShutdown()
        return "".join(information)

    def __call__(self):
        return self.get_gpu_info(use_index=self.use_index)


def rename_parameter(model_ms):
    layer_names = list(model_ms.layer_names.keys())
    for layer_name in layer_names:
        layer = model_ms.get_layers(layer_name)
        params_generator = layer.get_parameters()
        nums = 0
        for p in params_generator:
            nums += 1

        if list(params_generator) == []:
            continue
        p.name = layer_name + "." + p.name


def scan_replacecell(model, batch_size, framework):
    if framework == "torch":
        create = create2
    elif framework == "mindspore":
        create = create1

    layer_names = list(model.origin_layer_names.keys())
    for layer_name in layer_names:
        layer = model.origin_layer_names[layer_name]

        layer_type_name = layer.__class__.__name__
        if "Sequential" in layer_type_name:
            new_layer = []

            for l in range(len(layer)):
                if "Replace" in layer[l].__class__.__name__:
                    in_shape, out_shape = list(layer[l].in_shape), list(layer[l].out_shape)
                    in_shape[0] = batch_size
                    out_shape[0] = batch_size
                    new_layer.append(create(in_shape, out_shape))
                else:
                    new_layer.append(layer[l])
            if framework == "mindspore":
                set_layer = nn_ms.SequentialCell(new_layer)
            else:
                set_layer = nn_torch.Sequential(*new_layer)
            model.set_origin_layers(layer_name, set_layer)

        elif "Replace" in layer_type_name:
            in_shape, out_shape = list(layer.in_shape), list(layer.out_shape)
            in_shape[0] = batch_size
            out_shape[0] = batch_size
            new_layer = create(in_shape, out_shape)
            model.set_origin_layers(layer_name, new_layer)


class ToolUtils:
    @staticmethod
    def select_mutant(roulette, **kwargs):
        return roulette.choose_mutant()

    @staticmethod
    def select_mutator(logic, **kwargs):
        last_used_mutator = kwargs['last_used_mutator']
        return logic.choose_mutator(last_used_mutator)

    @staticmethod
    def get_HH_mm_ss(td):
        days, seconds = td.days, td.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return hours, minutes, secs
