import torch

import copy
import torch.fx.graph
import os
import platform
import numpy as np
import  time
from numpy import ndarray
from torch.fx import symbolic_trace
import onnx
import mindspore as ms
from logger import Logger
from infoplus.TorchInfoPlus import torchinfoplus
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
import tqdm
import json
from onnx import shape_inference

from onnx_mutation.cargo import *
from arg import init_config, mutation_args
from onnx_mutation.node_gen import handle_tuple
from onnx_mutation.distance import ChebyshevDistance, distance, distance_MODE
from CKPT_Converter import compare_layer
from onnx_mutation.handel_data import dataset, create_dataset
from onnx_mutation.utils.onnx_utils import name_obj_dict
args = init_config(mutation_args)
import onnx_mutation.edge_node
from onnx_mutation.node_gen import make_node_chain_generator
from onnx_mutation.mutations import *
from onnx_mutation.deadcode import DeadGenerator

def info_com(model, np_data, dtypes, verbose=0):  # 计算每一层的输出
    torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, device)
    result, global_layer_info = torchinfoplus.summary(
        model=model,
        input_data=torch_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=verbose)
    # print("result", result)
    input_datas = torchinfoplus.get_input_datas(global_layer_info)
    output_datas = torchinfoplus.get_output_datas(global_layer_info)
    return input_datas, output_datas


def save_layer_info(input_data_dict_new, input_data_dict_old, Mutate_time, filename="input_"):
    # pprint(input_data_dict_new)
    f = open(filename + "_" + str(device).replace(':', '_') + "_" + str(Mutate_time) + "times" + ".txt",
             "w")
    for layer in input_data_dict_new.keys():
        if input_data_dict_new[layer] is not None and input_data_dict_old[layer] is not None:
            layer_np_new = input_data_dict_new[layer][0]
            layer_up_old = input_data_dict_old[layer][0]
            # if isinstance(layer_np_new, tuple):
            layer_np_new = handle_tuple(layer_np_new)
            # if isinstance(layer_up_old, tuple):
            layer_up_old = handle_tuple(layer_up_old)
            f.write("=====================================================" + "\n")
            f.write(layer.replace('.', '_') + " distance is " + str(distance(layer_np_new, layer_up_old))
                    + " ChebyshevDistance is " + str(ChebyshevDistance(layer_np_new, layer_up_old)) + "\n")
    f.close()

class onnx_Mutator:
    def __init__(self, net_name, method, distance_mode, time_time, frame_name):

        self.net = onnx.load("./vgg11.onnx")

        self.old_net = copy.deepcopy(self.net)
        
        self.method = method

        self.distance_real = distance_MODE[distance_mode]

        self.time_time = time_time
        
        self.frame_name = frame_name

        if not os.path.exists("onnx_mutated_net"):
            os.mkdir("onnx_mutated_net")
        if not os.path.exists("onnx_mutated_net/" + str(self.net.__class__.__name__) + "/"):
            os.mkdir("onnx_mutated_net/" + str(self.net.__class__.__name__) + "/")
        if not os.path.exists(
                "onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(time_time) + "/"):
            os.mkdir("onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(time_time) + "/")
        if not os.path.exists(
                "onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(time_time) + "/MUTANTS/"):
            os.mkdir("onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(time_time) + "/MUTANTS/")
        if not os.path.exists(
                "onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/ONNX/"):
            os.mkdir("onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(time_time) + "/ONNX/")
        if not os.path.exists(
                "onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(method) + "/" + str(self.time_time) + "/ONNX/"):
            os.makedirs("onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(method) + "/" + str(time_time) + "/ONNX/")
        


    def mutate(self, Mutate_time, Mutate_Batch_size):

        shapes = shape_cargo["vgg11"]

        TRAIN_FLAG = False
        HOOK_FLAG = True
        INDEX = -1000
        # model_list = []

        if args.LOG_FLAG:
            with open(args.LOG_PATH, 'r', encoding='utf-8') as file:
                log_dict = json.load(file)
            # pprint(log_dict)
        else:
            log_dict = {}

        # distance_list = []

        dtypes = [torch.float32 for _ in shapes] if self.net.__class__.__name__ not in nlp_cargo \
            else [torch.int32 for _ in shapes]
    
        data_dir = path_cargo["vgg11"]

        image_size = size_cargo["vgg11"]

        f = open("onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/MUTATE_LOG_"
                 + str(platform.system()) + "_" + str(device).replace(':', '_') + ".txt", "w")
        
        # print("f", f.name)

        if isinstance(image_size, list):

            path = path_cargo["vgg11"]
            data_0 = np.load("./data.npy")

            print("111111111111")
            seed_model = shape_inference.infer_shapes(self.net)
            print("111111111111")
            model = copy.deepcopy(seed_model)

            all_edges = onnx_mutation.edge_node.convert_onnx_to_edge(model.graph)


            methods[self.method](model, all_edges, args.log_flag, log_dict, self.net, self.old_net, self.time_time, f, Mutate_time, data_0)

        elif self.net.__class__.__name__ == "FastText_torch":

            data0, data1 = create_dataset(self.net, data_dir, image_size, Mutate_Batch_size)

            np_data = [data0.cpu().numpy(), data1.cpu().numpy()]

            torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, device)

            result, global_layer_info = torchinfoplus.summary(
                model=self.net,
                input_data=torch_data,
                # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
                dtypes=dtypes,
                col_names=['input_size', 'output_size', 'name'], depth=8,
                verbose=1)
            
            count = 0

            for _ in self.net.modules():
                count += 1

            print("Depth of the network is:", count - 1)  # subtract 1 to not count the network itself as a module
            # print("result", result)
            # exit(0)

            print("Mutate method: ", self.method.upper())
            # print(symbolic_traced.code)
            # print(net)

            # methods[self.method](symbolic_traced, args.log_flag, log_dict, self.net, self.old_net, self.time_time, f, data0, data1, times = Mutate_time)
        elif self.net.__class__.__name__ == "TextCNN":
            data0= create_dataset(self.net, data_dir, image_size, Mutate_Batch_size)

            np_data = [data0.cpu().numpy()]

            torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, device)

            result, global_layer_info = torchinfoplus.summary(
                model=self.net,
                input_data=torch_data,
                # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
                dtypes=dtypes,
                col_names=['input_size', 'output_size', 'name'], depth=8,
                verbose=1)
            
            count = 0

            for _ in self.net.modules():
                count += 1
            print("Depth of the network is:", count - 1)  # subtract 1 to not count the network itself as a module
            # print("result", result)
            # exit(0)
            # methods[self.method](symbolic_traced, args.log_flag, log_dict, self.net, self.old_net, self.time_time, f, data0, times = Mutate_time)
        else:
            raise NotImplementedError("dataset is not implemented")
        
        # print(args.distance_list)
        distance_np: ndarray = np.array(args.distance_list)
        # print("distance_np", distance_np)
        b = np.argsort(-distance_np)
        try:
            print("maximum model index is", b[0] if INDEX == -1000 else INDEX)
            print("maximum model distance is", distance_np[b[0] if INDEX == -1000 else INDEX])
            max_model = args.model_list[b[0] if INDEX == -1000 else INDEX]
            # print("max_model def __init__(self)\n", max_model)
        except IndexError as e:
            print("IndexError", e)
            max_model = self.old_net
        max_model.eval()
        print("max_model", max_model)
        # traced_model = torch.jit.trace(max_model, data0)
        if not os.path.exists("onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/ONNX/"):
            os.mkdir("onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/ONNX/")
        try:
            self.net.eval()
            # torch.onnx.export(self.net, data0,
            #                   f="torch_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/ONNX/"
            #                     + str(self.net.__class__.__name__) + "_" + 'seed' + ".onnx", verbose=False)
            
            # torch.onnx.export(max_model, data0,
            #                   f="torch_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/ONNX/"
            #                     + str(self.net.__class__.__name__) + "_" + str(b[0]) + ".onnx", verbose=False)
        except RuntimeError as e:
            print("RuntimeError", e)
        # if TRAIN_FLAG:
        #     if net.__class__.__name__ in train_cargo:
        #         print("===================================")
        #         print("start train and eval", net.__class__.__name__)
        #         train_cargo[net.__class__.__name__](old_net, max_model, data_dir, batch_size=batch_size)
        #     else:
        #         print(net.__class__.__name__, "is not in train_cargo")
        banned_hook_list = []
        if HOOK_FLAG and self.net.__class__.__name__ not in banned_hook_list:
            np_data = [np.ones(shape) for shape in shapes]
            input_data_dict_old, output_data_dict_old = info_com(self.old_net, np_data, dtypes, verbose=0)
            input_data_dict_new, output_data_dict_new = info_com(max_model, np_data, dtypes, verbose=0)
            filename1 = str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/" + str(platform.system()) + "_" + str(
                self.net.__class__.__name__) + "_" \
                        + self.method + str(self.time_time) + "_input"
            filename2 = str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/" + str(platform.system()) + "_" + str(
                self.net.__class__.__name__) + "_" \
                        + self.method + "_output"
            if not os.path.exists(os.path.join("torch_mutated_net")):
                os.mkdir(os.path.join("torch_mutated_net"))
            if not os.path.exists(os.path.join("torch_mutated_net", str(self.net.__class__.__name__))):
                os.mkdir(os.path.join("torch_mutated_net", str(self.net.__class__.__name__)))
            filename1 = os.path.join("torch_mutated_net", filename1)
            filename2 = os.path.join("torch_mutated_net", filename2)
            print('31564655545')
            print(filename1)
            filename1 = filename1 + "_" + str(device).replace(':', '_') + "_" + str(Mutate_time) + "times" + ".txt"
            filename2 = filename2 + "_" + str(device).replace(':', '_') + "_" + str(Mutate_time) + "times" + ".txt"
            save_layer_info(input_data_dict_old, input_data_dict_new, Mutate_time, filename1)
            save_layer_info(output_data_dict_old, output_data_dict_new, Mutate_time, filename2)
            compare_layer(output_data_dict_old, output_data_dict_new)
        f.close()
        # lp.print_stats()
        print("Mutate&Test finished at", time.asctime(time.localtime(time.time())))

    def net_to_onnx(self, frame_name, data0):

        if frame_name == 'pytorch':
            torch.onnx.export(self.net, data0,
                                f="onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/ONNX/"
                                + str(self.net.__class__.__name__) + "_" + 'seed' + ".onnx", input_names= ['input'], verbose=False)
            model_path = "onnx_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/ONNX/"+ str(self.net.__class__.__name__) + "_" + 'seed' + ".onnx"
            model = onnx.load(model_path)
        if frame_name == 'mindspore':
            ms.export(self.net, data0, file_name="mindspore_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/ONNX/"
                                + str(self.net.__class__.__name__) + "_" + 'seed' + ".onnx", file_format='ONNX')
            
            model_path = "mindspore_mutated_net/" + str(self.net.__class__.__name__) + "/" + str(self.time_time) + "/ONNX/"+ str(self.net.__class__.__name__) + "_" + 'seed' + ".onnx"
            model = onnx.load(model_path)   
        return model