import torch
from infoplus.TorchInfoPlus import torchinfoplus
import gc
import json
from union import union_json
from Coverage import CoverageCalculatornew
import os
# from memory_profiler import profile



def get_module_config(module):
    config_params = {}
    for attr_name, attr_value in module.__dict__.items():
        if not attr_name.startswith('_'):
            if isinstance(attr_value, torch.dtype):
                attr_value = str(attr_value)
            config_params[attr_name] = attr_value
    return config_params


def traverse_network(net, layer_config):

    for name, sub_module in net.named_children():

        layer_type_name = type(sub_module).__name__
        if layer_type_name not in layer_config.keys():
            layer_config[layer_type_name] = []

        config_params = get_module_config(sub_module)
        layer_config[layer_type_name].append(config_params)

        if isinstance(sub_module, torch.nn.Module):
            traverse_network(sub_module, layer_config)
    # print(layer_config)
    for key, value in layer_config.items():
        layer_config[key] = [dict(t) for t in {tuple(d.items()) for d in layer_config[key]}]

    return layer_config

# @profile
def torch_model2json(model, input_tensor, input_dtypes):
    with torch.no_grad():
        result, global_layer_info, summary_list = torchinfoplus.summary( # 占用内存大
            model=model,
            input_data=input_tensor,
            dtypes=input_dtypes,
            col_names=['input_size', 'output_size', 'name'], depth=8,
            verbose=1)

        # import sys
        # print(f"变量result占用的内存大小为: {sys.getsizeof(result)} 字节")
        # print(f"变量global_layer_info占用的内存大小为: {sys.getsizeof(global_layer_info)} 字节")
        # print(f"变量summary_list占用的内存大小为: {sys.getsizeof(summary_list)} 字节")

        # print('(((((((((((((((((((((((((')
        # print('result:')
        # print(result)
        # print('global_layer_info:')
        # print(global_layer_info)
        # print('summary_list:')
        # print(summary_list)
        # print('(((((((((((((((((((((((((')

        model_json = {}
        edge_list_list = []
        cur_edge_num = 0
        for index in range(len(summary_list) - 1):
            if int(summary_list[index].depth) >= 1:
                edge_list = []
                input_type = summary_list[index].class_name
                output_type = summary_list[index + 1].class_name
                # output_name = ooo.name
                edge_list.append(input_type)
                edge_list.append(output_type)
                cur_edge_num += 1
                if edge_list not in edge_list_list:
                    edge_list_list.append(edge_list)
        model_json["edges"] = edge_list_list
        layer_config = {}
        layer_config = traverse_network(model, layer_config)
        model_json["layer_config"] = layer_config

        layer_input_info = {}
        for layer_info in summary_list:
            layer_input_info_dist = {}
            if int(layer_info.depth) >= 1 and layer_info.input_size != []:
                input_name = layer_info.class_name
                # if input_name not in layer_input_info.keys():
                output_name = layer_info.name
                # print(layer_info.input_size)
                input_size = layer_info.input_size[0]
                output_size = layer_info.output_size
                len_input_size = len(input_size)
                len_output_size = len(output_size)
                if input_name not in layer_input_info.keys():
                    layer_input_info_dist["input_dims"] = [len_input_size]
                    layer_input_info_dist["dtype"] = [str(input_dtypes)]
                    layer_input_info_dist["shape"] = [str(input_size)]
                    layer_input_info[input_name] = layer_input_info_dist
                else:
                    if len_input_size not in layer_input_info[input_name]["input_dims"]:
                        layer_input_info[input_name]["input_dims"].append(len_input_size)
                    if str(input_dtypes) not in layer_input_info[input_name]["dtype"]:
                        layer_input_info[input_name]["dtype"].append(str(input_dtypes))
                    if str(input_size) not in layer_input_info[input_name]["shape"]:
                        layer_input_info[input_name]["shape"].append(str(input_size))
        model_json["layer_input_info"] = layer_input_info

        current_layer_num = 0
        layer_type_list = []
        for ooo in summary_list:
            if int(ooo.depth) >= 1:
                input_name = ooo.class_name
                if input_name not in layer_type_list:
                    layer_type_list.append(input_name)
                current_layer_num += 1
        model_json["layer_num"] = current_layer_num
        model_json["layer_type"] = layer_type_list

        layer_dims = {}
        for layer_info in summary_list:
            if int(layer_info.depth) >= 1 and layer_info.input_size != []:
                input_name = layer_info.class_name
                # output_name = layer_info.name
                input_size = layer_info.input_size[0]
                output_size = layer_info.output_size
                len_input_size = len(input_size)
                len_output_size = len(output_size)
                if input_name not in layer_dims:
                    layer_dims[input_name] = {
                        "input_dims": [len_input_size],
                        "output_dims": [len_output_size]
                    }
                else:
                    if len_input_size not in layer_dims[input_name]["input_dims"]:
                        layer_dims[input_name]["input_dims"].append(len_input_size)
                    if len_output_size not in layer_dims[input_name]["output_dims"]:
                        layer_dims[input_name]["output_dims"].append(len_output_size)
        # cur_edge_num
        model_json["cur_edge_num"] = cur_edge_num
        model_json["layer_dims"] = layer_dims

        # print(model_json)
        return model_json

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # 转换张量为列表
        elif isinstance(obj, torch.nn.Module):
            # 如果是模型或层的实例，返回一个标识字符串
            return "Instance of {}".format(obj.__class__.__name__)
        # 增加一个通用的后备处理
        try:
            return super().default(obj)  # 尝试默认的处理
        except TypeError:
            return str(obj)  # 作为最后的手段，转换为字符串

# @profile
def model2cov(model,input,dtype,d_file_path,api_config_pool_path):
    model_json_1 = torch_model2json(model, input, dtype) # 本次变异的模型
    os.makedirs(os.path.dirname(d_file_path), exist_ok=True)
    with open(d_file_path, 'w', encoding='utf-8') as file:
        json.dump(model_json_1, file,cls=CustomEncoder, ensure_ascii=False, indent=4)
        # json.dump(model_json_1, file,  ensure_ascii=False, indent=4)

    # d_file_path="/root/zgb/SemTest24.0/pytorch/results/DeepLabV3/2024_10_03_15_48_19/model_json/DeepLabV3-UOC1.json"
    cal_cov = CoverageCalculatornew(api_config_pool_path)
    cal_cov.load_json(d_file_path)
    input_cov,config_cov,api_cov= cal_cov.cal_coverage()

    # 删除相关变量
    # del model, input, dtype,cal_cov
    del model_json_1,model, input, dtype,cal_cov
    # 手动调用垃圾回收
    gc.collect()
    return input_cov,config_cov,api_cov




### 以下为测试部分
import torch.nn as nn
import torch.fx as fx
class vgg11(nn.Module):
    def __init__(self):
        super(vgg11, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) #所有卷积层都一致
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1) #一致
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1) #一致
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1) #一致
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.averagepool = nn.AvgPool2d(kernel_size=1, stride=1)#一致
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)#不一致，高宽都减半
        self.flatten = torch.nn.Flatten()

    def forward(self, inputs):
        x = self.relu1(self.conv1(inputs))
        x = self.maxpool(x)
        x = self.relu1(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu1(self.conv3(x))
        x = self.relu1(self.conv4(x))
        x = self.maxpool(x)
        x = self.relu1(self.conv5(x))
        x = self.relu1(self.conv6(x))
        x = self.maxpool(x)
        x = self.relu1(self.conv7(x))
        x = self.relu1(self.conv8(x))
        x = self.maxpool(x)
        x = self.averagepool(x)
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu1(self.fc2(x))
        x = self.fc3(x)
        return x

selected_MR_structure_name ="UOC"
from torch_mutation.MR_structure import *
MR_structures_map = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B}

if __name__ == '__main__':
    with torch.no_grad():
        net = vgg11()
        a = torch.randn(5, 3, 32, 32)
        d=fx.symbolic_trace(net)
        # print(d)

        print('1111111111111111111111')
        model_json_1 = torch_model2json(d, a, [torch.float32])  # 本次变异的模型
        print('1111111111111111111111')
        with torch.no_grad():
            graph = d.graph
            nodelist = []
            for node in graph.nodes:
                if node.op in ['call_module', 'root'] or \
                        (node.op == "call_function" and any(
                            substring in node.name for substring in ['uoc', 'pioc', 'absoc_a', 'absoc_b'])):
                    nodelist.append(node)
            # print(nodelist)

            try:
                add_module = MR_structures_map[selected_MR_structure_name]('conv')
            except Exception as e:
                exit(e)

            aa = nodelist[6]
            bb = nodelist[13]
            cc = nodelist[3]
            dd = nodelist[22]
            print(aa,bb,cc,dd) # conv3 conv6 conv2 flatten


            if selected_MR_structure_name == "PIOC":
                with cc.graph.inserting_after(cc):
                    new_hybrid_node = cc.graph.call_function(add_module, args=(cc, cc, cc))
                    cc.replace_all_uses_with(new_hybrid_node)
                    new_hybrid_node.update_arg(0, aa)
                    new_hybrid_node.update_arg(1, bb)
                    new_hybrid_node.update_arg(2, cc)
            else:  # selected_MR_structure_name != "PIOC"
                with dd.graph.inserting_after(dd):
                    new_hybrid_node = dd.graph.call_function(add_module, args=(dd, dd, dd, dd))
                    dd.replace_all_uses_with(new_hybrid_node)
                    new_hybrid_node.update_arg(0, aa)
                    new_hybrid_node.update_arg(1, bb)
                    new_hybrid_node.update_arg(2, cc)
                    new_hybrid_node.update_arg(3, dd)
            graph.lint()  # 检查是否有图错误并重新编译图
            d.recompile()

        print('2222222222222222')
        print(d)
        model_json_1 = torch_model2json(d, a, [torch.float32])  # 本次变异的模型
        print('2222222222222222')
        # print(d)
        # torch.onnx.export(d, a, r"D:\张广倍大学\张广倍科研\南大沐燕舟\画MR图\导出onnx\PIOC.onnx", verbose=True, opset_version=11)

