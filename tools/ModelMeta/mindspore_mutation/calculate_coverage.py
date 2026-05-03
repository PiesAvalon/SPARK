
import mindspore_mutation.msmodel2json as msmodel2json
from mindspore_mutation.msmodel2json import *
import torch
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")

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

def model2cov(model, input, dtype, file_path_1, all_json_path, api_config_pool_path, folder_path):
    model_json_1, inside, output_datas = ms_model2json(model, input, dtype)
    with open(file_path_1, 'w', encoding='utf-8') as file:
        json.dump(model_json_1, file, ensure_ascii=False, indent=4)
    file.close()

    cal_cov = CoverageCalculatornew(all_json_path, api_config_pool_path)
    cal_cov.load_json(file_path_1)
    # print(f'The inside Coverage Is: {inside}')
    input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov = cal_cov.cal_coverage()
    return input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside, output_datas


def json2cov(file_path_1, all_json_path, api_config_pool_path):
    cal_cov = CoverageCalculatornew(all_json_path, api_config_pool_path)
    cal_cov.load_json(file_path_1)
    # print(f'The inside Coverage Is: {inside}')
    input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov = cal_cov.cal_coverage()
    return input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov



def find_layer_type(new_net):
    layer_type_list = []
    layers = new_net.cells_and_names()
    for i, j in layers:
        if i != '':
            if not has_child_node(new_net, i):
                if j.__class__.__name__ not in layer_type_list:
                    layer_type_list.append(j.__class__.__name__)
    return layer_type_list

    
# @profile
# def model2cov(model,input,dtype,d_file_path,api_config_pool_path):
#     model_json_1 = torch_model2json(model, input, dtype) # 本次变异的模型
#     os.makedirs(os.path.dirname(d_file_path), exist_ok=True)
#     with open(d_file_path, 'w', encoding='utf-8') as file:
#         json.dump(model_json_1, file,cls=CustomEncoder, ensure_ascii=False, indent=4)
#         # json.dump(model_json_1, file,  ensure_ascii=False, indent=4)

#     # d_file_path="/root/zgb/SemTest24.0/pytorch/results/DeepLabV3/2024_10_03_15_48_19/model_json/DeepLabV3-UOC1.json"
#     cal_cov = CoverageCalculatornew(api_config_pool_path)
#     cal_cov.load_json(d_file_path)
#     input_cov,config_cov,api_cov= cal_cov.cal_coverage()

#     # 删除相关变量
#     # del model, input, dtype,cal_cov
#     del model_json_1,model, input, dtype,cal_cov
#     # 手动调用垃圾回收
#     gc.collect()
#     return input_cov,config_cov,api_cov