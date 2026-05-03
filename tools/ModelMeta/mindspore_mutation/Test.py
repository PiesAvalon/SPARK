


from mindspore_mutation.msmodel2json import CoverageCalculatornew
from mindspore_mutation.cargo import get_model
from mindspore_mutation.calculate_coverage import model2cov,find_layer_type,json2cov
import mindspore
import numpy as np
import os
import json

seed_model = "vgg16"
all_json_path = "/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/ms_all_layer_info.json"
api_config_pool_path = '/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/mindspore_api_config_pool.json'


cal_cov = CoverageCalculatornew(all_json_path, api_config_pool_path)

seed_model_net = get_model(seed_model)
input_data = [mindspore.Tensor(np.random.randn(1,3,224,224),mindspore.float32)]
dtypes = [mindspore.float32]

json_file_path = os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results/", f"{seed_model}.json")
os.makedirs(json_file_path, exist_ok=True)

json_file_path = os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results", seed_model, "model_json1"+".json")
os.makedirs(os.path.join("/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results", seed_model,"model_json"), exist_ok=True)

all_json_path = "/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/ms_all_layer_info.json"
api_config_pool_path = '/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/mindspore_api_config_pool.json'
folder_path = '/home/cvgroup/myz/czx/semtest-gitee/'

input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside, output_datas = model2cov(seed_model_net,
                                                                                                            input_data,
                                                                                                            dtypes,
                                                                                                            json_file_path,
                                                                                                            all_json_path,
                                                                                                            api_config_pool_path,
                                                                                                            folder_path)
from mindspore_mutation.msmodel2json import *




model_json_1, inside, output_datas = ms_model2json(seed_model_net, input_data, dtypes)

with open(json_file_path, 'w', encoding='utf-8') as file:
    json.dump(model_json_1, file, ensure_ascii=False, indent=4)
file.close()

cal_cov = CoverageCalculatornew(all_json_path, api_config_pool_path)
cal_cov.load_json(json_file_path)





print("input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside: ", input_cov, config_cov,
              api_cov, op_type_cov, op_num_cov, edge_cov, inside)









