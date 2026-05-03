
import troubleshooter as ts
# seed = 20230808
# ts.widget.fix_random(seed)
# ts.fix_random(seed=seed)

import torch
import mindspore

from common.model_utils import get_model
import numpy as np
from utils.infoplus_compare import compare_models


# 0 为完全相等；负数为含有不同的keys；正数为keys相同，但是对应的值不完全相同
def compare_dict(dict1,dict2):
    cnt = 0
    dif_keys = []
    if (dict1.keys() ^ dict2.keys()):
        print('Two dictionaries have different keys!')
        dif_keys = list(dict1.keys() ^ dict2.keys())
        cnt = cnt - len(dif_keys)
        return cnt, dif_keys
    else:
        keys = list(dict1.keys())
        for key in keys:
            print(f"{key} distance: ",dict1[key]- dict2[key])



def ts_apicompare():

    #mindspore dump
    ts.migrator.api_dump_init(model_ms_origin, output_path="/data1/myz/empirical_exp/log/ms_dump")
    ts.migrator.api_dump_start()
    data = mindspore.Tensor(np.ones(input_size), mindspore.float32)
    print(model_ms_origin(data).shape)

    ts.migrator.api_dump_stop()

    #torch dump
    ts.migrator.api_dump_init(model_torch_origin, output_path="/data1/myz/empirical_exp/log/torch_dump")
    ts.migrator.api_dump_start()

    data = torch.tensor(np.ones(input_size),dtype=torch.float32).to(final_device)
    print(model_torch_origin(data).shape)

    ts.migrator.api_dump_stop()


    # origin_path与target_path为api_dump_init中的output_path
    origin_path = "/data1/myz/empirical_exp/log/ms_dump"
    target_path = "/data1/myz/empirical_exp/log/torch_dump"
    # 输出结果保存路径
    output_path = "/data1/myz/empirical_exp/log/compare_result"

    # 对比完成之后会生成ts_api_mapping.csv（API映射文件）、 ts_api_forward_compare.csv（正向比对结果）、ts_api_backward_compare.csv（反向比对结果）
    ts.migrator.api_dump_compare(origin_path, target_path, output_path)


if __name__ == '__main__':
    model_name = "wide_and_deep"
    device_target = "GPU"
    device_id = 0
    input_size = (16000, 39)

    if device_target == "GPU":
        final_device = "cuda:"+str(device_id)

    model_ms_origin1, model_torch_origin1 = get_model(model_name=model_name, input_size=input_size, scaned=True)
    data1 = [np.ones(input_size), np.ones(input_size)]
    data2 = [data1[0], data1[1]]
    ms_dtypes = [mindspore.int32]
    torch_dtypes = [torch.int32]

    data1_ms = mindspore.Tensor(data1[0], mindspore.int32)
    data2_ms = mindspore.Tensor(data1[1], mindspore.float32)
    out1 = model_ms_origin1(data1_ms, data2_ms)

    # data2 = torch.tensor(data2[0], dtype=torch.int32).cuda()
    # out2 = model_torch_origin1(data2)

    # # 5. 设置dump的网络
    # ts.migrator.api_dump_init(model_torch_origin1, output_path="./torch_dump", retain_backward=True)
    #
    # # 6. 在迭代开始时开启dump
    # ts.migrator.api_dump_start()
    #
    # # 7. 执行训练流程
    # out2 = model_torch_origin1(data2)
    #
    # # 8. 在反向计算结束，优化器更新前关闭dump
    # ts.migrator.api_dump_stop()
    #
    # # dis_dict1 = compare_models(model_ms_origin1, model_torch_origin1,np_data = data,ms_dtypes=ms_dtypes,torch_dtypes=torch_dtypes)
    # # for key in dis_dict1.keys():
    # #     print(f"{key}: ",dis_dict1[key])





