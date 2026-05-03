import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mindspore_mutation.generate_models.run_random_ms import run_random_ms
from mindspore_mutation.generate_models.run_mcmc_ms import run_mcmc_ms
from mindspore_mutation.generate_models.run_log_ms import run_log_ms
from mindspore_mutation.generate_models.run_q_ms import run_q_ms
# from detect_bugs import detect_bugs
import time
import yaml
import argparse


# 图像分类：1."vgg11"  2."vgg16"  3."vgg19"   4."resnet"
# 目标检测：5."yolov3"  6."openpose"  7."SSDresnet50fpn" 8."SSDmobilenetv1"
# 语义分割： 9."UNetMedical"  10."DeepLabV3"
# 文本分类： 11."LSTM"(弃用)  12."textcnn"  13."FastText"
# 异常检测：  14."patchcore"  15."ssimae"
import sys
from mindspore_mutation.cargo import net_cargo
import mindspore

mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")

class Logger(object):
    def __init__(self, filename="print_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)  # 输出到控制台
        self.log.write(message)       # 输出到文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()



if __name__ == '__main__':
    



    basic_dir = "/home/cvgroup/myz/czx/semtest-gitee/modelmeta"
    config_path_main = "/home/cvgroup/myz/czx/semtest-gitee/modelmeta/configs/main.yaml"
    with open(config_path_main, 'r', encoding="utf-8") as f:
        main_config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    main_config = main_config_yaml['config']




    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    sys.stdout = Logger(f"/home/cvgroup/myz/czx/SemTest_master/print/print_log_{current_time}.txt")

    # config_path = basic_dir + '/configs/rq3_exp1.yaml'
    config_path = main_config["yaml_path"]
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    execution_config = config['execution_config']
    train_config = config['train_config']

    # # 设置参数 
    seed_model = execution_config['seed_model'] 
    # resnet yolov3 yolov4 mobilenetv2 openpose crnn unet DeepLabV3 patchcore SSDresnet50fpn SSDmobilenetv1
    print(seed_model)
    mutate_times = int(execution_config['mutate_times'])  # 变异次数
    ifapimut = execution_config['ifapimut']
    num_samples = int(execution_config['num_samples'])  # 随机选择几个数据
    num_quantiles = execution_config['num_quantiles']
    print(type(execution_config['MR']))
    if type(execution_config['MR']) == int:
        MR = [execution_config['MR']]
    else:
        MR_str = execution_config['MR'].split(",")
        MR = [int(val) for val in MR_str]
    device = int(execution_config['device'])
    run_option = int(execution_config['run_option'])  # 0代表随机选择，1代表通过MCMC选择，2代表Q网络方法，3代表根据日志对模型detect_bugs
    ifeplison = float(execution_config['ifeplison'])
    ifTompson = execution_config['ifTompson']

    path_flag = False


    MR_structures_map = {0: "UOC", 1: "PIOC", 2: "ABSOC_A", 3: "ABSOC_B"}

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    sstart = time.time()

    if run_option == 0:  # 随机选择
        run_random_ms(seed_model, mutate_times, num_samples, MR, ifapimut, ifTompson,device,train_config)
        pass
    elif run_option == 1:  # MCMC选择
        run_mcmc_ms(seed_model, mutate_times, num_samples, MR, ifapimut, ifTompson,device,train_config)
        pass
    elif run_option == 2:  # Q网络方法
        run_q_ms(seed_model, mutate_times, num_samples, MR, ifapimut, num_quantiles, ifeplison, ifTompson,device,train_config,main_config["csv_path"])
        # test(seed_model, mutate_times, num_samples)
    elif run_option == 3:  # 根据日志对模型detect_bugs
        #$ run_log_ms(seed_model, mutate_times, log_path, num_samples, data_x_path, data_y_path, path_flag)
        pass





    
