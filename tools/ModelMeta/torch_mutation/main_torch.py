import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
from torch_mutation.generate_models.run_random_torch import run_random_torch
from torch_mutation.generate_models.run_mcmc_torch import run_mcmc_torch
from torch_mutation.generate_models.run_q_torch import run_q_torch
from torch_mutation.generate_models.run_log_torch import run_log_torch
import time


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
    # seed_model ="resnet" # 选择模型
    log_path ="/home/cvgroup/myz/czx/semtest-gitee/modelmeta/log/unet/TORCH_LOG_DICT_gpu.json" #变异日志
    
    # mutate_times = 100 # 变异次数
    # num_samples = 1  # 随机选择几个数据
    # data_x_path='' #输入的x
    # data_y_path='' #输入的y，这两个用于缺陷检测：detect_bugs.py
    # path_flag=False #用于缺陷检测：detect_bugs.py numpy数据是否用选定的
    base_path = log_path.replace("TORCH_LOG_DICT_gpu.json","")
    # run_option = 2 # 0代表随机选择，1代表通过MCMC选择，2代表Q网络方法，3代表根据日志对模型detect_bugs

    basic_dir = "/home/cvgroup/myz/czx/semtest-gitee/modelmeta"
    # config_path_main = basic_dir + '/configs/main.yaml'
    # with open(config_path_main, 'r', encoding="utf-8") as f:
    #     main_config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    # main_config = main_config_yaml['config']
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    sys.stdout = Logger(f"/home/cvgroup/myz/czx/SemTest_master/print/print_log_{current_time}.txt")
    # config_path = basic_dir + '/configs/rq3_exp1.yaml'
    config_path = "/home/cvgroup/myz/czx/semtest-gitee/modelmeta/configs/rq3_MRs/unet_False.yaml"
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    execution_config = config['execution_config']
    train_config = config['train_config']
    # # 设置参数 
    seed_model = execution_config['seed_model'] 
    # resnet yolov3 yolov4 mobilenetv2 openpose crnn unet DeepLabV3 patchcore SSDresnet50fpn SSDmobilenetv1
    print(seed_model)
    mutate_times = int(execution_config['mutate_times'])  # 变异次数
    mutate_times = 20
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
    run_option = 3 # int(execution_config['run_option'])  # 0代表随机选择，1代表通过MCMC选择，2代表Q网络方法，3代表根据日志对模型detect_bugs
    ifeplison = float(execution_config['ifeplison'])
    ifTompson = execution_config['ifTompson']
    path_flag = False
    MR_structures_map = {0: "UOC", 1: "PIOC", 2: "ABSOC_A", 3: "ABSOC_B"}


    if run_option == 0: # 随机选择
        run_random_torch(seed_model, mutate_times,num_samples)
    elif run_option == 1: # MCMC选择
        run_mcmc_torch(seed_model, mutate_times,num_samples)
    elif run_option == 2: # Q网络方法
        run_q_torch(seed_model, mutate_times,num_samples, MR, ifapimut, num_quantiles, ifeplison, ifTompson,device,train_config,main_config["csv_path"])
    elif run_option == 3:  # 根据日志对模型detect_bugs
        run_log_torch(seed_model, mutate_times,log_path, num_samples,base_path,train_config,execution_config)

