import yaml
import mindspore.nn as nn
from mindspore_mutation.rules_ms import rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, \
                                        rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18
basic_dir = "/home/cvgroup/myz/czx/semtest-gitee/modelmeta/"
config_path = basic_dir + 'configs/rq3_exp1.yaml'
with open(config_path.lower(), 'r', encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
train_config = config['execution_config']


dev = int(train_config['device'])
if dev == -1:
    pt_device = "cpu"
    ms_device = "cpu"
else:
    pt_device = "cuda:" + str(dev)
    ms_device = "gpu"



rules_dict = {
    # nn.Conv2d: [rule1, rule2, rule3, rule5, rule6, rule7, rule8],  # 图像分类
    nn.Conv2d: [rule1, rule3, rule5, rule6, rule7, rule8],  # 非图像分类
    nn.AvgPool2d: [rule1, rule3, rule12, rule13, rule14],
    # nn.GlobalAvgPooling: [rule1, rule3, rule12, rule13, rule14],
    nn.MaxPool2d: [rule1, rule3, rule12, rule13, rule14],
    nn.ReLU: [rule1, rule15],
    nn.ReLU6: [rule1],
    nn.BatchNorm2d: [rule1, rule4, rule9, rule10, rule11],
    nn.Dense: [rule1],
    nn.Flatten: [rule1],
    nn.HSigmoid: [rule1],
    nn.Sigmoid: [rule16, rule1],
    nn.Softmax: [rule17, rule1],
    nn.Tanh: [rule18, rule1],
    
    nn.Conv2dTranspose: [rule1],
    nn.LeakyReLU: [rule1, rule15],
    nn.AdaptiveAvgPool2d: [rule1, rule12, rule13, rule14],
    nn.Dropout: [rule1],
    nn.Embedding: [rule1],
    nn.LSTM: [rule1]
}