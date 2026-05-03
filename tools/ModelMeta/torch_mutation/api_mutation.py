import numpy as np
from torch_mutation.cargo import *
import torch
import copy
import time
import json
import torch.fx as fx
# from MR_structure import *
from cargo import match_rule,reflect_name,MCMC,rule_reflect_class
import config

device=config.device
rules_dict = config.rules_dict # 对于每一个类型的算子，有哪些规则适用

def api_mutation(d,log_dict,n,LOG_FLAG):
    graph=d.graph
    if LOG_FLAG == False:
        option_layers = list()  # 可被选择的算子列表（形式：(node,instance,name)）
        for node in graph.nodes:
            if node.op == 'call_module' and '_mutate' not in node.name:  # 如果这个算子是module类型的且不是被变异过的算子，就加入列表
                module_name = node.target
                module_instance = d.get_submodule(module_name)  # 获取模块实例
                option_layers.append((node, module_instance, node.name))
        # print(option_layers)
        # print(len(option_layers))

        if len(option_layers) != 0:
            option_node, option_instance, option_name = random.choice(
                option_layers)  # 随机从option_layers列表中选1个被变异的算子
            option_rule = random.choice(rules_dict[type(option_instance)])  # 根据rules_dict随机选择适用于该算子的规则
            new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

            print('选择对其变异的node：', option_node)
            print('选择对其变异的node的名字：', option_name)
            print('选择对其变异的instance：', option_instance)
            print('选择的变异规则：', option_rule)
            print('变异后新的层：', new_instance)

            log_dict[n]['seedmodel_api_name'] = option_name
            if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                log_dict[n]['seedmodel_api_rule'] = option_rule.__name__[-6:]
            else:
                log_dict[n]['seedmodel_api_rule'] = option_rule.__name__[-5:]
            # print(log_dict[n])
    else:  # 根据日志变异
        option_name = log_dict[str(n)]['seedmodel_api_name']
        option_rule_name = log_dict[str(n)]['seedmodel_api_rule']

        for node in graph.nodes:
            if node.name == option_name:
                module_name = node.target
                module_instance = d.get_submodule(module_name)  # 获取模块实例
                option_node, option_instance, option_name = node, module_instance, node.name
                break

        option_rule = match_rule(option_rule_name)
        new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

        # print('选择对其变异的node：', option_node)
        # print('选择对其变异的node的名字：', option_name)
        # print('选择对其变异的instance：', option_instance)
        # print('选择的变异规则：', option_rule)
        # print('变异后新的层：', new_instance)

    
    # print(graph)  # delete

    # 新层名字
    new_name = reflect_name(option_name, option_rule)

    # 将新层绑定到符号追踪的模块中
    d.add_module(new_name, new_instance)
    # 插入新的节点并替换旧的节点
    with option_node.graph.inserting_after(option_node):
        new_node = option_node.graph.call_module(new_name, args=option_node.args)
        option_node.replace_all_uses_with(new_node)
        d.graph.erase_node(option_node)
    # 检查并重新编译
    graph.lint()
    d.recompile()
    print('finished api mutation')
    # return d,




