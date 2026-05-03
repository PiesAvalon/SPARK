import numpy as np
import mindspore.nn as nn
from mindspore.rewrite import ScopedValue, NodeType
from mindspore.rewrite.node import Node, NodeManager
from mindspore.rewrite import SymbolTree, Node, NodeType
from mindspore_mutation.cargo import *
import copy
import time
import json
import mindspore_mutation.config as ms_config
import collections
import mindspore
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")

rules_dict = ms_config.rules_dict # 对于每一个类型的算子，有哪些规则适用

def has_child_node(net, node_name):
    layers = net.cells_and_names()
    parent_node = None
    for name, _ in layers:
        if name == node_name:
            parent_node = name
            continue
        if parent_node is not None and name.startswith(parent_node + '.'):
            return True
    return False


MR_structure_name_list = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']


def api_mutation(net, option_layers, option_index, log_dict, n, LOG_FLAG):
    if LOG_FLAG == False:

        # print([node.get_name() for node in d.nodes()])
        # option_name,option_instance,option_instance_type,option_node_type=random.choice(option_layers)
        available_indices = [i for i in range(len(option_layers)) if i not in option_index]

        # 检查 available_indices 是否为空，确保可以进行随机选择
        random_index = random.choice(available_indices)
        
        
        # random_index = random.randrange(len(option_layers))
        
        # 获取对应的元素
        option_name, option_instance, option_instance_type, option_node_type = option_layers[random_index]
        # print('选择对其变异的node的名字：', option_name)
        # print('选择对其变异的type：', option_node_type)
        
        option_rule = random.choice(rules_dict[type(option_instance)])  # 根据rules_dict随机选择适用于该算子的规则
        # print('选择的变异规则：', option_rule)
        new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层
        
        
        # 以追加模式打开文件
        # with open('/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/results/example.txt', 'a', encoding='utf-8') as file:
        #     # 写入固定分隔符

        #     file.write('\n' + 'node:' + str(option_name) + '\n')  # 确保将 n 转换为字符串写入文件
        #     file.write('type:'+ str(option_node_type) + '\n')  # 确保将 n 转换为字符串写入文件
        #     # file.write('index:'+ str(random_index) + '\n')  # 确保将 n 转换为字符串写入文件
        #     file.write('relu:'+ str(option_rule) + '\n')  # 确保将 n 转换为字符串写入文件

        # print('选择对其变异的node的名字：', option_name)
        # print('选择对其变异的type：', option_node_type)
        # print('选择的变异规则：', option_rule)
        # print('变异后新的层：', new_instance)

        log_dict[n]['seedmodel_api_name'] = option_name
        if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
            log_dict[n]['seedmodel_api_rule'] = option_rule.__name__[-6:]
        else:
            log_dict[n]['seedmodel_api_rule'] = option_rule.__name__[-5:]

    else:  # 根据日志变异 to do,还没写
        option_name = log_dict[str(n)]['seedmodel_api_name']
        option_rule_name = log_dict[str(n)]['seedmodel_api_rule']
        option_instance = layer._cells.get(option_name, None)

        if option_instance is not None:
            option_rule = match_rule(option_rule_name)
            new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  # 根据规则变异过的新层
            net._cells[option_name] = new_instance
            net.update_parameters_name()
        return net, SymbolTree.create(net), log_dict, option_index

    # new_name = reflect_name(option_name, option_rule) 
    new_name = reflect_name(option_name, option_rule)
    net._cells[option_name] = new_instance
    net.update_parameters_name()
    i = 0
    for name, child in net.cells_and_names():
        if not has_child_node(net, name) and not name == '' and not 'deadcode' in str(type(child)):
            i += 1
            if name == option_name and i not in option_index:
                option_index.append(i)
                break
    # option_node = d.get_node(option_name)
    # new_node = option_node.create_call_cell(cell=new_instance, targets=[d.unique_name("x")], args=[ScopedValue.create_naming_value('x')],name=new_name)
    # d.replace(option_node, [new_node])
    # print("option_index",option_index)
    return net, SymbolTree.create(net), log_dict, option_index





