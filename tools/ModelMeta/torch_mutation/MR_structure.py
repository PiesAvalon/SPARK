import numpy as np
import torch
from torch import nn, fx
from torch_mutation.deadcode import SELayer, Inception_A, DenseLayer, DropPath, Dense, ResidualBlock, PWDWPW_ResidualBlock
from torch_mutation.handel_shape import handle_shape_final, handle_shape_strict, make_unsqueeze, make_reduce
from cargo import match_rule,reflect_name,rule_reflect_class
import random
import config

device=config.device
rules_dict = config.rules_dict # 对于每一个类型的算子，有哪些规则适用

# # rule1:torch.nn.Conv2d,torch.nn.AvgPool2d,torch.nn.MaxPool2d,torch.nn.ReLU,torch.nn.ReLU6,torch.nn.BatchNorm2d,torch.nn.Linear,torch.nn.Flatten
# torch.nn.Hardsigmoid...所有
# rule2:torch.nn.Conv2d
# # rule3:torch.nn.Conv2d,torch.nn.AvgPool2d,torch.nn.MaxPool2d
# rule4: torch.nn.BatchNorm2d
# rule5:torch.nn.Conv2d
# rule6:torch.nn.Conv2d
# rule7:torch.nn.Conv2d
# rule8:torch.nn.Conv2d
# rule9:torch.nn.BatchNorm2d
# rule10:torch.nn.BatchNorm2d
# rule11:torch.nn.BatchNorm2d
# # rule12:torch.nn.AvgPool2d,torch.nn.MaxPool2d,torch.nn.AdaptiveAvgPool2d
# # rule13:torch.nn.AvgPool2d,torch.nn.MaxPool2d,torch.nn.AdaptiveAvgPool2d
# # rule14:torch.nn.AvgPool2d,torch.nn.MaxPool2d,torch.nn.AdaptiveAvgPool2d
# # rule15:torch.nn.ReLU,torch.nn.LeakyReLU
# rule16:torch.nn.Sigmoid
# rule17:torch.nn.Softmax
# rule18:torch.nn.Tanh


class UOC(nn.Module):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "UOC"
        self.op_type = op_type
        if self.op_type == "SELayer":
            layer = SELayer()
        if self.op_type == "DenseLayer":
            layer = DenseLayer()
        if self.op_type == "Inception_A":
            layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            layer = ResidualBlock()
        if self.op_type == "DropPath":
            layer = DropPath()
        if self.op_type == "Dense":
            layer = Dense()

        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(layer, log_dict, i, LOG_FLAG)  # 对层进行 torch.fx 的更改
        else:
            self.op_layer = layer

    def modify_layer_with_fx(self, layer,log_dict,i,LOG_FLAG):
        new_module=torch.fx.symbolic_trace(layer)
        graph=new_module.graph

        if LOG_FLAG == False:
            option_layers = list()  # 可被选择的算子列表（形式：(node,instance,name)）
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  # 如果这个算子是module类型的且不是被变异过的算子，就加入列表
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_layers.append((node, module_instance, node.name))
            # print(option_layers)

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  # 随机从option_layers列表中选1个被变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据rules_dict随机选择适用于该算子的规则
                # print(option_rule)# delete
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

                print('uoc：选择对其变异的node：', option_node)
                print('uoc：选择对其变异的node的名字：', option_name)
                print('uoc：选择对其变异的instance：', option_instance)
                print('uoc：选择的变异规则：', option_rule)
                print('uoc：变异后新的层：', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]
                # print(log_dict[i])
        elif 'deadcode_api_name' not in log_dict[str(i)].keys():
            option_layers = list()  # 可被选择的算子列表（形式：(node,instance,name)）
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  # 如果这个算子是module类型的且不是被变异过的算子，就加入列表
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_layers.append((node, module_instance, node.name))
            # print(option_layers)

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  # 随机从option_layers列表中选1个被变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据rules_dict随机选择适用于该算子的规则
                # print(option_rule)# delete
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

                print('uoc：选择对其变异的node：', option_node)
                print('uoc：选择对其变异的node的名字：', option_name)
                print('uoc：选择对其变异的instance：', option_instance)
                print('uoc：选择的变异规则：', option_rule)
                print('uoc：变异后新的层：', new_instance)

                log_dict[str(i)]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[str(i)]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[str(i)]['deadcode_api_rule'] = option_rule.__name__[-5:]
                # print(log_dict[i])
            
        else:  # 根据日志变异
            option_name = log_dict[str(i)]['deadcode_api_name']
            option_rule_name = log_dict[str(i)]['deadcode_api_rule']

            for node in graph.nodes:
                # print(node.name,option_name)
                if node.name == option_name.lower().replace(".","_"):
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_node, option_instance, option_name = node, module_instance, node.name
                    break

            option_rule = match_rule(option_rule_name)
            new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

            print('uoc：选择对其变异的node：', option_node)
            print('uoc：选择对其变异的node的名字：', option_name)
            print('uoc：选择对其变异的instance：', option_instance)
            print('uoc：选择的变异规则：', option_rule)
            print('uoc：变异后新的层：', new_instance)

        # print('变异前的旧模型：')
        # print(graph)  # delete

        # 新层名字
        new_name = reflect_name(option_name, option_rule)

        # 将新层绑定到符号追踪的模块中
        new_module.add_module(new_name, new_instance)
        # 插入新的节点并替换旧的节点
        with option_node.graph.inserting_after(option_node):
            new_node = option_node.graph.call_module(new_name, args=option_node.args)
            option_node.replace_all_uses_with(new_node)
            new_module.graph.erase_node(option_node)
        # 检查并重新编译
        graph.lint()
        new_module.recompile()
        print("finish MR")
        return new_module

    def forward(self, a, b, deada, deadb):
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb

        a, b = handle_shape_strict(a, b)  # 处理维度

        a2 = torch.mul(a, a)
        b2 = torch.mul(b, b)
        ab = torch.mul(a, b)
        ab2 = torch.mul(ab, -2)
        uoc = torch.add(a2, b2)
        uoc = torch.add(uoc, ab2)
        uoc = torch.add(uoc, 1e-10)
        uoc = torch.neg(uoc)
        uoc = torch.sub(uoc, 1e-5)
        # uoc = nn.ReLU()(uoc)
        uoc = random.choice([nn.ReLU(), nn.ReLU6(), nn.Hardtanh()])(uoc) # 随机选择

        dead = make_unsqueeze(deada)  # 调整维度
        add_edge = self.op_layer(dead)
        uoc, add_edge = handle_shape_strict(uoc, add_edge)  # 确保两个张量具有相同的形状以便进行后续操作
        out0 = torch.mul(uoc, add_edge)
        dtype = deadb.dtype
        out, deadbb = handle_shape_final(out0, deadb)
        out = torch.add(out, deadbb)
        out = out.to(dtype)

        return out


class ABSOC_A(nn.Module):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "ABSOC_A"
        self.op_type = op_type
        if self.op_type == "SELayer":
            layer = SELayer()
        if self.op_type == "DenseLayer":
            layer = DenseLayer()
        if self.op_type == "Inception_A":
            layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            layer = ResidualBlock()
        if self.op_type == "DropPath":
            layer = DropPath()
        if self.op_type == "Dense":
            layer = Dense()

        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(layer,log_dict,i,LOG_FLAG) # 对层进行 torch.fx 的更改
        else:
            self.op_layer = layer

    def modify_layer_with_fx(self, layer,log_dict,i,LOG_FLAG):
        new_module=torch.fx.symbolic_trace(layer)
        graph=new_module.graph

        if LOG_FLAG == False:
            option_layers = list()  # 可被选择的算子列表（形式：(node,instance,name)）
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  # 如果这个算子是module类型的且不是被变异过的算子，就加入列表
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_layers.append((node, module_instance, node.name))
            # print(option_layers)

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  # 随机从option_layers列表中选1个被变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据rules_dict随机选择适用于该算子的规则
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

                print('ABSOC_A：选择对其变异的node：', option_node)
                print('ABSOC_A：选择对其变异的node的名字：', option_name)
                print('ABSOC_A：选择对其变异的instance：', option_instance)
                print('ABSOC_A：选择的变异规则：', option_rule)
                print('ABSOC_A：变异后新的层：', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]
        elif 'deadcode_api_name' not in log_dict[str(i)].keys():
            option_layers = list()  # 可被选择的算子列表（形式：(node,instance,name)）
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  # 如果这个算子是module类型的且不是被变异过的算子，就加入列表
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_layers.append((node, module_instance, node.name))
            # print(option_layers)

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  # 随机从option_layers列表中选1个被变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据rules_dict随机选择适用于该算子的规则
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

                print('ABSOC_A：选择对其变异的node：', option_node)
                print('ABSOC_A：选择对其变异的node的名字：', option_name)
                print('ABSOC_A：选择对其变异的instance：', option_instance)
                print('ABSOC_A：选择的变异规则：', option_rule)
                print('ABSOC_A：变异后新的层：', new_instance)

                log_dict[str(i)]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[str(i)]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[str(i)]['deadcode_api_rule'] = option_rule.__name__[-5:]
            
        else:  # 根据日志变异
            option_name = log_dict[str(i)]['deadcode_api_name']
            option_rule_name = log_dict[str(i)]['deadcode_api_rule']

            for node in graph.nodes:
                if node.name == option_name.lower().replace(".","_"):
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_node, option_instance, option_name = node, module_instance, node.name
                    break

            option_rule = match_rule(option_rule_name)
            new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

            print('ABSOC_A：选择对其变异的node：', option_node)
            print('ABSOC_A：选择对其变异的node的名字：', option_name)
            print('ABSOC_A：选择对其变异的instance：', option_instance)
            print('ABSOC_A：选择的变异规则：', option_rule)
            print('ABSOC_A：变异后新的层：', new_instance)

        # print('变异前的旧模型：')
        # print(graph)  # delete

        # 新层名字
        new_name = reflect_name(option_name, option_rule)

        # 将新层绑定到符号追踪的模块中
        new_module.add_module(new_name, new_instance)
        # 插入新的节点并替换旧的节点
        with option_node.graph.inserting_after(option_node):
            new_node = option_node.graph.call_module(new_name, args=option_node.args)
            option_node.replace_all_uses_with(new_node)
            new_module.graph.erase_node(option_node)
        # 检查并重新编译
        graph.lint()
        new_module.recompile()
        # EAGLE变异结束-----------------------------------------------------------------------------

        return new_module

    def forward(self, a, b, deada, deadb):
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb
        a, b = handle_shape_strict(a, b)
        a1 = torch.abs(a)
        b1 = torch.abs(b)
        a1b1 = torch.add(a1, b1)  # |a|+|b|
        ab = torch.abs(torch.add(a, b))  # |a+b|
        absoc_a = torch.sub(a1b1, ab)  # |a|+|b| - |a+b|
        absoc_a = torch.add(absoc_a, 1e-10)  # # |a|+|b| - |a+b| + 1e-10
        absoc_a = torch.neg(absoc_a)  # 取负 -|a|- |b| + |a+b| - 1e-10
        # absoc_a = nn.ReLU()(absoc_a)  # relu(|a+b|-(|a|+|b|)-1e-10)
        absoc_a = random.choice([nn.ReLU(), nn.ReLU6(), nn.Hardtanh()])(absoc_a)  # relu(|a+b|-(|a|+|b|)-1e-10)
        dead = make_unsqueeze(deada)
        add_edge = self.op_layer(dead)
        absoc_a, add_edge = handle_shape_strict(absoc_a, add_edge)
        out = torch.mul(absoc_a, add_edge)
        dtype = deadb.dtype
        out, deadb = handle_shape_final(out, deadb)
        out = torch.add(out, deadb)
        out = out.to(dtype)
        return out


class ABSOC_B(nn.Module):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "ABSOC_B"
        self.op_type = op_type
        if self.op_type == "SELayer":
            layer = SELayer()
        if self.op_type == "DenseLayer":
            layer = DenseLayer()
        if self.op_type == "Inception_A":
            layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            layer = ResidualBlock()
        if self.op_type == "DropPath":
            layer = DropPath()
        if self.op_type == "Dense":
            layer = Dense()

        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(layer, log_dict, i, LOG_FLAG)  # 对层进行 torch.fx 的更改
        else:
            self.op_layer = layer

    def modify_layer_with_fx(self, layer,log_dict,i,LOG_FLAG):
        new_module=torch.fx.symbolic_trace(layer)
        graph=new_module.graph

        if LOG_FLAG == False:
            option_layers = list()  # 可被选择的算子列表（形式：(node,instance,name)）
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  # 如果这个算子是module类型的且不是被变异过的算子，就加入列表
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_layers.append((node, module_instance, node.name))
            # print(option_layers)

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  # 随机从option_layers列表中选1个被变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据rules_dict随机选择适用于该算子的规则
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

                print('ABSOC_B：选择对其变异的node：', option_node)
                print('ABSOC_B：选择对其变异的node的名字：', option_name)
                print('ABSOC_B：选择对其变异的instance：', option_instance)
                print('ABSOC_B：选择的变异规则：', option_rule)
                print('ABSOC_B：变异后新的层：', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]
        elif 'deadcode_api_name' not in log_dict[str(i)].keys():
            option_layers = list()  # 可被选择的算子列表（形式：(node,instance,name)）
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  # 如果这个算子是module类型的且不是被变异过的算子，就加入列表
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_layers.append((node, module_instance, node.name))
            # print(option_layers)

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  # 随机从option_layers列表中选1个被变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据rules_dict随机选择适用于该算子的规则
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

                print('ABSOC_B：选择对其变异的node：', option_node)
                print('ABSOC_B：选择对其变异的node的名字：', option_name)
                print('ABSOC_B：选择对其变异的instance：', option_instance)
                print('ABSOC_B：选择的变异规则：', option_rule)
                print('ABSOC_B：变异后新的层：', new_instance)

                log_dict[str(i)]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[str(i)]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[str(i)]['deadcode_api_rule'] = option_rule.__name__[-5:]
        else:  # 根据日志变异
            option_name = log_dict[str(i)]['deadcode_api_name']
            option_rule_name = log_dict[str(i)]['deadcode_api_rule']

            for node in graph.nodes:
                print(node.name,option_name.lower())
                if node.name == option_name.lower().replace(".","_"):
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_node, option_instance, option_name = node, module_instance, node.name
                    break

            option_rule = match_rule(option_rule_name)
            new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

            print('ABSOC_B：选择对其变异的node：', option_node)
            print('ABSOC_B：选择对其变异的node的名字：', option_name)
            print('ABSOC_B：选择对其变异的instance：', option_instance)
            print('ABSOC_B：选择的变异规则：', option_rule)
            print('ABSOC_B：变异后新的层：', new_instance)

        # print('变异前的旧模型：')
        # print(graph)  # delete

        # 新层名字
        new_name = reflect_name(option_name, option_rule)

        # 将新层绑定到符号追踪的模块中
        new_module.add_module(new_name, new_instance)
        # 插入新的节点并替换旧的节点
        with option_node.graph.inserting_after(option_node):
            new_node = option_node.graph.call_module(new_name, args=option_node.args)
            option_node.replace_all_uses_with(new_node)
            new_module.graph.erase_node(option_node)
        # 检查并重新编译
        graph.lint()
        new_module.recompile()
        # EAGLE变异结束-----------------------------------------------------------------------------

        return new_module

    def forward(self, a, b, deada, deadb):
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb
        # print("a.dtype", a.dtype)
        # print("b.dtype", b.dtype)
        # print("deada.dtype", deada.dtype)
        # print("deadb.dtype", deadb.dtype)
        # print("ABSOC_A old a", a.shape)
        # print("ABSOC_A old b", b.shape)
        a, b = handle_shape_strict(a, b)
        # print("a", a.shape)
        # print("b", b.shape)
        a1 = torch.abs(a)
        b1 = torch.abs(b)
        a1b1 = torch.sub(a1, b1)  # |a|-|b|
        a1b1 = torch.abs(a1b1)  # ||a|-|b||
        ab = torch.abs(torch.add(a, b))  # |a+b|
        absoc_b = torch.sub(a1b1, ab)  # ||a|-|b|| - |a+b|
        absoc_b = torch.sub(absoc_b, 1e-5)  # ||a|-|b|| - |a+b| - 1e-5
        # absoc_b = nn.ReLU()(absoc_b)  # relu(||a|-|b|| - |a+b| - 1e-5)
        absoc_b = random.choice([nn.ReLU(), nn.ReLU6(), nn.Hardtanh()])(absoc_b)  # relu(||a|-|b|| - |a+b| - 1e-5)
        dead = make_unsqueeze(deada)
        add_edge = self.op_layer(dead)
        absoc_b, add_edge = handle_shape_strict(absoc_b, add_edge)
        # dead, uoc = handle_shape_final(dead, uoc)
        out = torch.mul(absoc_b, add_edge)
        dtype = deadb.dtype
        out, deadb = handle_shape_final(out, deadb)
        out = torch.add(out, deadb)
        # print("uoc dtype", uoc.dtype)
        out = out.to(dtype)
        # print("ABSOC_A out shape: ", out.shape)
        return out


class PIOC(nn.Module):
    def __init__(self, op_type, operator_mutation_type, log_dict, i, LOG_FLAG):
        super().__init__()
        self.__name__ = "PIOC"
        self.op_type = op_type
        if self.op_type == "SELayer":
            layer = SELayer()
        if self.op_type == "DenseLayer":
            layer = DenseLayer()
        if self.op_type == "Inception_A":
            layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            layer = ResidualBlock()
        if self.op_type == "DropPath":
            layer = DropPath()
        if self.op_type == "Dense":
            layer = Dense()

        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(layer, log_dict, i, LOG_FLAG)  # 对层进行 torch.fx 的更改
        else:
            self.op_layer = layer

    def modify_layer_with_fx(self, layer, log_dict, i, LOG_FLAG):
        new_module = torch.fx.symbolic_trace(layer)
        graph = new_module.graph

        if LOG_FLAG == False:
            option_layers = list()  # 可被选择的算子列表（形式：(node,instance,name)）
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  # 如果这个算子是module类型的且不是被变异过的算子，就加入列表
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_layers.append((node, module_instance, node.name))
            # print(option_layers)

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  # 随机从option_layers列表中选1个被变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据rules_dict随机选择适用于该算子的规则
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

                print('PIOC：选择对其变异的node：', option_node)
                print('PIOC：选择对其变异的node的名字：', option_name)
                print('PIOC：选择对其变异的instance：', option_instance)
                print('PIOC：选择的变异规则：', option_rule)
                print('PIOC：变异后新的层：', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]
        elif 'deadcode_api_name' not in log_dict[str(i)].keys():
            option_layers = list()  # 可被选择的算子列表（形式：(node,instance,name)）
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  # 如果这个算子是module类型的且不是被变异过的算子，就加入列表
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_layers.append((node, module_instance, node.name))
            # print(option_layers)

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  # 随机从option_layers列表中选1个被变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据rules_dict随机选择适用于该算子的规则
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

                print('PIOC：选择对其变异的node：', option_node)
                print('PIOC：选择对其变异的node的名字：', option_name)
                print('PIOC：选择对其变异的instance：', option_instance)
                print('PIOC：选择的变异规则：', option_rule)
                print('PIOC：变异后新的层：', new_instance)

                log_dict[str(i)]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[str(i)]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[str(i)]['deadcode_api_rule'] = option_rule.__name__[-5:]
        else:  # 根据日志变异
            option_name = log_dict[str(i)]['deadcode_api_name']
            option_rule_name = log_dict[str(i)]['deadcode_api_rule']

            for node in graph.nodes:
                if node.name == option_name.lower().replace(".","_"):
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  # 获取模块实例
                    option_node, option_instance, option_name = node, module_instance, node.name
                    break

            option_rule = match_rule(option_rule_name)
            new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  # 根据规则变异过的新层

            print('PIOC：选择对其变异的node：', option_node)
            print('PIOC：选择对其变异的node的名字：', option_name)
            print('PIOC：选择对其变异的instance：', option_instance)
            print('PIOC：选择的变异规则：', option_rule)
            print('PIOC：变异后新的层：', new_instance)

        # print('变异前的旧模型：')
        # print(graph)  # delete

        # 新层名字
        new_name = reflect_name(option_name, option_rule)

        # 将新层绑定到符号追踪的模块中
        new_module.add_module(new_name, new_instance)
        # 插入新的节点并替换旧的节点
        with option_node.graph.inserting_after(option_node):
            new_node = option_node.graph.call_module(new_name, args=option_node.args)
            option_node.replace_all_uses_with(new_node)
            new_module.graph.erase_node(option_node)
        # 检查并重新编译
        graph.lint()
        new_module.recompile()
        # EAGLE变异结束-----------------------------------------------------------------------------

        return new_module

    def forward(self, input, deada, final_dead, deadb=None):
        if isinstance(input, tuple) or isinstance(deada, tuple) or isinstance(final_dead, tuple):
            print("mutate failed for input tuple")
            print("type a", type(input))
            print("type b", type(deada))
            print("type c", type(final_dead))
            return input, deada, final_dead
        reduce_edge = make_reduce(input)
        dtype = reduce_edge.dtype
        # print("reduce_edge shape is", reduce_edge.shape)
        if torch.get_device(reduce_edge) != "CPU":
            const_np = reduce_edge.detach().cpu().numpy()
        else:
            const_np = reduce_edge.detach().numpy()
        const_edge = torch.tensor(const_np, dtype=dtype).to(device)
        sub_edge = torch.sub(reduce_edge, const_edge)

        deada, _ = handle_shape_strict(deada, deada)
        deada = make_unsqueeze(deada)
        add_edge = self.op_layer(deada)
        sub_edge, add_edge = handle_shape_strict(sub_edge, add_edge)
        mul_edge = torch.mul(add_edge, sub_edge)

        mul_edge_1, final_dead_1 = handle_shape_final(mul_edge, final_dead)
        # print("mul_edge_1 is", mul_edge_1.shape)
        # print("final_dead_1 is", final_dead_1.shape)
        out = torch.add(mul_edge_1, final_dead_1)
        pioc_equal = np.allclose(out.detach().cpu().numpy(), final_dead_1.detach().cpu().numpy())
        # if not pioc_equal:
        #     print("pioc不相等！")
        #assert pioc_equal
        # print("pioc out shape is", out.shape)
        # print("pioc dtype is", out.dtype)
        return out
