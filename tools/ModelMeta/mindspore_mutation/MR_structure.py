import mindspore
import mindspore.ops as ops
import numpy as np
from mindspore import nn
import random
from mindspore_mutation.deadcode import SELayer, Inception_A, DenseLayer, DropPath, Dense, ResidualBlock, PWDWPW_ResidualBlock, MyConvPoolLayerSameShape, MyAdaptiveMaxPoolLayer, MyTransposeConvLayer
from mindspore_mutation.handel_shape import handle_shape_final, handle_shape_strict, make_unsqueeze, make_reduce
from mindspore_mutation.cargo import match_rule,reflect_name,rule_reflect_class
import mindspore_mutation.config as config
import collections
from mindspore.rewrite.node import NodeManager
from mindspore.rewrite import ScopedValue, NodeType
from mindspore.rewrite import SymbolTree


banned_ops = [mindspore.ops.operations.array_ops.Shape,
              mindspore.ops.operations.array_ops.Concat,
              mindspore.ops.operations.array_ops.TupleToArray,
              mindspore.ops.operations.array_ops.Reshape,
              mindspore.ops.operations.array_ops.Tile,
              type(None)
              ]
banned_cell = [mindspore.nn.layer.CentralCrop, ]
banned_trees = [mindspore.ops.ResizeBilinearV2,
                mindspore.ops.operations.Shape,
                type(None)
                ]


# def scan_node(stree, hash_table, nodedict=None, depth=0):
#     # global hash_table
#     # for node in stree.nodes(all_nodes=False):
#     if type(stree) == mindspore.rewrite.api.symbol_tree.SymbolTree:
#         stree = stree._symbol_tree
#     for node in stree.all_nodes():
#         if isinstance(node, NodeManager):
#             for sub_node in node.get_tree_nodes():
#                 subtree = sub_node.symbol_tree
#                 scan_node(subtree, hash_table, nodedict=nodedict, depth=depth + 1)
#         if (node.get_node_type() == NodeType.CallCell and node.get_instance_type() not in banned_cell) or (
#                 node.get_node_type() == NodeType.CallPrimitive and node.get_instance_type() not in banned_ops) \
#                 or (node.get_node_type() == NodeType.Tree and node.get_instance_type() not in banned_trees) \
#                 or node.get_node_type() == NodeType.CellContainer:
#             if hash_table[mindspore.rewrite.api.node.Node(node).get_handler()] == 1:
#                 continue
#             hash_table[mindspore.rewrite.api.node.Node(node).get_handler()] += 1
#             if node.get_node_type() not in [NodeType.CellContainer, NodeType.Tree]:
#                 nodedict[mindspore.rewrite.api.node.Node(node).get_handler()] = node.get_belong_symbol_tree()
#     return True,nodedict



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



# def scan_layer(layer,option_layers):
#     layers = layer.cells_and_names()
#     for name, submodule in layers:
#         if i != '':
#             if not has_child_node(new_net, i):
#                 option_layers.append((name, submodule, name))
            
#     return option_layers

rules_dict = config.rules_dict # 对于每一个类型的算子，有哪些规则适用
class UOC(nn.Cell):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "UOC"
        self.op_type = op_type
        if self.op_type == "SELayer":
            self.op_layer = SELayer()
        if self.op_type == "DenseLayer":
            self.op_layer = DenseLayer()
        if self.op_type == "Inception_A":
            self.op_layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            self.op_layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            self.op_layer = ResidualBlock()
        if self.op_type == "DropPath":
            self.op_layer = DropPath()
        if self.op_type == "Dense":
            self.op_layer = Dense()
        if self.op_type == "MyConvPoolLayerSameShape":
            self.op_layer = MyConvPoolLayerSameShape()
        if self.op_type == "MyAdaptiveMaxPoolLayer":
            self.op_layer = MyAdaptiveMaxPoolLayer()
        if self.op_type == "MyTransposeConvLayer":
            self.op_layer = MyTransposeConvLayer()
        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(self.op_layer, log_dict, i, LOG_FLAG)  # 对层进行变异
        else:
            self.op_layer = self.op_layer

    def modify_layer_with_fx(self, layer, log_dict, i, LOG_FLAG):
        # MindSpore 没有 torch.fx，因此手动修改网络层
        hash_table = collections.defaultdict(int)
        nodedict = collections.OrderedDict()
        if LOG_FLAG == False:
            
            option_layers = list()  # 可被选择的算子列表（形式：(node, instance, name)）

            # print("layer_type:",type(layer))
            for name, child in layer.cells_and_names():
                if not has_child_node(layer,name) and not name == '' and not 'deadcode' in str(type(child)):
                    # print(name)
                    if type(child) in rules_dict.keys():
                        option_layers.append((name, child, name))
            print(option_layers)
            if len(option_layers) != 0:
                option_name, option_instance, option_node_name = random.choice(option_layers)  # 随机选择一个可变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据 rules_dict 随机选择适用于该算子的规则
                
                new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  # 根据规则变异过的新层

                print('UOC:node', option_node_name)
                print('UOC:node name', option_name)
                print('UOC:instance', option_instance)
                print('UOC:rule', option_rule)
                print('UOC:new layer', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

                # 替换原始层
                layer._cells[option_name] = new_instance

        else:  # 根据日志变异
            try:
                option_name = log_dict[str(i)]['deadcode_api_name']
                option_rule_name = log_dict[str(i)]['deadcode_api_rule']

                option_instance = layer._cells.get(option_name, None)
                if option_instance is not None:
                    option_rule = match_rule(option_rule_name)
                    new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  # 根据规则变异过的新层

                    print('UOC:option_name', option_name)
                    print('UOC:instance', option_instance)
                    print('UOC:rule', option_rule)
                    print('UOC:new layer', new_instance)

                    # 替换原始层
                    layer._cells[option_name] = new_instance
            except Exception as e:
                print(e)

        # 更新图结构（模拟 symbolic_trace 的效果）
        layer.update_parameters_name()
        return layer


    def construct(self, a, b, deada, deadb):
        # print("deada", type(deada)) Stubtensor
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            return a, b, deada, deadb
        # print("old a", a.shape)
        # print("old b", b.shape)
        a, b = handle_shape_strict(a, b)
        # print("a", a.shape)
        # print("b", b.shape)
        a2 = ops.mul(a, a)
        # print("a2 min", ops.min(a2))
        b2 = ops.mul(b, b)
        # print("b2 min", ops.min(b2))
        ab = ops.mul(a, b)
        # print("ab max", ops.min(ab))
        ab2 = ops.mul(ab, -2)
        # print("ab2 min", ops.max(ab2))
        uoc = ops.add(a2, b2)
        # print("a2b2 min", ops.min(uoc))
        uoc = ops.add(uoc, ab2)
        # tmp, _ = ops.min(uoc)
        # if tmp < 0:
        #     np.save("issue/issue_uoc/a.npy", a.asnumpy())
        #     np.save("issue/issue_uoc/b.npy", b.asnumpy())
        #     exit(0)
        # print("uoc min", ops.min(uoc))
        uoc = ops.add(uoc, 1e-10)
        # print("uoc before neg min", ops.min(uoc))
        uoc = ops.neg(uoc)
        # print("uoc before relu max", ops.max(uoc))
        uoc = ops.ReLU()(uoc)
        # print("uoc max", ops.max(uoc))
        dead = make_unsqueeze(deada)
        add_edge = self.op_layer(dead)
        uoc, add_edge = handle_shape_strict(uoc, add_edge)
        # dead, uoc = handle_shape_final(dead, uoc)
        out = ops.mul(uoc, add_edge)
        dtype = deadb.dtype
        # print("deadb shape", deadb.shape)
        out, deadb = handle_shape_final(out, deadb)
        # print("out max", ops.max(out))
        # print("out shape", out.shape)
        # print("deadb shape", deadb.shape)
        out = ops.add(out, deadb)
        # uoc_equal = np.allclose(out.asnumpy(), deadb.asnumpy())
        # if not uoc_equal:
        #     print("uoc equal", uoc_equal)
        #     print("uoc distance", ChebyshevDistance(out, deadb))
        # assert uoc_equal
        # print("out shape: ", out.shape)
        out = out.to(dtype)
        return out


class PIOC(nn.Cell):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "PIOC"
        self.op_type = op_type
        if self.op_type == "SELayer":
            self.op_layer = SELayer()
        if self.op_type == "DenseLayer":
            self.op_layer = DenseLayer()
        if self.op_type == "Inception_A":
            self.op_layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            self.op_layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            self.op_layer = ResidualBlock()
        if self.op_type == "DropPath":
            self.op_layer = DropPath()
        if self.op_type == "Dense":
            self.op_layer = Dense()
        if self.op_type == "MyConvPoolLayerSameShape":
            self.op_layer = MyConvPoolLayerSameShape()
        if self.op_type == "MyAdaptiveMaxPoolLayer":
            self.op_layer = MyAdaptiveMaxPoolLayer()
        if self.op_type == "MyTransposeConvLayer":
            self.op_layer = MyTransposeConvLayer()
        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(self.op_layer, log_dict, i, LOG_FLAG)  # 对层进行变异
        else:
            self.op_layer = self.op_layer

    def modify_layer_with_fx(self, layer, log_dict, i, LOG_FLAG):
        # MindSpore 没有 torch.fx，因此手动修改网络层
        hash_table = collections.defaultdict(int)
        nodedict = collections.OrderedDict()
        if LOG_FLAG == False:
            option_layers = list()  # 可被选择的算子列表（形式：(node, instance, name)）

            #$ print("layer_type:",type(layer))
            for name, child in layer.cells_and_names():
                if not has_child_node(layer,name) and not name == '' and not 'deadcode' in str(type(child)):
                    option_layers.append((name, child, name))

            if len(option_layers) != 0:
                option_name, option_instance, option_node_name = random.choice(option_layers)  # 随机选择一个可变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据 rules_dict 随机选择适用于该算子的规则
                new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  # 根据规则变异过的新层

                print('PIOC:node', option_node_name)
                print('PIOC:node name', option_name)
                print('PIOC:instance', option_instance)
                print('PIOC:rule', option_rule)
                print('PIOC:new layer', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

                # 替换原始层
                layer._cells[option_name] = new_instance

        else:  # 根据日志变异
            try:
                option_name = log_dict[str(i)]['deadcode_api_name']
                option_rule_name = log_dict[str(i)]['deadcode_api_rule']

                option_instance = layer._cells.get(option_name, None)
                if option_instance is not None:
                    option_rule = match_rule(option_rule_name)
                    new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  # 根据规则变异过的新层

                    print('uoc:option_name', option_name)
                    print('uoc:instance', option_instance)
                    print('uoc:rule', option_rule)
                    print('uoc:new layer', new_instance)

                    # 替换原始层
                    layer._cells[option_name] = new_instance
            except Exception as e:
                print(e)

        # 更新图结构（模拟 symbolic_trace 的效果）
        layer.update_parameters_name()
        return layer

    def construct(self, input, deada, final_dead, deadb=None):
        reduce_edge = make_reduce(input)
        dtype = reduce_edge.dtype
        # print("reduce_edge shape is", reduce_edge.shape)
        const_np = reduce_edge.asnumpy()
        # print("max of reduce_edge is", np.max(const_np))
        const_edge = mindspore.Tensor(const_np, dtype)
        sub_edge = ops.Sub()(reduce_edge, const_edge)
        if self.op_type == "Add":
            deada, deadb = handle_shape_strict(deada, deadb)
            add_edge = ops.Add()(deada, deadb)
            # print("old add_edge shape is", add_edge.shape)
            # print("old sub_edge shape is", sub_edge.shape)
            add_edge, sub_edge = handle_shape_strict(add_edge, sub_edge)
            # print("add_edge shape is", add_edge.shape)
            # print("sub_edge shape is", sub_edge.shape)
            mul_edge = ops.Mul()(add_edge, sub_edge)
            # print("mul_edge shape is", mul_edge.shape)
        elif self.op_type in ["DenseLayer", "SELayer", "Inception_A", "PWDWPW_ResidualBlock",
                              "ResidualBlock", "DropPath", "Dense"]:
            deada, _ = handle_shape_strict(deada, deada)
            deada = make_unsqueeze(deada)
            add_edge = self.op_layer(deada)
            sub_edge, add_edge = handle_shape_strict(sub_edge, add_edge)
            mul_edge = ops.Mul()(add_edge, sub_edge)
        else:
            raise NotImplementedError("optype Not Implemented for optype: {}".format(self.op_type))

        # else:
        #     raise ValueError("PIOC len Not Implemented")
        # print("mul_edge is", mul_edge.shape)
        # print("final_dead is", final_dead.shape)
        mul_edge_1, final_dead_1 = handle_shape_final(mul_edge, final_dead)
        # print("mul_edge_1 is", mul_edge_1.shape)
        # print("final_dead_1 is", final_dead_1.shape)
        out = ops.Add()(mul_edge_1, final_dead_1)
        # print("pioc out shape is", out.shape)
        return out


class ABSOC_A(nn.Cell):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "uoc"
        self.op_type = op_type
        if self.op_type == "Add":
            self.op_layer = SELayer()
        if self.op_type == "SELayer":
            self.op_layer = SELayer()
        if self.op_type == "DenseLayer":
            self.op_layer = DenseLayer()
        if self.op_type == "Inception_A":
            self.op_layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            self.op_layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            self.op_layer = ResidualBlock()
        if self.op_type == "DropPath":
            self.op_layer = DropPath()
        if self.op_type == "Dense":
            self.op_layer = Dense()
        if self.op_type == "MyConvPoolLayerSameShape":
            self.op_layer = MyConvPoolLayerSameShape()
        if self.op_type == "MyAdaptiveMaxPoolLayer":
            self.op_layer = MyAdaptiveMaxPoolLayer()
        if self.op_type == "MyTransposeConvLayer":
            self.op_layer = MyTransposeConvLayer()  
        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(self.op_layer, log_dict, i, LOG_FLAG)  # 对层进行变异
        else:
            self.op_layer = self.op_layer

    def modify_layer_with_fx(self, layer, log_dict, i, LOG_FLAG):
        # MindSpore 没有 torch.fx，因此手动修改网络层
        hash_table = collections.defaultdict(int)
        nodedict = collections.OrderedDict()
         #print("1")
        if LOG_FLAG == False:
            print("1")
            option_layers = list()  # 可被选择的算子列表（形式：(node, instance, name)）

            # print("layer_type:",type(layer))
            for name, child in layer.cells_and_names():
                if not has_child_node(layer,name) and not name == '' and not 'deadcode' in str(type(child)):
                    option_layers.append((name, child, name))
            print("1")
            if len(option_layers) != 0:
                option_name, option_instance, option_node_name = random.choice(option_layers)  # 随机选择一个可变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据 rules_dict 随机选择适用于该算子的规则
                new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  # 根据规则变异过的新层

                print('ABSOC_A:node', option_node_name)
                print('ABSOC_A:node name', option_name)
                print('ABSOC_A:instance', option_instance)
                print('ABSOC_A:rule', option_rule)
                print('ABSOC_A:new layer', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

                # 替换原始层
                layer._cells[option_name] = new_instance
            # print("1")
        else:  # 根据日志变异
            try:
                option_name = log_dict[str(i)]['deadcode_api_name']
                option_rule_name = log_dict[str(i)]['deadcode_api_rule']

                option_instance = layer._cells.get(option_name, None)
                if option_instance is not None:
                    option_rule = match_rule(option_rule_name)
                    new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  # 根据规则变异过的新层

                    print('ABSOC_A:option_name', option_name)
                    print('ABSOC_A:instance', option_instance)
                    print('ABSOC_A:rule', option_rule)
                    print('ABSOC_A:new layer', new_instance)

                    # 替换原始层
                    layer._cells[option_name] = new_instance
            except Exception as e:
                print(e)
        # 更新图结构（模拟 symbolic_trace 的效果）
        layer.update_parameters_name()
        return layer

    def construct(self, a, b, deada, deadb):
        # print("deada", type(deada)) Stubtensor
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
        a1 = mindspore.ops.abs(a)
        b1 = mindspore.ops.abs(b)
        a1b1 = mindspore.ops.add(a1, b1)  # |a|+|b|
        ab = mindspore.ops.abs(mindspore.ops.add(a, b))  # |a+b|
        absoc_a = mindspore.ops.sub(a1b1, ab)  # |a|+|b| - |a+b|
        absoc_a = mindspore.ops.add(absoc_a, 1e-10)  # # |a|+|b| - |a+b| + 1e-10
        absoc_a = mindspore.ops.neg(absoc_a)  # 取负 -|a|+|b| + |a+b| - 1e-10
        absoc_a = mindspore.nn.ReLU()(absoc_a)  # relu(|a+b|-(|a|+|b|)-1e-10)
        # print("ABSOC_A", ABSOC_A.shape)
        dead = make_unsqueeze(deada)
        add_edge = self.op_layer(dead)
        absoc_a, add_edge = handle_shape_strict(absoc_a, add_edge)
        # dead, uoc = handle_shape_final(dead, uoc)
        out = mindspore.ops.mul(absoc_a, add_edge)
        dtype = deada.dtype
        out, deada = handle_shape_final(out, deada)
        out = mindspore.ops.add(out, deada)
        # print("uoc dtype", uoc.dtype)
        out = out.to(dtype)
        # print("ABSOC_A out shape: ", out.shape)
        return out


class ABSOC_B(nn.Cell):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "uoc"
        self.op_type = op_type
        if self.op_type == "Add":
            self.op_layer = SELayer()
        if self.op_type == "SELayer":
            self.op_layer = SELayer()
        if self.op_type == "DenseLayer":
            self.op_layer = DenseLayer()
        if self.op_type == "Inception_A":
            self.op_layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            self.op_layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            self.op_layer = ResidualBlock()
        if self.op_type == "DropPath":
            self.op_layer = DropPath()
        if self.op_type == "Dense":
            self.op_layer = Dense()
        if self.op_type == "MyConvPoolLayerSameShape":
            self.op_layer = MyConvPoolLayerSameShape()
        if self.op_type == "MyAdaptiveMaxPoolLayer":
            self.op_layer = MyAdaptiveMaxPoolLayer()
        if self.op_type == "MyTransposeConvLayer":
            self.op_layer = MyTransposeConvLayer()
        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(self.op_layer, log_dict, i, LOG_FLAG)  # 对层进行变异
        else:
            self.op_layer = self.op_layer

    def modify_layer_with_fx(self, layer, log_dict, i, LOG_FLAG):
        # MindSpore 没有 torch.fx，因此手动修改网络层
        hash_table = collections.defaultdict(int)
        nodedict = collections.OrderedDict()
        if LOG_FLAG == False:
            option_layers = list()  # 可被选择的算子列表（形式：(node, instance, name)）

            # print("layer_type:",type(layer))
            for name, child in layer.cells_and_names():
                if not has_child_node(layer,name) and not name == '' and not 'deadcode' in str(type(child)):
                    option_layers.append((name, child, name))

            if len(option_layers) != 0:
                option_name, option_instance, option_node_name = random.choice(option_layers)  # 随机选择一个可变异的算子
                option_rule = random.choice(rules_dict[type(option_instance)])  # 根据 rules_dict 随机选择适用于该算子的规则
                new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  # 根据规则变异过的新层

                print('ABSOC_B:node', option_node_name)
                print('ABSOC_B:node name', option_name)
                print('ABSOC_B:instance', option_instance)
                print('ABSOC_B:rule', option_rule)
                print('ABSOC_B:new layer', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

                # 替换原始层
                layer._cells[option_name] = new_instance

        else:  # 根据日志变异
            try:
                option_name = log_dict[str(i)]['deadcode_api_name']
                option_rule_name = log_dict[str(i)]['deadcode_api_rule']

                option_instance = layer._cells.get(option_name, None)
                if option_instance is not None:
                    option_rule = match_rule(option_rule_name)
                    new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  # 根据规则变异过的新层

                    print('ABSOC_B:option_name', option_name)
                    print('ABSOC_B:instance', option_instance)
                    print('ABSOC_B:rule', option_rule)
                    print('ABSOC_B:new layer', new_instance)

                    # 替换原始层
                    layer._cells[option_name] = new_instance
            except Exception as e:
                print(e)
        # 更新图结构（模拟 symbolic_trace 的效果）
        layer.update_parameters_name()
        return layer


    def construct(self, a, b, deada, deadb):
        # print("deada", type(deada)) Stubtensor
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb

        a, b = handle_shape_strict(a, b)

        a1 = mindspore.ops.abs(a)
        b1 = mindspore.ops.abs(b)
        a1b1 = mindspore.ops.subtract(a1, b1)  # |a|-|b|
        a1b1 = mindspore.ops.abs(a1b1)  # ||a|-|b||
        ab = mindspore.ops.abs(mindspore.ops.add(a, b))  # |a+b|
        absoc_b = mindspore.ops.subtract(a1b1, ab)  # ||a|-|b|| - |a+b|
        absoc_b = mindspore.ops.subtract(absoc_b, 1e-5)  # ||a|-|b|| - |a+b| - 1e-5
        absoc_b = mindspore.nn.ReLU()(absoc_b)  # relu(||a|-|b|| - |a+b| - 1e-5)
        # print("ABSOC_A", ABSOC_A.shape)
        dead = make_unsqueeze(deada)
        add_edge = self.op_layer(dead)
        absoc_b, add_edge = handle_shape_strict(absoc_b, add_edge)
        # dead, uoc = handle_shape_final(dead, uoc)
        out = mindspore.ops.mul(absoc_b, add_edge)
        dtype = deada.dtype
        out, deada = handle_shape_final(out, deada)
        out = mindspore.ops.add(out, deada)
        # print("uoc dtype", uoc.dtype)
        out = out.to(dtype)
        # print("ABSOC_A out shape: ", out.shape)
        return out
