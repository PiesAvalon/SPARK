# import torch
# from torch import nn
# from torch.nn import functional as F
import random
import numpy as np
import onnx
from onnx_mutation.deadcode import DeadGenerator
from onnx_mutation.distance import ChebyshevDistance, ManhattanDistance, EuclideanDistance
from onnx_mutation.node_gen import NodeChainGen
from onnx_mutation.edge_node import *
from onnx_mutation.utils.onnx_utils import onnx_run, name_obj_dict
import onnxruntime as rt
# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
#     device = 'cpu'
from collections import OrderedDict
class UOC():
    def __init__(self, op_type, max_node_idx , max_edge_idx):
        super().__init__()
        self.__name__ = "uoc"
        self.op_type = op_type
        self.gen = NodeChainGen(max_node_idx ,max_edge_idx)
        self.make_dead = DeadGenerator(self.gen)


    def run_uoc(self, edge_a: EdgeNode, edge_b: EdgeNode, deada: EdgeNode, deadb: EdgeNode, sub_node: EdgeNode):

        edges, (edge_a, edge_b), c_shape = self.gen.bilateral_shape_matching(
            [edge_a, edge_b], True)
        
        a2_edge = self.gen.make_edge_node(
            'Mul', [edge_a, edge_a], edge_a.shape, True)
        edges.append(a2_edge)

        b2_edge = self.gen.make_edge_node(
            'Mul', [edge_b, edge_b], edge_b.shape, True)
        edges.append(b2_edge)

        ab_edge = self.gen.make_edge_node(
            'Mul', [edge_a, edge_b], c_shape, True)
        edges.append(ab_edge)

        minus_two_edge = self.gen.make_constant(
            np.array([-2], dtype=np.float32))
        edges.append(minus_two_edge)

        ab2_edge = self.gen.make_edge_node(
            'Mul', [ab_edge, minus_two_edge], c_shape, True)
        edges.append(ab2_edge)

        sum_e = self.gen.make_edge_node(
            'Add', [a2_edge, b2_edge], c_shape,
            True
        )
        edges.append(sum_e)

        sum_e = self.gen.make_edge_node(
            'Add', [sum_e, ab2_edge], c_shape,
            True
        )
        edges.append(sum_e)

        exp_e = self.gen.make_constant(
            np.array([1e-10], dtype=np.float32))
        edges.append(exp_e)

        uoc = self.gen.make_edge_node(
            'Add', [sum_e, exp_e], c_shape,
            True
        )
        edges.append(uoc)

        uoc = self.gen.make_edge_node(
            'Neg', uoc, c_shape, uoc.zero)
        edges.append(uoc)

        minus_five_edge = self.gen.make_constant(
            np.array([-0.00005], dtype=np.float32))
        edges.append(minus_five_edge)

        uoc = self.gen.make_edge_node(
            'Add', [uoc, minus_five_edge], c_shape,
            True
        )
        edges.append(uoc)

        uoc = self.gen.make_edge_node('Relu', uoc, c_shape, True)
        # edges.append(uoc)

        node, out_shape, out_name, new_edge = self.gen.gen_unsqueeze(
                deada.name, deada.shape, 4)
        # print('new_edge')
        # print(new_edge[1].name)

        for i in new_edge:
            edges.append(i)

        out_deada = EdgeNode(out_name, out_shape, node, deada.zero)
        
        # for i in (new_edge):
        edges.append(out_deada)
        add_edge, new_edges = self.make_dead.gen_dead(self.op_type.name, edge_a = out_deada, all_edges=edges)

        for i in new_edges:
            edges.append(i)
        print('print(uoc_shape, add_edge_shape)')
        print(uoc.shape, add_edge.shape)
        uoc_shape, add_edge_shape = self.gen.handle_shape_strict(uoc.shape, add_edge.shape)
        print(uoc_shape, add_edge_shape)

        uoc,new_edge = self.gen.unilateral_shape_matching(edges, uoc, uoc_shape, True)
        for i in new_edge:
            edges.append(i)
        # edges.append(uoc)
        add_edge,new_edge = self.gen.unilateral_shape_matching(edges, add_edge, add_edge_shape, True)
        for i in new_edge:
            edges.append(i)
        # edges.append(add_edge)

        # dead, uoc = handle_shape_final(dead, uoc)

        out01 = self.gen.make_edge_node(
            'Mul', [uoc, add_edge], add_edge.shape, True)
        edges.append(out01)
        # out0_shape, deadb_shape = self.gen.handle_shape_strict(out0.shape, deadb.shape)

        out0, new_edge = self.gen.unilateral_shape_matching(edges, out01, sub_node.shape, True)
        for i in new_edge:
            if i.name != out01.name:
                edges.append(i)

        # edges.append(out0)
        # deadb,new_edge = self.gen.unilateral_shape_matching(edges, deadb, deadb_shape, True)
        # for i in new_edge:
        #     edges.append(i)
        # edges.append(deadb)

        out,add_edge = self.gen.make_subs_add(sub_node, out0)
        # out = self.gen.make_edge_node(
        #     'Add', [out0, deadb], deadb.shape,
        #     True
        # )
        edges.append(out)
        edges.append(add_edge)

        # print("out shape: ", out.shape)
        return edges


class ABSOC_A():
    def __init__(self, op_type, max_node_idx , max_edge_idx):
        super().__init__()
        self.__name__ = "ABSOC_A"
        self.op_type = op_type
        self.gen = NodeChainGen(max_node_idx ,max_edge_idx)
        self.make_dead = DeadGenerator(self.gen)

    def run_ABSOC_A(self, a, b, deada, deadb, sub_node):
        # print("deada", type(deada)) Stubtensor
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb
            

        edges, (edge_a, edge_b), c_shape = self.gen.bilateral_shape_matching(
            [a, b], True)
        # a, b = handle_shape_strict(a, b)

        # print("a", a.shape)
        # print("b", b.shape)

        a1_edge = self.gen.make_edge_node(
            'abs', [edge_a], edge_a.shape, True)
        edges.append(a1_edge)
        # a1 = torch.abs(a)

        b1_edge = self.gen.make_edge_node(
            'abs', [edge_b], edge_b.shape, True)
        edges.append(b1_edge)
        # b1 = torch.abs(b) 

        a1b1 = self.gen.make_edge_node(
            'Add', [a1_edge, b1_edge], b1_edge.shape,
            True
        )
        edges.append(a1b1)
        # a1b1 = torch.add(a1, b1) # |a|+|b|

        ab = self.gen.make_edge_node(
            'Add', [edge_a, edge_b], edge_b.shape,
            True
        )
        edges.append(ab)

        ab = self.gen.make_edge_node(
            'abs', [ab], ab.shape, True)
        edges.append(ab)
        # ab = torch.abs(torch.add(a, b)) # |a+b|

        absoc_a = self.gen.make_edge_node(
            'sub', [a1b1, ab], ab.shape, True)
        edges.append(absoc_a)
        # absoc_a = torch.sub(a1b1, ab) # |a|+|b| - |a+b|

        exp_e = self.gen.make_constant(
            np.array([1e-10], dtype=np.float32))
        edges.append(exp_e)

        absoc_a = self.gen.make_edge_node(
            'Add', [absoc_a, exp_e], absoc_a.shape,
            True
        )
        edges.append(absoc_a)
        # absoc_a = torch.add(absoc_a, 1e-10) # # |a|+|b| - |a+b| + 1e-10

        absoc_a = self.gen.make_edge_node(
            'Neg', absoc_a, absoc_a.shape, True)
        edges.append(absoc_a)
        # absoc_a = torch.neg(absoc_a) #取负 -|a|+|b| + |a+b| - 1e-10

        absoc_a = self.gen.make_edge_node('Relu', absoc_a, absoc_a.shape, True)
        # absoc_a = nn.ReLU()(absoc_a) # relu(|a+b|-(|a|+|b|)-1e-10)

        node, out_shape, out_name, new_edge = self.gen.gen_unsqueeze(
                deada.name, deada.shape, 4)
        
        for i in new_edge:
            edges.append(i)

        out_deada = EdgeNode(out_name, out_shape, node, deada.zero)
        edges.append(out_deada)

        add_edge, new_edges = self.make_dead.gen_dead(self.op_type.name, edge_a = out_deada, all_edges=edges)
        for i in new_edges:
            edges.append(i)
        # add_edge = self.op_layer(dead)

        absoc_a_shape, add_edge_shape = self.gen.handle_shape_strict(absoc_a.shape, add_edge.shape)
        absoc_a,new_edge = self.gen.unilateral_shape_matching(edges, absoc_a, absoc_a_shape, True)
        for i in new_edge:
            edges.append(i)
        # edges.append(uoc)
        add_edge,new_edge = self.gen.unilateral_shape_matching(edges, add_edge, add_edge_shape, True)
        for i in new_edge:
            edges.append(i)
        # absoc_a, add_edge = handle_shape_strict(absoc_a, add_edge)
        # dead, uoc = handle_shape_final(dead, uoc)

        out0 = self.gen.make_edge_node(
            'Mul', [absoc_a, add_edge], c_shape, True)
        
        edges.append(out0)
        # out = torch.mul(absoc_a, add_edge)

        # dtype = deadb.dtype
        # out_shape, deadb_shape = self.gen.handle_shape_strict(out.shape, deadb.shape)

        out,new_edge = self.gen.unilateral_shape_matching(edges, out0, sub_node.shape, True)
        for i in new_edge:
            if i.name != out0.name:
                edges.append(i)
        # deadb,new_edge = self.gen.unilateral_shape_matching(edges, deadb, deadb_shape, True)
        # for i in new_edge:
        #     edges.append(i)
        # out, deadb = handle_shape_final(out, deadb)
        out,add_edge = self.gen.make_subs_add(sub_node, out)
        # out = self.gen.make_edge_node(
        #     'Add', [out, deadb], deadb.shape,
        #     True
        # )
        edges.append(out)
        edges.append(add_edge)
        # out = torch.add(out, deadb)
        # print("uoc dtype", uoc.dtype)
        # out = out.to(dtype)
        # print("ABSOC_A out shape: ", out.shape)
        return edges


class ABSOC_B():
    def __init__(self, op_type, max_node_idx , max_edge_idx):
        super().__init__()
        self.__name__ = "ABSOC_B"
        self.op_type = op_type
        self.gen = NodeChainGen(max_node_idx ,max_edge_idx)
        self.make_dead = DeadGenerator(self.gen)

    def run_ABSOC_B(self, a, b, deada, deadb, sub_node):
        # print("deada", type(deada)) Stubtensor
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb

        edges, (edge_a, edge_b), c_shape = self.gen.bilateral_shape_matching(
            [a, b], True)
        # a, b = handle_shape_strict(a, b)

        a1_edge = self.gen.make_edge_node(
            'abs', [edge_a], edge_a.shape, True)
        edges.append(a1_edge)
        # a1 = torch.abs(a) 

        b1_edge = self.gen.make_edge_node(
            'abs', [edge_b], edge_b.shape, True)
        edges.append(b1_edge)
        # b1 = torch.abs(b)

        a1b1 = self.gen.make_edge_node(
            'sub', [a1_edge, b1_edge], a1_edge.shape, True)
        edges.append(a1b1)
        # a1b1 = torch.sub(a1, b1) # |a|-|b|

        a1b1 = self.gen.make_edge_node(
            'abs', [a1b1], a1b1.shape, True)
        edges.append(a1b1)
        # a1b1 = torch.abs(a1b1) # ||a|-|b||

        ab = self.gen.make_edge_node(
            'Add', [edge_a, edge_b], edge_b.shape,
            True
        )
        edges.append(ab)

        ab = self.gen.make_edge_node(
            'abs', [ab], ab.shape, True)
        edges.append(ab)
        # ab = torch.abs(torch.add(a, b)) # |a+b|

        absoc_b = self.gen.make_edge_node(
            'sub', [a1b1, ab], ab.shape, True)
        edges.append(absoc_b)
        # absoc_b = torch.sub(a1b1, ab) # ||a|-|b|| - |a+b|

        exp_e = self.gen.make_constant(
            np.array([1e-5], dtype=np.float32))
        edges.append(exp_e)

        absoc_b = self.gen.make_edge_node(
            'sub', [absoc_b, exp_e], absoc_b.shape, True)
        edges.append(absoc_b)
        # absoc_b = torch.sub(absoc_b, 1e-5) # ||a|-|b|| - |a+b| - 1e-5

        absoc_b = self.gen.make_edge_node('Relu', absoc_b, absoc_b.shape, True)
        # absoc_b = nn.ReLU()(absoc_b) # relu(||a|-|b|| - |a+b| - 1e-5)

        # print("ABSOC_A", ABSOC_A.shape)

        node, out_shape, out_name, new_edge = self.gen.gen_unsqueeze(
                deada.name, deada.shape, 4)
        
        for i in new_edge:
            edges.append(i)
        out_deada = EdgeNode(out_name, out_shape, node, deada.zero)
        edges.append(out_deada)
        # dead = make_unsqueeze(deada)

        add_edge, new_edges = self.make_dead.gen_dead(self.op_type.name, edge_a = out_deada, all_edges=edges)
        for i in new_edges:
            edges.append(i)
        # add_edge = self.op_layer(dead)

        absoc_b_shape, add_edge_shape = self.gen.handle_shape_strict(absoc_b.shape, add_edge.shape)
        absoc_b,new_edge = self.gen.unilateral_shape_matching(edges, absoc_b, absoc_b_shape, True)
        for i in new_edge:
            edges.append(i)
        # edges.append(uoc)
        add_edge,new_edge = self.gen.unilateral_shape_matching(edges, add_edge, add_edge_shape, True)
        for i in new_edge:
            edges.append(i)
        # absoc_b, add_edge = handle_shape_strict(absoc_b, add_edge)


        # dead, uoc = handle_shape_final(dead, uoc)

        out0 = self.gen.make_edge_node(
            'Mul', [absoc_b, add_edge], c_shape, True)
        edges.append(out0)
        # out = torch.mul(absoc_b, add_edge)

        # dtype = deadb.dtype
        out,new_edge = self.gen.unilateral_shape_matching(edges, out0, sub_node.shape, True)
        for i in new_edge:
            if i.name != out0.name:
                edges.append(i)
        # out, deadb = handle_shape_final(out, deadb)

        out,add_edge = self.gen.make_subs_add(sub_node, out)
        edges.append(out)
        edges.append(add_edge)
        # out = torch.add(out, deadb)
        # print("uoc dtype", uoc.dtype)
        # out = out.to(dtype)
        # print("ABSOC_A out shape: ", out.shape)
        return edges


class PIOC():
    def __init__(self, op_type, max_node_idx , max_edge_idx, input_data, model_dir, netname):
        super().__init__()
        self.__name__ = "PIOC"
        self.op_type = op_type
        self.netname = netname
        self.gen = NodeChainGen(max_node_idx ,max_edge_idx)
        self.tmp_path = model_dir + self.netname + "_" + 'seed' + ".onnx"
        self.input_data = input_data
        self.make_dead = DeadGenerator(self.gen)

    def run_PIOC(self, input, deada, final_dead, deadb, sub_node):
        edges = []
        reduce_edge, edges = self.gen.make_reduce(edges, input)
        # for i in edges:
        edges.append(reduce_edge)
        # if boo:
        #     edges.append(reduce_edge)
        # new_model = 
        # print(self.tmp_path)
        # model = onnx.load(self.tmp_path)
        # # print(model)
        # sess = rt.InferenceSession(model.SerializeToString(),providers=['CUDAExecutionProvider'])
        input_data = self.input_data

        # const_np = reduce_edge.detach().cpu().numpy()

        const_edge = self.gen.make_constant(input_data)
        edges.append(const_edge)
        # const_edge = torch.tensor(const_np, dtype=dtype).to(device)

        sub_edge = self.gen.make_edge_node(
            'sub', [reduce_edge, const_edge], reduce_edge.shape, True)
        edges.append(sub_edge)
        # sub_edge = torch.sub(reduce_edge, const_edge)

        if self.op_type == "Add":

            deada_shape, deadb_shape = self.gen.handle_shape_strict(deada.shape, deadb.shape)
            deada,new_edge = self.gen.unilateral_shape_matching(edges, deada, deada_shape, True)
            for i in new_edge:
                edges.append(i)
            # edges.append(uoc)
            deadb,new_edge = self.gen.unilateral_shape_matching(edges, deadb, deadb_shape, True)
            for i in new_edge:
                edges.append(i)
            # deada, deadb = handle_shape_strict(deada, deadb)

            add_edge = self.gen.make_edge_node(
                'add', [deada, deadb], deada.shape, True)
            edges.append(add_edge)
            # add_edge = torch.add(deada, deadb)


            # print("old add_edge shape is", add_edge.shape)
            # print("old sub_edge shape is", sub_edge.shape)
            add_edge_shape, sub_edge_shape = self.gen.handle_shape_strict(add_edge.shape, sub_edge.shape)
            add_edge,new_edge = self.gen.unilateral_shape_matching(edges, add_edge, add_edge_shape, True)
            for i in new_edge:
                edges.append(i)
            # edges.append(uoc)
            sub_edge,new_edge = self.gen.unilateral_shape_matching(edges, sub_edge, sub_edge_shape, True)
            for i in new_edge:
                edges.append(i)
            # add_edge, sub_edge = handle_shape_strict(add_edge, sub_edge)

            # print("add_edge shape is", add_edge.shape)
            # print("sub_edge shape is", sub_edge.shape)

            mul_edge = self.gen.make_edge_node(
                'mul', [add_edge, sub_edge], sub_edge.shape, True)
            edges.append(mul_edge)
            # mul_edge = torch.mul(add_edge, sub_edge)

            # print("mul_edge shape is", mul_edge.shape)
        # elif self.op_type in ["DenseLayer", "SELayer", "Inception_A", "PWDWPW_ResidualBlock",
        #                       "ResidualBlock", "DropPath", "Dense"]:
        else:
            deada_shape, deadb_shape = self.gen.handle_shape_strict(deada.shape, deadb.shape)
            deada,new_edge = self.gen.unilateral_shape_matching(edges, deada, deada_shape, True)
            for i in new_edge:
                if i.name != deada.name:
                    edges.append(i)
            # edges.append(uoc)
            # deadb,new_edge = self.gen.unilateral_shape_matching(edges, deadb, deadb_shape, True)
            # for i in new_edge:
            #     if i.name != deadb.name:
            #         edges.append(i)

            # deada, _ = handle_shape_strict(deada, deada)
            node, out_shape, out_name, new_edge = self.gen.gen_unsqueeze(
                    deada.name, deada.shape, 4)
            for i in new_edge:
                if i.name != deada.name:
                    edges.append(i)
            # deada = EdgeNode(out_name, out_shape, node, deada.zero)
            # edges.append(deada)
            # deada = make_unsqueeze(deada)
            
            add_edge, new_edges = self.make_dead.gen_dead(self.op_type.name, edge_a = deada, all_edges=edges)
            for i in new_edges:
                edges.append(i)
            # add_edge = self.op_layer(deada)

            add_edge_shape, sub_edge_shape = self.gen.handle_shape_strict(add_edge.shape, sub_edge.shape)
            add_edge,new_edge = self.gen.unilateral_shape_matching(edges, add_edge, add_edge_shape, True)
            for i in new_edge:
                edges.append(i)
            sub_edge,new_edge = self.gen.unilateral_shape_matching(edges, sub_edge, sub_edge_shape, True)
            for i in new_edge:
                edges.append(i)

            mul_edge = self.gen.make_edge_node(
                'mul', [add_edge, sub_edge], sub_edge.shape, True)
            # edges.append(mul_edge)

            # sub_edge, add_edge = handle_shape_strict(sub_edge, add_edge)
            # mul_edge = torch.mul(add_edge, sub_edge)
        # else:
        #     raise NotImplementedError("optype Not Implemented for optype: {}".format(self.op_type))

        # else:
        #     raise ValueError("PIOC len Not Implemented")
        # print("mul_edge is", mul_edge.shape)
        # print("final_dead is", final_dead.shape)
        mul_edge_shape, final_dead_shape = self.gen.handle_shape_strict(mul_edge.shape, final_dead.shape)
        mul_edge,new_edge = self.gen.unilateral_shape_matching(edges, mul_edge, mul_edge_shape, True)
        for i in new_edge:
            edges.append(i)

        final_dead,new_edge = self.gen.unilateral_shape_matching(edges, final_dead, final_dead_shape, True)
        for i in new_edge:

            edges.append(i)
        for i in new_edge:
            if i.name == final_dead.name:
                edges.remove(i)
                break
        # mul_edge_1, final_dead_1 = handle_shape_final(mul_edge, final_dead)

        out1 = self.gen.make_edge_node(
            'Mul', [mul_edge, final_dead], final_dead.shape, True)
        
        edges.append(out1)
        # out = torch.add(mul_edge_1, final_dead_1)

        out,new_edge = self.gen.unilateral_shape_matching(edges, out1, sub_node.shape, True)
        for i in new_edge:
            edges.append(i)

        out,add_edge = self.gen.make_subs_add(sub_node, out)
        edges.append(out)
        edges.append(add_edge)
        # pioc_equal = np.allclose(out.detach().cpu().numpy(), final_dead_1.detach().cpu().numpy())
        # if not pioc_equal:
        #     print("pioc_equal", pioc_equal)
        #     print("pioc distance", ChebyshevDistance(out, final_dead_1))
        # assert pioc_equal
        # print("pioc out shape is", out.shape)
        # print("pioc dtype is", out.dtype)

        return edges
    

class Hybrid():
    def __init__(self, op_type, max_node_idx , max_edge_idx, input_data, model_dir, netname):

        self.uoc = UOC(op_type, max_node_idx , max_edge_idx)

        self.pioc = PIOC(op_type, max_node_idx , max_edge_idx, input_data, model_dir, netname)
        
        self.absoc_a = ABSOC_A(op_type, max_node_idx , max_edge_idx)

        self.absoc_b = ABSOC_B(op_type, max_node_idx , max_edge_idx)

        self.methods = ['uoc', 'pioc', 'absoc_a', 'absoc_b']
        
        self.methods = random.choice(self.methods)

    def run_Hybrid(self, a, b, deada, deadb, sub_node):
        if self.methods == 'uoc':
            return self.uoc.run_uoc(a, b, deada, deadb, sub_node)
        
        elif self.methods == 'pioc':
            return self.pioc.run_PIOC(a, b, deada, deadb, sub_node)
        
        elif self.methods == 'absoc_a':
            return self.absoc_a.run_ABSOC_A(a, b, deada, deadb, sub_node)
        
        elif self.methods == 'absoc_b':
            return self.absoc_b.run_ABSOC_B(a, b, deada, deadb, sub_node)



