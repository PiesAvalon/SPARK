from onnx_mutation.node_gen import NodeChainGen
import random
import numpy as np
import onnx
from onnx_mutation.edge_node import EdgeNode
import torch
class DeadGenerator:
    def __init__(self, generator: NodeChainGen):
        self.gen = generator
        self.op_types = ['Dense', 'SELayer', 'DenseLayer', 'Inception_A', 'PWDWPW_ResidualBlock',
            'ResidualBlock', 'DropPath']
        self.kernel_size = [1, 'all', 3]

    def gen_dead_edge(self, op_type, edge_a, edge_b, all_edges):
        zero = (edge_a.zero or edge_b.zero) if op_type == 'Mul' \
            else (edge_a.zero and edge_b.zero)
        return self.gen.make_multi_input_node(
            op_type, [edge_a, edge_b], True, zero)

    def gen_dead(self, op_type, edge_a, all_edges):
        if op_type == 'Dense':
            # print('Dense')
            new_edges = []
            Dense_node, new_edges = self.gen_dense(new_edges, edge_a, all_edges)
            return Dense_node, new_edges
        elif op_type == 'SELayer':
            # print('SELayer')
            new_edges = []
            Dense_node, new_edges = self.gen_SELayer(new_edges, edge_a, all_edges)
            return Dense_node, new_edges 
        elif op_type == 'DenseLayer':
            # print('DenseLayer')
            new_edges = []
            Dense_node, new_edges = self.gen_DenseLayer(new_edges, edge_a, all_edges)
            return Dense_node, new_edges 
        elif op_type == 'Inception_A':
            # print('Inception_A')
            new_edges = []
            Dense_node, new_edges = self.gen_Inception_A(new_edges, edge_a, all_edges)
            return Dense_node, new_edges 
        elif op_type == 'PWDWPW_ResidualBlock':
            # print('PWDWPW_ResidualBlock')
            new_edges = []
            Dense_node, new_edges = self.gen_PWDWPW_ResidualBlock(new_edges, edge_a, all_edges)
            return Dense_node, new_edges 
        elif op_type == 'ResidualBlock':
            # print('ResidualBlock')
            new_edges = []
            Dense_node, new_edges = self.gen_ResidualBlock(new_edges, edge_a, all_edges)
            return Dense_node, new_edges 
        elif op_type == 'DropPath':
            new_edges = []
            # print('DropPath')
            Dense_node, new_edges = self.gen_DropPath(new_edges, edge_a, all_edges)
            return Dense_node, new_edges 

    
    def gen_dense(self, new_edges, in_edge, all_edges):
        # self, new_edges, in_edge, tgt_rank
        edge, edge3 = self.gen.make_unsqueeze(new_edges, in_edge, tgt_rank=2)
 
        if len(edge3) > 0:
            for i in edge3:
                new_edges.append(i)
        num_features = edge.shape[-1]

        mul_val = np.random.randn(num_features, num_features).astype(np.float32)
        mul_val_edge = self.gen.make_constant(mul_val)
        new_edges.append(mul_val_edge)

        mul_out = self.gen.make_edge_node(
            'MatMul', [edge, mul_val_edge], edge.shape, edge.zero)
        new_edges.append(mul_out)

        add_val = np.random.randn(num_features).astype(np.float32)
        add_val_edge = self.gen.make_constant(add_val)
        new_edges.append(add_val_edge)

        add_out = self.gen.make_edge_node(
            'Add', [mul_out, add_val_edge], mul_out.shape, False)
        # new_edges.append(add_out)
        return add_out, new_edges
    
    def gen_SELayer(self, new_edges, in_edge, all_edges):
        out = self.gen.make_edge_node(
            'GlobalAveragePool', [in_edge], in_edge.shape, False)
        new_edges.append(out)

        out = self.gen.make_conv(new_edges, in_edge, self.foo(out.shape))

        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        
        out = self.gen.make_conv(new_edges, in_edge, self.foo(out.shape))

        out = self.gen.make_edge_node(
            'Hardsigmoid', [out], out.shape, False)
        new_edges.append(out)

        return out, new_edges
    
    def foo(self, out):
        out = tuple(out[:2])+ (1, 1)
        return np.array(out).astype(np.float32)
    
    def foo2(self, out):
        out = tuple(out[1:2]) + tuple(out[1:2]) + (1, 1)

        return np.array(out).astype(np.float32)

    def gen_DenseLayer(self, new_edges, in_edge, all_edges):
        out = self.gen.make_batch_norm(new_edges, in_edge)

        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))

        out = self.gen.make_batch_norm(new_edges, out)

        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))

        out = self.gen.make_edge_node('dropout', out, out.shape, True)
        new_edges.append(out)

        running_mean= np.array(1, dtype=np.float32)

        running_mean = self.gen.make_constant(running_mean)
        new_edges.append(running_mean)

        out = self.gen.make_edge_node('Concat', [in_edge, out, running_mean], out.shape, True)
        new_edges.append(out)
        return out, new_edges
    
    def gen_Inception_A(self, new_edges, in_edge, all_edges):

        out = self.gen.make_conv(new_edges, in_edge, self.foo2(in_edge.shape))
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)
        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)
        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)
        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        out = self.gen.make_edge_node('AveragePool', out, out.shape, True)
        new_edges.append(out)

        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)
        return out, new_edges
    
    def gen_PWDWPW_ResidualBlock(self, new_edges, in_edge, all_edges):
        out = self.gen.make_conv(new_edges, in_edge, self.foo2(in_edge.shape))
        out = self.gen.make_batch_norm(new_edges, out)
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))
        out = self.gen.make_batch_norm(new_edges, out)
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))
        out = self.gen.make_batch_norm(new_edges, out)
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        add_out = self.gen.make_edge_node(
            'Add', [out, in_edge], in_edge.shape, False)
        # new_edges.append(add_out)

        return add_out, new_edges
    

    def gen_ResidualBlock(self, new_edges, in_edge, all_edges):

        out = self.gen.make_conv(new_edges, in_edge, self.foo2(in_edge.shape))

        out = self.gen.make_batch_norm(new_edges, out)
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))
        out = self.gen.make_batch_norm(new_edges, out)
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        out = self.gen.make_conv(new_edges, out, self.foo2(out.shape))
        out = self.gen.make_batch_norm(new_edges, out)
        out = self.gen.make_edge_node('Relu', out, out.shape, True)
        new_edges.append(out)

        return out, new_edges
    
    def gen_DropPath(self, new_edges, in_edge, all_edges):
        keep_prob = np.array(0.5).astype(np.float32)
        keep_prob = self.gen.make_constant(keep_prob)
        new_edges.append(keep_prob)

        shape = (in_edge.shape[0],) + (1,) * (len(in_edge.shape) - 1)
        # print(shape)
        random_numpy = np.random.randn(*shape).astype(np.float32)
        random_numpy = self.gen.make_constant(random_numpy)
        new_edges.append(random_numpy)

        random_numpy = self.gen.make_edge_node(
            'Add', [keep_prob, random_numpy], random_numpy.shape, False)
        new_edges.append(random_numpy)

        random_tensor = self.gen.make_edge_node('Floor', random_numpy, random_numpy.shape, False)
        new_edges.append(random_tensor)

        out = self.gen.make_edge_node('Div', [in_edge, keep_prob], in_edge.shape, False)
        new_edges.append(out)

        out = self.gen.make_edge_node('Mul', [out, random_tensor], out.shape, False)
        # new_edges.append(out)

        return out, new_edges
    
    def get_kernel_shape(self, in_shape):
        min_dim = min(in_shape[2:])
        if min_dim < 3:
            k_size = self.kernel_size[random.randint(0, 1)]
        else:
            k_size = self.kernel_size[random.randint(0, 2)]

        if k_size == 'all':
            kernel_shape = in_shape[1], in_shape[1], in_shape[2], in_shape[3]
        else:
            kernel_shape = in_shape[1], in_shape[1], k_size, k_size
        return kernel_shape

    def gen_conv(self, new_edges, edge):
        rank_edge = self.gen.match_rank(new_edges, edge, 4)

        kernel_shape = self.get_kernel_shape(rank_edge.shape)
        np_kernel_val = np.random.randn(*kernel_shape).astype(np.float32)

        conv_edge = self.gen.make_conv(new_edges, rank_edge, np_kernel_val)
        return conv_edge

