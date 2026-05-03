import onnx

import onnx_mutation.utils.onnx_utils
from onnx_mutation import mutate_utils, attr_gen, shape_utils
from onnx_mutation.edge_node import EdgeNode
import copy
import numpy as np
import onnx.numpy_helper
import torch
class ElementGen:
    def __init__(self, next_node_idx, next_edge_idx):
        self.node_id = next_node_idx
        self.edge_id = next_edge_idx

    def new_node_name(self, node_type):
        self.node_id += 1
        return "%s_%d" % (node_type, self.node_id - 1)

    def new_edge_name(self):
        self.edge_id += 1
        return str(self.edge_id - 1)

    @staticmethod
    def new_tensor_name(node_name, attr_name):
        return "%s_%s" % (node_name, attr_name)

    def new_node(self, node_type, input_edges: list, **kwargs):
        print(node_type,
            input_edges,
            [self.new_edge_name()],
            self.new_node_name(node_type),)        
        node = onnx.helper.make_node(
            node_type,
            input_edges,
            [self.new_edge_name()],
            self.new_node_name(node_type),
            **kwargs
        )
        return node

    def new_node_specifying_output(self, node_type, input_edges: list,
                                   output_edge: str, **kwargs):
        node = onnx.helper.make_node(
            node_type,
            input_edges,
            [output_edge],
            self.new_node_name(node_type),
            **kwargs
        )
        return node

    @staticmethod
    def new_tensor(np_val, node_name, attr_name):
        data_type = mutate_utils.numpy_onnx_type_mapping(np_val.dtype)

        return onnx.helper.make_tensor(
            name=ElementGen.new_tensor_name(node_name, attr_name),
            data_type=data_type,
            dims=np_val.shape,
            vals=np_val.flatten()
        )


class NodeGen:
    def __init__(self, st_node_idx, st_edge_idx):
        self.elem_gen = ElementGen(st_node_idx, st_edge_idx)

    def gen_slice(self, input_name, src_shape, tgt_shape, broadcast=False):
        slice_shape = shape_utils.get_slice_shape(
            src_shape, tgt_shape, broadcast)
        if slice_shape != src_shape:
            new_constant_node = []
            op_type = 'Slice'
            slice_dict = attr_gen.slice_node(src_shape, slice_shape)

            # STARTS
            axes_data = slice_dict['starts']

            val = np.array(axes_data, dtype=np.int64)
            new_node1 = self.gen_constant(val)
            new_edge11 = EdgeNode(new_node1.output[0], val.shape, new_node1, False)
            new_constant_node.append(new_edge11)
            # ANDS
            axes_data = slice_dict['ends']
            val = np.array(axes_data, dtype=np.int64)
            new_node2 = self.gen_constant(val)
            new_edge21 = EdgeNode(new_node2.output[0], val.shape, new_node2, False)
            new_constant_node.append(new_edge21)

            # AXES  
            axes_data = slice_dict['axes']
            val = np.array(axes_data, dtype=np.int64)
            new_node3 = self.gen_constant(val)
            new_edge31 = EdgeNode(new_node3.output[0], val.shape, new_node3, False)
            new_constant_node.append(new_edge31)

            input_names = []
            input_names.append(input_name)
            input_names.append(new_node1.output[0])
            input_names.append(new_node2.output[0])
            input_names.append(new_node3.output[0])

            slice_node = self.make_node(op_type, input_names, slice_dict)
            new_edge4 = EdgeNode(slice_node.output[0], slice_shape, slice_node, False)
            new_constant_node.append(new_edge4)

            return slice_node, slice_shape, slice_node.output[0], new_constant_node
        
        else:

            return None, src_shape, input_name,[]

    def gen_unsqueeze(self, input_name, src_shape, tgt_rank):
        if len(src_shape) < tgt_rank:
            # print('111111111111111111111111111111111111111')
            new_constant_node = []
            # print(src_shape, tgt_rank)
            axes, unsqueeze_shape = attr_gen.unsqueeze_node(src_shape, tgt_rank)
            # 创建 axes 输入张量
            # print('axes')
            # print(axes)
            axes_data = axes['axes'].tolist()  # 以示例为准，这里使用 0 和 1 作为要展开的维度
            val = np.array(axes_data, dtype=np.int64)
            # print('val')
            # print(val)
            # axes = np.array([2, 3], dtype=np.int64)

            # print(new_node)
            # new_node = self.gen_constant(np.array(axes_data, dtype=np.float32))
            new_node = self.gen_constant(val)
            new_edge = EdgeNode(new_node.output[0], val.shape, new_node, False)
            new_constant_node.append(new_edge)
            # print('111111111111111111111111111111111111111')
            # print(new_constant_node)
            # axes_tensor = onnx.numpy_helper.from_list(axes_data, name='axes_tensor')
            input = []
            input.append(input_name)
            input.append(new_node.output[0])
            # print(input)
            node = self.make_node('Unsqueeze', input, None)
            new_edge = EdgeNode(node.output[0], unsqueeze_shape, node, False)
            new_constant_node.append(new_edge)
            # print('111111111111111111111111111111111111111')
            # print(new_constant_node)
            return node, unsqueeze_shape, node.output[0], new_constant_node
        else:
            return None, src_shape, input_name, []

    def gen_reduce(self, input_name, in_shape,
                   reduce='mean', keep_dims=False, rank=2):
        assert reduce.lower() in ['mean', 'max', 'min', 'l1', 'l2', 'sum']
        op_type = "Reduce%s%s" % (reduce[0].upper(), reduce[1:])
        if len(in_shape) > rank:
            attr, reduce_shape = attr_gen.reduce_node(in_shape, keep_dims, rank)
            node = self.make_node(op_type, input_name, attr)
            return node, reduce_shape, node.output[0]
        else:
            return None, in_shape, input_name

    def gen_pad(self, in_name, src_shape, tgt_shape,
                broadcast, mode='constant'):
        pad_shape = shape_utils.get_pad_shape(src_shape, tgt_shape, broadcast)
        if pad_shape != src_shape:

            pad_dict = attr_gen.pad_node(src_shape, pad_shape, mode)
            val = np.array(pad_dict['pads'], dtype=np.int64)
            pads = self.gen_constant(val)
            pads1 = EdgeNode(pads.output[0], val.shape, pads, False)
            
            node = self.make_node('Pad', [in_name, pads.output[0]], None)
            return node, pad_shape, node.output[0], pads1
        else:
            return None, pad_shape, in_name, None

    def gen_constant(self, val):
        attr_dict = {'value': self.elem_gen.new_tensor(
            val, self.elem_gen.new_node_name('Constant'), 'tensor'
        )}
        # print(attr_dict)
        return self.make_node('Constant', [], attr_dict)

    def gen_conv(self, in_name, in_shape, kernel_name, kernel_shape):
        attr, out_shape = attr_gen.conv_node(in_shape, kernel_shape)

        node = self.make_node('Conv', [in_name, kernel_name], attr)
        return node, out_shape, node.output[0]

    def make_node(self, op_type, input_edges_name, attr_dict=None):
        input_edges_name = mutate_utils.convert2iter(input_edges_name)

        if attr_dict is not None:
            node = self.elem_gen.new_node(op_type, input_edges_name, **attr_dict)
        else:
            node = self.elem_gen.new_node(op_type, input_edges_name)
        return node


class NodeChainGen(NodeGen):
    def make_batch_norm(self, new_edges, in_edge):

        running_mean1 = np.zeros(in_edge.shape[1], dtype=np.float32)
        scale = self.make_constant(running_mean1)
        new_edges.append(scale)
        input_B = self.make_constant(running_mean1)
        new_edges.append(input_B)
        input_mean = self.make_constant(running_mean1)
        new_edges.append(input_mean)
        input_var = self.make_constant(running_mean1)
        new_edges.append(input_var)

        LIST1 = []
        LIST1.append(in_edge)
        LIST1.append(scale)
        LIST1.append(input_B)
        LIST1.append(input_mean)
        LIST1.append(input_var)
        out = self.make_edge_node('BatchNormalization',LIST1,in_edge.shape,False)
        new_edges.append(out)

        return out
    
    def make_conv(self, new_edges, in_edge, np_kernel_val):
        weight = self.make_constant(np_kernel_val)
        new_edges.append(weight)
        
        conv_node, conv_out_shape, conv_out_name = self.gen_conv(
            in_edge.name, in_edge.shape, weight.name, weight.shape
        )

        conv_edge = EdgeNode(conv_out_name, conv_out_shape, conv_node,
                             in_edge.zero or mutate_utils.is_val_zero(np_kernel_val))
        new_edges.append(conv_edge)
        return conv_edge
    
    def make_unsqueeze(self, new_edges, in_edge, tgt_rank):
        # print('12333333333333333333333333')
        # print(len(in_edge.shape), tgt_rank)
        
        if len(in_edge.shape) < tgt_rank:
            rank_node, edge_shape, edge_name, new_edge = self.gen_unsqueeze(
                in_edge.name, in_edge.shape, tgt_rank)
            edge = EdgeNode(edge_name, edge_shape, rank_node)
            
            return edge, new_edge
        return in_edge, []

    def make_reduce(self, new_edges: list, input_edge,
                    reduce='mean', keep_dims=False, rank=2):
        node, edge_shape, edge_name = self.gen_reduce(
            input_edge.name, input_edge.shape, reduce, keep_dims, rank
        )

        if node:
            new_edge = EdgeNode(edge_name, edge_shape, node, input_edge.zero)
            # new_edges.append(new_edge)
            return new_edge, new_edges
        else:
            return input_edge, new_edges

    def make_constant(self, val):
        new_node = self.gen_constant(val)
        new_edge = EdgeNode(new_node.output[0], val.shape, new_node, False)
        return new_edge

    def make_subs_add(self, subs_edge, dead_edge):
        ori_output_name = subs_edge.name
        subs_edge = self.substitute_edge(subs_edge)
        add_node = self.elem_gen.new_node_specifying_output(
            'Add', [dead_edge.name, subs_edge.name],
            ori_output_name
        )
        add_edge = EdgeNode(ori_output_name, subs_edge.shape, add_node,
                            subs_edge.zero and dead_edge.zero)
        print(add_edge.def_node)
        return subs_edge, add_edge

    def substitute_edge(self, substituted_edge):
        node = substituted_edge.def_node
        print(node)
        new_output_name = self.elem_gen.new_edge_name()
        print('111111111111111111111111111111111111111')
        node = mutate_utils.replace_node_output(node, new_output_name)
        print(node)
        new_output_edge = EdgeNode(
            new_output_name, substituted_edge.shape, node, substituted_edge.zero
        )
        # substituted_edge.def_node = None
        return new_output_edge

    def make_edge_node(self, op_type, in_edges, out_shape, zero):
        in_edges = mutate_utils.convert2iter(in_edges)
        node = self.make_node(op_type, [e.name for e in in_edges], None)
        edge = EdgeNode(node.output[0], out_shape, node, zero)
        return edge

    def make_multi_input_node(self, op_type, in_edges, broadcast, out_zero):
        edges, node_in_edges, common_shape = self.bilateral_shape_matching(
            in_edges, broadcast)

        agg_edge = self.make_edge_node(
            op_type, node_in_edges, common_shape, out_zero)
        edges.append(agg_edge)
        return edges

    def unilateral_shape_matching(self, new_edges, in_edge,
                                  tgt_shape, broadcast):
        new_edge = []
        edge,eeeeeeeeee = self.match_rank(new_edges, in_edge, len(tgt_shape))
        for i in eeeeeeeeee:
            # if i.name != in_edge.name:
            new_edge.append(i)

        edge_name, edge_shape = edge.name, edge.shape

        slice_node, edge_shape, edge_name, new_edge111 = self.gen_slice(
            edge_name, edge_shape, tgt_shape, False)
        
        if slice_node:
            edge = EdgeNode(edge_name, edge_shape, slice_node)
            # new_edge.append(edge)
            for i in new_edge111:
                new_edge.append(i)

        pad_node, edge_shape, edge_name, pads = self.gen_pad(
            edge_name, edge_shape, tgt_shape, broadcast)
        
        if pad_node:
            edge = EdgeNode(edge_name, edge_shape, pad_node)
            new_edge.append(pads)
            new_edge.append(edge)

        return edge,new_edge

    def match_rank(self, new_edges, in_edge, tgt_rank):
        in_name, src_shape = in_edge.name, in_edge.shape
        if len(src_shape) < tgt_rank:
            # print('len(src_shape) < tgt_rank')
            edge,eeeeeeeeee = self.make_unsqueeze(new_edges, in_edge, tgt_rank)
            return edge,eeeeeeeeee
        elif len(src_shape) > tgt_rank:
            # print('len(src_shape) > tgt_rank')
            edge, eeeeee = self.make_reduce(new_edges, in_edge, 'max', rank=tgt_rank)

        else:
            edge = in_edge
        return edge,[edge]

    def bilateral_shape_matching(self, in_edges: list, broadcast):
        shape_list = [e.shape for e in in_edges]
        common_shape = shape_utils.get_common_shape(shape_list, broadcast)
        new_edges, out_edges = [], []
        for in_edge in in_edges:
            out_edge = in_edge
            node, out_shape, out_name, nodes = self.gen_unsqueeze(
                in_edge.name, in_edge.shape, len(common_shape))
            if node:
                out_edge = EdgeNode(out_name, out_shape, node, in_edge.zero)
                for i in nodes:
                    new_edges.append(i)

            node, out_shape, out_name, new_123= self.gen_slice(
                out_name, out_shape, common_shape, broadcast)
            if node:
                out_edge = EdgeNode(out_name, out_shape, node, in_edge.zero)
                for i in new_123:
                    new_edges.append(i)
                # new_edges.append(out_edge)

            out_edges.append(out_edge)

        return new_edges, out_edges, common_shape
    
    def find_unsqueeze(self, in_tensor):
        if len(in_tensor.shape) == 2:
            return 2
        if len(in_tensor.shape) == 4:
            return 0
        if len(in_tensor.shape) == 3:
            return 1
        if len(in_tensor.shape) == 1:
            return 3
        print("len error length:", len(in_tensor.shape))
        raise ValueError("make_unsqueeze len error")

    def handle_shape_strict(self, a_shape, b_shape):
        # print('len(a_shape),len(b_shape')
        # print(len(a_shape),len(b_shape))
        # print(type(a_shape))

        if len(a_shape) == 1:
            a_shape = a_shape + (1,)
        if len(b_shape) == 1:
            b_shape = b_shape + (1,)
        if len(a_shape) == 3:
            a_shape = a_shape[:2] + (1,)
        if len(b_shape) == 3:
            b_shape = b_shape[:2] + (1,)
        if a_shape == b_shape:
            return a_shape, b_shape
        
        if len(a_shape) == 2 and len(b_shape) == 2:
            if a_shape[1] < b_shape[1]:
                new_tuple = tuple(max(x, y) for x, y in zip(a_shape, b_shape))
                return a_shape, b_shape
            else:
                new_tuple = tuple(max(x, y) for x, y in zip(a_shape, b_shape))
                return a_shape, b_shape
        elif len(a_shape) == 4 and len(b_shape) == 4:
            new_tuple = tuple(max(x, y) for x, y in zip(a_shape, b_shape))
            return new_tuple, new_tuple
        elif len(a_shape) == 2:
            a_shape = np.array(a_shape)
            b_shape = np.array(b_shape)
            for i in range(4 - len(a_shape)):
                a_shape = np.expand_dims(a_shape, axis=-1)
            for i in range(4 - len(b_shape)):
                b_shape = np.expand_dims(b_shape, axis=-1)
            # 将列表再转换回元组

            print(len(a_shape), len(b_shape))
            # a_shape = a_shape[:1] + (1, 1)
            # print('a_shape')
            # print(a_shape)
            if a_shape[1] > b_shape[1]:
                a_shape = a_shape[:, :b_shape[1], :, :]
            elif a_shape[1] < b_shape[1]:
                b_shape = b_shape[:1] + a_shape[1:]
            
            if a_shape[2] < b_shape[2]:
                pad = (0, 0, 0, b_shape[2] - a_shape[2])
                a_shape = a_shape + pad
            elif a_shape[2] > b_shape[2]:
                pad = (0, 0, 0, a_shape[2] - b_shape[2])
                b_shape = b_shape + pad
            if a_shape[3] > b_shape[3]:
                pad = (a_shape[3] - b_shape[3], 0)
                b_shape = b_shape + pad
            elif a_shape[3] < b_shape[3]:
                b_shape = b_shape[:3] + (a_shape[3],)
            return a_shape, b_shape
        elif len(b_shape) == 2:
            b_shape = list(b_shape)
            # 在列表中插入 1 作为新的维度
            b_shape.extend([1, 1])
            # 将列表再转换回元组
            b_shape = tuple(b_shape)
            # b_shape = b_shape[:1] + (1, 1)
            if a_shape[1] > b_shape[1]:

                pad = (0, 0, 0, 0, 0, a_shape[1] - b_shape[1])
                b_shape = b_shape + pad
                # b_shape[1] = a_shape[1]
            elif a_shape[1] < b_shape[1]:
                b = b_shape[:, :a_shape[1], :, :]
                # b_shape[1] = a_shape[1]
                b_shape = b_shape[:1] + a_shape[1:]
            if a_shape[2] < b_shape[2]:
                b_shape = b_shape[:2] + (a_shape[2],)
                b = b_shape[:, :, :a_shape[2], :]
                # b_shape[2] = a_shape[2]
            elif a_shape[2] > b_shape[2]:
                # b_shape[2] = a_shape[2]
                pad = (0, 0, 0, a_shape[2] - b_shape[2])
                b_shape = b_shape + pad
            if a_shape[3] > b_shape[3]:
                # b_shape[3] = a_shape[3]
                pad = (a_shape[3] - b_shape[3], 0)
                b_shape = b_shape + pad
            elif a_shape[3] < b_shape[3]:
                b = b[:, :, :, :a_shape[3]]
            return a_shape, a_shape
        elif len(a_shape) == 5 and len(b_shape) == 5:
            if a_shape[-1] > b_shape[-1]:
                pad = (0, a_shape[-1] - b_shape[-1])
                b_shape = b_shape + pad
            elif a_shape[-1] < b_shape[-1]:
                b_shape = b_shape[:4] + (a_shape[-1],)
            if a_shape[-2] > b_shape[-2]:
                pad = (0, 0, 0, a_shape[-2] - b_shape[-2])
                b_shape = b_shape + pad
            elif a_shape[-2] < b_shape[-2]:
                b_shape = b_shape[:3] + (a_shape[-2], 1)
            if a_shape[2] > b_shape[2]:
                pad = (0, 0, 0, 0, 0, a_shape[2] - b_shape[2])
                b_shape = b_shape + pad
            elif a_shape[2] < b_shape[2]:
                b_shape = b_shape[:3] + (a_shape[2], 1, 1)
            if a_shape[1] > b_shape[1]:
                pad = (0, 0, 0, 0, 0, 0, 0, a_shape[1] - b_shape[1])
                b_shape = b_shape + pad
            elif a_shape[1] < b_shape[1]:
                b_shape = b_shape[:3] + (a_shape[1], 1, 1)
            return a_shape, b_shape
        else:
            # print("handle_shape error", a_shape, b_shape)
            return a_shape, b_shape
            # raise ValueError("handle_shape error", a_shape, b_shape)


def handle_tuple(outputs):
    if isinstance(outputs, (torch.Tensor,)):
        return outputs
    # elif isinstance(outputs[0], (torch.Tensor,)):
    #     length = len(outputs)
    #     ans = torch.zeros(outputs[0].shape).to(device)
    #     for out in outputs:
    #         ans, tmp = handle_shape_strict(ans, out)
    #         ans = torch.add(ans, tmp)
    #     ans = torch.div(ans, length)
    #     return ans
    # elif isinstance(outputs[0], tuple):
    #     big_length = len(outputs)
    #     ans = torch.zeros(outputs[0][0].shape).to(device)
    #     for out in outputs:
    #         temp = handle_tuple(out)
    #         # print("temp.shape: ", temp.shape)
    #         ans, tmp = handle_shape_strict(ans, temp)
    #         # print("ans.shape: ", ans.shape, "tmp.shape: ", tmp.shape)
    #         ans = torch.add(ans, tmp)
    #         # print("fucker", ans.shape)
    #     ans = torch.div(ans, big_length)
    #     return ans
    else:
        raise NotImplementedError("handle_tuple not implemented for type: ", type(outputs[0]))




def make_node_chain_generator(model):
    max_node_idx = onnx_mutation.utils.onnx_utils.get_max_node_idx(model.graph)
    max_edge_idx = onnx_mutation.utils.onnx_utils.get_max_edge_idx(model.graph)
    return max_node_idx + 1, max_edge_idx + 1
    # return NodeChainGen(max_node_idx + 1, max_edge_idx + 1)
