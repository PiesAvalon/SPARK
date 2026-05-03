import torch
import numpy as np


def check_illegal_mutant(mutant_model, mutant_output, output_time=-1):
    for val in mutant_output:
        if isinstance(val, list) or isinstance(val, tuple):

            for vall in val:
                vall_array = vall.asnumpy()
                if True in np.isnan(vall_array):
                    return False
                elif np.max(vall_array) > 1e38:
                    return False

        else:
            val_array = val.asnumpy()

            if True in np.isnan(val_array):
                return False
            elif np.max(val_array)>1e38:
                return False

    return True


def find_inshapes(net):
    layer_names = net.Basic_OPS
    for layer_name in layer_names:
        topology_info = net.get_order(layer_name)
        last_op, next_op = topology_info[0], topology_info[1]

        if isinstance(last_op, list):
            last_op = last_op[0]

        in_shape = net.out_shapes[last_op]
        net.in_shapes[layer_name] = in_shape

    shape_keys = list(net.out_shapes.keys())
    for shape_key in shape_keys:
        if "OUTPUT" in shape_key:
            net.in_shapes[shape_key] = net.out_shapes[shape_key]
        elif "INPUT" in shape_key:
            net.in_shapes[shape_key] = net.out_shapes[shape_key]


def find_Cascade_OP(layer_names):
    Cascade_ops = []
    for i in range(len(layer_names)):
        if "_del" in layer_names[i] or "empty" in layer_names[i]:
            continue
        c1 = layer_names[i].split(".")
        for j in range(i + 1, len(layer_names)):
            if "_del" in layer_names[j] or "empty" in layer_names[i]:
                continue
            c2 = layer_names[j].split(".")
            if layer_names[i] in layer_names[j] and len(c1) == len(c2) - 1:
                Cascade_ops.append(layer_names[i])
                break
    return Cascade_ops


def check_orderinfo_selfcorrect(model):
    orders = model.orders
    layer_names = list(orders.keys())
    for layer_name in layer_names:
        # print(f'debug check_orderinfo_selfcorrect layer name:{layer_name}')
        qianqu, houji = orders[layer_name]
        if isinstance(qianqu, list):
            for qianqu_single in qianqu:
                if "INPUT" in qianqu_single:
                    continue
                assert (orders[qianqu_single][1] == layer_name or layer_name in orders[qianqu_single][1])
        else:
            if not "INPUT" in qianqu:
                assert (orders[qianqu][1] == layer_name or layer_name in orders[qianqu][1])

        if isinstance(houji, list):
            for houji_single in houji:
                if "OUTPUT" in houji_single:
                    continue
                assert (orders[houji_single][0] == layer_name or layer_name in orders[houji_single][0])
        else:
            if not "OUTPUT" in houji:
                assert (orders[houji][0] == layer_name or layer_name in orders[houji][0])
    print('check_orderinfo_selfcorrect success!')


def write_setmethod(model, name, save_path):
    if "torch" in str(model.__class__.__bases__):
        layers = model.named_modules()
    elif "mindspore" in str(model.__class__.__bases__):
        layers = model.cells_and_names()

    flag = True
    f = open(save_path + "/set_method-{}.txt".format(name), "w")
    f.write("    def set_layers(self,layer_name,new_layer):\n")
    for layer_name in layers:

        layer_name = layer_name[0]
        if layer_name == "":
            continue

        elements = layer_name.split(".")
        s = "self."
        for element in elements:
            if element.isdigit():
                s = s[:-1] + "[" + element + "]."
            else:
                s = s + element + "."

        s = s[:-1] + "= new_layer\n"
        s2 = "self.layer_names[\"" + layer_name + "\"]=new_layer\n"
        s3 = "self.origin_layer_names[\"" + layer_name + "\"]=new_layer"
        if flag:
            ifs = "        if " + "\'" + layer_name + "\'" + " == layer_name:\n"
            flag = False
        else:
            ifs = "        elif " + "\'" + layer_name + "\'" + " == layer_name:\n"
        print(ifs + "            " + s + "\n")
        f.write(ifs + "            " + s + "            " + s2 + "\n")  # + "            " + s3
    f.close()


def write_layernames(model, name, save_path):
    if "torch" in str(model.__class__.__bases__):
        layers = model.named_modules()
    elif "mindspore" in str(model.__class__.__bases__):
        layers = model.cells_and_names()

    flag = True
    f = open(save_path+"/layernames_method-{}.txt".format(name), "w")
    f.write("self.layer_names{\n")
    for layer in layers:
        layer_name = layer[0]
        if layer_name == "":
            continue
        elements = layer_name.split(".")
        s = "self."
        for element in elements:
            if element.isdigit():
                s = s[:-1] + "[" + element + "]."
            else:
                s = s + element + "."

        f.write('"' + layer_name + '"' + ":" + s[:-1] + ",\n")
    f.write("}\n")
    f.close()


def check_layers_and_shapes(model):
    print(f'check layers and shapes keys')
    det = model.in_shapes.keys() - model.layer_names.keys()
    # print(f'check finish, find:')
    # for i in det:
    #     print(i)
    print(f'{len(det)} inconsistency')


class QNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_size,64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_size)
        self.activation = torch.nn.Softmax()

    def forward(self, x):
        x = torch.nn.ReLU()(self.fc1(x))
        x = torch.nn.ReLU()(self.fc2(x))
        return self.activation(self.fc3(x))
    

def check_layers_and_shapes(model):
    print(f'check layers and shapes keys')
    det = model.in_shapes.keys() - model.layer_names.keys()
    # print(f'check finish, find:')
    # for i in det:
    #     print(i)
    print(f'{len(det)} inconsistency')
