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


def write_setmethod(model, name):
    if "torch" in str(model.__class__.__bases__):
        layers = model.named_modules()
    elif "mindspore" in str(model.__class__.__bases__):
        layers = model.cells_and_names()

    flag = True
    f = open("./set_method-{}.txt".format(name), "w")
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

def write_setmethod_tf(model, name):
    if "torch" in str(model.__class__.__bases__):
        layers = model.named_modules()
    elif "mindspore" in str(model.__class__.__bases__):
        layers = model.cells_and_names()

    flag = True
    f = open("./set_method-{}.txt".format(name), "w")
    f.write("    def set_layers(self,layer_name,new_layer):\n")
    for layer_name in layers:

        layer_name = layer_name[0]
        if layer_name == "":
            continue

        elements = layer_name.split(".")
        s = "self."
        for element in elements:
            if element.isdigit():
                s = s[:-1] + ".layers[" + element + "]."
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


def write_layernames(model, name='net'):
    if "torch" in str(model.__class__.__bases__):
        layers = model.named_modules()
    elif "mindspore" in str(model.__class__.__bases__):
        layers = model.cells_and_names()

    flag = True
    f = open("./layernames_method-{}.txt".format(name), "w")
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

def write_layernames_tf(model, name='net'):
    if "torch" in str(model.__class__.__bases__):
        layers = model.named_modules()
    elif "mindspore" in str(model.__class__.__bases__):
        layers = model.cells_and_names()

    flag = True
    f = open("./layernames_method-{}_tf.txt".format(name), "w")
    f.write("self.layer_names{\n")
    for layer in layers:
        layer_name = layer[0]
        if layer_name == "":
            continue
        elements = layer_name.split(".")
        s = "self."
        for element in elements:
            if element.isdigit():
                s = s[:-1] + ".layers[" + element + "]."
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
