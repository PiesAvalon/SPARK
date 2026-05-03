import math
import torch
import torch.nn as nn
from common.mutation_torch.OP_weight_utils import _shuffle_conv2d, _shuffle_conv3d, _shuffle_dense, generate_permutation
from common.mutation_torch.Other_utils import *
from common.mutation_ms.OP_parameter_mutate_utils import get_PM_new_layer_torch


torch_dtypes = [torch.float32, torch.int32, torch.float16]
def PM_mut(model, input_size, mutate_layer_name, mutate_layer_indice, mutate_param_selname, mut_value,
           train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False
    mutate_layer = model.get_layers(mutate_layer_name)

    # select Basic OP contains mulpltiy layers
    candidate_in_mutlayers = []
    if "Sequential" in mutate_layer.__class__.__name__:
        for l in range(len(mutate_layer)):
            if "_del" in mutate_layer[l].__class__.__name__ or "empty" in mutate_layer[l].__class__.__name__ or \
                    "Replace" in l.__class__.__name__:
                continue
            if hasattr(mutate_layer[l], mutate_param_selname) or (
                    "channels" in mutate_param_selname and mutate_layer[l].__class__.__name__ == "Linear"):
                candidate_in_mutlayers.append(l)

        mutate_layer = mutate_layer[candidate_in_mutlayers[0]]

    # find the input and out shape
    mutate_layer_input_shape = model.get_inshape(mutate_layer_name)
    mutate_layer_output_shape = model.get_outshape(mutate_layer_name)

    mutate_layer_type = mutate_layer.__class__.__name__
    new_value = deepcopy(mut_value)
    mutate_replace_layer_inshape = deepcopy(mutate_layer_input_shape)

    if "sequential" in mutate_layer_type.lower():
        raise RuntimeError("try to adpot PM mutate to sequential!")

    copy_result = get_PM_new_layer_torch(mutate_layer, mutate_layer_type, mutate_param_selname, new_value, mutate_replace_layer_inshape,)
    if isinstance(copy_result, str):
        return copy_result
    else:
        mutate_replace_layer, mutate_replace_layer_inshape = copy_result[0], copy_result[1]


    tcflag = False
    test_data = np.random.randn(*tuple(mutate_replace_layer_inshape))
    for dtype in torch_dtypes:
        test_input_data = torch.tensor(test_data, dtype=dtype).to(final_device)

        try:
            new_op_outshape = mutate_replace_layer(test_input_data).shape
        except Exception:
            pass
        else:
            tcflag = True
            break

    if not tcflag:
        return "PM Create illegal layer!"

    if mutate_param_selname == "in_channels" or mutate_param_selname == "num_features":
        replace_cell1 = create_replacecell(tuple(mutate_layer_input_shape), tuple(mutate_replace_layer_inshape))
        if not tuple(new_op_outshape) == tuple(mutate_layer_output_shape):
            replace_cell2 = create_replacecell(tuple(new_op_outshape), tuple(mutate_layer_output_shape))
        else:
            replace_cell2 = EmptyCell()

        replace_layer = nn.Sequential(replace_cell1, mutate_replace_layer, replace_cell2)
        set_result = set_layer(model, replace_layer, mutate_layer_name, "PM")
        if set_result is not True:
            return set_result

        if test_flag:
            mut_result = judge_legenacy(model, input_size, train_configs)
        else:
            mut_result = True

        return mut_result

    if not tuple(new_op_outshape) == tuple(mutate_layer_output_shape):
        replace_cell = create_replacecell(tuple(new_op_outshape), tuple(mutate_layer_output_shape))
        replace_layer = nn.Sequential(mutate_replace_layer, replace_cell)
        set_result = set_layer(model, replace_layer, mutate_layer_name, "PM")
    else:
        set_result = set_layer(model, mutate_replace_layer, mutate_layer_name, "PM")

    if set_result is not True:
        return set_result

    if test_flag:
        mut_result = judge_legenacy(model, input_size, train_configs)
    else:
        mut_result = "No need to Test"

    return mut_result


def LD_mut(model, input_size, del_layer_name="", mutate_layer_indice=-1, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False

    layer_names = list(model.layer_names.keys())
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if del_layer_name in Cascade_OPs:
        del_layer_type = "Cascade_op"
    else:
        del_layer_type = "Basic_op"

    if del_layer_type == "Basic_op":
        if model.get_outshape(del_layer_name) is False:
            raise RuntimeError("No such layer!")

        topology_info = model.get_order(del_layer_name)
        last_ops, next_ops = topology_info[0], topology_info[1]
        in_shape = model.get_inshape(del_layer_name)
        out_shape = model.get_outshape(del_layer_name)
        mut_layer = model.get_layers(del_layer_name)
        if not mutate_layer_indice == -1:
            idx = mutate_layer_indice
            while not "replace" in str(mut_layer[idx].__class__.__name__).lower() and idx >= 0:
                idx -= 1

            if idx < 0:
                idx = mutate_layer_indice

            replace_cell = mut_layer[:idx]

            tcflag = True
            for dtype in torch_dtypes:
                test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
                try:
                    new_outshape = replace_cell(test_insert_layer_data).shape
                except Exception:
                    pass
                else:
                    tcflag = True
                    break
            if not tcflag:
                raise RuntimeError("can not achieve the correct dtype")



            if not tuple(new_outshape) == tuple(out_shape):
                op_replace_cell = create_replacecell(new_outshape, out_shape)
                replace_cell.add_module(str(len(replace_cell)), op_replace_cell)
        else:
            if in_shape == out_shape:
                replace_cell = EmptyCell()
            else:
                replace_cell = create_replacecell(in_shape, out_shape)

        set_result = set_layer(model, replace_cell, del_layer_name, "LD")
        if set_result is not True:
            return set_result

        if mutate_layer_indice == -1:
            # update model help parameters
            model.orders.pop(del_layer_name)
            if not isinstance(last_ops, list):
                if not ("INPUT" in last_ops):
                    lastop_houjiinfo = model.get_order(last_ops)[1]
                    if isinstance(lastop_houjiinfo, list):
                        del lastop_houjiinfo[lastop_houjiinfo.index(del_layer_name)]
                        if isinstance(next_ops, list):
                            lastop_houjiinfo_new = next_ops + lastop_houjiinfo
                        else:
                            lastop_houjiinfo_new = [next_ops] + lastop_houjiinfo
                        model.orders[last_ops] = [model.orders[last_ops][0], lastop_houjiinfo_new]
                    else:
                        model.orders[last_ops] = [model.orders[last_ops][0], next_ops]

            else:
                for last_op_single in last_ops:
                    if "INPUT" in last_op_single:
                        continue
                    lastop_houjiinfo = model.get_order(last_op_single)[1]
                    if isinstance(lastop_houjiinfo, list):
                        del lastop_houjiinfo[lastop_houjiinfo.index(del_layer_name)]
                        if isinstance(next_ops, list):
                            lastop_houjiinfo_new = next_ops + lastop_houjiinfo
                        else:
                            lastop_houjiinfo_new = [next_ops] + lastop_houjiinfo
                        model.orders[last_op_single] = [model.orders[last_op_single][0], lastop_houjiinfo_new]
                    else:
                        model.orders[last_op_single] = [model.orders[last_op_single][0], next_ops]

            if not isinstance(next_ops, list):
                if not ("OUTPUT" in next_ops):
                    nextop_qianquinfo = model.get_order(next_ops)[0]
                    if isinstance(nextop_qianquinfo, list):
                        del nextop_qianquinfo[nextop_qianquinfo.index(del_layer_name)]
                        if isinstance(last_ops, list):
                            nextop_qianquinfo_new = last_ops + nextop_qianquinfo
                        else:
                            nextop_qianquinfo_new = [last_ops] + nextop_qianquinfo
                        model.orders[next_ops] = [nextop_qianquinfo_new, model.orders[next_ops][1]]
                    else:
                        model.orders[next_ops] = [last_ops, model.orders[next_ops][1]]

            else:
                for next_op_single in next_ops:
                    if "OUTPUT" in next_op_single:
                        continue
                    nextop_qianquinfo = model.get_order(next_op_single)[0]
                    if isinstance(nextop_qianquinfo, list):
                        del nextop_qianquinfo[nextop_qianquinfo.index(del_layer_name)]
                        if isinstance(last_ops, list):
                            nextop_qianquinfo_new = last_ops + nextop_qianquinfo
                        else:
                            nextop_qianquinfo_new = [last_ops] + nextop_qianquinfo
                        model.orders[next_op_single] = [nextop_qianquinfo_new, model.orders[next_op_single][1]]
                    else:
                        model.orders[next_op_single] = [last_ops, model.orders[next_op_single][1]]

            model.out_shapes.pop(del_layer_name)
            model.in_shapes.pop(del_layer_name)
            model.layer_names.pop(del_layer_name)

            if del_layer_name in Basic_OPS:
                del Basic_OPS[Basic_OPS.index(del_layer_name)]
                model.set_Basic_OPS(Basic_OPS)
            elif del_layer_name in model.add_Cascade_OPs:
                del model.add_Cascade_OPs[model.add_Cascade_OPs.index(del_layer_name)]

    elif del_layer_type == "Cascade_op":
        yezi_ops = find_Child_leaf_OP(layer_names, del_layer_name, Basic_OPS, model.add_Cascade_OPs)
        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, del_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[del_layer_name]), list(
            model.Cascade_OPs_outshapes[del_layer_name])
        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)
        mut_layer = model.get_layers(del_layer_name)

        if len(last_ops) == 1:
            last_ops = last_ops[0]
        if len(next_ops) == 1:
            next_ops = next_ops[0]

        if not mutate_layer_indice == -1:
            idx = mutate_layer_indice
            while not "replace" in str(mut_layer[idx].__class__.__name__).lower() and idx >= 0:
                idx -= 1

            if idx < 0:
                idx = mutate_layer_indice

            replace_cell = mut_layer[:idx]

            tcflag = False
            for dtype in torch_dtypes:
                test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
                try:
                    new_outshape = replace_cell(test_insert_layer_data).shape
                except Exception:
                    pass
                else:
                    tcflag = True
                    break
            if not tcflag:
                raise RuntimeError("can not achieve the correct dtype")



            if not tuple(new_outshape) == tuple(out_shape):
                op_replace_cell = create_replacecell(new_outshape, out_shape)
                replace_cell.add_module(str(len(replace_cell)), op_replace_cell)

        else:
            if tuple(in_shape) == tuple(out_shape):
                replace_cell = EmptyCell()
            else:
                replace_cell = create_replacecell(in_shape, out_shape)

        set_result = set_layer(model, replace_cell, del_layer_name, "LD")
        if set_result is not True:
            return set_result
        if mutate_layer_indice == -1:
            LD_update_Cascade_lastandnext_info(model, last_ops, next_ops, del_layer_name)
            for child_op in yezi_ops:
                model.orders.pop(child_op)
                model.out_shapes.pop(child_op)
                model.in_shapes.pop(child_op)

            for layer_name in layer_names:
                if del_layer_name in layer_name:
                    model.layer_names.pop(layer_name)

            Cascade_OPs = del_Cascade_op_info(Cascade_OPs, del_layer_name)
            Basic_OPS = del_Cascade_op_info(Basic_OPS, del_layer_name)

            model.add_Cascade_OPs = deepcopy(del_Cascade_op_info(model.add_Cascade_OPs, del_layer_name))

            model.set_Cascade_OPS(Cascade_OPs)
            model.set_Basic_OPS(Basic_OPS)

    Cascade_OPs = model.get_Cascade_OPs()
    Cascade_OPs = remove_empty_Cascade_ops(model, Cascade_OPs, Basic_OPS)
    model.set_Cascade_OPS(Cascade_OPs)

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to Test"

    return test_result


def LA_mut(model, input_size, mut_layer_isBasic, mut_layer_name, add_layer_type, add_layer_info, activation_name,
           mutate_layer_indice=-1, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False

    layer_names = list(model.layer_names.keys())
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if len(Cascade_OPs) == 0 and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
        return "No suitable ops for LA mutation!\n"

    if mut_layer_isBasic:
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
        topology_info = model.get_order(mut_layer_name)
        last_ops, next_ops = topology_info[0], topology_info[1]
    else:
        yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)
        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), list(
            model.Cascade_OPs_outshapes[mut_layer_name])

        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)

    insert_layer_inshape = deepcopy(out_shape)
    insert_layer_outshape = deepcopy(out_shape)
    op_in_shape, op_out_shape = deepcopy(out_shape), deepcopy(out_shape)
    mut_layer = model.get_layers(mut_layer_name)

    if not mutate_layer_indice == -1:

        tcflag = False
        for dtype in torch_dtypes:
            test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
            try:
                insert_layer_inshape = mut_layer[:mutate_layer_indice + 1](test_insert_layer_data).shape
            except Exception as e:
                pass
            else:
                tcflag = True
                break
        if not tcflag:
            print("Exception: " + str(e))
            raise RuntimeError("can not achieve the correct dtype")


        insert_layer_outshape = deepcopy(out_shape)  # mut_layer[mutate_layer_indice](temp_data).shape
        op_in_shape, op_out_shape = deepcopy(insert_layer_inshape), deepcopy(insert_layer_outshape)

    alternative_insert_layers = []

    if add_layer_type == "Basic_op":
        alternative_insert_layers = get_alternative_Basicops(op_in_shape, op_out_shape, "LA")
    elif add_layer_type == "Cascade_op":
        if activation_name is None:
            activation_name = "relu"
        alternative_insert_layers = get_alternative_Cascadeops(op_in_shape, op_out_shape, "LA", activation_name)

    for alternative_insert_layer in alternative_insert_layers:
        if add_layer_info == alternative_insert_layer.__class__.__name__:
            insert_layer = alternative_insert_layer
            break

    lubrication_op = get_lubrication_op(insert_layer_inshape, insert_layer, input_size)

    if mutate_layer_indice == -1:
        if not tuple(insert_layer_outshape) == tuple(out_shape):
            replace_cell = create_replacecell(tuple(insert_layer_outshape), tuple(out_shape))
            if lubrication_op is None:
                insert_layer = nn.Sequential(mut_layer, insert_layer, replace_cell)
            else:
                insert_layer = nn.Sequential(mut_layer, lubrication_op, insert_layer, replace_cell)
        else:
            if lubrication_op is None:
                insert_layer = nn.Sequential(mut_layer, insert_layer)
            else:
                insert_layer = nn.Sequential(mut_layer, lubrication_op, insert_layer)
    else:
        if not tuple(insert_layer_outshape) == tuple(out_shape):
            replace_cell = create_replacecell(tuple(insert_layer_outshape), tuple(out_shape))
            if lubrication_op is None:
                insert_layer = nn.Sequential(mut_layer[:mutate_layer_indice + 1], insert_layer, replace_cell)
            else:
                insert_layer = nn.Sequential(mut_layer[:mutate_layer_indice + 1], lubrication_op, insert_layer,
                                             replace_cell)
        else:
            if lubrication_op is None:
                insert_layer = nn.Sequential(mut_layer[:mutate_layer_indice + 1], insert_layer)
            else:
                insert_layer = nn.Sequential(mut_layer[:mutate_layer_indice + 1], lubrication_op, insert_layer)


    tcflag = False
    test_data = np.random.randn(*tuple(in_shape))
    for dtype in torch_dtypes:
        test_insert_layer_data = torch.tensor(test_data, dtype=dtype).to(final_device)

        try:
            insert_layer(test_insert_layer_data)
        except Exception as e:
            pass
        else:
            tcflag = True
            break

    if not tcflag:
        return "LA Create illegal layer!"

    set_result = set_layer(model, insert_layer, mut_layer_name, "LA")
    if set_result is not True:
        return set_result

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to Test"

    # add
    if mutate_layer_indice == -1:
        if mut_layer_isBasic:
            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)

            if mut_layer_name in Basic_OPS:
                del Basic_OPS[Basic_OPS.index(mut_layer_name)]
            model.set_Basic_OPS(Basic_OPS)
        else:
            RA_update_Cascade_lastandnext_info(model, last_ops, next_ops, mut_layer_name)

            model.orders[mut_layer_name] = [last_ops, next_ops]
            model.in_shapes[mut_layer_name] = list(in_shape)
            model.out_shapes[mut_layer_name] = list(out_shape)

            for child_op in yezi_ops:
                model.orders.pop(child_op)
                model.in_shapes.pop(child_op)
                model.out_shapes.pop(child_op)

            for layer_name in layer_names:
                if mut_layer_name in layer_name and not mut_layer_name == layer_name:
                    model.layer_names.pop(layer_name)

            Cascade_OPs = del_Cascade_op_info(Cascade_OPs, mut_layer_name)
            Basic_OPS = del_Cascade_op_info(Basic_OPS, mut_layer_name)
            model.set_Cascade_OPS(Cascade_OPs)
            model.set_Basic_OPS(Basic_OPS)

            del_idxs = []
            for idx in range(len(model.add_Cascade_OPs)):
                op = model.add_Cascade_OPs[idx]
                if mut_layer_name in op and not mut_layer_name == op:
                    del_idxs.append(idx)
            del_flag = 0
            for idx in del_idxs:
                del model.add_Cascade_OPs[idx - del_flag]
                del_flag += 1

            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)

    Cascade_OPs = model.get_Cascade_OPs()
    Cascade_OPs = remove_empty_Cascade_ops(model, Cascade_OPs, Basic_OPS)
    model.set_Cascade_OPS(Cascade_OPs)

    return test_result


def RA_mut(model, input_size, mut_layer_isBasic, mut_layer_name, add_layer_type, add_layer_info, activation_name,
           mutate_layer_indice=-1, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False

    layer_names = list(model.layer_names.keys())
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if len(Cascade_OPs) == 0 and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
        return "No suitable ops for RA mutation!"

    if mut_layer_isBasic:
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
    else:
        yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)

        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), list(
            model.Cascade_OPs_outshapes[mut_layer_name])

        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)

        if len(last_ops) == 1:
            last_ops = last_ops[0]
        if len(next_ops) == 1:
            next_ops = next_ops[0]

    mut_layer = model.get_layers(mut_layer_name)
    insert_layer_inshape = deepcopy(in_shape)
    insert_layer_outshape = deepcopy(out_shape)
    op_in_shape, op_out_shape = deepcopy(in_shape), deepcopy(out_shape)
    if not mutate_layer_indice == -1:



        idx = mutate_layer_indice
        while not "replace" in str(mut_layer[idx].__class__.__name__).lower() and idx >= 0:
            idx -= 1

        if idx < 0:
            idx = mutate_layer_indice

        tcflag = False
        for dtype in torch_dtypes:
            test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
            try:
                temp_data = mut_layer[:idx](test_insert_layer_data)
            except Exception as e:
                pass
            else:
                tcflag = True
                break
        if not tcflag:
            raise RuntimeError("can not achieve the correct dtype")

        insert_layer_inshape = temp_data.shape
        insert_layer_outshape = deepcopy(out_shape)  # mut_layer[mutate_layer_indice](temp_data).shape
        op_in_shape, op_out_shape = deepcopy(insert_layer_inshape), deepcopy(insert_layer_outshape)

    if add_layer_type == "Basic_op":
        alternative_insert_layers = get_alternative_Basicops(op_in_shape, op_out_shape, "RA")

    elif add_layer_type == "Cascade_op":
        if activation_name is None:
            activation_name = "relu"
        alternative_insert_layers = get_alternative_Cascadeops(op_in_shape, op_out_shape, "RA", activation_name)

    for alternative_insert_layer in alternative_insert_layers:
        if add_layer_info == alternative_insert_layer.__class__.__name__:
            insert_layer = alternative_insert_layer
            break

    lubrication_op = get_lubrication_op(in_shape, insert_layer, input_size)



    if mutate_layer_indice == -1:

        if lubrication_op is None:
            test_in_shape = deepcopy(insert_layer_inshape)
        else:
            test_in_shape = deepcopy(lubrication_op.output_shape)

        for dtype in torch_dtypes:
            test_data = torch.tensor(np.random.randn(*test_in_shape), dtype=dtype).to(final_device)
            try:
                insertlayeroutshape = insert_layer(test_data).shape
            except Exception:
                pass
            else:
                break

        insert_layer_outshape = insertlayeroutshape


        if not tuple(insert_layer_outshape) == tuple(out_shape):
            replace_cell = create_replacecell(tuple(insert_layer_outshape), tuple(out_shape))

            if lubrication_op is None:
                final_insert_layer = nn.Sequential(insert_layer, replace_cell)
            else:
                final_insert_layer = nn.Sequential(lubrication_op, insert_layer, replace_cell)
        else:
            if lubrication_op is not None:
                final_insert_layer = nn.Sequential(lubrication_op, insert_layer)
            else:
                final_insert_layer = insert_layer
    else:
        mut_layer = mut_layer[:idx]
        if not tuple(insert_layer_outshape) == tuple(out_shape):
            replace_cell = create_replacecell(tuple(insert_layer_outshape), tuple(out_shape))
            if lubrication_op is None:
                mut_layer.add_module(str(len(mut_layer)), insert_layer)
                mut_layer.add_module(str(len(mut_layer)), replace_cell)
                final_insert_layer = mut_layer
            else:
                mut_layer.add_module(str(len(mut_layer)), lubrication_op)
                mut_layer.add_module(str(len(mut_layer)), insert_layer)
                mut_layer.add_module(str(len(mut_layer)), replace_cell)
                final_insert_layer = mut_layer
        else:
            if lubrication_op:
                mut_layer.add_module(str(len(mut_layer)), lubrication_op)
                mut_layer.add_module(str(len(mut_layer)), insert_layer)
                final_insert_layer = mut_layer
            else:
                mut_layer.add_module(str(len(mut_layer)), insert_layer)
                final_insert_layer = mut_layer

    tcflag = False
    test_data = np.random.randn(*tuple(insert_layer_inshape))
    for dtype in torch_dtypes:
        test_insert_layer_data = torch.tensor(test_data, dtype=dtype).to(final_device)

        try:
            final_insert_layer(test_insert_layer_data).shape
        except Exception as e:
            pass
        else:
            tcflag = True
            break

    if not tcflag:
        return "RA Create illegal layer!"


    set_result = set_layer(model, final_insert_layer, mut_layer_name, "RA")
    if set_result is not True:
        return set_result

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to Test"


    # update information
    if mutate_layer_indice == -1:
        if add_layer_type == "Cascade_op" and mut_layer_isBasic:  # Basic -> Cascade
            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)

            if mut_layer_name in Basic_OPS:
                del Basic_OPS[Basic_OPS.index(mut_layer_name)]
            model.set_Basic_OPS(Basic_OPS)

        elif add_layer_type == "Basic_op" and not mut_layer_isBasic:  # Cascade -> Basic
            RA_update_Cascade_lastandnext_info(model, last_ops, next_ops, mut_layer_name)

            model.orders[mut_layer_name] = [last_ops, next_ops]
            model.in_shapes[mut_layer_name] = list(in_shape)
            model.out_shapes[mut_layer_name] = list(out_shape)

            for child_op in yezi_ops:
                model.orders.pop(child_op)
                model.in_shapes.pop(child_op)
                model.out_shapes.pop(child_op)

            Cascade_OPs = del_Cascade_op_info(Cascade_OPs, mut_layer_name)  # delete all of its non-leaf children
            Basic_OPS = del_Cascade_op_info(Basic_OPS, mut_layer_name)  # delete all of its leaf children
            Basic_OPS.append(mut_layer_name)

            model.set_Cascade_OPS(Cascade_OPs)
            model.set_Basic_OPS(Basic_OPS)

            for layer_name in layer_names:
                if mut_layer_name in layer_name and not mut_layer_name == layer_name:
                    model.layer_names.pop(layer_name)

        elif add_layer_type == "Cascade_op" and not mut_layer_isBasic:  # Cascade -> Cascade
            # update last and next op opinfo
            RA_update_Cascade_lastandnext_info(model, last_ops, next_ops, mut_layer_name)

            # update own info
            model.orders[mut_layer_name] = [last_ops, next_ops]
            model.in_shapes[mut_layer_name] = list(in_shape)
            model.out_shapes[mut_layer_name] = list(out_shape)

            for child_op in yezi_ops:
                model.orders.pop(child_op)
                model.in_shapes.pop(child_op)
                model.out_shapes.pop(child_op)

            for layer_name in layer_names:
                if mut_layer_name in layer_name and not mut_layer_name == layer_name:
                    model.layer_names.pop(layer_name)

            Cascade_OPs = del_Cascade_op_info(Cascade_OPs, mut_layer_name)
            Basic_OPS = del_Cascade_op_info(Basic_OPS, mut_layer_name)
            model.set_Basic_OPS(Basic_OPS)
            model.set_Cascade_OPS(Cascade_OPs)

            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)

        del_idxs = []
        layer_names = model.layer_names.keys()
        for idx in range(len(model.add_Cascade_OPs)):
            op = model.add_Cascade_OPs[idx]
            if op not in layer_names:
                del_idxs.append(idx)

        del_flag = 0
        for idx in del_idxs:
            del model.add_Cascade_OPs[idx - del_flag]
            del_flag += 1

    Cascade_OPs = model.get_Cascade_OPs()
    Cascade_OPs = remove_empty_Cascade_ops(model, Cascade_OPs, Basic_OPS)
    model.set_Cascade_OPS(Cascade_OPs)
    return test_result


def CM_mut(model, input_size, mut_layer_isBasic, mut_layer_name, add_layer_info, activation_name,
           mutate_layer_indice=-1, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False

    layer_names = list(model.layer_names.keys())
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if len(Cascade_OPs) == 0 and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
        return "No suitable ops for CM mutation!"

    if mut_layer_isBasic:
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
    else:
        yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)
        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), list(
            model.Cascade_OPs_outshapes[mut_layer_name])

        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)

        if len(last_ops) == 1:
            last_ops = last_ops[0]
        if len(next_ops) == 1:
            next_ops = next_ops[0]

    mut_layer = model.get_layers(mut_layer_name)
    op_in_shape, op_out_shape = deepcopy(in_shape), deepcopy(out_shape)

    if not mutate_layer_indice == -1:


        idx = mutate_layer_indice
        while not "replace" in str(mut_layer[idx].__class__.__name__).lower() and idx >= 0:
            idx -= 1

        if idx < 0:
            idx = mutate_layer_indice

        tcflag = False
        for dtype in torch_dtypes:
            test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
            try:
                temp_data = mut_layer[:idx](test_insert_layer_data)
            except Exception:
                pass
            else:
                tcflag = True
                break

        if not tcflag:
            raise RuntimeError("can not achieve the correct dtype")


        insert_layer_inshape = temp_data.shape
        insert_layer_outshape = deepcopy(out_shape)  # mut_layer[mutate_layer_indice](temp_data).shape
        op_in_shape, op_out_shape = deepcopy(insert_layer_inshape), deepcopy(insert_layer_outshape)

    alternative_insert_layers_Basicops = get_alternative_Basicops(op_in_shape, op_out_shape, "CM")
    if activation_name is None:
        activation_name = "relu"
    alternative_insert_layers_Cascadeops = get_alternative_Cascadeops(op_in_shape, op_out_shape, "CM", activation_name)
    alternative_insert_layers = alternative_insert_layers_Basicops + alternative_insert_layers_Cascadeops

    for alternative_insert_layer in alternative_insert_layers:
        if add_layer_info == alternative_insert_layer.__class__.__name__:
            insert_layer = alternative_insert_layer
            break



    lubrication_op = get_lubrication_op(in_shape, insert_layer, input_size)
    if lubrication_op is not None:
        tcflag = False
        for dtype in torch_dtypes:
            test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
            try:
                lubrication_data = lubrication_op(test_insert_layer_data)
                insert_layer(lubrication_data)
            except Exception as e:
                pass
            else:
                tcflag = True
                break

        if not tcflag:
            return "CM Create illegal layer!"

        insert_layer = nn.Sequential(lubrication_op, insert_layer)

    if mutate_layer_indice == -1:
        branch_insert_layer = CM_branchCell(mut_layer, insert_layer, in_shape, out_shape)
    else:
        branch_insert_layer = CM_branchCell(mut_layer[idx:mutate_layer_indice + 1], insert_layer, in_shape, out_shape)

    tcflag = False
    for dtype in torch_dtypes:
        test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
        try:
            branch_insert_layer(test_insert_layer_data)
        except Exception as e:
            pass
        else:
            tcflag = True
            break

    if not tcflag:
        return "CM Create illegal layer!"

    if mutate_layer_indice == -1:
        set_result = set_layer(model, branch_insert_layer, mut_layer_name, "CM")
    else:
        mut_layer[mutate_layer_indice] = branch_insert_layer
        set_result = set_layer(model, mut_layer, mut_layer_name, "CM")

    if set_result is not True:
        return set_result

    # update information
    if mutate_layer_indice == -1:
        if mut_layer_isBasic:  # Basic -> Cascade_op
            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)
            if mut_layer_name in Basic_OPS:
                del Basic_OPS[Basic_OPS.index(mut_layer_name)]
            model.set_Basic_OPS(Basic_OPS)

        else:  # Cascade_op  -> Cascade_op
            # update last and next op opinfo
            RA_update_Cascade_lastandnext_info(model, last_ops, next_ops, mut_layer_name)

            # update own info
            model.orders[mut_layer_name] = [last_ops, next_ops]
            model.in_shapes[mut_layer_name] = list(in_shape)
            model.out_shapes[mut_layer_name] = list(out_shape)

            for child_op in yezi_ops:
                model.orders.pop(child_op)
                model.in_shapes.pop(child_op)
                model.out_shapes.pop(child_op)

            for layer_name in layer_names:
                if mut_layer_name in layer_name and not mut_layer_name == layer_name:
                    model.layer_names.pop(layer_name)

            Cascade_OPs = del_Cascade_op_info(Cascade_OPs, mut_layer_name)
            Basic_OPS = del_Cascade_op_info(Basic_OPS, mut_layer_name)
            model.set_Basic_OPS(Basic_OPS)
            model.set_Cascade_OPS(Cascade_OPs)

            del_idxs = []
            for idx in range(len(model.add_Cascade_OPs)):
                op = model.add_Cascade_OPs[idx]
                if mut_layer_name in op and not mut_layer_name == op:
                    del_idxs.append(idx)
            del_flag = 0
            for idx in del_idxs:
                del model.add_Cascade_OPs[idx - del_flag]
                del_flag += 1

            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)

    Cascade_OPs = model.get_Cascade_OPs()
    Cascade_OPs = remove_empty_Cascade_ops(model, Cascade_OPs, Basic_OPS)
    model.set_Cascade_OPS(Cascade_OPs)

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to Test"

    return test_result


def WS_mut(model, input_size, mut_layer_name, mutation_ratio, mutate_layer_indice=-1, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False

    if 0 < mutation_ratio <= 1.0:
        layer = model.get_layers(mut_layer_name)
        weights = []

        if not mutate_layer_indice == -1:
            layer = layer[mutate_layer_indice]

        params_generator = layer.state_dict()
        params_dict_keys = list(params_generator.keys())
        for param_name in params_dict_keys:
            if torch.cuda.is_available():
                weights.append(params_generator[param_name].cpu().numpy())
            else:
                weights.append(params_generator[param_name].numpy())
        layer_name = type(layer).__name__

        if "conv" in layer_name.lower() and len(weights) != 0:
            if "2d" in layer_name.lower():
                new_weights = _shuffle_conv2d(weights, mutation_ratio)
            elif "3d" in layer_name.lower():
                new_weights = _shuffle_conv3d(weights, mutation_ratio)

            old_params = layer.state_dict()
            for params_dict_key in range(len(params_dict_keys)):
                old_params[params_dict_keys[params_dict_key]] = torch.tensor(new_weights[params_dict_key],
                                                                             dtype=torch.float32).to(final_device)
            layer.load_state_dict(old_params)
        elif layer_name.lower() == "linear" and len(weights) != 0:
            new_weights = _shuffle_dense(weights, mutation_ratio)
            old_params = layer.state_dict()
            for params_dict_key in range(len(params_dict_keys)):
                old_params[params_dict_keys[params_dict_key]] = torch.tensor(new_weights[params_dict_key],
                                                                             dtype=torch.float32).to(final_device)
            layer.load_state_dict(old_params)

    else:
        raise Exception("mutation_ratio or index are wrong")

    if not mutate_layer_indice == -1:
        ori_layer = model.get_layers(mut_layer_name)
        ori_layer[mutate_layer_indice] = layer
        set_result = set_layer(model, ori_layer, mut_layer_name, "WS")
    else:
        set_result = set_layer(model, layer, mut_layer_name, "WS")

    if set_result is not True:
        return set_result

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to Test"

    return test_result


def NS_mut(model, input_size, mut_layer_name, mutation_ratio, mutate_layer_indice=-1, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False
    layer = model.get_layers(mut_layer_name)

    if not mutate_layer_indice == -1:
        layer = layer[mutate_layer_indice]

    weights = []
    params_generator = layer.state_dict()
    params_dict_keys = list(params_generator.keys())
    for param_name in params_dict_keys:
        if torch.cuda.is_available():
            weights.append(params_generator[param_name].cpu().numpy())
        else:
            weights.append(params_generator[param_name].numpy())

    if len(weights) == 2:
        weights_w, weights_b = weights
        if weights_w.shape[0] >= 2:
            permutation = np.random.permutation(weights_w.shape[0])[:2]
            weights_w[permutation[0]], weights_w[permutation[1]] = weights_w[permutation[1]].copy(), weights_w[
                permutation[0]].copy()

            weights_b[permutation[0]], weights_b[permutation[1]] = weights_b[permutation[1]].copy(), weights_b[
                permutation[0]].copy()

            old_params = layer.state_dict()
            new_weights = [weights_w, weights_b]
            for params_dict_key in range(len(params_dict_keys)):
                old_params[params_dict_keys[params_dict_key]] = torch.tensor(new_weights[params_dict_key],
                                                                             dtype=torch.float32).to(final_device)
            layer.load_state_dict(old_params)
    elif len(weights) == 1:
        weights_w = weights[0]
        if weights_w.shape[0] >= 2:
            permutation = np.random.permutation(weights_w.shape[0])[:2]
            weights_w[permutation[0]], weights_w[permutation[1]] = weights_w[permutation[1]].copy(), weights_w[
                permutation[0]].copy()

            old_params = layer.state_dict()
            new_weights = [weights_w]
            for params_dict_key in range(len(params_dict_keys)):
                old_params[params_dict_keys[params_dict_key]] = torch.tensor(new_weights[params_dict_key],
                                                                             dtype=torch.float32).to(final_device)
            layer.load_state_dict(old_params)

    if not mutate_layer_indice == -1:
        ori_layer = model.get_layers(mut_layer_name)
        ori_layer[mutate_layer_indice] = layer
        set_result = set_layer(model, ori_layer, mut_layer_name, "WS")
    else:
        set_result = set_layer(model, layer, mut_layer_name, "WS")

    if set_result is not True:
        return set_result

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to Test"

    return test_result


def GF_mut(model, input_size, mut_layer_name, mutation_ratio, mutate_layer_indice=-1, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False
    layer = model.get_layers(mut_layer_name)
    if not mutate_layer_indice == -1:
        layer = layer[mutate_layer_indice]

    STD = 0.1
    weights = []
    params_generator = layer.state_dict()
    params_dict_keys = list(params_generator.keys())

    for params_dict_key in range(len(params_dict_keys)):
        param_name = params_dict_keys[params_dict_key]
        if torch.cuda.is_available():
            val = params_generator[param_name].cpu().numpy()
        else:
            val = params_generator[param_name].numpy()
        weights.append(val)

    new_weights = []
    for weight in weights:
        weight_shape = weight.shape
        weight_flat = weight.flatten()
        permu_num = math.floor(len(weight_flat) * mutation_ratio)
        permutation = np.random.permutation(len(weight_flat))[:permu_num]
        STD = math.sqrt(weight_flat.var()) * STD
        weight_flat[permutation] = weight_flat[permutation] + np.random.normal(scale=STD, size=len(permutation))
        weight = weight_flat.reshape(weight_shape)
        new_weights.append(torch.tensor(weight, dtype=torch.float32).to(final_device))

    old_params = layer.state_dict()
    for params_dict_key in range(len(params_dict_keys)):
        old_params[params_dict_keys[params_dict_key]] = torch.tensor(new_weights[params_dict_key],
                                                                     dtype=torch.float32).to(final_device)

    layer.load_state_dict(old_params)

    if not mutate_layer_indice == -1:
        ori_layer = model.get_layers(mut_layer_name)
        ori_layer[mutate_layer_indice] = layer
        set_result = set_layer(model, ori_layer, mut_layer_name, "GF")
    else:
        set_result = set_layer(model, layer, mut_layer_name, "GF")

    if set_result is not True:
        return set_result

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to Test"

    return test_result


def NEB_mut(model, input_size, mut_layer_name, mutation_ratio, mutate_layer_indice=-1, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False
    layer = model.get_layers(mut_layer_name)

    if not mutate_layer_indice == -1:
        layer = layer[mutate_layer_indice]

    weights = []
    params_generator = layer.state_dict()
    params_dict_keys = list(params_generator.keys())

    for param_name in params_dict_keys:
        if torch.cuda.is_available():
            weights.append(params_generator[param_name].cpu().numpy())
        else:
            weights.append(params_generator[param_name].numpy())

    if len(weights) == 2:
        weights_w, weights_b = weights
        permutation = generate_permutation(weights_w.shape[0], mutation_ratio)
        weights_w[permutation] = np.zeros(weights_w[0].shape)
        weights_b[permutation] = 0
        old_params = layer.state_dict()
        new_weights = [weights_w, weights_b]
        for params_dict_key in range(len(params_dict_keys)):
            old_params[params_dict_keys[params_dict_key]] = torch.tensor(new_weights[params_dict_key],
                                                                         dtype=torch.float32).to(final_device)
        layer.load_state_dict(old_params)

    else:
        weights_w = weights[0]
        permutation = generate_permutation(weights_w.shape[0], mutation_ratio)
        weights_w[permutation] = np.zeros(weights_w[0].shape)

        old_params = layer.state_dict()
        new_weights = [weights_w]
        for params_dict_key in range(len(params_dict_keys)):
            old_params[params_dict_keys[params_dict_key]] = torch.tensor(new_weights[params_dict_key],
                                                                         dtype=torch.float32).to(final_device)
        layer.load_state_dict(old_params)

    if not mutate_layer_indice == -1:
        ori_layer = model.get_layers(mut_layer_name)
        ori_layer[mutate_layer_indice] = layer
        set_result = set_layer(model, ori_layer, mut_layer_name, "NEB")
    else:
        set_result = set_layer(model, layer, mut_layer_name, "NEB")

    if set_result is not True:
        return set_result

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to Test"
    return test_result


def NAI_mut(model, input_size, mut_layer_name, mutation_ratio, mutate_layer_indice=-1, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False
    layer = model.get_layers(mut_layer_name)

    if not mutate_layer_indice == -1:
        layer = layer[mutate_layer_indice]

    weights = []
    params_generator = layer.state_dict()
    params_dict_keys = list(params_generator.keys())

    for param_name in params_dict_keys:
        if torch.cuda.is_available():
            weights.append(params_generator[param_name].cpu().numpy())
        else:
            weights.append(params_generator[param_name].numpy())

    if len(weights) == 2:
        weights_w, weights_b = weights
        permutation = generate_permutation(weights_w.shape[0], mutation_ratio)
        weights_w[permutation] *= -1
        weights_b[permutation] *= -1

        old_params = layer.state_dict()
        new_weights = [weights_w, weights_b]
        for params_dict_key in range(len(params_dict_keys)):
            old_params[params_dict_keys[params_dict_key]] = torch.tensor(new_weights[params_dict_key],
                                                                         dtype=torch.float32).to(final_device)
        layer.load_state_dict(old_params)

    else:
        weights_w = weights[0]
        permutation = generate_permutation(weights_w.shape[0], mutation_ratio)
        weights_w[permutation] *= -1

        old_params = layer.state_dict()
        new_weights = [weights_w]
        for params_dict_key in range(len(params_dict_keys)):
            old_params[params_dict_keys[params_dict_key]] = torch.tensor(new_weights[params_dict_key],
                                                                         dtype=torch.float32).to(final_device)
        layer.load_state_dict(old_params)

    if not mutate_layer_indice == -1:
        ori_layer = model.get_layers(mut_layer_name)
        ori_layer[mutate_layer_indice] = layer
        set_result = set_layer(model, ori_layer, mut_layer_name, "NAI")
    else:
        set_result = set_layer(model, layer, mut_layer_name, "NAI")

    if set_result is not True:
        return set_result

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to Test"

    return test_result


def LS_mut(model, input_size, mut_layer_name1, mut_layer_name2, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False
    switch_layer1, switch_layer2 = model.get_layers(mut_layer_name1), model.get_layers(mut_layer_name2)
    try:
        temp1, temp2 = deepcopy(switch_layer1), deepcopy(switch_layer2)
    except Exception as e:
        temp1, temp2 = switch_layer1, switch_layer2

    set_layer(model, temp2, mut_layer_name1, "LS")
    set_layer(model, temp1, mut_layer_name2, "LS")

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to test"
    return test_result


def LC_mut(model, input_size, mut_layer_isBasic, mut_layer_name, add_layer_type, add_layer_info, activation_name,
           train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False

    layer_names = list(model.layer_names.keys())
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())
    if len(Cascade_OPs) == 0 and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
        return "No suitable ops for LC mutation!\n"

    if mut_layer_isBasic:
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
        topology_info = model.get_order(mut_layer_name)
        last_ops, next_ops = topology_info[0], topology_info[1]
    else:
        yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)
        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), \
            list(model.Cascade_OPs_outshapes[mut_layer_name])
        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)

    mut_layer = model.get_layers(mut_layer_name)

    if add_layer_type == "Basic_op":
        insert_layer = BasicOPUtils.copy_basicop(mut_layer)
    elif add_layer_type == "Cascade_op":
        insert_layer = CascadeOPUtils.copy_cascadeop(mut_layer)


    tcflag = False
    for dtype in torch_dtypes:
        test_insert_layer_data = torch.tensor(np.random.randn(*tuple(out_shape)), dtype=dtype).to(final_device)

        try:
            insert_layer_outshape = insert_layer(test_insert_layer_data).shape
        except Exception as e:
            print(insert_layer)
            print("Exception: ", e)
        else:
            tcflag = True
            break

    if not tcflag:
        return "LC Create illegal layer!"

    if not tuple(insert_layer_outshape) == tuple(out_shape):
        replace_cell = create_replacecell(tuple(insert_layer_outshape), tuple(out_shape))
        insert_layer = nn.Sequential(mut_layer, insert_layer, replace_cell)
    else:
        insert_layer = nn.Sequential(mut_layer, insert_layer)

    set_result = set_layer(model, insert_layer, mut_layer_name, "LC")
    if set_result is not True:
        return set_result

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to test"

    # add
    if mut_layer_isBasic:
        if mut_layer_name not in model.add_Cascade_OPs:
            model.add_Cascade_OPs.append(mut_layer_name)
        if mut_layer_name in Basic_OPS:
            del Basic_OPS[Basic_OPS.index(mut_layer_name)]
        model.set_Basic_OPS(Basic_OPS)

    else:
        RA_update_Cascade_lastandnext_info(model, last_ops, next_ops, mut_layer_name)
        model.orders[mut_layer_name] = [last_ops, next_ops]
        model.in_shapes[mut_layer_name] = list(in_shape)
        model.out_shapes[mut_layer_name] = list(out_shape)
        for child_op in yezi_ops:
            model.orders.pop(child_op)
            model.in_shapes.pop(child_op)
            model.out_shapes.pop(child_op)

        for layer_name in layer_names:
            if mut_layer_name in layer_name and not mut_layer_name == layer_name:
                model.layer_names.pop(layer_name)

        Cascade_OPs = del_Cascade_op_info(Cascade_OPs, mut_layer_name)
        Basic_OPS = del_Cascade_op_info(Basic_OPS, mut_layer_name)
        model.set_Cascade_OPS(Cascade_OPs)
        model.set_Basic_OPS(Basic_OPS)

        del_idxs = []
        for idx in range(len(model.add_Cascade_OPs)):
            op = model.add_Cascade_OPs[idx]
            if mut_layer_name in op and not mut_layer_name == op:
                del_idxs.append(idx)
        del_flag = 0
        for idx in del_idxs:
            del model.add_Cascade_OPs[idx - del_flag]
            del_flag += 1

        if mut_layer_name not in model.add_Cascade_OPs:
            model.add_Cascade_OPs.append(mut_layer_name)

    Cascade_OPs = model.get_Cascade_OPs()
    Cascade_OPs = remove_empty_Cascade_ops(model, Cascade_OPs, Basic_OPS)
    model.set_Cascade_OPS(Cascade_OPs)
    return test_result


def SM_mut(model, input_size, mut_layer_isBasic, mut_layer_name, mut_state, input_shape_mut, output_shape_mut,
           mutate_layer_indice=-1, train_configs=""):
    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False

    layer_names = list(model.layer_names.keys())
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if mut_layer_isBasic:
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
        topology_info = model.get_order(mut_layer_name)
        last_ops, next_ops = topology_info[0], topology_info[1]
    else:
        yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)
        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), list(
            model.Cascade_OPs_outshapes[mut_layer_name])
        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)

    mut_layer = model.get_layers(mut_layer_name)

    if not mutate_layer_indice == -1:



        idx = mutate_layer_indice

        while not "replace" in str(mut_layer[idx].__class__.__name__).lower() and idx >= 0:
            idx -= 1

        if idx < 0:
            idx = mutate_layer_indice

        tcflag = False
        for dtype in torch_dtypes:
            test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
            try:
               mut_layer[:mutate_layer_indice](test_insert_layer_data)
            except Exception:
                pass
            else:
                tcflag = True
                break
        if not tcflag:
            raise RuntimeError("can not achieve the correct dtype")




    input_replace_cell1, input_replace_cell2 = Replace_torch(in_shape, input_shape_mut), \
        Replace_torch(input_shape_mut, in_shape)
    output_replace_cell1, output_replace_cell2 = Replace_torch(out_shape, output_shape_mut), \
        Replace_torch(output_shape_mut, out_shape)

    if not mutate_layer_indice == -1:
        if mut_state == 0:
            mut_layer_slice = mut_layer[:idx + 1]
            mut_layer_slice.add_module(str(len(mut_layer_slice)), input_replace_cell1)
            mut_layer_slice.add_module(str(len(mut_layer_slice)), input_replace_cell2)
            mut_layer_slice.add_module(str(len(mut_layer_slice)), mut_layer[mutate_layer_indice])
            insert_layer = mut_layer_slice

        elif mut_state == 1:
            mut_layer_slice = mut_layer[:mutate_layer_indice + 1]
            mut_layer_slice.add_module(str(len(mut_layer_slice)), output_replace_cell1)
            mut_layer_slice.add_module(str(len(mut_layer_slice)), output_replace_cell2)
            insert_layer = mut_layer_slice

        elif mut_state == 2:
            mut_layer_slice = mut_layer[:mutate_layer_indice + 1]
            mut_layer_slice.add_module(str(len(mut_layer_slice)), input_replace_cell1)
            mut_layer_slice.add_module(str(len(mut_layer_slice)), input_replace_cell2)
            mut_layer_slice.add_module(str(len(mut_layer_slice)), mut_layer[mutate_layer_indice])
            mut_layer_slice.add_module(str(len(mut_layer_slice)), output_replace_cell1)
            mut_layer_slice.add_module(str(len(mut_layer_slice)), output_replace_cell2)
            insert_layer = mut_layer_slice


    else:
        if mut_state == 0:
            insert_layer = nn.Sequential(input_replace_cell1, input_replace_cell2, mut_layer)

        elif mut_state == 1:
            insert_layer = nn.Sequential(mut_layer, output_replace_cell1, output_replace_cell2)

        elif mut_state == 2:
            insert_layer = nn.Sequential(input_replace_cell1, input_replace_cell2, mut_layer, output_replace_cell1,
                                         output_replace_cell2)


    tcflag = False
    for dtype in torch_dtypes:
        test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
        try:
            insert_layer(test_insert_layer_data)
        except Exception as e:
            pass
        else:
            tcflag = True
            break

    if not tcflag:
        return "SM Create illegal layer!"

    set_result = set_layer(model, insert_layer, mut_layer_name, "SM")
    if set_result is not True:
        return set_result

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs)
    else:
        test_result = "No need to test"

    # add
    if mutate_layer_indice == -1:
        if mut_layer_isBasic:
            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)
            if mut_layer_name in Basic_OPS:
                del Basic_OPS[Basic_OPS.index(mut_layer_name)]
            model.set_Basic_OPS(Basic_OPS)
        else:
            RA_update_Cascade_lastandnext_info(model, last_ops, next_ops, mut_layer_name)
            model.orders[mut_layer_name] = [last_ops, next_ops]
            model.in_shapes[mut_layer_name] = list(in_shape)
            model.out_shapes[mut_layer_name] = list(out_shape)
            for child_op in yezi_ops:
                model.orders.pop(child_op)
                model.in_shapes.pop(child_op)
                model.out_shapes.pop(child_op)

            for layer_name in layer_names:
                if mut_layer_name in layer_name and not mut_layer_name == layer_name:
                    model.layer_names.pop(layer_name)

            Cascade_OPs = del_Cascade_op_info(Cascade_OPs, mut_layer_name)
            Basic_OPS = del_Cascade_op_info(Basic_OPS, mut_layer_name)
            model.set_Cascade_OPS(Cascade_OPs)
            model.set_Basic_OPS(Basic_OPS)

            del_idxs = []
            for idx in range(len(model.add_Cascade_OPs)):
                op = model.add_Cascade_OPs[idx]
                if mut_layer_name in op and not mut_layer_name == op:
                    del_idxs.append(idx)
            del_flag = 0
            for idx in del_idxs:
                del model.add_Cascade_OPs[idx - del_flag]
                del_flag += 1

            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)

    Cascade_OPs = model.get_Cascade_OPs()
    Cascade_OPs = remove_empty_Cascade_ops(model, Cascade_OPs, Basic_OPS)
    model.set_Cascade_OPS(Cascade_OPs)

    return test_result

def DM_mut(model, input_size, mut_layer_isBasic, mut_layer_name, newdtype, mutate_layer_indice=-1, train_configs=""):

    test_size_threshold = int(train_configs['test_size'])
    test_flag = True
    if input_size[0] > test_size_threshold:
        test_flag = False

    layer_names = list(model.layer_names.keys())
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if mut_layer_isBasic:
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
        topology_info = model.get_order(mut_layer_name)
        last_ops, next_ops = topology_info[0], topology_info[1]
    else:
        yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)
        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), list(
            model.Cascade_OPs_outshapes[mut_layer_name])
        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)



    mut_layer = model.get_layers(mut_layer_name)

    if not mutate_layer_indice == -1:

        idx = mutate_layer_indice

        while not "replace" in str(mut_layer[idx].__class__.__name__).lower() and idx >= 0:
            idx -= 1

        if idx < 0:
            idx = mutate_layer_indice

        tcflag = False
        for dtype in torch_dtypes:
            test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
            try:
                temp_data = mut_layer[:idx](test_insert_layer_data)
            except Exception as e:
                pass
            else:
                tcflag = True
                break
        if not tcflag:
            raise RuntimeError("can not achieve the correct dtype")


    else:
        tcflag = False
        for dtype in torch_dtypes:
            test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
            try:
                mut_layer(test_insert_layer_data)
            except Exception as e:
                pass
            else:
                tcflag = True
                break
        if not tcflag:
            raise RuntimeError("can not achieve the correct dtype")


    in_dtype = dtypecast(newdtype)
    out_dtype = dtypecast(torch.float32)

    if not mutate_layer_indice == -1:
        mut_layer[idx].to_float(newdtype)
        new_layer = nn.Sequential(in_dtype, mut_layer, out_dtype).to(final_device)
    else:
        new_layer = nn.Sequential(in_dtype, mut_layer.to(newdtype), out_dtype).to(final_device)

    tcflag = False
    error_info = ""
    for dtype in torch_dtypes:
        test_insert_layer_data = torch.tensor(np.random.randn(*tuple(in_shape)), dtype=dtype).to(final_device)
        new_layer(test_insert_layer_data)
        try:
            new_layer(test_insert_layer_data)
        except Exception as e:
            error_info = str(e)
        else:
            tcflag = True
            break

    if not tcflag:
        return "DM Create illegal layer!"

    set_result = set_layer(model, new_layer, mut_layer_name, "DM")
    if not (set_result == True):
        return set_result

    if test_flag:
        test_result = judge_legenacy(model, input_size, train_configs=train_configs)
    else:
        test_result = "No need to test"
    # add
    if mutate_layer_indice == -1:
        if mut_layer_isBasic:
            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)
            if mut_layer_name in Basic_OPS:
                del Basic_OPS[Basic_OPS.index(mut_layer_name)]
            model.set_Basic_OPS(Basic_OPS)

        else:
            RA_update_Cascade_lastandnext_info(model, last_ops, next_ops, mut_layer_name)
            model.orders[mut_layer_name] = [last_ops, next_ops]
            model.in_shapes[mut_layer_name] = list(in_shape)
            model.out_shapes[mut_layer_name] = list(out_shape)
            for child_op in yezi_ops:
                model.orders.pop(child_op)
                model.in_shapes.pop(child_op)
                model.out_shapes.pop(child_op)

            for layer_name in layer_names:
                if mut_layer_name in layer_name and not mut_layer_name == layer_name:
                    model.layer_names.pop(layer_name)

            Cascade_OPs = del_Cascade_op_info(Cascade_OPs, mut_layer_name)
            Basic_OPS = del_Cascade_op_info(Basic_OPS, mut_layer_name)
            model.set_Cascade_OPS(Cascade_OPs)
            model.set_Basic_OPS(Basic_OPS)

            del_idxs = []
            for idx in range(len(model.add_Cascade_OPs)):
                op = model.add_Cascade_OPs[idx]
                if mut_layer_name in op and not mut_layer_name == op:
                    del_idxs.append(idx)
            del_flag = 0
            for idx in del_idxs:
                del model.add_Cascade_OPs[idx - del_flag]
                del_flag += 1

            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)

    Cascade_OPs = model.get_Cascade_OPs()
    Cascade_OPs = remove_empty_Cascade_ops(model, Cascade_OPs, Basic_OPS)
    model.set_Cascade_OPS(Cascade_OPs)

    return test_result

