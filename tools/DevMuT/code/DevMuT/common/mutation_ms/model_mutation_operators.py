import math
import mindspore
import mindspore as ms
import mindspore.nn as nn
from common.mutation_ms.Layer_utils import *
from common.mutation_ms.OP_parameter_mutate_utils import *
from common.mutation_ms.OP_weight_utils import _shuffle_conv2d, _shuffle_conv3d, weighted_layer_indices, \
    assert_indices, _shuffle_dense, generate_permutation
from common.mutation_ms.Other_utils import *
import numpy as np

basicop_copy_whitelist = ['AvgPool1d', 'MaxPool1d', 'AvgPool2d', 'MaxPool2d', 'BatchNorm1d', 'BatchNorm2d',
                          'BatchNorm3d', 'Conv2d', 'Conv2dTranspose', 'Conv1d', 'Conv1dTranspose', 'Conv3d',
                          'Conv3dTranspose',
                          "Dense", "Dropout", "relu", "relu6", "tanh", "sigmoid", "leakyrelu", "elu", "gelu", "mish",
                          "softmax"
                          ]
cascadeop_copy_whitelist = ['convbnrelu', 'downsample', 'dwpw_group', 'ResidualBlock']
ms_dtypes = [mindspore.float32, mindspore.int32, mindspore.float16]

def PM_mut(model, input_size, mut_file_path="", generations=-1, mutate_logger="", train_configs=""):
    f = open(mut_file_path, "a+")
    f.write("Adopt PM mut_strategy!\n")
    mutate_logger.info("Adopt PM mut_strategy!")

    params_ori = list(mutate_OPparam_names.keys())
    params_ran = np.random.permutation(params_ori)
    mutate_param_selname = params_ran[0]

    candidate_ops = []
    add_Cascade_OPs_indices = {}
    add_Cascade_OPs = deepcopy(model.add_Cascade_OPs)
    for op in add_Cascade_OPs:
        op_layer = model.get_layers(op)
        if str(op_layer.__class__.__name__) in basicop_copy_whitelist:
            if hasattr(op_layer, mutate_param_selname):
                candidate_ops.append(op)
        elif "sequential" in str(op_layer.__class__.__name__).lower():
            suitable_indices = []
            for i in range(len(op_layer)):
                l = op_layer[i]
                if str(l.__class__.__name__) in basicop_copy_whitelist:
                    suitable_indices.append(i)

            if len(suitable_indices) > 1 or len(suitable_indices) == 0:
                continue

            l = op_layer[suitable_indices[0]]
            if hasattr(l, mutate_param_selname):
                candidate_ops.append(op)
                add_Cascade_OPs_indices[op] = suitable_indices[0]

    Basic_OPS = deepcopy(model.Basic_OPS)
    for op in Basic_OPS:
        op_layer = model.get_layers(op)
        if hasattr(op_layer, mutate_param_selname):
            candidate_ops.append(op)

    if len(candidate_ops) == 0:
        f.write("The target model do not have the mutator operators {}!\n".format(mutate_param_selname))
        f.write("mut_result:{}\n".format("Parameter Miss"))
        f.write("{} generation!\n\n".format(generations))
        mutate_logger.info("The target model do not have the mutator operators {}!".format(mutate_param_selname))
        mutate_logger.info("mut_result:{}".format("Parameter Miss"))
        mutate_logger.info("{} generation!\n".format(generations))
        return "Parameter Miss"

    suit_for_mutate_layer_idxs = np.random.permutation(candidate_ops)
    mutate_layer_name = suit_for_mutate_layer_idxs[0]
    mutate_layer = model.get_layers(mutate_layer_name)

    # select Basic OP contains mulpltiy layers
    if "Sequential" in mutate_layer.__class__.__name__:
        mutate_layer_indice = add_Cascade_OPs_indices[mutate_layer_name]
        mutate_layer = mutate_layer[mutate_layer_indice]
        f.write("candidate_in_mutlayers_indice:" + str(mutate_layer_indice) + "\n")
        mutate_logger.info("candidate_in_mutlayers_indice:" + str(mutate_layer_indice))
    else:
        f.write("candidate_in_mutlayers_indice:-1\n")
        mutate_logger.info("candidate_in_mutlayers_indice:-1")

    # find the input and out shape
    mutate_layer_input_shape = model.get_inshape(mutate_layer_name)
    mutate_layer_output_shape = model.get_outshape(mutate_layer_name)

    mutate_layer_type = mutate_layer.__class__.__name__

    f.write("select op: " + str(mutate_layer_name) + " layer_type: " + str(
        mutate_layer_type) + " selected param:" + mutate_param_selname + " input_shape:" + str(
        mutate_layer_input_shape) + " output_shape:" + str(mutate_layer_output_shape) + "\n")
    mutate_logger.info("select op: " + str(mutate_layer_name) + " layer_type: " + str(
        mutate_layer_type) + " selected param:" + mutate_param_selname + " input_shape:" + str(
        mutate_layer_input_shape) + " output_shape:" + str(mutate_layer_output_shape))

    previous_value = getattr(mutate_layer, mutate_param_selname)

    value_range_type = mutate_OPparam_names[mutate_param_selname]
    if "tuple_int" == value_range_type:
        val_ranges = mutate_OPparam_valranges[mutate_param_selname]
        l_flag, r_flag = val_ranges[0] == "[", val_ranges[-1] == "]"
        l_range, r_range = val_ranges[1:-1].split(",")

        if l_flag and r_flag:
            l, r = 0, 0
            while l == 0 and r == 0:
                l, r = random.randint(int(l_range), int(r_range) + 1), random.randint(int(l_range), int(r_range) + 1)
        elif not l_flag and r_flag:
            l, r = 0, 0
            while l == 0 and r == 0:
                l, r = random.randint(int(l_range) + 1, int(r_range) + 1), random.randint(int(l_range) + 1,
                                                                                          int(r_range) + 1)
        elif l_flag and not r_flag:
            l, r = 0, 0
            while l == 0 and r == 0:
                l, r = random.randint(int(l_range), int(r_range)), random.randint(int(l_range), int(r_range))
        elif not l_flag and not r_flag:
            l, r = 0, 0
            while l == 0 and r == 0:
                l, r = random.randint(int(l_range) + 1, int(r_range)), random.randint(int(l_range) + 1, int(r_range))

        new_value = (l, r)
        if "3d" in mutate_layer_type.lower():
            new_value = (l, int((l + r) / 2), r)

    elif "int" == value_range_type or "float" == value_range_type:
        Magnification = int(mutate_OPparam_valranges[mutate_param_selname])
        index = [i for i in range(-1 * Magnification, Magnification + 1)]
        change_rates = [-1 / val for val in index if val < 0] + [val for val in index if val > 0]

        change_rate = 1.0
        new_value = 0
        if previous_value == 0:
            previous_value = 1e-2

        while change_rate == 1.0 or new_value == 0:  # or (mutate_param_selname=="group" and )
            change_rates = np.random.permutation(change_rates)
            change_rate = change_rates[0]

            if "float" == value_range_type:

                new_value = float(change_rate * previous_value)
                while previous_value < 1 and new_value > 1:
                    new_value = new_value * 0.75

            elif "int" == value_range_type:
                new_value = int(change_rate * previous_value)

    elif "Bool" == value_range_type:
        new_value = not getattr(mutate_layer, mutate_param_selname)

    mutate_replace_layer_inshape = deepcopy(mutate_layer_input_shape)

    copy_result = get_PM_new_layer_ms(mutate_layer, mutate_layer_type, mutate_param_selname, new_value, mutate_replace_layer_inshape, f, mutate_logger,generations)

    if isinstance(copy_result, str):
        return copy_result
    else:
        mutate_replace_layer, mutate_replace_layer_inshape, new_value = copy_result[0], copy_result[1], copy_result[2]

    mutate_logger.info("mutate_replace_layer: " + str(mutate_replace_layer))

    tc_flag = False
    error_info = ""
    for dtype in ms_dtypes:
        test_input_data = mindspore.Tensor(np.random.randn(*tuple(mutate_replace_layer_inshape)), dtype)

        try:
            new_op_outshape = mutate_replace_layer(test_input_data).shape
        except Exception as e:
            error_info = str(e)
        else:
            tc_flag = True
            break
    if not tc_flag:
        f.write("Illegal PM mutate!\n")
        f.write(error_info + "\n")
        f.write("mut_result:{}\n".format("PM Create illegal layer!"))
        f.write("{} generation!\n\n".format(generations))

        mutate_logger.info("Illegal PM mutate!")
        mutate_logger.info(error_info )
        mutate_logger.info("mut_result:{}".format("PM Create illegal layer!"))
        mutate_logger.info("Exception information: \n" + str(e))
        mutate_logger.info("{} generation!\n".format(generations))
        return "PM Create illegal layer!"

    f.write("Edit value: " + str(new_value) + " new_inshape: " + str(test_input_data.shape) + " new_outshape: " + str(
        new_op_outshape) + "\n")
    mutate_logger.info(
        "Previous value: " + str(previous_value) + " Edit value: " + str(new_value) + " new_inshape: " + str(
            test_input_data.shape) + " new_outshape: " + str(
            new_op_outshape))

    if mutate_param_selname == "in_channels" or mutate_param_selname == "num_features":
        f.write("mutate op infor:\n")
        replace_cell1 = create_replacecell(tuple(mutate_layer_input_shape), tuple(mutate_replace_layer_inshape))
        if not tuple(new_op_outshape) == tuple(mutate_layer_output_shape):
            replace_cell2 = create_replacecell(tuple(new_op_outshape), tuple(mutate_layer_output_shape))
            replace_layer = nn.SequentialCell([replace_cell1, mutate_replace_layer, replace_cell2])
        else:
            replace_layer = nn.SequentialCell([replace_cell1, mutate_replace_layer])

        set_result = set_layer(model, replace_layer, mutate_layer_name, f, "PM", generations, mutate_logger)
        if set_result is not True:
            return set_result

        mut_result = judge_legenacy(model, input_size, mutate_logger, train_configs)
        f.write(str(model.get_layers(mutate_layer_name)) + "\n")
        f.write("mut_result:{}\n".format(str(mut_result)))
        f.write("{} generation!\n\n".format(generations))
        mutate_logger.info(str(model.get_layers(mutate_layer_name)))
        mutate_logger.info("mut_result:{}".format(str(mut_result)))
        mutate_logger.info("{} generation!\n".format(generations))
        f.close()
        return mut_result

    if not tuple(new_op_outshape) == tuple(mutate_layer_output_shape):
        f.write("mutate op infor:\n")
        replace_cell = create_replacecell(tuple(new_op_outshape), tuple(mutate_layer_output_shape))
        replace_layer = nn.SequentialCell([mutate_replace_layer, replace_cell])
        set_result = set_layer(model, replace_layer, mutate_layer_name, f, "PM", generations, mutate_logger)
    else:
        set_result = set_layer(model, mutate_replace_layer, mutate_layer_name, f, "PM", generations, mutate_logger)
    if set_result is not True:
        return set_result

    f.write(str(model.get_layers(mutate_layer_name)) + "\n")
    mut_result = judge_legenacy(model, input_size, mutate_logger, train_configs)
    f.write("mut_result:{}\n".format(str(mut_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info(str(model.get_layers(mutate_layer_name)))
    mutate_logger.info("mut_result:{}".format(str(mut_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return mut_result


def LD_mut(model, layer_names, input_size, del_layer_type="", mut_file_path="", generations=-1, mutate_logger="",
           train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt LD mut_strategy!\n")
    mutate_logger.info("Adopt LD mut_strategy!")

    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())
    if len(layer_names) < 2:
        f.write("mut_result:" + "not enough layers to delete!" + "\n")
        f.write("{} generation!\n\n".format(generations))
        mutate_logger.info("mut_result:" + "not enough layers to delete!")
        mutate_logger.info("not enough layers to delete!")
        return "not enough layers to delete!"

    if del_layer_type == "Basic_op":
        yezi_ops = deepcopy(Basic_OPS + model.add_Cascade_OPs)
        del_layer_loction = random.randint(0, len(yezi_ops) - 1)
        del_layer_name = yezi_ops[del_layer_loction]
        f.write("delete layer_name:" + del_layer_name + "\n")
        mutate_logger.info("delete layer_name:" + del_layer_name)
        if model.get_outshape(del_layer_name) is False:
            raise RuntimeError("No such layer!")

        topology_info = model.get_order(del_layer_name)
        last_ops, next_ops = topology_info[0], topology_info[1]
        in_shape = model.get_inshape(del_layer_name)
        out_shape = model.get_outshape(del_layer_name)
        del_layer = model.get_layers(del_layer_name)

        mutate_layer_indice = -1

        if "Sequential" in str(del_layer.__class__.__name__):
            mutate_layer_indices = []
            for i in range(len(del_layer)):
                if not "replace" in str(del_layer[mutate_layer_indice].__class__.__name__).lower():
                    mutate_layer_indices.append(i)
            if len(mutate_layer_indices) == 0:
                f.write("mut_result:No suitable ops for LD mutation!\n")
                f.write("{} generation!\n\n".format(generations))
                mutate_logger.info("mut_result:No suitable ops for LD mutation!")
                mutate_logger.info("{} generation!".format(generations))
                return "mut_result:No suitable ops for LD mutation!"

            mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])

        if not mutate_layer_indice == -1:
            idx = mutate_layer_indice

            mutate_logger.info("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))
            f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

            while not "replace" in str(del_layer[idx].__class__.__name__).lower() and idx >= 0:
                idx -= 1

            if idx < 0:
                idx = mutate_layer_indice

            replace_cell = del_layer[:idx]

            tcflag = False
            for dtype in ms_dtypes:
                test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)
                try:
                    new_outshape = replace_cell(test_insert_layer_data).shape
                    
                except Exception as e:
                    pass
                else:
                    tcflag = True
                    break
            
            if not tcflag:
                raise RuntimeError("can not detect the correct dtype")
                    
            if not tuple(new_outshape) == tuple(out_shape):
                op_replace_cell = create_replacecell(new_outshape, out_shape)
                f.write("adopt DeleteCell " + str(op_replace_cell.__class__.__name__))
                mutate_logger.info("adopt DeleteCell " + str(op_replace_cell.__class__.__name__))
                replace_cell.append(op_replace_cell)

        else:
            mutate_logger.info("candidate_in_mutlayers_indice:-1")
            f.write("candidate_in_mutlayers_indice:-1\n")
            if in_shape == out_shape:
                replace_cell = EmptyCell()
                f.write("adopt DeleteCell " + str(replace_cell.__class__.__name__) + "\n")
                mutate_logger.info("adopt DeleteCell " + str(replace_cell.__class__.__name__))
            else:
                replace_cell = create_replacecell(in_shape, out_shape)
                f.write("adopt DeleteCell " + str(replace_cell.__class__.__name__) + "\n")
                mutate_logger.info("adopt DeleteCell " + str(replace_cell.__class__.__name__))

        set_result = set_layer(model, replace_cell, del_layer_name, f, "LD", generations, mutate_logger)
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
        del_layer_name = np.random.permutation(Cascade_OPs)[0]
        del_layer = model.get_layers(del_layer_name)
        yezi_ops = find_Child_leaf_OP(layer_names, del_layer_name, Basic_OPS, model.add_Cascade_OPs)
        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, del_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[del_layer_name]), list(
            model.Cascade_OPs_outshapes[del_layer_name])
        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)
        if len(last_ops) == 1:
            last_ops = last_ops[0]
        if len(next_ops) == 1:
            next_ops = next_ops[0]

        f.write("delete layer_name:" + del_layer_name + "\n")
        mutate_logger.info("delete layer_name:" + del_layer_name)

        mutate_layer_indice = -1
        if "Sequential" in str(del_layer.__class__.__name__):
            mutate_layer_indices = []
            for i in range(len(del_layer)):
                if not "replace" in str(del_layer[mutate_layer_indice].__class__.__name__).lower():
                    mutate_layer_indices.append(i)
            if len(mutate_layer_indices) == 0:
                f.write("mut_result:No suitable ops for LD mutation!\n")
                f.write("{} generation!\n\n".format(generations))
                mutate_logger.info("mut_result:No suitable ops for LD mutation!")
                mutate_logger.info("{} generation!".format(generations))
                return "mut_result:No suitable ops for LD mutation!"
            mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])

        if not mutate_layer_indice == -1:
            idx = mutate_layer_indice

            mutate_logger.info("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))
            f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

            while not "replace" in str(del_layer[idx].__class__.__name__).lower() and idx >= 0:
                idx -= 1

            if idx < 0:
                idx = mutate_layer_indice

            replace_cell = del_layer[:idx]

            tcflag = False
            for dtype in ms_dtypes:
                test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)
                try:
                    new_outshape = replace_cell(test_insert_layer_data).shape
                except Exception as e:
                    pass
                else:
                    tcflag = True
                    break
            if not tcflag:
                raise RuntimeError("can not detect the correct dtype")


            if not tuple(new_outshape) == tuple(out_shape):
                op_replace_cell = create_replacecell(new_outshape, out_shape)
                f.write("adopt DeleteCell " + str(op_replace_cell.__class__.__name__))
                mutate_logger.info("adopt DeleteCell " + str(op_replace_cell.__class__.__name__))
                replace_cell.append(op_replace_cell)

        else:
            mutate_logger.info("candidate_in_mutlayers_indice:-1")
            f.write("candidate_in_mutlayers_indice:-1\n")
            if in_shape == out_shape:
                replace_cell = EmptyCell()
                f.write("adopt DeleteCell " + str(replace_cell.__class__.__name__) + "\n")
                mutate_logger.info("adopt DeleteCell " + str(replace_cell.__class__.__name__))
            else:
                replace_cell = create_replacecell(in_shape, out_shape)
                f.write("adopt DeleteCell " + str(replace_cell.__class__.__name__) + "\n")
                mutate_logger.info("adopt DeleteCell " + str(replace_cell.__class__.__name__))

        set_result = set_layer(model, replace_cell, del_layer_name, f, "LD", generations, mutate_logger)
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

    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)
    f.write("mut_result:" + str(test_result) + "\n")
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:" + str(test_result))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def LA_mut(model, layer_names, input_size, add_layer_type, mut_file_path, generations, mut_layer_isBasic="",
           mutate_logger="", train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt LA mut_strategy!\n")
    mutate_logger.info("Adopt LA mut_strategy!")
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if add_layer_type == "":
        add_layer_type = random.choice(["Basic_op", "Cascade_op"])
    if mut_layer_isBasic == "":
        mut_layer_isBasic = np.random.permutation([True, False])[0]
        if len(Cascade_OPs) == 0 and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            f.write("mut_result:No suitable ops for LA mutation!\n")
            mutate_logger.info("mut_result:No suitable ops for LA mutation!")
            return "mut_result:No suitable ops for LA mutation!\n"
        elif not mut_layer_isBasic and len(Cascade_OPs) == 0:
            mut_layer_isBasic = True
        elif mut_layer_isBasic and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            mut_layer_isBasic = False

    if mut_layer_isBasic:
        mut_layer_name = np.random.permutation(Basic_OPS + model.add_Cascade_OPs)[0]
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
        topology_info = model.get_order(mut_layer_name)
        last_ops, next_ops = topology_info[0], topology_info[1]
    else:
        mut_layer_name = np.random.permutation(Cascade_OPs)[0]
        yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)
        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), list(
            model.Cascade_OPs_outshapes[mut_layer_name])
        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)

    insert_layer_inshape = deepcopy(out_shape)
    insert_layer_outshape = deepcopy(out_shape)
    op_in_shape, op_out_shape = deepcopy(out_shape), deepcopy(out_shape)
    mut_layer = model.get_layers(mut_layer_name)

    mutate_layer_indice = -1
    if "Sequential" in str(mut_layer.__class__.__name__):
        mutate_layer_indices = []
        for i in range(len(mut_layer)):
            if not "replace" in str(mut_layer[mutate_layer_indice].__class__.__name__).lower():
                mutate_layer_indices.append(i)
        if len(mutate_layer_indices) == 0:
            f.write("mut_result:No suitable ops for LA mutation!\n")
            f.write("{} generation!\n\n".format(generations))
            mutate_logger.info("mut_result:No suitable ops for LA mutation!")
            mutate_logger.info("{} generation!".format(generations))
            return "mut_result:No suitable ops for LA mutation!"

        mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])

    if not mutate_layer_indice == -1:

        tcflag = False
        for dtype in ms_dtypes:

            test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)
            try:
                insert_layer_inshape = mut_layer[:mutate_layer_indice + 1](test_insert_layer_data).shape
            except Exception as e:
                pass
            else:
                tcflag = True
                break
        if not tcflag:
            raise RuntimeError("mutation_ratio or index are wrong")

        mutate_logger.info("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))
        f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

        insert_layer_outshape = deepcopy(out_shape)  # mut_layer[mutate_layer_indice](temp_data).shape
        op_in_shape, op_out_shape = deepcopy(insert_layer_inshape), deepcopy(insert_layer_outshape)


    else:
        mutate_logger.info("candidate_in_mutlayers_indice:-1")
        f.write("candidate_in_mutlayers_indice:-1\n")

    f.write("select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
        in_shape) + " out_shape: " + str(out_shape) + "\n")
    f.write("mut Basic type: " + str(mut_layer_isBasic) + "\n")
    f.write("add Basic layer : " + str(add_layer_type) + "\n")
    mutate_logger.info(
        "select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
            in_shape) + " out_shape: " + str(out_shape))
    mutate_logger.info("mut Basic type: " + str(mut_layer_isBasic))
    mutate_logger.info("add Basic layer : " + str(add_layer_type))

    alternative_insert_layers = []
    if add_layer_type == "Basic_op":
        alternative_insert_layers = get_alternative_Basicops(op_in_shape, op_out_shape, "LA")
    elif add_layer_type == "Cascade_op":
        alternative_insert_layers = get_alternative_Cascadeops(op_in_shape, op_out_shape, "LA")

    alternative_insert_layers = np.random.permutation(alternative_insert_layers)
    insert_layer = alternative_insert_layers[0]

    lubrication_op = get_lubrication_op(insert_layer_inshape, insert_layer, input_size)

    f.write("select insert layer: " + str(insert_layer) + "\n")
    mutate_logger.info("select insert layer: " + str(insert_layer))

    if mutate_layer_indice == -1:
        if not tuple(insert_layer_outshape) == tuple(out_shape):
            f.write("insert_layer_outshape not equal!: " + str(out_shape) + "\n")
            mutate_logger.info("insert_layer_outshape not equal!: " + str(out_shape))
            replace_cell = create_replacecell(tuple(insert_layer_outshape), tuple(out_shape))
            if lubrication_op is None:
                insert_layer = nn.SequentialCell([mut_layer, insert_layer, replace_cell])
            else:
                insert_layer = nn.SequentialCell([mut_layer, lubrication_op, insert_layer, replace_cell])
        else:
            f.write("insert_layer_outshape equal!" + "\n")
            mutate_logger.info("insert_layer_outshape equal!")

            if lubrication_op is None:
                insert_layer = nn.SequentialCell([mut_layer, insert_layer])
            else:
                insert_layer = nn.SequentialCell([mut_layer, lubrication_op, insert_layer])
    else:
        if not tuple(insert_layer_outshape) == tuple(out_shape):
            f.write("insert_layer_outshape not equal!: " + str(out_shape) + "\n")
            mutate_logger.info("insert_layer_outshape not equal!: " + str(out_shape))
            replace_cell = create_replacecell(tuple(insert_layer_outshape), tuple(out_shape))
            if lubrication_op is None:
                insert_layer = nn.SequentialCell([mut_layer[:mutate_layer_indice + 1], insert_layer, replace_cell])
            else:
                insert_layer = nn.SequentialCell(
                    [mut_layer[:mutate_layer_indice + 1], lubrication_op, insert_layer, replace_cell])
        else:
            f.write("insert_layer_outshape equal!" + "\n")
            mutate_logger.info("insert_layer_outshape equal!")

            if lubrication_op is None:
                insert_layer = nn.SequentialCell([mut_layer[:mutate_layer_indice + 1], insert_layer])
            else:
                insert_layer = nn.SequentialCell([mut_layer[:mutate_layer_indice + 1], lubrication_op, insert_layer])

    tcflag = False
    error_info = ""
    for dtype in ms_dtypes:
        test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)

        try:
            insert_layer(test_insert_layer_data)
        except Exception as e:
            error_info = str(e)
        else:
            tcflag = True
            break

    if not tcflag:
        f.write("Illegal LA mutate!\n")
        f.write(error_info + "\n")
        f.write("mut_result:{}\n".format("LA Create illegal layer!"))
        f.write("{} generation!\n\n".format(generations))

        mutate_logger.info("Illegal LA mutate!")
        mutate_logger.info("mut_result:{}".format("LA Create illegal layer!"))
        mutate_logger.info("Exception information: \n" + error_info)
        mutate_logger.info("{} generation!\n".format(generations))
        return "LA Create illegal layer!"


    set_result = set_layer(model, insert_layer, mut_layer_name, f, "LA", generations, mutate_logger)
    if set_result is not True:
        return set_result

    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)

    # update information
    if mutate_layer_indice == -1:
        if mut_layer_isBasic:
            if add_layer_type == "Baisc_op":  # Basic -> Cascade[Basic,Basic]
                f.write("add basicop after basicop!\n")
                mutate_logger.info("add basicop after basicop!")
            elif add_layer_type == "Cascade_op":  # Basic -> Cascade[Basic,Cascade]
                f.write("add Cascade_op after basicop!\n")
                mutate_logger.info("add Cascade_op after basicop!")
            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)
            if mut_layer_name in Basic_OPS:
                del Basic_OPS[Basic_OPS.index(mut_layer_name)]
            model.set_Basic_OPS(Basic_OPS)

        else:
            if add_layer_type == "Basic_op":  # Cascade -> Cascade[Cascade,Basic]
                f.write("add Basic_op after Cascade_op!\n")
                mutate_logger.info("add Basic_op after Cascade_op!")
            elif add_layer_type == "Cascade_op":  # Cascade -> Cascade[Cascade,Cascade]
                f.write("add Cascade_op after basicop!\n")
                mutate_logger.info("add Cascade_op after basicop!")

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

    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def RA_mut(model, layer_names, input_size, add_layer_type, mut_file_path, generations, mut_layer_isBasic="",
           mutate_logger="", train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt RA mut_strategy!\n")
    mutate_logger.info("Adopt RA mut_strategy!")

    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())
    if add_layer_type == "":
        add_layer_type = random.choice(["Basic_op", "Cascade_op"])

    if mut_layer_isBasic == "":
        mut_layer_isBasic = np.random.permutation([True, False])[0]
        if len(Cascade_OPs) == 0 and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            f.write("mut_result:No suitable ops for RA mutation!\n")
            mutate_logger.info("mut_result:No suitable ops for RA mutation!")
            return "mut_result:No suitable ops for RA mutation!"
        elif not mut_layer_isBasic and len(Cascade_OPs) == 0:
            mut_layer_isBasic = True
        elif mut_layer_isBasic and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            mut_layer_isBasic = False

    if mut_layer_isBasic:
        canditidate_select_ops = Basic_OPS + deepcopy(model.add_Cascade_OPs)
        mut_layer_name = np.random.permutation(canditidate_select_ops)[0]
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
    else:
        mut_layer_name = np.random.permutation(Cascade_OPs)[0]
        yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)

        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), list(
            model.Cascade_OPs_outshapes[mut_layer_name])

        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)
        if len(last_ops) == 1:
            last_ops = last_ops[0]
        if len(next_ops) == 1:
            next_ops = next_ops[0]

    insert_layer_inshape = deepcopy(in_shape)
    insert_layer_outshape = deepcopy(out_shape)
    op_in_shape, op_out_shape = deepcopy(in_shape), deepcopy(out_shape)
    mut_layer = model.get_layers(mut_layer_name)

    mutate_layer_indice = -1
    if "Sequential" in str(mut_layer.__class__.__name__):
        mutate_layer_indices = []
        for i in range(len(mut_layer)):
            if not "replace" in str(mut_layer[mutate_layer_indice].__class__.__name__).lower():
                mutate_layer_indices.append(i)
        if len(mutate_layer_indices) == 0:
            f.write("mut_result:No suitable ops for RA mutation!\n")
            f.write("{} generation!\n\n".format(generations))
            mutate_logger.info("mut_result:No suitable ops for RA mutation!")
            mutate_logger.info("{} generation!".format(generations))
            return "mut_result:No suitable ops for RA mutation!"

        mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])

    if not mutate_layer_indice == -1:



        idx = mutate_layer_indice
        mutate_logger.info("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))
        f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

        while not "replace" in str(mut_layer[idx].__class__.__name__).lower() and idx >= 0:
            idx -= 1

        if idx < 0:
            idx = mutate_layer_indice

        mut_layer_slice = mut_layer[:idx]

        tcflag = False
        for dtype in ms_dtypes:
            test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)

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
        mut_layer = mut_layer_slice


    else:
        mutate_logger.info("candidate_in_mutlayers_indice:-1")
        f.write("candidate_in_mutlayers_indice:-1\n")

    f.write("select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
        in_shape) + " out_shape: " + str(out_shape) + "\n")
    f.write("mut Basic type: " + str(mut_layer_isBasic) + "\n")
    f.write("add Basic layer : " + str(add_layer_type) + "\n")
    mutate_logger.info(
        "select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
            in_shape) + " out_shape: " + str(out_shape))
    mutate_logger.info("mut Basic type: " + str(mut_layer_isBasic))
    mutate_logger.info("add Basic layer : " + str(add_layer_type))

    if add_layer_type == "Basic_op":
        alternative_insert_layers = get_alternative_Basicops(op_in_shape, op_out_shape, "RA")
    elif add_layer_type == "Cascade_op":
        alternative_insert_layers = get_alternative_Cascadeops(op_in_shape, op_out_shape, "RA")

    alternative_insert_layers = np.random.permutation(alternative_insert_layers)
    insert_layer = alternative_insert_layers[0]

    lubrication_op = get_lubrication_op(insert_layer_inshape, insert_layer, input_size)

    f.write("select insert layer: " + str(insert_layer) + "\n")
    mutate_logger.info("select insert layer: " + str(insert_layer))




    if mutate_layer_indice == -1:

        if lubrication_op is None:
            test_in_shape = deepcopy(insert_layer_inshape)
        else:
            test_in_shape = deepcopy(lubrication_op.output_shape)

        tcflag = False
        error_info = ""
        for dtype in ms_dtypes:
            test_data = mindspore.Tensor(np.random.randn(*test_in_shape), dtype)

            try:
                insertlayeroutshape = insert_layer(test_data).shape
            except Exception as e:
                error_info = str(e)
            else:
                tcflag = True
                break

        if not tcflag:
            f.write("Illegal RA mutate!\n")
            f.write(error_info + "\n")
            f.write("mut_result:{}\n".format("RA Create illegal layer!"))
            f.write("{} generation!\n\n".format(generations))

            mutate_logger.info("Illegal RA mutate!")
            mutate_logger.info("mut_result:{}".format("RA Create illegal layer!"))
            mutate_logger.info("Exception information: \n" + error_info)
            mutate_logger.info("{} generation!\n".format(generations))
            return "RA Create illegal layer!"

        insert_layer_outshape = insertlayeroutshape
        if not tuple(insert_layer_outshape) == tuple(out_shape):
            f.write("insert_layer_outshape not equal!: " + str(out_shape) + "\n")
            replace_cell = create_replacecell(tuple(insert_layer_outshape), tuple(out_shape))
            if lubrication_op is None:
                final_insert_layer = nn.SequentialCell([insert_layer, replace_cell])
            else:
                final_insert_layer = nn.SequentialCell([lubrication_op, insert_layer, replace_cell])
            mutate_logger.info("insert_layer_outshape not equal!: " + str(out_shape))
        else:
            f.write("insert_layer_outshape equal!" + "\n")
            mutate_logger.info("insert_layer_outshape equal!")
            if lubrication_op is not None:
                final_insert_layer = nn.SequentialCell([lubrication_op, insert_layer])
            else:
                final_insert_layer = insert_layer
    else:
        mut_layer = mut_layer[:idx]
        if not tuple(insert_layer_outshape) == tuple(out_shape):
            mutate_logger.info("insert_layer_outshape not equal!: " + str(out_shape))
            replace_cell = create_replacecell(tuple(insert_layer_outshape), tuple(out_shape))
            if lubrication_op == None:
                mut_layer.append(insert_layer)
                mut_layer.append(replace_cell)
                final_insert_layer = mut_layer
            else:
                mut_layer.append(lubrication_op)
                mut_layer.append(insert_layer)
                mut_layer.append(replace_cell)
                final_insert_layer = mut_layer
        else:
            mutate_logger.info("insert_layer_outshape equal!")
            if not lubrication_op == None:
                mut_layer.append(lubrication_op)
                mut_layer.append(insert_layer)
                final_insert_layer = mut_layer
            else:
                mut_layer.append(insert_layer)
                final_insert_layer = mut_layer

    tcflag = False
    error_info = ""
    for dtype in ms_dtypes:
        test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(insert_layer_inshape)), dtype)

        try:
            final_insert_layer(test_insert_layer_data)

        except Exception as e:
            error_info = str(e)
        else:
            tcflag = True
            break

    if not tcflag:
        f.write("Illegal RA mutate!\n")
        f.write(error_info + "\n")
        f.write("mut_result:{}\n".format("RA Create illegal layer!"))
        f.write("{} generation!\n\n".format(generations))

        mutate_logger.info("Illegal RA mutate!")
        mutate_logger.info("mut_result:{}".format("RA Create illegal layer!"))
        mutate_logger.info("Exception information: \n" + error_info)
        mutate_logger.info("{} generation!\n".format(generations))
        return "RA Create illegal layer!"

    set_result = set_layer(model, final_insert_layer, mut_layer_name, f, "RA", generations, mutate_logger)
    if set_result is not True:
        return set_result

    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)

    # update information
    if mutate_layer_indice == 1:
        if add_layer_type == "Cascade_op" and mut_layer_isBasic:  # Basic -> Cascade
            f.write("replace Basic with Cascade!\n")
            mutate_logger.info("replace Basic with Cascade!")

            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)
            if mut_layer_name in Basic_OPS:
                del Basic_OPS[Basic_OPS.index(mut_layer_name)]
            model.set_Basic_OPS(Basic_OPS)

        elif add_layer_type == "Basic_op" and not mut_layer_isBasic:  # Cascade -> Basic
            f.write("replace Cascade with Basic!\n")
            mutate_logger.info("replace Cascade with Basic!")

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
            f.write("replace Cascade with Cascade!\n")
            mutate_logger.info("replace Cascade with Cascade!")

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
            if not op in layer_names:
                del_idxs.append(idx)

        del_flag = 0
        for idx in del_idxs:
            del model.add_Cascade_OPs[idx - del_flag]
            del_flag += 1

    Cascade_OPs = model.get_Cascade_OPs()
    Cascade_OPs = remove_empty_Cascade_ops(model, Cascade_OPs, Basic_OPS)
    model.set_Cascade_OPS(Cascade_OPs)

    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def CM_mut(model, layer_names, input_size, mut_file_path, generations, mut_layer_isBasic="", mutate_logger="",
           train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt CM mut_strategy!\n")
    mutate_logger.info("Adopt CM mut_strategy!")

    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if mut_layer_isBasic == "":
        mut_layer_isBasic = np.random.permutation([True, False])[0]
        if len(Cascade_OPs) == 0 and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            f.write("mut_result:No suitable ops for CM mutation!\n")
            mutate_logger.info("mut_result:No suitable ops for CM mutation!")
            return "mut_result:No suitable ops for CM mutation!"
        elif not mut_layer_isBasic and len(Cascade_OPs) == 0:
            mut_layer_isBasic = True
        elif mut_layer_isBasic and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            mut_layer_isBasic = False

    if mut_layer_isBasic:
        mut_layer_name = np.random.permutation(Basic_OPS + model.add_Cascade_OPs)[0]
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
    else:
        mut_layer_name = np.random.permutation(Cascade_OPs)[0]
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

    mutate_layer_indice = -1
    if "Sequential" in str(mut_layer.__class__.__name__):
        mutate_layer_indices = []
        for i in range(len(mut_layer)):
            if not "replace" in str(mut_layer[mutate_layer_indice].__class__.__name__).lower():
                mutate_layer_indices.append(i)

        if len(mutate_layer_indices) == 0:
            f.write("mut_result:No suitable ops for CM mutation!\n")
            f.write("{} generation!\n\n".format(generations))
            mutate_logger.info("mut_result:No suitable ops for CM mutation!")
            mutate_logger.info("{} generation!".format(generations))
            return "mut_result:No suitable ops for CM mutation!"

        mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])

    if not mutate_layer_indice == -1:


        idx = mutate_layer_indice
        while not "replace" in str(mut_layer[idx].__class__.__name__).lower() and idx >= 0:
            idx -= 1

        if idx < 0:
            idx = mutate_layer_indice

        tcflag = False
        for dtype in ms_dtypes:
            test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)
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
        mutate_logger.info("candidate_in_mutlayers_indice:{}".format(mutate_layer_indice))
        f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

    else:
        mutate_logger.info("candidate_in_mutlayers_indice:-1")
        f.write("candidate_in_mutlayers_indice:-1\n")

    f.write("select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
        in_shape) + " out_shape: " + str(out_shape) + "\n")
    f.write("mut Basic type: " + str(mut_layer_isBasic) + "\n")

    mutate_logger.info(
        "select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
            in_shape) + " out_shape: " + str(out_shape))
    mutate_logger.info("mut Basic type: " + str(mut_layer_isBasic))

    alternative_insert_layers = get_alternative_Cascadeops(op_in_shape, op_out_shape, "CM") + get_alternative_Basicops(
        op_in_shape, op_out_shape, "CM")

    idxs = np.random.permutation([i for i in range(len(alternative_insert_layers))])
    insert_layer_candidate = alternative_insert_layers[idxs[0]]

    lubrication_op = get_lubrication_op(in_shape, insert_layer_candidate, input_size)



    if lubrication_op is not None:
        tcflag = False
        error_info = ""
        for dtype in ms_dtypes:
            test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)
            try:
                lubrication_data = lubrication_op(test_insert_layer_data)
                insert_layer_candidate(lubrication_data)
            except Exception as e:
                error_info = str(e)
            else:
                tcflag = True
                break

        if not tcflag:
            f.write("Illegal CM mutate!\n")
            f.write(error_info + "\n")
            f.write("mut_result:{}\n".format("CM Create illegal layer!"))
            f.write("{} generation!\n\n".format(generations))

            mutate_logger.info("Illegal CM mutate!")
            mutate_logger.info("mut_result:{}".format("CM Create illegal layer!"))
            mutate_logger.info("Exception information: \n" + error_info)
            mutate_logger.info("{} generation!\n".format(generations))
            return "CM Create illegal layer!"

        insert_layer = nn.SequentialCell([lubrication_op, insert_layer_candidate])
    else:
        insert_layer = insert_layer_candidate

    f.write("select insert layer: " + str(insert_layer_candidate) + "\n")
    mutate_logger.info("select insert layer: " + str(insert_layer_candidate))

    if mutate_layer_indice == -1:
        branch_insert_layer = CM_branchCell(mut_layer, insert_layer, in_shape, out_shape)
    else:
        branch_insert_layer = CM_branchCell(mut_layer[idx:mutate_layer_indice + 1], insert_layer, in_shape, out_shape)

    tcflag = False
    error_info = ""
    for dtype in ms_dtypes:
        test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)

        try:
            branch_insert_layer(test_insert_layer_data)
        except Exception as e:
            error_info = str(e)
        else:
            tcflag = True
            break

    if not tcflag:
        f.write("Illegal CM mutate!\n")
        f.write(error_info + "\n")
        f.write("mut_result:{}\n".format("CM Create illegal layer!"))
        f.write("{} generation!\n\n".format(generations))

        mutate_logger.info("Illegal CM mutate!")
        mutate_logger.info("mut_result:{}".format("CM Create illegal layer!"))
        mutate_logger.info("Exception information: \n" + error_info)
        mutate_logger.info("{} generation!\n".format(generations))
        return "CM Create illegal layer!"

    if mutate_layer_indice == -1:
        set_result = set_layer(model, branch_insert_layer, mut_layer_name, f, "CM", generations, mutate_logger)
    else:
        mut_layer[mutate_layer_indice] = branch_insert_layer
        set_result = set_layer(model, mut_layer, mut_layer_name, f, "CM", generations, mutate_logger)

    if set_result is not True:
        return set_result

    # update information
    if mutate_layer_indice == -1:
        if mut_layer_isBasic:  # Basic -> Cascade_op
            f.write("replace Basic with CM op!\n")
            mutate_logger.info("replace Basic with CM op!")
            if mut_layer_name not in model.add_Cascade_OPs:
                model.add_Cascade_OPs.append(mut_layer_name)

            if mut_layer_name in Basic_OPS:
                del Basic_OPS[Basic_OPS.index(mut_layer_name)]
            model.set_Basic_OPS(Basic_OPS)
        else:  # Cascade_op  -> Cascade_op
            f.write("replace Cascade with CM op!\n")
            mutate_logger.info("replace Cascade with CM op!")
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

    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)

    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def WS_mut(model, layer_names, input_size, mut_file_path, generations, mutate_logger="", mutation_ratio=0.4,
           train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt WS mut_strategy!\n")
    mutate_logger.info("Adopt WS mut_strategy!")

    candidate_layer_names = model.add_Cascade_OPs + model.Basic_OPS
    mutated_layer_indices = weighted_layer_indices(model, candidate_layer_names)
    depth_layer = len(candidate_layer_names)
    mutated_layer_indices = np.arange(depth_layer) if mutated_layer_indices is None else mutated_layer_indices

    mutate_layer_indice = -1
    execution_flag = False
    if 0 < mutation_ratio <= 1.0:
        assert_indices(mutated_layer_indices, depth_layer)
        layer_name = ""
        search_times = 0
        while search_times <= depth_layer:
            mutated_layer_indices = np.random.permutation(mutated_layer_indices)
            sel_mut_layer_name_idx = mutated_layer_indices[0]

            mut_layer_name = candidate_layer_names[sel_mut_layer_name_idx]
            layer = model.get_layers(mut_layer_name)

            layer_name = type(layer).__name__
            search_times += 1
            if ("conv" in layer_name.lower() or "dense" in layer_name.lower()):
                break

            if "sequential" in str(layer.__class__.__name__):
                mutate_layer_indices = []
                for i in range(len(layer)):
                    layeri_type = str(layer[i].__class__.__name__).lower()
                    if ("conv" in layeri_type or "dense" in layeri_type):
                        mutate_layer_indices.append(i)

                if len(mutate_layer_indices) == 0:
                    f.write("mut_result:No suitable ops for WS mutation!\n")
                    f.write("{} generation!\n\n".format(generations))
                    mutate_logger.info("mut_result:No suitable ops for WS mutation!")
                    mutate_logger.info("{} generation!".format(generations))
                    return "mut_result:No suitable ops for WS mutation!"

                mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])

        if layer == None:
            f.write("mut_result:No suitable ops for WS mutation!\n")
            f.write("{} generation!\n\n".format(generations))
            mutate_logger.info("mut_result:No suitable ops for WS mutation!")
            mutate_logger.info("{} generation!".format(generations))
            return "mut_result:No suitable ops for WS mutation!"

        if not mutate_layer_indice == -1:
            layer = layer[mutate_layer_indice]
            mutate_logger.info("candidate_in_mutlayers_indice:{}".format(mutate_layer_indice))
            f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

        else:
            mutate_logger.info("candidate_in_mutlayers_indice:-1")
            f.write("candidate_in_mutlayers_indice:-1\n")

        weights = []
        params_generator = layer.get_parameters()
        params_dict_keys = []
        for param in params_generator:
            params_dict_keys.append(param.name)
            weights.append(param.init_data().asnumpy())

        if "conv" in layer_name.lower() and len(weights) != 0:
            if "2d" in layer_name.lower():
                new_weights = _shuffle_conv2d(weights, mutation_ratio)
            elif "3d" in layer_name.lower():
                new_weights = _shuffle_conv3d(weights, mutation_ratio)

            for params_dict_key in params_dict_keys:
                f.write(f"select layer:{candidate_layer_names[sel_mut_layer_name_idx]}\n")
                f.write(f"layer type:{str(type(layer).__name__)}\n")
                mutate_logger.info(
                    f"select layer:{candidate_layer_names[sel_mut_layer_name_idx]} " + " layer_type:" + str(
                        type(layer).__name__))
                execution_flag = True
                param = layer.parameters_dict()[params_dict_key]
                update = nn.ParameterUpdate(param)
                update(ms.Tensor(new_weights[params_dict_keys.index(params_dict_key)], ms.float32))

        elif layer_name.lower() == "dense" and len(weights) != 0:
            new_weights = _shuffle_dense(weights, mutation_ratio)
            for params_dict_key in params_dict_keys:
                f.write(f"select layer:{candidate_layer_names[sel_mut_layer_name_idx]}\n")
                f.write(f"layer type:{str(type(layer).__name__)}\n")
                mutate_logger.info(
                    f"select layer:{candidate_layer_names[sel_mut_layer_name_idx]} " + " layer_type:" + str(
                        type(layer).__name__))
                execution_flag = True
                param = layer.parameters_dict()[params_dict_key]
                update = nn.ParameterUpdate(param)
                update(ms.Tensor(new_weights[params_dict_keys.index(params_dict_key)], ms.float32))
                # load_param_into_net equals to pytorch load_state_to
        else:
            pass
    else:
        raise RuntimeError("mutation_ratio or index are wrong")

    if execution_flag is False:
        f.write("mut_result:No suitable layer for WS!\n")
        f.write("{} generation!\n".format(generations))
        mutate_logger.info("mut_result:No suitable layer for WS!")
        mutate_logger.info("{} generation!\n".format(generations))
        f.close()
        return "No suitable layer for WS!"

    if not mutate_layer_indice == -1:
        ori_layer = model.get_layers(mut_layer_name)
        ori_layer[mutate_layer_indice] = layer
        set_result = set_layer(model, ori_layer, mut_layer_name, f, "WS", generations, mutate_logger)
    else:
        set_result = set_layer(model, layer, mut_layer_name, f, "WS", generations, mutate_logger)

    if set_result is not True:
        return set_result

    f.write(f"mutation_ratio:{mutation_ratio}\n")
    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)
    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def NS_mut(model, layer_names, input_size, mut_file_path, generations, mutate_logger="", mutation_ratio=0.4,
           train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt NS mut_strategy!\n")
    mutate_logger.info("Adopt NS mut_strategy!")

    candidate_layer_names = model.add_Cascade_OPs + model.Basic_OPS
    mutated_layer_indices = weighted_layer_indices(model, candidate_layer_names)
    mutated_layer_indices = np.arange(
        len(candidate_layer_names)) if mutated_layer_indices is None else mutated_layer_indices
    depth_layer = len(candidate_layer_names)
    assert_indices(mutated_layer_indices, depth_layer)
    layer_utils = Layer_helpUtils()

    mutated_layer_indices = np.random.permutation(mutated_layer_indices)

    execution_flag = False

    mutate_layer_indice = -1
    for layer_name_index in mutated_layer_indices:
        layer_name = candidate_layer_names[layer_name_index]
        layer = model.get_layers(layer_name)
        mut_layer_name = deepcopy(layer_name)

        if "sequential" in str(layer.__class__.__name__):
            mutate_layer_indices = []
            for i in range(len(layer)):
                if layer_utils.is_layer_in_weight_change_white_list(layer[i]):
                    mutate_layer_indices.append(i)

            if len(mutate_layer_indices) == 0:
                continue

            mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])
            layer = layer[mutate_layer_indice]
            mutate_logger.info("candidate_in_mutlayers_indice:{}".format(mutate_layer_indice))
            f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

        else:
            if not layer_utils.is_layer_in_weight_change_white_list(layer):
                continue
            mutate_logger.info("candidate_in_mutlayers_indice:-1")
            f.write("candidate_in_mutlayers_indice:-1\n")

        weights = []
        params_generator = layer.get_parameters()
        params_dict_keys = []
        for param in params_generator:
            params_dict_keys.append(param.name)
            weights.append(param.init_data().asnumpy())

        if len(weights) > 0:
            if len(weights) == 2:

                f.write(f"select layer:{layer_name}\n")
                f.write(f"layer type:{str(type(layer).__name__)}\n")
                mutate_logger.info(f"select layer:{layer_name} " + " layer_type:" + str(type(layer).__name__))
                weights_w, weights_b = weights

                if weights_w.shape[0] >= 2:
                    permutation = np.random.permutation(weights_w.shape[0])[:2]
                    weights_w[permutation[0]], weights_w[permutation[1]] = weights_w[permutation[1]].copy(), weights_w[
                        permutation[0]].copy()
                    weights_b[permutation[0]], weights_b[permutation[1]] = weights_b[permutation[1]].copy(), weights_b[
                        permutation[0]].copy()

                    layer.weight.set_data(ms.Tensor(weights_w, ms.float32))
                    layer.bias.set_data(ms.Tensor(weights_b, ms.float32))
                    execution_flag = True
                else:
                    f.write(f"layer type:{str(type(layer).__name__)}\n")
                    mutate_logger.info(f"select layer:{layer_name} " + " layer_type:" + str(type(layer).__name__))

            elif len(weights) == 1:
                f.write(f"select layer:{layer_name}\n")
                f.write(f"layer type:{str(type(layer).__name__)}\n")
                mutate_logger.info(f"select layer:{layer_name} " + " layer_type:" + str(type(layer).__name__))
                weights_w = weights[0]

                if weights_w.shape[0] >= 2:
                    permutation = np.random.permutation(weights_w.shape[0])[:2]
                    weights_w[permutation[0]], weights_w[permutation[1]] = weights_w[permutation[1]].copy(), weights_w[
                        permutation[0]].copy()

                    layer.weight.set_data(ms.Tensor(weights_w, ms.float32))
                    execution_flag = True
                else:
                    f.write(f"layer type:{str(type(layer).__name__)}\n")
                    mutate_logger.info(f"select layer:{layer_name} " + " layer_type:" + str(type(layer).__name__))

            break

    if execution_flag is False:
        f.write("mut_result:No suitable layer for NS!\n")
        f.write("{} generation!\n\n".format(generations))
        mutate_logger.info("mut_result:No suitable layer for NS!")
        mutate_logger.info("{} generation!\n".format(generations))
        f.close()
        return "No suitable layer for NS!"
    else:

        if not mutate_layer_indice == -1:
            ori_layer = model.get_layers(mut_layer_name)
            ori_layer[mutate_layer_indice] = layer
            set_result = set_layer_nolog(model, ori_layer, mut_layer_name, "NS")
        else:
            set_result = set_layer_nolog(model, layer, mut_layer_name, "NS")

        if set_result is not True:
            return set_result

        f.write(f"mutation_ratio:{mutation_ratio}\n")
        test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)
        f.write("mut_result:{}\n".format(str(test_result)))
        f.write("{} generation!\n\n".format(generations))
        mutate_logger.info("mut_result:{}".format(str(test_result)))
        mutate_logger.info("{} generation!\n".format(generations))
        f.close()
        return test_result


def GF_mut(model, layer_names, input_size, mut_file_path, generations, mutate_logger="", mutation_ratio=0.4,
           train_configs=""):
    distribution = 'normal'
    STD = 0.1
    f = open(mut_file_path, 'a+')
    f.write("Adopt GF mut_strategy!\n")
    mutate_logger.info("Adopt GF mut_strategy!")
    valid_distributions = ['normal', 'uniform']
    assert distribution in valid_distributions, 'Distribution %s is not support.' % distribution

    candidate_layer_names = model.add_Cascade_OPs + model.Basic_OPS

    chosed_index = np.random.randint(0, len(candidate_layer_names))
    layer_name = candidate_layer_names[chosed_index]
    layer = model.get_layers(layer_name)

    mutate_layer_indice = -1
    if "sequential" in str(layer.__class__.__name__):
        mutate_layer_indices = []
        for i in range(len(layer)):
            if not "replace" in str(layer[i].__class__.__name__).lower():
                mutate_layer_indices.append(i)

        if len(mutate_layer_indices) == 0:
            f.write("mut_result:No suitable ops for GF mutation!\n")
            f.write("{} generation!\n\n".format(generations))
            mutate_logger.info("mut_result:No suitable ops for GF mutation!")
            mutate_logger.info("{} generation!".format(generations))
            return "mut_result:No suitable ops for GF mutation!"

        mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])
        layer = layer[mutate_layer_indice]
        mutate_logger.info("candidate_in_mutlayers_indice:{}".format(mutate_layer_indice))
        f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

    else:
        mutate_logger.info("candidate_in_mutlayers_indice:-1")
        f.write("candidate_in_mutlayers_indice:-1\n")

    weights = []
    params_generator = layer.get_parameters()
    params_dict_keys = []
    for param in params_generator:
        params_dict_keys.append(param.name)
        weights.append(param.init_data().asnumpy())

    f.write(f"select layer:{layer_name}\n")
    f.write(f"layer type:{str(type(layer).__name__)}\n")
    mutate_logger.info(f"select layer:{layer_name} " + " layer_type:" + str(type(layer).__name__))

    new_weights = []
    for weight in weights:
        weight_shape = weight.shape
        weight_flat = weight.flatten()
        permu_num = math.floor(len(weight_flat) * mutation_ratio)
        permutation = np.random.permutation(len(weight_flat))[:permu_num]
        STD = math.sqrt(weight_flat.var()) * STD
        weight_flat[permutation] = weight_flat[permutation] + np.random.normal(scale=STD, size=len(permutation))
        weight = weight_flat.reshape(weight_shape)
        new_weights.append(mindspore.Tensor(weight, ms.float32))

    for params_dict_key in params_dict_keys:
        param = layer.parameters_dict()[params_dict_key]
        update = nn.ParameterUpdate(param)
        update(ms.Tensor(new_weights[params_dict_keys.index(params_dict_key)], ms.float32))

    if not mutate_layer_indice == -1:
        ori_layer = model.get_layers(layer_name)
        ori_layer[mutate_layer_indice] = layer
        set_result = set_layer_nolog(model, ori_layer, layer_name, "GF")
    else:
        set_result = set_layer_nolog(model, layer, layer_name, "GF")

    if set_result is not True:
        return set_result

    f.write(f"mutation_ratio:{mutation_ratio}\n")
    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)
    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def NEB_mut(model, layer_names, input_size, mut_file_path, generations, mutate_logger="", mutation_ratio=0.4,
            train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt NEB mut_strategy!\n")
    mutate_logger.info("Adopt NEB mut_strategy!")
    candidate_layer_names = model.add_Cascade_OPs + model.Basic_OPS
    mutated_layer_indices = np.arange(len(candidate_layer_names) - 1)

    execution_flag = False
    if 0 < mutation_ratio <= 1.0:
        assert_indices(mutated_layer_indices, len(candidate_layer_names))
        layer_utils = Layer_helpUtils()

        mutate_layer_indice = -1

        for i in mutated_layer_indices:
            layer_name = candidate_layer_names[mutated_layer_indices[i]]
            layer = model.get_layers(layer_name)
            # skip if layer is not in white list

            if "sequential" in str(layer.__class__.__name__):
                mutate_layer_indices = []
                for i in range(len(layer)):
                    if layer_utils.is_layer_in_weight_change_white_list(layer[i]):
                        mutate_layer_indices.append(i)

                if len(mutate_layer_indices) == 0:
                    continue

                mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])
                layer = layer[mutate_layer_indice]
                mutate_logger.info("candidate_in_mutlayers_indice:{}".format(mutate_layer_indice))
                f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

            else:
                if not layer_utils.is_layer_in_weight_change_white_list(layer):
                    continue
                mutate_logger.info("candidate_in_mutlayers_indice:-1")
                f.write("candidate_in_mutlayers_indice:-1\n")

            weights = []
            params_generator = layer.get_parameters()
            params_dict_keys = []
            for param in params_generator:
                params_dict_keys.append(param.name)
                weights.append(param.init_data().asnumpy())

            if len(weights) > 0:
                if len(weights) == 2:
                    execution_flag = True
                    weights_w, weights_b = weights
                    permutation = generate_permutation(weights_w.shape[0], mutation_ratio)
                    weights_w[permutation] = np.zeros(weights_w[0].shape)
                    weights_b[permutation] = 0
                    new_weights = [weights_w, weights_b]

                    f.write(f"select layer:{layer_name}\n")
                    f.write(f"layer type:{str(type(layer).__name__)}\n")
                    mutate_logger.info(f"select layer:{layer_name} " + " layer_type:" + str(type(layer).__name__))

                    for params_dict_key in params_dict_keys:
                        param = layer.parameters_dict()[params_dict_key]
                        update = nn.ParameterUpdate(param)
                        update(ms.Tensor(new_weights[params_dict_keys.index(params_dict_key)], ms.float32))

                elif len(weights) == 1:
                    execution_flag = True
                    f.write(f"select layer:{layer_name}\n")
                    f.write(f"layer type:{str(type(layer).__name__)}\n")
                    mutate_logger.info(f"select layer:{layer_name} " + " layer_type:" + str(type(layer).__name__))

                    weights_w = weights[0]
                    permutation = generate_permutation(weights_w.shape[0], mutation_ratio)
                    weights_w[permutation] = np.zeros(weights_w[0].shape)
                    new_weights = [weights_w]

                    for params_dict_key in params_dict_keys:
                        param = layer.parameters_dict()[params_dict_key]
                        update = nn.ParameterUpdate(param)
                        update(ms.Tensor(new_weights[params_dict_keys.index(params_dict_key)], ms.float32))
                break
    else:
        raise Exception("mutation_ratio or index are wrong")

    if not execution_flag:
        mutate_logger.info("mut_result:Do not find suitable layer for NEB mutation!")
        f.write(f"mutation_ratio:{mutation_ratio}\n")
        f.write(f"mut_result:Do not find suitable layer for NEB mutation!\n")
        f.write("{} generation!\n\n".format(generations))
        mutate_logger.info("{} generation!\n".format(generations))

        f.close()
        return "Do not find suitable layer for NEB mutation!"
    else:
        if not mutate_layer_indice == -1:
            ori_layer = model.get_layers(layer_name)
            ori_layer[mutate_layer_indice] = layer
            set_result = set_layer_nolog(model, ori_layer, layer_name, "NEB")
        else:
            set_result = set_layer_nolog(model, layer, layer_name, "NEB")

        if set_result is not True:
            return set_result

    f.write(f"mutation_ratio:{mutation_ratio}\n")
    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)
    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def NAI_mut(model, layer_names, input_size, mut_file_path, generations, mutate_logger="", mutation_ratio=0.4,
            train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt NAI mut_strategy!\n")
    mutate_logger.info("Adopt NAI mut_strategy!")

    candidate_layer_names = model.add_Cascade_OPs + model.Basic_OPS
    mutated_layer_indices = np.arange(len(candidate_layer_names) - 1)

    execution_flag = False
    if 0 < mutation_ratio <= 1.0:
        assert_indices(mutated_layer_indices, len(candidate_layer_names))
        np.random.shuffle(mutated_layer_indices)
        layer_utils = Layer_helpUtils()

        mutate_layer_indice = -1
        for i in mutated_layer_indices:
            layer_name = candidate_layer_names[mutated_layer_indices[i]]
            layer = model.get_layers(layer_name)

            if "sequential" in str(layer.__class__.__name__):
                mutate_layer_indices = []
                for i in range(len(layer)):
                    if layer_utils.is_layer_in_weight_change_white_list(layer[i]):
                        mutate_layer_indices.append(i)

                if len(mutate_layer_indices) == 0:
                    continue

                mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])
                layer = layer[mutate_layer_indice]
                mutate_logger.info("candidate_in_mutlayers_indice:{}".format(mutate_layer_indice))
                f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

            else:
                if not layer_utils.is_layer_in_weight_change_white_list(layer):
                    continue
                mutate_logger.info("candidate_in_mutlayers_indice:-1")
                f.write("candidate_in_mutlayers_indice:-1\n")

            weights = []
            params_generator = layer.get_parameters()
            params_dict_keys = []
            for param in params_generator:
                params_dict_keys.append(param.name)
                weights.append(param.init_data().asnumpy())

            if len(weights) > 0:
                if len(weights) == 2:
                    execution_flag = True
                    weights_w, weights_b = weights
                    permutation = generate_permutation(weights_w.shape[0], mutation_ratio)
                    weights_w[permutation] *= -1
                    weights_b[permutation] *= -1
                    new_weights = [weights_w, weights_b]
                    f.write(f"select layer:{layer_name}\n")
                    f.write(f"layer type:{str(type(layer).__name__)}\n")
                    mutate_logger.info(f"select layer:{layer_name} " + " layer_type:" + str(type(layer).__name__))
                    for params_dict_key in params_dict_keys:
                        param = layer.parameters_dict()[params_dict_key]
                        update = nn.ParameterUpdate(param)
                        update(ms.Tensor(new_weights[params_dict_keys.index(params_dict_key)], ms.float32))

                elif len(weights) == 1:
                    execution_flag = True
                    f.write(f"select layer:{layer_name}\n")
                    f.write(f"layer type:{str(type(layer).__name__)}\n")
                    mutate_logger.info(f"select layer:{layer_name} " + " layer_type:" + str(type(layer).__name__))
                    weights_w = weights[0]
                    permutation = generate_permutation(weights_w.shape[0], mutation_ratio)
                    weights_w[permutation] *= -1
                    new_weights = [weights_w]

                    for params_dict_key in params_dict_keys:
                        param = layer.parameters_dict()[params_dict_key]
                        update = nn.ParameterUpdate(param)
                        update(ms.Tensor(new_weights[params_dict_keys.index(params_dict_key)], ms.float32))
                break
    else:
        raise Exception("mutation_ratio or index are wrong")

    if not execution_flag:
        mutate_logger.info("mut_result:Do not find suitable layer for NAI mutation!")
        f.write(f"mutation_ratio:{mutation_ratio}\n")
        f.write("mut_result:Do not find suitable layer for NAI mutation!\n")
        f.write("{} generation!\n\n".format(generations))
        mutate_logger.info("{} generation!\n".format(generations))

        f.close()
        return "Do not find suitable layer for NAI mutation!"

    else:
        if not mutate_layer_indice == -1:
            ori_layer = model.get_layers(layer_name)
            ori_layer[mutate_layer_indice] = layer
            set_result = set_layer_nolog(model, ori_layer, layer_name, "NAI")
        else:
            set_result = set_layer_nolog(model, layer, layer_name, "NAI")

        if set_result is not True:
            return set_result

    f.write(f"mutation_ratio:{mutation_ratio}\n")
    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)
    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def LS_mut(model, layer_names, input_size, mut_file_path, generations, mut_layer_isBasic="", mutate_logger="",
           train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt LS mut_strategy!\n")
    mutate_logger.info("Adopt LS mut_strategy!")

    candidate_layers = model.add_Cascade_OPs + model.Basic_OPS
    search_times = 0
    while True:
        candidate_layers = np.random.permutation(candidate_layers)
        switch_layer1_name = candidate_layers[0]
        switch_layer2_name = scan_same_inout(model, switch_layer1_name, candidate_layers)
        if switch_layer2_name is False:
            search_times += 1
            if search_times == 100:
                mutate_logger.info("mut_result: LS search no suitable layer\n")
                f.write("mut_result:LS search no suitable layer\n")
                f.write("{} generation!\n\n".format(generations))
                f.close()
                return "LS search no suitable layer"

        switch_layer1, switch_layer2 = model.get_layers(switch_layer1_name), model.get_layers(switch_layer2_name)
        if not switch_layer1.__class__.__name__ == switch_layer2.__class__.__name__ and model.get_outshape(
                switch_layer1_name) == model.get_outshape(switch_layer2_name):
            break

    mutate_logger.info("switch layer1:" + switch_layer1_name)
    mutate_logger.info("switch layer2:" + switch_layer2_name)
    f.write("switch layer1:" + switch_layer1_name + "\n")
    f.write("switch layer2:" + switch_layer2_name + "\n")

    try:
        temp1, temp2 = deepcopy(switch_layer1), deepcopy(switch_layer2)
    except Exception as e:
        temp1, temp2 = switch_layer1, switch_layer2


    set_layer(model, temp2, switch_layer1_name, f, "LS", generations, mutate_logger)
    set_layer(model, temp1, switch_layer2_name, f, "LS", generations, mutate_logger)

    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)
    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def LC_mut(model, layer_names, input_size, add_layer_type, mut_file_path, generations, mut_layer_isBasic="",
           mutate_logger="", train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt LC mut_strategy!\n")
    mutate_logger.info("Adopt LC mut_strategy!")
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if mut_layer_isBasic == "":

        mut_layer_isBasic = True
        if mut_layer_isBasic:
            add_layer_type = "Basic_op"
        else:
            add_layer_type = "Cascade_op"

        if len(Cascade_OPs) == 0 and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            f.write("No suitable ops for LC mutation!\n")
            mutate_logger.info("No suitable ops for LC mutation!")
            return "No suitable ops for LC mutation!\n"

        elif not mut_layer_isBasic and len(Cascade_OPs) == 0:
            mut_layer_isBasic = True
            add_layer_type = "Basic_op"

        elif mut_layer_isBasic and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            mut_layer_isBasic = False
            add_layer_type = "Cascade_op"

    copy_flag = False
    search_times = 0
    while not copy_flag:
        if mut_layer_isBasic:
            mut_layer_name = np.random.permutation(Basic_OPS + model.add_Cascade_OPs)[0]
            in_shape = model.get_inshape(mut_layer_name)
            out_shape = model.get_outshape(mut_layer_name)
            topology_info = model.get_order(mut_layer_name)
            last_ops, next_ops = topology_info[0], topology_info[1]
        else:
            mut_layer_name = np.random.permutation(Cascade_OPs)[0]
            yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)
            last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
            in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), list(
                model.Cascade_OPs_outshapes[mut_layer_name])
            in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)

        mut_layer = model.get_layers(mut_layer_name)
        mut_layer_type = str(mut_layer.__class__.__name__)
        if mut_layer_type in basicop_copy_whitelist or mut_layer_type in cascadeop_copy_whitelist:
            copy_flag = True

        search_times +=1
        if search_times > len(layer_names):
            break
    if not copy_flag:
        f.write("Illegal LC mutate!\n")
        f.write("mut_result:{}\n".format("LC Create illegal layer!"))
        f.write("{} generation!\n\n".format(generations))

        mutate_logger.info("Illegal LC mutate!")
        mutate_logger.info("mut_result:{}".format("LC Create illegal layer!"))
        mutate_logger.info("{} generation!\n".format(generations))
        return "LC Create illegal layer!"


    f.write("select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
        in_shape) + " out_shape: " + str(out_shape) + "\n")
    f.write("mut Basic type: " + str(mut_layer_isBasic) + "\n")
    f.write("add Basic layer : " + str(add_layer_type) + "\n")
    mutate_logger.info(
        "select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
            in_shape) + " out_shape: " + str(out_shape))
    mutate_logger.info("mut Basic type: " + str(mut_layer_isBasic))
    mutate_logger.info("add Basic layer : " + str(add_layer_type))

    if add_layer_type == "Basic_op":
        insert_layer = BasicOPUtils.copy_basicop(mut_layer)

    elif add_layer_type == "Cascade_op":
        insert_layer = CascadeOPUtils.copy_cascadeop(mut_layer)

    if insert_layer is None:
        mutate_logger.info("Unknown operator copied!")
        raise RuntimeError("Unknown operator copied!")

    f.write("select insert layer: " + str(insert_layer) + "\n")
    mutate_logger.info("select insert layer: " + str(insert_layer))


    tcflag = False
    error_info = ""
    for dtype in ms_dtypes:
        test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(out_shape)), dtype)

        try:
            insert_layer_outshape = insert_layer(test_insert_layer_data).shape
        except Exception as e:
            error_info = str(e)
        else:
            tcflag = True
            break

    if not tcflag:
        f.write("Illegal LC mutate!\n")
        f.write(error_info + "\n")
        f.write("mut_result:{}\n".format("LC Create illegal layer!"))
        f.write("{} generation!\n\n".format(generations))

        mutate_logger.info("Illegal LC mutate!")
        mutate_logger.info("mut_result:{}".format("LC Create illegal layer!"))
        mutate_logger.info("Exception information: \n" + error_info)
        mutate_logger.info("{} generation!\n".format(generations))
        return "LC Create illegal layer!"

    if not tuple(insert_layer_outshape) == tuple(out_shape):
        f.write("insert_layer_outshape not equal!: " + str(out_shape) + "\n")
        mutate_logger.info("insert_layer_outshape not equal!: " + str(out_shape))
        replace_cell = create_replacecell(tuple(insert_layer_outshape), tuple(out_shape))
        insert_layer = nn.SequentialCell([mut_layer, insert_layer, replace_cell])
    else:
        f.write("insert_layer_outshape equal!" + "\n")
        mutate_logger.info("insert_layer_outshape equal!")
        insert_layer = nn.SequentialCell([mut_layer, insert_layer])

    set_result = set_layer(model, insert_layer, mut_layer_name, f, "LC", generations, mutate_logger)
    if set_result is not True:
        return set_result

    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)
    # add
    if mut_layer_isBasic:
        if add_layer_type == "Baisc_op":  # Basic -> Cascade[Basic,Basic]
            f.write("basic_op copied!\n")
            mutate_logger.info("basic op copied!\n")
        elif add_layer_type == "Cascade_op":  # Basic -> Cascade[Basic,Cascade]
            f.write("Cascade_op copied!\n")
            mutate_logger.info("Cascade_op copied!")

        if mut_layer_name not in model.add_Cascade_OPs:
            model.add_Cascade_OPs.append(mut_layer_name)
        if mut_layer_name in Basic_OPS:
            del Basic_OPS[Basic_OPS.index(mut_layer_name)]
        model.set_Basic_OPS(Basic_OPS)

    else:
        if add_layer_type == "Basic_op":  # Cascade -> Cascade[Cascade,Basic]
            f.write("add Basic_op after Cascade_op!\n")
            mutate_logger.info("add Basic_op after Cascade_op!")
        elif add_layer_type == "Cascade_op":  # Cascade -> Cascade[Cascade,Cascade]
            f.write("add Cascade_op after basicop!\n")
            mutate_logger.info("add Cascade_op after basicop!")

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

    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def SM_mut(model, layer_names, input_size, mut_file_path, generations, mut_layer_isBasic="", mutate_logger="",
           train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt SM mut_strategy!\n")
    mutate_logger.info("Adopt SM mut_strategy!")
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if mut_layer_isBasic == "":
        mut_layer_isBasic = np.random.permutation([True, False])[0]
        if len(Cascade_OPs) == 0 and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            f.write("mut_result:No suitable ops for SM mutation!\n")
            mutate_logger.info("mut_result:No suitable ops for SM mutation!")
            return "mut_result:No suitable ops for SM mutation!\n"
        elif not mut_layer_isBasic and len(Cascade_OPs) == 0:
            mut_layer_isBasic = True
        elif mut_layer_isBasic and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            mut_layer_isBasic = False

    if mut_layer_isBasic:
        mut_layer_name = np.random.permutation(Basic_OPS + model.add_Cascade_OPs)[0]
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
        topology_info = model.get_order(mut_layer_name)
        last_ops, next_ops = topology_info[0], topology_info[1]
    else:
        mut_layer_name = np.random.permutation(Cascade_OPs)[0]
        yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)
        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), list(
            model.Cascade_OPs_outshapes[mut_layer_name])
        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)

    mut_layer = model.get_layers(mut_layer_name)
    op_in_shape, op_out_shape = deepcopy(in_shape), deepcopy(out_shape)

    f.write("select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
        in_shape) + " out_shape: " + str(out_shape) + "\n")
    f.write("mut Basic type: " + str(mut_layer_isBasic) + "\n")

    mutate_logger.info(
        "select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
            in_shape) + " out_shape: " + str(out_shape))
    mutate_logger.info("mut Basic type: " + str(mut_layer_isBasic))

    mutate_layer_indice = -1
    if "Sequential" in str(mut_layer.__class__.__name__):
        mutate_layer_indices = []
        for i in range(len(mut_layer)):
            if not "replace" in str(mut_layer[mutate_layer_indice].__class__.__name__).lower():
                mutate_layer_indices.append(i)
        if len(mutate_layer_indices) == 0:
            f.write("mut_result:No suitable ops for SM mutation!\n")
            f.write("{} generation!\n\n".format(generations))
            mutate_logger.info("No suitable ops for SM mutation!")
            mutate_logger.info("{} generation!".format(generations))
            return "mut_result:No suitable ops for SM mutation!\n"

        mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])

    if not mutate_layer_indice == -1:


        idx = mutate_layer_indice

        while not "replace" in str(mut_layer[idx].__class__.__name__).lower() and idx >= 0:
            idx -= 1

        if idx < 0:
            idx = mutate_layer_indice

        tcflag = False
        for dtype in ms_dtypes:
            test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)
            try:
                temp_data = mut_layer[:mutate_layer_indice](test_insert_layer_data)
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
        mutate_logger.info("candidate_in_mutlayers_indice:{}".format(mutate_layer_indice))
        f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

    else:
        mutate_logger.info("candidate_in_mutlayers_indice:-1")
        f.write("candidate_in_mutlayers_indice:-1\n")

    input_shape_dimension_mut, output_shape_dimension_mut = np.random.randint(2, 6), np.random.randint(2, 6)
    input_shape_mut, output_shape_mut = [input_size[0]], [input_size[0]]

    high = 51
    for i in range(input_shape_dimension_mut - 1):
        input_shape_mut.append(np.random.randint(1, high))

    for i in range(output_shape_dimension_mut - 1):
        output_shape_mut.append(np.random.randint(1, high))

    input_replace_cell1, input_replace_cell2 = Replace_ms(op_in_shape, input_shape_mut), Replace_ms(input_shape_mut,
                                                                                                    op_in_shape)
    output_replace_cell1, output_replace_cell2 = Replace_ms(op_out_shape, output_shape_mut), Replace_ms(
        output_shape_mut, op_out_shape)

    mut_state = np.random.randint(0, 3)

    if not mutate_layer_indice == -1:
        if mut_state == 0:
            mut_layer_slice = mut_layer[:idx + 1]
            mut_layer_slice.append(input_replace_cell1)
            mut_layer_slice.append(input_replace_cell2)
            mut_layer_slice.append(mut_layer[mutate_layer_indice])
            insert_layer = mut_layer_slice
            f.write("mutate state: before\n")
            f.write("mutate input_shape: " + str(input_shape_mut) + "\n")
            mutate_logger.info("mutate input_shape: " + str(input_shape_mut))

        elif mut_state == 1:
            mut_layer_slice = mut_layer[:mutate_layer_indice + 1]
            mut_layer_slice.append(output_replace_cell1)
            mut_layer_slice.append(output_replace_cell2)
            insert_layer = mut_layer_slice
            f.write("mutate state: after\n")
            f.write("mutate output_shape: " + str(output_shape_mut) + "\n")
            mutate_logger.info("mutate output_shape: " + str(output_shape_mut))

        elif mut_state == 2:
            mut_layer_slice = mut_layer[:mutate_layer_indice + 1]
            mut_layer_slice.append(input_replace_cell1)
            mut_layer_slice.append(input_replace_cell2)
            mut_layer_slice.append(mut_layer[mutate_layer_indice])
            mut_layer_slice.append(output_replace_cell1)
            mut_layer_slice.append(output_replace_cell2)
            insert_layer = mut_layer_slice
            f.write("mutate state: all\n")
            f.write("mutate input_shape: " + str(input_shape_mut) + "\n")
            f.write("mutate output_shape: " + str(output_shape_mut) + "\n")
            mutate_logger.info("mutate input_shape: " + str(input_shape_mut))
            mutate_logger.info("mutate output_shape: " + str(output_shape_mut))

    else:
        if mut_state == 0:
            insert_layer = nn.SequentialCell([input_replace_cell1, input_replace_cell2, mut_layer])
            f.write("mutate state: before\n")
            f.write("mutate input_shape: " + str(input_shape_mut) + "\n")
            mutate_logger.info("mutate input_shape: " + str(input_shape_mut))

        elif mut_state == 1:
            insert_layer = nn.SequentialCell([mut_layer, output_replace_cell1, output_replace_cell2])
            f.write("mutate state: after\n")
            f.write("mutate output_shape: " + str(output_shape_mut) + "\n")
            mutate_logger.info("mutate output_shape: " + str(output_shape_mut))

        elif mut_state == 2:
            insert_layer = nn.SequentialCell(
                [input_replace_cell1, input_replace_cell2, mut_layer, output_replace_cell1, output_replace_cell2])
            f.write("mutate state: all\n")
            f.write("mutate input_shape: " + str(input_shape_mut) + "\n")
            f.write("mutate output_shape: " + str(output_shape_mut) + "\n")
            mutate_logger.info("mutate input_shape: " + str(input_shape_mut))
            mutate_logger.info("mutate output_shape: " + str(output_shape_mut))

    tcflag = False
    error_info = ""
    for dtype in ms_dtypes:
        test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)

        try:
            insert_layer(test_insert_layer_data)
        except Exception as e:
            error_info = str(e)
        else:
            tcflag = True
            break

    if not tcflag:
        f.write("Illegal SM mutate!\n")
        f.write(error_info + "\n")
        f.write("mut_result:{}\n".format("SM Create illegal layer!"))
        f.write("{} generation!\n\n".format(generations))

        mutate_logger.info("Illegal SM mutate!")
        mutate_logger.info("mut_result:{}".format("SM Create illegal layer!"))
        mutate_logger.info("Exception information: \n" + error_info)
        mutate_logger.info("{} generation!\n".format(generations))
        return "SM Create illegal layer!"

    set_result = set_layer(model, insert_layer, mut_layer_name, f, "SM", generations, mutate_logger)
    if set_result is not True:
        return set_result

    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)

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

    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result


def DM_mut(model, layer_names, input_size, mut_file_path, generations, mut_layer_isBasic="", mutate_logger="",
           train_configs=""):
    f = open(mut_file_path, 'a+')
    f.write("Adopt DM mut_strategy!\n")
    mutate_logger.info("Adopt DM mut_strategy!")
    Cascade_OPs = deepcopy(model.get_Cascade_OPs())
    Basic_OPS = deepcopy(model.get_Basic_OPS())

    if mut_layer_isBasic == "":
        mut_layer_isBasic = np.random.permutation([True, False])[0]
        if len(Cascade_OPs) == 0 and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            f.write("mut_result:No suitable ops for DM mutation!\n")
            mutate_logger.info("mut_result:No suitable ops for SM mutation!")
            return "mut_result:No suitable ops for DM mutation!\n"
        elif not mut_layer_isBasic and len(Cascade_OPs) == 0:
            mut_layer_isBasic = True
        elif mut_layer_isBasic and (len(Basic_OPS) + len(model.add_Cascade_OPs)) == 0:
            mut_layer_isBasic = False

    if mut_layer_isBasic:
        mut_layer_name = np.random.permutation(Basic_OPS + model.add_Cascade_OPs)[0]
        in_shape = model.get_inshape(mut_layer_name)
        out_shape = model.get_outshape(mut_layer_name)
        topology_info = model.get_order(mut_layer_name)
        last_ops, next_ops = topology_info[0], topology_info[1]
    else:
        mut_layer_name = np.random.permutation(Cascade_OPs)[0]
        yezi_ops = find_Child_leaf_OP(layer_names, mut_layer_name, Basic_OPS, model.add_Cascade_OPs)
        last_ops, next_ops, _, _ = find_Cascade_OP_shape(model, input_size, mut_layer_name, yezi_ops)
        in_shape_list, out_shape_list = list(model.Cascade_OPs_inshapes[mut_layer_name]), list(
            model.Cascade_OPs_outshapes[mut_layer_name])
        in_shape, out_shape = tuple(in_shape_list), tuple(out_shape_list)

    mut_layer = model.get_layers(mut_layer_name)

    f.write("select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
        in_shape) + " out_shape: " + str(out_shape) + "\n")
    f.write("mut Basic type: " + str(mut_layer_isBasic) + "\n")

    mutate_logger.info(
        "select layer: " + mut_layer_name + " layer_type: " + str(mut_layer.__class__) + " " + "in_shape: " + str(
            in_shape) + " out_shape: " + str(out_shape))
    mutate_logger.info("mut Basic type: " + str(mut_layer_isBasic))

    mutate_layer_indice = -1
    if "Sequential" in str(mut_layer.__class__.__name__):
        mutate_layer_indices = []
        for i in range(len(mut_layer)):
            if not "replace" in str(mut_layer[mutate_layer_indice].__class__.__name__).lower():
                mutate_layer_indices.append(i)
        if len(mutate_layer_indices) == 0:
            f.write("mut_result:No suitable ops for DM mutation!\n")
            f.write("{} generation!\n\n".format(generations))
            mutate_logger.info("No suitable ops for DM mutation!")
            mutate_logger.info("{} generation!".format(generations))
            return "mut_result:No suitable ops for DM mutation!\n"

        mutate_layer_indice = int(np.random.permutation(mutate_layer_indices)[0])

    if not mutate_layer_indice == -1:


        idx = mutate_layer_indice

        while not "replace" in str(mut_layer[idx].__class__.__name__).lower() and idx >= 0:
            idx -= 1

        if idx < 0:
            idx = mutate_layer_indice

        tcflag = False
        for dtype in ms_dtypes:
            test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)
            try:
                mut_layer[:idx](test_insert_layer_data)
            except Exception as e:
                pass
            else:
                tcflag = True
                break
        if not tcflag:
            raise RuntimeError("can not achieve the correct dtype")

        mutate_logger.info("candidate_in_mutlayers_indice:{}".format(mutate_layer_indice))
        f.write("candidate_in_mutlayers_indice:{}\n".format(mutate_layer_indice))

    else:
        tcflag = False
        for dtype in ms_dtypes:
            test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)
            try:
                mut_layer(test_insert_layer_data)
            except Exception as e:
                pass
            else:
                tcflag = True
                break
        if not tcflag:
            raise RuntimeError("can not achieve the correct dtype")

        mutate_logger.info("candidate_in_mutlayers_indice:-1")
        f.write("candidate_in_mutlayers_indice:-1\n")

    if "float" in str(dtype.__class__.__name__).lower():
        newdtype = np.random.permutation([mindspore.float16, mindspore.float32])[0]

    elif "int" in str(dtype.__class__.__name__).lower():
        newdtype = np.random.permutation([mindspore.int16, mindspore.int32])[0]

    mutate_logger.info("in_dtype:{}".format(str(newdtype)))
    f.write("in_dtype:{}\n".format(str(newdtype)))


    in_dtype = dtypecast(newdtype)
    out_dtype = dtypecast(mindspore.float32)
    if not mutate_layer_indice == -1:
        mut_layer[idx].to_float(newdtype)
        new_layer = nn.SequentialCell([in_dtype, mut_layer, out_dtype])
    else:
        new_layer = nn.SequentialCell([in_dtype, mut_layer.to_float(newdtype), out_dtype])

    tcflag = False
    error_info = ""
    for dtype in ms_dtypes:
        test_insert_layer_data = mindspore.Tensor(np.random.randn(*tuple(in_shape)), dtype)

        try:
            new_layer(test_insert_layer_data)
        except Exception as e:
            error_info = str(e)
        else:
            tcflag = True
            break

    if not tcflag:
        f.write("Illegal DM mutate!\n")
        f.write(error_info + "\n")
        f.write("mut_result:{}\n".format("DM Create illegal layer!"))
        f.write("{} generation!\n\n".format(generations))

        mutate_logger.info("Illegal DM mutate!")
        mutate_logger.info("mut_result:{}".format("DM Create illegal layer!"))
        mutate_logger.info("Exception information: \n" + error_info)
        mutate_logger.info("{} generation!\n".format(generations))
        return "DM Create illegal layer!"

    set_result = set_layer(model, new_layer, mut_layer_name, f, "DM", generations, mutate_logger)
    if set_result is not True:
        return set_result

    test_result = judge_legenacy(model, input_size, mutate_logger, train_configs)

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

    f.write("mut_result:{}\n".format(str(test_result)))
    f.write("{} generation!\n\n".format(generations))
    mutate_logger.info("mut_result:{}".format(str(test_result)))
    mutate_logger.info("{} generation!\n".format(generations))
    f.close()
    return test_result
