import math
import random
import numpy as np
from common.mutation_torch.Layer_utils import BasicOPUtils

mutate_OPparam_names = {
    "num_features": "int",
    "keep_prob": "float",
    "group": "int",
    "eps": "float",
    "momentum": "float",
    "kernel_size": "tuple_int",
    "in_channels": "int",
    "out_channels": "int",
    "stride": "tuple_int",
}
mutate_OPparam_valranges = {
    "keep_prob": "4",
    "group": "4",
    "num_features": "4",
    "eps": "4",
    "momentum": "4",
    "kernel_size": "(1,4)",
    "in_channels": "4",
    "out_channels": "4",
    "stride": "(1,4)"
}


def get_PM_new_layer_ms(mutate_layer, mutate_layer_type, mutate_param_selname, new_value, mutate_replace_layer_inshape,
                        f="", mutate_logger="", generations=-1):
    from common.mutation_ms.Layer_utils import BasicOPUtils
    basic_ops_dict = BasicOPUtils()
    if "conv" in mutate_layer_type.lower():
        copy_op = basic_ops_dict.extension_ops['conv']
        param1, param2, param3, param4, param5, param6, param7 = mutate_layer.in_channels, mutate_layer.out_channels, \
            mutate_layer.kernel_size, mutate_layer.stride, mutate_layer.group, mutate_layer.has_bias, mutate_layer_type

        if "stride" == mutate_param_selname:
            param4 = new_value
        elif "kernel_size" == mutate_param_selname:
            param3 = new_value
        elif "in_channels" == mutate_param_selname:
            while not (new_value % mutate_layer.group == 0 and new_value > mutate_layer.group):
                new_value += 1
            param1 = new_value
            mutate_replace_layer_inshape[1] = param1
        elif "out_channels" == mutate_param_selname:
            while not (new_value % mutate_layer.group == 0 and new_value > mutate_layer.group):
                new_value += 1
            param2 = new_value
        elif "group" == mutate_param_selname:
            if "3d" in mutate_layer_type.lower():
                f.write("conv3d can not mutate group param!\n")
                f.write("mut_result:{}\n".format("conv3d can not mutate group param!"))
                f.write("{} generation!\n\n".format(generations))
                mutate_logger.info("conv3d can not mutate group param!")
                mutate_logger.info("mut_result:{}".format("conv3d can not mutate group param"))
                mutate_logger.info("{} generation!\n".format(generations))
                return "conv3d can not mutate group param!"

            while not (mutate_layer.in_channels % new_value == 0 and mutate_layer.out_channels % new_value == 0
                       and new_value > 0):
                new_value -= 1
            param5 = new_value
        elif "has_bias" == mutate_param_selname:
            param6 = new_value

        mutate_replace_layer = copy_op(param1=param1, param2=param2, param3=param3, param4=param4, param5=param5,
                                       param6=param6, param7=param7)

    elif "pool" in mutate_layer_type.lower():
        copy_op = basic_ops_dict.extension_ops['pool']
        param1, param2, param3 = mutate_layer.kernel_size, mutate_layer.stride, mutate_layer_type

        if "kernel_size" == mutate_param_selname:
            param1 = new_value
        elif "stride" == mutate_param_selname:
            param2 = new_value
        mutate_replace_layer = copy_op(param1=param1, param2=param2, param3=param3)

    elif "dense" in mutate_layer_type.lower():
        copy_op = basic_ops_dict.extension_ops["dense"]
        param1, param2, param3 = mutate_layer.in_channels, mutate_layer.out_channels, mutate_layer.has_bias

        if "in_channels" == mutate_param_selname:
            param1 = new_value
            mutate_replace_layer_inshape[1] = new_value

        elif "out_channels" == mutate_param_selname:
            param2 = new_value

        elif "has_bias" == mutate_param_selname:
            param3 = new_value

        mutate_replace_layer = copy_op(param1=param1, param2=param2, param3=param3)

    elif "batchnorm" in mutate_layer_type.lower():
        copy_op = basic_ops_dict.extension_ops["batchnorm"]
        param1, param2, param3, param4 = mutate_layer.num_features, mutate_layer.eps, mutate_layer.momentum, mutate_layer_type

        if "num_features" == mutate_param_selname:
            if "3d" in mutate_layer_type.lower():
                f.write("batchnorm3d can not mutate num_features param!\n")
                f.write("mut_result:{}\n".format("batchnorm3d can not mutate num_features param!"))
                f.write("{} generation!\n\n".format(generations))
                mutate_logger.info("batchnorm3d can not mutate num_features param!")
                mutate_logger.info("mut_result:{}".format("batchnorm3d can not mutate num_features param!"))
                mutate_logger.info("{} generation!\n".format(generations))
                return "batchnorm3d can not mutate num_features param!"

            param1 = new_value
            mutate_replace_layer_inshape[1] = param1

        elif "eps" == mutate_param_selname:
            param2 = new_value

        elif "momentum" == mutate_param_selname:

            if new_value > 1:
                new_value = new_value / math.ceil(new_value)

            param3 = new_value

        mutate_replace_layer = copy_op(param1=param1, param2=param2, param3=param3, param4=param4)

    elif "dropout" in mutate_layer_type.lower():
        copy_op = basic_ops_dict.extension_ops['dropout']
        param1 = new_value
        if "keep_prob" == mutate_param_selname:
            mutate_replace_layer = copy_op(param1=param1)

    elif "droppath" in mutate_layer_type.lower():
        if "keep_prob" == mutate_param_selname:
            mutate_replace_layer = BasicOPUtils.available_Droppath(new_value)

    return mutate_replace_layer, mutate_replace_layer_inshape, new_value


def get_PM_new_layer_torch(mutate_layer, mutate_layer_type, mutate_param_selname, new_value,
                           mutate_replace_layer_inshape):
    basic_ops_dict = BasicOPUtils()
    if "conv" in mutate_layer_type.lower():
        copy_op = basic_ops_dict.extension_ops['conv']
        param1, param2, param3, param4, param5, param6, param7 = mutate_layer.in_channels, mutate_layer.out_channels, \
            mutate_layer.kernel_size, mutate_layer.stride, mutate_layer.groups, mutate_layer.bias is not None, \
            mutate_layer_type

        if "stride" == mutate_param_selname:
            param4 = new_value
        elif "kernel_size" == mutate_param_selname:
            param3 = new_value
        elif "in_channels" == mutate_param_selname:
            param1 = new_value
            mutate_replace_layer_inshape[1] = param1
        elif "out_channels" == mutate_param_selname:
            param2 = new_value
        elif "groups" in mutate_param_selname:
            param5 = new_value

        mutate_replace_layer = copy_op(param1=param1, param2=param2, param3=param3, param4=param4, param5=param5,
                                       param6=param6, param7=param7)

    elif "pool" in mutate_layer_type.lower():
        copy_op = basic_ops_dict.extension_ops['pool']
        param1, param2, param3 = mutate_layer.kernel_size, mutate_layer.stride, mutate_layer_type

        if "kernel_size" == mutate_param_selname:
            param1 = new_value
        elif "stride" == mutate_param_selname:
            param2 = new_value

        mutate_replace_layer = copy_op(param1=param1, param2=param2, param3=param3)

    elif "linear" in mutate_layer_type.lower():
        copy_op = basic_ops_dict.extension_ops["linear"]
        param1, param2, param3 = mutate_layer.in_features, mutate_layer.out_features, mutate_layer.bias is not None

        if "in_channels" == mutate_param_selname:
            param1 = new_value
            mutate_replace_layer_inshape[1] = new_value
        elif "out_channels" == mutate_param_selname:
            param2 = new_value

        mutate_replace_layer = copy_op(param1=param1, param2=param2, param3=param3)

    elif "batchnorm" in mutate_layer_type.lower():
        copy_op = basic_ops_dict.extension_ops["batchnorm"]
        param1, param2, param3, param4 = mutate_layer.num_features, mutate_layer.eps, mutate_layer.momentum, mutate_layer_type

        if "num_features" == mutate_param_selname:
            param1 = new_value
            mutate_replace_layer_inshape[1] = param1
        elif "eps" == mutate_param_selname:
            param2 = new_value
        elif "momentum" == mutate_param_selname:
            param3 = new_value

        mutate_replace_layer = copy_op(param1=param1, param2=param2, param3=param3, param4=param4)

    elif "dropout" in mutate_layer_type.lower():
        copy_op = basic_ops_dict.extension_ops['dropout']
        param1 = new_value
        if "keep_prob" == mutate_param_selname:
            mutate_replace_layer = copy_op(param1)

    elif "droppath" in mutate_layer_type.lower():
        if "keep_prob" == mutate_param_selname:
            mutate_replace_layer = BasicOPUtils.available_Droppath(new_value)

    return mutate_replace_layer, mutate_replace_layer_inshape


def get_new_basicop(alternative_insert_layers, activation_names, insert_inchannels, insert_outchannels, kernel_size,
                    stride, dimension):
    alternative_insert_layers_instances = []
    keys = alternative_insert_layers.keys()
    for mutate_layer_type in keys:

        flag = False
        if "conv" in mutate_layer_type.lower():
            flag = True
            copy_op = alternative_insert_layers['conv']
            mutate_replace_layer = list(copy_op(insert_inchannels, insert_outchannels, kernel_size, stride, dimension))

        elif mutate_layer_type.lower() in activation_names:
            flag = True
            copy_op = alternative_insert_layers[mutate_layer_type]
            mutate_replace_layer = copy_op

        elif "pool" in mutate_layer_type.lower():
            flag = True
            copy_op = alternative_insert_layers['pool']
            mutate_replace_layer = list(copy_op(kernel_size, stride, dimension))

        elif "dense" == mutate_layer_type.lower() or "linear" == mutate_layer_type.lower():
            flag = True
            if "dense" == mutate_layer_type.lower():
                copy_op = alternative_insert_layers["dense"]
            else:
                copy_op = alternative_insert_layers["linear"]
            mutate_replace_layer = copy_op(param1=insert_inchannels, param2=insert_outchannels,
                                           param3=random.choice([True, False]))

        elif "batchnorm" in mutate_layer_type.lower():
            flag = True
            copy_op = alternative_insert_layers["batchnorm"]
            mutate_replace_layer = copy_op(param1=insert_inchannels, param2=dimension)

        elif "dropout" == mutate_layer_type.lower():
            flag = True
            copy_op = alternative_insert_layers['dropout']
            mutate_replace_layer = copy_op(param1=0.5)

        if flag:
            if isinstance(mutate_replace_layer, list) or isinstance(mutate_replace_layer, type(np.array)):
                alternative_insert_layers_instances = alternative_insert_layers_instances + mutate_replace_layer
            else:
                alternative_insert_layers_instances.append(mutate_replace_layer)

    return alternative_insert_layers_instances


def get_new_cascadeop(alternative_insert_layers, insert_inchannels, insert_outchannels, kernel_size, stride,
                      activation):
    alternative_insert_layers_instances = []
    keys = alternative_insert_layers.keys()
    for mutate_layer_type in keys:
        if "convbnrelu" in mutate_layer_type.lower():
            copy_op = alternative_insert_layers['convbnrelu']
            mutate_replace_layer = copy_op(insert_inchannels, insert_outchannels, kernel_size, stride)

        elif "downsample" in mutate_layer_type.lower():
            copy_op = alternative_insert_layers['downsample']
            mutate_replace_layer = copy_op(insert_inchannels, insert_outchannels, kernel_size, stride)

        elif "dwpw_group" in mutate_layer_type.lower() and (
                insert_inchannels % insert_outchannels == 0 and insert_inchannels <= insert_outchannels):
            copy_op = alternative_insert_layers["dwpw_group"]
            mutate_replace_layer = copy_op(insert_inchannels, insert_outchannels, kernel_size, stride, activation)

        elif "se" in mutate_layer_type.lower():
            copy_op = alternative_insert_layers["se"]
            mutate_replace_layer = copy_op()

        elif "denselayer" in mutate_layer_type.lower():
            copy_op = alternative_insert_layers['denselayer']
            mutate_replace_layer = copy_op(insert_inchannels, insert_outchannels)

        elif "residualblock" in mutate_layer_type.lower() and insert_outchannels // 4 > 0:
            copy_op = alternative_insert_layers['residualblock']
            mutate_replace_layer = copy_op(insert_inchannels, insert_outchannels, kernel_size, stride, activation)

        elif "pwdwpw_residualblock" in mutate_layer_type.lower() and insert_outchannels // 4 > 0:
            copy_op = alternative_insert_layers['pwdwpw_residualblock']
            mutate_replace_layer = copy_op(insert_inchannels, insert_outchannels, kernel_size, stride, activation)

        elif "inception" in mutate_layer_type.lower():
            copy_op = alternative_insert_layers['inception']
            mutate_replace_layer = copy_op()





        if isinstance(mutate_replace_layer, list) or isinstance(mutate_replace_layer, type(np.array)):
            alternative_insert_layers_instances = alternative_insert_layers_instances + mutate_replace_layer
        else:
            alternative_insert_layers_instances.append(mutate_replace_layer)

    return alternative_insert_layers_instances
