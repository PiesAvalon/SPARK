import mindspore
import mindspore.nn as nn
import numpy as np
import torch




if __name__ == '__main__':
    dtypes = [mindspore.float32,mindspore.int32]
    mutate_replace_layer_inshape = (2, 3, 224, 224)
    mutate_replace_layer = nn.Conv2d(3, 64, 3, 3)

    for dtype in dtypes:
        test_input_data = mindspore.Tensor(np.random.randn(*tuple(mutate_replace_layer_inshape)), dtype)

        try:
            new_op_outshape = mutate_replace_layer(test_input_data).shape
        except Exception as e:
            print("fuck: ", dtype)
        else:
            print("success: ",dtype)
            break