import mindspore

from common.model_utils import get_model
import torch
import numpy as np
if __name__ == '__main__':

    model_name = "pinns"
    input_size = (20150, 2)

    model1, model2 = get_model(model_name, input_size,scaned=True)

    data = np.random.randn(*input_size)
    x = mindspore.Tensor(data,mindspore.float32)
    output = model1(x)
    print(output[0].shape)


