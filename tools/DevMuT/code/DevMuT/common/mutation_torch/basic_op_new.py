import torch.nn as nn




def get_biDense(*kwargs):
    m = nn.Bilinear(kwargs[0], kwargs[1], kwargs[2])
    return m


def get_upsample(**kwargs):
    upsample = nn.Upsample(size=kwargs['param1'], mode='nearest')
    return upsample


new_basic_ops = {
    'bidense': get_biDense,
    'upsample': get_upsample
}
