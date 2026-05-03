import mindspore.nn as nn


def get_biDense(**kwargs):
    m = nn.BiDense(kwargs['param1'], kwargs['param2'], kwargs['param3'])
    return m

def get_upsample(**kwargs):
    upsample = nn.Upsample(size=kwargs['param1'], mode='nearest')
    return upsample

new_basic_ops = {
    'bidense': get_biDense,
    'upsample': get_upsample
}