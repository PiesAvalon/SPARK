"""
    优化器配置规范
"""
import src.optimizers as optim


optimizer_optional = {
    'Adam': {'lr': float or int, 'beta1': float or int, 'beta2': float or int, 'eps': float or int,
             'decay': float or int},
    'RMSprop': {'lr': float or int, 'momentum': float or int, 'centered': bool, 'eps': float or int},
    'SGD': {'lr': float or int, 'momentum': float or int, 'nesterov': bool, 'decay': float or int},
}

optimizer_method_table = {
    'Adam': 'optim.adam.get_optimizer(framework, params, network_parameters)',
    'RMSprop': 'optim.rms_prop.get_optimizer(framework, params, network_parameters)',
    'SGD': 'optim.sgd.get_optimizer(framework, params, network_parameters)',
}
