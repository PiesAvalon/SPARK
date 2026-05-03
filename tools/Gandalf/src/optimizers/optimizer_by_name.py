import src.optimizers as optim
from src.optimizers.optimizer_configuration import optimizer_optional, optimizer_method_table

framework_need_network = ['PyTorch', 'Jittor']


def optimizer_by_name(name, framework, params, network_parameters=None):
    # 检查语法规则
    check_optimizer_configuration(name, framework, params, network_parameters)
    # 提取优化器
    optimizer_method = optimizer_method_table[name]
    return eval(optimizer_method)


def check_optimizer_configuration(name, framework, params, network_parameters):
    if framework in framework_need_network and network_parameters is None:
        raise Exception('The current framework need the parameters of your model as its params.')
    if not optimizer_optional.__contains__(name):
        raise Exception('No support Optimizer.')
    else:
        configuration = optimizer_optional[name]
        for k, v in params.items():
            if not configuration.__contains__(k):
                raise Exception('The current optimizer do not support param \"{0}\"'.format(k))
            else:
                if not isinstance(v, configuration[k]):
                    raise Exception('Something wrong with param \"{0}\" and value \"{1}\"'.format(k, v))
