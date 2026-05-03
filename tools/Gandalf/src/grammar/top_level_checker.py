"""
    顶层json语法检查器
"""
from src.configuration.ops.network_dictionary import layers_required, layers_range, layers_optional


class TopLevelChecker:
    framework_allowed = ['TensorFlow', 'PyTorch', 'Jittor']

    def check_json(self, json_ob):
        # 检查必需字段
        # framework字段
        if not json_ob.__contains__('framework'):
            raise Exception('Key \"framework\" missing.')
        else:
            if json_ob['framework'] not in TopLevelChecker.framework_allowed:
                raise Exception('No support DL framework.')
        # input_shape字段
        if not json_ob.__contains__('input_shape'):
            raise Exception('Key \"input_shape\" missing.')
        else:
            if not isinstance(json_ob['input_shape'], list):
                raise Exception('Key \"input_shape\" must be List of Integer.')
            for ob in json_ob['input_shape']:
                if not isinstance(ob, int):
                    raise Exception('Key \"input_shape\" must be List of Integer.')
        # network字段
        if not json_ob.__contains__('network'):
            raise Exception('Key \"network\" missing.')
        else:
            if not isinstance(json_ob['network'], list):
                raise Exception('Key \"network\" must be List of JSON.')
            for layer in json_ob['network']:
                if not isinstance(layer, dict):
                    raise Exception('Key \"network\" must be List of JSON.')

    def check_network(self, network):
        # 检查network字段的具体内容
        branch_to = set()
        index = set()
        for layer in network:
            # 检查name字段
            if not layer.__contains__('name'):
                raise Exception('Some layer\'s name in your network missing.')
            else:
                layer_name = layer['name']
                layer_params = layer['params'] if layer.__contains__('params') else {}
                if layer.__contains__('branch_to'):
                    if branch_to.__contains__(layer['branch_to']):
                        raise Exception('Duplicate \"branch_to\" value: {0}'.format(layer['branch_to']))
                    branch_to.add(layer['branch_to'])
                if layer.__contains__('index'):
                    # if index.__contains__(layer['index']):
                    #     raise Exception('Duplicate \"index\" value: {0}'.format(layer['index']))
                    if not branch_to.__contains__(layer['index']):
                        raise Exception('Unused \"index\" value {0} or defined before \"branch_to\"'.format(
                            layer['index']))
                    index.add(layer['index'])
                # 检查name是否合法
                if not layers_required.__contains__(layer_name):
                    raise Exception('No support layer name {0}.'.format(layer_name))
                # 检查必需参数项
                for k, v in layers_required[layer_name].items():
                    if not layer_params.__contains__(k):
                        raise Exception('Layer {0} missing parameter {1}.'.format(layer_name, k))
                    elif not isinstance(layer_params[k], v):
                        raise Exception('Layer {0} comes across wrong parameter format with {1}.'.format(layer_name, k))
                # 检查可选项
                for k, v in layers_optional[layer_name].items():
                    if layer_params.__contains__(k) and not isinstance(layer_params[k], v):
                        raise Exception('Layer {0} comes across wrong parameter format with {1}.'.format(layer_name, k))
                # 检查存在取值选项的参数
                for k, v in layers_range[layer_name].items():
                    if layer_params.__contains__(k) and layer_params[k] not in v:
                        raise Exception('Parameter {1} of layer {0} should be chosen between {2}.'.format(layer_name,
                                                                                                          k, v))
        # 检查 branch_to 和 index 字段
        if len(branch_to - index) != 0:
            raise Exception('Unimplemented \"branch_to\" values {0} exist'.format(branch_to - index))

    def grammar_checker(self, json_ob):
        self.check_json(json_ob)
        self.check_network(json_ob['network'])
