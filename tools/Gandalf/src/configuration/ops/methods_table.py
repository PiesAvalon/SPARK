"""
    算子封装方法对照表
"""

methods_table = {
    'Conv1d': 'ops.conv1d.get_op_and_shape',
    'Conv2d': 'ops.conv2d.get_op_and_shape',
    'Conv3d': 'ops.conv3d.get_op_and_shape',
    'Conv2dTranspose': 'ops.conv2dtranspose.get_op_and_shape',
    'Conv3dTranspose': 'ops.conv3dtranspose.get_op_and_shape',
    'DepthwiseConv1d': 'ops.depthwise_conv1d.get_op_and_shape',
    'DepthwiseConv2d': 'ops.depthwise_conv2d.get_op_and_shape',
    'SeparableConv1d': 'ops.separable_conv1d.get_op_and_shape',
    'SeparableConv2d': 'ops.separable_conv2d.get_op_and_shape',

    'Embedding': 'ops.embedding.get_op_and_shape',

    'Flatten': 'ops.flatten.get_op_and_shape',
    'Reshape': 'ops.reshape.get_op_and_shape',
    'Squeeze': 'ops.squeeze.get_op_and_shape',
    'Unsqueeze': 'ops.unsqueeze.get_op_and_shape',
    'Transpose': 'ops.transpose.get_op_and_shape',

    'Shape': 'ops.shape.get_op_and_shape',
    'Gather': 'ops.gather.get_op_and_shape',

    'Resize': 'ops.resize.get_op_and_shape',
    'Upsample1d': 'ops.upsample1d.get_op_and_shape',
    'Upsample2d': 'ops.upsample2d.get_op_and_shape',
    'Upsample3d': 'ops.upsample3d.get_op_and_shape',
    'CropAndResize': 'ops.crop_and_resize.get_op_and_shape',

    'Dense': 'ops.dense.get_op_and_shape',
    'BiasAdd': 'ops.bias_add.get_op_and_shape',

    'Dropout': 'ops.dropout.get_op_and_shape',

    'ReduceMean': 'ops.reduce_mean.get_op_and_shape',
    'ReduceMax': 'ops.reduce_max.get_op_and_shape',
    'ReduceSum': 'ops.reduce_sum.get_op_and_shape',
    'ReduceProd': 'ops.reduce_prod.get_op_and_shape',

    'Argmax': 'ops.argmax.get_op_and_shape',
    'Argmin': 'ops.argmin.get_op_and_shape',

    'Cast': 'ops.cast.get_op_and_shape',

    'Ceil': 'ops.ceil.get_op_and_shape',
    'Floor': 'ops.floor.get_op_and_shape',
    "Exp": 'ops.exp.get_op_and_shape',
    "Compare": 'ops.compare.get_op_and_shape',
    "Sqrt": 'ops.sqrt.get_op_and_shape',
    "Square": 'ops.square.get_op_and_shape',
    "Rsqrt": 'ops.rsqrt.get_op_and_shape',
    "Tile": 'ops.tile.get_op_and_shape',
    "TopK": 'ops.top_k.get_op_and_shape',
    "Slice": 'ops.slice.get_op_and_shape',
    "StridedSlice": 'ops.strided_slice.get_op_and_shape',
    'Repeat': 'ops.repeat.get_op_and_shape',

    'Threshold': 'ops.threshold.get_op_and_shape',
    'ReLU': 'ops.relu.get_op_and_shape',
    'ReLU6': 'ops.relu6.get_op_and_shape',
    'ELU': 'ops.elu.get_op_and_shape',
    'SeLU': 'ops.selu.get_op_and_shape',
    'PReLU': 'ops.p_relu.get_op_and_shape',
    'LeakyReLU': 'ops.leaky_relu.get_op_and_shape',
    'Softmax': 'ops.softmax.get_op_and_shape',
    'Sigmoid': 'ops.sigmoid.get_op_and_shape',
    'Tanh': 'ops.tanh.get_op_and_shape',

    'GlobalAvgPool1d': 'ops.global_avg_pool1d.get_op_and_shape',
    'GlobalAvgPool2d': 'ops.global_avg_pool2d.get_op_and_shape',
    'GlobalAvgPool3d': 'ops.global_avg_pool3d.get_op_and_shape',
    'GlobalMaxPool1d': 'ops.global_max_pool1d.get_op_and_shape',
    'GlobalMaxPool2d': 'ops.global_max_pool2d.get_op_and_shape',
    'GlobalMaxPool3d': 'ops.global_max_pool3d.get_op_and_shape',
    'AvgPool1d': 'ops.avg_pool1d.get_op_and_shape',
    'MaxPool1d': 'ops.max_pool1d.get_op_and_shape',
    'AvgPool2d': 'ops.avg_pool2d.get_op_and_shape',
    'MaxPool2d': 'ops.max_pool2d.get_op_and_shape',
    'AvgPool3d': 'ops.avg_pool3d.get_op_and_shape',
    'MaxPool3d': 'ops.max_pool3d.get_op_and_shape',

    'ZeroPadding2d': 'ops.zero_pad2d.get_op_and_shape',

    'BatchNorm1d': 'ops.batch_norm1d.get_op_and_shape',
    'BatchNorm2d': 'ops.batch_norm2d.get_op_and_shape',
    'BatchNorm3d': 'ops.batch_norm3d.get_op_and_shape',
    'LayerNorm1d': 'ops.layer_norm1d.get_op_and_shape',
    'LayerNorm2d': 'ops.layer_norm2d.get_op_and_shape',
    'LayerNorm3d': 'ops.layer_norm3d.get_op_and_shape',

    'Lambda': 'ops.lamb.get_op_and_shape',
    'GaussianNoise': 'ops.gaussian_noise.get_op_and_shape',

    'Add': 'ops.add.get_op_and_shape',
    'Subtract': 'ops.subtract.get_op_and_shape',
    'Multiply': 'ops.multiply.get_op_and_shape',
    'Divide': 'ops.divide.get_op_and_shape',
    'Maximum': 'ops.maximum.get_op_and_shape',
    'Minimum': 'ops.minimum.get_op_and_shape',
    'Average': 'ops.average.get_op_and_shape',
    'Concatenate': 'ops.concatenate.get_op_and_shape',

    'RNN': 'ops.rnn.get_op_and_shape',
    'GRU': 'ops.gru.get_op_and_shape',
    'LSTM': 'ops.lstm.get_op_and_shape',
    'RNNCell': 'ops.rnn_cell.get_op_and_shape',
    'GRUCell': 'ops.grun_cell.get_op_and_shape',
    'LSTMCell': 'ops.lstmn_cell.get_op_and_shape',
    'BiRNN': 'ops.bi_rnn.get_op_and_shape',
    'BiGRU': 'ops.bi_gru.get_op_and_shape',
    'BiLSTM': 'ops.bi_lstm.get_op_and_shape',
}