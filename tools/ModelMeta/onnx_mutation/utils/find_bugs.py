import onnxruntime as ort
import numpy as np


# Load ONNX model
model_path = './testing/14300_vgg.onnx'
session = ort.InferenceSession(model_path)

# Get input name and shape
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Set input data
input_data = np.zeros(input_shape, np.float32)

# Run the model
outputs = session.run(None, {input_name: input_data})

# Get the output of the 2000-th node
print(outputs)
node_index = 1
node_output = outputs[node_index]

# Get the operator type of the 2000-th node
node_type = session.get_modelmeta().graph.node[node_index].op_type
print(f'Node {node_index} operator type:', node_type)
