import onnx
import sclblonnx as so
from onnxsim import simplify
from sclblonnx.input import rename_input_image
from sclblonnx.output import rename_class_probabilities_output

onnx_path = 'banana_model.onnx'
output_onnx_path = 'named_banana_model.onnx'
classes = ['Too early', 'Ripe', 'Too late', 'No banana']

# Load the model
graph = so.graph_from_file(onnx_path)

# print IO names
print('Original IO names:')
so.list_inputs(graph)
so.list_outputs(graph)

# simplify the model
model_simp, check = simplify(onnx_path)
assert check, "Simplified ONNX model could not be validated"
# save the model
print('Saving simplified model to: ', output_onnx_path)
onnx.save(model_simp, output_onnx_path)

# rename the input and output names
graph = so.graph_from_file(output_onnx_path)
rename_input_image(graph, 'serving_default_sequential_1_input:0')
rename_class_probabilities_output(graph, 'StatefulPartitionedCall:0', classes)
so.graph_to_file(graph, output_onnx_path)
print('Saving renamed model to: ', output_onnx_path)

# Print new IO names
print('New IO names:')
graph = so.graph_from_file(output_onnx_path)
so.list_inputs(graph)
so.list_outputs(graph)
