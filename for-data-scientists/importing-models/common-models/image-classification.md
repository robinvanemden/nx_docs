# Image classification

> Image Classification is a fundamental task in vision recognition that aims to understand and categorize an image as a whole under a specific label. Unlike object detection, which involves classification and location of multiple objects within an image, image classification typically pertains to single-object images.

In this section, we'll go through the process of exporting a trained image classification model and export to an ONNX file that conforms to the requirements mentioned [before](../../onnx-requirements.md).\
For the sake of example, we'll be using a model available in the PyTorch image models (aka. timm) library, specifically a ResNet-18 model.

This guide shows how a **trained** image classification model (implemented using PyTorch) can be exported to ONNX the right way. We'll be using image classification models implemented in the [timm](https://huggingface.co/docs/timm/quickstart) library.

### Requirements

Make sure to install the required packages:

```sh
pip install -r requirements.txt
```

<details>

<summary>requirements.txt</summary>

```
timm
sclblonnx
torch
onnxsim
```

</details>

### Exporting the model to ONNX

The following command exports a trained image classification model to ONNX and saves it locally:

```sh
python export_to_onnx.py
```

<details>

<summary>export_to_onnx.py</summary>

```python
import json
from os.path import join, dirname, abspath

import onnx
import sclblonnx as so
import timm
import torch
from torch import nn
from onnxsim import simplify

PATH = dirname(abspath(__file__))
classes_path = join(PATH, 'imagenet-classes.json') # https://github.com/anishathalye/imagenet-simple-labels/blob/master/imagenet-simple-labels.json

model_name = 'resnet18'
classes = json.load(open(classes_path))  # Replace with the classes that the model was trained on.
concatenated_classes = ';'.join([f'{i}:{c}' for i, c in enumerate(classes)])
onnx_opset_version = 12
output_onnx_path = join(PATH, f'{model_name}.onnx')

input_width = 224  # Replace with your input width
input_height = 224  # Replace with your input height
model_means = [0.485, 0.456, 0.406]  # Replace with your model means
model_means = [255 * m for m in model_means]  # Convert to 0-255 range
model_stds = [0.229, 0.224, 0.225]  # Replace with your model standard deviations
model_stds = [255 * s for s in model_stds]  # Convert to 0-255 range


# Define the model since the model's output needs to be softmaxed
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.sotfmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.sotfmax(x)
        return x


# Load the model
model = Model()
# Set the model to evaluation mode
model.eval()
# Define onnx IO names
input_names = ['image-']
output_names = [f'scores-{concatenated_classes}']
dummy_input = torch.rand(1, 3, input_width, input_height)

# Export model to ONNX
torch.onnx.export(model, dummy_input, output_onnx_path,
                  input_names=input_names, output_names=output_names,
                  opset_version=onnx_opset_version)

# Update the ONNX description
graph = so.graph_from_file(output_onnx_path)
# Add the model means and standard deviations to the ONNX graph description,
# because that's used by the toolchain to populate some settings.
graph.doc_string = json.dumps({'means': model_means, 'vars': model_stds})
so.graph_to_file(graph, output_onnx_path, onnx_opset_version=onnx_opset_version)

# Simplify the ONNX model
# This step is optional, but it is recommended to reduce the size of the model
# optimize the model for inference
try:
    model = onnx.load(output_onnx_path)
    model, check = simplify(model, check_n=10)
    assert check, "Couldn't simplify the ONNX model"
    onnx.save_model(model, output_onnx_path)
except Exception as e:
    print(f'Simplification failed: {e}')
    exit(1)
```

</details>

The exported ONNX can be uploaded on the platform and used for inference.

### In-depth inspection of the export script

* The ONNX model is expected to output probabilities (values in range \[0, 1]). However, most image classification models don't have the Softmax activation by default. Therefore, we need to append Softmax to the model manually.

```python
# Define the model since the model's output needs to be softmaxed
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.sotfmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.sotfmax(x)
        return x
```

* The ONNX IO names must adhere to the convention that was detailed in the requirements section. For image classification models, the input and output names must be "image-" and "scores-...", as shown in the Python code below:

```python
input_names = ['image-']
output_names = [f'scores-{concatenated_classes}']
```

* We export the model to ONNX using PyTorch's onnx exporter.\
  Please note that the Operator Set version is set to 12 (which less than 18, the upper bound).

```python
torch.onnx.export(model, dummy_input, output_onnx_path,
                  input_names=input_names, output_names=output_names,
                  opset_version=onnx_opset_version)
```

* Setting the model normalization values can be done in two different ways. The first option is specifying those values in Nx cloud when uploading the ONNX model.\
  The second option is to carve the ONNX with those values by setting them as JSON in the ONNX `doc_string` attribute as shown below:

```python
# Update the ONNX description
graph = so.graph_from_file(output_onnx_path)
# Add the model means and standard deviations to the ONNX graph description,
# because that's used by the toolchain to populate some settings.
graph.doc_string = json.dumps({'means': model_means, 'vars': model_stds})
so.graph_to_file(graph, output_onnx_path, onnx_opset_version=onnx_opset_version)
```

* Optimizing the model is an optional step as it's always done in the Nx cloud during the conversion. But, it's useful to run locally as a validation step to check whether the ONNX model conforms to the ONNX specification:

```python
# Simplify the ONNX model
# This step is optional, but it is recommended to reduce the size of the model
# optimize the model for inference
try:
    model = onnx.load(output_onnx_path)
    model, check = simplify(model, check_n=10)
    assert check, "Couldn't simplify the ONNX model"
    onnx.save_model(model, output_onnx_path)
except Exception as e:
    print(f'Simplification failed: {e}')
    exit(1)
```

### Beyond this example

This example is a starting point for exporting image classification models to ONNX. Yet, this approach is valid for any image classification model implemented in PyTorch, given that the model has one input (image input) and one output (representing the probabilities).

To adapt this example to your own model, you need to:

* update these variables in the Python script: `classes`, `input_width`, `input_height`, `model_means`, and `model_stds`.&#x20;
* update the `Model` class to load your own model.
