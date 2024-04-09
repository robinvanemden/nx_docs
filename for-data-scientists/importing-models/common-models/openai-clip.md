# OpenAI CLIP

> [CLIP](https://openai.com/research/clip) (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.

OpenAI's CLIP model provides a versatile foundation for constructing image classifiers without necessitating the training of a new classifier. Instead, classifiers can be created simply by specifying the classe names.

This guide shows the process of preparing the CLIP model for deployment with a defined set of classes. For the sake of example, we'll use `RN50` model version of CLIP.

### Requirements

Make sure to install the required packages:

```sh
pip install -r requirements.txt
```

<details>

<summary>requirements.txt</summary>

```
git+https://github.com/openai/CLIP.git
torch
onnxruntime
onnxsim
sclblonnx
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

import clip
import onnx
import torch
from PIL import Image
from onnxsim import simplify
from torch import nn

input_width = 224
input_height = 224
model_means = [0.48145466, 0.4578275, 0.40821073]
model_means = [round(255 * m, 2) for m in model_means]  # Convert to 0-255 range
model_stds = [0.26862954, 0.26130258, 0.27577711]
model_stds = [round(255 * s, 2) for s in model_stds]  # Convert to 0-255 range
model_name = "RN50" # TODO: Replace with the desired clip model version
text_classes = ["a person", "a car", "a dog", "a cat"] # TODO: Replace with the desired class names
class_names = ";".join([f'{i}:{c}' for i, c in enumerate(text_classes)])
scores_output_name = f'scores-{class_names}'
opset_version = 12
onnx_path = "clip_visual.onnx"

class ClipTextualModel(nn.Module):
    """Copied from https://github.com/Lednik7/CLIP-ONNX"""

    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.token_embedding = model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # needs .float() before .argmax(  ) to work
        x = x[torch.arange(x.shape[0]), text.float().argmax(dim=-1)] @ self.text_projection

        return x


class ClipVisionModel(nn.Module):
    """Adapted from https://github.com/Lednik7/CLIP-ONNX"""

    def __init__(self, model, text_classes):
        super(ClipVisionModel, self).__init__()

        self.logit_scale = model.logit_scale.exp().detach()
        text_model = ClipTextualModel(model)
        self.model = model.visual
        self.model.eval()

        self.text_features = text_model(clip.tokenize(text_classes).cpu())
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        self.text_features = self.text_features.detach()

    def forward(self, x):
        image_features = self.model(x)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * (image_features @ self.text_features.t())
        probs = logits_per_image.softmax(dim=-1)
        return probs


# onnx cannot work with cuda
model, preprocess = clip.load(model_name, device="cpu", jit=False)

# Create the model
clip_model = ClipVisionModel(model, text_classes)

# batch first
image = preprocess(Image.open("person.jpg")).unsqueeze(0).cpu()  # [1, 3, 224, 224]

# Run the model
scores = clip_model(image)
print('PyTorch outputs:', scores)

# Export to ONNX
torch.onnx.export(clip_model, image, onnx_path, opset_version=opset_version,
                  input_names=["image-"], output_names=[scores_output_name])

# Update the ONNX description
import sclblonnx as so

graph = so.graph_from_file(onnx_path)
# Add the model means and standard deviations to the ONNX graph description,
# because that's used by the toolchain to populate some settings.
graph.doc_string = json.dumps({'means': model_means, 'vars': model_stds})
so.graph_to_file(graph, onnx_path, onnx_opset_version=opset_version)

# Simplify the ONNX model
# This step is optional, but it is recommended to reduce the size of the model
# optimize the model for inference
try:
    model = onnx.load(onnx_path)
    model, check = simplify(model, check_n=3)
    assert check, "Couldn't simplify the ONNX model"
    onnx.save_model(model, onnx_path)
except Exception as e:
    print(f'Simplification failed: {e}')
    exit(1)

# Load the ONNX model
import onnxruntime

ort_session = onnxruntime.InferenceSession("clip_visual.onnx")

# Run the model
ort_inputs = {ort_session.get_inputs()[0].name: image.detach().numpy()}
ort_outs = ort_session.run(None, ort_inputs)
print('ONNX Runtime outputs:', ort_outs)
```

</details>

The exported ONNX can be uploaded on the platform and used for inference.

### Beyond this example

This example is a starting point for exporting image classification models based on CLIP to ONNX. And you can adapt it to in the following ways:

* Use a different version of CLIP, by change the value of "model\_name" in the Python file.
* Change the class names by updating the value of "text\_classes" in the Python file.
