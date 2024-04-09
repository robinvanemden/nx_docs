# YoloS

> The YOLOS model was proposed in [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/abs/2106.00666) by Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu. YOLOS proposes to just leverage the plain [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/en/model\_doc/vit) for object detection, inspired by DETR. It turns out that a base-sized encoder-only Transformer can also achieve 42 AP on COCO, similar to DETR and much more complex frameworks such as Faster R-CNN.

This guide shows how to export a trained YoloS model to an ONNX conforming to the requirements mentioned [before](../../onnx-requirements.md).

### Requirements

Make sure to install the required packages:

```sh
pip install -r requirements.txt
```

<details>

<summary>requirements.txt</summary>

```
optimum[exporters]
accelerate
onnxruntime
onnxsim
```

</details>

### Exporting the model to ONNX

1. The following command exports a trained YoloS model to ONNX and saves it locally:

```sh
bash export-to-onnx.sh hustvl/yolos-tiny yolos-tiny
```

<details>

<summary>export-to-onnx.sh</summary>

```sh
# This script is used to convert HuggingFace models to ONNX format opset version 12
set -e

# Check if the user has provided the model name
if [ -z "$2" ]
then
    echo "Usage: $0 <model_name> <folder_path>"
    exit 1
fi

model_name=$1 # exampples: hustvl/yolos-tiny
folder_path=$2 # example: yolos-model

optimum-cli export onnx --model "$model_name" \
  --opset 12 \
  --framework pt \
  --batch_size 1 \
  --atol 0.001 \
  "$folder_path"
```

</details>

2. The exported ONNX needs a little bit of fiddling to make usable with the Nx AI Manager.

<pre class="language-sh"><code class="lang-sh"><strong>python complete_onnx.py
</strong></code></pre>

<details>

<summary>complete_onnx.py</summary>

```python
from os.path import dirname, abspath, join, splitext
from glob import glob
from utils import add_post_processing_to_onnx, simplify_onnx, set_input_shape, read_configs, update_onnx_doc_string

HERE = dirname(abspath(__file__))

if __name__ == '__main__':
    model_folder_path = join(HERE, 'yolos-tiny')

    onnx_path = glob(join(model_folder_path, '*.onnx'))[0]
    output_onnx_path = splitext(onnx_path)[0] + '-complete.onnx'

    metadata = read_configs(model_folder_path)
    image_size = metadata['image_size']
    if image_size is None:
        image_size = [416, 416]

    # Update input shape to static shape
    set_input_shape(onnx_path, output_onnx_path, (1, 3, image_size[0], image_size[1]))

    # Update ONNX doc string
    update_onnx_doc_string(output_onnx_path, metadata['means'], metadata['stds'])

    # Simplify  ONNX model
    simplify_onnx(output_onnx_path, output_onnx_path)

    add_post_processing_to_onnx(output_onnx_path, output_onnx_path, id2labels=metadata['id2label'],
                                image_size=tuple(image_size))

    # Simplify ONNX model again
    simplify_onnx(output_onnx_path, output_onnx_path)

```

</details>

<details>

<summary>utils.py</summary>

```python
import json
import re

import numpy as np
import onnx
import sclblonnx as so
from onnx import save
from onnxsim import simplify


def add_post_processing_to_onnx(onnx_path: str, output_onnx_path: str, id2labels: dict[str, str],
                                image_size: tuple[int, int]):
    base_graph = so.graph_from_file(onnx_path)
    logits_name = base_graph.output[0].name
    boxes_name = base_graph.output[1].name
    input_name = base_graph.input[0].name

    transformation_matrix = np.array([[[1, 0, 1, 0], [0, 1, 0, 1], [-1/2, 0, 1/2, 0], [0, -1/2, 0, 1/2]]],
                                     dtype=np.float32)
    so.add_constant(base_graph, 'transformation_matrix', transformation_matrix, 'FLOAT')

    class_ids = np.array(sorted(list(id2labels.keys())), dtype=np.int64)
    so.add_constant(base_graph, 'class_ids', class_ids, 'INT64')

    so.add_constant(base_graph, 'C1', np.array(1, dtype=np.int64), 'INT64')

    # Softmax the logits
    softmax_output = so.node('Softmax', inputs=[logits_name], outputs=['softmax_output'], axis=-1)  # [1, M, C]

    # Compute classes
    classes = so.node('ArgMax', inputs=['softmax_output'], outputs=['classes'], axis=-1, keepdims=1)  # [1, M, 1]
    float_classes = so.node('Cast', inputs=['classes'], outputs=['float_classes'], to=1)  # [1, M, 1]
    equal_classes = so.node('Equal', inputs=['classes', 'class_ids'], outputs=['equal_classes'])  # [1, M, C]
    equal_classes_int = so.node('Cast', inputs=['equal_classes'], outputs=['equal_classes_int'], to=7)  # [1, M, C]
    reducesum_classes = so.node('ReduceSum', inputs=['equal_classes_int'], outputs=['reducesum_classes'],
                                axes=(-1,), keepdims=1)  # [1, M, 1]
    mask_classes = so.node('Cast', inputs=['reducesum_classes'], outputs=['mask_classes'], to=9)  # [1, M, 1]

    # Compute scores
    scores = so.node('ReduceMax', inputs=['softmax_output'], outputs=['scores'], axes=(-1,), keepdims=1)  # [1, M, 1]
    mask_scores = so.node('Greater', inputs=['scores', 'nms_sensitivity-'], outputs=['mask_scores'])  # [1, M, 1]
    boxes_to_keep = so.node('And', inputs=['mask_classes', 'mask_scores'], outputs=['boxes_to_keep'])  # [1, M, 1]

    # box ids to keep
    box_ids = so.node('NonZero', inputs=['boxes_to_keep'], outputs=['box_ids'])  # [3, M]
    box_ids_row1 = so.node('Gather', inputs=['box_ids', 'C1'], outputs=['box_ids_row1'], axis=0)  # [M]

    # xywh to xyxy
    xyxy = so.node('MatMul', inputs=[boxes_name, 'transformation_matrix'],
                   outputs=['xyxy'])  # [1, M, 4]

    # Concat the boxes, score & classes
    bboxes = so.node('Concat', inputs=['xyxy', 'scores', 'float_classes'],
                     outputs=['bboxes'], axis=2)  # [1, M, 6]

    # Keep only the boxes to keep
    bboxes_to_keep = so.node('Gather', inputs=['bboxes', 'box_ids_row1'],
                             outputs=['bboxes_to_keep'], axis=1)  # [1, M, 6]
    bboxes_squeezed = so.node('Squeeze', inputs=['bboxes_to_keep'], outputs=['bboxes_squeezed'],
                              axes=(0,))  # [M, 6]

    so.add_nodes(base_graph,
                 [softmax_output, classes, float_classes, equal_classes, equal_classes_int, reducesum_classes,
                  mask_classes,
                  scores, mask_scores, boxes_to_keep, box_ids, box_ids_row1, xyxy, bboxes, bboxes_to_keep,
                  bboxes_squeezed])

    # Add mask to the model
    so.delete_output(base_graph, boxes_name)
    so.delete_output(base_graph, logits_name)

    mask_bboxes(base_graph, 'bboxes_squeezed', 'mask-', image_size[1], image_size[0])
    so.add_output(base_graph, 'unmasked_bboxes', 'FLOAT', dimensions=[20, 6])

    so.add_input(base_graph, name='nms_sensitivity-', dimensions=[1], data_type='FLOAT')

    # Save the model
    so.graph_to_file(base_graph, output_onnx_path, onnx_opset_version=get_onnx_opset_version(onnx_path))

    # Rename model IO
    classes_str = ';'.join([f'{k}:{v}' for k, v in id2labels.items()])
    rename_io(output_onnx_path, output_onnx_path, **{input_name: 'image-',
                                                     'unmasked_bboxes': f'bboxes-format:xyxysc;{classes_str}'
                                                     })


def mask_bboxes(graph, bboxes_name, mask_name, w, h):
    so.add_input(graph, name=mask_name, dimensions=[h, w], data_type='BOOL')

    so.add_constant(graph, 'index_one_three', np.array([0, 2]), 'INT64')
    so.add_constant(graph, 'index_four', np.array([3, 3]), 'INT64')
    so.add_constant(graph, 'hw_clip_min', np.array(0), 'FLOAT')
    so.add_constant(graph, 'w_clip_max', np.array(w - 1), 'FLOAT')
    so.add_constant(graph, 'h_clip_max', np.array(h - 1), 'FLOAT')

    x_coordinates = so.node('Gather', inputs=[bboxes_name, 'index_one_three'], outputs=['x_coordinates'], axis=1)
    y_coordinates = so.node('Gather', inputs=[bboxes_name, 'index_four'], outputs=['y_coordinates'], axis=1)
    x_reducemean = so.node('ReduceMean', inputs=['x_coordinates'], outputs=['x_reducemean'], axes=(1,), keepdims=1)
    y_coordinate = so.node('ReduceMean', inputs=['y_coordinates'], outputs=['y_coordinate'], axes=(1,), keepdims=1)
    x_clipped = so.node('Clip', inputs=['x_reducemean', 'hw_clip_min', 'w_clip_max'], outputs=['x_clipped'])
    y_clipped = so.node('Clip', inputs=['y_coordinate', 'hw_clip_min', 'h_clip_max'], outputs=['y_clipped'])

    bottom_center_corner = so.node('Concat', inputs=['y_clipped', 'x_clipped'], outputs=['bottom_center_corner'],
                                   axis=1)
    bottom_center_corner_int = so.node('Cast', inputs=['bottom_center_corner'], outputs=['bottom_center_corner_int'],
                                       to=7)
    bboxes_mask1 = so.node('GatherND', inputs=[mask_name, 'bottom_center_corner_int'], outputs=['bboxes_mask1'])
    bboxes_indices1 = so.node('NonZero', inputs=['bboxes_mask1'], outputs=['bboxes_indices1'])
    bboxes_indices1_squeezed = so.node('Squeeze', inputs=['bboxes_indices1'], outputs=['bboxes_indices1_squeezed'],
                                       axes=(0,))
    new_bboxes = so.node('Gather', inputs=[bboxes_name, 'bboxes_indices1_squeezed'], outputs=['unmasked_bboxes'],
                         axis=0)

    so.add_nodes(graph, [x_coordinates, y_coordinates, x_reducemean, y_coordinate, y_clipped, x_clipped,
                         bottom_center_corner,
                         bottom_center_corner_int,
                         bboxes_mask1, bboxes_indices1,
                         bboxes_indices1_squeezed, new_bboxes])
    return graph


def rename_io(model_path, new_model_path=None, **io_names):
    if new_model_path is None:
        new_model_path = model_path

    g = so.graph_from_file(model_path)

    def log(old: bool = True):
        s = 'Old' if old else 'New'
        assert so.list_inputs(g)
        assert so.list_outputs(g)

    if io_names == {}:
        return

    inputs = [i.name for i in g.input]
    outputs = [i.name for i in g.output]

    for k, v in io_names.items():
        pattern = re.compile(k)
        renamed = False

        for i in inputs:
            if pattern.match(i):
                renamed = True
                so.rename_input(g, i, v)
                break

        if not renamed:
            for o in outputs:
                if pattern.match(o):
                    renamed = True
                    so.rename_output(g, o, v)
                    break

        if not renamed:
            continue

    log(False)

    so.graph_to_file(g, new_model_path, onnx_opset_version=get_onnx_opset_version(model_path))


def get_onnx_opset_version(onnx_path):
    model = onnx.load(onnx_path)
    opset_version = model.opset_import[0].version if len(model.opset_import) > 0 else 0
    return opset_version


def simplify_onnx(onnx_path: str, output_onnx_path: str):
    try:
        model, check = simplify(onnx_path, check_n=1)
        assert check, 'Failed to simplify ONNX model'
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        raise Exception('Failed to simplify ONNX model')

    save(model, output_onnx_path)


def update_onnx_doc_string(onnx_path: str, model_means: list[float], model_stds: list[float]):
    # Update the ONNX description
    graph = so.graph_from_file(onnx_path)
    # Add the model means and standard deviations to the ONNX graph description,
    # because that's used by the toolchain to populate some settings.
    graph.doc_string = json.dumps({'means': model_means, 'vars': model_stds})
    so.graph_to_file(graph, onnx_path, onnx_opset_version=12)


def set_input_shape(input_onnx: str, output_onnx: str, new_shape: tuple[int, int, int, int]):
    graph = so.graph_from_file(input_onnx)
    input_shape = graph.input[0].type.tensor_type.shape.dim
    for i, d in enumerate(input_shape):
        d.dim_value = new_shape[i]

    so.graph_to_file(graph, output_onnx, onnx_opset_version=12)


def read_configs(model_folder_path: str):
    from os.path import join
    import json

    preproccessing_config_path = join(model_folder_path, 'preprocessor_config.json')
    config_path = join(model_folder_path, 'config.json')

    with open(preproccessing_config_path, 'r') as f:
        preprocessor_config = json.load(f)

    with open(config_path, 'r') as f:
        config = json.load(f)

    metadata = {'means': None, 'stds': None, 'id2label': None, 'image_size': None}

    # Get model means and stds
    reescale_factor = preprocessor_config.get('rescale_factor', 1)
    do_rescale = preprocessor_config.get('do_rescale', False)
    image_mean = preprocessor_config.get('image_mean', [0, 0, 0])
    image_std = preprocessor_config.get('image_std', [1, 1, 1])
    if do_rescale:
        metadata['means'] = [round(m / reescale_factor, 2) for m in image_mean]
        metadata['stds'] = [round(s / reescale_factor, 2) for s in image_std]

    # Get model image size
    metadata['image_size'] = config.get('image_size', None)

    # Get model id2label
    id2label = config.get('id2label', None)
    id2label = {k: v for k, v in id2label.items() if v != 'N/A'}  # Eliminate N/A labels
    metadata['id2label'] = id2label

    return metadata

```

</details>

3. Finally, here's a Python script that can be used in order to test the ONNX file using ONNXRuntime.

```bash
python test_onnx.py
```

<details>

<summary>test_onnx.py</summary>

```python
from os.path import join, dirname, abspath

import cv2
import numpy as np
import onnxruntime as rt

PATH = dirname(abspath(__file__))


def test_model(model_path, img_path):
    sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # get input name
    input_name1 = sess.get_inputs()[0].name
    input_name2 = sess.get_inputs()[1].name
    input_name3 = sess.get_inputs()[2].name

    # get input dimensions
    input_shape = sess.get_inputs()[0].shape
    if input_shape[1] <= 3:  # nchw
        height, width = input_shape[2], input_shape[3]
    else:  # nhwc
        height, width = input_shape[1], input_shape[2]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # nwhc nchw
    img = (img.astype('float32') - np.array([123.67, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
    img = np.transpose(img, (2, 0, 1)).astype('float32')
    img = np.expand_dims(img, axis=0)

    mask_area = np.repeat(1, width * height).astype('bool')
    mask_area = mask_area.reshape((height, width))
    mask_area[:, :width // 2] = 1  # mask the left part of the image

    bboxes = sess.run(None, {
        input_name1: img,
        input_name2: mask_area,
        input_name3: np.array([0.9]).astype('float32')
    })
    bboxes = bboxes[-1]

    return bboxes, (width, height)


def visualize_bboxes(bboxes, img_path, width, height):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    if bboxes[:, :4].max() <= 1.5:
        bboxes[:, :4] = bboxes[:, :4] * [width, height, width, height]
    for bbox in bboxes:
        x1, y1, x2, y2, score, class_id = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{int(class_id)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    from glob import glob

    model_path = glob(join(PATH, '**', '*-complete.onnx'), recursive=True)[0]
    img_path = join(PATH, 'pedestrians.jpg')
    bboxes, (width, height) = test_model(model_path, img_path)
    visualize_bboxes(bboxes, img_path, width, height)

```

</details>

### Beyond this example

This example is a starting point for exporting YoloS-tiny model to ONNX. Yet, this approach is valid for any YoloS model trained on any object detection dataset.

To adapt this example to your own model, you need to:

* update the content of the "export-to-onnx.sh" and "complete\_onnx.py" scripts accordingly.
