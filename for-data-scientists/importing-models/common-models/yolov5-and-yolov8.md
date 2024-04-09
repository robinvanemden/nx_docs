# Yolov5 and Yolov8

This guide shows how to export a **trained** model based on **Ultralytics** to ONNX that can be directly uploaded and deployed on servers.

We'll be using the YOLOv8n as an example to show how each Ultralytics models can be exported and trasformed into an ONNX that's compatible with the Nx AI Manager.

### Requirements

Make sure to install the required packages:

```bash
pip install -r requirements.txt
```

<details>

<summary>requirements.txt</summary>

```
onnx==1.13.1
onnx-simplifier==0.4.35
onnxoptimizer==0.3.13
onnxruntime==1.17.0
sclblonnx==0.2.1
ultralytics==8.1.14
```

</details>

### Exporting the model to ONNX

The ONNX model exported directly using Ultralytics CLI doesn't come with the necessary post-processing steps (namely, Non-Maximum Suppression). Therefore, we need to add that post-processing to the ONNX model using another script.

1. So, first we export the model to ONNX using the following command:

```bash
bash export-to-onnx.sh yolov8n yolov8n.onnx 640
```

<details>

<summary>export-to-onnx.sh</summary>

```bash
# This script is used to convert the YOLOv3/5/8 model to ONNX format opset version 12
set -e

# Check if the user has provided the model name
if [ -z "$3" ]
then
    echo "Usage: $0 <model_name> <onnx_path> <img_size>"
    exit 1
fi

model_name=$1 # exampples: yolov5su, yolov8n, yolov8s, ../custom-model.pt
onnx_path=$2 # example: ../yolov8n.onnx
img_size=$3 # example: 640

# additional arguments: half, int8, optimize
yolo export model="$model_name" format=onnx imgsz="$img_size" opset=12 simplify=true

# Move the file to the specified path
mv "$model_name.onnx" "$onnx_path" || echo "Failed to move the file to '$onnx_path'"
```

</details>

2. Then, we add the post-processing steps to the ONNX model using the following command:

```bash
python complete_onnx.py 
```

<details>

<summary>complete_onnx.py</summary>

```python
from os.path import dirname, abspath, join, splitext

from utils import add_pre_post_processing_to_onnx, simplify_onnx

from glob import glob

HERE = dirname(abspath(__file__))

if __name__ == '__main__':
    onnx_path = glob(join(HERE, '*.onnx'))[0]
    output_onnx_path = splitext(onnx_path)[0] + '-complete.onnx'
    classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    add_pre_post_processing_to_onnx(onnx_path, output_onnx_path, classes)
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
from typing import List

def add_pre_post_processing_to_onnx(onnx_path: str, output_onnx_path: str, classes: List[str]):
    base_graph = so.graph_from_file(onnx_path)
    output_name = base_graph.output[0].name
    input_name = base_graph.input[0].name

    # get input shape
    input_shape = base_graph.input[0].type.tensor_type.shape.dim
    input_shape = [d.dim_value for d in input_shape]
    if input_shape[1] <= 3:  # NCHW
        width, height = input_shape[2], input_shape[3]
    else:  # NHWC
        width, height = input_shape[1], input_shape[2]

    # cleanup useless IO
    so.delete_output(base_graph, output_name)
    so.delete_input(base_graph, input_name)

    # Normalize the input by dividing by 255
    so.add_constant(base_graph, 'c_255', np.array([255], dtype=np.float32), 'FLOAT')
    div = so.node('Div', inputs=['image-', 'c_255'], outputs=[input_name])
    base_graph.node.insert(0, div)
    so.add_input(base_graph, name='image-', dimensions=input_shape, data_type='FLOAT')

    # move constant nodes to the beginning of the graph
    constant_nodes = [n for n in base_graph.node if n.op_type == 'Constant']
    for n in constant_nodes:
        base_graph.node.remove(n)
        base_graph.node.insert(0, n)

    # Add NMS to the model
    make_yolov5_complementary_graph(base_graph, output_name)

    # Add mask to the model
    so.delete_output(base_graph, 'bboxes-')
    mask_bboxes(base_graph, 'bboxes-', 'mask-', width, height)
    so.add_output(base_graph, 'unmasked_bboxes', 'FLOAT', dimensions=[20, 6])

    # Save the model
    so.graph_to_file(base_graph, output_onnx_path, onnx_opset_version=get_onnx_opset_version(onnx_path))

    # Rename model IO
    classes_str = ';'.join([f'{i}:{c}' for i, c in enumerate(classes)])
    rename_io(output_onnx_path, output_onnx_path, **{'image': 'image-',
                                                     'unmasked_bboxes': f'bboxes-format:xyxysc;{classes_str}'
                                                     })

    update_onnx_doc_string(output_onnx_path, [0, 0, 0], [1, 1, 1])


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


def make_yolov5_complementary_graph(g, output_name):
    so.add_input(g, name='nms_sensitivity-', dimensions=[1], data_type='FLOAT')

    # constants
    so.add_constant(g, name='C0', value=np.array(0), data_type='INT64')
    so.add_constant(g, name='C1', value=np.array(1), data_type='INT64')
    so.add_constant(g, name='C2', value=np.array(2), data_type='INT64')

    so.add_constant(g, name='c0', value=np.array([0]), data_type='INT64')
    so.add_constant(g, name='c1', value=np.array([1]), data_type='INT64')
    so.add_constant(g, name='c2', value=np.array([2]), data_type='INT64')
    so.add_constant(g, name='c4', value=np.array([4]), data_type='INT64')
    so.add_constant(g, name='c5', value=np.array([5]), data_type='INT64')
    so.add_constant(g, name='c_-1-111', value=np.array([-1 / 2, -1 / 2, 1 / 2, 1 / 2]), data_type='FLOAT')
    so.add_constant(g, name='c_2323', value=np.array([2, 3, 2, 3]), data_type='INT64')
    so.add_constant(g, name='c_0101', value=np.array([0, 1, 0, 1]), data_type='INT64')
    so.add_constant(g, name='c_2048', value=np.array([2048, 2048, 0, 0]).astype('float32'), data_type='FLOAT')
    so.add_constant(g, name='c_end', value=np.array([10000]), data_type='INT64')
    so.add_constant(g, name='c_20', value=np.array([20]), data_type='INT64')
    so.add_constant(g, name='c_0.35', value=np.array([0.35]), data_type='FLOAT')

    # nodes
    transpose_output = so.node('Transpose', inputs=[output_name], outputs=['transposed_output'], perm=(0, 2, 1))

    slice_x_4_end = so.node('Slice', inputs=['transposed_output', 'c4', 'c_end', 'c2', 'c1'], outputs=['all_scores'])
    transposed_all_scores = so.node('Transpose', inputs=['all_scores'], outputs=['transposed_all_scores'],
                                    perm=(0, 2, 1))  # [1, C, N]

    _classes = so.node('ArgMax', inputs=['all_scores'], outputs=['_classes'], axis=2, keepdims=True)  # [1, N, 1]
    _classes_float = so.node('Cast', inputs=['_classes'], outputs=['_classes_float'], to=1)  # [1, N, 1]
    _scores = so.node('ReduceMax', inputs=['all_scores'], outputs=['_scores'], axes=(2,), keepdims=True)  # [1, N, 1]

    _boxes = so.node('Slice', inputs=['transposed_output', 'c0', 'c4', 'c2', 'c1'], outputs=['_boxes'])  # [1, N, 4]

    offset = so.node('Mul', inputs=['_classes_float', 'c_2048'], outputs=['offset'])  # [1, N, 1]
    shifted_boxes = so.node('Add', inputs=['_boxes', 'offset'], outputs=['shifted_boxes'])  # [1, N, 4]

    nms_indices = so.node('NonMaxSuppression', inputs=['shifted_boxes',
                                                       'transposed_all_scores',
                                                       'c_20',
                                                       'c_0.35',
                                                       'nms_sensitivity-'],
                          outputs=['nms_indices'],
                          center_point_box=1)

    boxes_indices = so.node('Gather', inputs=['nms_indices', 'C2'], outputs=['boxes_indices'], axis=1)

    _classes2 = so.node('Gather', inputs=['nms_indices', 'C1'], outputs=['_classes2'], axis=1)  # [M]
    _classes2_float = so.node('Cast', inputs=['_classes2'], outputs=['_classes2_float'], to=1)  # [M]
    classes = so.node('Unsqueeze', inputs=['_classes2_float'], outputs=['classes'], axes=(0, 2))  # [1, M, 1]
    scores = so.node('Gather', inputs=['_scores', 'boxes_indices'], outputs=['scores'], axis=1)  # [1, M, 1]
    boxes = so.node('Gather', inputs=['_boxes', 'boxes_indices'], outputs=['boxes'], axis=1)  # [1, M, 4]

    boxes_0101 = so.node('Gather', inputs=['boxes', 'c_0101'], outputs=['boxes_0101'], axis=2)
    boxes_2323 = so.node('Gather', inputs=['boxes', 'c_2323'], outputs=['boxes_2323'], axis=2)
    something = so.node('Mul', inputs=['c_-1-111', 'boxes_2323'], outputs=['something'])
    xyxy_boxes = so.node('Add', inputs=['boxes_0101', 'something'], outputs=['xyxy_boxes'])

    _bboxes = so.node('Concat', inputs=['xyxy_boxes', 'scores', 'classes'], outputs=['_bboxes'], axis=2)
    bboxes = so.node('Squeeze', inputs=['_bboxes'], outputs=['bboxes-'], axes=(0,))

    so.add_nodes(g,
                 [transpose_output, slice_x_4_end, transposed_all_scores,
                  _classes, _classes_float, _scores, _boxes,
                  offset, shifted_boxes,
                  nms_indices, boxes_indices, _classes2, _classes2_float,
                  classes, scores, boxes,
                  boxes_0101, boxes_2323, something, xyxy_boxes,
                  _bboxes, bboxes
                  ])

    so.add_output(g, 'bboxes-', 'FLOAT', dimensions=[20, 6])

    return g


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
        model, check = simplify(onnx_path)
        assert check, 'Failed to simplify ONNX model'
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        raise Exception('Failed to simplify ONNX model')

    save(model, output_onnx_path)


def update_onnx_doc_string(onnx_path: str, model_means: List[float], model_stds: List[float]):
    # Update the ONNX description
    graph = so.graph_from_file(onnx_path)
    # Add the model means and standard deviations to the ONNX graph description,
    # because that's used by the toolchain to populate some settings.
    graph.doc_string = json.dumps({'means': model_means, 'vars': model_stds})
    so.graph_to_file(graph, onnx_path, onnx_opset_version=12)

```

</details>

3. Finally, to test the ONNX model, we can use the following command:

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
        channels, height, width = input_shape[1], input_shape[2], input_shape[3]
    else:  # nhwc
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)  # add channel dimension
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.transpose(img, (2, 0, 1)).astype('float32')
    img = np.expand_dims(img, axis=0)

    mask_area = np.repeat(1, width * height).astype('bool')
    mask_area = mask_area.reshape((height, width))
    mask_area[:, :width // 2] = 0  # mask the left part of the image

    bboxes = sess.run(None, {
        input_name1: img,
        input_name2: np.array([0.15]).astype('float32'),
        input_name3: mask_area
    })[0]

    return bboxes, (width, height)


def visualize_bboxes(bboxes, img_path, width, height):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    for bbox in bboxes:
        x1, y1, x2, y2, score, class_id = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{int(class_id)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    from glob import glob

    model_path = glob(join(PATH, '*-complete.onnx'))[0]
    img_path = join(PATH, 'pedestrians.jpg')
    bboxes, (width, height) = test_model(model_path, img_path)
    visualize_bboxes(bboxes, img_path, width, height)

```

</details>

The exported ONNX can be uploaded on the platform and used for inference directly.

### Beyond this example

This example is just a starting point for exporting Ultralytics models to ONNX. Therefore any model based on Yolov5 or Yolov8 can be easily deployed by slightly adapting the ONNX model.

To adapt this example to your own model, you need to:

* Call `bash export-to-onnx.sh` with the appropriate arguments (in particular the first and last arguments: model name, and input size).
* Update the `complete_onnx.py` script to add the classes your model is trained on.