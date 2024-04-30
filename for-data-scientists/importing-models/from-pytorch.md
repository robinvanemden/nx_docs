# From PyTorch

## About PyTorch

![](../../.gitbook/assets/Unknown-1.png)

[PyTorch](https://pytorch.org) is a [machine learning](https://en.wikipedia.org/wiki/Machine\_learning) [framework](https://en.wikipedia.org/wiki/Software\_framework) based on the [Torch](https://en.wikipedia.org/wiki/Torch\_\(machine\_learning\)) library, used for applications such as [computer vision](https://en.wikipedia.org/wiki/Computer\_vision) and [natural language processing](https://en.wikipedia.org/wiki/Natural\_language\_processing), originally developed by [Meta AI ](https://en.wikipedia.org/wiki/Meta\_AI)and now part of the [Linux Foundation](https://en.wikipedia.org/wiki/Linux\_Foundation) umbrella. It is [free and open-source software](https://en.wikipedia.org/wiki/Free\_and\_open-source\_software) released under the [modified BSD license](https://en.wikipedia.org/wiki/Modified\_BSD\_license). Although the [Python](https://en.wikipedia.org/wiki/Python\_\(programming\_language\)) interface is more polished and the primary focus of development, PyTorch also has a [C++](https://en.wikipedia.org/wiki/C%2B%2B) interface.

## Model deployment from PyTorch

Model deployment from PyTorch is simple to achieve by exporting your PyTorch model to ONNX and subsequently using (if neccesary) the `sclblonnx` package to [clean and check the resulting graph](../onnx-requirements.md#automatic-checking-using-the-sclblonnx-check-function) for an upload to the Nx AI cloud.&#x20;

* You can find details on PyTorch to ONNX exports [here](https://pytorch.org/tutorials/advanced/super\_resolution\_with\_onnxruntime.html). You can find an insightful tutorial [here](https://deci.ai/blog/how-to-convert-a-pytorch-model-to-onnx/).
* You can find an example using PyTorch and sclblonnx [here](https://github.com/scailable/sclblonnx/blob/master/examples/example\_03.py).

After obtaining a [clean ONNX graph that adheres to our requirements](../onnx-requirements.md), you can [upload it to the Nx AI cloud](../../nx-ai-cloud/upload-your-model/) for deployment.
