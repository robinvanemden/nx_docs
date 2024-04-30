# From Scikit-learn

## About Scikit-learn

![](../../.gitbook/assets/sklear.png)

[**scikit-learn**](https://scikit-learn.org/stable/) is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license. The project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed.

Scikit-learn contains a collection of classic ML models that can be used to solve various practical tasks: classification, regression, data points clustering, etc. using for instance SVM, logistic regression, decision tree, isolation forest, etc.

## Model deployment from Scikit-learn

Model deployment from scikit-learn is simple to achieve by exporting your sklearn model to ONNX using the [skl2onnx](https://onnx.ai/sklearn-onnx/) package.

After obtaining a [clean ONNX graph that adheres to our requirements](../onnx-requirements.md), you can upload it to the [Nx AI cloud for deployment](../../nx-ai-cloud/upload-your-model/).
