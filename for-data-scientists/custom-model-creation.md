# Custom model creation

Here, we demonstrate the creation of _custom_ ONNX models/pipelines for use within the Nx ecosystem. To export a model from your preferred training platform, please look at the[ importing model pages](importing-models/).

{% hint style="info" %}
Note that throughout this section, we heavily rely on the open-source [sclblonnx](https://pypi.org/project/sclblonnx/) python package; you can find more examples in the [sclblonnx git repository](https://github.com/scailable/sclblonnx/tree/master/examples).
{% endhint %}

This article explains how to build ONNX graphs from scratch that encode a pipeline (i.e., an AI model including potential pre- and post-processing) that can be deployed to any edge device running the Nx AI manager.&#x20;

We cover the following steps:

1. Basic background regarding the `sclblonnx` package.
2. Using `sclblonnx` to create an ONNX graph from scratch.

{% hint style="info" %}
If you do not know the ONNX format, we encourage you to read our[ about ONNX page](about-onnx.md) at this point.
{% endhint %}

## 1. Basic background regarding the `sclblonnx` package

Because at Nx, we use ONNX often, and because our use of ONNX models/pipelines almost always extends (far) beyond simply [storing a fitted model in a single environment](importing-models/) to use it in that exact same environment later on, we often find ourselves in the situation that we would like to _create,_ _inspect_, _alter_, _test_, or _merge_ existing ONNX graphs. For example, we often add image resizing to an existing vision model such that the resulting ONNX pipeline can be put into production for cameras with different resolutions. However, in our view, the existing [onnx.helper API](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md) is challenging to use. Thus, internally, we have developed (and are continuously trying to improve) a higher-level API for the manipulation of ONNX graphs. This higher level tooling is openly available in the `sclblonnx` python package.

{% hint style="info" %}
The source for the `sclblonnx` package can be found on [git](https://github.com/scailable/sclblonnx). Easy installation of the package can be done using [pip](https://pypi.org/project/sclblonnx/).
{% endhint %}

In its bare essence, the `sclblonnx` package provides a number of high-level utility functions to deal with ONNX graphs. We try to use a consistent syntax, which looks as follows:

```
# Importing the package
import sclblonnx as so

# Assuming we have a graph object g:
g = so.FUNCTION(g, ...)
```

Thus, we provide a number of functions to operate on a graph (and often alter an existing graph), which results in an updated version of the graph. Common functions are:

* `add_node(g, node)`: Add a node to an existing graph (and yeah, obviously, you can also delete\_node(g, node)).
* `add_input(g, input)`: Add a new input to an existing graph. You can also delete or change inputs.
* `add_output(g, output)`: Add a new output to an existing graph.
* `add_constant(g, constant)`: Add a constant to a graph.
* `clean(g)`: Clean up the graph; this is important as exported graphs are often bloated or inconsistent.
* `check(g)`: Check whether the graph is valid, can be run, and can be deployed using the Nx AI Manager (the latter you can turn off)
* `display(g)`: Visually inspect the graph using Netron.
* `merge(g1, g2, outputs, inputs)`: Merge two (sub) graphs into a single graph. E.g., add preprocessing to a trained model.

{% hint style="info" %}
The `sclblonnx` git repository contains many [examples](https://github.com/scailable/sclblonnx/tree/master/examples) that should help you get started.
{% endhint %}

## 2. Using `sclblonnx` to create an ONNX graph from scratch.

Here we provide the syntax to use the `sclblonnx` package to create a super simple ONNX graph to add two scalars.

{% hint style="warning" %}
The code example is the first example of creating an ONNX graph from scratch.&#x20;

Note that the resulting graph cannot be deployed to the AI manager as it does not operate on an image input and does not adhere to our [ONNX requirements](onnx-requirements.md).&#x20;
{% endhint %}

We start by creating an empty graph:

```
# Use the empty_graph() method to create a named xpb2.GraphProto object:
g = so.empty_graph()
```

Next, we add the `Add` node to the graph (you can find the list off all possible nodes, or ONNX operators) [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md)).

```
# Add a node to the graph.
n1 = so.node('Add', inputs=['x1', 'x2'], outputs=['sum'])
g = so.add_node(g, n1)
```

By now, we have a graph with a single computational operator called Add. The inputs and output of the add operator are named, but we have not specified their types yet. This is our next step:

```
# We should explicitly specify the named inputs to the graph -- note that the names determine the graph topology.
# Also, we should specify the data type and dimensions of any input.
# Use so.list_data_types() to see available data types.
g = so.add_input(g, 'x1', "FLOAT", [1])
g = so.add_input(g, 'x2', "FLOAT", [1])

# Similarly, we add the named output with its corresponding type and dimension.
# Note that types will need to "match", as do dimensions. Please see the operator docs for more info.
g = so.add_output(g, 'sum', "FLOAT", [1])
```

By now we have effectively created a fully functioning ONNX graph: we specified all our operators and the inputs and outputs to the graph (including their types and dimensions).

Next, we provide a few options to check, clean, and inspect the resulting graph:

```
# so.check() checks the current graph to see if it matches Nx upload criteria for .wasm conversion.
so.check(g)

# Now, a few tricks to sanitize the graph which are always useful.
# so.clean() provides lossless reduction of the graph. If successful cleaned graph is returned.
g = so.clean(g)

# so.display() tries to open the graph using Netron to inspect it. This worsk on most systems if Netron is installed.
# Get Netron at https://github.com/onnx/onnx/blob/master/docs/Operators.md
so.display(g)
```

If the created graph `g` passes the `so.check()` function you can be sure your ONNX graph is proper.

> **Note:** The `_sclbl_check` argument of the `so.check()` function can be used to toggle whether or not you would like to check the graph for usage within the Nx ecosystem.

After finalizing and checking the graph, it's easy to test the resulting graph locally using the [onnx runtime](https://github.com/microsoft/onnxruntime):

```
# Now, use the default ONNX runtime to do a test run of the graph.
# Note that the inputs dimensions and types need to match the specification of the graph.
# The outputs returns all the outputs named in the list.
example = {"x1": np.array([1.2]).astype(np.float32), "x2": np.array([2.5]).astype(np.float32)}
result = so.run(g,
                inputs=example,
                outputs=["sum"]
                )
print(result)
```

Finally, a created graph can easily be stored:

```
# We can easily store the graph to a file for upload at https://admin.sclbl.nxvms.com/:
so.graph_to_file(g, "onnx/add-scalars.onnx")
```

After storing a complete graph, you can upload it to the Nx AI Cloud platform by logging into your account at [https://admin.sclbl.nxvms.com/](https://admin.sclbl.nxvms.com/) and going to the "CREATE" tab.&#x20;

{% hint style="info" %}
After creating your ONNX file, and before uploading it to the Nx platform, please check whether your ONNX file meets all the [ONNX requirements](onnx-requirements.md) imposed by the Nx platform.
{% endhint %}

{% hint style="success" %}
Note that the conversion of ONNX to SPMF is one-to-one: i.e., the output produced by the ONNX graph will exactly match the output produced by the AI manager when a model is deployed to any supported edge device.
{% endhint %}
