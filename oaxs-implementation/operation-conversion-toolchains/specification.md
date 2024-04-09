# Specification

As discussed before, the conversion toolchain takes a trained AI model and returns one or more model files (or return nothing if that the input model is not supported).

The conversion step is usually complicated and requires many dependencies, thus, the recommended way is host it inside a Docker container with an entrypoint, which is invoked to execute the conversion process on an input model. To standardize the process, the entrypoint should expect  a couple of arguments: the path to the input model and path of a directory where the toolchain will save its artifacts.

### Toolchain artifacts

* **Binary files**\
  The model files generated by the toolchain must have a media type (formerly known as a MIME type), for example: `application/x-onnx`, `application/x-tensorflow-lite`, `application/zip`, etc.
* **Log messages** \
  A JSON file containing information about the conversion process. It should be structured like this: \
  \[{"Message": \<message>, "Data": {"\<field1>": \<value1>, ...\}}]. \
  For example:

```json
[
  {
    "Message": "Pulled model file",
    "Data": {
      "fileHash": "5a79643c223584f5796cd40356dbddd2"
    }
  },
  {
    "Message": "Simplifying ONNX model",
    "Data": {
      "error": "Failed to simplify ONNX model: The shape of input \"image-\" has dynamic size, please set an input shape manually with --test-input-shape",
      "solution": "Fallback to original model"
    }
  },
  {
    "Message": "Sending response",
    "Data": {
      "mimetype": "application/x-onnx",
      "fileHash": "5a79643c223584f5796cd40356dbddd2",
      "IOHash": "20b9e0bc3c0b9952673c28e17f31d9ca"
    }
  },
  {
    "Message": "Conversion process completed successfully"
  }
]
```

### Recommendations

* In the event of a failed or declined conversion, it is recommended, for user experience (UX) purposes, to provide a human-readable message in the logs. This message should clearly explain the reason for the conversion failure, aiding users or developers in understanding the issue and facilitating its resolution.