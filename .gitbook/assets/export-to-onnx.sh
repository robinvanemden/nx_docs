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
  "$folder_path"