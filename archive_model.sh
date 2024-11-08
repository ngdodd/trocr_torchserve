#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <model-name> <version>"
  read -p "Press Enter to exit..."
  exit 1
fi

# Assign input arguments to variables
MODEL_NAME="$1"
VERSION="$2"

MODEL_STORE="model_store"

echo "Finding relevant files..."

# model.safetensors is the serialized model file we will use to serve the model
MODEL_FILE=$(python -c "
import sys
sys.path.append('.')
from pathlib import Path
from utils import MODEL_PATH
model_path = Path(MODEL_PATH) / 'model.safetensors'
print(model_path)
")

# Check if model.safetensors was found
if [ $? -ne 0 ]; then
  echo "Error: model.safetensors file not found."
  exit 1
fi

# Get the CSV string of all files in MODEL_PATH (including in any child directories)
EXTRA_FILES=$(python -c "
import sys
sys.path.append('.')
from pathlib import Path
from utils import MODEL_PATH
files = [str(path) for path in Path(MODEL_PATH).rglob('*') if path.is_file() and path.name != 'model.safetensors']
print(','.join(files))
")

echo "Archiving model..."
torch-model-archiver --model-name "$MODEL_NAME" --version "$VERSION" --serialized-file "$MODEL_FILE" --handler handler.py --extra-files "$EXTRA_FILES" --export-path "$MODEL_STORE"

echo "Model archive saved to $MODEL_STORE/trocr_base.mar"
read -p "Press Enter to exit..."