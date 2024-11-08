#!/bin/bash

# Define the directory to store the model
MODEL_STORE="model_store"

# Create model_store directory if it doesn't exist
mkdir -p $MODEL_STORE

python -c "
import sys
sys.path.append('.')  # Make sure the current directory is in the Python path
from utils import load_hf_model, save_hf_model, MODEL_PATH

# Load the model and processor from Hugging Face
model, processor = load_hf_model()

# Save the model and processor to the specified directory
save_hf_model(model, processor, model_path=MODEL_PATH)
"