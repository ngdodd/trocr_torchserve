from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import requests
from pathlib import Path

HF_DATASET_ID = 'Teklia/IAM-line'
HF_MODEL_ID = 'microsoft/trocr-base-handwritten'
MODEL_PATH = './trocr_model'

def load_hf_model(path=MODEL_PATH):
    model_path = Path(path)
    processor_path = model_path / "processor"

    if model_path.is_dir() and processor_path.is_dir(): # Exists locally
        print(f"Loading model and processor from local directory {model_path}")
        processor = TrOCRProcessor.from_pretrained(processor_path) # contains image encoder and text tokenizer
        model = VisionEncoderDecoderModel.from_pretrained(model_path)

    else: # download from hugging face
        path = HF_MODEL_ID
        print("Downloading model and processor from Hugging Face...")
        processor = TrOCRProcessor.from_pretrained(path) # contains image encoder and text tokenizer
        model = VisionEncoderDecoderModel.from_pretrained(path)

    return model, processor

def save_hf_model(model, processor=None, scripted=False, model_path=MODEL_PATH, filename=None):
    base_path = Path(model_path)
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {base_path}...")

    if scripted:
        file_path_name = HF_MODEL_ID[HF_MODEL_ID.rfind('/')+1:] + "_scripted.pt"
        torch.jit.save(model, base_path/file_path_name)
    else:
        model.save_pretrained(base_path)
        processor_path = base_path / "processor"
        print(f"Saving processor to {processor_path}")
        processor_path.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(processor_path)
	
def load_hf_dataset(dataset_id=HF_DATASET_ID, split=None, streaming=False):
	# TODO: handle any required authorization and credentials here
	return load_dataset(dataset_id, split=split, streaming=streaming)