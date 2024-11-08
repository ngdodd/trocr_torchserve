import torch
from utils import load_hf_model, load_hf_dataset, MODEL_PATH
from profile_model import inference_step, profile_inference

print("Loading huggingface model...")
model, processor = load_hf_model()
print("Loading huggingface dataset...")
dataset = load_hf_dataset()

BATCH_SIZE = 4
sample_images = [img.convert('RGB') for img in dataset['validation'][:BATCH_SIZE]['image']]

def preprocess_images(batch, processor):
    return processor(images=batch, return_tensors='pt').pixel_values

def postprocess_text(batch, processor):
    return processor.batch_decode(batch, skip_special_tokens=True)

processed_images = processor(images=sample_images, return_tensors='pt').pixel_values
print("Profiling inference for loaded model...")
profile_inference(model,
                  processor,
                  sample_images,
                  log_out_dir='./log_original/', 
                  device='cpu',
                  preprocessing_fn=preprocess_images,
                  postprocessing_fn=postprocess_text)