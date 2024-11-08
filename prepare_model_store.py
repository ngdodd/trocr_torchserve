from utils import load_hf_model, load_hf_dataset, save_hf_model
from pathlib import Path

path = Path("./model_store")
path.mkdir(parents=True, exist_ok=True)

model, processor = load_hf_model()
dataset = load_hf_dataset()
save_hf_model(model, BASE_PATH)