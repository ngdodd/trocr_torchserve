import io
import torch
import base64
from ts.torch_handler.base_handler import BaseHandler
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class CustomHandler(BaseHandler):
  def initialize(self, context):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = context.system_properties.get("model_dir")
    self.processor = TrOCRProcessor.from_pretrained(model_dir) # contains image encoder and text tokenizer
    self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    self.model = self.model.to(self.device)
    self.model.eval()

  def preprocess(self, requests):
    image_list = requests[0].get("data")
    if image_list is None:
        image_list = requests[0].get("body")
    print(requests)
    
    # capable of accepting multiple images at a time for batching inference calls
    images = []
    for item in image_list:
        image_data = base64.b64decode(item['data'])
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        images.append(image)
    encodings = self.processor(images=images, return_tensors='pt').pixel_values
    return encodings.to(self.device)

  def inference(self, data):
    with torch.no_grad():
      outputs = self.model.generate(data)
    return outputs

  def postprocess(self, outputs):
    decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)
    print(decoded_outputs)
    return [decoded_outputs]