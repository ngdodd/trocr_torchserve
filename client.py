import requests
import base64
import io
import argparse
from PIL import Image
from utils import load_hf_dataset
import matplotlib.pyplot as plt

def encode_pil_image(pil_image):
    """Encodes a PIL image to a base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")  # Choose the format that suits your model (e.g., JPEG, PNG)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
def create_batch_payload(pil_images):
    """Creates a JSON payload with base64-encoded PIL images."""
    batch_payload = {
        "instances": [{"data": encode_pil_image(image)} for image in pil_images]
    }
    return batch_payload

def send_images_to_model(images, model_url):  
    # Send POST request and check response
    headers = {'Content-Type': 'application/json'}
    payload = [{"data": encode_pil_image(image)} for image in images]  # serialize all images in the batch
    response = requests.post(model_url, json=payload, headers=headers)
    print(response.json())
    if response.status_code == 200:
        return response
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def main(test_ids, max_batch_size, model_url):
    print("Loading HF dataset...")
    dataset = load_hf_dataset(split='test')

    # convert test_ids from string to a list of integers
    ids = [int(i) for i in test_ids.split(',')]

    # Loop through specified image IDs in batches
    batch_size = min(max_batch_size, len(ids))
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        images = []
        
        for k, img_id in enumerate(batch_ids):
            img_data = dataset[img_id]
            image = img_data['image']
            
            # display the image. This will block data form being sent to the server
            # until the image display is closed.
            plt.imshow(image)
            title = f'Image {k+1}/{batch_size}'
            if k == batch_size-1:
                title += ". Close window to send batch to the model for inference."
            else:
                title += ". Close window to display the next image in the batch."
            plt.title(title)
            plt.axis('off')
            plt.show()
            images.append(image)

        print(f"Sending batch of images {batch_ids} to the model...")
        
        # Perform inference on the batch of images and receive the response
        output = send_images_to_model(images, model_url)
        print(f"Response for batch {batch_ids}: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send images to TorchServe model.")
    parser.add_argument("--ids", type=str, required=True, help="Comma-separated IDs of the test dataset images to process.")
    parser.add_argument("--max_batch_size", type=int, default=4, help="Max number of images to process in a single request.")
    parser.add_argument("--model_url", type=str, default="http://127.0.0.1:8081/predictions/trocr_base", help="Prediction server url.")
    args = parser.parse_args()

    main(args.ids, args.max_batch_size, args.model_url)
