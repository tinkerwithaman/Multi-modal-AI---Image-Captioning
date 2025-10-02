import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def load_model_and_processor():
    """
    Loads the BLIP model and processor from Hugging Face.
    The model and processor will be downloaded and cached on the first run.
    """
    print("Loading model and processor... This may take a while on the first run.")
    
    # Define the model ID
    model_id = "Salesforce/blip-image-captioning-large"
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the processor and model
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    
    print("Model and processor loaded.")
    return processor, model, device

def generate_caption(processor, model, image_url, device):
    """
    Generates a caption for an image from a URL.

    Args:
        processor: The BLIP processor.
        model: The BLIP model.
        image_url (str): The URL of the image.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        str: The generated caption.
    """
    try:
        # Load image from URL
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    except requests.exceptions.RequestException as e:
        return f"Error: Could not retrieve image from URL. {e}"

    # --- Conditional generation ---
    # You can provide a text prompt to guide the caption generation
    # text = "a photography of"
    # inputs = processor(raw_image, text, return_tensors="pt").to(device)

    # --- Unconditional generation ---
    inputs = processor(raw_image, return_tensors="pt").to(device)

    # Generate the caption
    out = model.generate(**inputs, max_new_tokens=50) # Increase max_new_tokens for longer captions
    
    # Decode the generated tokens to text
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

if __name__ == "__main__":
    IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg" # A photo of cats and a remote
    
    processor, model, device = load_model_and_processor()
    
    print(f"Generating caption for image: {IMAGE_URL}")
    caption_text = generate_caption(processor, model, IMAGE_URL, device)
    
    print(f"\nGenerated Caption: {caption_text}")
