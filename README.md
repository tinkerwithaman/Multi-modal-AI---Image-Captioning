# Image Captioning with Hugging Face Transformers

## Description
This project demonstrates a modern multi-modal AI task: image captioning. It uses a powerful, pre-trained **BLIP (Bootstrapping Language-Image Pre-training)** model from the Hugging Face `transformers` library to automatically generate a textual description of an image.

The script loads an image from a URL, processes it, and then feeds it into the model to generate a descriptive caption. This is a great example of combining state-of-the-art Computer Vision and NLP models.

## Features
-   Leverages the Hugging Face `transformers` and `torch` libraries.
-   Uses a pre-trained `Salesforce/blip-image-captioning-large` model.
-   Models and processors are downloaded and cached automatically.
-   Simple and clean code to generate captions for any image URL.

## Setup and Installation

1.  **Clone the repository and navigate to the directory.**
2.  **Create a virtual environment and activate it.**
3.  **Install the dependencies:** `pip install -r requirements.txt`
    *Note: This will download several large libraries, including PyTorch.*
4.  **Run the script:** `python src/main.py`
    *Note: The first run will download the pre-trained model (approx. 1.8 GB), which will take some time depending on your internet connection.*

## Example Output
```
Loading model and processor... This may take a while on the first run.
Model and processor loaded.
Generating caption for image: [http://images.cocodataset.org/val2017/000000039769.jpg](http://images.cocodataset.org/val2017/000000039769.jpg)
Generated Caption: two cats sleeping on a couch next to a remote control
```
