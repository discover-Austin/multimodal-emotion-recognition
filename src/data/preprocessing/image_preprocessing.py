from PIL import Image

def preprocess_image(image: Image.Image, transform):
    return transform(image)