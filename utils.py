import cv2
import os

# Function to load an image from a file path
def load_image(image_path, loaded_images):
    if image_path in loaded_images:
        return loaded_images[image_path]

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0

    loaded_images[image_path] = img
    return img