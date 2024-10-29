import os

import numpy as np
from tensorflow.keras.models import Model, load_model

from utils import load_image


def compute_l2_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def generate_embeddings_and_compute_l2(image1_path, image2_path, embedding_model, image_folder):
    loaded_images = {}
    img1 = load_image(os.path.join(image_folder, image1_path), loaded_images)
    img2 = load_image(os.path.join(image_folder, image2_path), loaded_images)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    embedding1 = embedding_model.predict(img1)[0]
    embedding2 = embedding_model.predict(img2)[0]
    l2_distance = compute_l2_distance(embedding1, embedding2)
    return l2_distance

image_folder= 'assets/HouseImages'

embedding_model_file = 'embedding_model.keras'
embedding_model = load_model(embedding_model_file, custom_objects={'compute_l2_distance': compute_l2_distance},
                             safe_mode=False)


print(f"\nL2 distance between Lijnmarkt.jpg and Lijnmarkt.jpg is {generate_embeddings_and_compute_l2('Lijnmarkt.jpg', 'Lijnmarkt.jpg', embedding_model, image_folder)}")
print(f"\nL2 distance between Lijnmarkt.jpg and LijnmarktKopie.jpg is {generate_embeddings_and_compute_l2('Lijnmarkt.jpg', 'LijnmarktKopie.jpg', embedding_model, image_folder)}")
print(f"\nL2 distance between Lijnmarkt.jpg and flip_Lijnmarkt.jpeg is {generate_embeddings_and_compute_l2('Lijnmarkt.jpg', 'flip_Lijnmarkt_0_747.jpeg', embedding_model, image_folder)}")
print(f"\nL2 distance between RandomHouse.jpg and RandomHouse.jpeg is {generate_embeddings_and_compute_l2('RandomHouse.jpg', 'RandomHouse.jpg', embedding_model, image_folder)}")
print(f"\nL2 distance between RandomHouse.jpg and RandomHouse_rotated.jpg is {generate_embeddings_and_compute_l2('RandomHouse.jpg', 'RandomHouse_rotated.jpg', embedding_model, image_folder)}")
print(f"\nL2 distance between RandomHouse.jpg and RandomHouse_cropped.jpg is {generate_embeddings_and_compute_l2('RandomHouse.jpg', 'RandomHouse_cropped.jpg', embedding_model, image_folder)}")
print(f"\nL2 distance between RandomHouse.jpg and RandomHouse_less_cropped.jpg is {generate_embeddings_and_compute_l2('RandomHouse.jpg', 'RandomHouse_less_cropped.jpg', embedding_model, image_folder)}")
print(f"\nL2 distance between RandomHouse.jpg and RandomHouse_different_color.jpg is {generate_embeddings_and_compute_l2('RandomHouse.jpg', 'RandomHouse_different_color.jpg', embedding_model, image_folder)}")
print(f"\nL2 distance between RandomHouse.jpg and RandomHouse_watermark.png is {generate_embeddings_and_compute_l2('RandomHouse.jpg', 'randomhouse_watermark.png', embedding_model, image_folder)}")
print(f"\nL2 distance between RandomHouse.jpg and pikachu.jpeg is {generate_embeddings_and_compute_l2('RandomHouse.jpg', 'pikachu.jpeg', embedding_model, image_folder)}")
print(f"\nL2 distance between RandomHouse.jpg and Lijnmarkt.jpg is {generate_embeddings_and_compute_l2('RandomHouse.jpg', 'Lijnmarkt.jpg', embedding_model, image_folder)}")



