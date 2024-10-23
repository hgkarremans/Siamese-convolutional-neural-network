import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.utils import register_keras_serializable
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections

from ImageDataGenerator import ImageDataGenerator

# Function to load and preprocess images
def load_image(image_path, loaded_images):
    if image_path in loaded_images:
        return loaded_images[image_path]

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")
    img = cv2.resize(img, (128, 128))  # Resize to the input size
    img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

    loaded_images[image_path] = img
    return img

# Load data from CSV and prepare image pairs and labels
def load_data(csv_file, image_folder):
    data = pd.read_csv(csv_file)
    image_pairs = []
    labels = []
    loaded_images = {}

    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Loading images"):
        img1_path = os.path.join(image_folder, row['Image1'])
        img2_path = os.path.join(image_folder, row['Image2'])

        try:
            img1 = load_image(img1_path, loaded_images)
            img2 = load_image(img2_path, loaded_images)
        except FileNotFoundError as e:
            print(e)
            continue

        image_pairs.append([img1, img2])
        labels.append(row['Label'])

    return np.array(image_pairs), np.array(labels)

# Define the base network for the Siamese model
def build_base_model(input_shape):
    input_layer = Input(shape=input_shape, name='input_1')

    x = Conv2D(32, (10, 10), activation='relu')(input_layer)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='linear')(x)

    return Model(inputs=input_layer, outputs=x)

# Define the Lambda function for L1 distance
@register_keras_serializable()
def compute_l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])

def build_siamese_model(input_shape):
    base_model = build_base_model(input_shape)

    input_a = Input(shape=input_shape, name='input_1')
    input_b = Input(shape=input_shape, name='input_2')

    # Get embeddings for both inputs
    encoded_a = base_model(input_a)
    encoded_b = base_model(input_b)

    # Compute the L1 distance between the two encodings
    l1_distance = Lambda(compute_l1_distance)([encoded_a, encoded_b])

    # Add a Dense layer with a single unit and sigmoid activation
    output = Dense(1, activation='sigmoid')(l1_distance)

    # Model to output the 128-dimensional embedding for a single image
    embedding_model = Model(inputs=input_a, outputs=encoded_a)

    return Model(inputs=[input_a, input_b], outputs=output), embedding_model

# Function to test the similarity between two images
def test_similarity(image1_path, image2_path, model, image_folder):
    loaded_images = {}
    img1 = load_image(os.path.join(image_folder, image1_path), loaded_images)
    img2 = load_image(os.path.join(image_folder, image2_path), loaded_images)

    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    similarity_score = model.predict([img1, img2])[0][0]
    return similarity_score



# Define model file names
model_file = 'siamese_model.keras'
embedding_model_file = 'embedding_model.keras'

if not os.path.exists(model_file):
    # Define model file names
    model_file = 'siamese_model.keras'
    embedding_model_file = 'embedding_model.keras'

    # Load and preprocess data using the generator
    csv_file = 'assets/training_data.csv'
    image_folder = 'assets/AugmentedImages'
    batch_size = 16
    data_generator = ImageDataGenerator(csv_file, image_folder, batch_size)

    # Build the model
    input_shape = (128, 128, 3)
    siamese_model, embedding_model = build_siamese_model(input_shape)

    # Compile the model
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    # Train the model using the generator
    siamese_model.fit(data_generator, epochs=20 , validation_data=data_generator)

    # Save the models
    siamese_model.save(model_file)
    embedding_model.save(embedding_model_file)
else:
    # Load the trained models with custom objects
    siamese_model = load_model(model_file, custom_objects={'compute_l1_distance': compute_l1_distance})
    embedding_model = load_model(embedding_model_file, custom_objects={'compute_l1_distance': compute_l1_distance})
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


# Function to test the similarity between two images
def test_similarity(image1_path, image2_path, model, image_folder):
    loaded_images = {}
    img1 = load_image(os.path.join(image_folder, image1_path), loaded_images)
    img2 = load_image(os.path.join(image_folder, image2_path), loaded_images)

    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    similarity_score = model.predict([img1, img2])[0][0]
    return similarity_score

# # simalirity test
pikachu = 'pikachu.jpeg'
lijnmarkt = 'Lijnmarkt.jpg'
lijnmarktKopie = 'LijnmarktKopie.jpg'
flippedLijnmarkt = 'flip_Lijnmarkt_0_747.jpeg'
randomHouse = 'RandomHouse.jpg'
randomHouseCropped = 'RandomHouse_cropped.jpg'
randomHouseLessCropped = 'RandomHouse_less_cropped.jpg'
randomHouseColor = 'RandomHouse_different_color.jpg'
RandomHouseRotated = 'RandomHouse_rotated.jpg'
RandomHouseWatermark = 'randomhouse_watermark.png'
print(f"Similarity score between {pikachu} and {lijnmarkt}: {test_similarity(pikachu, lijnmarkt, siamese_model, 'assets/HouseImages')}")
print(f"Similarity score between {lijnmarkt} and {lijnmarktKopie}: {test_similarity(lijnmarkt, lijnmarktKopie, siamese_model, 'assets/HouseImages')}")
print(f"Similarity score between {lijnmarkt} and {flippedLijnmarkt}: {test_similarity(lijnmarkt, flippedLijnmarkt, siamese_model, 'assets/HouseImages')}")
print(f"Similarity score between {randomHouse} and {randomHouseCropped}: {test_similarity(randomHouse, randomHouseCropped, siamese_model, 'assets/HouseImages')}")
print(f"Similarity score between {randomHouse} and {randomHouseLessCropped}: {test_similarity(randomHouse, randomHouseLessCropped, siamese_model, 'assets/HouseImages')}")
print(f"Similarity score between {randomHouse} and {randomHouseColor}: {test_similarity(randomHouse, randomHouseColor, siamese_model, 'assets/HouseImages')}")
print(f"Similarity score between {randomHouse} and {RandomHouseRotated}: {test_similarity(randomHouse, RandomHouseRotated, siamese_model, 'assets/HouseImages')}")
print(f"Similarity score between {randomHouse} and {RandomHouseWatermark}: {test_similarity(randomHouse, RandomHouseWatermark, siamese_model, 'assets/HouseImages')}")


