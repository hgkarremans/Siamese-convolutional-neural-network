import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
import os
from keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Load images from assets folder
def load_image(image_path, loaded_images):
    if image_path in loaded_images:
        return loaded_images[image_path]

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded. Check the file path and integrity.")
    img = cv2.resize(img, (128, 128))  # Resize to the input size
    img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

    loaded_images[image_path] = img
    return img


# Read CSV and prepare image pairs and labels
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
            print(e)  # Log the error message
            continue  # Skip this pair and continue with the next

        image_pairs.append([img1, img2])
        labels.append(row['Label'])  # Ensure 'Label' matches your CSV header case sensitivity

    return np.array(image_pairs), np.array(labels)


# Define the base network for the Siamese model with minimal complexity
def build_base_model(input_shape):
    input_layer = Input(shape=input_shape)

    x = Conv2D(32, (10, 10), activation='relu')(input_layer)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (7, 7), activation='relu')(input_layer)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (4, 4), activation='relu')(input_layer)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(128, activation='sigmoid')(x)

    return Model(inputs=input_layer, outputs=x)


# Define the Lambda function outside the model to ensure tf is in scope
@register_keras_serializable()
def compute_l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])


# Define the Siamese network
def build_siamese_model(input_shape):
    base_model = build_base_model(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    encoded_a = base_model(input_a)
    encoded_b = base_model(input_b)

    # Lambda layer to compute the absolute difference between the two embeddings
    l1_layer = Lambda(
        compute_l1_distance,
        output_shape=lambda input_shapes: input_shapes[0]
    )
    l1_distance = l1_layer([encoded_a, encoded_b])

    # Dense layer to output the similarity score
    output_layer = Dense(1, activation='sigmoid')(l1_distance)

    siamese_model = Model(inputs=[input_a, input_b], outputs=output_layer)

    return siamese_model


# Function to test the similarity between two images
def test_similarity(image1_path, image2_path, model, image_folder):
    loaded_images = {}  # Initialize an empty dictionary for loaded images
    img1 = load_image(os.path.join(image_folder, image1_path), loaded_images)
    img2 = load_image(os.path.join(image_folder, image2_path), loaded_images)

    # Reshape to add batch dimension
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    # Predict similarity score (1 is duplicate, 0 is unique)
    similarity_score = model.predict([img1, img2])[0][0]

    return similarity_score


# Load and preprocess data
csv_file = 'assets/training_data.csv'
image_folder = 'assets/AugmentedImages'
image_pairs, labels = load_data(csv_file, image_folder)

# Split data into two input arrays
X1 = np.array([pair[0] for pair in image_pairs])
X2 = np.array([pair[1] for pair in image_pairs])
y = np.array(labels)

# Define model file name
model_file = 'siamese_model.keras'

if not os.path.exists(model_file):
    # Build the model
    input_shape = (128, 128, 3)
    siamese_model = build_siamese_model(input_shape)

    # Compile the model
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    # Train the model
    siamese_model.fit([X1, X2], y, batch_size=16, epochs=5, validation_split=0.2)

    # Save the model
    siamese_model.save(model_file)
else:
    # Load the trained model with custom objects
    siamese_model = load_model(model_file, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# --- TESTING TWO NEW IMAGES ---
new_image_directory = 'assets/HouseImages'  # Specify the new directory where the images are located
image1 = 'Lijnmarkt.jpg'  # Replace with your test image file name
image2 = 'LijnmarktKopie.jpg'  # Replace with your test image file name

# Calculate similarity
similarity = test_similarity(image1, image2, siamese_model, new_image_directory)
print(f"Similarity score between {image1} and {image2}: {similarity:.2f}")

similarity = test_similarity('Lijnmarkt.jpg', 'pikachu.jpeg', siamese_model, new_image_directory)
print(f"Similarity score between Lijnmarkt.jpg and pikachu.jpeg: {similarity:.2f}")

similarity = test_similarity('Lijnmarkt.jpg', 'RandomHouse.jpg', siamese_model, new_image_directory)
print(f"Similarity score between Lijnmarkt.jpg and RandomHouse.jpg: {similarity:.2f}")

similarity = test_similarity('RandomHouse.jpg', 'RandomHouse.jpg', siamese_model, new_image_directory)
print(f"Similarity score between RandomHouse.jpg and RandomHouse.jpg: {similarity:.2f}")

similarity = test_similarity('RandomHouse.jpg', 'RandomHouse_cropped.jpg', siamese_model, new_image_directory)
print(f"Similarity score between RandomHouse.jpg and RandomHouse_cropped.jpg: {similarity:.2f}")

similarity = test_similarity('Lijnmarkt.jpg', 'flip_Lijnmarkt_0_747.jpeg', siamese_model, new_image_directory)
print(f"Similarity score between Lijnmarkt.jpg and Lijnmarkt-flipped.jpg: {similarity:.2f}")
