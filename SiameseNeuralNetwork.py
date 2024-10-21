import tensorflow as tf
import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


# --- Data Loading with tf.data ---

def preprocess_image(image_path):
    """ Function to load and preprocess a single image """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Adjust based on your image type
    img = tf.image.resize(img, (128, 128))  # Resize to the input size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img


def preprocess_pair(img1_path, img2_path, label):
    """ Preprocess two images and a label """
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)
    return (img1, img2), label


def create_dataset(csv_file, image_folder, batch_size):
    """ Create a tf.data.Dataset from the CSV file """
    data = pd.read_csv(csv_file)

    img1_paths = [os.path.join(image_folder, img) for img in data['Image1']]
    img2_paths = [os.path.join(image_folder, img) for img in data['Image2']]
    labels = data['Label'].values

    # Convert to tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((img1_paths, img2_paths, labels))

    # Shuffle, preprocess, batch, and prefetch
    dataset = dataset.shuffle(buffer_size=len(labels))
    dataset = dataset.map(lambda img1, img2, label: preprocess_pair(img1, img2, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch for efficiency

    return dataset


# --- Model Definition ---

def build_base_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (10, 10), activation='relu')(input_layer)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    output_layer = Dense(128, activation='sigmoid')(x)
    return Model(inputs=input_layer, outputs=output_layer)


def compute_l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])


def build_siamese_model(input_shape):
    base_model = build_base_model(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    encoded_a = base_model(input_a)
    encoded_b = base_model(input_b)

    l1_layer = Lambda(compute_l1_distance)([encoded_a, encoded_b])
    output_layer = Dense(1, activation='sigmoid')(l1_layer)

    siamese_model = Model(inputs=[input_a, input_b], outputs=output_layer)
    return siamese_model


# --- Train Function ---

def train_model(csv_file, image_folder, model_file, input_shape, batch_size=4, epochs=5):
    # Create dataset
    dataset = create_dataset(csv_file, image_folder, batch_size)

    if not os.path.exists(model_file):
        # Build and compile the Siamese model
        siamese_model = build_siamese_model(input_shape)
        siamese_model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )

        # Train the model using the tf.data dataset
        siamese_model.fit(dataset, epochs=epochs)
        siamese_model.save(model_file)
    else:
        # Load the model if it already exists
        siamese_model = load_model(model_file, custom_objects={'compute_l1_distance': compute_l1_distance})

    return siamese_model


# --- Test Similarity ---

def test_similarity(image1_path, image2_path, model, image_folder):
    img1 = preprocess_image(os.path.join(image_folder, image1_path))
    img2 = preprocess_image(os.path.join(image_folder, image2_path))

    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    similarity_score = model.predict([img1, img2])[0][0]
    return similarity_score


# --- Main Execution ---

csv_file = 'assets/training_data.csv'
image_folder = 'assets/AugmentedImages'
input_shape = (128, 128, 3)
model_file = 'siamese_model.keras'

# Train the model using lazy loading and small batches
siamese_model = train_model(csv_file, image_folder, model_file, input_shape)

# Test the model with two images
new_image_directory = 'assets/HouseImages'
image1 = 'Lijnmarkt.jpg'
image2 = 'LijnmarktKopie.jpg'
similarity = test_similarity(image1, image2, siamese_model, new_image_directory)
print(f"Similarity score between {image1} and {image2}: {similarity:.2f}")
