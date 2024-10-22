import tensorflow as tf
import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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


def create_dataset(csv_file, image_folder, batch_size, val_split=0.2):
    """ Create a tf.data.Dataset from the CSV file, and split it into training and validation sets """
    data = pd.read_csv(csv_file)

    img1_paths = [os.path.join(image_folder, img) for img in data['Image1']]
    img2_paths = [os.path.join(image_folder, img) for img in data['Image2']]
    labels = data['Label'].values

    # Convert to tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((img1_paths, img2_paths, labels))

    # Shuffle, preprocess, and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(labels))
    dataset = dataset.map(lambda img1, img2, label: preprocess_pair(img1, img2, label),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Calculate the number of training samples
    num_samples = len(labels)
    val_size = int(val_split * num_samples)
    train_size = num_samples - val_size

    # Split the dataset into training and validation sets
    train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


# --- Model Definition ---

def build_base_model(input_shape):
    input_layer = Input(shape=input_shape, name='input_1')

    x = Conv2D(32, (10, 10), activation='relu')(input_layer)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    # Use linear activation to allow output beyond [0, 1]
    x = Dense(128, activation='linear')(x)

    return Model(inputs=input_layer, outputs=x)


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

def train_model(csv_file, image_folder, model_file, input_shape, batch_size=4, epochs=5, val_split=0.2):
    # Create training and validation datasets
    train_dataset, val_dataset = create_dataset(csv_file, image_folder, batch_size, val_split)

    if not os.path.exists(model_file):
        # Build and compile the Siamese model
        siamese_model = build_siamese_model(input_shape)
        siamese_model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )

        # Define early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # You can also monitor 'val_accuracy'
            patience=5,  # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
        )

        # Train the model with validation and early stopping
        siamese_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stopping]
        )
        # Save the model in the SavedModel format
        siamese_model.save(model_file, save_format='tf')
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
model_file = 'siamese_model.h5'

# Train the model using lazy loading, small batches, and with validation accuracy
siamese_model = train_model(csv_file, image_folder, model_file, input_shape, batch_size=32, epochs=10, val_split=0.2)

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