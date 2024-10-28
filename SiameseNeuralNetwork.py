import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
import os
from keras.saving import register_keras_serializable
from tensorflow.python.layers.core import Dropout
from tqdm import tqdm

from ImageDataGenerator import ImageDataGenerator
from utils import load_image

# Function to calculate L1 distance between two tensors (Manhattan distance)
@register_keras_serializable()
def compute_l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])

# Load data from CSV and prepare image pairs and labels for training
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

    # Define the convolutional layers with neurons and filters
    # https://medium.com/advanced-deep-learning/cnn-operation-with-2-kernels-resulting-in-2-feature-mapsunderstanding-the-convolutional-filter-c4aad26cf32
    x = Conv2D(32, (10, 10), activation='relu')(input_layer)
    # Downsample the output of the convolutional layers https://keras.io/api/layers/pooling_layers/max_pooling2d/
    x = MaxPooling2D()(x)

    x = Conv2D(64, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)

    # Flatten the output of the convolutional layers to feed into the dense layers https://www.educative.io/answers/what-is-a-neural-network-flatten-layer
    x = Flatten()(x)
    # Dropout is a regularization technique to prevent overfitting https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9
    x = Dropout(0.5)(x)
    # Layer with 128 fully connected neurons to output 128-dimensional embeddings using linear activation function to
    # calculate the L1 distance between the embeddings
    x = Dense(128, activation='linear')(x)

    return Model(inputs=input_layer, outputs=x)

#define the neuron network for the Siamese model
def build_siamese_model(input_shape):
    # Build the base model
    base_model = build_base_model(input_shape)

    # Define the two input layers
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

# Function to test the similarity between two images used for testing while training network
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

# check if model already exists otherwise train the model
if not os.path.exists(model_file):

    # Load and preprocess data using the generator
    csv_file = 'assets/training_data.csv'
    image_folder = 'assets/AugmentedImages'
    batch_size = 16
    data_generator = ImageDataGenerator(csv_file, image_folder, batch_size)

    # Build the model
    input_shape = (128, 128, 3)
    siamese_model, embedding_model = build_siamese_model(input_shape)

    # Compile the model with binary crossentropy loss and Adam optimizer
    #https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
    # small learning rate to avoid overshooting the weight change to avoid overfitting
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    # Train the model using the generator
    # Epoch is the number of times the model will see the entire dataset
    siamese_model.fit(data_generator, epochs=10 , validation_data=data_generator)

    # Save the models
    siamese_model.save(model_file)
    embedding_model.save(embedding_model_file)
# if model already exists load the model to avoid training again
else:
    # Load the trained models with custom objects
    siamese_model = load_model(model_file, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)
    embedding_model = load_model(embedding_model_file, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


