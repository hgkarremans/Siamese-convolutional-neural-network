import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
import os
from keras.saving import register_keras_serializable
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections


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
    input_layer = Input(shape=input_shape)

    x = Conv2D(32, (10, 10), activation='relu')(input_layer)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(128, activation='sigmoid')(x)

    return Model(inputs=input_layer, outputs=x)


# Define the Lambda function for L1 distance
@register_keras_serializable()
def compute_l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])


def build_siamese_model(input_shape):
    base_model = build_base_model(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

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


# Load and preprocess data
csv_file = 'assets/training_data.csv'
image_folder = 'assets/AugmentedImages'
image_pairs, labels = load_data(csv_file, image_folder)

# Split data into two input arrays
X1 = np.array([pair[0] for pair in image_pairs])
X2 = np.array([pair[1] for pair in image_pairs])
y = np.array(labels)

# Define model file names
model_file = 'siamese_model.keras'
embedding_model_file = 'embedding_model.keras'

if not os.path.exists(model_file):
    # Build the model
    input_shape = (128, 128, 3)
    siamese_model, embedding_model = build_siamese_model(input_shape)

    # Compile the model
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    # Train the model
    siamese_model.fit([X1, X2], y, batch_size=16, epochs=5, validation_split=0.2)

    # Save the models
    siamese_model.save(model_file)
    embedding_model.save(embedding_model_file)
else:
    # Load the trained models with custom objects
    siamese_model = load_model(model_file, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)
    embedding_model = load_model(embedding_model_file, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# # --- TESTING TWO NEW IMAGES ---
# new_image_directory = 'assets/HouseImages'
# image1 = 'Lijnmarkt.jpg'
# image2 = 'LijnmarktKopie.jpg'
#
# # Calculate similarity
# similarity = test_similarity(image1, image2, siamese_model, new_image_directory)
# print(f"Similarity score between {image1} and {image2}: {similarity:.2f}")

# Milvus Integration
# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, "HouseImages collection")

# # Drop the collection if it exists
# if "house_images" in list_collections():
#     collection = Collection("house_images")
#     collection.drop()

# Create the collection
collection = Collection("house_images", schema)


# Function to extract vectors using the Siamese model
def extract_vector(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Get the vector from the embedding model
    vector = model.predict(img)

    # Ensure the vector has the correct dimension
    if vector.shape[1] != 128:
        raise ValueError(f"Vector dimension mismatch: expected 128, got {vector.shape[1]}")

    return vector[0]


image_id_mapping = {}  # Dictionary to store mapping between ID and image name


def insert_vectors(image_folder, model):
    vectors = []
    image_names = []  # List to store image names
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        vector = extract_vector(image_path, model)
        if vector is not None:  # Check if vector is not None
            vectors.append(vector.tolist())  # Convert numpy array to list
            image_names.append(image_name)  # Store the image name

    insert_result = collection.insert([vectors])
    ids = insert_result.primary_keys  # Get the IDs generated by Milvus

    # Store the mapping between ID and image name
    for id, name in zip(ids, image_names):
        image_id_mapping[id] = name


# Example usage
image_folder = 'assets/HouseImages'
insert_vectors(image_folder, embedding_model)

# Create an index for faster search
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

# Load the collection into memory
collection.load()


def search_similar(image_path, model, top_k=5):
    query_vector = extract_vector(image_path, model)
    results = collection.search([query_vector.tolist()], "embedding", {"metric_type": "L2", "params": {"nprobe": 10}},
                                limit=top_k)

    # Print the corresponding image names based on IDs
    for result in results[0]:
        image_name = image_id_mapping.get(result.id, "Unknown")
        print(f"ID: {result.id}, Image: {image_name}, Distance: {result.distance}")

    return results
