import os
import cv2
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import tensorflow as tf

# Define the Lambda function outside the model to ensure tf is in scope
@register_keras_serializable()
def compute_l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])

# Load the Siamese and embedding models
model_file = 'siamese_model.keras'
embedding_model_file = 'embedding_model.keras'
siamese_model = load_model(model_file, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)
embedding_model = load_model(embedding_model_file, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=255)
]
schema = CollectionSchema(fields, "HouseImages collection")

# Drop the collection if it exists
if "house_images" in list_collections():
    collection = Collection("house_images")
    collection.drop()

# Create the collection
collection = Collection("house_images", schema)

# Function to extract vectors using the embedding model
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

# Insert vectors into Milvus
def insert_vectors(image_folder, model):
    vectors = []
    image_names = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        vector = extract_vector(image_path, model)
        if vector is not None:
            vectors.append(vector.tolist())  # Convert numpy array to list
            image_names.append(image_name)
    collection.insert([vectors, image_names])

# Example usage
image_folder = 'assets/HouseImages'
insert_vectors(image_folder, embedding_model)

# Create an index for faster search
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

# Load the collection into memory
collection.load()

# Function to search for similar vectors
def search_similar(image_path, model, top_k=5):
    query_vector = extract_vector(image_path, model)
    if query_vector is None:
        print(f"Error: Unable to extract vector for image at {image_path}")
        return []
    results = collection.search([query_vector.tolist()], "embedding", {"metric_type": "L2", "params": {"nprobe": 10}}, limit=top_k)
    return results

# Example search
results = search_similar('assets/HouseImages/Lijnmarkt.jpg', embedding_model)
for result in results[0]:
    print(f"ID: {result.id}, Distance: {result.distance}, Image Name: {result.entity.get('image_name')}")