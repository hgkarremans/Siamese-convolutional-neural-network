from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable

# Define the Lambda function outside the model to ensure tf is in scope
@register_keras_serializable()
def compute_l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])

# Load the Siamese model
model_file = 'siamese_model.keras'
siamese_model = load_model(model_file, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, "HouseImages collection")

# Create the collection
collection = Collection("house_images", schema)

# Function to extract vectors using the Siamese model
def extract_vector(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    vector = model.predict(img)[0]
    return vector

# Insert vectors into Milvus
def insert_vectors(image_folder, model):
    vectors = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        vector = extract_vector(image_path, model)
        vectors.append(vector)
    collection.insert([vectors])

# Example usage
image_folder = 'assets/HouseImages'
insert_vectors(image_folder, siamese_model)

# Create an index for faster search
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

# Load the collection into memory
collection.load()

# Function to search for similar vectors
def search_similar(image_path, model, top_k=5):
    query_vector = extract_vector(image_path, model)
    results = collection.search([query_vector], "embedding", {"metric_type": "L2", "params": {"nprobe": 10}}, limit=top_k)
    return results

# Example search
results = search_similar('assets/HouseImages/Lijnmarkt.jpg', siamese_model)
for result in results[0]:
    print(f"ID: {result.id}, Distance: {result.distance}")