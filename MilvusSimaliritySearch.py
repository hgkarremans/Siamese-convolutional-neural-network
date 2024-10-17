import os
import cv2
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import tensorflow as tf
from tqdm import tqdm


# Define the Lambda function outside the model to ensure tf is in scope
@register_keras_serializable()
def compute_l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])


# Load the Siamese and embedding models
model_file = 'siamese_model.keras'
embedding_model_file = 'embedding_model.keras'
siamese_model = load_model(model_file, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)
embedding_model = load_model(embedding_model_file, custom_objects={'compute_l1_distance': compute_l1_distance},
                             safe_mode=False)

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


def insert_vectors(image_folder, model, collection: Collection):
    vectors = []
    image_names = []
    image_list = os.listdir(image_folder)

    for image_name in tqdm(image_list, desc="Inserting vectors into DB", leave=True, ncols=100,
                           bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}'):
        image_path = os.path.join(image_folder, image_name)
        # print(image_name)
        vector = extract_vector(image_path, model)
        if vector is not None:
            vectors.append(vector.tolist())  # Convert numpy array to list
            image_names.append(image_name)

    # Insert vectors and image names into the Milvus collection
    if vectors and image_names:
        # Assuming your collection schema has two fields: 'image_name' and 'vector'
        entities = [
            image_names,  # Field 1: image names
            vectors  # Field 2: vectors
        ]

        collection.insert(entities)  # Insert entities into Milvus
        print("Insertion complete!")


# Example usage
image_folder = 'assets/HouseImages'
insert_vectors(image_folder, embedding_model, collection)

# Create an index for faster search
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

# Load the collection into memory
collection.load()


# Function to search for similar vectors
def search_similar(image_path, model, top_k=1):
    query_vector = extract_vector(image_path, model)
    if query_vector is None:
        print(f"Error: Unable to extract vector for image at {image_path}")
        return []
    results = collection.search([query_vector.tolist()], "embedding", {"metric_type": "L2", "params": {"nprobe": 10}},
                                limit=top_k)
    return results

# approx time taken is between 18ms to 80ms for 5 images
# Example search
results = search_similar('assets/HouseImages/Lijnmarkt.jpg', embedding_model)
for result in results[0]:
    print("Matching Object Attributes:")
    for attr, value in result.entity.__dict__.items():
        print(f"{attr}: {value}")
