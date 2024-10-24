import os
import cv2
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections
from keras.saving import register_keras_serializable
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from SiameseNeuralNetwork import load_image

# Function to calculate L1 distance between two tensors (Manhattan distance)
@register_keras_serializable()
def compute_l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])


# Load the Siamese and embedding models
model_file = 'siamese_model.keras'
embedding_model_file = 'embedding_model.keras'
siamese_model = load_model(model_file, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)
embedding_model = load_model(embedding_model_file, custom_objects={'compute_l1_distance': compute_l1_distance},
                             safe_mode=False)

# Connect to the milvus database
connections.connect("default", host="localhost", port="19530")

# Define the schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=255)
]
schema = CollectionSchema(fields, "HouseImages collection")

# Drop the collection if it exists for testing purposes
# if "house_images" in list_collections():
#     collection = Collection("house_images")
#     collection.drop()

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


# Function to process and insert a single image
def process_and_insert_image(image_name, image_folder, model):
    image_path = os.path.join(image_folder, image_name)
    vector = extract_vector(image_path, model)
    if vector is not None:
        return vector.tolist(), image_name
    return None, None


def insert_vectors(image_folder, model, collection: Collection):
    vectors = []
    image_names = []
    image_list = os.listdir(image_folder)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_and_insert_image, image_name, image_folder, model) for image_name in image_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Inserting vectors into DB", leave=True, ncols=100, bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}'):
            vector, image_name = future.result()
            if vector is not None:
                vectors.append(vector)
                image_names.append(image_name)

    # Insert vectors and image names into the Milvus collection
    if vectors and image_names:
        entities = [
            vectors,  # This is the list of embeddings
            image_names  # This is the list of image names
        ]

        # Insert entities into the collection
        try:
            collection.insert(entities)
            print("Insertion complete!")
        except Exception as e:
            print(f"Error during insertion: {e}")

        # Flush the collection to ensure data is written to disk
        collection.flush()

        # Load the collection into memory before counting entities
        collection.load()

        # Count the number of records in the collection
        num_records = collection.num_entities
        print(f"Total number of records in the database: {num_records}")

    else:
        print("No vectors or image names were found for insertion.")




image_folder = 'assets/combinedImages'

# Insert vectors into the collection for the first time
insert_vectors(image_folder, embedding_model, collection)

# Create an index for faster search
collection.create_index("embedding",
                        {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

# Load the collection into memory
collection.load()


# Function to search for similar vectors returning top K similar vectors based on L2 distance
# K refers to number of similar images to return
# l2 is the distance metric used to calculate the similarity (Euclidean distance)
def search_similar(image_path, model, top_k=10):
    query_vector = extract_vector(image_path, model)
    if query_vector is None:
        print(f"Error: Unable to extract vector for image at {image_path}")
        return []
    results = collection.search([query_vector.tolist()],
                                "embedding", {"metric_type": "L2", "params": {"nprobe": 10}},
                                limit=top_k, output_fields=["image_name"])
    return results


# search with a certain image, in real product this will be image that needs to be tested for duplicates
SEARCH_SIMILAR = search_similar('assets/HouseImages/RandomHouse.jpg', embedding_model)

# search in the collection
results = SEARCH_SIMILAR
for result in results[0]:
    print("\nMatching Object Attributes:")
    for attr, value in result.entity.__dict__.items():
        print(f"{attr}: {value}")


