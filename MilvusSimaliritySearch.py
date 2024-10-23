import os
import cv2
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections
from keras.saving import register_keras_serializable
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import load_model
from SiameseNeuralNetwork import load_image


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
            vectors,
            image_names
        ]

        collection.insert(entities)  # Insert entities into Milvus
        print("Insertion complete!")


image_folder = 'assets/combinedImages'
# insert_vectors(image_folder, embedding_model, collection)

# Create an index for faster search
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

# Load the collection into memory
collection.load()


def l2_to_similarity(l2_distance):
    # Convert L2 distance to similarity score (0 to 1)
    return np.exp(-l2_distance)


def search_similar(image_path, model, top_k=1):
    query_vector = extract_vector(image_path, model)
    if query_vector is None:
        print(f"Error: Unable to extract vector for image at {image_path}")
        return []

    results = collection.search(
        [query_vector.tolist()],
        "embedding",
        {"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["image_name"]
    )

    # Convert L2 distances to similarity scores
    similarity_results = []
    for result in results[0]:
        l2_distance = result.distance
        similarity_score = l2_to_similarity(l2_distance)
        similarity_results.append((result.entity.image_name, similarity_score))

    return similarity_results


# Example usage
SEARCH_SIMILAR = search_similar('assets/HouseImages/flip_Lijnmarkt_0_747.jpeg', embedding_model)
for image_name, similarity_score in SEARCH_SIMILAR:
    print(f"Image: {image_name}, Similarity Score: {similarity_score}")

# search in the collection
results = SEARCH_SIMILAR
for result in results[0]:
    print("Matching Object Attributes:")
    for attr, value in result.entity.__dict__.items():
        print(f"{attr}: {value}")



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
# pikachu = 'pikachu.jpeg'
# lijnmarkt = 'Lijnmarkt.jpg'
# lijnmarktKopie = 'LijnmarktKopie.jpg'
# flippedLijnmarkt = 'flip_Lijnmarkt_0_747.jpeg'
# randomHouse = 'RandomHouse.jpg'
# randomHouseCropped = 'RandomHouse_cropped.jpg'
# randomHouseLessCropped = 'RandomHouse_less_cropped.jpg'
# randomHouseColor = 'RandomHouse_different_color.jpg'
# RandomHouseRotated = 'RandomHouse_rotated.jpg'
# RandomHouseWatermark = 'randomhouse_watermark.png'
# print(f"Similarity score between {pikachu} and {lijnmarkt}: {test_similarity(pikachu, lijnmarkt, siamese_model, 'assets/HouseImages')}")
# print(f"Similarity score between {lijnmarkt} and {lijnmarktKopie}: {test_similarity(lijnmarkt, lijnmarktKopie, siamese_model, 'assets/HouseImages')}")
# print(f"Similarity score between {lijnmarkt} and {flippedLijnmarkt}: {test_similarity(lijnmarkt, flippedLijnmarkt, siamese_model, 'assets/HouseImages')}")
# print(f"Similarity score between {randomHouse} and {randomHouseCropped}: {test_similarity(randomHouse, randomHouseCropped, siamese_model, 'assets/HouseImages')}")
# print(f"Similarity score between {randomHouse} and {randomHouseLessCropped}: {test_similarity(randomHouse, randomHouseLessCropped, siamese_model, 'assets/HouseImages')}")
# print(f"Similarity score between {randomHouse} and {randomHouseColor}: {test_similarity(randomHouse, randomHouseColor, siamese_model, 'assets/HouseImages')}")
# print(f"Similarity score between {randomHouse} and {RandomHouseRotated}: {test_similarity(randomHouse, RandomHouseRotated, siamese_model, 'assets/HouseImages')}")
# print(f"Similarity score between {randomHouse} and {RandomHouseWatermark}: {test_similarity(randomHouse, RandomHouseWatermark, siamese_model, 'assets/HouseImages')}")


