import os
import time
import numpy as np
import psycopg2
from keras.src.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from tqdm import tqdm
from utils import load_image
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed

@register_keras_serializable()
def compute_l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])

class ImageEmbeddingDatabase:
    def __init__(self, model_path, image_folder, db_config):
        self.model_path = model_path
        self.image_folder = image_folder
        self.db_config = db_config
        self.model = self.load_model()

    def load_model(self):
        model = load_model(self.model_path, custom_objects={'compute_l1_distance': compute_l1_distance}, safe_mode=False)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def generate_embedding(self, image_path):
        img = load_image(image_path, {})
        img = np.expand_dims(img, axis=0)
        embedding = self.model.predict(img)
        return embedding[0]

    def store_embeddings(self):
        def process_and_insert_image(image_name):
            img_path = os.path.join(self.image_folder, image_name)
            try:
                embedding = self.generate_embedding(img_path)
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor()
                cur.execute("INSERT INTO images (image_name, image_vector) VALUES (%s, %s)",
                            (image_name, embedding.tolist()))
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

        image_files = os.listdir(self.image_folder)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_and_insert_image, img_name) for img_name in image_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images", unit="image"):
                future.result()

    def get_stored_embeddings(self):
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        cur.execute("SELECT image_name, image_vector FROM images")
        stored_embeddings = cur.fetchall()
        cur.close()
        conn.close()
        return stored_embeddings

    def find_top_similar_images(self, search_embedding, top_n=10):
        query = f"""
        SELECT image_name, image_vector <-> %s::vector AS distance
        FROM images
        ORDER BY distance
        LIMIT %s;
        """
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        cur.execute(query, (np.array(search_embedding).tolist(), top_n))  # Ensure it's a list
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results

    def count_vectors(self):
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM images")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count

    def show_top_similar_images(self, image_path, top_n=5):
        search_embedding = self.generate_embedding(image_path)
        top_similar_images = self.find_top_similar_images(search_embedding, top_n)

        for image_name, similarity in top_similar_images:
            print(f"Image: {image_name}, Distance: {similarity}")

    def check_duplicate(self, image_path, similarity_threshold=0.8):
        search_embedding = self.generate_embedding(image_path)
        top_similar_images = self.find_top_similar_images(search_embedding, top_n=1)

        if top_similar_images:
            closest_image, similarity = top_similar_images[0]
            if similarity >= similarity_threshold:
                print(f"Duplicate image found: {closest_image} with similarity: {similarity}")
            else:
                print(f"No duplicate image found. Highest similarity: {similarity}")
        else:
            print("No images found in the database.")

# Usage
if __name__ == "__main__":
    db_config = {
        'dbname': 'image_db',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432'
    }

    model_path = 'embedding_model.keras'
    image_folder = 'assets/combinedImages'
    total_time = time.time()
    image_embedding_db = ImageEmbeddingDatabase(model_path, image_folder, db_config)
    # image_embedding_db.store_embeddings()

    # Check for duplicate image
    image_path = 'assets/HouseImages/Lijnmarkt.jpg'
    start_time = time.time()
    image_embedding_db.show_top_similar_images(image_path, top_n=30)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTime taken to check for duplicates: {elapsed_time:.2f} seconds")

    vector_count = image_embedding_db.count_vectors()
    print(f"\nTotal number of vectors in the database: {vector_count}")
    total_elapsed_time = time.time() - total_time

    print(f"\nTotal time taken: {total_elapsed_time:.2f} seconds")