import numpy as np
import psycopg2
import time
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import faiss

class ImageSearcher:
    def __init__(self, model_path, db_config):
        # Load the trained Siamese model
        self.model = load_model(model_path)

        # Connect to the AlloyDB Omni database
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()

    def preprocess_image(self, image_path):
        """Preprocess the input image to the format expected by the model."""
        img = image.load_img(image_path, target_size=(128, 128))  # Ensure size matches model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def generate_embedding(self, image_path):
        """Generate the embedding for the provided image."""
        img_array = self.preprocess_image(image_path)
        embedding = self.model.predict(img_array)
        return embedding.flatten()

    def fetch_all_embeddings(self):
        """Retrieve all image embeddings from the database."""
        self.cursor.execute("SELECT image_name, embedding FROM image_embeddings")
        return self.cursor.fetchall()

    def build_faiss_index(self, embeddings):
        """Build a faiss index for fast similarity search."""
        embedding_matrix = np.array([e[1] for e in embeddings]).astype('float32')
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        return index

    def find_similar_images(self, target_image_path, top_n=10):
        """Find the top-N most similar images to the given image."""
        start_time = time.time()  # Start the timer

        # Step 1: Generate embedding for the target image
        target_embedding = self.generate_embedding(target_image_path).astype('float32')

        # Step 2: Fetch all stored embeddings from the database
        embeddings = self.fetch_all_embeddings()

        # Step 3: Build faiss index
        index = self.build_faiss_index(embeddings)

        # Step 4: Search for the most similar embeddings
        distances, indices = index.search(np.array([target_embedding]), top_n)

        # Get the top-N most similar images
        similar_images = [(embeddings[i][0], distances[0][j]) for j, i in enumerate(indices[0])]

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        print(f"Search time: {elapsed_time:.4f} seconds")

        return similar_images

    def close(self):
        """Close the database connection."""
        self.cursor.close()
        self.conn.close()