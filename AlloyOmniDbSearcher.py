import numpy as np
import psycopg2
import time
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

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

    def compute_similarity(self, target_embedding, embeddings):
        """Compute cosine similarity between the target embedding and the list of embeddings."""
        embedding_matrix = np.array([e[1] for e in embeddings])  # Extract just the embeddings
        similarities = cosine_similarity([target_embedding], embedding_matrix)[0]  # Compute cosine similarities
        return similarities

    def find_similar_images(self, target_image_path, top_n=10):
        """Find the top-N most similar images to the given image."""
        start_time = time.time()  # Start the timer

        # Step 1: Generate embedding for the target image
        target_embedding = self.generate_embedding(target_image_path)

        # Step 2: Fetch all stored embeddings from the database
        embeddings = self.fetch_all_embeddings()

        # Step 3: Compute similarities
        similarities = self.compute_similarity(target_embedding, embeddings)

        # Step 4: Sort by similarity and get top-N results
        sorted_indices = np.argsort(similarities)[::-1][:top_n]  # Sort in descending order

        # Get the top-N most similar images
        similar_images = [(embeddings[i][0], similarities[i]) for i in sorted_indices]

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        print(f"Search time: {elapsed_time:.4f} seconds")

        return similar_images

    def close(self):
        """Close the database connection."""
        self.cursor.close()
        self.conn.close()

# Example usage
if __name__ == "__main__":
    model_path = 'embedding_model.keras'
    db_config = {
        'host': 'localhost',
        'port': '5433',
        'database': 'Images',
        'user': 'beheerder',
        'password': 'Borghoek2003'
    }
    searcher = ImageSearcher(model_path, db_config)
    similar_images = searcher.find_similar_images('assets/HouseImages/Lijnmarkt.jpg')
    print(similar_images)
    searcher.close()