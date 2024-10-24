import os
import psycopg2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tqdm import tqdm  # Import tqdm for the loading bar

# Load your trained Siamese model (modify the path if needed)
model = load_model('embedding_model.keras')

# Define the folder containing the images
image_folder = 'assets/combinedImages'

# Connect to AlloyDB Omni
conn = psycopg2.connect(
    host="localhost",
    port="5433",  # Change this if you're using a different port
    database="Images",
    user="beheerder",
    password="Borghoek2003"
)

cursor = conn.cursor()

# Drop the table if it exists
# cursor.execute("DROP TABLE IF EXISTS image_embeddings;")
# conn.commit()
#
# # Create a new table for storing embeddings
# cursor.execute("""
#     CREATE TABLE image_embeddings (
#         id SERIAL PRIMARY KEY,
#         image_name TEXT NOT NULL,
#         embedding FLOAT8[] NOT NULL
#     );
# """)
# conn.commit()


# Function to preprocess images for the model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # Adjust size to 128x128
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Get the list of images in the folder
image_files = os.listdir(image_folder)

# Filter out only valid image files based on extensions
valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}  # Add more formats as needed
image_files = [img for img in image_files if os.path.splitext(img)[1].lower() in valid_extensions]

# Iterate through all images in the folder and generate embeddings with a loading bar
for img_name in tqdm(image_files, desc="Processing Images", unit="image"):
    img_path = os.path.join(image_folder, img_name)

    try:
        img_array = preprocess_image(img_path)

        # Generate the embedding using the Siamese model
        embedding = model.predict(img_array)

        # Flatten the embedding if needed
        embedding = embedding.flatten().tolist()

        # Insert the image name and embedding into the database
        cursor.execute("""
            INSERT INTO image_embeddings (image_name, embedding)
            VALUES (%s, %s);
        """, (img_name, embedding))
        conn.commit()
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")

# Query the database to count the total number of records
cursor.execute("SELECT COUNT(*) FROM image_embeddings;")
record_count = cursor.fetchone()[0]

print(f"Total number of records in the database: {record_count}")

# Close the database connection
cursor.close()
conn.close()

print("All embeddings have been inserted into AlloyDB Omni.")
