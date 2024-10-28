import os
import psycopg2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load your trained Siamese model (modify the path if needed)
model = load_model('embedding_model.keras')

# Define the folder containing the images
image_folder = 'assets/combinedImages'

# Connect to AlloyDB Omni
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="Images",
    user="beheerder",
    password="test12345"
)

cursor = conn.cursor()

# Function to preprocess images for the model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to process and insert a single image
def process_and_insert_image(img_name):
    img_path = os.path.join(image_folder, img_name)
    try:
        img_array = preprocess_image(img_path)
        embedding = model.predict(img_array)
        embedding = embedding.flatten().tolist()
        cursor.execute("""
            INSERT INTO image_embeddings (image_name, embedding)
            VALUES (%s, %s);
        """, (img_name, embedding))
        conn.commit()
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")

# Get the list of images in the folder
image_files = os.listdir(image_folder)
valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
image_files = [img for img in image_files if os.path.splitext(img)[1].lower() in valid_extensions]

# Use ThreadPoolExecutor to process images in parallel
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_and_insert_image, img_name) for img_name in image_files]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Images", unit="image"):
        future.result()

# Query the database to count the total number of records
cursor.execute("SELECT COUNT(*) FROM image_embeddings;")
record_count = cursor.fetchone()[0]

print(f"Total number of records in the database: {record_count}")

# Close the database connection
cursor.close()
conn.close()

print("All embeddings have been inserted into AlloyDB Omni.")