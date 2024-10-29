import tensorflow as tf
from tensorflow import keras

# Load the .keras model file
model = keras.models.load_model("embedding_model.keras")

# Specify the directory where you want to save the model in SavedModel format
export_dir = "converted_model"

# Save the model in the SavedModel format
model.export(export_dir)
