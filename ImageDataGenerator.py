import tensorflow as tf
from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import os
from utils import load_image

class ImageDataGenerator(Sequence):
    def __init__(self, csv_file, image_folder, batch_size=16):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.loaded_images = {}

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        image_pairs = []
        labels = []

        for _, row in batch_data.iterrows():
            img1_path = os.path.join(self.image_folder, row['Image1'])
            img2_path = os.path.join(self.image_folder, row['Image2'])

            try:
                img1 = load_image(img1_path, self.loaded_images)
                img2 = load_image(img2_path, self.loaded_images)
            except FileNotFoundError as e:
                print(e)
                continue

            image_pairs.append([img1, img2])
            labels.append(row['Label'])

        X1 = np.array([pair[0] for pair in image_pairs])
        X2 = np.array([pair[1] for pair in image_pairs])
        y = np.array(labels)

        return {'input_1': X1, 'input_2': X2}, y