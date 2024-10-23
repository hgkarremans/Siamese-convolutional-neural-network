# House Image Embedding and Search System using a siamese neural network (SNN)

This project uses a Siamese Neural Network for image embedding and Milvus for efficient image similarity search. 
The goal is to detect duplicate or similar house images from a collection of property images by embedding them into a 128-dimensional vector space and searching for similar vectors based on Euclidean (L2) distance.

**Overview**

Siamese Neural Network: This model calculates the similarity between two images by embedding them into a vector space using a convolutional neural network (CNN). The L1 distance between image embeddings is used to measure their similarity.

Milvus Database: Milvus is used to store the image embeddings and enables fast similarity search based on the Euclidean distance (L2 distance) between image vectors.

**Components** 

- Siamese Neural Network (Embedding Model):
- MilvusSimaliritySearch

**prerequisites**

- Docker installed
- Python 3.12.2 installed (or similar version)

**How to Use:**

Start Milvus database using:
_bash standalone_embed.sh start_


To insert image vectors into the Milvus collection, specify the image folder path in image_folder and call insert_vectors(image_folder, embedding_model, collection).


Searching for Similar Images:

Use search_similar(image_path, embedding_model) to find the top K similar images in the collection. The default value of top_k is 10, but it can be adjusted as needed.
Example Usage

Uncomment code and add list of photos to be added to database in assets file. In this case combinedImages is folder with required images
**Insert vectors into the collection for the first time**
insert_vectors('assets/combinedImages', embedding_model, collection)

**Search for similar images using an input image. In this case picture is from folder called HouseImages. **
results = search_similar('assets/HouseImages/RandomHouse.jpg', embedding_model)

**Dependencies**
Dependencies are all in requirements.txt

**Notes**
Ensure that the milvus instance is running before attempting to insert or search vectors.
