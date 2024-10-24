# House Image Embedding and Search System using a siamese neural network (SNN)

This project uses a Siamese Neural Network for image embedding and several databases which support vectorizing the images for efficient image similarity search. 
The goal is to detect duplicate or similar house images from a collection of property images by embedding them into a (128)-dimensional vector space and searching for similar vectors based on Euclidean (L2) distance.

**Overview**

Siamese Neural Network: This model calculates the similarity between two images by embedding them into a vector space using a convolutional neural network (CNN). The L1 distance between image embeddings is used to measure their similarity.

Milvus Database: Milvus is used to store the image embeddings and enables fast similarity search based on the Euclidean distance (L2 distance) between image vectors, this is a vector database.

PostgreSQL with pgvector: This is a standard SQL database with added functionality for vectors, and is able to use L1 and L2 algoritms to find top K's fast.

AlloyDB Omni: This is implementation of google for a PostgreSQL database with pgvectors, which utilatizes optimations for using algoritms to be faster then standard PostgreSQL with Pgvector.

**Components** 

- Siamese Neural Network (Embedding Model):
- MilvusSimaliritySearch
- PostgreSQL
- AlloyDB Omni

**Prerequisites**

- Docker installed
- Python 3.12.2 installed (or similar version)
- Depending of database used, starting a local AlloyDB Omni or PostgreSQL database

# How to Use:

Start for example Milvus using:

_bash standalone_embed.sh start_

Insert image vectors into the Milvus collection, specify the image folder path in image_folder and call insert_vectors(image_folder, embedding_model, collection) in **MilvusSimaliritySearch**.

**Insert vectors into the collection for the first time**

insert_vectors('assets/combinedImages', embedding_model, collection)

**Search for similar images using an input image. In this case picture is from folder called HouseImages.**

results = search_similar('assets/HouseImages/RandomHouse.jpg', embedding_model)

**Dependencies**

Dependencies are all in requirements.txt and can be installed there or using:

_pip install -r requirements.txt_

**Notes**

- Ensure that the database instance is running before attempting to insert or search vectors.
- Code is still a prototype, so a lot of improvements can be made.
- Different databases could be used.
