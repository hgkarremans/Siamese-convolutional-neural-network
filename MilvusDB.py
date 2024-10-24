from pymilvus import MilvusClient

# Connect to the Milvus database
client = MilvusClient("milvus_demo.db")

# if client.has_collection(collection_name="images"):
#     client.drop_collection(collection_name="images")
# client.create_collection(
#     collection_name="images",
#     dimension=768,  # The vectors in this collection have a dimension of 768
# )
client.get_collection_stats(collection_name="milvus_demo")