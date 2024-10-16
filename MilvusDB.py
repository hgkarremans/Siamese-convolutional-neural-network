from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")

if client.has_collection(collection_name="images"):
    client.drop_collection(collection_name="images")
client.create_collection(
    collection_name="images",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)
