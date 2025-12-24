import chromadb
# Initialize the Chroma client
CHROMA_PATH = "chromadb_data"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
 
COLLECTION_NAME = "project_vectors"
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
    # Use cosine similarity for retrieval
)