import chromadb
import os
from sentence_transformers import SentenceTransformer
from chromadb.errors import NotFoundError

# Load HuggingFace token from environment (Ensure HF_TOKEN is set in your environment)
token = os.getenv("HF_TOKEN")

# --- CUSTOM EMBEDDER CLASS ---
class GemmaEmbedder:
    def __init__(self, model_name="google/embeddinggemma-300m", token=None):
        self.model = SentenceTransformer(model_name, use_auth_token=token)
    def __call__(self, input):
        return self.model.encode(input).tolist() 
    def name(self):
        return "google/embeddinggemma-300m"

# --- CONFIGURATION FIX: Store DB relative to current script (src/database/) ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(current_script_dir, "chromadb_data") 

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
COLLECTION_NAME = "project_vectors"

# --- Delete and Recreate Collection ---
try:
    chroma_client.delete_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' deleted for a clean start.")
except NotFoundError: 
    print(f"Collection '{COLLECTION_NAME}' did not exist, proceeding to create it.")
    pass 

# Initialize the embedder instance
embedder = GemmaEmbedder(token=token)

# Recreate the collection with the corrected custom embedding function
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedder,
    metadata={"hnsw:space": "cosine"}
)
print(f"Collection '{COLLECTION_NAME}' created/recreated successfully with GemmaEmbedder.")

# --- Sample Data and Insertion ---
documents = [
    "Trainee A has a Technical Score of 95 and completed all modules successfully.",
    "Trainer John assessed Trainee B as having excellent communication skills and a positive attitude, making them suitable for client-facing roles.",
    "Trainee C took 5 PTO days during the training period and scored 78 on the final technical exam.",
    "Trainer Jane specializes in Python and SQL database management, and she is available for new assignments.",
    "Trainee D received an Attitude score of 4.5/5.0 and a Communication score of 4.8/5.0 from Trainer Sarah.",
    "Trainer Mark is proficient in AWS Cloud services and DevOps methodologies."
]
metadatas = [
    {"source": "KPI_Report.xlsx", "row_id": 10, "trainee_name": "A", "type": "KPI"},
    {"source": "Trainer_Assessment_Q3.json", "row_id": 25, "trainee_name": "B", "type": "Assessment", "assessor": "John"},
    {"source": "KPI_Report.xlsx", "row_id": 12, "trainee_name": "C", "type": "KPI"},
    {"source": "Trainer_Profile.csv", "row_id": 5, "trainer_name": "Jane", "type": "Trainer", "tech_skill": "Python"},
    {"source": "Trainer_Assessment_Q3.json", "row_id": 30, "trainee_name": "D", "type": "Assessment", "assessor": "Sarah"},
    {"source": "Trainer_Profile.csv", "row_id": 8, "trainer_name": "Mark", "type": "Trainer", "tech_skill": "AWS"}
]
ids = ["doc1_kpi_A", "doc2_assessment_B", "doc3_kpi_C", "doc4_trainer_Jane", "doc5_assessment_D", "doc6_trainer_Mark"]

try:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    print("\nSuccessfully added sample data to the collection.")
    count = collection.count()
    print(f"\nTotal documents in collection '{COLLECTION_NAME}': {count}")

except Exception as e:
    print(f"\nAn error occurred during data insertion: {e}")