import os
import json
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random
import warnings
import psycopg2

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings('ignore')

# ===================================================================
# .env LOADING
# ===================================================================
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
env_path = os.path.join(project_root, '.env')

print(f"🔍 Looking for .env at: {env_path}")
if not os.path.exists(env_path):
    raise FileNotFoundError(f".env file not found! Place it at: {env_path}")

load_dotenv(env_path, override=True)
print("✅ .env loaded successfully")

# ===================================================================
# CONFIGURATION
# ===================================================================
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DATABASE = os.getenv("PG_DATABASE")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_TABLE = os.getenv("PG_TABLE", "document_chunks")

COLLECTION_NAME = "project_vectors"
CHROMA_PATH = os.path.join(current_file_dir, "chromadb_data")

print("\n🔍 Configuration:")
print(f"  💾 ChromaDB Path: {CHROMA_PATH}")

# ===================================================================
# BEST EMBEDDER
# ===================================================================
class MiniLMEmbedder(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
            **kwargs
        )

# ===================================================================
# NATURAL LANGUAGE TEXT EXTRACTION
# ===================================================================
def extract_searchable_text(json_record: dict) -> str:
    record_type = json_record.get("record_type", "")
    
    if record_type == "TRAINER_ASSESSMENT":
        trainer = json_record.get("Trainer_Name", "trainer")
        python = json_record.get("Technical_Aggregate_Python", "N/A")
        snowflake = json_record.get("Technical_Aggregate_Snowflake", "N/A")
        databricks = json_record.get("Technical_Aggregate_Databricks", "N/A")
        comments = json_record.get("Trainer_Comments", "")
        to_focus = json_record.get("To_Focus", "")
        
        return f"Trainer {trainer} assessed a trainee. Python skill level {python}/10. " \
               f"Snowflake {snowflake}/10, Databricks {databricks}/10. " \
               f"Comments: {comments}. Focus area: {to_focus}."
    
    elif record_type == "TRAINEE_LOG":
        name = json_record.get("Trainee_Name", "trainee")
        task = json_record.get("Task_Name", "task")
        status = json_record.get("Task_Completed", "")
        pto = json_record.get("PTO", 0)
        pto_reason = json_record.get("PTO_Reason", "")
        participation = json_record.get("Org_Participation", "")
        
        pto_text = f"took {pto} PTO days ({pto_reason})" if pto > 0 else "no PTO"
        
        return f"Trainee {name} worked on {task}. Status: {status}. {pto_text}. " \
               f"Participated in: {participation}."
    
    return "Unknown record"

def create_json_record(data: dict, record_type: str) -> dict:
    clean_data = {"record_type": record_type}
    for k, v in data.items():
        if k in ['data_hash', 'version_hash']:
            continue
        if pd.notna(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                clean_data[k] = v
            else:
                clean_data[k] = str(v)
    return clean_data

# ===================================================================
# RICH DATA GENERATION
# ===================================================================
def generate_rich_training_data():
    trainees = [
        {"name": "Aarav Sharma", "emp_id": "EMP1001", "email": "aarav@company.com"},
        {"name": "Isha Patel", "emp_id": "EMP1002", "email": "isha@company.com"},
        {"name": "Vihaan Kumar", "emp_id": "EMP1003", "email": "vihaan@company.com"},
        {"name": "Ananya Singh", "emp_id": "EMP1004", "email": "ananya@company.com"},
        {"name": "Arjun Gupta", "emp_id": "EMP1005", "email": "arjun@company.com"},
    ]

    trainers = ["Alice Johnson", "Bob Smith", "Carol Lee", "David Kim", "Eve Chen"]

    training_id = "TRAIN2025-001"

    all_documents = []

    for idx, trainee in enumerate(trainees):
        name = trainee["name"]
        emp_id = trainee["emp_id"]
        trainer = trainers[idx]

        # 5 daily logs
        base_date = datetime(2025, 12, 15)
        for day in range(5):
            log_date = base_date + timedelta(days=day)
            record = {
                "Trainee_Name": name,
                "Employee_id": emp_id,
                "Email_address": trainee["email"],
                "Training_id": training_id,
                "Last_log_date": log_date.strftime("%Y-%m-%d"),
                "PTO": random.choice([0, 1, 2]),
                "PTO_Reason": random.choice(["Sick", "Personal", "Vacation", ""]),
                "WFH": random.choice(["Yes", "No"]),
                "Task_Name": random.choice(["Python Project", "Snowflake Setup", "Databricks Pipeline", "ETL Development", "ML Model"]),
                "Task_Completed": random.choice(["Yes", "No", "In Progress"]),
                "Start_Date": (log_date - timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d"),
                "Expected_date": (log_date + timedelta(days=14)).strftime("%Y-%m-%d"),
                "End_Date": "",
                "Org_Participation": random.choice(["Team Meeting", "Knowledge Share", "Hackathon", "None"]),
                "Certifications": random.choice(["SnowPro", "Databricks Certified", "PCAP", "None"]),
                "Additional_Skills": random.choice(["Git", "Docker", "Airflow", "Spark", "None"])
            }
            doc_id = f"TRAINEE_LOG_{emp_id}_{log_date.strftime('%Y%m%d')}"
            all_documents.append({"doc_id": doc_id, "json_record": create_json_record(record, "TRAINEE_LOG")})

        # Trainer assessment
        assessment = {
            "Training_id": training_id,
            "Trainer_Name": trainer,
            "attending_calls": random.choice(["Yes", "No", "Partial"]),
            "reporting_time": random.choice(["On Time", "Late", "Early"]),
            "Review_date": "2025-12-19",
            "Technical_Aggregate_Python": random.randint(6, 10),
            "Technical_Aggregate_Snowflake": random.randint(5, 9),
            "Technical_Aggregate_Databricks": random.randint(4, 9),
            "Problem_Solving": random.randint(7, 10),
            "Communication_Score": random.randint(6, 10),
            "Professionalism_Score": random.randint(8, 10),
            "Presentation_skills": random.randint(5, 9),
            "Team_Collaboration": random.randint(7, 10),
            "Leadership_Score": random.randint(4, 8),
            "To_Focus": random.choice(["Improve Databricks", "Excellent progress", "Focus on presentation", "Strengthen Python"]),
            "Trainer_Comments": random.choice(["Strong Python skills and initiative", "Great team player", "Needs Databricks practice", "Outstanding performance", "Good analytical skills"])
        }
        doc_id = f"TRAINER_ASSESSMENT_{emp_id}"
        all_documents.append({"doc_id": doc_id, "json_record": create_json_record(assessment, "TRAINER_ASSESSMENT")})

    print(f"📊 Generated {len(all_documents)} rich records")
    return all_documents

# ===================================================================
# PG & CHROMA SYNC
# ===================================================================
def get_pg_connection():
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        print("✅ PostgreSQL connected successfully.")
        return conn
    except Exception as e:
        print(f"❌ PG Connection failed: {e}")
        return None

def insert_and_sync_data(conn, documents_data: List[Dict]):
    if not conn:
        return False
    
    # Clear PG table
    try:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {PG_TABLE}")
        conn.commit()
        cursor.close()
        print("🧹 Cleared PG table")
    except Exception as e:
        print(f"Clear failed: {e}")
        conn.rollback()
        return False
    
    # Insert to PG
    try:
        cursor = conn.cursor()
        inserted = 0
        for i, item in enumerate(documents_data):
            doc_id = item["doc_id"]
            json_record = item["json_record"]
            
            cursor.execute(f"""
                INSERT INTO {PG_TABLE} 
                (document_id, data_hash, version_hash, raw_text_chunk, date_ingested)
                VALUES (%s, %s, %s, %s, %s)
            """, (doc_id, i + 1000, "v1.0", json.dumps(json_record), datetime.now()))
            inserted += 1
        
        conn.commit()
        cursor.close()
        print(f"✅ Inserted {inserted} records to PG")
    except Exception as e:
        print(f"Insert failed: {e}")
        conn.rollback()
        return False
    
    # Create Chroma documents (CORRECTED)
    print("\n📝 Creating Chroma documents...")
    documents = []
    for item in documents_data:
        json_data = item["json_record"]  # Original JSON
        searchable_text = extract_searchable_text(json_data)  # For semantic search
        
        doc = Document(
            page_content=json.dumps(json_data),  # ✅ JSON in page_content
            metadata={
                "document_id": item["doc_id"],
                "record_type": json_data.get("record_type"),
                "searchable_text": searchable_text,  # ✅ Text for BM25/semantic
                "json_preview": str(json_data)[:200]  # Debug
            }
        )
        documents.append(doc)
    
    print(f"📄 Prepared {len(documents)} Chroma documents")
    
    # Index to Chroma
    os.makedirs(CHROMA_PATH, exist_ok=True)
    embedder = MiniLMEmbedder()
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ Indexed {vectorstore._collection.count()} vectors in Chroma")
    return True

# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("🚀 LOADING RICH TRAINING DATA (FIXED JSON)...")
    
    data = generate_rich_training_data()
    conn = get_pg_connection()
    
    if conn:
        success = insert_and_sync_data(conn, data)
        conn.close()
        if success:
            print("\n🎉 SUCCESS! PG + Chroma fully synced with proper JSON!")
        else:
            print("❌ PG sync failed")
    else:
        print("⚠️  Skipping PG - run retrieval tests anyway")
    
    print("\n✅ READY! Run your retrieval code now.")