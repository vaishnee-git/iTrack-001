# indexing_langchain_way.py
# Fully LangChain-native, clean, and ready for any vectorstore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import DataFrameLoader
from cleaning_hashing import get_processed_dataframes
import pandas as pd
import os
from dotenv import load_dotenv
import json

# ──────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))
MAIN_FOLDER = os.getenv("MAIN_FOLDER")
OUTPUT_FOLDER = os.path.join(MAIN_FOLDER, "tests")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
FINAL_INDEX_FILE = os.path.join(OUTPUT_FOLDER, "itrack_hybrid_index.jsonl")
# ──────────────────────────────────────────────────────────────

 
def create_id(source_type: str, index: int) -> str:
    prefix = "STU" if source_type == "trainee" else "MAS"
    return f"{prefix}{index + 1:04d}"


#LangChain-native loader that converts DataFrame → Documents with metadata
class TraineeTrainerLoader(DataFrameLoader):
    def __init__(self, dataframe: pd.DataFrame, source_type: str):
        # Tell LangChain which column has the main text
        super().__init__(dataframe, page_content_column="page_content")
        self.df = dataframe.copy()
        self.source_type = source_type

        # Pre-build clean text + metadata for each row
        docs = []
        for idx, row in self.df.iterrows():
            friendly_id = create_id(source_type, idx)

            # Build clean text (same logic as before, but cleaner)
            parts = []
            for col, val in row.items():
                if col in ["data_hash", "version_hash", "page_content"]:
                    continue
                if pd.isna(val) or str(val).strip() in ["", "N/A", "0"]:
                    continue
                pretty_col = col.replace("_", " ").title()
                parts.append(f"{pretty_col}: {val}")

            content = "\n".join(parts) if parts else "No data available"

            metadata = {
                "id": friendly_id,
                "source": source_type,
                "trainee_name": str(row.get("Trainee_Name", "Unknown")),
                "employee_id": str(row.get("Employee_id", "")),
                "data_hash": row["data_hash"],
                "version_hash": row.get("version_hash", ""),
                "original_row": int(idx),
                "record_type": source_type,
                "source_file": "Trainee_Processed" if source_type == "trainee" else "Trainer_Processed"
            }

            docs.append(Document(page_content=content, metadata=metadata))

        self.docs = docs

    def load(self):
        return self.docs


if __name__ == "__main__":
    print("LangChain-Native Indexing (2025 Style)".center(90, "="))

    trainee_df, trainer_df = get_processed_dataframes(MAIN_FOLDER)
    print(f"Trainee records : {len(trainee_df):,}")
    print(f"Trainer records  : {len(trainer_df):,}\n")

    # LangChain splitter (same as before, but now official)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        keep_separator=True
    )

    # Load + convert + split — all in LangChain style
    all_docs = []

    # Process Trainees
    print("Loading & splitting trainees...")
    trainee_loader = TraineeTrainerLoader(trainee_df, "trainee")
    trainee_docs = trainee_loader.load()
    trainee_chunks = splitter.split_documents(trainee_docs)
    all_docs.extend(trainee_chunks)

    # Process Trainers
    print("Loading & splitting trainers...")
    trainer_loader = TraineeTrainerLoader(trainer_df, "trainer")
    trainer_docs = trainer_loader.load()
    trainer_chunks = splitter.split_documents(trainer_docs)
    all_docs.extend(trainer_chunks)

    # Auto-assign chunk IDs (LangChain doesn't do this by default)
    for i, chunk in enumerate(all_docs):
        base_id = chunk.metadata["id"]
        if len(splitter.split_text(chunk.page_content)) > 1:
            chunk.metadata["id"] = f"{base_id}_{i % 1000}"  # simple way
        chunk.metadata["chunk_index"] = i % 1000
        chunk.metadata["total_chunks"] = len(all_docs)

    print(f"\nTotal chunks: {len(all_docs):,}")

    # Save as JSONL (same format — works with any loader)
    print(f"Saving → {FINAL_INDEX_FILE}")
    with open(FINAL_INDEX_FILE, "w", encoding="utf-8") as f:
        for doc in all_docs:
            json.dump({
                "id": doc.metadata["id"],
                "text": doc.page_content,
                "metadata": doc.metadata
            }, f, ensure_ascii=False)
            f.write("\n")

    print("\nchunking and indexing part has been done successfully!")