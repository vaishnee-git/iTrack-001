# indexing_chunking.py
# Converts processed Trainee & Trainer DataFrames into LangChain Documents
# Filters metadata, exports to JSONL (for checking), and returns JSON data structure

import os
import json
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from cleaning_hashing import get_processed_dataframes


# ──────────────────────────────────────────────────────────────
# Configuration 
# ──────────────────────────────────────────────────────────────
MAIN_FOLDER = r"C:\Users\devavaishnee.m\Desktop\itrack-chatbot\itrackTraining-001"
OUTPUT_FOLDER = os.path.join(MAIN_FOLDER, "tests")
FINAL_INDEX_FILE = os.path.join(OUTPUT_FOLDER, "itrack_hybrid_index.jsonl")


class TraineeTrainerLoader:
    """
    Custom loader to convert a processed DataFrame into filtered LangChain Documents.
    Ensures only safe/relevant metadata is kept.
    """
    def __init__(self, dataframe: pd.DataFrame, source_type: str):
        if dataframe.empty:
            self.loader = None
            return

        df = dataframe.copy()

        # Create id from data_hash and add source tag
        df["id"] = df["data_hash"].astype(str)
        df["source"] = source_type

        # Keep only essential columns
        keep_columns = ["page_content", "id", "data_hash", "version_hash", "source"]
        df_filtered = df[keep_columns]

        # Initialize LangChain DataFrameLoader
        self.loader = DataFrameLoader(df_filtered, page_content_column="page_content")

    def load(self):
        """Returns list of Document objects (or empty list if no data)"""
        if self.loader is None:
            return []
        return self.loader.load()


def create_index_jsonl(main_folder_path: str) -> list:
    """
    Main processing function:
    - Loads processed DataFrames
    - Converts to filtered LangChain Documents
    - Creates JSON data structure
    - Saves to JSONL file (for checking)
    - Returns JSON data for next step (embedd_loadTochroma.py)
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("INDEXING & CHUNKING PIPELINE".center(90, "="))
    print("Creating documents with metadata...\n")

    try:
        trainee_df, trainer_df = get_processed_dataframes(main_folder_path)
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return []

    print(f"Trainee records : {len(trainee_df):,}")
    print(f"Trainer records : {len(trainer_df):,}\n")

    all_docs = []

    # Process trainee data
    if not trainee_df.empty:
        print("Processing trainees for indexing and chunking...")
        all_docs.extend(TraineeTrainerLoader(trainee_df, "trainee").load())
    else:
        print("No trainee data to process.")

    # Process trainer data
    if not trainer_df.empty:
        print("Processing trainers for indexing and chunking...")
        all_docs.extend(TraineeTrainerLoader(trainer_df, "trainer").load())
    else:
        print("No trainer data to process.")

    print(f"\nTotal documents created: {len(all_docs):,}")

    if not all_docs:
        print("No documents to save. Skipping JSONL export.")
        return []  # Return empty list if no data

    # === CREATE JSON DATA STRUCTURE (for next pipeline step) ===
    json_data = []
    for doc in all_docs:
        doc_dict = {
            "id": doc.metadata.get("id"),
            "text": doc.page_content,
            "metadata": doc.metadata,
        }
        json_data.append(doc_dict)

    print(f"Created JSON data structure with {len(json_data):,} documents (ready for embedding)")

    # === SAVE TO JSONL (for manual checking) ===
    print(f"\nSaving final index → {FINAL_INDEX_FILE}")
    try:
        with open(FINAL_INDEX_FILE, "w", encoding="utf-8") as f:
            for doc_dict in json_data:
                f.write(json.dumps(doc_dict, ensure_ascii=False) + "\n")
        print(f"Successfully saved {len(json_data):,} documents to JSONL")
    except Exception as e:
        print(f"Failed to write JSONL file: {e}")

    print("\nSUCCESS! Json data is ready for embedding")
    print("=" * 90)

    return json_data  # Return for embedd_loadTochroma.py


# ===================================================================
# Manual testing entry point
# ===================================================================
def main():
    """Run when executing this file directly"""
    json_data = create_index_jsonl(MAIN_FOLDER)
    print(f"\nPipeline complete! Generated {len(json_data):,} documents")
    if json_data:
        print("\nSample document (first 500 chars):")
        print(json.dumps(json_data[0], indent=2, ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    main()